import time
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement as FullCraftaxAchievements
from craftax.craftax_classic.constants import Achievement as ClassicAchievements
from src.planners.utils import _build_model, _load_checkpoint, _make_apply_fns

def sample_plan_inpainting(
    apply_fn, params, rng, obs, history, hist_len,
    num_actions, plan_horizon, diffusion_steps, temperature, top_k
):
    """Custom Diffusion Loop implementing the Historical Inpainting Paradox."""
    batch_size = obs.shape[0]
    mask_token_id = num_actions

    def diff_step(carry, step):
        seq, step_rng = carry
        step_rng, model_rng, sample_rng, remask_rng = jax.random.split(step_rng, 4)

        ratio = step / diffusion_steps
        t_val = 1.0 - ratio
        t_tensor = jnp.full((batch_size,), t_val)

        logits = apply_fn(params, obs, seq, t_tensor, model_rng)
        scaled_logits = logits / jnp.maximum(temperature, 1e-8)

        if top_k is not None:
            top_k_vals, _ = jax.lax.top_k(scaled_logits, top_k)
            kth_vals = top_k_vals[..., -1:]
            filtered_logits = jnp.where(scaled_logits >= kth_vals, scaled_logits, -jnp.inf)
        else:
            filtered_logits = scaled_logits

        preds = jax.random.categorical(sample_rng, filtered_logits, axis=-1)
        probs = jax.nn.softmax(filtered_logits, axis=-1)
        confidences = jnp.take_along_axis(probs, preds[..., None], axis=-1).squeeze(-1)

        num_to_unmask = jnp.maximum(1, (plan_horizon * ratio).astype(jnp.int32))
        sorted_conf = jnp.sort(confidences, axis=-1)[..., ::-1]
        thresholds = sorted_conf[jnp.arange(batch_size), num_to_unmask - 1]

        mask_condition = confidences < thresholds[:, None]
        seq_new = jnp.where(mask_condition, mask_token_id, preds)

        remask_prob = 0.15 * (1.0 - ratio)
        is_unmasked = seq_new != mask_token_id
        remdm_mask = (jax.random.uniform(remask_rng, seq_new.shape) < remask_prob) & is_unmasked
        seq_new = jnp.where(remdm_mask, mask_token_id, seq_new)

        idx_matrix = jnp.arange(plan_horizon)[None, :].repeat(batch_size, axis=0)
        lock_mask = idx_matrix < hist_len[:, None]
        seq_new = jnp.where(lock_mask, history, seq_new)

        return (seq_new, step_rng), None

    init_seq = jnp.full((batch_size, plan_horizon), mask_token_id, dtype=jnp.int32)
    idx_matrix = jnp.arange(plan_horizon)[None, :].repeat(batch_size, axis=0)
    lock_mask = idx_matrix < hist_len[:, None]
    init_seq = jnp.where(lock_mask, history, init_seq)

    steps_array = jnp.arange(1, diffusion_steps + 1)
    (final_seq, _), _ = jax.lax.scan(diff_step, (init_seq, rng), steps_array)

    return final_seq


def run_inference(config: Dict[str, Any]) -> None:
    env_name = config["ENV_NAME"]
    
    # Notice we REMOVED the AutoReset and Log wrappers. We want raw game access.
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    num_actions: int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs: int = config.get("NUM_ENVS", 32)
    plan_horizon: int = config["PLAN_HORIZON"]
    diffusion_steps: int = config.get("DIFFUSION_STEPS_EVAL", 10)
    temperature: float = config.get("TEMPERATURE", 0.5)
    top_k: int = config.get("TOP_K", 4)
    
    # 10,000 steps is exactly ONE maximum length Craftax game.
    eval_steps = 10000 

    model = _build_model(config, num_actions)
    apply_inference, _ = _make_apply_fns(model)

    print(f"Loading checkpoint: {config['CHECKPOINT_PATH']}")
    model_params = _load_checkpoint(config, model, obs_dim, config["CHECKPOINT_PATH"])

    rng = jax.random.PRNGKey(config.get("SEED", 42))

    @jax.jit
    def mpc_step(carry, step_idx):
        obs, state, rng, history, hist_len = carry
        rng, plan_rng, env_rng = jax.random.split(rng, 3)

        seq_full = hist_len >= plan_horizon
        hist_len = jnp.where(seq_full, 0, hist_len)
        history = jnp.where(seq_full[:, None], num_actions, history)

        plan = sample_plan_inpainting(
            apply_fn=apply_inference, params=model_params, rng=plan_rng,
            obs=obs, history=history, hist_len=hist_len,
            num_actions=num_actions, plan_horizon=plan_horizon,
            diffusion_steps=diffusion_steps, temperature=temperature, top_k=top_k
        )

        action = jnp.take_along_axis(plan, hist_len[:, None], axis=-1).squeeze(-1)
        history = history.at[jnp.arange(num_envs), hist_len].set(action)
        hist_len += 1

        obs_next, state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(env_rng, num_envs), state, action, env_params
        )

        hist_len = jnp.where(done, 0, hist_len)
        history = jnp.where(done[:, None], num_actions, history)

        # We return the RAW achievements array directly from the state!
        return (obs_next, state_next, rng, history, hist_len), (reward, done, state_next.achievements)

    print(f"\nDropping {num_envs} agents into {env_name} for 1 full life (10,000 steps)...")
    
    rng, env_rng = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(env_rng, num_envs), env_params
    )
    
    history = jnp.full((num_envs, plan_horizon), num_actions, dtype=jnp.int32)
    hist_len = jnp.zeros((num_envs,), dtype=jnp.int32)
    carry = (obs, state, rng, history, hist_len)

    t0 = time.time()
    # Run the full 10,000 steps
    _, (rewards, dones, achievements) = jax.lax.scan(mpc_step, carry, jnp.arange(eval_steps))
    elapsed = time.time() - t0

    # achievements shape: [10000 steps, 32 envs, num_achievements]
    # Find the maximum achievement unlocked for each of the 32 agents across the 10,000 steps
    max_achievements_per_agent = jnp.max(achievements, axis=0) # Shape: [32, num_achievements]
    
    # Count how many total agents (out of 32) got each achievement
    total_agents_with_achievement = jnp.sum(max_achievements_per_agent, axis=0)
    
    # Calculate the total reward each agent got
    total_reward_per_agent = jnp.sum(rewards, axis=0)

    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE IN {elapsed:.1f} SECONDS")
    # --- CALCULATE EXACT REPORT CARD ---
    max_achievements_per_agent = jnp.max(achievements, axis=0)
    total_agents_with_achievement = jnp.sum(max_achievements_per_agent, axis=0)
    total_reward_per_agent = jnp.sum(rewards, axis=0)

    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE IN {elapsed:.1f} SECONDS")
    print("="*50)
    print(f"Average Score across 32 games: {float(jnp.mean(total_reward_per_agent)):.1f}")
    print(f"Highest Score in a single game: {float(jnp.max(total_reward_per_agent)):.1f}")
    print("\nACHIEVEMENT REPORT CARD (Out of 32 Agents):")
    
    if "Classic" in env_name:
        achievement_names = [a.name.replace("_", " ").title() for a in ClassicAchievements]
    else:
        achievement_names = [a.name.replace("_", " ").title() for a in FullCraftaxAchievements]

    # REMOVED the "if count > 0" check. Now it prints the entire tech tree!
    for i, count in enumerate(total_agents_with_achievement):
        name = achievement_names[i] if i < len(achievement_names) else f"Achievement {i}"
        
        # Color code it slightly: show hits clearly, and misses as 0
        if count > 0:
            print(f"  ✅ {name}: {int(count)} / {num_envs} agents")
        else:
            print(f"  ❌ {name}: 0 / {num_envs} agents")

    print("="*50)

    # --- WANDB LOGGING ---
    if config.get("USE_WANDB", True):
        wandb.init(
            project=config.get("WANDB_PROJECT", "craftax-remdm"),
            name=f"ReportCard-T{temperature}-K{top_k}",
            config=config,
            job_type="evaluation"
        )
        
        summary_log = {"eval/average_score": float(jnp.mean(total_reward_per_agent))}
        for i, count in enumerate(total_agents_with_achievement):
            name = achievement_names[i] if i < len(achievement_names) else f"Ach_{i}"
            # Log all of them to WandB as well
            summary_log[f"eval/achievements/{name.replace(' ', '_')}"] = float(count) / num_envs * 100
        wandb.log(summary_log)
        
        table = wandb.Table(columns=["Game ID", "Total Score", "Max Achievements Unlocked"])
        for env_id in range(num_envs):
            score = float(total_reward_per_agent[env_id])
            unlocked = int(jnp.sum(max_achievements_per_agent[env_id]))
            table.add_data(f"Agent {env_id+1}", score, unlocked)
            
        wandb.log({"Individual Game Results": table})
        wandb.finish()