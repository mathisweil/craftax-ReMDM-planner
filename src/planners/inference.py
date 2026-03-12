import time
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement as FullCraftaxAchievements
from craftax.craftax_classic.constants import Achievement as ClassicAchievements
from src.planners.utils import _build_model, _load_checkpoint, _make_apply_fns

def sample_plan_inpainting(
    apply_fn, params, rng, obs, history, hist_len,
    num_actions, plan_horizon, diffusion_steps, temperature, top_p
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

        if top_p is not None:
            probs = jax.nn.softmax(scaled_logits, axis=-1)
            sorted_indices = jnp.argsort(-probs, axis=-1)
            sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
            
            # Exclusive cumsum to keep the first token that pushes us over the threshold
            cutoff = jnp.cumsum(sorted_probs, axis=-1) - sorted_probs 
            mask = cutoff >= top_p # True means throw away
            
            # Map the mask back to the original vocabulary positions
            inv_indices = jnp.argsort(sorted_indices, axis=-1)
            original_mask = jnp.take_along_axis(mask, inv_indices, axis=-1)
            
            filtered_logits = jnp.where(original_mask, -jnp.inf, scaled_logits)
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

        idx_matrix = jnp.broadcast_to(jnp.arange(plan_horizon)[None, :], (batch_size, plan_horizon))
        lock_mask = idx_matrix < hist_len[:, None]
        seq_new = jnp.where(lock_mask, history, seq_new)

        return (seq_new, step_rng), None

    init_seq = jnp.full((batch_size, plan_horizon), mask_token_id, dtype=jnp.int32)
    idx_matrix = jnp.broadcast_to(jnp.arange(plan_horizon)[None, :], (batch_size, plan_horizon))
    lock_mask = idx_matrix < hist_len[:, None]
    init_seq = jnp.where(lock_mask, history, init_seq)

    steps_array = jnp.arange(1, diffusion_steps + 1)
    (final_seq, _), _ = jax.lax.scan(diff_step, (init_seq, rng), steps_array)

    return final_seq


def run_inference(config: dict[str, Any]) -> None:
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
    top_p: float = config.get("TOP_P", 0.95)
    eval_steps: int = int(float(config.get("EVAL_STEPS", 10000)))

    model = _build_model(config, num_actions)
    apply_inference, _ = _make_apply_fns(model)

    print(f"Loading checkpoint: {config['CHECKPOINT_PATH']}")
    model_params = _load_checkpoint(config, model, obs_dim, config["CHECKPOINT_PATH"])

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    env_indices = jnp.arange(num_envs)
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
            diffusion_steps=diffusion_steps, temperature=temperature, top_p=top_p
        )

        action = jnp.take_along_axis(plan, hist_len[:, None], axis=-1).squeeze(-1)
        history = history.at[env_indices, hist_len].set(action)
        hist_len += 1

        obs_next, state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(env_rng, num_envs), state, action, env_params
        )

        hist_len = jnp.where(done, 0, hist_len)
        history = jnp.where(done[:, None], num_actions, history)

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

    import numpy as np

    # Convert to numpy for easy slicing
    rewards_np = np.array(rewards)          # Shape: [10000, 32]
    dones_np = np.array(dones)              # Shape: [10000, 32]
    achievements_np = np.array(achievements)# Shape: [10000, 32, num_achievements]

    # Arrays to hold the STRICT single-episode data
    first_episode_rewards = np.zeros(num_envs)
    first_episode_achievements = np.zeros((num_envs, achievements_np.shape[2]))
    first_episode_lengths = np.zeros(num_envs, dtype=int)

    # --- THE FIRST LIFE FILTER ---
    for i in range(num_envs):
        # Find every timestep where this specific agent died
        done_indices = np.where(dones_np[:, i])[0]
        
        if len(done_indices) > 0:
            end_idx = done_indices[0] # Cut the timeline at their FIRST death
        else:
            end_idx = eval_steps - 1  # The absolute legend survived all 10k steps

        # Sum rewards only up to the first death
        first_episode_rewards[i] = np.sum(rewards_np[:end_idx + 1, i])
        
        # Take the max achievements unlocked strictly within this single life
        first_episode_achievements[i] = np.max(achievements_np[:end_idx + 1, i], axis=0)
        
        # Record how long they lived
        first_episode_lengths[i] = end_idx + 1

    # Calculate the TRUE percentage of agents that unlocked each achievement in 1 life
    final_pct = np.mean(first_episode_achievements, axis=0) * 100.0
    
    # --- REPORT CARD ---
    print("\n" + "="*50)
    print(f"STRICT SINGLE-EPISODE EVALUATION COMPLETE ({elapsed:.1f}s)")
    print("="*50)
    print(f"Average Score across {num_envs} games: {float(np.mean(first_episode_rewards)):.1f}")
    print(f"Highest Score in a single game: {float(np.max(first_episode_rewards)):.1f}")
    print("\nACHIEVEMENT REPORT CARD (Single Life Only):")

    achievement_cls = ClassicAchievements if "Classic" in env_name else FullCraftaxAchievements
    achievement_names = [(a.name.replace("_", " ").title(), a.name.lower()) for a in achievement_cls]

    # Find the deepest achievement unlocked by ANY agent
    valid_indices = [i for i, pct in enumerate(final_pct) if pct > 0]
    highest_idx = max(valid_indices) if valid_indices else 5

    for i in range(highest_idx + 1):
        name, _ = achievement_names[i]
        count = int(final_pct[i] / 100.0 * num_envs)
        icon = "✅" if count > 0 else "❌"
        print(f"  {icon} {name}: {count} / {num_envs} agents")
    print("="*50)

    # --- WANDB LOGGING (Summary & Table Only) ---
    if config.get("USE_WANDB", True):
        wandb.init(
            project=config.get("WANDB_PROJECT", "craftax-remdm"),
            name=f"Eval-Strict-T{temperature}-P{top_p}",
            config=config,
            job_type="evaluation"
        )
        
        # Log Summary (Bar Chart Data)
        summary_log = {"eval_strict/average_score": float(np.mean(first_episode_rewards))}
        for i in range(highest_idx + 1):
            _, key = achievement_names[i]
            summary_log[f"eval_strict/achievements/{key}"] = final_pct[i]
            
        wandb.log(summary_log)
        
        # Log Individual Game Table
        table = wandb.Table(columns=["Game ID", "Total Score", "Max Achievements", "Lifespan (Steps)"])
        unlocked_counts = np.sum(first_episode_achievements, axis=-1)
        
        for env_id in range(num_envs):
            table.add_data(
                f"Agent {env_id + 1}", 
                float(first_episode_rewards[env_id]), 
                int(unlocked_counts[env_id]),
                int(first_episode_lengths[env_id])
            )
            
        wandb.log({"Strict Individual Results": table})
        wandb.finish()