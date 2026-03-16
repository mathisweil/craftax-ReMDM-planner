"""Evaluation: run a trained diffusion planner with pure TWM Guided MPC."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
import optax
import orbax.checkpoint as ocp
from flax.training import train_state
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement as FullCraftaxAchievements
from craftax.craftax_classic.constants import Achievement as ClassicAchievements

from .state import build_model, load_checkpoint, make_apply_fns, create_train_state
from src.models.worldmodel import TransformerWorldModel

# ---------------------------------------------------------------------------
# Standard Diffusion Sampler (No History/Inpainting)
# ---------------------------------------------------------------------------

def sample_plan(
    apply_fn, params, rng, obs,
    num_actions, plan_horizon, diffusion_steps, temperature, top_p,
):
    """Standard Masked Discrete Diffusion sampling from scratch."""
    B = obs.shape[0]
    mask_id = num_actions

    def _step(carry, step):
        seq, rng = carry
        rng, model_rng, sample_rng, remask_rng = jax.random.split(rng, 4)

        ratio = step / diffusion_steps
        t_tensor = jnp.full((B,), 1.0 - ratio)
        logits = apply_fn(params, obs, seq, t_tensor, model_rng) / jnp.maximum(temperature, 1e-8)

        if top_p is not None:
            probs = jax.nn.softmax(logits, axis=-1)
            sorted_idx = jnp.argsort(-probs, axis=-1)
            sorted_p = jnp.take_along_axis(probs, sorted_idx, axis=-1)
            cutoff = jnp.cumsum(sorted_p, axis=-1) - sorted_p
            inv_idx = jnp.argsort(sorted_idx, axis=-1)
            nucleus_mask = jnp.take_along_axis(cutoff >= top_p, inv_idx, axis=-1)
            logits = jnp.where(nucleus_mask, -jnp.inf, logits)

        preds = jax.random.categorical(sample_rng, logits, axis=-1)
        conf = jnp.take_along_axis(
            jax.nn.softmax(logits, axis=-1), preds[..., None], axis=-1,
        ).squeeze(-1)

        num_unmask = jnp.maximum(1, (plan_horizon * ratio).astype(jnp.int32))
        sorted_conf = jnp.sort(conf, axis=-1)[..., ::-1]
        thresh = sorted_conf[jnp.arange(B), num_unmask - 1]
        seq_new = jnp.where(conf < thresh[:, None], mask_id, preds)

        remask_prob = 0.15 * (1.0 - ratio)
        do_remask = (jax.random.uniform(remask_rng, seq_new.shape) < remask_prob) & (seq_new != mask_id)
        seq_new = jnp.where(do_remask, mask_id, seq_new)

        return (seq_new, rng), None

    # Initialize with pure noise/masks
    init_seq = jnp.full((B, plan_horizon), mask_id, dtype=jnp.int32)
    (final_seq, _), _ = jax.lax.scan(_step, (init_seq, rng), jnp.arange(1, diffusion_steps + 1))
    
    return final_seq

# ---------------------------------------------------------------------------
# World Model Scoring
# ---------------------------------------------------------------------------

def score_candidates_twm(wm_net, wm_params, initial_obs, plan_candidates):
    """
    Autoregressively rolls out the TWM to score a batch of candidate plans.
    initial_obs: (Obs_Dim)
    plan_candidates: (Num_Candidates, Plan_Horizon)
    Returns: (Num_Candidates,) total predicted rewards.
    """
    num_candidates, plan_horizon = plan_candidates.shape
    obs_dim = initial_obs.shape[0]

    def _score_single_plan(plan_act_seq):
        def _twm_step(current_obs_seq, step_idx):
            # 1. Forward pass using sequence built so far
            next_obs_preds, rew_preds = wm_net.apply(
                {"params": wm_params}, 
                current_obs_seq[None, ...], 
                plan_act_seq[None, ...], 
                deterministic=True
            )
            
            # 2. Extract specific step predictions
            next_obs = next_obs_preds[0, step_idx]
            reward = rew_preds[0, step_idx]
            
            # 3. Inject predicted next_obs into the sequence for the next loop
            next_obs_seq = jax.lax.cond(
                step_idx + 1 < plan_horizon,
                lambda: current_obs_seq.at[step_idx + 1].set(next_obs),
                lambda: current_obs_seq
            )
            return next_obs_seq, reward

        # Initialize sequence with current real observation at t=0
        init_obs_seq = jnp.zeros((plan_horizon, obs_dim))
        init_obs_seq = init_obs_seq.at[0].set(initial_obs)

        _, rewards = jax.lax.scan(_twm_step, init_obs_seq, jnp.arange(plan_horizon))
        return jnp.sum(rewards)

    # vmap the scoring over all candidates
    return jax.vmap(_score_single_plan)(plan_candidates)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_inference(config: dict[str, Any]) -> None:
    env_name = config["ENV_NAME"]
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs = config.get("NUM_ENVS", 32)
    plan_horizon = config["PLAN_HORIZON"]
    diffusion_steps = config.get("DIFFUSION_STEPS_EVAL", 10)
    temperature = config.get("TEMPERATURE", 0.5)
    top_p = config.get("TOP_P", 0.95)
    eval_steps = int(float(config.get("EVAL_STEPS", 10000)))
    num_candidates = config.get("NUM_CANDIDATES", 4) # MPC search width

    # 1. Load Diffusion Brain
    model = build_model(config, num_actions)
    apply_eval, _ = make_apply_fns(model)
    rng = jax.random.PRNGKey(config.get("SEED", 42))
    rng, ckpt_rng = jax.random.split(rng)
    model_params = load_checkpoint(model, ckpt_rng, obs_dim, plan_horizon, config["CHECKPOINT_PATH"])

    # 2. Load World Model Engine
    wm_net = TransformerWorldModel(
        num_actions=num_actions, obs_dim=obs_dim,
        d_model=config.get("WM_D_MODEL", config["D_MODEL"]), 
        n_heads=config.get("WM_N_HEADS", config["N_HEADS"]),
        n_layers=config.get("WM_N_LAYERS", config["N_LAYERS"]),
        dropout_rate=config.get("DROPOUT_RATE", 0.1),
    )
    
    print(f"Loading World Model from {config['WM_CHECKPOINT_PATH']}...")
    wm_mgr = ocp.CheckpointManager(config["WM_CHECKPOINT_PATH"])
    dummy_obs = jnp.zeros((1, plan_horizon, obs_dim))
    dummy_act = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
    wm_params_init = wm_net.init(jax.random.PRNGKey(0), dummy_obs, dummy_act)["params"]
    wm_state_abstract = create_train_state(wm_net, wm_params_init, config.get("LR", 1e-4), config.get("MAX_GRAD_NORM", 1.0))
    restored_wm = wm_mgr.restore(wm_mgr.latest_step(), args=ocp.args.StandardRestore(wm_state_abstract))
    wm_params = restored_wm.params

    @jax.jit
    def mpc_step(carry, _step_idx):
        obs, state, rng = carry
        rng, plan_rng, env_rng = jax.random.split(rng, 3)

        # 1. Expand inputs to generate NUM_CANDIDATES per environment
        # Shape becomes (Num_Envs * Num_Candidates, ...)
        obs_expanded = jnp.repeat(obs, num_candidates, axis=0)
        
        # 2. Brain generates fresh plans from scratch
        plans_expanded = sample_plan(
            apply_eval, model_params, plan_rng, obs_expanded,
            num_actions, plan_horizon, diffusion_steps, temperature, top_p,
        )
        
        # Reshape to (Num_Envs, Num_Candidates, Plan_Horizon)
        candidate_plans = plans_expanded.reshape(num_envs, num_candidates, plan_horizon)

        # 3. TWM Physics Engine scores plans
        def _score_env(env_obs, env_candidates):
            return score_candidates_twm(wm_net, wm_params, env_obs, env_candidates)
        
        candidate_scores = jax.vmap(_score_env)(obs, candidate_plans)

        # 4. Pick best plan per environment
        best_plan_idx = jnp.argmax(candidate_scores, axis=1)
        best_plans = candidate_plans[jnp.arange(num_envs), best_plan_idx]

        # 5. Execute ONLY the first action (Pure MPC)
        action = best_plans[:, 0]

        obs_next, state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(env_rng, num_envs), state, action, env_params,
        )

        return (obs_next, state_next, rng), (reward, done, state_next.achievements)

    print(f"\nRunning {num_envs} agents in {env_name} for {eval_steps} steps...")
    print(f"MPC Candidates: {num_candidates} | Diffusion Steps: {diffusion_steps}")

    rng, env_rng = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(env_rng, num_envs), env_params,
    )

    t0 = time.time()
    _, (rewards, dones, achievements) = jax.lax.scan(
        mpc_step, (obs, state, rng), jnp.arange(eval_steps),
    )
    elapsed = time.time() - t0

    # First-episode extraction
    rewards_np = np.array(rewards)
    dones_np = np.array(dones)
    ach_np = np.array(achievements)

    ep_rewards = np.zeros(num_envs)
    ep_ach = np.zeros((num_envs, ach_np.shape[2]))
    ep_lengths = np.zeros(num_envs, dtype=int)

    for i in range(num_envs):
        death = np.where(dones_np[:, i])[0]
        end = death[0] if len(death) > 0 else eval_steps - 1
        ep_rewards[i] = rewards_np[:end + 1, i].sum()
        ep_ach[i] = ach_np[:end + 1, i].max(axis=0)
        ep_lengths[i] = end + 1

    pct = ep_ach.mean(axis=0) * 100.0

    print(f"\n{'=' * 50}")
    print(f"EVALUATION COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 50}")
    print(f"Average Score: {ep_rewards.mean():.1f}  |  Best: {ep_rewards.max():.1f}")

    ach_cls = ClassicAchievements if "Classic" in env_name else FullCraftaxAchievements
    ach_names = [(a.name.replace("_", " ").title(), a.name.lower()) for a in ach_cls]
    valid = [i for i, p in enumerate(pct) if p > 0]
    top_idx = max(valid) if valid else 5

    for i in range(top_idx + 1):
        name, _ = ach_names[i]
        count = int(pct[i] / 100.0 * num_envs)
        icon = "+" if count > 0 else "-"
        print(f"  [{icon}] {name}: {count}/{num_envs}")
    print(f"{'=' * 50}")

    if config.get("USE_WANDB", True):
        wandb.init(
            project=config.get("WANDB_PROJECT", "craftax-remdm"),
            name=f"Eval-PureMPC-TWM-Cand{num_candidates}",
            config=config, job_type="evaluation",
        )
        summary = {"eval/average_score": float(ep_rewards.mean())}
        for i in range(top_idx + 1):
            summary[f"eval/achievements/{ach_names[i][1]}"] = pct[i]
        wandb.log(summary)
        wandb.finish()