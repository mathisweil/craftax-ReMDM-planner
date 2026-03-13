"""Evaluation: run a trained diffusion planner with MPC + historical inpainting."""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement as FullCraftaxAchievements
from craftax.craftax_classic.constants import Achievement as ClassicAchievements

from .state import build_model, load_checkpoint, make_apply_fns


# ---------------------------------------------------------------------------
# Inpainting sampler: locks historical actions, diffuses the rest
# ---------------------------------------------------------------------------

def sample_plan_inpainting(
    apply_fn, params, rng, obs, history, hist_len,
    num_actions, plan_horizon, diffusion_steps, temperature, top_p,
):
    """Diffusion sampling with locked historical prefix (inpainting)."""
    B = obs.shape[0]
    mask_id = num_actions

    def _step(carry, step):
        seq, rng = carry
        rng, model_rng, sample_rng, remask_rng = jax.random.split(rng, 4)

        ratio = step / diffusion_steps
        t_tensor = jnp.full((B,), 1.0 - ratio)
        logits = apply_fn(params, obs, seq, t_tensor, model_rng) / jnp.maximum(temperature, 1e-8)

        # Optional nucleus filtering
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

        # Keep top-k most confident predictions unmasked
        num_unmask = jnp.maximum(1, (plan_horizon * ratio).astype(jnp.int32))
        sorted_conf = jnp.sort(conf, axis=-1)[..., ::-1]
        thresh = sorted_conf[jnp.arange(B), num_unmask - 1]
        seq_new = jnp.where(conf < thresh[:, None], mask_id, preds)

        # Light remasking (ReMDM-style)
        remask_prob = 0.15 * (1.0 - ratio)
        do_remask = (jax.random.uniform(remask_rng, seq_new.shape) < remask_prob) & (seq_new != mask_id)
        seq_new = jnp.where(do_remask, mask_id, seq_new)

        # Lock historical prefix
        pos = jnp.broadcast_to(jnp.arange(plan_horizon)[None, :], (B, plan_horizon))
        seq_new = jnp.where(pos < hist_len[:, None], history, seq_new)

        return (seq_new, rng), None

    # Initialize with history locked, rest masked
    init_seq = jnp.full((B, plan_horizon), mask_id, dtype=jnp.int32)
    pos = jnp.broadcast_to(jnp.arange(plan_horizon)[None, :], (B, plan_horizon))
    init_seq = jnp.where(pos < hist_len[:, None], history, init_seq)

    (final_seq, _), _ = jax.lax.scan(_step, (init_seq, rng), jnp.arange(1, diffusion_steps + 1))
    return final_seq


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

    model = build_model(config, num_actions)
    apply_eval, _ = make_apply_fns(model)

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    rng, ckpt_rng = jax.random.split(rng)
    model_params = load_checkpoint(model, ckpt_rng, obs_dim, plan_horizon, config["CHECKPOINT_PATH"])
    env_indices = jnp.arange(num_envs)

    @jax.jit
    def mpc_step(carry, _step_idx):
        obs, state, rng, history, hist_len = carry
        rng, plan_rng, env_rng = jax.random.split(rng, 3)

        # Reset history when plan is exhausted
        seq_full = hist_len >= plan_horizon
        hist_len = jnp.where(seq_full, 0, hist_len)
        history = jnp.where(seq_full[:, None], num_actions, history)

        plan = sample_plan_inpainting(
            apply_eval, model_params, plan_rng, obs,
            history, hist_len, num_actions, plan_horizon,
            diffusion_steps, temperature, top_p,
        )

        action = jnp.take_along_axis(plan, hist_len[:, None], axis=-1).squeeze(-1)
        history = history.at[env_indices, hist_len].set(action)
        hist_len = hist_len + 1

        obs_next, state_next, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(env_rng, num_envs), state, action, env_params,
        )

        hist_len = jnp.where(done, 0, hist_len)
        history = jnp.where(done[:, None], num_actions, history)
        return (obs_next, state_next, rng, history, hist_len), (reward, done, state_next.achievements)

    print(f"\nRunning {num_envs} agents in {env_name} for {eval_steps} steps...")

    rng, env_rng = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(env_rng, num_envs), env_params,
    )
    history = jnp.full((num_envs, plan_horizon), num_actions, dtype=jnp.int32)
    hist_len = jnp.zeros(num_envs, dtype=jnp.int32)

    t0 = time.time()
    _, (rewards, dones, achievements) = jax.lax.scan(
        mpc_step, (obs, state, rng, history, hist_len), jnp.arange(eval_steps),
    )
    elapsed = time.time() - t0

    # First-episode extraction (strict single-life evaluation)
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

    # Report
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
            name=f"Eval-T{temperature}-P{top_p}",
            config=config, job_type="evaluation",
        )
        summary = {"eval/average_score": float(ep_rewards.mean())}
        for i in range(top_idx + 1):
            summary[f"eval/achievements/{ach_names[i][1]}"] = pct[i]
        wandb.log(summary)

        table = wandb.Table(columns=["Agent", "Score", "Achievements", "Lifespan"])
        unlocked = ep_ach.sum(axis=-1)
        for i in range(num_envs):
            table.add_data(f"Agent {i + 1}", float(ep_rewards[i]), int(unlocked[i]), int(ep_lengths[i]))
        wandb.log({"Individual Results": table})
        wandb.finish()
