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

from src.diffusion.sampling import sample_plan_inpainting
from src.diffusion.schedules import SCHEDULE_MAP
from .model import build_model, load_checkpoint, make_apply_fns


def run_inference(config: dict[str, Any]) -> None:
    env_name = config["ENV_NAME"]
    env = make_craftax_env_from_name(env_name, auto_reset=True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs = config.get("EVAL_NUM_ENVS", 32)
    plan_horizon = config["PLAN_HORIZON"]
    diffusion_steps = config.get("DIFFUSION_STEPS_EVAL", 10)
    temperature = config.get("TEMPERATURE", 0.5)
    top_p = config.get("TOP_P", 0.95)
    eval_steps = int(float(config.get("EVAL_STEPS", 10000)))
    schedule_fn, _ = SCHEDULE_MAP[config.get("DIFFUSION_SCHEDULE", "cosine")]
    remask_strategy = config.get("REMASK_STRATEGY", "rescale")
    eta = config.get("ETA", 0.5)
    use_loop = config.get("USE_LOOP", True)
    t_on = config.get("T_ON", 0.7)
    t_off = config.get("T_OFF", 0.3)

    model = build_model(config, num_actions)
    apply_eval, _ = make_apply_fns(model)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, ckpt_rng = jax.random.split(rng)
    model_params = load_checkpoint(
        model, ckpt_rng, obs_dim, plan_horizon, config["CHECKPOINT_PATH"]
    )
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
            apply_eval,
            model_params,
            plan_rng,
            obs,
            history,
            hist_len,
            num_actions,
            plan_horizon,
            diffusion_steps,
            schedule_fn,
            remask_strategy,
            eta,
            use_loop,
            t_on,
            t_off,
            temperature,
            top_p,
        )

        action = jnp.take_along_axis(plan, hist_len[:, None], axis=-1).squeeze(-1)
        history = history.at[env_indices, hist_len].set(action)
        hist_len = hist_len + 1

        obs_next, state_next, reward, done, info = jax.vmap(
            env.step, in_axes=(0, 0, 0, None)
        )(
            jax.random.split(env_rng, num_envs),
            state,
            action,
            env_params,
        )

        hist_len = jnp.where(done, 0, hist_len)
        history = jnp.where(done[:, None], num_actions, history)
        return (obs_next, state_next, rng, history, hist_len), (
            reward,
            done,
            state_next.achievements,
        )

    print(f"\nRunning {num_envs} agents in {env_name} for {eval_steps} steps...")

    rng, env_rng = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(env_rng, num_envs),
        env_params,
    )
    history = jnp.full((num_envs, plan_horizon), num_actions, dtype=jnp.int32)
    hist_len = jnp.zeros(num_envs, dtype=jnp.int32)

    t0 = time.time()
    _, (rewards, dones, achievements) = jax.lax.scan(
        mpc_step,
        (obs, state, rng, history, hist_len),
        jnp.arange(eval_steps),
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
        ep_rewards[i] = rewards_np[: end + 1, i].sum()
        ep_ach[i] = ach_np[: end + 1, i].max(axis=0)
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

    out_path = config.get("INFERENCE_OUTPUT")
    if out_path:
        import json
        from pathlib import Path

        n_ach = min(len(ach_names), len(pct))
        payload = {
            "checkpoint": str(config.get("CHECKPOINT_PATH")),
            "env_name": env_name,
            "seed": int(config.get("SEED", 0)),
            "eval_num_envs": int(num_envs),
            "eval_steps": int(eval_steps),
            "diffusion_steps_eval": int(diffusion_steps),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "mean_score": float(ep_rewards.mean()),
            "best_score": float(ep_rewards.max()),
            "mean_episode_length": float(ep_lengths.mean()),
            "achievement_rates": {
                ach_names[i][1]: float(pct[i]) / 100.0 for i in range(n_ach)
            },
        }
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved inference results to {out_path}")

    if config.get("USE_WANDB", True):
        wandb.init(
            project=config.get("WANDB_PROJECT", "remdm-craftax"),
            name=f"Eval-T{temperature}-P{top_p}",
            config=config,
            job_type="evaluation",
        )
        summary = {"eval/average_score": float(ep_rewards.mean())}
        for i in range(top_idx + 1):
            summary[f"eval/achievements/{ach_names[i][1]}"] = pct[i]
        wandb.log(summary)

        table = wandb.Table(columns=["Agent", "Score", "Achievements", "Lifespan"])
        unlocked = ep_ach.sum(axis=-1)
        for i in range(num_envs):
            table.add_data(
                f"Agent {i + 1}",
                float(ep_rewards[i]),
                int(unlocked[i]),
                int(ep_lengths[i]),
            )
        wandb.log({"Individual Results": table})
        wandb.finish()
