#!/usr/bin/env python3
"""Evaluate a committed PPO expert checkpoint.

Mirrors the first-episode protocol of ``src/planners/inference.py``: N
vectorised environments stepped for M steps, per-env return, episode length
and achievements taken over the first life only.

Usage:
  uv run python scripts/eval_ppo_expert.py \
      --path checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M \
      --env-name Craftax-Classic-Symbolic-v1 \
      --num-envs 256 --steps 1024 --seed 0 \
      --output outputs/expert_eval/classic_seed0.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp
import numpy as np

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement as FullCraftaxAchievements
from craftax.craftax_classic.constants import Achievement as ClassicAchievements
from src.planners.ppo import load_ppo_agent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Orbax checkpoint directory.")
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-type", default="ppo_rnn")
    ap.add_argument("--layer-size", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--output", default=None)
    a = ap.parse_args()

    env = make_craftax_env_from_name(a.env_name, auto_reset=True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]

    agent = load_ppo_agent(
        a.path, num_actions, obs_dim, a.layer_size, a.model_type,
        config={"SEED": a.seed}, num_envs=a.num_envs,
    )

    rng = jax.random.PRNGKey(a.seed)
    rng, reset_rng = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_rng, a.num_envs), env_params,
    )
    hidden = agent.init_hidden(a.num_envs)
    done0 = jnp.zeros((a.num_envs,), dtype=bool)

    def step_fn(carry, _):
        obs, state, rng, hidden, done = carry
        rng, act_rng, env_rng = jax.random.split(rng, 3)
        action, hidden = agent.act(obs, done, hidden, act_rng, temperature=a.temperature)
        action = jnp.reshape(jnp.asarray(action), (a.num_envs,))
        obs2, state2, reward, done2, _info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
            jax.random.split(env_rng, a.num_envs), state, action, env_params,
        )
        return (obs2, state2, rng, hidden, done2), (reward, done2, state2.achievements)

    print(f"Evaluating {a.model_type} expert on {a.env_name}: "
          f"{a.num_envs} envs x {a.steps} steps, seed {a.seed}")
    t0 = time.time()
    _, (rewards, dones, achievements) = jax.lax.scan(
        step_fn, (obs, state, rng, hidden, done0), jnp.arange(a.steps),
    )
    elapsed = time.time() - t0

    rewards_np = np.array(rewards)
    dones_np = np.array(dones)
    ach_np = np.array(achievements)

    ep_rewards = np.zeros(a.num_envs)
    ep_ach = np.zeros((a.num_envs, ach_np.shape[2]))
    ep_lengths = np.zeros(a.num_envs, dtype=int)
    for i in range(a.num_envs):
        death = np.where(dones_np[:, i])[0]
        end = death[0] if len(death) > 0 else a.steps - 1
        ep_rewards[i] = rewards_np[: end + 1, i].sum()
        ep_ach[i] = ach_np[: end + 1, i].max(axis=0)
        ep_lengths[i] = end + 1

    pct = ep_ach.mean(axis=0)
    ach_cls = ClassicAchievements if "Classic" in a.env_name else FullCraftaxAchievements
    ach_names = [ach.name.lower() for ach in ach_cls]
    n_ach = min(len(ach_names), pct.shape[0])

    print(f"done in {elapsed:.1f}s | mean first-episode return "
          f"{ep_rewards.mean():.4f} | best {ep_rewards.max():.4f} "
          f"| mean length {ep_lengths.mean():.1f}")

    payload = {
        "checkpoint": a.path,
        "model_type": a.model_type,
        "env_name": a.env_name,
        "seed": a.seed,
        "num_envs": a.num_envs,
        "steps": a.steps,
        "temperature": a.temperature,
        "mean_score": float(ep_rewards.mean()),
        "best_score": float(ep_rewards.max()),
        "mean_episode_length": float(ep_lengths.mean()),
        "achievement_rates": {ach_names[i]: float(pct[i]) for i in range(n_ach)},
        "achievements_at_0.5": int(sum(1 for i in range(n_ach) if pct[i] >= 0.5)),
        "wall_clock_s": round(elapsed, 1),
    }
    if a.output:
        Path(a.output).parent.mkdir(parents=True, exist_ok=True)
        with open(a.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved expert evaluation to {a.output}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
