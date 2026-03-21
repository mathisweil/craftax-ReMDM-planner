"""Collect offline trajectories from a trained PPO agent."""

from __future__ import annotations

import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .data import load_ppo_agent, make_env


def collect_offline_data(config: dict[str, Any]) -> None:
    """Roll out a PPO agent and save (obs, actions, rewards, dones) to disk.

    Args:
        config: Upper-cased hyperparameter dict.  Must contain
            ``PPO_CHECKPOINT_PATH``, ``OFFLINE_DATA_PATH``,
            ``COLLECT_NUM_STEPS``, and ``COLLECT_NUM_ENVS``.
    """
    assert config.get("PPO_CHECKPOINT_PATH"), (
        "--ppo_checkpoint_path is required for --mode collect."
    )

    num_envs: int = config["COLLECT_NUM_ENVS"]
    num_iters: int = config["COLLECT_NUM_STEPS"] // num_envs

    env_w, env_params = make_env(config, num_envs)
    num_actions = env_w.action_space(env_params).n
    obs_dim = env_w.observation_space(env_params).shape[0]

    ppo = load_ppo_agent(
        config["PPO_CHECKPOINT_PATH"], num_actions, obs_dim,
        config.get("LAYER_SIZE", 512),
        model_type=config.get("PPO_MODEL_TYPE", "ppo_rnn"),
        config=config, num_envs=num_envs,
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, env_rng, collect_rng = jax.random.split(rng, 3)
    obs, env_state = env_w.reset(env_rng, env_params)
    done = jnp.zeros(num_envs, dtype=bool)
    hidden = ppo.init_hidden(num_envs)

    def _step(carry, _):
        rng, es, obs, done, hs = carry
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        action, new_hs = ppo.act(obs, done, hs, act_rng,
                                  temperature=config.get("COLLECT_TEMPERATURE", 1.0))
        obs_next, es, reward, done_next, _ = env_w.step(step_rng, es, action, env_params)
        return (rng, es, obs_next, done_next, new_hs), (obs, action, reward, done)

    rollout_fn = jax.jit(lambda c: jax.lax.scan(_step, c, None, length=num_iters))
    _, (obs_arr, act_arr, rew_arr, done_arr) = rollout_fn(
        (collect_rng, env_state, obs, done, hidden),
    )

    # [steps, envs, ...] -> [envs, steps, ...]
    obs_np  = np.array(obs_arr).transpose(1, 0, 2)
    act_np  = np.array(act_arr).transpose(1, 0)
    rew_np  = np.array(rew_arr).transpose(1, 0)
    done_np = np.array(done_arr).transpose(1, 0)

    out_path = config["OFFLINE_DATA_PATH"]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs_np, actions=act_np, rewards=rew_np, dones=done_np)
    total = obs_np.shape[0] * obs_np.shape[1]
    print(f"Saved {obs_np.shape[0]}×{obs_np.shape[1]} transitions ({total:,}) to '{out_path}'")


def run_collect(config: dict[str, Any]) -> None:
    collect_offline_data(config)
