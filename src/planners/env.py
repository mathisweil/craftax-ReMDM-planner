"""Craftax environment construction and trajectory data structures."""

from __future__ import annotations

import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
from typing import NamedTuple

from Craftax_Baselines.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: dict


def make_env(config: dict, num_envs: int):
    """Build a wrapped Craftax environment.

    Args:
        config:    Upper-cased config dict with ``ENV_NAME``,
                   ``USE_OPTIMISTIC_RESETS``, ``OPTIMISTIC_RESET_RATIO``.
        num_envs:  Number of parallel environments.

    Returns:
        Tuple of ``(env, env_params)``.
    """
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=num_envs,
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], num_envs),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=num_envs)
    return env, env_params
