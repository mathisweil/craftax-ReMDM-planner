"""Shared fixtures for the ReMDM test suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def rng():
    """Deterministic JAX PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def small_config():
    """Minimal config dict for fast tests."""
    return {
        "PLAN_HORIZON": 4,
        "D_MODEL": 32,
        "N_HEADS": 2,
        "N_LAYERS": 1,
        "D_FF": 64,
        "OBS_ENCODER_LAYERS": 1,
        "OBS_ENCODER_WIDTH": 32,
        "DROPOUT_RATE": 0.0,
        "LR": 1e-3,
        "MAX_GRAD_NORM": 1.0,
        "BATCH_SIZE": 4,
        "DIFFUSION_SCHEDULE": "cosine",
        "REMASK_STRATEGY": "rescale",
        "ETA": 0.5,
        "DIFFUSION_STEPS": 5,
        "NUM_ACTIONS": 5,
        "OBS_DIM": 8,
        "NUM_ENVS": 2,
        "NUM_STEPS": 4,
        "REPLAN_EVERY": 2,
        "NUM_UPDATES": 1,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "NUM_TRAIN_STEPS": 2,
        "DEBUG": False,
        "USE_WANDB": False,
        "SEED": 0,
        "NUM_REPEATS": 1,
        "SAVE_POLICY": False,
    }


# ---------------------------------------------------------------------------
# Dummy / mock environment for wrapper tests
# ---------------------------------------------------------------------------


class _DummyEnvParams:
    pass


class _DummyEnvState:
    """Trivial env state that is just a counter."""

    def __init__(self, step_count: int = 0):
        self.step_count = step_count


class DummyGymnaxEnv:
    """Minimal Gymnax-like environment for testing wrappers in isolation.

    - obs_dim dimensional float observations
    - num_actions discrete actions
    - Episodes never end (done=False always) unless step_count >= max_steps.
    """

    def __init__(self, obs_dim: int = 4, num_actions: int = 3, max_steps: int = 100):
        self.obs_dim = obs_dim
        self._num_actions = num_actions
        self.max_steps = max_steps
        self.default_params = _DummyEnvParams()

    def reset(self, key, params=None):
        obs = jax.random.normal(key, shape=(self.obs_dim,))
        state = {"step_count": jnp.int32(0)}
        return obs, state

    def step(self, key, state, action, params=None):
        new_count = state["step_count"] + 1
        obs = jax.random.normal(key, shape=(self.obs_dim,))
        reward = jnp.float32(1.0)
        done = new_count >= self.max_steps
        info = {}
        new_state = {"step_count": new_count}
        return obs, new_state, reward, done, info

    class _ActionSpace:
        def __init__(self, n):
            self.n = n

    class _ObsSpace:
        def __init__(self, shape):
            self.shape = shape

    def action_space(self, params=None):
        return self._ActionSpace(self._num_actions)

    def observation_space(self, params=None):
        return self._ObsSpace((self.obs_dim,))


@pytest.fixture
def dummy_env():
    return DummyGymnaxEnv(obs_dim=4, num_actions=3)


def make_dummy_model_apply(num_actions: int):
    """Create a trivial model_apply closed over num_actions (JIT-safe)."""
    def apply_fn(params, obs, z_t, t, rng=None):
        batch_size = obs.shape[0]
        plan_horizon = z_t.shape[1]
        return jnp.zeros((batch_size, plan_horizon, num_actions))
    return apply_fn


@pytest.fixture
def dummy_model_apply():
    return make_dummy_model_apply(5)


@pytest.fixture
def dummy_params():
    return {}
