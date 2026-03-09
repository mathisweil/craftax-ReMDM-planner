"""Tests for src/planners/offline.py and common utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.remdm import cosine_schedule, linear_schedule


class TestScheduleMap:
    def test_contains_cosine(self):
        from src.planners.common import SCHEDULE_MAP
        assert "cosine" in SCHEDULE_MAP
        assert SCHEDULE_MAP["cosine"] is cosine_schedule

    def test_contains_linear(self):
        from src.planners.common import SCHEDULE_MAP
        assert "linear" in SCHEDULE_MAP
        assert SCHEDULE_MAP["linear"] is linear_schedule


class TestMakeTrainOffline:
    """Integration test with tiny synthetic data — no Craftax dependency."""

    def test_returns_callable(self, small_config):
        from src.planners.offline import make_train_offline

        num_envs, traj_len, obs_dim = 1, 20, 8
        num_actions = small_config["NUM_ACTIONS"]
        plan_horizon = small_config["PLAN_HORIZON"]

        offline_data = {
            "obs": np.random.randn(num_envs, traj_len, obs_dim).astype(np.float32),
            "actions": np.random.randint(0, num_actions, (num_envs, traj_len)).astype(np.int32),
            "dones": np.zeros((num_envs, traj_len), dtype=bool),
        }

        train_fn, data_arrays = make_train_offline(small_config, offline_data)
        assert callable(train_fn)
        assert isinstance(data_arrays, dict)

    def test_train_fn_runs(self, small_config):
        from src.planners.offline import make_train_offline

        num_envs, traj_len, obs_dim = 1, 20, 8
        num_actions = small_config["NUM_ACTIONS"]

        offline_data = {
            "obs": np.random.randn(num_envs, traj_len, obs_dim).astype(np.float32),
            "actions": np.random.randint(0, num_actions, (num_envs, traj_len)).astype(np.int32),
            "dones": np.zeros((num_envs, traj_len), dtype=bool),
        }

        train_fn, data_arrays = make_train_offline(small_config, offline_data)
        rng = jax.random.PRNGKey(0)

        # Unpack the tuple and pass data_arrays explicitly to the JIT function
        result = jax.jit(train_fn)(rng, data_arrays)
        assert "train_state" in result
        assert "final_loss" in result

    def test_loss_decreases_or_finite(self, small_config):
        """With enough steps on a tiny dataset, loss should remain finite."""
        from src.planners.offline import make_train_offline

        small_config = {**small_config, "NUM_TRAIN_STEPS": 5}
        num_envs, traj_len, obs_dim = 1, 20, 8
        num_actions = small_config["NUM_ACTIONS"]

        offline_data = {
            "obs": np.random.randn(num_envs, traj_len, obs_dim).astype(np.float32),
            "actions": np.random.randint(0, num_actions, (num_envs, traj_len)).astype(np.int32),
            "dones": np.zeros((num_envs, traj_len), dtype=bool),
        }

        train_fn, data_arrays = make_train_offline(small_config, offline_data)
        result = jax.jit(train_fn)(jax.random.PRNGKey(0), data_arrays)

        loss = result["final_loss"]
        assert jnp.isfinite(loss)