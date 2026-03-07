"""Tests for src/envs/wrappers.py — environment wrappers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.envs.wrappers import (
    DiscreteTokenizationWrapper,
    OfflineTrajectoryWrapper,
    PlannerWrapper,
    SequenceHistoryWrapper,
)
from src.tests.conftest import DummyGymnaxEnv


# =============================================================================
# SequenceHistoryWrapper
# =============================================================================


class TestSequenceHistoryWrapper:
    @pytest.fixture
    def wrapped(self):
        env = DummyGymnaxEnv(obs_dim=4, num_actions=3)
        return SequenceHistoryWrapper(env, history_len=3, obs_shape=(4,))

    def test_reset_obs_shape(self, wrapped):
        obs, state = wrapped.reset(jax.random.PRNGKey(0))
        assert obs.shape == (4,)

    def test_reset_history_shapes(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        assert state.obs_history.shape == (3, 4)
        assert state.act_history.shape == (3,)

    def test_reset_history_filled_with_initial_obs(self, wrapped):
        obs, state = wrapped.reset(jax.random.PRNGKey(0))
        for i in range(3):
            assert jnp.allclose(state.obs_history[i], obs)

    def test_reset_act_history_zeros(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        assert jnp.all(state.act_history == 0)

    def test_step_updates_history(self, wrapped):
        obs, state = wrapped.reset(jax.random.PRNGKey(0))
        action = jnp.int32(1)
        obs_next, state_next, reward, done, info = wrapped.step(
            jax.random.PRNGKey(1), state, action
        )
        assert jnp.allclose(state_next.obs_history[-1], obs_next)
        assert state_next.act_history[-1] == 1

    def test_step_rolls_history(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        old_obs_0 = state.obs_history[1].copy()
        _, state2, _, _, _ = wrapped.step(jax.random.PRNGKey(1), state, jnp.int32(0))
        # After rolling, the old second element should now be first
        assert jnp.allclose(state2.obs_history[0], old_obs_0)


# =============================================================================
# DiscreteTokenizationWrapper
# =============================================================================


class TestDiscreteTokenizationWrapper:
    @pytest.fixture
    def wrapped(self):
        env = DummyGymnaxEnv(obs_dim=4, num_actions=3)
        obs_min = jnp.array([-2.0, -2.0, -2.0, -2.0])
        obs_max = jnp.array([2.0, 2.0, 2.0, 2.0])
        return DiscreteTokenizationWrapper(env, n_bins=10, obs_min=obs_min, obs_max=obs_max)

    def test_reset_returns_int32(self, wrapped):
        obs, _ = wrapped.reset(jax.random.PRNGKey(0))
        assert obs.dtype == jnp.int32

    def test_reset_obs_in_range(self, wrapped):
        obs, _ = wrapped.reset(jax.random.PRNGKey(0))
        assert jnp.all(obs >= 0)
        assert jnp.all(obs < 10)

    def test_step_returns_int32(self, wrapped):
        obs, state = wrapped.reset(jax.random.PRNGKey(0))
        obs_next, _, _, _, _ = wrapped.step(jax.random.PRNGKey(1), state, jnp.int32(0))
        assert obs_next.dtype == jnp.int32

    def test_tokenize_clamps_correctly(self, wrapped):
        # Test _tokenize directly with extreme values
        extreme_obs = jnp.array([100.0, -100.0, 0.0, 2.0])
        tokens = wrapped._tokenize(extreme_obs)
        assert jnp.all(tokens >= 0)
        assert jnp.all(tokens < 10)

    def test_tokenize_boundary_lower(self, wrapped):
        obs = jnp.array([-2.0, -2.0, -2.0, -2.0])
        tokens = wrapped._tokenize(obs)
        assert jnp.all(tokens == 0)

    def test_tokenize_boundary_upper(self, wrapped):
        obs = jnp.array([2.0, 2.0, 2.0, 2.0])
        tokens = wrapped._tokenize(obs)
        assert jnp.all(tokens == 9)  # n_bins - 1


# =============================================================================
# PlannerWrapper
# =============================================================================


class TestPlannerWrapper:
    def test_replan_every_exceeds_horizon_raises(self):
        env = DummyGymnaxEnv()
        with pytest.raises(ValueError, match="replan_every.*must be <= plan_horizon"):
            PlannerWrapper(
                env,
                num_envs=2,
                plan_horizon=4,
                replan_every=5,
                planner_apply_fn=lambda *a: None,
            )

    def test_replan_every_equals_horizon_ok(self):
        env = DummyGymnaxEnv()
        wrapper = PlannerWrapper(
            env,
            num_envs=2,
            plan_horizon=4,
            replan_every=4,
            planner_apply_fn=lambda *a: None,
        )
        assert wrapper.replan_every == 4

    def test_reset_creates_zero_plan(self):
        env = DummyGymnaxEnv()

        # We need a batched environment for PlannerWrapper
        class FakeBatchedEnv:
            _env = env
            def __getattr__(self, name):
                return getattr(self._env, name)
            def reset(self, key, params=None):
                obs, state = self._env.reset(key, params)
                return obs[None].repeat(2, axis=0), state
            def step(self, key, state, action, params=None):
                obs, state, r, d, i = self._env.step(key, state, action[0], params)
                return (
                    obs[None].repeat(2, axis=0),
                    state,
                    jnp.array([r, r]),
                    jnp.array([d, d]),
                    i,
                )

        batched = FakeBatchedEnv()
        wrapper = PlannerWrapper(
            batched,
            num_envs=2,
            plan_horizon=4,
            replan_every=2,
            planner_apply_fn=lambda *a: jnp.zeros((2, 4), dtype=jnp.int32),
        )
        _, state = wrapper.reset(jax.random.PRNGKey(0))
        assert state.current_plan.shape == (2, 4)
        assert jnp.all(state.current_plan == 0)
        assert state.plan_step == 0


# =============================================================================
# OfflineTrajectoryWrapper
# =============================================================================


class TestOfflineTrajectoryWrapper:
    @pytest.fixture
    def wrapped(self):
        env = DummyGymnaxEnv(obs_dim=4)
        return OfflineTrajectoryWrapper(env, max_size=10, obs_shape=(4,))

    def test_reset_empty_buffer(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        assert state.buf_obs.shape == (10, 4)
        assert state.buf_act.shape == (10,)
        assert state.buf_reward.shape == (10,)
        assert state.buf_done.shape == (10,)
        assert state.buf_next_obs.shape == (10, 4)
        assert int(state.write_idx) == 0
        assert int(state.num_valid) == 0

    def test_step_increments_buffer(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        _, state, _, _, _ = wrapped.step(jax.random.PRNGKey(1), state, jnp.int32(0))
        assert int(state.write_idx) == 1
        assert int(state.num_valid) == 1

    def test_buffer_wraps_around(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        for i in range(12):
            _, state, _, _, _ = wrapped.step(
                jax.random.PRNGKey(i + 1), state, jnp.int32(i % 3)
            )
        assert int(state.write_idx) == 2  # 12 % 10
        assert int(state.num_valid) == 10

    def test_sample_sequences_shape(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        for i in range(10):
            _, state, _, _, _ = wrapped.step(
                jax.random.PRNGKey(i + 1), state, jnp.int32(0)
            )
        obs, act, rew, done, next_obs = wrapped.sample_sequences(
            jax.random.PRNGKey(99), state, n_samples=3, seq_len=4
        )
        assert obs.shape == (3, 4, 4)
        assert act.shape == (3, 4)
        assert rew.shape == (3, 4)
        assert done.shape == (3, 4)
        assert next_obs.shape == (3, 4, 4)

    def test_sample_sequences_deterministic(self, wrapped):
        _, state = wrapped.reset(jax.random.PRNGKey(0))
        for i in range(10):
            _, state, _, _, _ = wrapped.step(
                jax.random.PRNGKey(i + 1), state, jnp.int32(0)
            )
        rng = jax.random.PRNGKey(42)
        r1 = wrapped.sample_sequences(rng, state, n_samples=2, seq_len=3)
        r2 = wrapped.sample_sequences(rng, state, n_samples=2, seq_len=3)
        assert jnp.allclose(r1[0], r2[0])
