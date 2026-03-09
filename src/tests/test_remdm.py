"""Tests for src/models/remdm.py — noise schedules, forward process, loss, remasking, sampling."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.models.remdm import (
    cosine_schedule,
    linear_schedule,
    forward_process,
    compute_loss,
    remask_rescale,
    remask_cap,
    compute_sigma_conf,
    sample_plan,
    _sigma_max,
    STRATEGY_MAP,
)


# =============================================================================
# Noise schedules
# =============================================================================


class TestCosineSchedule:
    def test_boundary_t0(self):
        assert jnp.isclose(cosine_schedule(jnp.array(0.0)), 1.0)

    def test_boundary_t1(self):
        assert jnp.isclose(cosine_schedule(jnp.array(1.0)), 0.0, atol=1e-6)

    def test_monotonically_decreasing(self):
        ts = jnp.linspace(0.0, 1.0, 50)
        alphas = cosine_schedule(ts)
        diffs = alphas[1:] - alphas[:-1]
        assert jnp.all(diffs <= 1e-7)

    def test_array_input(self):
        ts = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        result = cosine_schedule(ts)
        assert result.shape == (5,)
        assert result.dtype == jnp.float32


class TestLinearSchedule:
    def test_boundary_t0(self):
        assert jnp.isclose(linear_schedule(jnp.array(0.0)), 1.0)

    def test_boundary_t1(self):
        assert jnp.isclose(linear_schedule(jnp.array(1.0)), 0.0)

    def test_midpoint(self):
        assert jnp.isclose(linear_schedule(jnp.array(0.5)), 0.5)

    def test_array_input(self):
        ts = jnp.array([0.0, 0.5, 1.0])
        expected = jnp.array([1.0, 0.5, 0.0])
        assert jnp.allclose(linear_schedule(ts), expected)


# =============================================================================
# Forward process
# =============================================================================


class TestForwardProcess:
    @pytest.fixture
    def setup(self):
        batch, horizon, num_actions = 8, 6, 5
        mask_token_id = num_actions
        rng = jax.random.PRNGKey(0)
        x_0 = jax.random.randint(rng, (batch, horizon), 0, num_actions)
        return rng, x_0, mask_token_id, batch, horizon

    def test_shape_preserved(self, setup):
        rng, x_0, mask_token_id, batch, horizon = setup
        alpha_t = jnp.ones(batch)
        z_t = forward_process(rng, x_0, alpha_t, mask_token_id)
        assert z_t.shape == (batch, horizon)

    def test_dtype_preserved(self, setup):
        rng, x_0, mask_token_id, batch, _ = setup
        alpha_t = jnp.ones(batch)
        z_t = forward_process(rng, x_0, alpha_t, mask_token_id)
        assert z_t.dtype == jnp.int32

    def test_fully_clean_at_alpha_one(self, setup):
        rng, x_0, mask_token_id, batch, _ = setup
        alpha_t = jnp.ones(batch)
        z_t = forward_process(rng, x_0, alpha_t, mask_token_id)
        assert jnp.array_equal(z_t, x_0)

    def test_fully_masked_at_alpha_zero(self, setup):
        rng, x_0, mask_token_id, batch, _ = setup
        alpha_t = jnp.zeros(batch)
        z_t = forward_process(rng, x_0, alpha_t, mask_token_id)
        assert jnp.all(z_t == mask_token_id)

    def test_deterministic_with_same_key(self, setup):
        rng, x_0, mask_token_id, batch, _ = setup
        alpha_t = jnp.full(batch, 0.5)
        z1 = forward_process(rng, x_0, alpha_t, mask_token_id)
        z2 = forward_process(rng, x_0, alpha_t, mask_token_id)
        assert jnp.array_equal(z1, z2)

    def test_different_key_gives_different_result(self, setup):
        _, x_0, mask_token_id, batch, _ = setup
        alpha_t = jnp.full(batch, 0.5)
        z1 = forward_process(jax.random.PRNGKey(0), x_0, alpha_t, mask_token_id)
        z2 = forward_process(jax.random.PRNGKey(1), x_0, alpha_t, mask_token_id)
        assert not jnp.array_equal(z1, z2)

    def test_scalar_alpha_t(self, setup):
        rng, x_0, mask_token_id, _, _ = setup
        alpha_t = jnp.array(0.5)
        z_t = forward_process(rng, x_0, alpha_t, mask_token_id)
        assert z_t.shape == x_0.shape

    def test_jit_compatible(self, setup):
        rng, x_0, mask_token_id, batch, _ = setup
        alpha_t = jnp.full(batch, 0.5)
        jit_fp = jax.jit(forward_process, static_argnums=(3,))
        z_t = jit_fp(rng, x_0, alpha_t, mask_token_id)
        assert z_t.shape == x_0.shape


# =============================================================================
# Compute loss
# =============================================================================


class TestComputeLoss:
    @pytest.fixture
    def loss_setup(self, dummy_model_apply, dummy_params, rng):
        batch, horizon, num_actions = 4, 6, 5
        x_0 = jax.random.randint(rng, (batch, horizon), 0, num_actions)
        obs = jax.random.normal(rng, (batch, 8))
        return dummy_model_apply, dummy_params, rng, x_0, obs, num_actions

    def test_returns_scalar_loss(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        loss, info = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule)
        assert loss.shape == ()

    def test_info_dict_keys(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        _, info = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule)
        assert "loss" in info
        assert "mean_t" in info
        assert "frac_masked" in info

    def test_loss_non_negative(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        loss, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule)
        assert loss >= 0.0

    def test_mean_t_in_range(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        _, info = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule)
        assert 0.0 < float(info["mean_t"]) < 1.0

    def test_frac_masked_in_range(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        _, info = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule)
        assert 0.0 <= float(info["frac_masked"]) <= 1.0

    def test_works_with_linear_schedule(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        loss, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, linear_schedule)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_jit_compatible(self, loss_setup):
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        jit_loss = jax.jit(compute_loss, static_argnums=(0, 5, 6, 7))
        loss, info = jit_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule, 0.0)
        assert jnp.isfinite(loss)

    def test_sigma_zero_matches_default(self, loss_setup):
        """Passing sigma_t=0.0 explicitly should match omitting it."""
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        loss_default, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule)
        loss_explicit, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule, sigma_t=0.0)
        assert jnp.isclose(loss_default, loss_explicit)

    def test_sigma_positive_reduces_weight(self, loss_setup):
        """ReMDM NELBO with sigma_t > 0 scales the weight by (1 - sigma_t), so loss should decrease."""
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        loss_mdlm, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule, sigma_t=0.0)
        loss_remdm, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule, sigma_t=0.5)
        assert float(loss_remdm) < float(loss_mdlm)

    def test_sigma_one_gives_zero_loss(self, loss_setup):
        """sigma_t=1.0 makes weight = 0, so loss should be zero."""
        apply_fn, params, rng, x_0, obs, num_actions = loss_setup
        loss, _ = compute_loss(apply_fn, params, rng, x_0, obs, num_actions, cosine_schedule, sigma_t=1.0)
        assert jnp.isclose(loss, 0.0)


# =============================================================================
# Remasking strategies
# =============================================================================


class TestSigmaMax:
    def test_alpha_t_zero_alpha_s_zero(self):
        result = _sigma_max(jnp.array(0.0), jnp.array(0.0))
        assert jnp.isclose(result, 1.0)

    def test_alpha_t_one_alpha_s_zero(self):
        result = _sigma_max(jnp.array(1.0), jnp.array(0.0))
        assert jnp.isclose(result, 1.0)

    def test_alpha_s_one(self):
        result = _sigma_max(jnp.array(0.5), jnp.array(1.0))
        # (1 - 1) / 0.5 = 0
        assert jnp.isclose(result, 0.0)

    def test_clamped_to_one(self):
        # (1 - 0) / 0.1 = 10, clamped to 1
        result = _sigma_max(jnp.array(0.1), jnp.array(0.0))
        assert jnp.isclose(result, 1.0)


class TestRemaskRescale:
    def test_proportional_to_eta(self):
        alpha_t = jnp.array(0.5)
        alpha_s = jnp.array(0.3)
        s1 = remask_rescale(alpha_t, alpha_s, 0.5)
        s2 = remask_rescale(alpha_t, alpha_s, 1.0)
        assert jnp.isclose(s1 * 2, s2)

    def test_eta_zero_gives_zero(self):
        result = remask_rescale(jnp.array(0.5), jnp.array(0.3), 0.0)
        assert jnp.isclose(result, 0.0)


class TestRemaskCap:
    def test_capped_at_eta(self):
        # When sigma_max is large, result should be capped at eta
        result = remask_cap(jnp.array(0.5), jnp.array(0.0), 0.3)
        sigma_max_val = _sigma_max(jnp.array(0.5), jnp.array(0.0))
        assert float(result) <= 0.3 + 1e-6

    def test_sigma_max_smaller_than_eta(self):
        # When sigma_max < eta, result = sigma_max
        alpha_t = jnp.array(0.8)
        alpha_s = jnp.array(0.85)
        sm = _sigma_max(alpha_t, alpha_s)
        result = remask_cap(alpha_t, alpha_s, 10.0)
        assert jnp.isclose(result, sm)


class TestRemaskConf:
    def test_output_shape(self):
        batch, horizon = 4, 6
        alpha_t = jnp.array(0.5)
        alpha_s = jnp.array(0.3)
        psi = jnp.full((batch, horizon), 0.8)
        is_unmasked = jnp.ones((batch, horizon), dtype=bool)
        result = compute_sigma_conf(alpha_t, alpha_s, 0.5, psi, is_unmasked)
        assert result.shape == (batch, horizon)

    def test_high_confidence_low_remask(self):
        batch, horizon = 1, 4
        alpha_t = jnp.array(0.5)
        alpha_s = jnp.array(0.3)
        is_unmasked = jnp.ones((batch, horizon), dtype=bool)
        # One token decoded with low confidence, rest high.
        # The low-confidence token should get higher sigma.
        psi = jnp.array([[0.99, 0.99, 0.99, 0.01]])
        sigma = compute_sigma_conf(alpha_t, alpha_s, 0.5, psi, is_unmasked)
        # Token 3 (psi=0.01) should be remasked more than token 0 (psi=0.99)
        assert float(sigma[0, 3]) > float(sigma[0, 0])


# =============================================================================
# Sample plan
# =============================================================================


class TestSamplePlan:
    @pytest.fixture
    def plan_setup(self, dummy_model_apply, dummy_params, rng):
        batch = 4
        obs = jnp.zeros((batch, 8))
        return dummy_model_apply, dummy_params, rng, obs, batch

    def test_output_shape(self, plan_setup):
        apply_fn, params, rng, obs, batch = plan_setup
        plan = sample_plan(apply_fn, params, rng, obs, 5, 6, 3, cosine_schedule, "rescale")
        assert plan.shape == (batch, 6)

    def test_output_dtype(self, plan_setup):
        apply_fn, params, rng, obs, batch = plan_setup
        plan = sample_plan(apply_fn, params, rng, obs, 5, 6, 3, cosine_schedule, "rescale")
        assert plan.dtype == jnp.int32

    def test_actions_in_valid_range(self, plan_setup):
        apply_fn, params, rng, obs, _ = plan_setup
        num_actions = 5
        plan = sample_plan(apply_fn, params, rng, obs, num_actions, 6, 3, cosine_schedule, "rescale")
        assert jnp.all(plan >= 0)
        assert jnp.all(plan < num_actions)

    def test_deterministic_with_same_key(self, plan_setup):
        apply_fn, params, rng, obs, _ = plan_setup
        p1 = sample_plan(apply_fn, params, rng, obs, 5, 6, 3, cosine_schedule, "rescale")
        p2 = sample_plan(apply_fn, params, rng, obs, 5, 6, 3, cosine_schedule, "rescale")
        assert jnp.array_equal(p1, p2)

    @pytest.mark.parametrize("strategy", ["rescale", "cap", "conf"])
    def test_all_strategies_run(self, plan_setup, strategy):
        apply_fn, params, rng, obs, batch = plan_setup
        plan = sample_plan(
            apply_fn, params, rng, obs, 5, 6, 3, cosine_schedule, strategy, eta=0.5
        )
        assert plan.shape == (batch, 6)

    def test_jit_compatible(self, plan_setup):
        apply_fn, params, rng, obs, batch = plan_setup
        jit_plan = jax.jit(
            sample_plan, static_argnums=(0, 4, 5, 6, 7, 8, 10)
        )
        plan = jit_plan(
            apply_fn, params, rng, obs, 5, 6, 3, cosine_schedule, "rescale",
            0.5, False,
        )
        assert plan.shape == (batch, 6)

    def test_single_step(self, plan_setup):
        apply_fn, params, rng, obs, batch = plan_setup
        plan = sample_plan(apply_fn, params, rng, obs, 5, 6, 1, cosine_schedule, "rescale")
        assert plan.shape == (batch, 6)

    def test_loop_mode_runs(self, plan_setup):
        apply_fn, params, rng, obs, batch = plan_setup
        plan = sample_plan(
            apply_fn, params, rng, obs, 5, 6, 10, cosine_schedule, "cap",
            eta=0.5, use_loop=True, t_on=0.55, t_off=0.05,
        )
        assert plan.shape == (batch, 6)
        assert jnp.all(plan >= 0)
        assert jnp.all(plan < 5)


class TestStrategyMap:
    def test_contains_all_strategies(self):
        assert set(STRATEGY_MAP.keys()) == {"rescale", "cap", "conf"}

    def test_unique_values(self):
        vals = list(STRATEGY_MAP.values())
        assert len(vals) == len(set(vals))
