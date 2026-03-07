"""Tests for src/models/denoiser.py — SinusoidalPosEmbed, TransformerBlock, DenoisingTransformer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from src.models.denoiser import (
    SinusoidalPosEmbed,
    TransformerBlock,
    DenoisingTransformer,
)


# =============================================================================
# SinusoidalPosEmbed
# =============================================================================


class TestSinusoidalPosEmbed:
    def test_output_shape_even(self):
        model = SinusoidalPosEmbed(d_model=16)
        params = model.init(jax.random.PRNGKey(0), jnp.array([0.0, 0.5, 1.0]))
        out = model.apply(params, jnp.array([0.0, 0.5, 1.0]))
        assert out.shape == (3, 16)

    def test_output_shape_odd(self):
        model = SinusoidalPosEmbed(d_model=17)
        params = model.init(jax.random.PRNGKey(0), jnp.array([0.0, 0.5]))
        out = model.apply(params, jnp.array([0.0, 0.5]))
        assert out.shape == (2, 17)

    def test_deterministic(self):
        model = SinusoidalPosEmbed(d_model=16)
        x = jnp.array([0.1, 0.5, 0.9])
        params = model.init(jax.random.PRNGKey(0), x)
        out1 = model.apply(params, x)
        out2 = model.apply(params, x)
        assert jnp.allclose(out1, out2)

    def test_different_inputs_different_outputs(self):
        model = SinusoidalPosEmbed(d_model=16)
        x = jnp.array([0.0, 1.0])
        params = model.init(jax.random.PRNGKey(0), x)
        out = model.apply(params, x)
        assert not jnp.allclose(out[0], out[1])

    def test_scalar_input(self):
        model = SinusoidalPosEmbed(d_model=8)
        x = jnp.array(0.5)
        params = model.init(jax.random.PRNGKey(0), x)
        out = model.apply(params, x)
        assert out.shape == (8,)


# =============================================================================
# TransformerBlock
# =============================================================================


class TestTransformerBlock:
    def test_output_shape_preserved(self):
        d_model, n_heads = 32, 4
        block = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=64, deterministic=True
        )
        x = jnp.ones((2, 5, d_model))
        params = block.init(jax.random.PRNGKey(0), x)
        out = block.apply(params, x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (not identity) but be influenced by it."""
        d_model, n_heads = 32, 4
        block = TransformerBlock(
            d_model=d_model, n_heads=n_heads, d_ff=64, deterministic=True
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 3, d_model))
        params = block.init(jax.random.PRNGKey(0), x)
        out = block.apply(params, x)
        # Not identical (non-trivial transform)
        assert not jnp.allclose(out, x, atol=1e-5)

    def test_dtype_float32(self):
        block = TransformerBlock(d_model=16, n_heads=2, d_ff=32, deterministic=True)
        x = jnp.ones((1, 2, 16))
        params = block.init(jax.random.PRNGKey(0), x)
        out = block.apply(params, x)
        assert out.dtype == jnp.float32


# =============================================================================
# DenoisingTransformer
# =============================================================================


class TestDenoisingTransformer:
    @pytest.fixture
    def model_and_params(self):
        num_actions, plan_horizon = 5, 4
        model = DenoisingTransformer(
            num_actions=num_actions,
            plan_horizon=plan_horizon,
            d_model=32,
            n_heads=2,
            n_layers=1,
            d_ff=64,
            obs_encoder_layers=1,
            obs_encoder_width=32,
            dropout_rate=0.0,
        )
        rng = jax.random.PRNGKey(0)
        batch, obs_dim = 2, 8
        dummy_obs = jnp.zeros((batch, obs_dim))
        dummy_act = jnp.zeros((batch, plan_horizon), dtype=jnp.int32)
        dummy_t = jnp.zeros((batch,))
        params = model.init(rng, dummy_obs, dummy_act, dummy_t)
        return model, params, num_actions, plan_horizon, batch, obs_dim

    def test_output_shape(self, model_and_params):
        model, params, num_actions, plan_horizon, batch, obs_dim = model_and_params
        obs = jnp.ones((batch, obs_dim))
        actions = jnp.zeros((batch, plan_horizon), dtype=jnp.int32)
        t = jnp.array([0.5, 0.3])
        logits = model.apply(params, obs, actions, t)
        assert logits.shape == (batch, plan_horizon, num_actions)

    def test_output_dtype(self, model_and_params):
        model, params, _, plan_horizon, batch, obs_dim = model_and_params
        obs = jnp.ones((batch, obs_dim))
        actions = jnp.zeros((batch, plan_horizon), dtype=jnp.int32)
        t = jnp.array([0.5, 0.3])
        logits = model.apply(params, obs, actions, t)
        assert logits.dtype == jnp.float32

    def test_handles_mask_token(self, model_and_params):
        model, params, num_actions, plan_horizon, batch, obs_dim = model_and_params
        obs = jnp.ones((batch, obs_dim))
        mask_token_id = num_actions
        actions = jnp.full((batch, plan_horizon), mask_token_id, dtype=jnp.int32)
        t = jnp.array([0.5, 0.3])
        logits = model.apply(params, obs, actions, t)
        assert logits.shape == (batch, plan_horizon, num_actions)
        assert jnp.all(jnp.isfinite(logits))

    def test_deterministic_mode(self, model_and_params):
        model, params, _, plan_horizon, batch, obs_dim = model_and_params
        obs = jnp.ones((batch, obs_dim))
        actions = jnp.zeros((batch, plan_horizon), dtype=jnp.int32)
        t = jnp.array([0.5, 0.3])
        out1 = model.apply(params, obs, actions, t, deterministic=True)
        out2 = model.apply(params, obs, actions, t, deterministic=True)
        assert jnp.allclose(out1, out2)

    def test_jit_compatible(self, model_and_params):
        model, params, _, plan_horizon, batch, obs_dim = model_and_params
        obs = jnp.ones((batch, obs_dim))
        actions = jnp.zeros((batch, plan_horizon), dtype=jnp.int32)
        t = jnp.array([0.5, 0.3])

        @jax.jit
        def forward(params, obs, actions, t):
            return model.apply(params, obs, actions, t)

        logits = forward(params, obs, actions, t)
        assert jnp.all(jnp.isfinite(logits))

    def test_different_timesteps_different_outputs(self, model_and_params):
        model, params, _, plan_horizon, batch, obs_dim = model_and_params
        obs = jnp.ones((batch, obs_dim))
        actions = jnp.zeros((batch, plan_horizon), dtype=jnp.int32)
        out1 = model.apply(params, obs, actions, jnp.array([0.0, 0.0]))
        out2 = model.apply(params, obs, actions, jnp.array([1.0, 1.0]))
        assert not jnp.allclose(out1, out2)

    def test_batch_size_one(self):
        model = DenoisingTransformer(
            num_actions=3, plan_horizon=2, d_model=16, n_heads=2,
            n_layers=1, d_ff=32, obs_encoder_layers=1, obs_encoder_width=16,
        )
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((1, 4))
        act = jnp.zeros((1, 2), dtype=jnp.int32)
        t = jnp.zeros((1,))
        params = model.init(rng, obs, act, t)
        logits = model.apply(params, obs, act, t)
        assert logits.shape == (1, 2, 3)
