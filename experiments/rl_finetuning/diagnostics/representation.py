"""Representation-space diagnostics: KL drift and CKA similarity.

KL drift measures how far the output distribution has moved from the
pretrained model.  CKA (Centred Kernel Alignment) measures similarity
of internal activations at the representation level.

All functions return JAX arrays and are fully JIT-compatible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from src.diffusion.forward import forward_process
from src.diffusion.schedules import ScheduleFn


def make_repr_drift_fn(
    apply_fn: Callable,
    schedule_fn: ScheduleFn,
    num_actions: int,
) -> Callable:
    """Build a JIT-compiled representation drift function.

    Returns a function producing four JAX scalars (kl_mean, kl_low,
    kl_mid, kl_high) — safe for ``jax.lax.scan`` / ``jax.lax.cond``.

    Args:
        apply_fn:    Eval apply fn (no dropout): (params, obs, z_t, t) -> logits.
        schedule_fn: alpha(t) noise schedule.
        num_actions: Action vocabulary size.

    Returns:
        JIT-compiled fn(params, ref_params, obs, acts, rng)
        -> (kl_mean, kl_low, kl_mid, kl_high) as JAX scalars.
    """
    _EPS = 1e-5

    def _kl_at_range(
        params: Any,
        ref_params: Any,
        obs: jax.Array,
        acts: jax.Array,
        rng: jax.Array,
        t_min: float,
        t_max: float,
    ) -> jax.Array:
        """Compute mean KL(ref || current) for t sampled in [t_min, t_max].

        Args:
            params:     Current model parameters.
            ref_params: Pretrained reference parameters.
            obs:        ``[B, obs_dim]`` observations.
            acts:       ``[B, H]`` action sequences.
            rng:        PRNG key.
            t_min:      Lower t bound.
            t_max:      Upper t bound.

        Returns:
            Scalar mean KL divergence.
        """
        B = obs.shape[0]
        rng, t_rng, mask_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (B,), minval=t_min, maxval=t_max)
        alpha_t = schedule_fn(t)
        z_t = forward_process(mask_rng, acts, alpha_t, num_actions)

        cur_logits = apply_fn(params, obs, z_t, t)
        ref_logits = apply_fn(jax.lax.stop_gradient(ref_params), obs, z_t, t)

        cur_log = jax.nn.log_softmax(cur_logits, axis=-1)
        ref_log = jax.nn.log_softmax(ref_logits, axis=-1)
        ref_prob = jnp.exp(ref_log)

        kl = (ref_prob * (ref_log - cur_log)).sum(-1)  # [B, H]
        return kl.mean()

    @jax.jit
    def repr_drift(
        params: Any,
        ref_params: Any,
        obs: jax.Array,
        acts: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute KL divergence drift from pretrained model.

        Args:
            params:     Current model parameters.
            ref_params: Pretrained reference parameters.
            obs:        ``[B, obs_dim]`` float32 observations.
            acts:       ``[B, H]`` int32 action sequences.
            rng:        PRNG key.

        Returns:
            Tuple of (kl_mean, kl_low_t, kl_mid_t, kl_high_t) as JAX scalars.
        """
        rng, r1, r2, r3, r4 = jax.random.split(rng, 5)
        kl_mean = _kl_at_range(params, ref_params, obs, acts, r1, _EPS, 1.0)
        kl_low = _kl_at_range(params, ref_params, obs, acts, r2, _EPS, 0.2)
        kl_mid = _kl_at_range(params, ref_params, obs, acts, r3, 0.3, 0.7)
        kl_high = _kl_at_range(params, ref_params, obs, acts, r4, 0.8, 1.0)
        return kl_mean, kl_low, kl_mid, kl_high

    return repr_drift


@jax.jit
def _linear_cka(x: jax.Array, y: jax.Array) -> jax.Array:
    """Linear CKA between activation matrices X and Y.

    CKA(X, Y) = ||Y.T @ X||_F^2 / (||X.T @ X||_F * ||Y.T @ Y||_F)

    Computed using the HSIC estimator with linear kernels.

    Args:
        x: ``[N, D_x]`` activation matrix from current model.
        y: ``[N, D_y]`` activation matrix from reference model.

    Returns:
        Scalar CKA value in [0, 1]; 1 = identical representations.
    """
    n = x.shape[0]
    h = jnp.eye(n) - jnp.ones((n, n)) / n

    kx = x @ x.T  # [N, N]
    ky = y @ y.T  # [N, N]

    hkx = h @ kx @ h  # centred K
    hky = h @ ky @ h  # centred L

    hsic_xy = jnp.sum(hkx * hky.T)
    hsic_xx = jnp.sum(hkx * hkx.T)
    hsic_yy = jnp.sum(hky * hky.T)

    return hsic_xy / (jnp.sqrt(hsic_xx * hsic_yy) + 1e-10)


def make_cka_fn(
    model_apply: Callable,
    schedule_fn: ScheduleFn,
    num_actions: int,
    cka_batch_size: int = 64,  # override via config CKA_BATCH_SIZE
) -> Callable:
    """Build a JIT-compiled CKA function returning a JAX scalar.

    Args:
        model_apply:    Eval apply fn: (params, obs, z_t, t) -> logits.
        schedule_fn:    alpha(t) schedule.
        num_actions:    Action vocabulary size.
        cka_batch_size: Number of samples for CKA (keep <= 128).

    Returns:
        JIT-compiled fn(params, ref_params, obs, acts, rng) -> cka_scalar.
    """
    _EPS = 1e-5

    @jax.jit
    def compute_cka(
        params: Any,
        ref_params: Any,
        obs: jax.Array,
        acts: jax.Array,
        rng: jax.Array,
    ) -> jax.Array:
        """Compute CKA similarity between current and reference representations.

        Args:
            params:     Current model parameters.
            ref_params: Pretrained reference parameters.
            obs:        ``[B, obs_dim]`` observations.
            acts:       ``[B, H]`` action sequences.
            rng:        PRNG key.

        Returns:
            Scalar CKA value.

        Note:
            Caller must ensure ``obs.shape[0] >= cka_batch_size``.  In
            practice ``batch_size`` (typically 128) should exceed the
            default ``cka_batch_size`` (64).
        """
        obs_b = jax.lax.dynamic_slice(obs, (0, 0), (cka_batch_size, obs.shape[1]))
        acts_b = jax.lax.dynamic_slice(acts, (0, 0), (cka_batch_size, acts.shape[1]))

        rng, t_rng, mask_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (cka_batch_size,), minval=0.3, maxval=0.7)
        alpha_t = schedule_fn(t)
        z_t = forward_process(mask_rng, acts_b, alpha_t, num_actions)

        cur_logits = model_apply(params, obs_b, z_t, t)  # [B, H, V]
        ref_logits = model_apply(ref_params, obs_b, z_t, t)  # [B, H, V]

        cur_repr = cur_logits.mean(axis=1)  # [B, V]
        ref_repr = ref_logits.mean(axis=1)  # [B, V]

        return _linear_cka(cur_repr, ref_repr)

    return compute_cka
