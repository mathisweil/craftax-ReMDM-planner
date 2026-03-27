"""Representation-space diagnostics: KL drift and CKA similarity.

KL drift measures how far the output distribution has moved from the
pretrained model.  CKA (Centred Kernel Alignment) measures similarity
of internal activations at the representation level, independently of
parameter values.

CKA is computed on a small fixed batch (N ≤ 128) to stay memory-safe
on a single GPU.  It should be called every ``cka_every`` iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from src.diffusion.forward import forward_process
from src.diffusion.schedules import ScheduleFn


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ReprDriftResult:
    """Output of a representation drift computation.

    Args:
        kl_mean:   Mean KL(ref || current) over the batch and masked positions.
        kl_low_t:  Mean KL measured at t ~ U(0.05, 0.2) — low-noise regime.
        kl_mid_t:  Mean KL measured at t ~ U(0.3, 0.7) — mid-noise regime.
        kl_high_t: Mean KL measured at t ~ U(0.8, 1.0) — high-noise regime.
    """

    kl_mean: float
    kl_low_t: float
    kl_mid_t: float
    kl_high_t: float


@dataclass
class CKAResult:
    """Output of a CKA similarity computation.

    Args:
        cka: Scalar linear CKA between current and reference activations.
    """

    cka: float


# ---------------------------------------------------------------------------
# KL drift
# ---------------------------------------------------------------------------


def make_repr_drift_fn(
    apply_fn: Callable,
    schedule_fn: ScheduleFn,
    num_actions: int,
) -> Callable:
    """Build a JIT-compiled representation drift function.

    Args:
        apply_fn:    Eval apply fn (no dropout): (params, obs, z_t, t) -> logits.
        schedule_fn: alpha(t) noise schedule.
        num_actions: Action vocabulary size.

    Returns:
        JIT-compiled fn(params, ref_params, obs, acts, rng) -> ReprDriftResult.
    """
    _EPS = 1e-5

    def _kl_at_range(params, ref_params, obs, acts, rng, t_min, t_max):
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

        # KL(ref || current): how much has current diverged from ref?
        kl = (ref_prob * (ref_log - cur_log)).sum(-1)  # [B, H]
        return kl.mean()

    @jax.jit
    def _drift(params, ref_params, obs, acts, rng):
        rng, r1, r2, r3, r4 = jax.random.split(rng, 5)
        kl_mean = _kl_at_range(params, ref_params, obs, acts, r1, _EPS, 1.0)
        kl_low = _kl_at_range(params, ref_params, obs, acts, r2, _EPS, 0.2)
        kl_mid = _kl_at_range(params, ref_params, obs, acts, r3, 0.3, 0.7)
        kl_high = _kl_at_range(params, ref_params, obs, acts, r4, 0.8, 1.0)
        return kl_mean, kl_low, kl_mid, kl_high

    def compute_repr_drift(params, ref_params, obs, acts, rng) -> ReprDriftResult:
        """Compute KL divergence drift from pretrained model.

        Args:
            params:     Current model parameters.
            ref_params: Pretrained reference parameters.
            obs:        ``[B, obs_dim]`` float32 observations.
            acts:       ``[B, H]`` int32 action sequences.
            rng:        PRNG key.

        Returns:
            ``ReprDriftResult`` with KL values at different noise levels.
        """
        kl_mean, kl_low, kl_mid, kl_high = _drift(params, ref_params, obs, acts, rng)
        return ReprDriftResult(
            kl_mean=float(kl_mean),
            kl_low_t=float(kl_low),
            kl_mid_t=float(kl_mid),
            kl_high_t=float(kl_high),
        )

    return compute_repr_drift


# ---------------------------------------------------------------------------
# CKA (Centred Kernel Alignment)
# ---------------------------------------------------------------------------


@jax.jit
def _linear_cka(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
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
    # Centre: H @ K @ H where H = I - 1/n * ones
    # For linear kernel K = X @ X.T, centred HSIC = ||Hx.T @ Hy||_F^2 ...
    # Using the equivalent: HSIC(K, L) = tr(KHLH) / (n-1)^2
    # With K = X@X.T and L = Y@Y.T
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
    cka_batch_size: int = 64,
) -> Callable:
    """Build a function that computes CKA between current and reference activations.

    CKA is computed at the final pre-head representation (penultimate layer output).
    To extract intermediate activations, the model is called with a dummy
    ``flax.linen.capture_intermediates`` context; if unavailable, we use the
    final logits as a proxy (less informative but always available).

    Args:
        model_apply:    Eval apply fn: (params, obs, z_t, t) -> logits ``[B, H, V]``.
        schedule_fn:    alpha(t) schedule.
        num_actions:    Action vocabulary size.
        cka_batch_size: Number of samples for CKA (keep ≤ 128 to avoid OOM).

    Returns:
        fn(params, ref_params, obs, acts, rng) -> CKAResult.
    """
    _EPS = 1e-5

    def compute_cka(params, ref_params, obs, acts, rng) -> CKAResult:
        """Compute CKA similarity between current and reference model representations.

        Uses mean-pooled logits as a proxy for the final representation.
        N is capped to cka_batch_size to bound memory usage.

        Args:
            params:     Current model parameters.
            ref_params: Pretrained reference parameters.
            obs:        ``[B, obs_dim]`` observations (uses first cka_batch_size rows).
            acts:       ``[B, H]`` action sequences.
            rng:        PRNG key.

        Returns:
            ``CKAResult`` with scalar CKA value.
        """
        B = min(obs.shape[0], cka_batch_size)
        obs_b = obs[:B]
        acts_b = acts[:B]

        rng, t_rng, mask_rng = jax.random.split(rng, 3)
        t = jax.random.uniform(t_rng, (B,), minval=0.3, maxval=0.7)
        alpha_t = schedule_fn(t)
        z_t = forward_process(mask_rng, acts_b, alpha_t, num_actions)

        # Use logits (mean-pooled over H) as the representation proxy
        cur_logits = model_apply(params, obs_b, z_t, t)           # [B, H, V]
        ref_logits = model_apply(ref_params, obs_b, z_t, t)       # [B, H, V]

        cur_repr = cur_logits.mean(axis=1)   # [B, V]
        ref_repr = ref_logits.mean(axis=1)   # [B, V]

        cka_val = _linear_cka(cur_repr, ref_repr)
        return CKAResult(cka=float(cka_val))

    return compute_cka


# ---------------------------------------------------------------------------
# Activation norm statistics
# ---------------------------------------------------------------------------


@dataclass
class ActivationNormStats:
    """Summary statistics of activation norms at a single checkpoint.

    Args:
        mean:   Mean L2 norm of activation vectors.
        std:    Standard deviation of L2 norms.
        p50:    Median L2 norm.
        p90:    90th percentile L2 norm.
    """

    mean: float
    std: float
    p50: float
    p90: float


def compute_activation_norm_stats(
    model_apply: Callable,
    params: Any,
    obs: jnp.ndarray,
    acts: jnp.ndarray,
    rng: jax.Array,
    schedule_fn: ScheduleFn,
    num_actions: int,
) -> ActivationNormStats:
    """Compute statistics of output logit norms as activation proxy.

    Args:
        model_apply: Eval apply fn.
        params:      Model parameters.
        obs:         ``[B, obs_dim]`` observations.
        acts:        ``[B, H]`` action sequences.
        rng:         PRNG key.
        schedule_fn: Noise schedule.
        num_actions: Action vocabulary size.

    Returns:
        ``ActivationNormStats`` summarising ||logits||_2 per sample.
    """
    B = obs.shape[0]
    rng, t_rng, mask_rng = jax.random.split(rng, 3)
    t = jax.random.uniform(t_rng, (B,), minval=0.3, maxval=0.7)
    alpha_t = schedule_fn(t)
    z_t = forward_process(mask_rng, acts, alpha_t, num_actions)

    logits = model_apply(params, obs, z_t, t)        # [B, H, V]
    norms = jnp.linalg.norm(logits, axis=-1).mean(axis=-1)  # [B]

    norms_np = jax.device_get(norms)
    import numpy as np

    return ActivationNormStats(
        mean=float(np.mean(norms_np)),
        std=float(np.std(norms_np)),
        p50=float(np.percentile(norms_np, 50)),
        p90=float(np.percentile(norms_np, 90)),
    )
