"""Timestep (t) distribution diagnostics for RL fine-tuning analysis.

Analyses how gradient contributions vary across the continuous diffusion
time t in [0, 1], partitioned into N_BINS equal bins.

All functions return JAX arrays and are fully JIT-compatible.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

from src.diffusion.loss import compute_loss
from src.diffusion.schedules import ScheduleFn

N_BINS: int = 10  # number of t bins for analysis


def make_t_analysis_fn(
    apply_fn: Callable,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    num_actions: int,
    sigma_t: float = 0.0,
    n_bins: int = N_BINS,
) -> Callable:
    """Build a JIT-compiled t-analysis function.

    The returned function computes per-bin gradient norms via ``jax.vmap``
    and measures alignment between low-t and high-t gradients.  All
    outputs are JAX arrays, safe for use inside ``jax.lax.scan``.

    Args:
        apply_fn:          Training apply fn.
        schedule_fn:       alpha(t) schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Action vocabulary size.
        sigma_t:           ReMDM remasking correction.
        n_bins:            Number of equal t bins to analyse.

    Returns:
        JIT-compiled fn(params, acts, obs, valid, advantages, rng)
        -> (bin_norms, low_high_cos, norm_low_t, norm_high_t).
    """
    _EPS = 1e-5
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)
    # Pre-compute bin boundaries as arrays: [n_bins, 2]
    bin_lo = jnp.maximum(bin_edges[:-1], _EPS)  # [n_bins]
    bin_hi = bin_edges[1:]  # [n_bins]

    def _grad_flat_at_range(
        params: Any,
        acts: jax.Array,
        obs: jax.Array,
        valid: jax.Array,
        advantages: jax.Array,
        rng: jax.Array,
        t_lo: jax.Array,
        t_hi: jax.Array,
    ) -> jax.Array:
        """Compute flattened gradient for a single t-range.

        Args:
            params:     Model parameters.
            acts:       ``[B, H]`` action sequences.
            obs:        ``[B, obs_dim]`` observations.
            valid:      ``[B]`` validity mask.
            advantages: ``[B]`` advantage weights.
            rng:        PRNG key.
            t_lo:       Lower t bound (scalar).
            t_hi:       Upper t bound (scalar).

        Returns:
            Flattened gradient vector ``[D_total]``.
        """

        def loss_in_range(p: Any) -> jax.Array:
            # Use jnp.where to handle degenerate ranges without Python branching
            safe_lo = jnp.maximum(t_lo, _EPS)
            safe_hi = jnp.maximum(t_hi, safe_lo + _EPS)
            loss_val, _ = compute_loss(
                apply_fn,
                p,
                rng,
                acts,
                obs,
                valid,
                num_actions,
                schedule_fn,
                schedule_deriv_fn,
                sigma_t=sigma_t,
                advantages=advantages,
                t_min=safe_lo,
                t_max=safe_hi,
            )
            return loss_val

        g = jax.grad(loss_in_range)(params)
        flat = jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(g)])
        return flat  # [D_total]

    @jax.jit
    def t_analysis(
        params: Any,
        acts: jax.Array,
        obs: jax.Array,
        valid: jax.Array,
        advantages: jax.Array,
        rng: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        """Compute per-t-bin gradient norms and low/high-t alignment.

        Args:
            params:     Current model parameters.
            acts:       ``[B, H]`` int32 action sequences.
            obs:        ``[B, obs_dim]`` float32 observations.
            valid:      ``[B]`` validity mask.
            advantages: ``[B]`` advantage weights.
            rng:        PRNG key.

        Returns:
            Tuple of:
            - bin_norms:    ``[n_bins]`` per-bin gradient L2 norms.
            - low_high_cos: Scalar cosine similarity between low-t and high-t gradients.
            - norm_low_t:   Scalar L2 norm of low-t gradient.
            - norm_high_t:  Scalar L2 norm of high-t gradient.
        """
        all_rngs = jax.random.split(rng, n_bins + 2)

        # Compute per-bin gradient norms by scanning over bins
        # (vmap over bin index doesn't work cleanly with grad; use scan instead)
        def _bin_step(carry: None, bin_idx: jax.Array) -> tuple[None, jax.Array]:
            flat = _grad_flat_at_range(
                params,
                acts,
                obs,
                valid,
                advantages,
                all_rngs[bin_idx],
                bin_lo[bin_idx],
                bin_hi[bin_idx],
            )
            return None, jnp.linalg.norm(flat)

        _, bin_norms = jax.lax.scan(_bin_step, None, jnp.arange(n_bins))  # [n_bins]

        # Low-t and high-t gradient vectors
        flat_low = _grad_flat_at_range(
            params,
            acts,
            obs,
            valid,
            advantages,
            all_rngs[n_bins],
            jnp.array(_EPS),
            jnp.array(0.2),
        )
        flat_high = _grad_flat_at_range(
            params,
            acts,
            obs,
            valid,
            advantages,
            all_rngs[n_bins + 1],
            jnp.array(0.8),
            jnp.array(1.0),
        )
        norm_low = jnp.linalg.norm(flat_low)
        norm_high = jnp.linalg.norm(flat_high)
        cos = jnp.dot(flat_low, flat_high) / (norm_low * norm_high + 1e-10)

        return bin_norms, cos, norm_low, norm_high

    return t_analysis
