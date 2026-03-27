"""Timestep (t) distribution diagnostics for RL fine-tuning analysis.

Analyses how gradient contributions vary across the continuous diffusion
time t ∈ [0, 1], partitioned into N_BINS equal bins.  Also checks whether
the t distribution of training data is uniform as expected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp

from src.diffusion.loss import compute_loss
from src.diffusion.schedules import ScheduleFn

N_BINS: int = 10  # number of t bins for analysis


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class TBinGradNorms:
    """Per-t-bin gradient norms for a single diagnostic step.

    Args:
        bin_norms:     Dict mapping bin label (e.g., "t_0.0-0.1") to gradient L2 norm.
        low_high_cos:  Cosine similarity between gradients from the lowest and highest bins.
        norm_low_t:    L2 norm of gradients from t ∈ [0, 0.2].
        norm_high_t:   L2 norm of gradients from t ∈ [0.8, 1.0].
    """

    bin_norms: dict[str, float] = field(default_factory=dict)
    low_high_cos: float = 0.0
    norm_low_t: float = 0.0
    norm_high_t: float = 0.0


@dataclass
class TBinLoss:
    """Per-t-bin mean loss values.

    Args:
        bin_losses: Dict mapping bin label to mean loss in that bin.
    """

    bin_losses: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# t-bin gradient norm analysis
# ---------------------------------------------------------------------------


def make_t_analysis_fn(
    apply_fn: Callable,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    num_actions: int,
    sigma_t: float = 0.0,
    n_bins: int = N_BINS,
) -> Callable:
    """Build a JIT-compatible t-analysis function.

    The returned function computes gradient norms separately for each t bin
    and also measures alignment between low-t and high-t gradients.

    Args:
        apply_fn:          Training apply fn.
        schedule_fn:       alpha(t) schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Action vocabulary size.
        sigma_t:           ReMDM remasking correction.
        n_bins:            Number of equal t bins to analyse.

    Returns:
        fn(params, acts, obs, valid, advantages, rng) -> TBinGradNorms.
    """
    _EPS = 1e-5
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)

    def _grad_norm_at_range(params, acts, obs, valid, advantages, rng, t_lo, t_hi):
        def loss_in_range(p):
            lo = float(jnp.maximum(t_lo, _EPS))
            hi = float(jnp.minimum(t_hi, 1.0))
            if lo >= hi:
                return jnp.array(0.0)
            loss_val, _ = compute_loss(
                apply_fn, p, rng, acts, obs, valid,
                num_actions, schedule_fn, schedule_deriv_fn,
                sigma_t=sigma_t, advantages=advantages,
                t_min=lo, t_max=hi,
            )
            return loss_val

        g = jax.grad(loss_in_range)(params)
        flat = jnp.concatenate([leaf.ravel() for leaf in jax.tree.leaves(g)])
        return flat

    def compute_t_analysis(params, acts, obs, valid, advantages, rng) -> TBinGradNorms:
        """Compute per-t-bin gradient norms and low/high-t alignment.

        Args:
            params:     Current model parameters.
            acts:       ``[B, H]`` int32 action sequences.
            obs:        ``[B, obs_dim]`` float32 observations.
            valid:      ``[B]`` validity mask.
            advantages: ``[B]`` advantage weights.
            rng:        PRNG key.

        Returns:
            ``TBinGradNorms`` with per-bin norms and alignment metrics.
        """
        bin_norms: dict[str, float] = {}
        all_rngs = jax.random.split(rng, n_bins + 2)

        all_flats = []
        for i in range(n_bins):
            t_lo = float(bin_edges[i])
            t_hi = float(bin_edges[i + 1])
            label = f"t_{t_lo:.1f}-{t_hi:.1f}"
            flat = _grad_norm_at_range(
                params, acts, obs, valid, advantages, all_rngs[i], t_lo, t_hi
            )
            norm = float(jnp.linalg.norm(flat))
            bin_norms[label] = norm
            all_flats.append(flat)

        # Low-t and high-t gradient norms and alignment
        flat_low = _grad_norm_at_range(
            params, acts, obs, valid, advantages, all_rngs[-2], _EPS, 0.2
        )
        flat_high = _grad_norm_at_range(
            params, acts, obs, valid, advantages, all_rngs[-1], 0.8, 1.0
        )
        norm_low = float(jnp.linalg.norm(flat_low))
        norm_high = float(jnp.linalg.norm(flat_high))
        cos = float(
            jnp.dot(flat_low, flat_high) / (norm_low * norm_high + 1e-10)
        )

        return TBinGradNorms(
            bin_norms=bin_norms,
            low_high_cos=cos,
            norm_low_t=norm_low,
            norm_high_t=norm_high,
        )

    return compute_t_analysis


# ---------------------------------------------------------------------------
# Per-t-bin loss decomposition
# ---------------------------------------------------------------------------


def compute_t_bin_losses(
    apply_fn: Callable,
    params: Any,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
    rng: jax.Array,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    num_actions: int,
    sigma_t: float = 0.0,
    n_bins: int = N_BINS,
) -> TBinLoss:
    """Compute mean loss separately in each t bin.

    Reveals which noise-level regime contributes most to the total loss.

    Args:
        apply_fn:          Training apply fn.
        params:            Current model parameters.
        acts:              ``[B, H]`` int32 action sequences.
        obs:               ``[B, obs_dim]`` float32 observations.
        valid:             ``[B]`` validity mask.
        rng:               PRNG key.
        schedule_fn:       alpha(t) schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Action vocabulary size.
        sigma_t:           ReMDM remasking correction.
        n_bins:            Number of t bins.

    Returns:
        ``TBinLoss`` with per-bin mean loss values.
    """
    _EPS = 1e-5
    bin_edges = jnp.linspace(0.0, 1.0, n_bins + 1)
    bin_losses: dict[str, float] = {}
    rngs = jax.random.split(rng, n_bins)

    for i in range(n_bins):
        t_lo = float(jnp.maximum(bin_edges[i], _EPS))
        t_hi = float(bin_edges[i + 1])
        label = f"t_{float(bin_edges[i]):.1f}-{t_hi:.1f}"
        if t_lo >= t_hi:
            bin_losses[label] = 0.0
            continue
        loss_val, _ = compute_loss(
            apply_fn, params, rngs[i], acts, obs, valid,
            num_actions, schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, t_min=t_lo, t_max=t_hi,
        )
        bin_losses[label] = float(loss_val)

    return TBinLoss(bin_losses=bin_losses)
