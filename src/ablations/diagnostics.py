"""Diagnostic metric functions for ablation training runs.

All functions are pure JAX and JIT-compatible unless stated otherwise.
Functions that involve double differentiation (e.g. Fisher-based metrics)
are marked as *not JIT-compatible* and are intended for host-level calls.

Metric categories
-----------------
Gradient health
    ``compute_gradient_alignment`` — cosine similarity between two gradient pytrees.
    ``compute_per_layer_grad_norm`` — per-sub-module gradient L2 norm.

Representation drift
    ``compute_representation_drift`` — per-layer and total L2 parameter distance.
    ``compute_output_kl`` — KL divergence of output distributions on a probe batch.

Loss diagnostics
    ``compute_per_t_loss`` — mean ELBO per diffusion-time bin.

Policy quality
    ``compute_token_entropy`` — mean entropy of action distribution per denoising step.
    ``compute_collapse_fraction`` — fraction of plans with ≥50% the same token.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from src.diffusion.forward import forward_process
from src.diffusion.schedules import ScheduleFn

_EPS: float = 1e-5
_MAX_WEIGHT: float = 1000.0

ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]


# ---------------------------------------------------------------------------
# Gradient health
# ---------------------------------------------------------------------------

def compute_gradient_alignment(
    grads_a: Any,
    grads_b: Any,
) -> jnp.ndarray:
    """Cosine similarity between two gradient pytrees.

    Both pytrees must have identical structure.  The similarity is computed
    on the concatenated flat vector of all leaf arrays.

    Args:
        grads_a: First gradient pytree.
        grads_b: Second gradient pytree.

    Returns:
        Scalar cosine similarity in ``[-1, 1]``.
    """
    flat_a = jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree.leaves(grads_a)]
    )
    flat_b = jnp.concatenate(
        [leaf.ravel() for leaf in jax.tree.leaves(grads_b)]
    )
    norm_a = jnp.linalg.norm(flat_a) + _EPS
    norm_b = jnp.linalg.norm(flat_b) + _EPS
    return jnp.dot(flat_a, flat_b) / (norm_a * norm_b)


def compute_per_layer_grad_norm(grads: Any) -> dict[str, jnp.ndarray]:
    """Compute L2 gradient norm for each named top-level sub-module.

    Assumes a Flax ``params`` pytree structure where ``grads['params']`` is a
    dict of sub-module names to parameter dicts.

    Args:
        grads: Full gradient pytree (Flax ``TrainState.params`` structure).

    Returns:
        Dict mapping sub-module name to its scalar L2 gradient norm.
        Also includes ``'__total__'`` for the global norm.
    """
    result: dict[str, jnp.ndarray] = {}

    if "params" in grads:
        param_grads = grads["params"]
        for name, sub_grads in param_grads.items():
            leaves = jax.tree.leaves(sub_grads)
            flat = jnp.concatenate([leaf.ravel() for leaf in leaves])
            result[name] = jnp.linalg.norm(flat)

    all_leaves = jax.tree.leaves(grads)
    all_flat = jnp.concatenate([leaf.ravel() for leaf in all_leaves])
    result["__total__"] = jnp.linalg.norm(all_flat)
    return result


# ---------------------------------------------------------------------------
# Representation drift
# ---------------------------------------------------------------------------

def compute_representation_drift(
    params: Any,
    ref_params: Any,
) -> dict[str, jnp.ndarray]:
    """Compute L2 parameter distance from a reference (pretrained) checkpoint.

    Args:
        params:     Current model parameter pytree.
        ref_params: Reference parameter pytree (same structure).

    Returns:
        Dict mapping sub-module name to its L2 drift scalar.
        Includes ``'__total__'`` for the global L2 distance.
    """
    diffs = jax.tree.map(lambda p, r: p - r, params, ref_params)
    result: dict[str, jnp.ndarray] = {}

    if "params" in diffs:
        param_diffs = diffs["params"]
        for name, sub_diff in param_diffs.items():
            leaves = jax.tree.leaves(sub_diff)
            flat = jnp.concatenate([leaf.ravel() for leaf in leaves])
            result[name] = jnp.linalg.norm(flat)

    all_leaves = jax.tree.leaves(diffs)
    all_flat = jnp.concatenate([leaf.ravel() for leaf in all_leaves])
    result["__total__"] = jnp.linalg.norm(all_flat)
    return result


def compute_output_kl(
    apply_fn: ModelApplyFn,
    params: Any,
    ref_params: Any,
    probe_obs: jnp.ndarray,
    rng: jax.Array,
    num_actions: int,
    plan_horizon: int,
) -> jnp.ndarray:
    """KL(p_theta || p_ref) on a fixed probe batch at mid-range t=0.5.

    Measures how much the current model's predicted action distribution has
    shifted from the pretrained reference on a held-out set of observations.

    Args:
        apply_fn:    Model apply closure (eval mode, no dropout).
        params:      Current model parameters.
        ref_params:  Reference (pretrained) parameters.
        probe_obs:   ``[B, D]`` held-out observations.
        rng:         PRNG key (unused; kept for API consistency).
        num_actions: Real action vocabulary size.
        plan_horizon: Plan sequence length H.

    Returns:
        Scalar mean KL divergence (averaged over batch and positions).
    """
    B = probe_obs.shape[0]
    mask_id = num_actions

    t_probe = jnp.full((B,), 0.5)
    z_probe = jnp.full((B, plan_horizon), mask_id, dtype=jnp.int32)

    logits_cur = apply_fn(params, probe_obs, z_probe, t_probe, None)      # [B, H, A]
    logits_ref = apply_fn(ref_params, probe_obs, z_probe, t_probe, None)  # [B, H, A]

    log_p = jax.nn.log_softmax(logits_cur, axis=-1)
    log_q = jax.nn.log_softmax(logits_ref, axis=-1)
    p = jnp.exp(log_p)

    kl = jnp.sum(p * (log_p - log_q), axis=-1)  # [B, H]
    return jnp.mean(kl)


# ---------------------------------------------------------------------------
# Loss diagnostics
# ---------------------------------------------------------------------------

def compute_per_t_loss(
    apply_fn: ModelApplyFn,
    params: Any,
    rng: jax.Array,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    n_bins: int = 10,
) -> jnp.ndarray:
    """Compute mean ELBO loss per diffusion-time bin.

    Samples t uniformly over ``[0, 1]`` and accumulates the per-sample loss
    into ``n_bins`` equal-width bins.  Bins with no samples return zero.

    Args:
        apply_fn:          Model apply closure (eval mode, no dropout).
        params:            Model parameters.
        rng:               PRNG key.
        acts:              ``[B, H]`` int32 action sequences.
        obs:               ``[B, D]`` float32 observations.
        valid:             ``[B]`` bool validity mask.
        num_actions:       Real action vocabulary size.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        n_bins:            Number of t-bins (default 10).

    Returns:
        ``[n_bins]`` float32 array of mean per-bin ELBO loss.
    """
    B = acts.shape[0]
    mask_id = num_actions
    rng, t_rng, mask_rng = jax.random.split(rng, 3)

    # Sample t uniformly; use a deterministic linear grid for reproducibility.
    t = jax.random.uniform(t_rng, (B,), minval=_EPS, maxval=1.0)
    alpha_t = schedule_fn(t)

    # Compute ELBO weight.
    neg_alpha_dot = -schedule_deriv_fn(t)
    weight = neg_alpha_dot / jnp.maximum(1.0 - alpha_t, _EPS)
    weight = jnp.minimum(weight, _MAX_WEIGHT)

    # Forward noise and model prediction.
    z_t = forward_process(mask_rng, acts, alpha_t, mask_id)
    logits = apply_fn(params, obs, z_t, t, None)  # [B, H, A]

    is_masked = (z_t == mask_id).astype(jnp.float32)
    valid_masked = is_masked * valid[:, None].astype(jnp.float32)

    targets = jax.nn.one_hot(acts, num_actions)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(targets * log_probs, axis=-1)  # [B, H]

    n_masked = jnp.maximum(valid_masked.sum(axis=-1), 1.0)
    per_sample_loss = weight * (ce * valid_masked).sum(axis=-1) / n_masked  # [B]

    # Assign each sample to its t-bin (0-indexed).
    bin_idx = jnp.floor(t * n_bins).astype(jnp.int32)
    bin_idx = jnp.minimum(bin_idx, n_bins - 1)

    bin_losses = jnp.zeros(n_bins).at[bin_idx].add(per_sample_loss)
    bin_counts = jnp.zeros(n_bins).at[bin_idx].add(1.0)
    return bin_losses / jnp.maximum(bin_counts, 1.0)


# ---------------------------------------------------------------------------
# Policy quality
# ---------------------------------------------------------------------------

def compute_token_entropy(
    logits: jnp.ndarray,
) -> jnp.ndarray:
    """Mean entropy of the predicted action distribution.

    Args:
        logits: ``[B, H, A]`` raw model output logits.

    Returns:
        Scalar mean entropy in nats (averaged over batch and positions).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(
        probs * jnp.log(jnp.where(probs > 0, probs, 1.0)), axis=-1
    )  # [B, H]
    return jnp.mean(entropy)


def compute_collapse_fraction(
    plan_tokens: jnp.ndarray,
    threshold: float = 0.5,
) -> jnp.ndarray:
    """Fraction of plans where a single action constitutes ≥ ``threshold`` of tokens.

    Mode-collapse is signalled when the model repeats one action token in
    more than half the positions of a plan.

    Args:
        plan_tokens: ``[B, H]`` int32 sampled action plans.
        threshold:   Fraction of repeated tokens that marks collapse (default 0.5).

    Returns:
        Scalar in ``[0, 1]``: fraction of plans considered collapsed.
    """
    B, H = plan_tokens.shape
    # Most-frequent token count per plan.
    def _max_freq(row: jnp.ndarray) -> jnp.ndarray:
        counts = jnp.bincount(row, length=256)  # 256 > max num_actions in Craftax
        return jnp.max(counts).astype(jnp.float32) / H

    max_freqs = jax.vmap(_max_freq)(plan_tokens)  # [B]
    return jnp.mean(max_freqs >= threshold)
