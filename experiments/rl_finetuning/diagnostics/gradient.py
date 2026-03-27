"""Gradient-space diagnostics for RL fine-tuning analysis.

All JIT-able functions are decorated with ``@jax.jit`` or returned from
factories.  Python-side aggregation (per-layer norm collection) is done
outside JIT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class GradAlignResult:
    """Output of a gradient alignment computation.

    Args:
        cos_sim:      Cosine similarity between RL and BC gradient vectors.
        rl_grad_norm: L2 norm of the RL gradient.
        bc_grad_norm: L2 norm of the BC gradient.
    """

    cos_sim: float
    rl_grad_norm: float
    bc_grad_norm: float


@dataclass
class PerLayerGradNorms:
    """Per-layer gradient norms for a single training step.

    Args:
        layer_norms: Dict mapping layer path prefix to its L2 gradient norm.
    """

    layer_norms: dict[str, float] = field(default_factory=dict)


@dataclass
class GradSurgeryMetrics:
    """Metrics from a gradient surgery (PCGrad) step.

    Args:
        projected_mass_fraction: Fraction of gradient L2 mass that was projected away.
        n_conflicting_params:    Number of parameter tensors where dot(g_rl, g_bc) < 0.
    """

    projected_mass_fraction: float
    n_conflicting_params: int


# ---------------------------------------------------------------------------
# Gradient alignment
# ---------------------------------------------------------------------------


def make_grad_alignment_fn(
    apply_fn: Callable,
    schedule_fn: Callable,
    schedule_deriv_fn: Callable,
    num_actions: int,
    sigma_t: float = 0.0,
) -> Callable:
    """Build a JIT-compiled gradient alignment function.

    The returned function computes:
    - RL gradient: grad of return-weighted ELBO w.r.t. current params
    - BC gradient: grad of unweighted ELBO w.r.t. pretrained (reference) params
    - Cosine similarity between the two gradient vectors

    Args:
        apply_fn:          Training apply fn (params, obs, z_t, t, rng) -> logits.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Action vocabulary size.
        sigma_t:           ReMDM remasking correction.

    Returns:
        JIT-compiled fn(params, ref_params, acts, obs, valid, rng, advantages)
        -> GradAlignResult.
    """
    from src.diffusion.loss import compute_loss

    @jax.jit
    def _align(params, ref_params, acts, obs, valid, rng, advantages):
        rng_rl, rng_bc = jax.random.split(rng)

        def rl_loss(p):
            loss, _ = compute_loss(
                apply_fn, p, rng_rl, acts, obs, valid,
                num_actions, schedule_fn, schedule_deriv_fn,
                sigma_t=sigma_t, advantages=advantages,
            )
            return loss

        def bc_loss(p):
            loss, _ = compute_loss(
                apply_fn, p, rng_bc, acts, obs, valid,
                num_actions, schedule_fn, schedule_deriv_fn,
                sigma_t=sigma_t, advantages=None,
            )
            return loss

        g_rl = jax.grad(rl_loss)(params)
        g_bc = jax.grad(bc_loss)(ref_params)

        rl_flat = jnp.concatenate([g.ravel() for g in jax.tree.leaves(g_rl)])
        bc_flat = jnp.concatenate([g.ravel() for g in jax.tree.leaves(g_bc)])

        rl_norm = jnp.linalg.norm(rl_flat)
        bc_norm = jnp.linalg.norm(bc_flat)
        cos_sim = jnp.dot(rl_flat, bc_flat) / (rl_norm * bc_norm + 1e-10)
        return cos_sim, rl_norm, bc_norm

    def compute_grad_alignment(params, ref_params, acts, obs, valid, rng, advantages) -> GradAlignResult:
        """Compute cosine similarity between RL and BC gradient vectors.

        Args:
            params:     Current model parameters.
            ref_params: Pretrained reference parameters.
            acts:       ``[B, H]`` int32 action sequences.
            obs:        ``[B, obs_dim]`` float32 observations.
            valid:      ``[B]`` validity mask.
            rng:        PRNG key.
            advantages: ``[B]`` return weights.

        Returns:
            ``GradAlignResult`` with cosine similarity and norms.
        """
        cos_sim, rl_norm, bc_norm = _align(params, ref_params, acts, obs, valid, rng, advantages)
        return GradAlignResult(
            cos_sim=float(cos_sim),
            rl_grad_norm=float(rl_norm),
            bc_grad_norm=float(bc_norm),
        )

    return compute_grad_alignment


# ---------------------------------------------------------------------------
# Per-layer gradient norms
# ---------------------------------------------------------------------------


def compute_per_layer_grad_norms(grads: Any) -> PerLayerGradNorms:
    """Compute L2 gradient norm per top-level parameter group.

    Groups are identified by the first key in each parameter path
    (e.g., "TransformerBlock_0", "Dense_0").

    Args:
        grads: Gradient pytree (same structure as params).

    Returns:
        ``PerLayerGradNorms`` with per-group L2 norms.
    """
    group_sq_sums: dict[str, float] = {}

    def _collect(path: tuple, leaf):
        group = str(path[0].key) if path and hasattr(path[0], "key") else str(path[0]) if path else "root"
        sq = float(jnp.sum(leaf ** 2))
        group_sq_sums[group] = group_sq_sums.get(group, 0.0) + sq

    jax.tree_util.tree_map_with_path(_collect, grads)
    layer_norms = {k: float(v ** 0.5) for k, v in group_sq_sums.items()}
    return PerLayerGradNorms(layer_norms=layer_norms)


# ---------------------------------------------------------------------------
# Gradient surgery metrics
# ---------------------------------------------------------------------------


def compute_surgery_metrics(g_rl_before: Any, g_rl_after: Any) -> GradSurgeryMetrics:
    """Measure how much gradient mass was removed by gradient surgery.

    Args:
        g_rl_before: RL gradient pytree before PCGrad projection.
        g_rl_after:  RL gradient pytree after PCGrad projection.

    Returns:
        ``GradSurgeryMetrics`` with projected mass fraction and conflict count.
    """
    before_sq = sum(float(jnp.sum(g ** 2)) for g in jax.tree.leaves(g_rl_before))
    after_sq = sum(float(jnp.sum(g ** 2)) for g in jax.tree.leaves(g_rl_after))
    projected_mass = max(before_sq - after_sq, 0.0)
    fraction = projected_mass / max(before_sq, 1e-10)

    # Count conflicting param tensors
    n_conflicting = sum(
        1 for g_r, g_b in zip(jax.tree.leaves(g_rl_before), jax.tree.leaves(g_rl_after))
        if float(jnp.sum(g_r * (g_r - g_b))) > 0  # projection was applied
    )
    return GradSurgeryMetrics(
        projected_mass_fraction=fraction,
        n_conflicting_params=n_conflicting,
    )
