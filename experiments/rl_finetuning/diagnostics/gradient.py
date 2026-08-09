"""Gradient-space diagnostics for RL fine-tuning analysis.

All functions return JAX arrays and are fully JIT-compatible.
Python-side wrappers (returning dataclasses) are provided for
non-compiled call sites; the inner ``_*`` functions are used
directly inside ``jax.lax.scan``.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def make_grad_alignment_fn(
    apply_fn: Callable,
    schedule_fn: Callable,
    schedule_deriv_fn: Callable,
    num_actions: int,
    sigma_t: float = 0.0,
) -> Callable:
    """Build a JIT-compiled gradient alignment function.

    Returns a function that computes cosine similarity, RL grad norm, and
    BC grad norm as a tuple of three JAX scalars — safe for use inside
    ``jax.lax.scan`` / ``jax.lax.cond``.

    Args:
        apply_fn:          Training apply fn (params, obs, z_t, t, rng) -> logits.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Action vocabulary size.
        sigma_t:           ReMDM remasking correction.

    Returns:
        JIT-compiled fn(params, ref_params, acts, obs, valid, rng, advantages)
        -> (cos_sim, rl_norm, bc_norm) as JAX scalars.
    """
    from src.diffusion.loss import compute_loss

    @jax.jit
    def grad_alignment(
        params: Any,
        ref_params: Any,
        acts: jax.Array,
        obs: jax.Array,
        valid: jax.Array,
        rng: jax.Array,
        advantages: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
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
            Tuple of (cos_sim, rl_grad_norm, bc_grad_norm) as JAX scalars.
        """
        rng_rl, rng_bc = jax.random.split(rng)

        def rl_loss(p: Any) -> jax.Array:
            loss, _ = compute_loss(
                apply_fn,
                p,
                rng_rl,
                acts,
                obs,
                valid,
                num_actions,
                schedule_fn,
                schedule_deriv_fn,
                sigma_t=sigma_t,
                advantages=advantages,
            )
            return loss

        def bc_loss(p: Any) -> jax.Array:
            loss, _ = compute_loss(
                apply_fn,
                p,
                rng_bc,
                acts,
                obs,
                valid,
                num_actions,
                schedule_fn,
                schedule_deriv_fn,
                sigma_t=sigma_t,
                advantages=None,
            )
            return loss

        g_rl = jax.grad(rl_loss)(params)
        g_bc = jax.grad(bc_loss)(ref_params)

        rl_flat = jnp.concatenate(
            [g.ravel() for g in jax.tree.leaves(g_rl)]
        )  # [D_total]
        bc_flat = jnp.concatenate(
            [g.ravel() for g in jax.tree.leaves(g_bc)]
        )  # [D_total]

        rl_norm = jnp.linalg.norm(rl_flat)
        bc_norm = jnp.linalg.norm(bc_flat)
        cos_sim = jnp.dot(rl_flat, bc_flat) / (rl_norm * bc_norm + 1e-10)
        return cos_sim, rl_norm, bc_norm

    return grad_alignment


def compute_per_layer_grad_norms_jax(grads: Any) -> jax.Array:
    """Compute L2 gradient norm per parameter leaf as a JAX array.

    Unlike the old ``PerLayerGradNorms`` dataclass, this returns a single
    1-D array of norms — one entry per pytree leaf — that is JIT-safe.
    The mapping from index to layer name can be recovered outside JIT via
    ``jax.tree.structure(params)``.

    Args:
        grads: Gradient pytree (same structure as params).

    Returns:
        ``[num_leaves]`` JAX array of per-leaf L2 norms.
    """
    leaves = jax.tree.leaves(grads)
    norms = jnp.stack([jnp.linalg.norm(leaf) for leaf in leaves])
    return norms  # [num_leaves]


def compute_surgery_metrics_jax(
    g_rl_before: Any,
    g_rl_after: Any,
) -> tuple[jax.Array, jax.Array]:
    """Measure gradient mass removed by PCGrad, returning JAX scalars.

    Args:
        g_rl_before: RL gradient pytree before projection.
        g_rl_after:  RL gradient pytree after projection.

    Returns:
        Tuple of (projected_mass_fraction, n_conflicting_params) as JAX arrays.
    """
    before_leaves = jax.tree.leaves(g_rl_before)
    after_leaves = jax.tree.leaves(g_rl_after)

    before_sq = jnp.stack([jnp.sum(g**2) for g in before_leaves])
    after_sq = jnp.stack([jnp.sum(g**2) for g in after_leaves])

    total_before = jnp.sum(before_sq)
    total_after = jnp.sum(after_sq)
    projected_mass = jnp.maximum(total_before - total_after, 0.0)
    fraction = projected_mass / jnp.maximum(total_before, 1e-10)

    # Count leaves where projection was applied (dot product of diff > 0)
    n_conflicting = jnp.sum(
        jnp.stack(
            [
                (jnp.sum(g_r * (g_r - g_a)) > 0).astype(jnp.int32)
                for g_r, g_a in zip(before_leaves, after_leaves)
            ]
        )
    )

    return fraction, n_conflicting
