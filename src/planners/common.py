"""Shared gradient-step factory and action-distribution diagnostics.

Both :mod:`src.planners.train` and :mod:`src.planners.online` use identical
gradient update logic.  Centralising it here eliminates duplication and ensures
that the cross-module import of ``_action_stats`` is no longer needed.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from src.diffusion.loss import compute_loss
from src.diffusion.schedules import ScheduleFn


def _action_stats(
    acts: jnp.ndarray,
    num_actions: int,
    valid: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute action-distribution entropy and unique-action fraction over valid windows.

    Args:
        acts:        ``[B, H]`` int32 action sequences.
        num_actions: Size of the real action vocabulary.
        valid:       ``[B]`` bool mask; invalid samples are excluded from counts.

    Returns:
        Dict with ``action_entropy`` and ``action_unique_frac``.
    """
    mask = jnp.broadcast_to(valid[:, None], acts.shape).reshape(-1)
    flat = jnp.where(mask, acts.reshape(-1), num_actions + 1)
    counts = jnp.bincount(flat, length=num_actions).astype(jnp.float32)
    probs = counts / jnp.maximum(counts.sum(), 1.0)
    entropy = -jnp.sum(probs * jnp.log(jnp.where(probs > 0, probs, 1.0)))
    return {
        "action_entropy": entropy,
        "action_unique_frac": jnp.sum(probs > 0).astype(jnp.float32) / num_actions,
    }


def make_grad_step(
    apply_train: Callable,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    sigma_t: float,
    label_smoothing: float,
) -> Callable:
    """Return a jittable gradient update function.

    Args:
        apply_train:       Model apply function with dropout enabled.
        num_actions:       Size of the action vocabulary.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        sigma_t:           ReMDM remasking strength during training.
        label_smoothing:   Cross-entropy label smoothing epsilon.

    Returns:
        A ``step(state, acts, obs, valid, rng, advantages) -> (state, metrics)``
        function ready for use inside ``jax.lax.scan``.
    """

    def _loss_fn(
        params: Any,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        rng: jax.Array,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        return compute_loss(
            apply_train, params, rng, acts, obs, valid,
            num_actions, schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )

    def step(
        state: Any,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        rng: jax.Array,
        advantages: jnp.ndarray,
    ) -> tuple[Any, dict]:
        """Single gradient update step.

        Args:
            state:      Current ``TrainState``.
            acts:       ``[B, H]`` int32 action sequences.
            obs:        ``[B, obs_dim]`` float32 observations.
            valid:      ``[B]`` bool validity mask (episode-boundary filter).
            rng:        PRNG key for dropout and noise sampling.
            advantages: ``[B]`` float per-sample weights applied before loss reduction.

        Returns:
            Updated ``TrainState`` and a metrics dict.
        """
        (_, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            state.params, acts, obs, valid, rng, advantages,
        )
        state = state.apply_gradients(grads=grads)
        info["grad_norm"] = optax.tree.norm(grads)
        info.update(_action_stats(acts, num_actions, valid))
        return state, info

    return step
