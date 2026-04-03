"""Shared gradient-step factory, validation rollout, and action diagnostics.

Both :mod:`src.planners.offline` and :mod:`src.planners.online` use identical
gradient update and validation logic.  Centralising it here eliminates
duplication.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import optax

from src.diffusion.loss import compute_loss
from src.diffusion.sampling import sample_plan
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


def make_validate(
    env: Any,
    env_params: Any,
    apply_eval: Callable,
    num_actions: int,
    plan_horizon: int,
    schedule_fn: ScheduleFn,
    config: dict[str, Any],
    val_replan_every: int,
    n_val_cycles: int,
) -> Callable:
    """Return a ``validate(state, rng) -> dict`` closure for periodic eval.

    The closure runs a held-out rollout using the diffusion model's current
    parameters and returns metrics under the ``val/`` namespace.

    Args:
        env:              Batched Gymnax environment.
        env_params:       Gymnax environment params.
        apply_eval:       Model apply function (eval mode, no dropout).
        num_actions:      Size of the action vocabulary.
        plan_horizon:     Action plan length H.
        schedule_fn:      alpha(t) noise schedule.
        config:           Training config dict (read-only).
        val_replan_every: Env steps executed per diffusion plan during validation.
        n_val_cycles:     Number of plan-execute cycles per validation rollout.

    Returns:
        A ``validate(state, rng) -> {str: jnp.ndarray}`` closure.
    """

    def validate(state: Any, rng: jax.Array) -> dict[str, jnp.ndarray]:
        """Run a validation rollout and return ``val/`` metrics.

        Args:
            state: Current ``TrainState`` (only ``.params`` is used).
            rng:   PRNG key.

        Returns:
            Dict with ``val/`` prefixed metric keys.
        """
        rng, val_rng = jax.random.split(rng)
        val_obs, val_env_state = env.reset(val_rng, env_params)

        def _val_cycle(carry, _):
            vs, vo, rng = carry
            rng, p_rng = jax.random.split(rng)
            plan = sample_plan(
                apply_eval,
                state.params,
                p_rng,
                vo,
                num_actions,
                plan_horizon,
                num_steps=config.get("VAL_DIFFUSION_STEPS", 50),
                schedule_fn=schedule_fn,
                remask_strategy=config.get("REMASK_STRATEGY", "rescale"),
                eta=config.get("ETA", 0.5),
                use_loop=config.get("USE_LOOP", True),
                t_on=config.get("T_ON", 0.7),
                t_off=config.get("T_OFF", 0.3),
                temperature=config.get("TEMPERATURE", 0.5),
                top_p=config.get("TOP_P", 0.95),
            )  # [num_envs, plan_horizon]

            def _exec_step(inner_carry, step_i):
                vs_i, vo_i, r = inner_carry
                r, s_rng = jax.random.split(r)
                vo_next, vs_next, _, _, info = env.step(
                    s_rng, vs_i, plan[:, step_i], env_params,
                )
                return (vs_next, vo_next, r), info

            (vs, vo, rng), step_infos = jax.lax.scan(
                _exec_step, (vs, vo, rng), jnp.arange(val_replan_every),
            )
            return (vs, vo, rng), step_infos

        _, cycle_infos = jax.lax.scan(
            _val_cycle, (val_env_state, val_obs, rng), None, n_val_cycles,
        )
        infos = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), cycle_infos,
        )
        returned = infos["returned_episode"]
        metrics = jax.tree.map(
            lambda x: (x * returned).sum() / (returned.sum() + 1e-8),
            infos,
        )
        return {f"val/{k}": v for k, v in metrics.items()}

    return validate
