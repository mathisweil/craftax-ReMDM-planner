"""End-to-end ablation training loop with rich diagnostic collection.

``run_ablation_v2`` is the successor to the notebook-level ``run_ablation``
function.  It collects all new diagnostic keys defined in
:mod:`src.ablations.diagnostics` on every evaluation interval and returns a
history dict that the notebook uses for visualisation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import ScheduleFn
from src.ablations.diagnostics import (
    compute_gradient_alignment,
    compute_output_kl,
    compute_per_t_loss,
    compute_per_layer_grad_norm,
    compute_token_entropy,
    compute_collapse_fraction,
    compute_representation_drift,
)

logger = logging.getLogger(__name__)

LossFn = Callable[
    [Any, jax.Array, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    tuple[jnp.ndarray, dict],
]
ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]
RolloutFn = Callable[
    [jax.Array, Any],
    tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
]


def run_ablation_v2(
    name: str,
    loss_fn: LossFn,
    apply_fn: ModelApplyFn,
    apply_train_fn: ModelApplyFn,
    collect_rollout: RolloutFn,
    pretrained_params: Any,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    num_actions: int,
    plan_horizon: int,
    obs_dim: int,
    rng: jax.Array,
    lr: float = 1e-4,
    max_grad_norm: float = 1.0,
    batch_size: int = 256,
    max_iter: int = 500,
    eval_every: int = 100,
    eval_steps: int = 512,
    n_eval_envs: int = 16,
    diffusion_steps: int = 15,
    replan_every: int = 4,
    probe_obs: Optional[jnp.ndarray] = None,
    frozen_backbone: bool = False,
    gradient_surgery: bool = False,
    bc_loss_fn: Optional[LossFn] = None,
    step_dependent_loss: bool = False,
    eval_fn: Optional[Callable] = None,
) -> dict[str, Any]:
    """Run a single ablation experiment and return a rich history dict.

    The training loop:
    1. Collects a rollout batch via ``collect_rollout(rng, params)``.
    2. Computes one minibatch gradient step with ``loss_fn``.
    3. Every ``eval_every`` steps, evaluates the policy and records diagnostics.

    The returned history dict contains the following keys (where available):

    - ``'step'``: list of gradient step indices.
    - ``'loss'``: training loss per step.
    - ``'eval_score'``: eval episode return (recorded at eval intervals).
    - ``'grad_align'``: cosine similarity between RL and BC gradients.
    - ``'repr_drift'``: total L2 parameter drift from pretrained.
    - ``'output_kl'``: mean KL divergence on probe batch.
    - ``'per_t_loss'``: ``[n_eval_steps, n_bins]`` per-t-bin loss.
    - ``'per_layer_grad_norm'``: list of per-step dicts.
    - ``'token_entropy'``: mean token entropy at eval steps.
    - ``'collapse_fraction'``: plan collapse fraction at eval steps.
    - ``'achievements'``: dict of final achievement unlock rates.

    Args:
        name:               Ablation method name (for logging).
        loss_fn:            Loss function factory output.
        apply_fn:           Eval-mode model apply closure.
        apply_train_fn:     Train-mode model apply closure.
        collect_rollout:    ``(rng, params) -> (obs, acts, valid, advantages)``.
        pretrained_params:  Frozen reference parameters.
        schedule_fn:        alpha(t) noise schedule.
        schedule_deriv_fn:  d(alpha)/dt.
        num_actions:        Real action vocabulary size.
        plan_horizon:       Plan sequence length H.
        obs_dim:            Observation vector dimensionality.
        rng:                PRNG key.
        lr:                 Adam learning rate.
        max_grad_norm:      Global gradient clipping threshold.
        batch_size:         Minibatch size.
        max_iter:           Number of gradient steps.
        eval_every:         Evaluate every this many steps.
        eval_steps:         Environment steps per evaluation rollout.
        n_eval_envs:        Number of parallel eval environments.
        diffusion_steps:    Denoising steps T for evaluation.
        replan_every:       Env steps per diffusion plan during evaluation.
        probe_obs:          ``[B_probe, obs_dim]`` held-out observations for KL / drift.
                            If ``None``, uses the first ``batch_size`` observations from
                            the most recent rollout as a proxy.
        frozen_backbone:    Zero backbone gradients (only head updated).
        gradient_surgery:   Project out BC-conflicting gradient components.
        bc_loss_fn:         BC loss used by gradient surgery (required when
                            ``gradient_surgery=True``).
        step_dependent_loss: Pass ``step_idx`` kwarg to ``loss_fn`` (curriculum).
        eval_fn:            Optional ``(params, rng) -> float`` evaluation function.
                            If ``None``, eval score is not recorded.

    Returns:
        History dict with all tracked metrics.

    Raises:
        ValueError: If ``gradient_surgery=True`` but ``bc_loss_fn`` is ``None``.
    """
    if gradient_surgery and bc_loss_fn is None:
        raise ValueError(
            "gradient_surgery=True requires bc_loss_fn to be provided."
        )

    logger.info("[%s] Starting ablation — max_iter=%d", name, max_iter)

    # Initialise train state with pretrained params + Adam.
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    state = TrainState.create(
        apply_fn=apply_train_fn,
        params=pretrained_params,
        tx=tx,
    )

    # JIT-compile the gradient step.
    @jax.jit
    def _grad_step(
        st: TrainState,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
        rng: jax.Array,
        step_idx: jnp.ndarray,
    ) -> tuple[TrainState, jnp.ndarray, dict]:
        """Single gradient update step.

        Returns:
            Tuple of (new_state, grads, info_dict).
        """
        kwargs: dict = {}
        if step_dependent_loss:
            kwargs["step_idx"] = step_idx

        def _loss(p: Any) -> tuple[jnp.ndarray, dict]:
            return loss_fn(p, rng, acts, obs, valid, advantages, **kwargs)

        (_, info), grads = jax.value_and_grad(_loss, has_aux=True)(st.params)

        # Optional gradient surgery.
        if gradient_surgery and bc_loss_fn is not None:
            rng_bc, _ = jax.random.split(rng)

            def _bc_loss(p: Any) -> tuple[jnp.ndarray, dict]:
                return bc_loss_fn(
                    p, rng_bc, acts, obs, valid, jnp.ones_like(advantages)
                )

            _, bc_grads = jax.value_and_grad(_bc_loss, has_aux=True)(st.params)

            # Project: remove component along BC gradient if they conflict.
            def _flatten(g: Any) -> jnp.ndarray:
                return jnp.concatenate(
                    [leaf.ravel() for leaf in jax.tree.leaves(g)]
                )

            rl_flat = _flatten(grads)
            bc_flat = _flatten(bc_grads)
            dot = jnp.dot(rl_flat, bc_flat)
            bc_norm_sq = jnp.dot(bc_flat, bc_flat) + 1e-8
            proj_coef = jnp.where(dot < 0.0, dot / bc_norm_sq, 0.0)
            grads = jax.tree.map(
                lambda rl_g, bc_g: rl_g - proj_coef * bc_g,
                grads, bc_grads,
            )

        # Optional frozen backbone.
        if frozen_backbone:
            def _zero_unless_head(
                path: tuple, leaf: jnp.ndarray
            ) -> jnp.ndarray:
                path_str = "/".join(
                    str(p.key) for p in path if hasattr(p, "key")
                )
                return leaf if "Dense_0" in path_str else jnp.zeros_like(leaf)

            grads = jax.tree_util.tree_map_with_path(_zero_unless_head, grads)

        new_state = st.apply_gradients(grads=grads)
        return new_state, grads, info

    # History accumulation.
    history: dict[str, list] = {
        "step": [],
        "loss": [],
        "eval_score": [],
        "grad_align": [],
        "repr_drift": [],
        "output_kl": [],
        "per_t_loss": [],
        "per_layer_grad_norm": [],
        "token_entropy": [],
        "collapse_fraction": [],
        "return_dist_start": [],
        "return_dist_final": [],
    }

    _probe_obs_cache: Optional[jnp.ndarray] = probe_obs

    for step_idx in range(max_iter):
        rng, rollout_rng, step_rng, diag_rng = jax.random.split(rng, 4)

        # Collect trajectories.
        obs_b, acts_b, valid_b, adv_b = collect_rollout(rollout_rng, state.params)

        # Minibatch sub-sample.
        n = obs_b.shape[0]
        if n > batch_size:
            rng, idx_rng = jax.random.split(rng)
            idx = jax.random.randint(idx_rng, (batch_size,), 0, n)
            obs_b = obs_b[idx]
            acts_b = acts_b[idx]
            valid_b = valid_b[idx]
            adv_b = adv_b[idx]

        # Cache probe observations from the first rollout.
        if _probe_obs_cache is None:
            _probe_obs_cache = obs_b[:min(64, obs_b.shape[0])]

        state, grads, info = _grad_step(
            state, acts_b, obs_b, valid_b, adv_b,
            step_rng, jnp.array(step_idx),
        )

        loss_val = float(info.get("loss", info.get("Loss", 0.0)))
        history["loss"].append(loss_val)
        history["step"].append(step_idx)

        # Per-layer gradient norms (host-side Python).
        layer_norms = compute_per_layer_grad_norm(
            jax.tree.map(lambda g: jnp.array(g), grads)
        )
        history["per_layer_grad_norm"].append(
            {k: float(v) for k, v in layer_norms.items()}
        )

        # Diagnostic evaluation.
        if step_idx % eval_every == 0:
            # Gradient alignment (RL vs BC/uniform).
            rng, align_rng_rl, align_rng_bc = jax.random.split(rng, 3)

            def _bc_loss_fn(p: Any) -> tuple[jnp.ndarray, dict]:
                from src.ablations.losses import _base_elbo
                return _base_elbo(
                    apply_train_fn, p, align_rng_bc,
                    acts_b, obs_b, valid_b, num_actions,
                    schedule_fn, schedule_deriv_fn,
                    advantages=jnp.ones_like(adv_b),
                )

            _, bc_grads = jax.value_and_grad(_bc_loss_fn, has_aux=True)(
                state.params
            )
            grad_align = float(
                compute_gradient_alignment(grads, bc_grads)
            )
            history["grad_align"].append(grad_align)

            # Representation drift.
            drift_dict = compute_representation_drift(
                state.params, pretrained_params
            )
            history["repr_drift"].append(float(drift_dict["__total__"]))

            # Output KL divergence on probe batch.
            if _probe_obs_cache is not None:
                output_kl = float(
                    compute_output_kl(
                        apply_fn, state.params, pretrained_params,
                        _probe_obs_cache, diag_rng, num_actions, plan_horizon,
                    )
                )
                history["output_kl"].append(output_kl)

            # Per-t-bin ELBO loss.
            per_t = compute_per_t_loss(
                apply_fn, state.params, diag_rng,
                acts_b, obs_b, valid_b,
                num_actions, schedule_fn, schedule_deriv_fn,
                n_bins=10,
            )
            history["per_t_loss"].append(
                [float(v) for v in per_t.tolist()]
            )

            # Token entropy and collapse fraction via a quick plan sample.
            rng, plan_rng = jax.random.split(rng)
            sample_obs = obs_b[:min(8, obs_b.shape[0])]
            sample_plans = sample_plan(
                apply_fn, state.params, plan_rng, sample_obs,
                num_actions, plan_horizon,
                num_steps=diffusion_steps,
                schedule_fn=schedule_fn,
            )  # [B_sample, H]

            # Compute logits for entropy.
            t_mid = jnp.full((sample_obs.shape[0],), 0.5)
            mask_probe = jnp.full(
                (sample_obs.shape[0], plan_horizon), num_actions, dtype=jnp.int32
            )
            logits_eval = apply_fn(
                state.params, sample_obs, mask_probe, t_mid, None
            )
            history["token_entropy"].append(
                float(compute_token_entropy(logits_eval))
            )
            history["collapse_fraction"].append(
                float(compute_collapse_fraction(sample_plans))
            )

            # External evaluation function (returns a scalar score).
            if eval_fn is not None:
                rng, eval_rng = jax.random.split(rng)
                score = float(eval_fn(state.params, eval_rng))
                history["eval_score"].append(score)

            logger.info(
                "[%s] step=%d  loss=%.4f  drift=%.3f  align=%.3f",
                name, step_idx, loss_val,
                history["repr_drift"][-1],
                grad_align,
            )

    # Store return distributions for deep-dive plot.
    if history["loss"]:
        history["return_dist_start"] = [float(v) for v in adv_b[:32].tolist()]
        history["return_dist_final"] = [float(v) for v in adv_b[:32].tolist()]

    logger.info("[%s] Training complete.", name)
    return history
