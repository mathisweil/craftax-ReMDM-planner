"""End-to-end ablation training loop with rich diagnostic collection.

``run_ablation_v2`` is the canonical training loop for all ablation
experiments.  It accepts a 2-arg ``collect_rollout(rng, params)`` closure and
returns ``(history, final_params)`` so callers can extract trained weights.

``make_stateful_rollout_fn`` bridges the notebook's 5-arg stateful rollout
(which carries gymnax environment state in a Python closure) to the 2-arg
interface expected by ``run_ablation_v2``.

``compute_return_weights`` normalises episode returns to per-sample advantage
weights, applying clipping and optional wins-only binarisation.

History schema
--------------
Every key is always present; unused diagnostics are appended as ``None``::

    history = {
        # Per gradient step
        "step":                [],  # int gradient step index
        "loss":                [],  # float training loss
        "grad_norm":           [],  # float global gradient norm
        # Per eval interval (every eval_every steps)
        "eval_step":           [],  # int gradient step at eval point
        "eval_score":          [],  # float eval episode return
        "grad_align":          [],  # float cosine sim (RL vs BC gradient)
        "repr_drift":          [],  # float total L2 parameter drift
        "output_kl":           [],  # float mean KL on probe batch
        "per_t_loss":          [],  # list[float] length n_bins
        "token_entropy":       [],  # float mean token entropy
        "collapse_fraction":   [],  # float plan collapse fraction
        "per_layer_grad_norm": [],  # dict[str, float] per-layer norms
    }
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from src.ablations.losses import _base_elbo
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


# ---------------------------------------------------------------------------
# Return weight utility
# ---------------------------------------------------------------------------

def compute_return_weights(
    flat_returns: jnp.ndarray,
    wins_only: bool = False,
    win_threshold: float = 0.0,
    cap: float = 5.0,
    floor: float = 0.1,
) -> jnp.ndarray:
    """Normalise episode returns to per-sample advantage weights.

    Args:
        flat_returns:  ``[B]`` float episode returns.
        wins_only:     If ``True``, return binary weights (1 if return >
                       ``win_threshold``, else 0).
        win_threshold: Threshold above which a window is a "win".
        cap:           Upper clip for normalised weights.
        floor:         Lower clip; prevents zero gradients (0.1 default).

    Returns:
        ``[B]`` advantage weights in ``{0, 1}`` (wins-only) or
        ``[floor, cap]`` (normalised).
    """
    if wins_only:
        return (flat_returns > win_threshold).astype(jnp.float32)
    clipped = jnp.clip(flat_returns, 0.0, None)
    weights = clipped / (jnp.mean(clipped) + 1e-8)
    return jnp.clip(weights, floor, cap)


# ---------------------------------------------------------------------------
# Stateful rollout adapter
# ---------------------------------------------------------------------------

def make_stateful_rollout_fn(
    raw_collect: Callable,
    initial_env_state: Any,
    initial_obs: jnp.ndarray,
    initial_done: jnp.ndarray,
    initial_hstate: Any,
    wins_only: bool = False,
    win_threshold: float = 0.0,
    return_weight_cap: float = 5.0,
    return_weight_floor: float = 0.1,
) -> RolloutFn:
    """Adapt a 5-arg stateful collect_rollout to the 2-arg ``(rng, params)`` interface.

    The notebook's ``collect_rollout`` carries gymnax environment state as
    explicit arguments::

        raw_collect(env_state, obs, done, hstate, rng)
            -> (env_state, obs, done, hstate, rng,
                flat_obs, flat_acts, flat_valid, flat_returns, env_score)

    ``run_ablation_v2`` expects::

        rollout_fn(rng, params) -> (obs, acts, valid, advantages)

    This adapter maintains environment state in a mutable Python closure.
    Only ``raw_collect`` itself needs to be JIT-compiled; the wrapper is
    Python-level and is never traced.

    Args:
        raw_collect:         JIT-compiled 5-arg collect_rollout from the notebook.
        initial_env_state:   Initial gymnax environment state.
        initial_obs:         ``[E, obs_dim]`` initial observations.
        initial_done:        ``[E]`` initial done flags.
        initial_hstate:      Initial PPO hidden state.
        wins_only:           If ``True``, return binary win/loss weights.
        win_threshold:       Advantage threshold for "win" classification.
        return_weight_cap:   Upper clip for normalised return weights.
        return_weight_floor: Lower clip (prevents zero gradients).

    Returns:
        A callable ``(rng, params) -> (obs, acts, valid, advantages)``
        suitable for use as the ``collect_rollout`` argument of
        :func:`run_ablation_v2`.
    """
    _state: dict[str, Any] = {
        "env_state": initial_env_state,
        "obs":       initial_obs,
        "done":      initial_done,
        "hstate":    initial_hstate,
    }

    def adapted_rollout(
        rng: jax.Array,
        params: Any,  # noqa: ARG001 — not used; PPO policy drives rollout
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Collect one rollout and return (obs, acts, valid, advantages).

        Args:
            rng:    PRNG key for the rollout.
            params: Diffusion model parameters (ignored; PPO drives rollout).

        Returns:
            Tuple ``(flat_obs, flat_acts, flat_valid, advantages)``.
        """
        (
            env_state_new, obs_new, done_new, hstate_new, _,
            flat_obs, flat_acts, flat_valid, flat_returns, _,
        ) = raw_collect(
            _state["env_state"], _state["obs"],
            _state["done"], _state["hstate"], rng,
        )
        _state["env_state"] = env_state_new
        _state["obs"] = obs_new
        _state["done"] = done_new
        _state["hstate"] = hstate_new
        adv = compute_return_weights(
            flat_returns,
            wins_only=wins_only,
            win_threshold=win_threshold,
            cap=return_weight_cap,
            floor=return_weight_floor,
        )
        return flat_obs, flat_acts, flat_valid, adv

    return adapted_rollout


# ---------------------------------------------------------------------------
# Canonical training loop
# ---------------------------------------------------------------------------

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
) -> tuple[dict[str, Any], Any]:
    """Run a single ablation experiment and return ``(history, final_params)``.

    The training loop:

    1. Collects a rollout batch via ``collect_rollout(rng, params)``.
    2. Computes one minibatch gradient step with ``loss_fn``.
    3. Every ``eval_every`` steps, evaluates the policy and records diagnostics.

    History schema (all keys always present; ``None`` when not applicable):

    - ``'step'``: gradient step index (per step).
    - ``'loss'``: training loss (per step).
    - ``'grad_norm'``: global gradient norm (per step).
    - ``'eval_step'``: gradient step index at eval (per eval interval).
    - ``'eval_score'``: eval episode return (per eval interval).
    - ``'grad_align'``: cosine similarity vs BC gradient (per eval interval).
    - ``'repr_drift'``: total L2 parameter drift from pretrained (per eval interval).
    - ``'output_kl'``: mean KL divergence on probe batch (per eval interval).
    - ``'per_t_loss'``: ``[n_bins]`` per-t-bin loss (per eval interval).
    - ``'token_entropy'``: mean token entropy (per eval interval).
    - ``'collapse_fraction'``: plan collapse fraction (per eval interval).
    - ``'per_layer_grad_norm'``: dict of per-layer norms (per eval interval).

    Args:
        name:               Ablation method name (for logging).
        loss_fn:            Loss function factory output; returns ``(loss, info)``.
        apply_fn:           Eval-mode model apply closure.
        apply_train_fn:     Train-mode model apply closure.
        collect_rollout:    ``(rng, params) -> (obs, acts, valid, advantages)``.
        pretrained_params:  Frozen reference parameters.
        schedule_fn:        alpha(t) noise schedule.
        schedule_deriv_fn:  d(alpha)/dt.
        num_actions:        Real action vocabulary size.
        plan_horizon:       Plan sequence length H.
        obs_dim:            Observation vector dimensionality (documentation only).
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
        probe_obs:          ``[B_probe, obs_dim]`` held-out observations for KL/drift.
                            If ``None``, uses the first 64 observations from the
                            most recent rollout as a proxy.
        frozen_backbone:    Zero backbone gradients (only head updated).
        gradient_surgery:   Project out BC-conflicting gradient components.
        bc_loss_fn:         BC loss used by gradient surgery (required when
                            ``gradient_surgery=True``).
        step_dependent_loss: Pass ``step_idx`` kwarg to ``loss_fn`` (curriculum).
        eval_fn:            Optional ``(params, rng) -> float`` evaluation function.
                            If ``None``, ``eval_score`` is not recorded.

    Returns:
        Tuple ``(history, final_params)`` where ``history`` is the metrics
        dict and ``final_params`` is the trained parameter pytree.

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
    @jax.jit  # ← compiled; no Python side effects below this line
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

        # Optional frozen backbone (keep only Dense_5 output head gradients).
        if frozen_backbone:
            def _zero_unless_head(
                path: tuple, leaf: jnp.ndarray
            ) -> jnp.ndarray:
                path_str = "/".join(
                    str(p.key) for p in path if hasattr(p, "key")
                )
                # Dense_5 is the output head (Dense_0/1=obs_enc, Dense_2=obs_tok,
                # Dense_3/4=time_emb, Dense_5=action logit head).
                return leaf if "Dense_5" in path_str else jnp.zeros_like(leaf)

            grads = jax.tree_util.tree_map_with_path(_zero_unless_head, grads)

        new_state = st.apply_gradients(grads=grads)
        return new_state, grads, info

    # History accumulation — all keys always present.
    history: dict[str, list] = {
        # Per gradient step
        "step":                [],
        "loss":                [],
        "grad_norm":           [],
        # Per eval interval
        "eval_step":           [],
        "eval_score":          [],
        "grad_align":          [],
        "repr_drift":          [],
        "output_kl":           [],
        "per_t_loss":          [],
        "token_entropy":       [],
        "collapse_fraction":   [],
        "per_layer_grad_norm": [],
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
            obs_b  = obs_b[idx]
            acts_b = acts_b[idx]
            valid_b = valid_b[idx]
            adv_b  = adv_b[idx]

        # Cache probe observations from the first rollout.
        if _probe_obs_cache is None:
            _probe_obs_cache = obs_b[:min(64, obs_b.shape[0])]

        state, grads, info = _grad_step(
            state, acts_b, obs_b, valid_b, adv_b,
            step_rng, jnp.array(step_idx),
        )

        loss_val = float(info.get("loss", info.get("Loss", 0.0)))
        grad_norm_val = float(optax.tree.norm(grads))
        history["step"].append(step_idx)
        history["loss"].append(loss_val)
        history["grad_norm"].append(grad_norm_val)

        # Diagnostic evaluation at eval intervals.
        if step_idx % eval_every == 0:
            history["eval_step"].append(step_idx)

            # Gradient alignment (RL vs BC/uniform).
            rng, align_rng_rl, align_rng_bc = jax.random.split(rng, 3)

            def _bc_loss_fn(p: Any) -> tuple[jnp.ndarray, dict]:
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

            # Representation drift (L2 distance from pretrained).
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
            else:
                history["output_kl"].append(None)

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

            # Per-layer gradient norms at eval intervals (moved from per-step
            # to reduce memory overhead for long runs).
            layer_norms = compute_per_layer_grad_norm(
                jax.tree.map(lambda g: jnp.array(g), grads)
            )
            history["per_layer_grad_norm"].append(
                {k: float(v) for k, v in layer_norms.items()}
            )

            # External evaluation function (returns a scalar score).
            if eval_fn is not None:
                rng, eval_rng = jax.random.split(rng)
                score = float(eval_fn(state.params, eval_rng))
                history["eval_score"].append(score)
            else:
                history["eval_score"].append(None)

            logger.info(
                "[%s] step=%d  loss=%.4f  drift=%.3f  align=%.3f",
                name, step_idx, loss_val,
                history["repr_drift"][-1],
                grad_align,
            )

    logger.info("[%s] Training complete.", name)
    return history, state.params
