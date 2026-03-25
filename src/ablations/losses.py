"""Ablation loss-function factories for ReMDM RL fine-tuning experiments.

All public functions follow the factory pattern::

    loss_fn = make_loss_<name>(apply_fn, ..., **hyperparams)
    loss, info = loss_fn(params, rng, acts, obs, valid, advantages)

The returned ``loss_fn`` is a pure JAX function, JIT- and vmap-compatible.

Standard signature
------------------
    loss_fn(
        params    : Any,         # model parameter pytree
        rng       : jax.Array,   # PRNG key
        acts      : Array[B, H], # int32 action sequences
        obs       : Array[B, D], # float32 observations
        valid     : Array[B],    # bool validity mask
        advantages: Array[B],    # float per-sample weights / returns
    ) -> tuple[Array, dict]      # (scalar loss, info dict)

For step-dependent losses (e.g. curriculum), the ``loss_fn`` accepts an
additional keyword argument ``step_idx: jax.Array = jnp.array(0)``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from src.diffusion.forward import forward_process
from src.diffusion.schedules import ScheduleFn

_EPS: float = 1e-5
_MAX_WEIGHT: float = 1000.0

# Type alias for model apply functions.
ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _base_elbo(
    apply_fn: ModelApplyFn,
    params: Any,
    rng: jax.Array,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    t_min: float = _EPS,
    t_max: float = 1.0,
    advantages: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """MDLM ELBO with configurable diffusion-time range [t_min, t_max].

    Mirrors :func:`src.diffusion.loss.compute_loss` but allows restricting
    the training signal to a sub-interval of the diffusion timeline.

    Args:
        apply_fn:          Model apply closure (train mode).
        params:            Model parameter pytree.
        rng:               PRNG key.
        acts:              ``[B, H]`` int32 ground-truth actions.
        obs:               ``[B, D]`` float32 observations.
        valid:             ``[B]`` bool validity mask.
        num_actions:       Real action vocabulary size (MASK = num_actions).
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        sigma_t:           ReMDM remasking correction (0 = standard MDLM).
        label_smoothing:   Cross-entropy smoothing epsilon.
        t_min:             Minimum sampled diffusion time (default: eps).
        t_max:             Maximum sampled diffusion time (default: 1.0).
        advantages:        ``[B]`` per-sample weights applied to per-sample loss.

    Returns:
        Tuple of ``(loss, info_dict)``.
    """
    B = acts.shape[0]
    mask_id = num_actions
    rng, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)

    t = jax.random.uniform(t_rng, (B,), minval=t_min, maxval=t_max)
    alpha_t = schedule_fn(t)

    neg_alpha_dot = -schedule_deriv_fn(t)
    weight = (1.0 - sigma_t) * neg_alpha_dot / jnp.maximum(1.0 - alpha_t, _EPS)
    weight = jnp.minimum(weight, _MAX_WEIGHT)

    z_t = forward_process(mask_rng, acts, alpha_t, mask_id)
    logits = apply_fn(params, obs, z_t, t, drop_rng)  # [B, H, num_actions]

    is_masked = (z_t == mask_id).astype(jnp.float32)  # [B, H]
    valid_masked = is_masked * valid[:, None].astype(jnp.float32)  # [B, H]

    targets = jax.nn.one_hot(acts, num_actions)
    if label_smoothing > 0.0:
        targets = (1.0 - label_smoothing) * targets + label_smoothing / num_actions

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(targets * log_probs, axis=-1)  # [B, H]

    n_masked = jnp.maximum(valid_masked.sum(axis=-1), 1.0)  # [B]
    per_sample = weight * (ce * valid_masked).sum(axis=-1) / n_masked  # [B]

    if advantages is not None:
        per_sample = per_sample * jax.lax.stop_gradient(advantages)

    loss = jnp.mean(per_sample)

    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == acts).astype(jnp.float32)
    acc = jnp.sum(correct * valid_masked) / jnp.maximum(valid_masked.sum(), 1.0)

    info: dict[str, jnp.ndarray] = {
        "loss": loss,
        "unweighted_loss": jnp.mean(
            (ce * valid_masked).sum(axis=-1) / n_masked
        ),
        "mean_t": jnp.mean(t),
        "frac_masked": jnp.mean(is_masked),
        "accuracy": acc,
    }
    if advantages is not None:
        info["adv_mean"] = jnp.mean(advantages)
        info["adv_std"] = jnp.std(advantages)
    return loss, info


def _output_kl_divergence(
    apply_fn: ModelApplyFn,
    params: Any,
    ref_params: Any,
    obs: jnp.ndarray,
    rng: jax.Array,
    num_actions: int,
    plan_horizon: int,
) -> jnp.ndarray:
    """Compute mean KL(p_theta || p_ref) on a probe batch at mid-range t.

    Args:
        apply_fn:    Model apply closure (eval mode, rng=None).
        params:      Current model parameters.
        ref_params:  Reference (pretrained) parameters.
        obs:         ``[B, D]`` probe observations.
        rng:         PRNG key.
        num_actions: Action vocabulary size.
        plan_horizon: Plan sequence length H.

    Returns:
        Scalar mean KL divergence.
    """
    B = obs.shape[0]
    mask_id = num_actions
    rng, z_rng = jax.random.split(rng)

    # Use a mid-range t and fully-masked sequence as probe.
    t_probe = jnp.full((B,), 0.5)
    z_probe = jnp.full((B, plan_horizon), mask_id, dtype=jnp.int32)

    logits_cur = apply_fn(params, obs, z_probe, t_probe, None)  # [B, H, A]
    logits_ref = apply_fn(ref_params, obs, z_probe, t_probe, None)  # [B, H, A]

    log_p = jax.nn.log_softmax(logits_cur, axis=-1)
    log_q = jax.nn.log_softmax(logits_ref, axis=-1)
    p = jnp.exp(log_p)

    kl = jnp.sum(p * (log_p - log_q), axis=-1)  # [B, H]
    return jnp.mean(kl)


# ---------------------------------------------------------------------------
# Public loss factories
# ---------------------------------------------------------------------------

def make_loss_baseline(
    apply_fn: ModelApplyFn,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """Standard return-weighted MDLM ELBO (offline BC reference).

    Args:
        apply_fn:          Model apply closure (train mode).
        num_actions:       Real action vocabulary size.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        sigma_t:           ReMDM remasking correction.
        label_smoothing:   Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        return _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )

    return loss_fn


def make_loss_kl(
    apply_fn: ModelApplyFn,
    pretrained_params: Any,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    kl_coef: float = 0.1,
    plan_horizon: int = 32,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """Return-weighted ELBO plus a KL penalty toward the pretrained model.

    The KL term ``KL(theta || theta_pretrained)`` is estimated on the current
    batch at a mid-range probe time t=0.5.

    Args:
        apply_fn:          Model apply closure (eval-compatible; no dropout needed).
        pretrained_params: Frozen reference parameters.
        num_actions:       Real action vocabulary size.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        kl_coef:           KL penalty coefficient lambda.
        plan_horizon:      Plan sequence length H.
        sigma_t:           ReMDM remasking correction.
        label_smoothing:   Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        rng, kl_rng = jax.random.split(rng)
        elbo_loss, info = _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )
        kl = _output_kl_divergence(
            apply_fn, params, pretrained_params, obs, kl_rng,
            num_actions, plan_horizon,
        )
        total = elbo_loss + kl_coef * kl
        info["kl_penalty"] = kl
        info["loss"] = total
        return total, info

    return loss_fn


def make_loss_bc_wins(
    apply_fn: ModelApplyFn,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    win_threshold: float = 0.0,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """Uniform behavioural cloning restricted to "winning" trajectories only.

    Samples with ``advantages > win_threshold`` are included with equal weight;
    losing samples are zeroed out.

    Args:
        apply_fn:       Model apply closure (train mode).
        num_actions:    Real action vocabulary size.
        schedule_fn:    alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        win_threshold:  Advantage threshold for "win" classification.
        sigma_t:        ReMDM remasking correction.
        label_smoothing: Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        # Gate validity on win condition; use uniform weights for wins.
        win_mask = (advantages > win_threshold).astype(jnp.float32)
        effective_valid = valid.astype(jnp.float32) * win_mask
        # Uniform advantages for winning samples.
        uniform_adv = jnp.ones_like(advantages)
        loss, info = _base_elbo(
            apply_fn, params, rng, acts, obs, effective_valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=uniform_adv,
        )
        info["win_frac"] = jnp.mean(win_mask)
        return loss, info

    return loss_fn


def make_loss_low_t(
    apply_fn: ModelApplyFn,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    t_max_low: float = 0.2,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """Return-weighted ELBO restricted to low-noise diffusion times t ∈ [ε, t_max_low].

    Focusing on low-t (high-quality signal) may stabilise RL fine-tuning by
    avoiding the noisy, highly-masked high-t regime.

    Args:
        apply_fn:       Model apply closure (train mode).
        num_actions:    Real action vocabulary size.
        schedule_fn:    alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        t_max_low:      Upper bound on sampled t (default 0.2).
        sigma_t:        ReMDM remasking correction.
        label_smoothing: Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        return _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            t_min=_EPS, t_max=t_max_low,
            advantages=advantages,
        )

    return loss_fn


def make_loss_ewc(
    apply_fn: ModelApplyFn,
    pretrained_params: Any,
    fisher: Any,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    ewc_lambda: float = 100.0,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """Return-weighted ELBO plus an Elastic Weight Consolidation penalty.

    EWC discourages deviation from pretrained weights in proportion to the
    diagonal Fisher information F_i: ``L_ewc = sum_i F_i (theta_i - theta*_i)^2``.

    The Fisher matrix ``fisher`` must be pre-computed via
    :func:`src.ablations.techniques.compute_ewc_fisher`.

    Args:
        apply_fn:          Model apply closure (train mode).
        pretrained_params: Frozen reference parameters theta*.
        fisher:            Diagonal Fisher information pytree (same structure as params).
        num_actions:       Real action vocabulary size.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        ewc_lambda:        EWC penalty coefficient.
        sigma_t:           ReMDM remasking correction.
        label_smoothing:   Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """
    # Pre-stop-gradient the Fisher and reference params so they are never
    # differentiated through.
    _frozen_fisher = jax.lax.stop_gradient(fisher)
    _frozen_ref = jax.lax.stop_gradient(pretrained_params)

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        elbo_loss, info = _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )
        # EWC penalty: sum_i F_i * (theta_i - theta*_i)^2 / 2
        ewc_terms = jax.tree.map(
            lambda f, p, p0: f * (p - p0) ** 2,
            _frozen_fisher, params, _frozen_ref,
        )
        ewc_loss = sum(
            jnp.sum(leaf) for leaf in jax.tree.leaves(ewc_terms)
        ) * 0.5
        total = elbo_loss + ewc_lambda * ewc_loss
        info["ewc_loss"] = ewc_loss
        info["loss"] = total
        return total, info

    return loss_fn


def make_loss_mixed_replay(
    apply_fn: ModelApplyFn,
    offline_obs: jnp.ndarray,
    offline_acts: jnp.ndarray,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    online_ratio: float = 0.5,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """Return-weighted ELBO on a mixture of online rollout and offline PPO data.

    At each step, a fraction ``online_ratio`` of the minibatch is drawn from
    the online batch (with GRPO advantages); the remainder is drawn from the
    frozen offline buffer (uniform advantages = 1).

    Args:
        apply_fn:      Model apply closure (train mode).
        offline_obs:   ``[N, D]`` offline observation buffer.
        offline_acts:  ``[N, H]`` offline action buffer.
        num_actions:   Real action vocabulary size.
        schedule_fn:   alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        online_ratio:  Fraction of minibatch from online data (rest from offline).
        sigma_t:       ReMDM remasking correction.
        label_smoothing: Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """
    _offline_obs = jax.lax.stop_gradient(offline_obs)
    _offline_acts = jax.lax.stop_gradient(offline_acts)
    offline_n = offline_obs.shape[0]

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        B = acts.shape[0]
        n_online = int(B * online_ratio)
        n_offline = B - n_online

        rng, idx_rng = jax.random.split(rng)
        idx = jax.random.randint(idx_rng, (n_offline,), 0, offline_n)

        mixed_obs = jnp.concatenate(
            [obs[:n_online], _offline_obs[idx]], axis=0
        )
        mixed_acts = jnp.concatenate(
            [acts[:n_online], _offline_acts[idx]], axis=0
        )
        mixed_valid = jnp.concatenate(
            [valid[:n_online], jnp.ones(n_offline)], axis=0
        )
        # Uniform advantages for offline portion.
        mixed_adv = jnp.concatenate(
            [advantages[:n_online], jnp.ones(n_offline)], axis=0
        )

        loss, info = _base_elbo(
            apply_fn, params, rng, mixed_acts, mixed_obs, mixed_valid,
            num_actions, schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=mixed_adv,
        )
        info["offline_frac"] = jnp.array(n_offline / B)
        return loss, info

    return loss_fn


def make_loss_t_curriculum(
    apply_fn: ModelApplyFn,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    t_curriculum_start: float = 0.2,
    t_curriculum_end: float = 1.0,
    t_curriculum_steps: int = 200,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """ELBO with a diffusion-time curriculum: t_max grows from t_start to t_end.

    Training begins restricted to low-noise (refinement) steps, then
    gradually expands to the full range over ``t_curriculum_steps`` gradient
    steps.  Requires ``step_idx`` to be passed at call time.

    Args:
        apply_fn:             Model apply closure (train mode).
        num_actions:          Real action vocabulary size.
        schedule_fn:          alpha(t) noise schedule.
        schedule_deriv_fn:    d(alpha)/dt.
        t_curriculum_start:   Initial t_max (low-noise restriction).
        t_curriculum_end:     Final t_max after curriculum completes.
        t_curriculum_steps:   Steps over which to expand the range.
        sigma_t:              ReMDM remasking correction.
        label_smoothing:      Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages,
                  step_idx=jnp.array(0)) -> (loss, info)``.
    """
    _t_range = t_curriculum_end - t_curriculum_start
    _steps = float(t_curriculum_steps)

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
        step_idx: jnp.ndarray = jnp.array(0),
    ) -> tuple[jnp.ndarray, dict]:
        progress = jnp.minimum(step_idx.astype(jnp.float32) / _steps, 1.0)
        t_max = t_curriculum_start + _t_range * progress
        loss, info = _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            t_min=_EPS, t_max=t_max,
            advantages=advantages,
        )
        info["curriculum_t_max"] = t_max
        return loss, info

    return loss_fn


def make_loss_entropy_reg(
    apply_fn: ModelApplyFn,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    entropy_coef: float = 0.01,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    plan_horizon: int = 32,
    **_kwargs: Any,
) -> Callable:
    """ELBO minus an entropy bonus to encourage diverse action distributions.

    Adds ``-entropy_coef * H[p(a|obs)]`` to the loss, where entropy is
    computed from the model's predicted distribution at a mid-range t.

    Args:
        apply_fn:      Model apply closure (train mode).
        num_actions:   Real action vocabulary size.
        schedule_fn:   alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        entropy_coef:  Weight of the entropy bonus (subtracted from loss).
        sigma_t:       ReMDM remasking correction.
        label_smoothing: Cross-entropy smoothing epsilon.
        plan_horizon:  Plan sequence length H (for probe computation).

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """
    mask_id = num_actions

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        rng, ent_rng, mask_rng = jax.random.split(rng, 3)
        elbo_loss, info = _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )
        # Entropy probe: mid-range t, fully-masked sequence.
        B = obs.shape[0]
        t_mid = jnp.full((B,), 0.5)
        z_probe = jnp.full((B, plan_horizon), mask_id, dtype=jnp.int32)
        logits = apply_fn(params, obs, z_probe, t_mid, ent_rng)  # [B, H, A]
        probs = jax.nn.softmax(logits, axis=-1)
        entropy = -jnp.sum(
            probs * jnp.log(jnp.where(probs > 0, probs, 1.0)), axis=-1
        )  # [B, H]
        mean_entropy = jnp.mean(entropy)
        total = elbo_loss - entropy_coef * mean_entropy
        info["entropy_bonus"] = mean_entropy
        info["loss"] = total
        return total, info

    return loss_fn


def make_loss_token_advantage(
    apply_fn: ModelApplyFn,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """ELBO with per-token advantage weighting proportional to prediction uncertainty.

    Each token position h receives a weight inversely proportional to the
    model's confidence at that position.  The weights are normalised per
    sample, so the total gradient magnitude is comparable to the baseline.

    Formally: ``token_weight[b, h] = H_h / sum_h H_h * adv_b``
    where ``H_h`` is the entropy of the predicted distribution at position h.

    Args:
        apply_fn:      Model apply closure (train mode).
        num_actions:   Real action vocabulary size.
        schedule_fn:   alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        sigma_t:       ReMDM remasking correction.
        label_smoothing: Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """
    mask_id = num_actions

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        B, H = acts.shape
        rng, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)

        t = jax.random.uniform(t_rng, (B,), minval=_EPS, maxval=1.0)
        alpha_t = schedule_fn(t)

        neg_alpha_dot = -schedule_deriv_fn(t)
        weight = (1.0 - sigma_t) * neg_alpha_dot / jnp.maximum(
            1.0 - alpha_t, _EPS
        )
        weight = jnp.minimum(weight, _MAX_WEIGHT)  # [B]

        z_t = forward_process(mask_rng, acts, alpha_t, mask_id)
        logits = apply_fn(params, obs, z_t, t, drop_rng)  # [B, H, num_actions]

        is_masked = (z_t == mask_id).astype(jnp.float32)
        valid_masked = is_masked * valid[:, None].astype(jnp.float32)  # [B, H]

        # Per-token entropy as uncertainty estimate.
        probs = jax.nn.softmax(logits, axis=-1)
        token_entropy = -jnp.sum(
            probs * jnp.log(jnp.where(probs > 0, probs, 1.0)), axis=-1
        )  # [B, H]

        # Normalise per sample over masked positions.
        entropy_masked = token_entropy * valid_masked
        total_entropy = jnp.maximum(entropy_masked.sum(axis=-1, keepdims=True), _EPS)
        token_w = entropy_masked / total_entropy  # [B, H], sums to 1 over masked positions

        # Scale by global advantage.
        token_adv = token_w * advantages[:, None]  # [B, H]

        targets = jax.nn.one_hot(acts, num_actions)
        if label_smoothing > 0.0:
            targets = (1.0 - label_smoothing) * targets + label_smoothing / num_actions

        log_probs = jax.nn.log_softmax(logits, axis=-1)
        ce = -jnp.sum(targets * log_probs, axis=-1)  # [B, H]

        # Weighted sum: weight[B] * token_adv[B, H] * ce[B, H]
        per_sample = weight * jnp.sum(ce * token_adv, axis=-1)  # [B]
        loss = jnp.mean(per_sample)

        preds = jnp.argmax(logits, axis=-1)
        correct = (preds == acts).astype(jnp.float32)
        acc = jnp.sum(correct * valid_masked) / jnp.maximum(valid_masked.sum(), 1.0)

        info: dict[str, jnp.ndarray] = {
            "loss": loss,
            "accuracy": acc,
            "mean_token_entropy": jnp.mean(token_entropy * valid_masked),
            "adv_mean": jnp.mean(advantages),
            "adv_std": jnp.std(advantages),
        }
        return loss, info

    return loss_fn


def make_loss_trust_region(
    apply_fn: ModelApplyFn,
    pretrained_params: Any,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    trust_region_kl: float = 0.05,
    kl_penalty_coef: float = 100.0,
    plan_horizon: int = 32,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    **_kwargs: Any,
) -> Callable:
    """ELBO with a hinge-loss trust-region constraint on the output KL.

    Adds a soft barrier: ``kl_penalty_coef * max(KL - trust_region_kl, 0)``.
    This quadratically penalises updates that push the policy more than
    ``trust_region_kl`` nats away from the pretrained reference.

    Args:
        apply_fn:          Model apply closure.
        pretrained_params: Frozen reference parameters.
        num_actions:       Real action vocabulary size.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt.
        trust_region_kl:   KL budget before the penalty activates (nats).
        kl_penalty_coef:   Coefficient of the hinge penalty.
        plan_horizon:      Plan sequence length H.
        sigma_t:           ReMDM remasking correction.
        label_smoothing:   Cross-entropy smoothing epsilon.

    Returns:
        ``loss_fn(params, rng, acts, obs, valid, advantages) -> (loss, info)``.
    """
    _frozen_ref = jax.lax.stop_gradient(pretrained_params)

    def loss_fn(
        params: Any,
        rng: jax.Array,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        rng, kl_rng = jax.random.split(rng)
        elbo_loss, info = _base_elbo(
            apply_fn, params, rng, acts, obs, valid, num_actions,
            schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )
        kl = _output_kl_divergence(
            apply_fn, params, _frozen_ref, obs, kl_rng,
            num_actions, plan_horizon,
        )
        # Hinge loss: only penalise if KL exceeds budget.
        penalty = kl_penalty_coef * jnp.maximum(kl - trust_region_kl, 0.0)
        total = elbo_loss + penalty
        info["kl"] = kl
        info["trust_region_penalty"] = penalty
        info["loss"] = total
        return total, info

    return loss_fn
