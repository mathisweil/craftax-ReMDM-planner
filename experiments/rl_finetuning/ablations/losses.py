"""Pure loss factory functions for all RL fine-tuning ablations.

Each factory returns a ``LossFn``:
    loss_fn(params, acts, obs, valid, rng, advantages) -> scalar

All factories accept a ``LossContext`` that bundles the shared context
(apply_fn, ref_params, schedule functions, config) so the factories
themselves are pure and free of global state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from src.diffusion.loss import compute_loss
from src.diffusion.forward import forward_process
from src.diffusion.schedules import ScheduleFn

# Loss signature: (params, acts, obs, valid, rng, advantages) -> scalar
LossFn = Callable[[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array, jnp.ndarray], jnp.ndarray]

_EPS: float = 1e-5
_MAX_WEIGHT: float = 1000.0


@dataclass
class LossContext:
    """Shared context for all loss factory functions.

    Args:
        apply_fn:          Training apply fn: (params, obs, z_t, t, rng) -> logits.
        ref_params:        Frozen pretrained parameters for regularisation losses.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Size of the discrete action vocabulary.
        config:            Full UPPERCASE config dict.
    """

    apply_fn: Callable
    ref_params: Any
    schedule_fn: ScheduleFn
    schedule_deriv_fn: ScheduleFn
    num_actions: int
    config: dict


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _core_loss(
    ctx: LossContext,
    params: Any,
    rng: jax.Array,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
    advantages: jnp.ndarray | None,
    t_min: float = _EPS,
    t_max: float = 1.0,
) -> jnp.ndarray:
    """Call ``compute_loss`` with context-provided schedule and config.

    Args:
        ctx:        Shared loss context.
        params:     Current model parameters.
        rng:        PRNG key.
        acts:       ``[B, H]`` int32 action sequences.
        obs:        ``[B, obs_dim]`` float32 observations.
        valid:      ``[B]`` validity mask.
        advantages: Optional ``[B]`` advantage weights.
        t_min:      Lower bound for uniform t sampling.
        t_max:      Upper bound for uniform t sampling.

    Returns:
        Scalar loss value.
    """
    loss, _ = compute_loss(
        ctx.apply_fn,
        params,
        rng,
        acts,
        obs,
        valid,
        ctx.num_actions,
        ctx.schedule_fn,
        ctx.schedule_deriv_fn,
        sigma_t=ctx.config.get("TRAIN_SIGMA", 0.0),
        label_smoothing=ctx.config.get("LABEL_SMOOTHING", 0.0),
        advantages=advantages,
        t_min=t_min,
        t_max=t_max,
    )
    return loss


def _kl_penalty(
    ctx: LossContext,
    params: Any,
    rng: jax.Array,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """KL divergence KL(current || pretrained) on masked positions.

    Args:
        ctx:    Loss context (ref_params must be set).
        params: Current model parameters.
        rng:    PRNG key.
        acts:   ``[B, H]`` int32 action sequences.
        obs:    ``[B, obs_dim]`` float32 observations.
        valid:  ``[B]`` validity mask.

    Returns:
        Scalar mean KL on masked positions.
    """
    B = acts.shape[0]
    rng, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)
    t = jax.random.uniform(t_rng, (B,), minval=_EPS, maxval=1.0)
    alpha_t = ctx.schedule_fn(t)
    z_t = forward_process(mask_rng, acts, alpha_t, ctx.num_actions)

    is_masked = (z_t == ctx.num_actions).astype(jnp.float32)
    valid_m = is_masked * valid[:, None].astype(jnp.float32)

    cur_logits = ctx.apply_fn(params, obs, z_t, t, drop_rng)
    ref_logits = ctx.apply_fn(
        jax.lax.stop_gradient(ctx.ref_params), obs, z_t, t, drop_rng
    )
    cur_log = jax.nn.log_softmax(cur_logits, axis=-1)
    ref_log = jax.nn.log_softmax(ref_logits, axis=-1)
    cur_prob = jnp.exp(cur_log)

    kl = (cur_prob * (cur_log - ref_log)).sum(-1)  # [B, H]
    kl_masked = (kl * valid_m).sum(-1) / jnp.maximum(valid_m.sum(-1), 1.0)
    return kl_masked.mean()


def _entropy_bonus(
    ctx: LossContext,
    params: Any,
    rng: jax.Array,
    acts: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
) -> jnp.ndarray:
    """Mean entropy of p_theta over masked positions.

    Args:
        ctx:    Loss context.
        params: Current model parameters.
        rng:    PRNG key.
        acts:   ``[B, H]`` int32 action sequences.
        obs:    ``[B, obs_dim]`` float32 observations.
        valid:  ``[B]`` validity mask.

    Returns:
        Scalar mean entropy.
    """
    B = acts.shape[0]
    rng, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)
    t = jax.random.uniform(t_rng, (B,), minval=_EPS, maxval=1.0)
    alpha_t = ctx.schedule_fn(t)
    z_t = forward_process(mask_rng, acts, alpha_t, ctx.num_actions)

    is_masked = (z_t == ctx.num_actions).astype(jnp.float32)
    valid_m = is_masked * valid[:, None].astype(jnp.float32)

    logits = ctx.apply_fn(params, obs, z_t, t, drop_rng)  # [B, H, V]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    entropy = -jnp.sum(probs * log_probs, axis=-1)  # [B, H]

    return (entropy * valid_m).sum() / jnp.maximum(valid_m.sum(), 1.0)


def _ewc_penalty(
    fisher: Any,
    params: Any,
    ref_params: Any,
) -> jnp.ndarray:
    """EWC penalty: lambda * sum(F_i * (theta_i - theta_i*)^2).

    Summation is over all parameters, using Python-level sum of JAX scalars
    (leaves count is fixed at trace time, so this is JIT-safe).

    Args:
        fisher:     Fisher diagonal pytree matching params structure.
        params:     Current model parameters.
        ref_params: Pretrained parameters (anchor).

    Returns:
        Scalar unweighted EWC penalty (caller multiplies by ewc_lambda).
    """
    return sum(
        (
            jnp.sum(f * (p - p_ref) ** 2)
            for f, p, p_ref in zip(
            jax.tree.leaves(fisher),
            jax.tree.leaves(params),
            jax.tree.leaves(ref_params),
        )
        ),
        jnp.array(0.0)
    )


# ---------------------------------------------------------------------------
# Group A: Regularisation / Constraint Methods
# ---------------------------------------------------------------------------


def make_loss_baseline(ctx: LossContext) -> LossFn:
    """Standard return-weighted ELBO — no modifications.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` implementing the baseline RL fine-tuning objective.
    """
    def loss_fn(params, acts, obs, valid, rng, advantages):
        return _core_loss(ctx, params, rng, acts, obs, valid, advantages)

    return loss_fn


def make_loss_kl_penalty(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO + soft KL penalty against the frozen pretrained model.

    Hypothesis: if this helps, catastrophic forgetting is the primary cause
    of collapse and soft regularisation suffices.

    Args:
        ctx: Shared loss context (ref_params required).

    Returns:
        ``LossFn`` with additive KL penalty.
    """
    kl_coef = ctx.config.get("KL_COEF", 0.1)

    def loss_fn(params, acts, obs, valid, rng, advantages):
        rng, kl_rng = jax.random.split(rng)
        rl = _core_loss(ctx, params, rng, acts, obs, valid, advantages)
        kl = _kl_penalty(ctx, params, kl_rng, acts, obs, valid)
        return rl + kl_coef * kl

    return loss_fn


def make_loss_ewc(ctx: LossContext, fisher: Any) -> LossFn:
    """Return-weighted ELBO + EWC penalty using pre-computed Fisher diagonal.

    Hypothesis: if EWC helps, catastrophic forgetting of pretrained
    representations is the proximate cause of collapse.

    Args:
        ctx:    Shared loss context (ref_params used as EWC anchor).
        fisher: Fisher diagonal pytree (pre-computed by estimate_fisher).

    Returns:
        ``LossFn`` with EWC regularisation.
    """
    ewc_lambda = ctx.config.get("EWC_LAMBDA", 100.0)
    ref_params = ctx.ref_params

    def loss_fn(params, acts, obs, valid, rng, advantages):
        rl = _core_loss(ctx, params, rng, acts, obs, valid, advantages)
        penalty = _ewc_penalty(fisher, params, ref_params)
        return rl + ewc_lambda * penalty

    return loss_fn


def make_loss_trust_region_kl(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO + hard KL trust region via quadratic barrier.

    When KL(current || pretrained) > threshold, applies a large quadratic
    penalty to enforce the trust region. This is a soft approximation
    of the hard constraint via dual ascent.

    Hypothesis: soft KL is insufficient; we need a hard constraint.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with hard KL trust region.
    """
    threshold = ctx.config.get("TRUST_REGION_KL", 0.05)

    def loss_fn(params, acts, obs, valid, rng, advantages):
        rng, kl_rng = jax.random.split(rng)
        rl = _core_loss(ctx, params, rng, acts, obs, valid, advantages)
        kl = _kl_penalty(ctx, params, kl_rng, acts, obs, valid)
        # Large quadratic penalty when KL exceeds threshold (barrier method)
        violation = jnp.maximum(kl - threshold, 0.0)
        return rl + 1e4 * violation ** 2

    return loss_fn


def make_loss_mixed_replay(ctx: LossContext) -> LossFn:
    """Baseline loss; mixed replay batching is handled at the training loop level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline (data mixing done externally).
    """
    return make_loss_baseline(ctx)


# ---------------------------------------------------------------------------
# Group B: Training Signal Modifications
# ---------------------------------------------------------------------------


def make_loss_bc_wins(ctx: LossContext) -> LossFn:
    """Uniform ELBO on all samples, ignoring advantages (BC on wins).

    The caller is expected to pre-filter to winning windows.
    This loss itself ignores advantages entirely.

    Hypothesis: if BC on wins helps, the issue is the return weighting,
    not the data distribution.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with advantages zeroed out.
    """
    def loss_fn(params, acts, obs, valid, rng, advantages):
        return _core_loss(ctx, params, rng, acts, obs, valid, advantages=None)

    return loss_fn


def make_loss_low_t(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO restricted to t ∈ [ε, t_max_low].

    Hypothesis: high-t gradients (coarse structure) dominate and are biased;
    restricting to low-t (fine detail) avoids the bias.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` using only low-t samples.
    """
    t_max = ctx.config.get("T_MAX_LOW", 0.2)

    def loss_fn(params, acts, obs, valid, rng, advantages):
        return _core_loss(ctx, params, rng, acts, obs, valid, advantages, t_min=_EPS, t_max=t_max)

    return loss_fn


def make_loss_t_curriculum(ctx: LossContext, current_iter: list[int]) -> LossFn:
    """Return-weighted ELBO with annealing t range (t-curriculum).

    The t range anneals from [t_start, 1.0] to [ε, t_end] over
    t_curriculum_steps iterations. ``current_iter`` is a mutable
    single-element list updated by the training loop.

    Hypothesis: the ordering of learning signals matters; coarse structure
    must be learned before fine detail.

    Args:
        ctx:          Shared loss context.
        current_iter: Mutable [int] container updated each iteration.

    Returns:
        ``LossFn`` with iteration-dependent t range.
    """
    t_start = ctx.config.get("T_CURRICULUM_START", 0.8)
    t_end = ctx.config.get("T_CURRICULUM_END", 0.2)
    steps = ctx.config.get("T_CURRICULUM_STEPS", 200)

    def loss_fn(params, acts, obs, valid, rng, advantages):
        frac = min(current_iter[0] / max(steps, 1), 1.0)
        t_min = _EPS + frac * (t_end - _EPS)
        t_max = 1.0 - frac * (1.0 - t_start)
        # Clamp to valid range
        t_min = float(jnp.clip(t_min, _EPS, 0.95))
        t_max = float(jnp.clip(t_max, t_min + 0.05, 1.0))
        return _core_loss(ctx, params, rng, acts, obs, valid, advantages, t_min=t_min, t_max=t_max)

    return loss_fn


def make_loss_entropy_bonus(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO minus entropy bonus.

    Entropy bonus encourages maintaining action diversity and prevents
    mode collapse.

    Hypothesis: collapse is a mode-collapse phenomenon; entropy regularisation
    is sufficient to stabilise training.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with entropy regularisation.
    """
    entropy_coef = ctx.config.get("ENTROPY_COEF", 0.01)

    def loss_fn(params, acts, obs, valid, rng, advantages):
        rng, ent_rng = jax.random.split(rng)
        rl = _core_loss(ctx, params, rng, acts, obs, valid, advantages)
        entropy = _entropy_bonus(ctx, params, ent_rng, acts, obs, valid)
        return rl - entropy_coef * entropy

    return loss_fn


def make_loss_gradient_surgery(ctx: LossContext) -> LossFn:
    """Baseline loss; gradient surgery projection handled at training loop level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline (PCGrad applied externally).
    """
    return make_loss_baseline(ctx)


def make_loss_advantage_clip(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO with PPO-style advantage clipping.

    Clips advantage weights to [1 - eps, 1 + eps] before applying.

    Hypothesis: large advantage magnitudes destabilise training; clipping
    is sufficient to stabilise.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with clipped advantages.
    """
    eps = ctx.config.get("ADV_CLIP_EPS", 0.2)

    def loss_fn(params, acts, obs, valid, rng, advantages):
        clipped = jnp.clip(advantages, 1.0 - eps, 1.0 + eps)
        return _core_loss(ctx, params, rng, acts, obs, valid, clipped)

    return loss_fn


def make_loss_normalized_adv(ctx: LossContext) -> LossFn:
    """Return-weighted ELBO with group-normalised advantages (GRPO-style).

    Normalises advantages as (A - mean(A)) / (std(A) + eps) over the batch,
    unlike the simpler mean-normalisation used in the baseline.

    Hypothesis: the current normalisation is too loose; std normalisation
    provides a cleaner gradient signal.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` with std-normalised advantages.
    """
    def loss_fn(params, acts, obs, valid, rng, advantages):
        mean = jnp.mean(advantages)
        std = jnp.std(advantages)
        norm_adv = (advantages - mean) / (std + 1e-8)
        return _core_loss(ctx, params, rng, acts, obs, valid, norm_adv)

    return loss_fn


# ---------------------------------------------------------------------------
# Group C: Architecture / Parameter Isolation — loss is always baseline
# ---------------------------------------------------------------------------


def make_loss_frozen_backbone(ctx: LossContext) -> LossFn:
    """Baseline loss; backbone freezing is handled at optimizer/mask level.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


def make_loss_param_isolation(ctx: LossContext) -> LossFn:
    """Baseline loss; parameter isolation is handled at optimizer/mask level.

    Shared by head-only, attention-only, FFN-only, layer ablation variants.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


# ---------------------------------------------------------------------------
# Group D: Reward / Data Quality — loss is baseline; data transforms external
# ---------------------------------------------------------------------------


def make_loss_reward_quality(ctx: LossContext) -> LossFn:
    """Baseline loss; reward filtering and normalisation are handled externally.

    Shared by reward_filtering, running_stats, action_diversity, reward_model.

    Args:
        ctx: Shared loss context.

    Returns:
        ``LossFn`` identical to baseline.
    """
    return make_loss_baseline(ctx)


# ---------------------------------------------------------------------------
# Fisher diagonal estimation (for EWC)
# ---------------------------------------------------------------------------


def estimate_fisher_diagonal(
    apply_fn: Callable,
    ref_params: Any,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    num_actions: int,
    batches: list[tuple],
    sigma_t: float = 0.0,
) -> Any:
    """Estimate the Fisher information diagonal on a set of held-out batches.

    F_i ≈ E[(d log p_theta(x|obs, z_t, t) / d theta_i)^2]

    Averaged over ``batches``, which are (acts, obs, valid) tuples.
    Returns a pytree of the same structure as ref_params.

    Args:
        apply_fn:          Training apply fn.
        ref_params:        Pretrained parameters (evaluation point).
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Size of the action vocabulary.
        batches:           List of (acts, obs, valid) tuples.
        sigma_t:           Remasking correction.

    Returns:
        Fisher diagonal pytree (same structure as ref_params).
    """
    ctx = LossContext(
        apply_fn=apply_fn,
        ref_params=ref_params,
        schedule_fn=schedule_fn,
        schedule_deriv_fn=schedule_deriv_fn,
        num_actions=num_actions,
        config={"TRAIN_SIGMA": sigma_t, "LABEL_SMOOTHING": 0.0},
    )
    bc_loss_fn = make_loss_baseline(ctx)

    accumulator = jax.tree.map(jnp.zeros_like, ref_params)
    rng = jax.random.PRNGKey(0)

    for acts, obs, valid in batches:
        rng, step_rng = jax.random.split(rng)
        ones = jnp.ones(acts.shape[0])
        grad = jax.grad(lambda p: bc_loss_fn(p, acts, obs, valid, step_rng, ones))(ref_params)
        accumulator = jax.tree.map(lambda acc, g: acc + g ** 2, accumulator, grad)

    n = max(len(batches), 1)
    return jax.tree.map(lambda acc: acc / n, accumulator)
