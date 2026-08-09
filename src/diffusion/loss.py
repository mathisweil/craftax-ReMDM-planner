"""MDLM ELBO loss for masked discrete diffusion training.

Shared pseudocode lines 2-6 (METHOD_PARITY 2.1); the minihack twin is
src/diffusion/loss.py:mdlm_loss.
"""

from __future__ import annotations
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .forward import forward_process
from .schedules import ScheduleFn

ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]

_MAX_WEIGHT: float = 1000.0
_EPS: float = 1e-5


def compute_loss(
    model_apply: ModelApplyFn,
    params: Any,
    rng: jax.Array,
    x_0: jnp.ndarray,
    obs: jnp.ndarray,
    valid: jnp.ndarray,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    sigma_t: float = 0.0,
    label_smoothing: float = 0.0,
    advantages: Optional[jnp.ndarray] = None,
    t_min: float | jax.Array = _EPS,
    t_max: float | jax.Array = 1.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Continuous-time ELBO loss on masked positions only.

    Args:
        model_apply:       fn(params, obs, z_t, t, rng) -> logits [B, H, V].
        params:            Model parameters.
        rng:               PRNG key.
        x_0:               [B, H] int32, ground-truth actions.
        obs:               [B, obs_dim] float32, observations.
        valid:             [B] bool/float, whether each sample is valid.
        num_actions:       Size of real action vocabulary.
        schedule_fn:       alpha(t).
        schedule_deriv_fn: d(alpha)/dt (analytic).
        sigma_t:           Remasking correction for ELBO weight (0 = standard MDLM).
        label_smoothing:   Smoothing epsilon (0 = exact ELBO targets).
        advantages:        Optional [B] per-sample weights.
        t_min:             Lower bound for uniform t sampling (default: _EPS).
        t_max:             Upper bound for uniform t sampling (default: 1.0).

    Returns:
        (loss, info_dict).
    """
    B = x_0.shape[0]
    mask_id = num_actions
    rng, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)

    # Sample t ~ U(t_min, t_max). Defaults give full ELBO; narrow range for ablations.
    t = jax.random.uniform(t_rng, (B,), minval=t_min, maxval=t_max)
    alpha_t = schedule_fn(t)

    # Analytic loss weight: w(t) = (1 - sigma) * (-d(alpha)/dt) / (1 - alpha(t))
    neg_alpha_dot = -schedule_deriv_fn(t)  # positive quantity
    weight = (1.0 - sigma_t) * neg_alpha_dot / jnp.maximum(1.0 - alpha_t, _EPS)
    weight = jnp.minimum(weight, _MAX_WEIGHT)

    # Forward noise
    z_t = forward_process(mask_rng, x_0, alpha_t, mask_id)

    # Model prediction
    logits = model_apply(params, obs, z_t, t, drop_rng)  # [B, H, V]

    # Cross-entropy on valid masked positions
    is_masked = (z_t == mask_id).astype(jnp.float32)        # [B, H]
    valid_masked = is_masked * valid[:, None].astype(jnp.float32)  # [B, H]

    targets = jax.nn.one_hot(x_0, num_actions)
    if label_smoothing > 0:
        targets = (1.0 - label_smoothing) * targets + label_smoothing / num_actions

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(targets * log_probs, axis=-1)             # [B, H]

    n_masked = jnp.maximum(valid_masked.sum(axis=-1), 1.0)  # [B]
    per_sample = weight * (ce * valid_masked).sum(axis=-1) / n_masked

    if advantages is not None:
        per_sample = per_sample * jax.lax.stop_gradient(advantages)

    loss = jnp.mean(per_sample)

    # Diagnostics
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == x_0).astype(jnp.float32)
    acc = jnp.sum(correct * valid_masked) / jnp.maximum(valid_masked.sum(), 1.0)

    t_bins = jnp.array([0.33, 0.66])
    t_lo = (t < t_bins[0])[:, None]
    t_mi = ((t >= t_bins[0]) & (t <= t_bins[1]))[:, None]
    t_hi = (t > t_bins[1])[:, None]

    def _binned_acc(mask):
        m = valid_masked * mask
        return jnp.sum(correct * m) / jnp.maximum(m.sum(), 1.0)

    info = {
        "loss": loss,
        "unweighted_loss": jnp.mean((ce * valid_masked).sum(axis=-1) / n_masked),
        "mean_t": jnp.mean(t),
        "frac_masked": jnp.mean(is_masked),
        "accuracy": acc,
        "acc_t_low": _binned_acc(t_lo),
        "acc_t_mid": _binned_acc(t_mi),
        "acc_t_high": _binned_acc(t_hi),
    }
    if advantages is not None:
        info["adv_mean"] = jnp.mean(advantages)
        info["adv_std"] = jnp.std(advantages)
    return loss, info
