"""Core discrete diffusion logic for ReMDM (Remasking Discrete Diffusion Model).

Notation
--------
alpha_t : probability that a token remains unmasked at time t.
          alpha_0 = 1 (fully clean), alpha_1 = 0 (fully masked).
z_t     : noisy action sequence at time t (contains real tokens and MASK tokens).
x_0     : clean action sequence (ground truth).
MASK    : special token with id = num_actions.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ScheduleFn = Callable[[jnp.ndarray], jnp.ndarray]
"""Maps a diffusion time array to alpha_t (retention probability)."""

ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]
"""fn(params, obs, z_t, t, rng=None) -> logits [batch, H, num_actions].

``rng`` is a JAX PRNG key used as the dropout RNG during training.
Pass ``None`` (or omit) for deterministic inference.
"""


# =============================================================================
# Noise Schedules
# =============================================================================


def cosine_schedule(t: jnp.ndarray) -> jnp.ndarray:
    """Cosine noise schedule. Returns alpha_t in [0, 1]."""
    return jnp.cos(t * jnp.pi / 2.0)


def linear_schedule(t: jnp.ndarray) -> jnp.ndarray:
    """Linear noise schedule. Returns alpha_t in [0, 1]."""
    return 1.0 - t


# =============================================================================
# Forward Process (Masking)
# =============================================================================


def forward_process(
    rng: chex.PRNGKey,
    x_0: jnp.ndarray,
    alpha_t: jnp.ndarray,
    mask_token_id: int,
) -> jnp.ndarray:
    """Sample z_t ~ q(z_t | x_0) by independently masking each token.

    Each token stays as x_0[i] with prob alpha_t, becomes MASK with prob 1-alpha_t.

    Args:
        rng:            PRNG key.
        x_0:            [batch, H] int32, clean actions in [0, num_actions).
        alpha_t:        [batch] or scalar, retention probability.
        mask_token_id:  int, the MASK token index (= num_actions).

    Returns:
        z_t: [batch, H] int32.
    """
    keep_probs = jax.random.uniform(rng, shape=x_0.shape)
    if alpha_t.ndim >= 1:
        alpha_t = alpha_t[:, None]
    mask_val = jnp.array(mask_token_id, dtype=x_0.dtype)
    return jnp.where(keep_probs < alpha_t, x_0, mask_val)


# =============================================================================
# Training Loss (MDLM SUBS parameterization)
# =============================================================================

_MAX_LOSS_WEIGHT: float = 1000.0


def compute_loss(
    model_apply: ModelApplyFn,
    params: Any,
    rng: chex.PRNGKey,
    x_0: jnp.ndarray,
    obs: jnp.ndarray,
    num_actions: int,
    schedule_fn: ScheduleFn,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute the MDLM SUBS training loss.

    1. Sample t ~ Uniform[eps, 1.0] per batch element.
    2. Compute alpha_t, mask tokens to get z_t.
    3. Predict logits from model.
    4. Weighted cross-entropy on masked positions only.

    The weight is -alpha'(t) / (1 - alpha_t), approximated via finite differences.

    Args:
        model_apply:    fn(params, obs, z_t, t) -> logits [batch, H, num_actions].
        params:         Model parameters.
        rng:            PRNG key.
        x_0:            [batch, H] int32, clean action sequences.
        obs:            [batch, obs_dim] float32, observations.
        num_actions:    int, number of real actions (MASK id = num_actions).
        schedule_fn:    cosine_schedule or linear_schedule.

    Returns:
        (scalar_loss, info_dict)
    """
    batch_size = x_0.shape[0]
    mask_token_id = num_actions
    eps = 1e-5
    dt = 1e-3

    rng, t_rng, mask_rng, dropout_rng = jax.random.split(rng, 4)

    t = jax.random.uniform(t_rng, shape=(batch_size,), minval=eps, maxval=1.0)
    alpha_t = schedule_fn(t)

    alpha_t_plus = schedule_fn(jnp.minimum(t + dt, 1.0))
    neg_alpha_dot = (alpha_t - alpha_t_plus) / dt
    weight = neg_alpha_dot / jnp.maximum(1.0 - alpha_t, eps)
    weight = jnp.minimum(weight, _MAX_LOSS_WEIGHT)

    z_t = forward_process(mask_rng, x_0, alpha_t, mask_token_id)

    logits = model_apply(params, obs, z_t, t, dropout_rng)  # [batch, H, num_actions]

    is_masked = (z_t == mask_token_id).astype(jnp.float32)  # [batch, H]
    targets_one_hot = jax.nn.one_hot(x_0, num_actions)  # [batch, H, num_actions]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(targets_one_hot * log_probs, axis=-1)  # [batch, H]

    masked_ce = ce * is_masked
    num_masked = jnp.maximum(is_masked.sum(axis=-1), 1.0)
    per_sample_loss = weight * (masked_ce.sum(axis=-1) / num_masked)
    loss = jnp.mean(per_sample_loss)

    info: Dict[str, jnp.ndarray] = {
        "loss": loss,
        "mean_t": jnp.mean(t),
        "frac_masked": jnp.mean(is_masked),
    }
    return loss, info


# =============================================================================
# Remasking Strategies
# =============================================================================


def _sigma_max(alpha_t: jnp.ndarray, alpha_s: jnp.ndarray) -> jnp.ndarray:
    """Maximum allowable remasking probability."""
    return jnp.minimum(1.0, (1.0 - alpha_s) / jnp.maximum(alpha_t, 1e-8))


def remask_rescale(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """sigma_t = eta * sigma_max. Simple proportional scaling."""
    return eta * _sigma_max(alpha_t, alpha_s)


def remask_cap(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """sigma_t = min(eta, sigma_max). Capped at a fixed rate."""
    return jnp.minimum(eta, _sigma_max(alpha_t, alpha_s))


def remask_conf(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
    logits: jnp.ndarray,
) -> jnp.ndarray:
    """Per-token remasking based on model confidence.

    Low-confidence predictions are remasked more aggressively.

    Args:
        alpha_t: Scalar or [batch].
        alpha_s: Scalar or [batch].
        eta:     Remasking strength.
        logits:  [batch, H, num_actions].

    Returns:
        sigma: [batch, H] per-token remasking probabilities.
    """
    s_max = _sigma_max(alpha_t, alpha_s)
    probs = jax.nn.softmax(logits, axis=-1)
    confidence = jnp.max(probs, axis=-1)  # [batch, H]
    if s_max.ndim >= 1:
        s_max = s_max[:, None]
    return eta * s_max * (1.0 - confidence)


def remask_loop(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
    t: jnp.ndarray,
    t_on: float = 0.7,
    t_off: float = 0.3,
) -> jnp.ndarray:
    """Remasking active only in time window [t_off, t_on] (reverse time).

    Returns:
        sigma: scalar or broadcastable.
    """
    in_window = jnp.logical_and(t >= t_off, t <= t_on)
    sigma = remask_rescale(alpha_t, alpha_s, eta)
    return jnp.where(in_window, sigma, 0.0)


# =============================================================================
# Reverse Sampling (Planning)
# =============================================================================

STRATEGY_RESCALE: int = 0
STRATEGY_CAP: int = 1
STRATEGY_CONF: int = 2
STRATEGY_LOOP: int = 3

STRATEGY_MAP: Dict[str, int] = {
    "rescale": STRATEGY_RESCALE,
    "cap": STRATEGY_CAP,
    "conf": STRATEGY_CONF,
    "loop": STRATEGY_LOOP,
}


def sample_plan(
    model_apply: ModelApplyFn,
    params: Any,
    rng: chex.PRNGKey,
    obs: jnp.ndarray,
    num_actions: int,
    plan_horizon: int,
    num_steps: int,
    schedule_fn: ScheduleFn,
    remask_strategy: str,
    eta: float = 0.5,
    t_on: float = 0.7,
    t_off: float = 0.3,
) -> jnp.ndarray:
    """Generate an action plan via reverse diffusion with ReMDM remasking.

    Args:
        model_apply:     fn(params, obs, z, t) -> logits [batch, H, num_actions].
        params:          Model parameters.
        rng:             PRNG key.
        obs:             [batch, obs_dim] float32.
        num_actions:     Number of real actions (MASK token id = num_actions).
        plan_horizon:    Length of action sequence H.
        num_steps:       T, number of denoising steps.
        schedule_fn:     Noise schedule function.
        remask_strategy: One of "rescale", "cap", "conf", "loop".
        eta:             Remasking strength.
        t_on:            Loop strategy: upper time bound for remasking window.
        t_off:           Loop strategy: lower time bound for remasking window.

    Returns:
        actions: [batch, plan_horizon] int32, predicted action sequence.
    """
    batch_size = obs.shape[0]
    mask_token_id = num_actions
    mask_val = jnp.array(mask_token_id, dtype=jnp.int32)
    strategy_idx = STRATEGY_MAP[remask_strategy]

    z = jnp.full((batch_size, plan_horizon), mask_token_id, dtype=jnp.int32)

    def _denoise_step(
        carry: Tuple[jnp.ndarray, chex.PRNGKey],
        step_idx: jnp.ndarray,
    ) -> Tuple[Tuple[jnp.ndarray, chex.PRNGKey], None]:
        z, rng = carry
        rng, unmask_rng, remask_rng = jax.random.split(rng, 3)

        # step_idx=0 → t = (T-1)/T (high noise), step_idx=T-1 → t = 0/T (clean)
        t = (num_steps - 1 - step_idx) / num_steps
        s = jnp.maximum(t - 1.0 / num_steps, 0.0)

        alpha_t = schedule_fn(t)
        alpha_s = schedule_fn(s)

        t_input = jnp.full((batch_size,), t)
        logits = model_apply(params, obs, z, t_input)
        x_hat = jnp.argmax(logits, axis=-1)  # [batch, H]

        is_masked = z == mask_token_id
        is_unmasked = ~is_masked

        # Unmask: for masked positions
        p_unmask = (alpha_s - alpha_t) / jnp.maximum(1.0 - alpha_t, 1e-8)
        p_unmask = jnp.clip(p_unmask, 0.0, 1.0)
        unmask_draw = jax.random.uniform(unmask_rng, shape=z.shape)
        do_unmask = is_masked & (unmask_draw < p_unmask)

        # Remask: for unmasked positions
        sigma_rescale = remask_rescale(alpha_t, alpha_s, eta)
        sigma_cap = remask_cap(alpha_t, alpha_s, eta)
        sigma_conf = remask_conf(alpha_t, alpha_s, eta, logits)
        sigma_loop = remask_loop(alpha_t, alpha_s, eta, t, t_on, t_off)

        sigma_rescale_bh = jnp.broadcast_to(sigma_rescale, z.shape)
        sigma_cap_bh = jnp.broadcast_to(sigma_cap, z.shape)
        sigma_loop_bh = jnp.broadcast_to(sigma_loop, z.shape)

        all_sigmas = jnp.stack(
            [sigma_rescale_bh, sigma_cap_bh, sigma_conf, sigma_loop_bh], axis=0
        )
        sigma = all_sigmas[strategy_idx]  # [batch, H]

        remask_draw = jax.random.uniform(remask_rng, shape=z.shape)
        do_remask = is_unmasked & (remask_draw < sigma)

        z_new = jnp.where(do_unmask, x_hat, z)
        z_new = jnp.where(do_remask, mask_val, z_new)

        return (z_new, rng), None

    (z_final, _), _ = jax.lax.scan(
        _denoise_step, (z, rng), jnp.arange(num_steps)
    )

    t_final = jnp.zeros((batch_size,))
    final_logits = model_apply(params, obs, z_final, t_final)
    x_final = jnp.argmax(final_logits, axis=-1)
    z_final = jnp.where(z_final == mask_token_id, x_final, z_final)

    return z_final
