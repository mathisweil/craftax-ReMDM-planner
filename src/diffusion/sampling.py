"""Reverse diffusion sampling with ReMDM remasking (Wang et al.).

Strategies (Section 4.1):
  rescale: sigma = eta * sigma_max
  cap:     sigma = min(eta, sigma_max)
  conf:    per-token confidence-based remasking

Loop mode (Section 4.2, Algorithm 3):
  Phase 1: standard MDLM decode, t in [1, t_on]
  Phase 2: constant alpha(t_on), remasking active
  Phase 3: standard MDLM decode, t in [t_off, 0]
"""

from __future__ import annotations
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .schedules import ScheduleFn

ModelApplyFn = Callable[
    [Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[Any]], jnp.ndarray
]


# ---------------------------------------------------------------------------
# Remasking sigma computation
# ---------------------------------------------------------------------------

def _sigma_max(alpha_t: jnp.ndarray, alpha_s: jnp.ndarray) -> jnp.ndarray:
    """sigma_max = min(1, (1 - alpha_s) / alpha_t).  [Eq. 7]"""
    return jnp.minimum(1.0, (1.0 - alpha_s) / jnp.maximum(alpha_t, 1e-8))


def sigma_rescale(alpha_t, alpha_s, eta):
    return eta * _sigma_max(alpha_t, alpha_s)


def sigma_cap(alpha_t, alpha_s, eta):
    return jnp.minimum(eta, _sigma_max(alpha_t, alpha_s))


def sigma_conf(alpha_t, alpha_s, eta, psi, is_unmasked):
    """Per-token confidence remasking.  Safe against all-masked rows."""
    base = eta * _sigma_max(alpha_t, alpha_s)
    any_unmasked = jnp.any(is_unmasked, axis=-1, keepdims=True)
    neg_psi = jnp.where(is_unmasked, -psi, -jnp.inf)
    safe_neg_psi = jnp.where(any_unmasked, neg_psi, 0.0)
    eta_conf = jax.nn.softmax(safe_neg_psi, axis=-1)
    return jnp.where(is_unmasked, eta_conf * base, 0.0)


_SIGMA_FNS = {
    "rescale": lambda a_t, a_s, eta, *_: sigma_rescale(a_t, a_s, eta),
    "cap": lambda a_t, a_s, eta, *_: sigma_cap(a_t, a_s, eta),
    "conf": lambda a_t, a_s, eta, psi, unm: sigma_conf(a_t, a_s, eta, psi, unm),
}


# ---------------------------------------------------------------------------
# Decoding helpers
# ---------------------------------------------------------------------------

def _nucleus_sample(rng, logits, top_p):
    """Top-p sampling from [B, H, V] logits -> [B, H] int32."""
    probs = jax.nn.softmax(logits, axis=-1)
    idx = jnp.argsort(-probs, axis=-1)
    sorted_p = jnp.take_along_axis(probs, idx, axis=-1)
    cum = jnp.cumsum(sorted_p, axis=-1)

    cutoff = cum - sorted_p
    sorted_p = jnp.where(cutoff >= top_p, 0.0, sorted_p)
    sorted_p = sorted_p / jnp.maximum(sorted_p.sum(axis=-1, keepdims=True), 1e-12)

    B, H, V = logits.shape
    flat = sorted_p.reshape(B * H, V)
    tokens = jax.random.categorical(rng, jnp.log(flat + 1e-12)).reshape(B, H)
    return jnp.take_along_axis(idx, tokens[..., None], axis=-1).squeeze(-1)


def _decode(rng, logits, temperature, top_p):
    """Sample tokens from logits.  Argmax only when temperature <= 0."""
    if top_p is not None:
        return _nucleus_sample(rng, logits / jnp.maximum(temperature, 1e-8), top_p)
    if temperature > 1e-8:
        B, H, V = logits.shape
        scaled = logits / temperature
        return jax.random.categorical(rng, scaled.reshape(-1, V)).reshape(B, H)
    return jnp.argmax(logits, axis=-1)


# ---------------------------------------------------------------------------
# Reverse sampling
# ---------------------------------------------------------------------------

def sample_plan(
    model_apply: ModelApplyFn,
    params: Any,
    rng: jax.Array,
    obs: jnp.ndarray,
    num_actions: int,
    plan_horizon: int,
    num_steps: int,
    schedule_fn: ScheduleFn,
    remask_strategy: str = "cap",
    eta: float = 0.5,
    use_loop: bool = False,
    t_on: float = 0.55,
    t_off: float = 0.05,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
) -> jnp.ndarray:
    """Generate an action plan via reverse diffusion with ReMDM remasking.

    Returns:
        actions: [B, H] int32.
    """
    B = obs.shape[0]
    mask_id = num_actions
    mask_val = jnp.array(mask_id, dtype=jnp.int32)

    if remask_strategy not in _SIGMA_FNS:
        raise ValueError(f"Unknown strategy {remask_strategy!r}. Options: {list(_SIGMA_FNS)}")
    get_sigma = _SIGMA_FNS[remask_strategy]

    # Phase allocation for loop mode
    if use_loop:
        f1, f3 = 1.0 - t_on, t_off
        denom = f1 + f3 + (t_on - t_off)
        n1 = max(int(round(num_steps * f1 / denom)), 1)
        n3 = max(int(round(num_steps * f3 / denom)), 1)
        n2 = max(num_steps - n1 - n3, 1)
    else:
        n1, n2, n3 = num_steps, 0, 0

    alpha_loop = schedule_fn(jnp.array(t_on))

    z_init = jnp.full((B, plan_horizon), mask_id, dtype=jnp.int32)
    psi_init = jnp.full((B, plan_horizon), jnp.inf)

    # ------------------------------------------------------------------
    # Core denoising step (ReMDM Eq. 6)
    # ------------------------------------------------------------------
    def _step(carry, _unused, t_val, alpha_t, alpha_s, sigma_on):
        z, rng, psi = carry
        rng, s_rng, u_rng, r_rng = jax.random.split(rng, 4)

        t_inp = jnp.full((B,), t_val)
        logits = model_apply(params, obs, z, t_inp, None)
        x_hat = _decode(s_rng, logits, temperature, top_p)

        is_masked = z == mask_id
        is_unmasked = ~is_masked

        sigma = get_sigma(alpha_t, alpha_s, eta, psi, is_unmasked)
        sigma = jnp.broadcast_to(sigma, z.shape)
        sigma = jnp.where(sigma_on, sigma, 0.0)

        # Masked -> unmask probability
        denom = jnp.maximum(1.0 - alpha_t, 1e-8)
        p_unmask = jnp.clip((alpha_s - (1.0 - sigma) * alpha_t) / denom, 0.0, 1.0)

        do_unmask = is_masked & (jax.random.uniform(u_rng, z.shape) < p_unmask)
        do_remask = is_unmasked & (jax.random.uniform(r_rng, z.shape) < sigma)

        z_new = jnp.where(do_unmask, x_hat, z)
        z_new = jnp.where(do_remask, mask_val, z_new)

        # Update confidence history
        probs = jax.nn.softmax(logits, axis=-1)
        decode_prob = jnp.take_along_axis(probs, x_hat[..., None], axis=-1).squeeze(-1)
        psi_new = jnp.where(do_unmask, decode_prob, psi)
        psi_new = jnp.where(do_remask, jnp.inf, psi_new)

        return (z_new, rng, psi_new), None

    # ------------------------------------------------------------------
    # Phase functions
    # ------------------------------------------------------------------
    def _phase1_step(carry, idx):
        t = 1.0 - idx * (1.0 - t_on) / n1
        s = jnp.maximum(1.0 - (idx + 1) * (1.0 - t_on) / n1, t_on)
        return _step(carry, idx, t, schedule_fn(t), schedule_fn(s), False)

    def _phase2_step(carry, idx):
        return _step(carry, idx, t_on, alpha_loop, alpha_loop, True)

    def _phase3_step(carry, idx):
        t = t_off - idx * t_off / n3
        s = jnp.maximum(t_off - (idx + 1) * t_off / n3, 0.0)
        return _step(carry, idx, t, schedule_fn(t), schedule_fn(s), False)

    def _simple_step(carry, idx):
        t = (num_steps - idx) / num_steps
        s = jnp.maximum((num_steps - idx - 1) / num_steps, 0.0)
        return _step(carry, idx, t, schedule_fn(t), schedule_fn(s), True)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    carry = (z_init, rng, psi_init)

    if use_loop:
        carry, _ = jax.lax.scan(_phase1_step, carry, jnp.arange(n1))
        carry, _ = jax.lax.scan(_phase2_step, carry, jnp.arange(n2))
        if n3 > 0:
            carry, _ = jax.lax.scan(_phase3_step, carry, jnp.arange(n3))
    else:
        carry, _ = jax.lax.scan(_simple_step, carry, jnp.arange(num_steps))

    z_final = carry[0]

    # Final greedy cleanup for any remaining masks
    final_logits = model_apply(params, obs, z_final, jnp.zeros((B,)), None)
    fallback = jnp.argmax(final_logits, axis=-1)
    return jnp.where(z_final == mask_id, fallback, z_final)
