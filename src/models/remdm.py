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

from typing import Any, Callable, Optional

import chex
import jax
import jax.numpy as jnp
import optax

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
    """Cosine noise schedule.  Returns alpha_t in [0, 1].

    alpha(t) = cos(pi * t / 2).  Used in D3PM-uniform and some MDLM configs.
    """
    return jnp.cos(t * jnp.pi / 2.0)


def linear_schedule(t: jnp.ndarray) -> jnp.ndarray:
    """Log-linear noise schedule.  Returns alpha_t in [0, 1].

    alpha(t) = 1 - t.  This is the default schedule used in MDLM / ReMDM
    experiments (OWT, QM9, ImageNet).
    """
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

    q(z_t | x_0) = Cat(z_t; alpha_t * x_0 + (1 - alpha_t) * m)   [ReMDM Eq. 1]

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
    alpha_t = jnp.reshape(alpha_t, (-1, 1))
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
        valid_batch: jnp.ndarray,  # <-- ADDED
        num_actions: int,
        schedule_fn: ScheduleFn,
        sigma_t: float = 0.0,
        advantages=None
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    batch_size = x_0.shape[0]
    mask_token_id = num_actions
    eps = 1e-5
    dt = 1e-3

    rng, t_rng, mask_rng, dropout_rng = jax.random.split(rng, 4)

    # --- sample time --------------------------------------------------------
    t = jax.random.uniform(t_rng, shape=(batch_size,), minval=eps, maxval=1.0)
    alpha_t = schedule_fn(t)

    # --- loss weight --------------------------------------------------------
    alpha_t_plus = schedule_fn(jnp.minimum(t + dt, 1.0))
    neg_alpha_dot = (alpha_t - alpha_t_plus) / dt
    weight = (1.0 - sigma_t) * neg_alpha_dot / jnp.maximum(1.0 - alpha_t, eps)
    weight = jnp.minimum(weight, _MAX_LOSS_WEIGHT)

    # --- forward noise ------------------------------------------------------
    z_t = forward_process(mask_rng, x_0, alpha_t, mask_token_id)

    # --- model prediction ---------------------------------------------------
    logits = model_apply(params, obs, z_t, t, dropout_rng)  # [B, H, num_actions]

    # --- cross-entropy on masked positions ----------------------------------
    is_masked = (z_t == mask_token_id).astype(jnp.float32)  # [B, H]
    valid_mask = valid_batch[:, None].astype(jnp.float32)  # [B, H]

    # Only penalize tokens that are both masked AND part of a valid trajectory
    valid_and_masked = is_masked * valid_mask  # [B, H]

    raw_one_hot = jax.nn.one_hot(x_0, num_actions)
    targets_one_hot = optax.smooth_labels(raw_one_hot, 0.05)

    log_probs = jax.nn.log_softmax(logits, axis=-1)
    ce = -jnp.sum(targets_one_hot * log_probs, axis=-1)

    masked_ce = ce * valid_and_masked
    num_valid_masked = jnp.maximum(valid_and_masked.sum(axis=-1), 1.0)  # [B]

    unweighted_loss = masked_ce.sum(axis=-1) / num_valid_masked
    per_sample_loss = weight * unweighted_loss

    # --- GRPO ADVANTAGE WEIGHTING ---
    if advantages is not None:
        adv_weights = jax.lax.stop_gradient(advantages)
        per_sample_loss = per_sample_loss * adv_weights

    loss = jnp.mean(per_sample_loss)

    # --- ACCURACY MATH ---
    predicted_actions = jnp.argmax(logits, axis=-1)
    correct_guesses = (predicted_actions == x_0)

    # Calculate accuracy ONLY on the valid, masked tokens
    masked_accuracy = jnp.sum(correct_guesses * valid_and_masked) / jnp.maximum(jnp.sum(valid_and_masked), 1.0)

    t_low = (t < 0.33)[:, None]  # Broadcast to [B, 1] to align with [B, H]
    t_mid = ((t >= 0.33) & (t <= 0.66))[:, None]
    t_high = (t > 0.66)[:, None]

    acc_low = jnp.sum(correct_guesses * valid_and_masked * t_low) / jnp.maximum(jnp.sum(valid_and_masked * t_low), 1.0)
    acc_mid = jnp.sum(correct_guesses * valid_and_masked * t_mid) / jnp.maximum(jnp.sum(valid_and_masked * t_mid), 1.0)
    acc_high = jnp.sum(correct_guesses * valid_and_masked * t_high) / jnp.maximum(jnp.sum(valid_and_masked * t_high),
                                                                                  1.0)

    info: dict[str, jnp.ndarray] = {
        "loss": loss,
        "unweighted_loss": jnp.mean(unweighted_loss),
        "mean_t": jnp.mean(t),
        "frac_masked": jnp.mean(is_masked),
        "accuracy": masked_accuracy,
        "acc_t_low": acc_low,
        "acc_t_mid": acc_mid,
        "acc_t_high": acc_high,
    }

    if advantages is not None:
        info["adv_mean"] = jnp.mean(advantages)
        info["adv_std"] = jnp.std(advantages)
    return loss, info


# =============================================================================
# Remasking Schedule Helpers
# =============================================================================


def _sigma_max(alpha_t: jnp.ndarray, alpha_s: jnp.ndarray) -> jnp.ndarray:
    """Maximum allowable remasking probability.  [ReMDM Eq. 7]

    sigma_max = min(1, (1 - alpha_s) / alpha_t)
    """
    return jnp.minimum(1.0, (1.0 - alpha_s) / jnp.maximum(alpha_t, 1e-8))


def remask_rescale(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """ReMDM-rescale: sigma_t = eta * sigma_max.  [Section 4.1]"""
    return eta * _sigma_max(alpha_t, alpha_s)


def remask_cap(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
) -> jnp.ndarray:
    """ReMDM-cap: sigma_t = min(eta, sigma_max).  [Section 4.1]"""
    return jnp.minimum(eta, _sigma_max(alpha_t, alpha_s))


def compute_sigma_conf(
    alpha_t: jnp.ndarray,
    alpha_s: jnp.ndarray,
    eta: float,
    psi: jnp.ndarray,
    is_unmasked: jnp.ndarray,
) -> jnp.ndarray:
    """ReMDM-conf: per-token remasking based on historical decode confidence.

    Implements the confidence-based schedule from Section 4.1 of Wang et al.:

        eta_conf^(l) = softmax(-psi)^(l)       (across sequence positions)
        sigma_t^(l)  = eta_conf^(l) * base_sigma

    where psi^(l) is the probability the model assigned to token l at the
    time it was last unmasked (stored across steps).  Masked positions are
    assigned psi = +inf so they get ~zero remasking weight.

    Args:
        alpha_t:      Scalar, current noise level.
        alpha_s:      Scalar, target noise level.
        eta:          Remasking strength (used as rescale on sigma_max).
        psi:          [batch, H] float32, historical decode confidence scores.
                      +inf for currently-masked positions.
        is_unmasked:  [batch, H] bool, True where z_t != MASK.

    Returns:
        sigma: [batch, H] per-token remasking probabilities.
    """
    base_sigma = eta * _sigma_max(alpha_t, alpha_s)   # scalar

    # Softmax of -psi across positions.  Masked positions have psi=+inf,
    # so -psi=-inf, and they get zero weight after softmax.
    any_unmasked = jnp.any(is_unmasked, axis=-1, keepdims=True)  # [B, 1]
    neg_psi = jnp.where(is_unmasked, -psi, -jnp.inf)
    safe_neg_psi = jnp.where(any_unmasked, neg_psi, 0.0)
    eta_conf = jax.nn.softmax(safe_neg_psi, axis=-1)

    # Only apply to unmasked positions.
    return jnp.where(is_unmasked, eta_conf * base_sigma, 0.0)


# =============================================================================
# Sampling Utilities
# =============================================================================


def _nucleus_sample(
    rng: chex.PRNGKey,
    logits: jnp.ndarray,
    top_p: float = 0.9,
) -> jnp.ndarray:
    """Nucleus (top-p) sampling from logits.

    Args:
        rng:    PRNG key.
        logits: [batch, H, V] float32.
        top_p:  Cumulative probability threshold.

    Returns:
        tokens: [batch, H] int32.
    """
    # Sort descending by probability.
    probs = jax.nn.softmax(logits, axis=-1)                    # [B, H, V]
    sorted_indices = jnp.argsort(-probs, axis=-1)              # [B, H, V]
    sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = jnp.cumsum(sorted_probs, axis=-1)

    # Mask tokens beyond the top-p nucleus (keep at least one token).
    cutoff = cumulative - sorted_probs  # exclusive cumsum
    mask = cutoff >= top_p
    sorted_probs = jnp.where(mask, 0.0, sorted_probs)

    # Re-normalise and sample.
    sorted_probs = sorted_probs / jnp.maximum(
        sorted_probs.sum(axis=-1, keepdims=True), 1e-12
    )
    # Flatten for sampling, then reshape.
    B, H, V = logits.shape
    flat_probs = sorted_probs.reshape(B * H, V)
    flat_rng = jax.random.split(rng, B * H)
    flat_samples = jax.vmap(
        lambda k, p: jax.random.categorical(k, jnp.log(p + 1e-12))
    )(flat_rng, flat_probs)                                     # [B*H]
    sorted_choice = flat_samples.reshape(B, H)

    # Map back from sorted indices to original vocabulary ids.
    tokens = jnp.take_along_axis(
        sorted_indices, sorted_choice[..., None], axis=-1
    ).squeeze(-1)
    return tokens


# =============================================================================
# Reverse Sampling (Planning)
# =============================================================================

STRATEGY_RESCALE = "rescale"
STRATEGY_CAP = "cap"
STRATEGY_CONF = "conf"

STRATEGY_MAP: dict[str, str] = {
    "rescale": STRATEGY_RESCALE,
    "cap": STRATEGY_CAP,
    "conf": STRATEGY_CONF,
}
"""Registry of valid remasking strategy names for ``sample_plan``."""


def sample_plan(
    model_apply: ModelApplyFn,
    params: Any,
    rng: chex.PRNGKey,
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

    Implements Algorithm 1 (basic) and Algorithm 3 (loop) from Wang et al.

    Strategies for sigma_t  (Section 4.1):
        - ``"rescale"``: sigma = eta * sigma_max
        - ``"cap"``:     sigma = min(eta, sigma_max)
        - ``"conf"``:    confidence-based per-token remasking with historical
                         decode probabilities (psi), combined with rescale.

    Loop mode  (Section 4.2, Algorithm 3):
        When ``use_loop=True``, sampling is split into three phases:
          Phase 1 (t from 1 to t_on): standard MDLM decode, sigma=0.
          Phase 2 (t_on to t_off):    alpha held constant at alpha(t_on),
                                      remasking active — "fix mistakes" loop.
          Phase 3 (t_off to 0):       standard MDLM decode, sigma=0.
        Time is rescaled within each phase so the full alpha schedule is
        covered in Phases 1+3.

    Args:
        model_apply:     fn(params, obs, z, t, rng) -> logits [B, H, num_actions].
        params:          Model parameters.
        rng:             PRNG key.
        obs:             [batch, obs_dim] float32.
        num_actions:     Number of real actions (MASK token id = num_actions).
        plan_horizon:    Length of action sequence H.
        num_steps:       T, total number of denoising steps across all phases.
        schedule_fn:     Noise schedule function (e.g. linear_schedule).
        remask_strategy: One of "rescale", "cap", "conf".
        eta:             Remasking strength hyperparameter.
        use_loop:        If True, use the three-phase ReMDM-loop (Algo 3).
        t_on:            Loop: time at which the remasking loop starts.
        t_off:           Loop: time at which the remasking loop ends.
        temperature:     Softmax temperature for sampling (1.0 = no scaling).
        top_p:           If not None, apply nucleus sampling with this threshold.

    Returns:
        actions: [batch, plan_horizon] int32, predicted action sequence.
    """
    batch_size = obs.shape[0]
    mask_token_id = num_actions
    mask_val = jnp.array(mask_token_id, dtype=jnp.int32)

    # --- Phase allocation for loop mode (Algorithm 3) -----------------------
    # We pre-compute how many discrete steps go into each phase.
    if use_loop:
        # Phase 1 covers alpha schedule from 1 → t_on  (initial decode)
        # Phase 2 is the remasking loop at constant alpha(t_on)
        # Phase 3 covers alpha schedule from t_off → 0  (finish decode)
        #
        # We allocate steps proportionally:
        #   phase1 gets a share proportional to (1 - t_on)
        #   phase3 gets a share proportional to t_off
        #   phase2 gets the remaining steps
        frac_phase1 = 1.0 - t_on
        frac_phase3 = t_off
        frac_total = frac_phase1 + frac_phase3
        # Avoid division by zero when t_on ≈ 1 and t_off ≈ 0.
        frac_total = max(frac_total, 1e-6)
        n_phase1 = max(int(round(num_steps * frac_phase1 / (frac_total + (t_on - t_off)))), 1)
        n_phase3 = max(int(round(num_steps * frac_phase3 / (frac_total + (t_on - t_off)))), 1)
        n_phase2 = max(num_steps - n_phase1 - n_phase3, 1)
    else:
        n_phase1 = num_steps
        n_phase2 = 0
        n_phase3 = 0

    # Pre-compute alpha(t_on) for the loop phase.
    alpha_loop = schedule_fn(jnp.array(t_on))

    # -----------------------------------------------------------------------
    # Initial state: all masked, psi = +inf (no decode history).
    # -----------------------------------------------------------------------
    z_init = jnp.full((batch_size, plan_horizon), mask_token_id, dtype=jnp.int32)
    psi_init = jnp.full((batch_size, plan_horizon), jnp.inf, dtype=jnp.float32)

    # -----------------------------------------------------------------------
    # Helper: compute sigma for the chosen strategy
    # -----------------------------------------------------------------------
    def _get_sigma(alpha_t, alpha_s, logits, psi, is_unmasked):
        """Return [batch, H] sigma given the selected strategy."""
        if remask_strategy == STRATEGY_RESCALE:
            sigma = remask_rescale(alpha_t, alpha_s, eta)
            return jnp.broadcast_to(sigma, (batch_size, plan_horizon))
        elif remask_strategy == STRATEGY_CAP:
            sigma = remask_cap(alpha_t, alpha_s, eta)
            return jnp.broadcast_to(sigma, (batch_size, plan_horizon))
        elif remask_strategy == STRATEGY_CONF:
            return compute_sigma_conf(
                alpha_t, alpha_s, eta, psi, is_unmasked
            )
        else:
            raise ValueError(
                f"Unknown remask_strategy: {remask_strategy!r}. "
                f"Valid options: {list(STRATEGY_MAP)}"
            )

    # -----------------------------------------------------------------------
    # Helper: decode logits → token ids
    # -----------------------------------------------------------------------
    def _decode(rng_key, logits):
        """Sample or argmax from logits, with optional temperature / top-p."""
        if top_p is not None:
            scaled = logits / jnp.maximum(temperature, 1e-8)
            return _nucleus_sample(rng_key, scaled, top_p=top_p)
        elif temperature > 1e-8:  # <-- changed condition
            scaled = logits / temperature
            B, H, V = scaled.shape
            flat = scaled.reshape(B * H, V)
            flat_rng = jax.random.split(rng_key, B * H)
            flat_tok = jax.vmap(
                lambda k, l: jax.random.categorical(k, l)
            )(flat_rng, flat)
            return flat_tok.reshape(B, H)
        else:
            return jnp.argmax(logits, axis=-1)

    # -----------------------------------------------------------------------
    # Core step: implements ReMDM Eq. 6 (full posterior)
    # -----------------------------------------------------------------------
    def _remdm_step(carry, step_idx, t_val, alpha_t, alpha_s, sigma_active):
        """One denoising step implementing the ReMDM posterior (Eq. 6).

        Args:
            carry:        (z, rng, psi)
            step_idx:     int (unused inside, just for scan compatibility)
            t_val:        scalar, diffusion time to pass to the model
            alpha_t:      scalar, alpha(t) — current noise level
            alpha_s:      scalar, alpha(s) — target noise level
            sigma_active: bool, whether remasking is active this step

        Returns:
            ((z_new, rng_new, psi_new), None)
        """
        z, rng, psi = carry
        rng, sample_rng, unmask_rng, remask_rng = jax.random.split(rng, 4)

        # --- Model prediction ------------------------------------------------
        t_input = jnp.full((batch_size,), t_val)
        logits = model_apply(params, obs, z, t_input, None)    # [B, H, V]
        x_hat = _decode(sample_rng, logits)                     # [B, H]

        is_masked = z == mask_token_id                           # [B, H]
        is_unmasked = ~is_masked

        # --- Compute sigma (only meaningful when sigma_active) ---------------
        sigma = _get_sigma(alpha_t, alpha_s, logits, psi, is_unmasked)
        # Zero out sigma when remasking is not active.
        sigma = jnp.where(sigma_active, sigma, 0.0)

        # --- ReMDM posterior (Eq. 6) -----------------------------------------
        #
        # Case z_t = m (masked):
        #   p(z_s = x | z_t = m) = (alpha_s - (1 - sigma)*alpha_t) / (1 - alpha_t)
        #   p(z_s = m | z_t = m) = (1 - alpha_s - sigma*alpha_t) / (1 - alpha_t)
        #
        # Case z_t ≠ m (unmasked):
        #   p(z_s = x | z_t = x) = 1 - sigma
        #   p(z_s = m | z_t = x) = sigma
        #
        # For masked positions: probability of unmasking.
        denom = jnp.maximum(1.0 - alpha_t, 1e-8)
        p_unmask = (alpha_s - (1.0 - sigma) * alpha_t) / denom  # [B,H] or scalar
        p_unmask = jnp.clip(p_unmask, 0.0, 1.0)

        unmask_draw = jax.random.uniform(unmask_rng, shape=z.shape)
        do_unmask = is_masked & (unmask_draw < p_unmask)

        # For unmasked positions: probability of remasking.
        remask_draw = jax.random.uniform(remask_rng, shape=z.shape)
        do_remask = is_unmasked & (remask_draw < sigma)

        # --- Apply transitions -----------------------------------------------
        z_new = jnp.where(do_unmask, x_hat, z)
        z_new = jnp.where(do_remask, mask_val, z_new)

        # --- Update confidence history (psi) ---------------------------------
        # For newly unmasked tokens, store the model's probability for that token.
        probs = jax.nn.softmax(logits, axis=-1)                 # [B, H, V]
        decode_prob = jnp.take_along_axis(
            probs, x_hat[..., None], axis=-1
        ).squeeze(-1)                                            # [B, H]

        newly_unmasked = do_unmask                               # was masked, now decoded
        psi_new = jnp.where(newly_unmasked, decode_prob, psi)
        # Remasked positions reset to +inf.
        psi_new = jnp.where(do_remask, jnp.inf, psi_new)

        return (z_new, rng, psi_new), None

    # =====================================================================
    # Phase 1: Standard MDLM decode (sigma = 0), t from 1 → t_on
    # =====================================================================
    def _phase1_step(carry, step_idx):
        """Decode without remasking, time rescaled to [1, t_on] (or [1, 0])."""
        if use_loop:
            # Map step_idx ∈ [0, n_phase1) → t ∈ (t_on, 1]
            t = 1.0 - step_idx * (1.0 - t_on) / n_phase1
            s = 1.0 - (step_idx + 1) * (1.0 - t_on) / n_phase1
            s = jnp.maximum(s, t_on)
        else:
            # No loop: map step_idx ∈ [0, num_steps) → t ∈ (0, 1]
            t = (num_steps - step_idx) / num_steps
            s = jnp.maximum((num_steps - step_idx - 1) / num_steps, 0.0)

        alpha_t_val = schedule_fn(t)
        alpha_s_val = schedule_fn(s)
        return _remdm_step(carry, step_idx, t, alpha_t_val, alpha_s_val,
                           sigma_active=False)

    # =====================================================================
    # Phase 2: Remasking loop at constant alpha(t_on)
    # =====================================================================
    def _phase2_step(carry, step_idx):
        """Remask + re-predict at constant alpha.  [Algorithm 3, Phase 2]"""
        # alpha_t = alpha_s = alpha(t_on)  →  denom = 1 - alpha(t_on).
        # For the model input time, we pass t_on.
        return _remdm_step(carry, step_idx, t_on, alpha_loop, alpha_loop,
                           sigma_active=True)

    # =====================================================================
    # Phase 3: Final MDLM decode (sigma = 0), t from t_off → 0
    # =====================================================================
    def _phase3_step(carry, step_idx):
        """Finish decoding without remasking, time rescaled to [t_off, 0]."""
        # Map step_idx ∈ [0, n_phase3) → t ∈ (0, t_off]
        t = t_off - step_idx * t_off / n_phase3
        s = t_off - (step_idx + 1) * t_off / n_phase3
        s = jnp.maximum(s, 0.0)
        alpha_t_val = schedule_fn(t)
        alpha_s_val = schedule_fn(s)
        return _remdm_step(carry, step_idx, t, alpha_t_val, alpha_s_val,
                           sigma_active=False)

    # --- Non-loop path (strategies: rescale, cap, conf without phases) ------
    def _simple_step(carry, step_idx):
        """Single-phase sampling with remasking active throughout."""
        t = (num_steps - step_idx) / num_steps
        s = jnp.maximum((num_steps - step_idx - 1) / num_steps, 0.0)
        alpha_t_val = schedule_fn(t)
        alpha_s_val = schedule_fn(s)
        return _remdm_step(carry, step_idx, t, alpha_t_val, alpha_s_val,
                           sigma_active=True)

    # --- Run the scan(s) ----------------------------------------------------
    carry = (z_init, rng, psi_init)

    if use_loop:
        # Phase 1
        carry, _ = jax.lax.scan(_phase1_step, carry, jnp.arange(n_phase1))
        # Phase 2
        carry, _ = jax.lax.scan(_phase2_step, carry, jnp.arange(n_phase2))
        # Phase 3
        if n_phase3 > 0:
            carry, _ = jax.lax.scan(_phase3_step, carry, jnp.arange(n_phase3))
    else:
        carry, _ = jax.lax.scan(_simple_step, carry, jnp.arange(num_steps))

    z_final, rng_final, _ = carry

    # --- Final cleanup: decode any remaining masked tokens -------------------
    t_final = jnp.zeros((batch_size,))
    final_logits = model_apply(params, obs, z_final, t_final, None)
    x_final = jnp.argmax(final_logits, axis=-1)
    z_final = jnp.where(z_final == mask_token_id, x_final, z_final)

    return z_final
