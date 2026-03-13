"""Forward process: q(z_t | x_0) by independent per-token masking."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def forward_process(
    rng: jax.Array,
    x_0: jnp.ndarray,
    alpha_t: jnp.ndarray,
    mask_id: int,
) -> jnp.ndarray:
    """Sample z_t ~ q(z_t | x_0).  Each token stays with prob alpha_t, else MASK.

    Args:
        rng:     PRNG key.
        x_0:     [B, H] int32, clean actions.
        alpha_t: [B] or scalar, retention probability.
        mask_id: MASK token index (= num_actions).

    Returns:
        z_t: [B, H] int32.
    """
    keep = jax.random.uniform(rng, shape=x_0.shape)
    alpha_t = jnp.reshape(alpha_t, (-1, 1))
    return jnp.where(keep < alpha_t, x_0, jnp.array(mask_id, dtype=x_0.dtype))
