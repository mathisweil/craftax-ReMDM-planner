"""Noise schedules for masked discrete diffusion.

alpha(t) is the retention probability: alpha(0)=1 (clean), alpha(1)=0 (fully masked).
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

ScheduleFn = Callable[[jnp.ndarray], jnp.ndarray]


def linear_schedule(t: jnp.ndarray) -> jnp.ndarray:
    """alpha(t) = 1 - t.  Default in MDLM / ReMDM."""
    return 1.0 - t


def linear_schedule_deriv(t: jnp.ndarray) -> jnp.ndarray:
    return jnp.full_like(t, -1.0)


def cosine_schedule(t: jnp.ndarray) -> jnp.ndarray:
    """alpha(t) = cos(pi * t / 2)."""
    return jnp.cos(t * jnp.pi / 2.0)


def cosine_schedule_deriv(t: jnp.ndarray) -> jnp.ndarray:
    return -(jnp.pi / 2.0) * jnp.sin(t * jnp.pi / 2.0)


SCHEDULE_MAP: dict[str, tuple[ScheduleFn, ScheduleFn]] = {
    "linear": (linear_schedule, linear_schedule_deriv),
    "cosine": (cosine_schedule, cosine_schedule_deriv),
}
