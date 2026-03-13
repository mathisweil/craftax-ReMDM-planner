"""Reward model architectures: MLP discriminator, RND, and Vision-RND."""

from __future__ import annotations
from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import orthogonal, lecun_normal


class DeterministicNeuralReward(nn.Module):
    """MLP mapping obs -> scalar reward (Meier & Mujika discriminator)."""
    hidden_dims: Sequence[int] = (512, 256, 128)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for dim in self.hidden_dims:
            x = nn.relu(nn.Dense(dim)(x))
        return jnp.squeeze(nn.Dense(1)(x), axis=-1)


class RNDReward(nn.Module):
    """Random Network Distillation.  Reward = MSE(predictor, frozen_target)."""
    hidden_dims: Sequence[int] = (256, 128, 64)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        def _mlp(x, prefix):
            for dim in self.hidden_dims:
                x = nn.relu(nn.Dense(dim, name=f"{prefix}_{dim}")(x))
            return x

        target_emb = jax.lax.stop_gradient(_mlp(obs, "target"))
        pred_emb = _mlp(obs, "pred")
        return jnp.mean((pred_emb - target_emb) ** 2, axis=-1)


class VisionRNDReward(nn.Module):
    """RND with conv backbone.  Orthogonal init on target for high entropy."""
    internal_w: int = 9
    internal_h: int = 9
    internal_c: int = 16
    scale: float = 100.0

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        def _cnn(x, prefix):
            w_init = orthogonal(1.414) if prefix == "target" else lecun_normal()
            flat_size = self.internal_w * self.internal_h * self.internal_c

            x = nn.relu(nn.Dense(flat_size, kernel_init=w_init, name=f"{prefix}_proj")(x))
            x = x.reshape(x.shape[:-1] + (self.internal_w, self.internal_h, self.internal_c))
            x = nn.relu(nn.Conv(64, (3, 3), (1, 1), "SAME", kernel_init=w_init, name=f"{prefix}_conv1")(x))
            x = nn.relu(nn.Conv(128, (3, 3), (2, 2), "VALID", kernel_init=w_init, name=f"{prefix}_conv2")(x))
            x = x.reshape(x.shape[:-3] + (-1,))
            return nn.Dense(128, kernel_init=w_init, name=f"{prefix}_dense")(x)

        target_emb = jax.lax.stop_gradient(_cnn(obs, "target"))
        pred_emb = _cnn(obs, "pred")
        return jnp.mean((pred_emb - target_emb) ** 2, axis=-1) * self.scale


REWARD_MODELS = {
    "mlp": DeterministicNeuralReward,
    "rnd": RNDReward,
    "vision_rnd": VisionRNDReward,
}


def get_reward_model(model_type: str) -> nn.Module:
    if model_type not in REWARD_MODELS:
        raise ValueError(f"Unknown reward model: {model_type!r}. Options: {list(REWARD_MODELS)}")
    return REWARD_MODELS[model_type]()
