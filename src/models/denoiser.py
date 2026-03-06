"""DenoisingTransformer: observation MLP encoder + sinusoidal time embedding
+ bidirectional transformer for masked discrete diffusion planning."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal positional embedding for continuous timesteps or integer positions."""

    d_model: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Embed scalar or integer values into d_model-dimensional sinusoidal features.

        Args:
            x: Arbitrary shape, float or int values to embed.

        Returns:
            Tensor of shape ``(*x.shape, d_model)``.
        """
        half = self.d_model // 2
        freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half) / half)
        angles = x[..., None] * freqs
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        if self.d_model % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[..., :1])], axis=-1)
        return emb


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHA -> residual -> LN -> FFN -> residual."""

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    deterministic: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            kernel_init=orthogonal(np.sqrt(2)),
            deterministic=self.deterministic,
        )(h, h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(h)
        x = x + h

        h = nn.LayerNorm()(x)
        h = nn.Dense(
            self.d_ff, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(h)
        h = nn.gelu(h)
        h = nn.Dense(
            self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(h)
        x = x + h
        return x


class DenoisingTransformer(nn.Module):
    """Denoising transformer for masked discrete diffusion planning.

    Takes an observation, a noisy action sequence (with MASK tokens), and a
    diffusion timestep. Outputs logits over the real action vocabulary for
    each position in the plan.

    The MASK token has id = ``num_actions`` (appended to the action vocabulary).
    Output logits have shape ``[batch, plan_horizon, num_actions]`` — no logit
    for the MASK token since the model only predicts real actions.
    """

    num_actions: int
    plan_horizon: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    obs_encoder_layers: int = 2
    obs_encoder_width: int = 512
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        noisy_actions: jnp.ndarray,
        timestep: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Forward pass.

        Args:
            obs:            [batch, obs_dim] float32.
            noisy_actions:  [batch, plan_horizon] int32, values in [0, num_actions].
            timestep:       [batch] float32, t in [0, 1].
            deterministic:  If ``False``, enables dropout during training.

        Returns:
            logits: [batch, plan_horizon, num_actions] float32.
        """
        batch_size = obs.shape[0]
        vocab_size = self.num_actions + 1  # +1 for MASK token

        # --- Observation encoder (MLP) ---
        obs_emb = obs
        for _ in range(self.obs_encoder_layers):
            obs_emb = nn.Dense(
                self.obs_encoder_width,
                kernel_init=orthogonal(np.sqrt(2)),
                bias_init=constant(0.0),
            )(obs_emb)
            obs_emb = nn.relu(obs_emb)
        obs_emb = nn.Dense(
            self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs_emb)  # [batch, d_model]

        # --- Timestep embedding ---
        t = timestep.reshape(batch_size)
        t_emb = SinusoidalPosEmbed(self.d_model)(t)
        t_emb = nn.Dense(
            self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(t_emb)
        t_emb = nn.gelu(t_emb)
        t_emb = nn.Dense(
            self.d_model, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(t_emb)  # [batch, d_model]

        # --- Action token embedding ---
        action_emb = nn.Embed(num_embeddings=vocab_size, features=self.d_model)(
            noisy_actions
        )  # [batch, plan_horizon, d_model]

        # --- Positional encoding for action sequence positions ---
        positions = jnp.arange(self.plan_horizon)
        pos_emb = SinusoidalPosEmbed(self.d_model)(positions)
        # pos_emb: [plan_horizon, d_model]
        action_emb = action_emb + pos_emb[None, :, :]

        # --- Assemble: [cond_token, action_1, ..., action_H] ---
        cond_token = (obs_emb + t_emb)[:, None, :]  # [batch, 1, d_model]
        seq = jnp.concatenate([cond_token, action_emb], axis=1)
        # seq: [batch, 1 + plan_horizon, d_model]

        # --- Transformer blocks (bidirectional) ---
        for _ in range(self.n_layers):
            seq = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout_rate=self.dropout_rate,
                deterministic=deterministic,
            )(seq)

        seq = nn.LayerNorm()(seq)

        # --- Output head: logits for action positions only ---
        action_features = seq[:, 1:, :]  # [batch, plan_horizon, d_model]
        logits = nn.Dense(
            self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(action_features)
        return logits
