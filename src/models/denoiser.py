"""Denoising Transformer for masked discrete diffusion planning.

Architecture: obs MLP encoder + sinusoidal time embedding + bidirectional
transformer.  Two prefix tokens (obs, time) precede the action sequence.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal

_INIT = orthogonal(np.sqrt(2))
_INIT_SMALL = orthogonal(0.01)
_BIAS = constant(0.0)


class SinusoidalPosEmbed(nn.Module):
    """Sinusoidal embedding for continuous timesteps or integer positions."""

    dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        half = self.dim // 2
        freqs = jnp.exp(-jnp.log(10_000.0) * jnp.arange(half) / half)
        angles = x[..., None] * freqs
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros_like(emb[..., :1])], axis=-1)
        return emb


class TransformerBlock(nn.Module):
    """Pre-norm transformer: LN -> MHA -> res -> LN -> FFN -> res.

    (A) framework default: flax LayerNorm eps 1e-6 vs torch 1e-5 in the
    minihack repo (METHOD_PARITY 2.3).
    """

    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1
    deterministic: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, kernel_init=_INIT, deterministic=self.deterministic,
        )(h, h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(h)
        x = x + h

        h = nn.LayerNorm()(x)
        h = nn.Dense(self.d_ff, kernel_init=_INIT, bias_init=_BIAS)(h)
        h = nn.gelu(h)
        h = nn.Dense(self.d_model, kernel_init=_INIT, bias_init=_BIAS)(h)
        h = nn.Dropout(rate=self.dropout_rate, deterministic=self.deterministic)(h)
        return x + h


class DenoisingTransformer(nn.Module):
    """Denoising transformer for masked discrete diffusion planning.

    Input:  (obs [B, D], noisy_actions [B, H], timestep [B])
    Output: logits [B, H, num_actions] (no MASK logit).
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
        B = obs.shape[0]
        vocab = self.num_actions + 1  # +1 for MASK token

        # Observation encoder
        h = nn.Dense(self.obs_encoder_width, kernel_init=_INIT, bias_init=_BIAS)(obs)
        h = nn.LayerNorm()(h)
        h = nn.relu(h)
        for _ in range(self.obs_encoder_layers - 1):
            h = nn.Dense(self.obs_encoder_width, kernel_init=_INIT, bias_init=_BIAS)(h)
            h = nn.relu(h)
        obs_tok = nn.Dense(self.d_model, kernel_init=_INIT, bias_init=_BIAS)(h)[:, None, :]

        # Time embedding
        t = timestep.reshape(B)
        t_emb = SinusoidalPosEmbed(self.d_model)(t)
        t_emb = nn.Dense(self.d_model, kernel_init=_INIT, bias_init=_BIAS)(t_emb)
        t_emb = nn.gelu(t_emb)
        t_tok = nn.Dense(self.d_model, kernel_init=_INIT, bias_init=_BIAS)(t_emb)[:, None, :]

        # Action token embedding
        act_emb = nn.Embed(num_embeddings=vocab, features=self.d_model)(noisy_actions)

        # Assemble sequence: [obs, time, actions]
        seq = jnp.concatenate([obs_tok, t_tok, act_emb], axis=1)
        seq_len = 2 + self.plan_horizon
        pos_emb = SinusoidalPosEmbed(self.d_model)(jnp.arange(seq_len))
        seq = seq + pos_emb[None, :, :]

        # Transformer
        for _ in range(self.n_layers):
            seq = TransformerBlock(
                d_model=self.d_model, n_heads=self.n_heads, d_ff=self.d_ff,
                dropout_rate=self.dropout_rate, deterministic=deterministic,
            )(seq)
        seq = nn.LayerNorm()(seq)

        # Output logits over real actions (skip 2 prefix tokens)
        return nn.Dense(self.num_actions, kernel_init=_INIT_SMALL, bias_init=_BIAS)(
            seq[:, 2:, :]
        )
