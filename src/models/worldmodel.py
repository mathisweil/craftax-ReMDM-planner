"""Causal Transformer World Model for predicting next states and rewards."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable

class CausalSelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=True):
        seq_len = x.shape[1]
        # Create a causal mask (lower triangular matrix)
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        causal_mask = causal_mask[None, None, :, :] # (1, 1, T, T) for multi-head attn

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.num_heads * self.head_dim,
            out_features=x.shape[-1],
            dropout_rate=self.dropout_rate
        )(x, x, mask=causal_mask, deterministic=deterministic)
        return x

class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=True):
        # Attention
        y = nn.LayerNorm()(x)
        y = CausalSelfAttention(
            num_heads=self.num_heads, 
            head_dim=self.head_dim, 
            dropout_rate=self.dropout_rate
        )(y, deterministic=deterministic)
        x = x + y

        # MLP
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.gelu(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        y = nn.Dense(x.shape[-1])(y)
        y = nn.Dropout(self.dropout_rate)(y, deterministic=deterministic)
        x = x + y
        
        return x

class TransformerWorldModel(nn.Module):
    num_actions: int
    obs_dim: int
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, obs_seq, act_seq, deterministic=True):
        """
        obs_seq: (Batch, Time, Obs_Dim)
        act_seq: (Batch, Time)
        Returns predictions for Next_Obs and Reward
        """
        B, T, _ = obs_seq.shape

        # 1. Embeddings
        obs_emb = nn.Dense(self.d_model)(obs_seq)
        act_emb = nn.Embed(num_embeddings=self.num_actions, features=self.d_model)(act_seq)
        
        # Positional Encoding
        positions = jnp.arange(T)[None, :]
        pos_emb = nn.Embed(num_embeddings=512, features=self.d_model)(positions)

        # Combine
        x = obs_emb + act_emb + pos_emb
        x = nn.Dropout(self.dropout_rate)(x, deterministic=deterministic)

        # 2. Transformer Blocks
        head_dim = self.d_model // self.n_heads
        for _ in range(self.n_layers):
            x = TransformerBlock(
                num_heads=self.n_heads, 
                head_dim=head_dim, 
                mlp_dim=self.d_model * 4, 
                dropout_rate=self.dropout_rate
            )(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)

        # 3. Prediction Heads
        next_obs_preds = nn.Dense(self.obs_dim, name="next_obs_head")(x)
        
        # New split reward heads
        rew_logits = nn.Dense(1, name="reward_bce_head")(x)
        rew_mags = nn.Dense(1, name="reward_mag_head")(x)
        
        rew_logits = jnp.squeeze(rew_logits, axis=-1) # (B, T)
        rew_mags = jnp.squeeze(rew_mags, axis=-1)     # (B, T)

        return next_obs_preds, rew_logits, rew_mags