"""Train a reward model (MLP discriminator or RND) on offline trajectory data."""

from __future__ import annotations

import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import serialization
from flax.training.train_state import TrainState

from src.models.reward_models import get_reward_model


# ---------------------------------------------------------------------------
# Loss functions per architecture
# ---------------------------------------------------------------------------

def _mlp_train_step(model):
    """Meier & Mujika discriminator: MSE to +1 (positive) and -1 (negative)."""

    @jax.jit
    def step(state, batch_neg, batch_pos):
        def loss_fn(params):
            r_neg = model.apply(params, batch_neg)
            r_pos = model.apply(params, batch_pos)
            l_neg = jnp.mean((r_neg - (-1.0)) ** 2)
            l_pos = jnp.mean((r_pos - 1.0) ** 2)
            return l_neg + l_pos, {
                "reward_loss": l_neg + l_pos,
                "loss_neg": l_neg, "loss_pos": l_pos,
                "pred_reward_neg": r_neg.mean(), "pred_reward_pos": r_pos.mean(),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads), metrics

    return step


def _rnd_train_step(model):
    """RND / Vision-RND: minimize predictor error on negative (boring) samples."""

    @jax.jit
    def step(state, batch_neg, batch_pos):
        def loss_fn(params):
            r_neg = model.apply(params, batch_neg)
            return jnp.mean(r_neg), r_neg

        (loss_neg, r_neg), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        r_pos = model.apply(state.params, batch_pos)
        metrics = {
            "reward_loss": loss_neg, "loss_neg": loss_neg, "loss_pos": jnp.mean(r_pos),
            "pred_reward_neg": r_neg.mean(), "pred_reward_pos": r_pos.mean(),
        }
        return state, metrics

    return step


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_train_reward(config: dict[str, Any]) -> None:
    data_path = config["OFFLINE_DATA_PATH"]
    print(f"Loading offline trajectories from {data_path}...")
    data = np.load(data_path)
    obs = data["obs"]
    if obs.ndim > 2:
        obs = obs.reshape(-1, obs.shape[-1])
    print(f"Frames: {obs.shape[0]}, obs_dim: {obs.shape[1]}")

    # Positive/negative split (last 20% = positive)
    split_idx = int(obs.shape[0] * 0.8)
    obs_neg, obs_pos = obs[:split_idx], obs[split_idx:]
    print(f"Negative: {obs_neg.shape[0]}, Positive: {obs_pos.shape[0]}")

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    rng, init_rng = jax.random.split(rng)

    model_type = config.get("REWARD_MODEL_TYPE", "mlp")
    model = get_reward_model(model_type)
    params = model.init(init_rng, jnp.zeros((1, obs.shape[-1])))

    load_path = config.get("REWARD_LOAD_PATH")
    if load_path and os.path.exists(load_path):
        with open(load_path, "rb") as f:
            params = serialization.from_bytes(params, f.read())
        print(f"Loaded reward weights from {load_path}")

    state = TrainState.create(
        apply_fn=model.apply, params=params,
        tx=optax.adam(config.get("REWARD_LR", 1e-4)),
    )

    train_step = _mlp_train_step(model) if model_type == "mlp" else _rnd_train_step(model)

    if config.get("USE_WANDB"):
        wandb.init(project=config.get("WANDB_PROJECT", "remdm-craftax"), name="Train-Neural-Reward")

    epochs = config.get("REWARD_EPOCHS", 10)
    batch_size = config.get("BATCH_SIZE", 256)

    for epoch in range(epochs):
        rng, shuffle_rng = jax.random.split(rng)
        perm_neg = jax.random.permutation(shuffle_rng, obs_neg.shape[0])
        perm_pos = jax.random.permutation(shuffle_rng, obs_pos.shape[0])
        neg_shuffled = obs_neg[perm_neg]
        pos_shuffled = obs_pos[perm_pos]

        n_batches = min(len(obs_neg), len(obs_pos)) // batch_size
        epoch_metrics = []
        for b in range(n_batches):
            s = b * batch_size
            state, mets = train_step(state, neg_shuffled[s:s + batch_size], pos_shuffled[s:s + batch_size])
            epoch_metrics.append(mets)

        avg = {k: np.mean([float(m[k]) for m in epoch_metrics]) for k in epoch_metrics[0]}
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg['reward_loss']:.3f} "
            f"| Pos: {avg['pred_reward_pos']:.2f} | Neg: {avg['pred_reward_neg']:.2f}"
        )
        if config.get("USE_WANDB"):
            wandb.log({f"reward_model/{k}": v for k, v in avg.items()}, step=epoch)

    save_path = config.get("REWARD_SAVE_PATH", "checkpoints/reward_model.msgpack")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(serialization.to_bytes(state.params))
    print(f"Saved reward weights to {save_path}")
