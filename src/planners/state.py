"""Model initialization, optimizer setup, checkpoint I/O, and apply-function closures."""

from __future__ import annotations
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from src.models.denoiser import DenoisingTransformer


def build_model(config: dict, num_actions: int) -> DenoisingTransformer:
    """Construct a DenoisingTransformer from a config dict."""
    return DenoisingTransformer(
        num_actions=num_actions,
        plan_horizon=config["PLAN_HORIZON"],
        d_model=config.get("D_MODEL", 256),
        n_heads=config.get("N_HEADS", 4),
        n_layers=config.get("N_LAYERS", 4),
        d_ff=config.get("D_FF", 512),
        obs_encoder_layers=config.get("OBS_ENCODER_LAYERS", 2),
        obs_encoder_width=config.get("OBS_ENCODER_WIDTH", 512),
        dropout_rate=config.get("DROPOUT_RATE", 0.1),
    )


def init_params(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
) -> Any:
    """Initialize model parameters with dummy inputs."""
    return model.init(
        rng,
        jnp.zeros((1, obs_dim)),
        jnp.zeros((1, plan_horizon), dtype=jnp.int32),
        jnp.zeros((1,)),
    )


def load_checkpoint(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
    path: str,
) -> Any:
    """Load diffusion model parameters from an Orbax checkpoint."""
    params = init_params(model, rng, obs_dim, plan_horizon)

    # Build an abstract/dummy TrainState matching the saved structure
    abstract_state = create_train_state(
        model=model,
        params=params,
        lr=1e-4,  # dummy, only used to match structure
        max_grad_norm=1.0,  # dummy, only used to match structure
    )

    with ocp.CheckpointManager(path) as mgr:
        step = mgr.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint at {path}")

        restored_state = mgr.restore(
            step,
            args=ocp.args.StandardRestore(item=abstract_state),
        )

    print(f"Loaded diffusion checkpoint from '{path}' (step {step})")
    return restored_state.params


def create_train_state(
    model: DenoisingTransformer,
    params: Any,
    lr: float,
    max_grad_norm: float,
) -> TrainState:
    """TrainState with gradient clipping + Adam."""
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=1e-5))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_apply_fns(model: DenoisingTransformer):
    """Return (apply_eval, apply_train) closures matching ModelApplyFn."""

    def apply_eval(params, obs, z_t, t, _rng=None):
        return model.apply(params, obs, z_t, t)

    def apply_train(params, obs, z_t, t, rng=None):
        return model.apply(
            params, obs, z_t, t,
            deterministic=False,
            rngs={"dropout": rng} if rng is not None else {},
        )

    return apply_eval, apply_train
