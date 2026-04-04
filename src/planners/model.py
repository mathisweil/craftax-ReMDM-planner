"""Diffusion model lifecycle: construction, parameter init, checkpoint I/O, and apply closures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from src.models.denoiser import DenoisingTransformer


def build_model(config: dict, num_actions: int) -> DenoisingTransformer:
    """Construct a :class:`DenoisingTransformer` from a config dict.

    Args:
        config:      Upper-cased config dict with architecture hyperparameters.
        num_actions: Size of the discrete action vocabulary.

    Returns:
        An uninitialised :class:`DenoisingTransformer` instance.
    """
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
    """Initialize model parameters with dummy inputs.

    Args:
        model:        Flax module to initialise.
        rng:          PRNG key.
        obs_dim:      Observation dimensionality.
        plan_horizon: Number of action steps in a plan.

    Returns:
        Initialised parameter pytree.
    """
    return model.init(
        rng,
        jnp.zeros((1, obs_dim)),
        jnp.zeros((1, plan_horizon), dtype=jnp.int32),
        jnp.zeros((1,)),
    )


def resolve_checkpoint_path(
    path: str,
    download_dir: str | None = None,
) -> str:
    """Resolve a checkpoint path, downloading from W&B if it is an artifact reference.

    Paths prefixed with ``wandb:`` are treated as W&B artifact references
    (e.g. ``wandb:entity/project/name:version``) and downloaded locally
    before returning the filesystem path.

    Args:
        path:         Local filesystem path or ``wandb:``-prefixed artifact
                      reference.
        download_dir: Root directory for downloaded artifacts.  When ``None``,
                      falls back to the wandb default (``./artifacts/``).

    Returns:
        Local filesystem path to the checkpoint directory.
    """
    if not path.startswith("wandb:"):
        return path

    import wandb

    artifact_ref = path.removeprefix("wandb:")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    local_path = (
        artifact.download(root=download_dir) if download_dir else artifact.download()
    )
    print(f"Downloaded W&B artifact '{artifact_ref}' -> '{local_path}'")
    return local_path


def load_checkpoint(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
    path: str,
) -> Any:
    """Load diffusion model parameters from an Orbax checkpoint.

    Args:
        model:        Flax module (used to build the abstract state structure).
        rng:          PRNG key for dummy initialisation.
        obs_dim:      Observation dimensionality.
        plan_horizon: Number of action steps in a plan.
        path:         Path to the Orbax checkpoint directory.

    Returns:
        Restored parameter pytree.

    Raises:
        FileNotFoundError: If the checkpoint directory contains no saved steps.
    """
    path = str(Path(path).resolve())
    params = init_params(model, rng, obs_dim, plan_horizon)
    abstract_state = create_train_state(model=model, params=params, lr=1e-4, max_grad_norm=1.0)

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
    lr: Union[float, Callable[[int], float]],
    max_grad_norm: float,
) -> TrainState:
    """Create a :class:`TrainState` with gradient clipping and Adam.

    Args:
        model:         Flax module (used only to bind ``apply_fn``).
        params:        Initialised parameter pytree.
        lr:            Constant learning rate or an optax schedule
                       (any callable ``step -> lr``).
        max_grad_norm: Global gradient clipping threshold.

    Returns:
        A Flax ``TrainState`` ready for ``apply_gradients``.
    """
    tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(lr, eps=1e-5))
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_apply_fns(
    model: DenoisingTransformer,
) -> tuple[Callable, Callable]:
    """Return ``(apply_eval, apply_train)`` closures matching ``ModelApplyFn``.

    Args:
        model: Flax module.

    Returns:
        Tuple of ``(apply_eval, apply_train)`` where ``apply_train`` enables
        dropout via ``rngs={"dropout": rng}``.
    """

    def apply_eval(params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray, _rng=None):
        return model.apply(params, obs, z_t, t)

    def apply_train(params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray, rng=None):
        return model.apply(
            params, obs, z_t, t,
            deterministic=False,
            rngs={"dropout": rng} if rng is not None else {},
        )

    return apply_eval, apply_train
