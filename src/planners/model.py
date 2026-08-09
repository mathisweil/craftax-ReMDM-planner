"""Diffusion model lifecycle: construction, parameter init, checkpoint I/O, and apply closures."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from src.models.denoiser import DenoisingTransformer

logger = logging.getLogger(__name__)

_METADATA_FILENAME = "resume_metadata.json"


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
        return str(Path(path).resolve())

    import wandb

    artifact_ref = path.removeprefix("wandb:")
    api = wandb.Api()
    artifact = api.artifact(artifact_ref)
    local_path = (
        artifact.download(root=download_dir) if download_dir else artifact.download()
    )
    print(f"Downloaded W&B artifact '{artifact_ref}' -> '{local_path}'")
    return local_path


def abstract_init(init_fn: Callable[[], Any]) -> Any:
    """Return the abstract pytree an init function would produce.

    Uses ``jax.eval_shape`` so no parameter memory is allocated; the result
    is a pytree of ``jax.ShapeDtypeStruct`` leaves suitable as an Orbax
    restore target.
    """
    return jax.eval_shape(init_fn)


def restore_latest(path: str, args: Any) -> tuple[Any, int]:
    """Restore the latest step of an Orbax checkpoint.

    Returns:
        ``(restored, step)`` tuple.

    Raises:
        FileNotFoundError: If the directory contains no saved steps.
    """
    path = str(Path(path).resolve())
    with ocp.CheckpointManager(path) as mgr:
        step = mgr.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint at {path}")
        restored = mgr.restore(step, args=args)
    return restored, step


def restore_latest_params(path: str, abstract_params_tree: Any) -> tuple[Any, int]:
    """Restore only the ``params`` entry from the latest checkpoint step.

    Ignores optimiser state and anything else in the checkpoint. The
    ``{"params": ...}`` wrapper must mirror the save side; do not change
    one without the other.
    """
    restored, step = restore_latest(
        path,
        ocp.args.PyTreeRestore(
            item={"params": abstract_params_tree},
            partial_restore=True,
        ),
    )
    return restored["params"], step


def abstract_params(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
) -> Any:
    """Abstract (shape/dtype only) version of :func:`init_params`.

    Traces the init with ``jax.eval_shape`` so no parameter memory is
    allocated. Used to build restore targets for checkpoint loading.
    """
    return abstract_init(lambda: init_params(model, rng, obs_dim, plan_horizon))


def load_checkpoint(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
    path: str,
) -> Any:
    """Load diffusion model parameters from an Orbax checkpoint.

    This function is intended for inference, evaluation, or initialising
    a new fine-tuning run. It restores model parameters only and deliberately
    ignores optimiser state and the saved training step.

    Use ``load_checkpoint_for_resume`` instead when continuing an interrupted
    training run.
    """
    abstract = abstract_params(model, rng, obs_dim, plan_horizon)
    params, step = restore_latest_params(path, abstract)
    print(f"Loaded diffusion parameters from '{path}' (checkpoint step {step})")
    return params


def load_checkpoint_for_resume(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
    path: str,
    lr_schedule: float | Callable[[int], float],
    max_grad_norm: float,
) -> TrainState:
    """Load a full ``TrainState`` (params + optimiser state) for resume.

    The abstract ``TrainState`` restore target is built with
    ``jax.eval_shape``, so no throwaway parameters or optimiser moments are
    allocated before restoring. The ``lr_schedule`` and ``max_grad_norm``
    must produce the same optimiser chain structure as the saved run.

    Returns:
        Restored ``TrainState``. The caller should call ``.replace(step=...)``
        to set the correct LR offset for the resumed run.
    """
    abstract_state = abstract_init(
        lambda: create_train_state(
            model,
            init_params(model, rng, obs_dim, plan_horizon),
            lr_schedule,
            max_grad_norm,
        )
    )

    restored_state, step = restore_latest(
        path, ocp.args.StandardRestore(abstract_state)
    )

    print(
        f"Loaded full TrainState for resume from '{path}' "
        f"(step {step}, opt_state step {restored_state.step})"
    )
    return restored_state


def create_train_state(
    model: DenoisingTransformer,
    params: Any,
    lr: float | Callable[[int], float],
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

    def apply_eval(
        params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray, _rng=None
    ):
        return model.apply(params, obs, z_t, t)

    def apply_train(
        params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray, rng=None
    ):
        return model.apply(
            params,
            obs,
            z_t,
            t,
            deterministic=False,
            rngs={"dropout": rng} if rng is not None else {},
        )

    return apply_eval, apply_train


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar types."""

    def default(self, o: Any) -> Any:
        """Serialize numpy scalars to native Python types.

        Args:
            o: Object to serialize.

        Returns:
            JSON-serializable object.
        """
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def save_checkpoint_metadata(
    checkpoint_dir: str,
    mode: str,
    update_step: int,
    total_gradient_steps: int,
    wandb_run_id: str | None,
    config: dict[str, Any],
) -> None:
    """Write a JSON metadata sidecar alongside an Orbax checkpoint.

    Args:
        checkpoint_dir: Root directory of the Orbax checkpoint manager.
        mode:           Training mode (``"offline"`` or ``"online"``).
        update_step:    Final update step index.
        total_gradient_steps: Total gradient steps completed.
        wandb_run_id:   Current W&B run ID, or ``None``.
        config:         Full training config snapshot.
    """
    metadata = {
        "mode": mode,
        "update_step": int(update_step),
        "total_gradient_steps_completed": int(total_gradient_steps),
        "wandb_run_id": wandb_run_id,
        "config_snapshot": config,
    }
    path = Path(checkpoint_dir) / _METADATA_FILENAME
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, cls=_NumpyEncoder)
    print(f"Saved checkpoint metadata to {path}")


def load_checkpoint_metadata(
    checkpoint_dir: str,
) -> dict[str, Any] | None:
    """Read the JSON metadata sidecar from a checkpoint directory.

    Args:
        checkpoint_dir: Root directory of the Orbax checkpoint manager.

    Returns:
        Parsed metadata dict, or ``None`` if the sidecar does not exist
        (backward-compatible with checkpoints created before this feature).
    """
    path = Path(checkpoint_dir) / _METADATA_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
