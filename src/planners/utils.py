from __future__ import annotations

import pathlib
from typing import Any, Callable, Dict, Optional, Tuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp

from src.models.denoiser import DenoisingTransformer
from src.models.remdm import (
    ScheduleFn,
    cosine_schedule,
    linear_schedule,
)

from Craftax_Baselines.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
    OptimisticResetVecEnvWrapper,
)
from src.envs.wrappers import SequenceHistoryWrapper

SCHEDULE_MAP: Dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# =============================================================================
# Model helpers
# =============================================================================


def _build_model(config: Dict[str, Any], num_actions: int) -> DenoisingTransformer:
    """Instantiate a ``DenoisingTransformer`` from the config dict."""
    return DenoisingTransformer(
        num_actions=num_actions,
        plan_horizon=config["PLAN_HORIZON"],
        d_model=config["D_MODEL"],
        n_heads=config["N_HEADS"],
        n_layers=config["N_LAYERS"],
        d_ff=config["D_FF"],
        obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
        obs_encoder_width=config["OBS_ENCODER_WIDTH"],
        dropout_rate=config["DROPOUT_RATE"],
    )


def _init_model_params(
    model: DenoisingTransformer,
    rng: jax.Array,
    obs_dim: int,
    plan_horizon: int,
) -> Any:
    """Create initial model parameters with dummy inputs."""
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
    dummy_t = jnp.zeros((1,))
    return model.init(rng, dummy_obs, dummy_act, dummy_t)


def _create_train_state(
    model: DenoisingTransformer,
    params: Any,
    lr: float,
    max_grad_norm: float,
) -> TrainState:
    """Create an optax ``TrainState`` with gradient clipping + Adam."""
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# =============================================================================
# Checkpoint I/O
# =============================================================================


def _restore_train_state(checkpoint_path: str, abstract_ts: TrainState) -> TrainState:
    """Handles core Orbax checkpoint restoration logic."""
    checkpointer = ocp.StandardCheckpointer()
    ckpt_mgr = ocp.CheckpointManager(
        checkpoint_path,
        checkpointer,
        options=ocp.CheckpointManagerOptions(max_to_keep=1),
    )

    latest_step = ckpt_mgr.latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No valid checkpoint found at '{checkpoint_path}'")

    restored_ts = ckpt_mgr.restore(
        latest_step,
        args=ocp.args.StandardRestore(item=abstract_ts)
    )

    print(f"Loaded checkpoint from '{checkpoint_path}' (step={latest_step})")
    return restored_ts


def _load_checkpoint(
        config: Dict[str, Any],
        model: Any,  # DenoisingTransformer
        obs_dim: int,
        checkpoint_path: str,
) -> Any:
    """Load model parameters from an Orbax checkpoint (outside JIT)."""

    def get_abstract_state():
        rng = jax.random.PRNGKey(0)
        params = _init_model_params(model, rng, obs_dim, config["PLAN_HORIZON"])
        tx = optax.adam(config["LR"])
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    abstract_ts = jax.eval_shape(get_abstract_state)
    restored_ts = _restore_train_state(checkpoint_path, abstract_ts)

    return restored_ts.params


def _load_ppo_checkpoint(
        ppo_checkpoint_path: str,
        num_actions: Sequence[int],
        obs_dim: int,
        layer_size: int,
        model_type: str,
) -> Tuple[Any, Any]:
    """Load a pre-trained ActorCritic (MLP) PPO checkpoint."""
    from Craftax_Baselines.models.rnd import ActorCriticRND
    import flax.core
    import orbax.checkpoint as ocp

    network = ActorCriticRND(num_actions, layer_size)

    # Use PyTreeCheckpointer to match the save mechanism in ppo_rnd.py
    checkpointer = ocp.PyTreeCheckpointer()
    ckpt_mgr = ocp.CheckpointManager(ppo_checkpoint_path, checkpointer)

    latest_step = ckpt_mgr.latest_step()
    if latest_step is None:
        raise FileNotFoundError(f"No valid checkpoint found at '{ppo_checkpoint_path}'")

    # Restore the raw dictionary, bypassing strict TrainState structure matching
    restored_dict = ckpt_mgr.restore(latest_step)

    # Re-freeze the parameters dictionary for Flax
    restored_params = flax.core.freeze(restored_dict["params"])
    print(f"Loaded checkpoint params from '{ppo_checkpoint_path}' (step={latest_step})")

    return network, restored_params


def _save_model(
        train_state: TrainState, config: Dict[str, Any], dir_name: str
) -> None:
    """Save a TrainState checkpoint using orbax."""

    # 1. Determine path
    if config.get("USE_WANDB") and wandb.run is not None:
        path = str(pathlib.Path(wandb.run.dir) / dir_name)
    else:
        path = dir_name

    # 2. Modern checkpointer setup
    checkpointer = ocp.StandardCheckpointer()
    ckpt_mgr = ocp.CheckpointManager(
        path,
        checkpointer,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )

    # 3. Modern save execution
    step = config.get("NUM_TRAIN_STEPS", config.get("NUM_UPDATES", 0))
    ckpt_mgr.save(
        step,
        args=ocp.args.StandardSave(train_state),
    )

    # 4. Ensure asynchronous saves complete before exiting
    ckpt_mgr.wait_until_finished()

    print(f"Saved model checkpoint to '{path}'")


# =============================================================================
# Environment stack construction
# =============================================================================


def _make_env_stack(
    config: Dict[str, Any],
    num_envs: int,
    *,
    use_optimistic_resets: bool = False,
    use_sequence_history: bool = False,
) -> Tuple[Any, Any]:
    """Build the wrapper stack for Craftax environments.

    Standard stack::

        env -> SequenceHistoryWrapper (optional) -> LogWrapper
             -> AutoResetEnvWrapper -> BatchEnvWrapper

    With optimistic resets::

        env -> SequenceHistoryWrapper (optional) -> LogWrapper
             -> OptimisticResetVecEnvWrapper

    Args:
        config:                 Configuration dict.
        num_envs:               Number of parallel environments.
        use_optimistic_resets:  Use ``OptimisticResetVecEnvWrapper``.
        use_sequence_history:   Wrap with ``SequenceHistoryWrapper`` (innermost).

    Returns:
        (wrapped_env, env_params)
    """
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not use_optimistic_resets
    )
    env_params = env.default_params

    if use_sequence_history:
        obs_shape: Tuple[int, ...] = env.observation_space(env_params).shape
        history_len: int = config["PLAN_HORIZON"]
        env = SequenceHistoryWrapper(env, history_len=history_len, obs_shape=obs_shape)

    env_w = LogWrapper(env)

    if use_optimistic_resets:
        reset_ratio = min(config.get("OPTIMISTIC_RESET_RATIO", 16), num_envs)
        env_w = OptimisticResetVecEnvWrapper(
            env_w, num_envs=num_envs, reset_ratio=reset_ratio
        )
    else:
        env_w = AutoResetEnvWrapper(env_w)
        env_w = BatchEnvWrapper(env_w, num_envs=num_envs)

    return env_w, env_params


# =============================================================================
# Offline data utilities
# =============================================================================


def _valid_window_mask(
    dones: np.ndarray,
    plan_horizon: int,
) -> np.ndarray:
    """Return a boolean mask of valid ``plan_horizon`` start positions.

    ``valid[e, t] = True`` iff ``dones[e, t:t+plan_horizon-1]`` are all False,
    i.e. the window ``[t, t+plan_horizon)`` does not cross an episode boundary.

    Args:
        dones:        [num_envs, num_steps] bool numpy array.
        plan_horizon: Window length.

    Returns:
        valid: [num_envs, num_steps] bool numpy array.
    """
    num_envs, num_steps = dones.shape
    valid = np.ones((num_envs, num_steps), dtype=bool)
    for offset in range(plan_horizon - 1):
        shifted = np.roll(dones, -offset, axis=1)
        if offset > 0:
            shifted[:, -offset:] = True
        valid &= ~shifted
    # Last (plan_horizon - 1) positions don't have enough room
    if plan_horizon > 1:
        valid[:, -(plan_horizon - 1):] = False
    return valid


def _sample_windows_from_chunk(
    chunk_obs: np.ndarray,
    chunk_acts: np.ndarray,
    chunk_dones: np.ndarray,
    plan_horizon: int,
    batch_size: int,
    np_rng: np.random.Generator,
) -> Optional[Tuple[jnp.ndarray, jnp.ndarray]]:
    """Sample ``batch_size`` valid windows from a collected chunk.

    Args:
        chunk_obs:    [num_envs, T, obs_dim].
        chunk_acts:   [num_envs, T].
        chunk_dones:  [num_envs, T].
        plan_horizon: Window length.
        batch_size:   Number of windows to sample.
        np_rng:       NumPy random generator.

    Returns:
        ``(obs_batch, act_batch)`` or ``None`` if no valid windows exist.
    """
    valid = _valid_window_mask(chunk_dones, plan_horizon)
    env_idxs, time_idxs = np.where(valid)

    if len(env_idxs) == 0:
        return None

    replace = len(env_idxs) < batch_size
    idx = np_rng.choice(len(env_idxs), size=batch_size, replace=replace)
    sel_e = env_idxs[idx]
    sel_t = time_idxs[idx]

    obs_batch = jnp.array(chunk_obs[sel_e, sel_t])
    act_batch = jnp.array(
        np.stack(
            [chunk_acts[e, t: t + plan_horizon] for e, t in zip(sel_e, sel_t)]
        )
    )
    return obs_batch, act_batch


# =============================================================================
# Model apply helpers (deterministic vs training)
# =============================================================================


def _make_apply_fns(
    model: DenoisingTransformer,
) -> Tuple[
    Callable[..., jnp.ndarray],
    Callable[..., jnp.ndarray],
]:
    """Return ``(apply_inference, apply_train)`` closures over the model.

    These are the ``ModelApplyFn`` signatures expected by ``compute_loss``
    and ``sample_plan``.
    """

    def apply_inference(
        params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        return model.apply(params, obs, z_t, t)

    def apply_train(
        params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        return model.apply(params, obs, z_t, t, deterministic=False)

    return apply_inference, apply_train