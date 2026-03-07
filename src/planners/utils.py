from __future__ import annotations

import os
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

    # 1. Evaluate shape without allocating memory
    def get_abstract_state():
        rng = jax.random.PRNGKey(0)
        params = _init_model_params(model, rng, obs_dim, config["PLAN_HORIZON"])
        tx = optax.adam(config["LR"])
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    abstract_ts = jax.eval_shape(get_abstract_state)
    restored_ts = _restore_train_state(checkpoint_path, abstract_ts)

    return restored_ts.params


class PPOAgent:
    """Unified wrapper for any Craftax_Baselines PPO policy.

    Normalises the apply interface across ``ppo`` (ActorCritic MLP),
    ``ppo_rnd`` (ActorCriticRND dual-critic MLP) and ``ppo_rnn``
    (ActorCriticRNN GRU-based) checkpoints so that collection code does
    not need to branch on the architecture.
    """

    def __init__(
        self,
        network: Any,
        params: Any,
        model_type: str,
        layer_size: int,
    ) -> None:
        self.network = network
        self.params = params
        self.model_type = model_type
        self.layer_size = layer_size

    def init_hidden(self, num_envs: int) -> Optional[jax.Array]:
        """Return an initial hidden state.  ``None`` for MLP-based models."""
        if self.model_type == "ppo_rnn":
            from Craftax_Baselines.ppo_rnn import ScannedRNN
            return ScannedRNN.initialize_carry(num_envs, self.layer_size)
        return None

    def apply(
        self,
        params: Any,
        obs: jax.Array,
        hidden: Optional[jax.Array] = None,
        done: Optional[jax.Array] = None,
    ) -> Tuple[Any, jax.Array, Optional[jax.Array]]:
        """Apply the policy network.

        Returns:
            ``(pi, value, new_hidden)`` where ``new_hidden`` is ``None`` for
            MLP models.  For ``ppo_rnn``, ``hidden`` and ``done`` are required.
        """
        if self.model_type == "ppo_rnn":
            # ActorCriticRNN.apply(params, hidden, (obs[T,B,D], done[T,B]))
            assert hidden is not None and done is not None, (
                "hidden and done must be provided for ppo_rnn"
            )
            ac_in = (obs[jnp.newaxis], done[jnp.newaxis])
            new_hidden, pi, value = self.network.apply(params, hidden, ac_in)
            return pi, value.squeeze(0), new_hidden
        else:
            result = self.network.apply(params, obs)
            return result[0], result[1], None


def _detect_ppo_model_type(checkpoint_path: str) -> str:
    """Detect PPO model architecture by inspecting checkpoint directory contents.

    Strategy (in order):
    1. Walk the checkpoint directory tree and look for characteristic
       layer names written by ``orbax.checkpoint.PyTreeCheckpointer``
       (e.g. ``ScannedRNN_0`` for RNN, ``Dense_8`` for RND dual-critic).
    2. Fall back to substring matching on the path string itself.
    3. Default to ``"ppo"`` (plain MLP ActorCritic).
    """
    try:
        step_dirs = sorted(
            d for d in os.listdir(checkpoint_path) if d.isdigit()
        )
        if step_dirs:
            latest_dir = os.path.join(checkpoint_path, step_dirs[-1])
            for root, dirs, files in os.walk(latest_dir):
                for name in dirs + files:
                    if "ScannedRNN" in name:
                        return "ppo_rnn"
            for root, dirs, files in os.walk(latest_dir):
                for name in dirs + files:
                    if "Dense_8" in name or "Dense_9" in name:
                        return "ppo_rnd"
    except OSError:
        pass

    # Path-name fallback
    path_lower = checkpoint_path.lower()
    if "rnn" in path_lower:
        return "ppo_rnn"
    if "rnd" in path_lower:
        return "ppo_rnd"
    return "ppo"


def _load_ppo_checkpoint(
        ppo_checkpoint_path: str,
        num_actions: Sequence[int],
        obs_dim: int,
        layer_size: int,
        model_type_override: Optional[str] = None,
) -> PPOAgent:
    """Load a Craftax_Baselines PPO checkpoint, auto-detecting the architecture.

    All three baselines (``ppo.py``, ``ppo_rnn.py``, ``ppo_rnd.py``) save
    checkpoints with the legacy ``orbax.checkpoint.PyTreeCheckpointer`` API.
    This function uses that same API for restoration and builds the matching
    Flax network based on the detected (or overridden) model type.

    Args:
        ppo_checkpoint_path: Directory created by ``CheckpointManager.save``.
        num_actions:         Number of discrete actions in the environment.
        obs_dim:             Flat observation dimension.
        layer_size:          Hidden-layer width of the policy network.
        model_type_override: Explicit architecture name (``"ppo"``,
                             ``"ppo_rnn"``, or ``"ppo_rnd"``).  When
                             ``None``, the type is detected automatically.

    Returns:
        A :class:`PPOAgent` with ``network``, ``params``, ``model_type``,
        and ``layer_size`` attributes.
    """
    model_type = model_type_override or _detect_ppo_model_type(ppo_checkpoint_path)

    # Build the matching network and dummy params for the restoration template.
    if model_type == "ppo_rnn":
        from Craftax_Baselines.ppo_rnn import ActorCriticRNN, ScannedRNN
        network = ActorCriticRNN(num_actions, config={"LAYER_SIZE": layer_size})
        dummy_hidden = ScannedRNN.initialize_carry(1, layer_size)
        dummy_obs = jnp.zeros((1, 1, obs_dim))   # [T=1, B=1, obs_dim]
        dummy_done = jnp.zeros((1, 1))            # [T=1, B=1]
        dummy_params = network.init(
            jax.random.PRNGKey(0), dummy_hidden, (dummy_obs, dummy_done)
        )
    elif model_type == "ppo_rnd":
        from Craftax_Baselines.models.rnd import ActorCriticRND
        network = ActorCriticRND(num_actions, layer_size)
        dummy_params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim)))
    else:
        from Craftax_Baselines.models.actor_critic import ActorCritic
        network = ActorCritic(num_actions, layer_size)
        dummy_params = network.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim)))

    # The baselines use optax.chain(clip, adam) — reproduce the chain structure
    # so the opt_state pytree shape matches and PyTreeCheckpointer.restore works.
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(2e-4, eps=1e-5),
    )
    dummy_ts = TrainState.create(apply_fn=network.apply, params=dummy_params, tx=tx)

    # All three baselines save with the legacy PyTreeCheckpointer API.
    # Use ocp.PyTreeCheckpointer directly on the step directory to avoid
    # the deprecated CheckpointManager(path, checkpointer, options) signature.
    try:
        step_dirs = sorted(
            int(d) for d in os.listdir(ppo_checkpoint_path) if d.isdigit()
        )
    except OSError as exc:
        raise FileNotFoundError(
            f"No PPO checkpoint found at '{ppo_checkpoint_path}'"
        ) from exc

    if not step_dirs:
        raise FileNotFoundError(
            f"No PPO checkpoint found at '{ppo_checkpoint_path}'"
        )

    latest_step = step_dirs[-1]
    ckpt_dir = os.path.join(ppo_checkpoint_path, str(latest_step))

    # ocp.PyTreeCheckpointer is the legacy checkpointer used by the baselines.
    checkpointer = ocp.PyTreeCheckpointer()  # type: ignore[attr-defined]
    restored_ts = checkpointer.restore(ckpt_dir, item=dummy_ts)

    print(
        f"Loaded {model_type} PPO checkpoint from '{ppo_checkpoint_path}'"
        f" (step={latest_step})"
    )
    return PPOAgent(
        network=network,
        params=restored_ts.params,
        model_type=model_type,
        layer_size=layer_size,
    )


def _save_model(
        train_state: TrainState, config: Dict[str, Any], dir_name: str
) -> None:
    """Save a TrainState checkpoint using orbax."""

    if config.get("USE_WANDB") and wandb.run is not None:
        path = str(pathlib.Path(wandb.run.dir) / dir_name)
    else:
        path = dir_name

    checkpointer = ocp.StandardCheckpointer()
    ckpt_mgr = ocp.CheckpointManager(
        path,
        checkpointer,
        options=ocp.CheckpointManagerOptions(max_to_keep=1, create=True),
    )

    step = config.get("NUM_TRAIN_STEPS", config.get("NUM_UPDATES", 0))
    ckpt_mgr.save(
        step,
        args=ocp.args.StandardSave(train_state),
    )

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