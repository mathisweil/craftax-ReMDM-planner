from __future__ import annotations

import os
import pathlib
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
from orbax.checkpoint.checkpoint_managers import (
    preservation_policy as preservation_policy_lib,
    save_decision_policy as save_decision_policy_lib,
)

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

SCHEDULE_MAP: dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# =============================================================================
# Model helpers
# =============================================================================


def _build_model(config: dict[str, Any], num_actions: int) -> DenoisingTransformer:
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


def _make_ckpt_manager(
    path: str,
    max_to_keep: int = 1,
    save_interval: int = 1,
) -> ocp.CheckpointManager:
    """Construct a CheckpointManager with current (non-deprecated) options."""
    return ocp.CheckpointManager(
        path,
        options=ocp.CheckpointManagerOptions(
            save_decision_policy=save_decision_policy_lib.FixedIntervalPolicy(
                save_interval
            ),
            preservation_policy=preservation_policy_lib.LatestN(max_to_keep),
            enable_async_checkpointing=True,
            enable_background_delete=True,
        ),
    )


def _restore_train_state(checkpoint_path: str, abstract_ts: TrainState) -> TrainState:
    """Restore a TrainState from an Orbax checkpoint."""
    with _make_ckpt_manager(checkpoint_path) as ckpt_mgr:
        latest_step = ckpt_mgr.latest_step()
        if latest_step is None:
            raise FileNotFoundError(
                f"No valid checkpoint found at '{checkpoint_path}'"
            )
        restored_ts = ckpt_mgr.restore(
            latest_step,
            args=ocp.args.StandardRestore(item=abstract_ts),
        )

    print(f"Loaded checkpoint from '{checkpoint_path}' (step={latest_step})")
    return restored_ts


def _load_checkpoint(
        config: dict[str, Any],
        model: Any,  # DenoisingTransformer
        obs_dim: int,
        checkpoint_path: str,
) -> Any:
    """Load model parameters from an Orbax checkpoint (outside JIT)."""
    def get_abstract_state():
        params = _init_model_params(
            model, jax.random.PRNGKey(0), obs_dim, config["PLAN_HORIZON"]
        )
        return _create_train_state(
            model, params, config["LR"], config["MAX_GRAD_NORM"]
        )

    abstract_ts = jax.eval_shape(get_abstract_state)
    return _restore_train_state(checkpoint_path, abstract_ts).params


class PPOAgent:
    """Unified wrapper for any Craftax_Baselines PPO policy.

    Normalises the apply interface across ``ppo`` (ActorCritic MLP),
    ``ppo_rnd`` (ActorCriticRND dual-critic MLP), and ``ppo_rnn``
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
        """Return an initial hidden state. ``None`` for MLP-based models."""
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
    ) -> tuple[Any, jax.Array, Optional[jax.Array]]:
        """Apply the policy network.

        Returns:
            ``(pi, value, new_hidden)`` where ``new_hidden`` is ``None`` for
            MLP models. For ``ppo_rnn``, ``hidden`` and ``done`` are required.
        """
        if self.model_type == "ppo_rnn":
            assert hidden is not None and done is not None, (
                "hidden and done must be provided for ppo_rnn"
            )
            new_hidden, pi, value = self.network.apply(
                params, hidden, (obs[jnp.newaxis], done[jnp.newaxis])
            )
            return pi, value.squeeze(0), new_hidden

        if self.model_type == "ppo_rnd":
            pi, value_e, _value_i = self.network.apply(params, obs)
            return pi, value_e, None

        pi, value = self.network.apply(params, obs)
        return pi, value, None


def _detect_ppo_model_type(checkpoint_path: str) -> str:
    """Detect PPO model architecture by inspecting the checkpoint directory.

    Strategy (in order):
    1. Walk the latest step directory, collecting all entry names in a single
       pass and checking for characteristic subtree names written by
       StandardCheckpointer (``ScannedRNN_0`` for RNN, ``Dense_8``/``Dense_9``
       for the RND dual-critic).
    2. Fall back to substring matching on the path string itself.
    3. Default to ``"ppo"`` (plain MLP ActorCritic).
    """
    try:
        step_dirs = sorted(d for d in os.listdir(checkpoint_path) if d.isdigit())
        if step_dirs:
            latest_dir = os.path.join(checkpoint_path, step_dirs[-1])
            all_names: set[str] = set()
            for _root, dirs, files in os.walk(latest_dir):
                all_names.update(dirs)
                all_names.update(files)
            if any("ScannedRNN" in name for name in all_names):
                return "ppo_rnn"
            if any("Dense_8" in name or "Dense_9" in name for name in all_names):
                return "ppo_rnd"
    except OSError:
        pass

    path_lower = checkpoint_path.lower()
    if "rnn" in path_lower:
        return "ppo_rnn"
    if "rnd" in path_lower:
        return "ppo_rnd"
    return "ppo"


def _build_ppo_network(
        model_type: str,
        num_actions: int,
        obs_dim: int,
        layer_size: int,
) -> tuple[Any, Any]:
    """Instantiate the correct network and return abstract params only.

    Returns:
        ``(network, abstract_params)`` — raw params pytree (no opt_state).
        ``partial_restore=True`` in the restore call handles the rest.
    """
    if model_type == "ppo_rnn":
        import sys
        import os
        
        # Inject the baselines folder into the path
        baselines_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Craftax_Baselines"))
        if baselines_path not in sys.path:
            sys.path.insert(0, baselines_path)
            
        # Import directly without the Craftax_Baselines prefix!
        from ppo_rnn import ActorCriticRNN, ScannedRNN
        #from Craftax_Baselines.ppo_rnn import ActorCriticRNN, ScannedRNN
        network = ActorCriticRNN(num_actions, config={"LAYER_SIZE": layer_size})
        def _init_params():
            dummy_hidden = ScannedRNN.initialize_carry(1, layer_size)
            return network.init(
                jax.random.PRNGKey(0), dummy_hidden,
                (jnp.zeros((1, 1, obs_dim)), jnp.zeros((1, 1))),
            )
    elif model_type == "ppo_rnd":
        from Craftax_Baselines.models.rnd import ActorCriticRND
        network = ActorCriticRND(num_actions, layer_size)
        def _init_params():
            return network.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim)))
    else:
        from Craftax_Baselines.models.actor_critic import ActorCritic
        network = ActorCritic(num_actions, layer_size)
        def _init_params():
            return network.init(jax.random.PRNGKey(0), jnp.zeros((1, obs_dim)))

    abstract_params = jax.eval_shape(_init_params)
    return network, abstract_params


def _load_ppo_checkpoint(
        ppo_checkpoint_path: str,
        num_actions: int,
        obs_dim: int,
        layer_size: int,
        model_type: Optional[str] = None,
) -> PPOAgent:
    model_type = model_type or _detect_ppo_model_type(ppo_checkpoint_path)
    network, abstract_params = _build_ppo_network(
        model_type, num_actions, obs_dim, layer_size
    )

    with _make_ckpt_manager(ppo_checkpoint_path) as ckpt_mgr:
        latest_step = ckpt_mgr.latest_step()
        if latest_step is None:
            raise FileNotFoundError(
                f"No PPO checkpoint found at '{ppo_checkpoint_path}'"
            )
        restored = ckpt_mgr.restore(
            latest_step,
            args=ocp.args.PyTreeRestore(
                item={"params": abstract_params},
                partial_restore=True,
            ),
        )

    print(f"Loaded {model_type} PPO checkpoint from '{ppo_checkpoint_path}'")
    return PPOAgent(
        network=network,
        params=restored["params"],
        model_type=model_type,
        layer_size=layer_size,
    )


def _resolve_ckpt_dir(config: dict[str, Any], subdir: str = "checkpoints") -> pathlib.Path:
    base = (
        pathlib.Path(wandb.run.dir)
        if config["USE_WANDB"] and wandb.run is not None
        else pathlib.Path(config["CKPT_DIR"])
    )
    path = base / config["ENV_NAME"] / subdir
    path.mkdir(parents=True, exist_ok=True)
    return path



def _make_periodic_ckpt_manager(config: dict[str, Any], subdir: str = "checkpoints") -> ocp.CheckpointManager:
    return ocp.CheckpointManager(
        _resolve_ckpt_dir(config, subdir),
        options=ocp.CheckpointManagerOptions(
            save_decision_policy=save_decision_policy_lib.FixedIntervalPolicy(config["CKPT_EVERY_STEPS"]),
            preservation_policy=preservation_policy_lib.LatestN(config["CKPT_MAX_TO_KEEP"]),
            enable_async_checkpointing=True,
            enable_background_delete=True,
        ),
    )


# =============================================================================
# Environment stack construction
# =============================================================================


def _make_env_stack(
    config: dict[str, Any],
    num_envs: int,
    *,
    use_optimistic_resets: bool = False,
    use_sequence_history: bool = False,
) -> tuple[Any, Any]:
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
        obs_shape: tuple[int, ...] = env.observation_space(env_params).shape
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
) -> Optional[tuple[jnp.ndarray, jnp.ndarray]]:
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
    offsets = np.arange(plan_horizon)[None, :] + sel_t[:, None]  # [batch, H]
    act_batch = jnp.array(chunk_acts[sel_e[:, None], offsets])
    return obs_batch, act_batch


# =============================================================================
# Model apply helpers (deterministic vs training)
# =============================================================================


def _make_apply_fns(
    model: DenoisingTransformer,
) -> tuple[
    Callable[..., jnp.ndarray],
    Callable[..., jnp.ndarray],
]:
    """Return ``(apply_inference, apply_train)`` closures over the model.

    These are the ``ModelApplyFn`` signatures expected by ``compute_loss``
    and ``sample_plan``.
    """

    def apply_inference(
        params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray,
        _rng: Optional[Any] = None,
    ) -> jnp.ndarray:
        return model.apply(params, obs, z_t, t)

    def apply_train(
        params: Any, obs: jnp.ndarray, z_t: jnp.ndarray, t: jnp.ndarray,
        rng: Optional[Any] = None,
    ) -> jnp.ndarray:
        return model.apply(
            params, obs, z_t, t, deterministic=False,
            rngs={"dropout": rng} if rng is not None else {},
        )

    return apply_inference, apply_train