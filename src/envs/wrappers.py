"""Project-specific Gymnax wrappers for the ReMDM planner."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct

from Craftax_Baselines.wrappers import GymnaxWrapper


# =============================================================================
# SequenceHistoryWrapper
# =============================================================================


@struct.dataclass
class SequenceHistoryState:
    env_state: Any
    obs_history: chex.Array  # [history_len, *obs_shape]
    act_history: chex.Array  # [history_len]  int32


class SequenceHistoryWrapper(GymnaxWrapper):
    """Augments env state with a sliding window of past observations and actions.

    After each step the histories satisfy::

        obs_history[-1]  = current observation
        act_history[i]   = action taken from obs_history[i] to reach obs_history[i+1]

    The wrapper returns the current observation unchanged; the sequence context is
    accessed via ``state.obs_history`` and ``state.act_history`` in the training loop.

    Place this as the **innermost** wrapper (before AutoReset / LogWrapper) so that
    episode boundaries trigger a proper history reset via the auto-reset mechanism.

    Args:
        env:          Single Gymnax environment.
        history_len:  Number of past timesteps to keep (including current).
        obs_shape:    Shape of a single observation, e.g. ``(obs_dim,)``.
    """

    def __init__(self, env: Any, history_len: int, obs_shape: Tuple[int, ...]) -> None:
        super().__init__(env)
        self.history_len = history_len
        self.obs_shape = obs_shape

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Any = None
    ) -> Tuple[chex.Array, SequenceHistoryState]:
        obs, env_state = self._env.reset(key, params)
        obs_history = jnp.tile(
            obs[None], [self.history_len] + [1] * len(self.obs_shape)
        )
        act_history = jnp.zeros(self.history_len, dtype=jnp.int32)
        state = SequenceHistoryState(
            env_state=env_state,
            obs_history=obs_history,
            act_history=act_history,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: SequenceHistoryState,
        action: Union[int, float],
        params: Any = None,
    ) -> Tuple[chex.Array, SequenceHistoryState, chex.Array, chex.Array, Any]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        act_history = jnp.roll(state.act_history, -1, axis=0).at[-1].set(action)
        obs_history = jnp.roll(state.obs_history, -1, axis=0).at[-1].set(obs)
        new_state = SequenceHistoryState(
            env_state=env_state,
            obs_history=obs_history,
            act_history=act_history,
        )
        return obs, new_state, reward, done, info


# =============================================================================
# DiscreteTokenizationWrapper
# =============================================================================


class DiscreteTokenizationWrapper(GymnaxWrapper):
    """Quantizes continuous observations into discrete token indices.

    Each observation element is mapped to one of ``n_bins`` integer tokens using
    uniform binning between ``obs_min`` and ``obs_max``.

    The returned observation dtype is int32 with values in ``[0, n_bins - 1]``.

    Args:
        env:      Gymnax environment (or wrapper).
        n_bins:   Number of discrete bins per observation element.
        obs_min:  Per-element lower bound, shape matching the observation.
        obs_max:  Per-element upper bound, shape matching the observation.
    """

    def __init__(
        self,
        env: Any,
        n_bins: int,
        obs_min: jnp.ndarray,
        obs_max: jnp.ndarray,
    ) -> None:
        super().__init__(env)
        self.n_bins = n_bins
        self.obs_min = obs_min
        self.obs_max = obs_max

    def _tokenize(self, obs: chex.Array) -> chex.Array:
        obs_clipped = jnp.clip(obs, self.obs_min, self.obs_max)
        normalized = (obs_clipped - self.obs_min) / (
            self.obs_max - self.obs_min + 1e-8
        )
        tokens = jnp.floor(normalized * self.n_bins).astype(jnp.int32)
        return jnp.clip(tokens, 0, self.n_bins - 1)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Any = None
    ) -> Tuple[chex.Array, Any]:
        obs, state = self._env.reset(key, params)
        return self._tokenize(obs), state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: Any,
        action: Union[int, float],
        params: Any = None,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Any]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self._tokenize(obs), state, reward, done, info


# =============================================================================
# PlannerWrapper
# =============================================================================


@struct.dataclass
class PlannerState:
    env_state: Any
    current_plan: chex.Array  # [num_envs, plan_horizon]  int32
    plan_step: int


class PlannerWrapper(GymnaxWrapper):
    """Manages the plan / replan cycle for a discrete diffusion planner.

    Expected wrapper stack (inner -> outer)::

        env  ->  LogWrapper  ->  AutoResetEnvWrapper  ->  BatchEnvWrapper
             ->  PlannerWrapper

    The ``planner_apply_fn`` must have the signature::

        fn(rng, model_params, obs) -> jnp.ndarray  # [num_envs, plan_horizon] int32

    Args:
        env:               Batched Gymnax environment (already handles num_envs).
        num_envs:          Number of parallel environments.
        plan_horizon:      Total number of actions the diffusion model outputs.
        replan_every:      Steps to execute before requesting a new plan (<= plan_horizon).
        planner_apply_fn:  Callable that invokes the diffusion model.
    """

    def __init__(
        self,
        env: Any,
        num_envs: int,
        plan_horizon: int,
        replan_every: int,
        planner_apply_fn: Callable[..., jnp.ndarray],
    ) -> None:
        super().__init__(env)
        if replan_every > plan_horizon:
            raise ValueError(
                f"replan_every ({replan_every}) must be <= plan_horizon ({plan_horizon})"
            )
        self.num_envs = num_envs
        self.plan_horizon = plan_horizon
        self.replan_every = replan_every
        self.planner_apply_fn = planner_apply_fn

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Any = None
    ) -> Tuple[chex.Array, PlannerState]:
        obs, env_state = self._env.reset(key, params)
        current_plan = jnp.zeros(
            (self.num_envs, self.plan_horizon), dtype=jnp.int32
        )
        state = PlannerState(
            env_state=env_state,
            current_plan=current_plan,
            plan_step=0,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: PlannerState,
        last_obs: chex.Array,
        model_params: Any,
        env_params: Any = None,
    ) -> Tuple[chex.Array, PlannerState, chex.Array, chex.Array, chex.Array, Any]:
        """Step the environment using the diffusion plan.

        Args:
            key:           PRNG key.
            state:         Current PlannerState.
            last_obs:      Most recent batched observation [num_envs, *obs_shape].
            model_params:  Parameters passed to planner_apply_fn.
            env_params:    Optional Gymnax environment params.

        Returns:
            (obs, state, action, reward, done, info)
        """
        key, plan_key, step_key = jax.random.split(key, 3)

        current_plan = jax.lax.cond(
            state.plan_step == 0,
            lambda operand: self.planner_apply_fn(*operand),
            lambda operand: state.current_plan,
            (plan_key, model_params, last_obs),
        )

        action = current_plan[:, state.plan_step]

        obs, env_state, reward, done, info = self._env.step(
            step_key, state.env_state, action, env_params
        )

        new_plan_step = (state.plan_step + 1) % self.replan_every
        new_state = PlannerState(
            env_state=env_state,
            current_plan=current_plan,
            plan_step=new_plan_step,
        )
        return obs, new_state, action, reward, done, info


# =============================================================================
# OfflineTrajectoryWrapper
# =============================================================================


@struct.dataclass
class TrajectoryBufferState:
    env_state: Any
    last_obs: Any          # [*obs_shape]
    buf_obs: Any           # [max_size, *obs_shape]
    buf_act: Any           # [max_size]  int32
    buf_reward: Any        # [max_size]  float32
    buf_done: Any          # [max_size]  bool
    buf_next_obs: Any      # [max_size, *obs_shape]
    write_idx: Any         # int32, wraps at max_size
    num_valid: Any         # int32, capped at max_size


class OfflineTrajectoryWrapper(GymnaxWrapper):
    """Accumulates transitions into a fixed-size circular replay buffer.

    The buffer overwrites the oldest entries once full.  Use ``sample_sequences``
    to draw contiguous subsequences for training a sequence model like ReMDM.

    Designed for a single environment; compose with ``BatchEnvWrapper`` *outside*
    this wrapper to collect from multiple envs simultaneously.

    Args:
        env:       Single Gymnax environment (or wrapper).
        max_size:  Maximum number of transitions to store.
        obs_shape: Shape of a single observation, e.g. ``(obs_dim,)``.
    """

    def __init__(
        self, env: Any, max_size: int, obs_shape: Tuple[int, ...]
    ) -> None:
        super().__init__(env)
        self.max_size = max_size
        self.obs_shape = obs_shape

    def _empty_buffer(
        self, env_state: Any, first_obs: chex.Array
    ) -> TrajectoryBufferState:
        return TrajectoryBufferState(
            env_state=env_state,
            last_obs=first_obs,
            buf_obs=jnp.zeros(
                (self.max_size, *self.obs_shape), dtype=jnp.float32
            ),
            buf_act=jnp.zeros(self.max_size, dtype=jnp.int32),
            buf_reward=jnp.zeros(self.max_size, dtype=jnp.float32),
            buf_done=jnp.zeros(self.max_size, dtype=jnp.bool_),
            buf_next_obs=jnp.zeros(
                (self.max_size, *self.obs_shape), dtype=jnp.float32
            ),
            write_idx=jnp.int32(0),
            num_valid=jnp.int32(0),
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, key: chex.PRNGKey, params: Any = None
    ) -> Tuple[chex.Array, TrajectoryBufferState]:
        obs, env_state = self._env.reset(key, params)
        state = self._empty_buffer(env_state, obs)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: TrajectoryBufferState,
        action: Union[int, float],
        params: Any = None,
    ) -> Tuple[chex.Array, TrajectoryBufferState, chex.Array, chex.Array, Any]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        idx = state.write_idx % self.max_size
        buf_obs = state.buf_obs.at[idx].set(state.last_obs)
        buf_act = state.buf_act.at[idx].set(action)
        buf_reward = state.buf_reward.at[idx].set(reward)
        buf_done = state.buf_done.at[idx].set(done)
        buf_next_obs = state.buf_next_obs.at[idx].set(obs)

        # Wrap write_idx at max_size to prevent unbounded growth / int32 overflow
        new_write_idx = (state.write_idx + 1) % self.max_size
        is_full = state.num_valid >= self.max_size
        new_num_valid = jnp.where(is_full, self.max_size, state.num_valid + 1)

        new_state = TrajectoryBufferState(
            env_state=env_state,
            last_obs=obs,
            buf_obs=buf_obs,
            buf_act=buf_act,
            buf_reward=buf_reward,
            buf_done=buf_done,
            buf_next_obs=buf_next_obs,
            write_idx=new_write_idx,
            num_valid=new_num_valid,
        )
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def sample_sequences(
        self,
        rng: chex.PRNGKey,
        state: TrajectoryBufferState,
        n_samples: int,
        seq_len: int,
    ) -> Tuple[
        chex.Array, chex.Array, chex.Array, chex.Array, chex.Array
    ]:
        """Sample ``n_samples`` contiguous subsequences of length ``seq_len``.

        Precondition: ``state.num_valid >= seq_len``.

        Returns:
            obs       [n_samples, seq_len, *obs_shape]
            act       [n_samples, seq_len]
            reward    [n_samples, seq_len]
            done      [n_samples, seq_len]
            next_obs  [n_samples, seq_len, *obs_shape]
        """
        max_start = jnp.maximum(state.num_valid - seq_len, 1)
        start_indices = jax.random.randint(
            rng, shape=(n_samples,), minval=0, maxval=max_start
        )

        def gather_seq(
            start_idx: jnp.ndarray,
        ) -> Tuple[
            chex.Array, chex.Array, chex.Array, chex.Array, chex.Array
        ]:
            indices = (start_idx + jnp.arange(seq_len)) % self.max_size
            return (
                state.buf_obs[indices],
                state.buf_act[indices],
                state.buf_reward[indices],
                state.buf_done[indices],
                state.buf_next_obs[indices],
            )

        return jax.vmap(gather_seq)(start_indices)
