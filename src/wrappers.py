import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial
from typing import Callable, Tuple, Union, Any


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class BatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info


class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree.map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree.map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


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

    After each step the histories satisfy:
        obs_history[-1]  = current observation
        act_history[i]   = action taken from obs_history[i] to reach obs_history[i+1]

    The wrapper returns the current observation unchanged; the sequence context is
    accessed via state.obs_history and state.act_history in the training loop.

    Place this as the innermost wrapper (before AutoReset / LogWrapper) so that
    episode boundaries trigger a proper history reset via the auto-reset mechanism.

    Args:
        env:          Single Gymnax environment.
        history_len:  Number of past timesteps to keep (including current).
        obs_shape:    Shape of a single observation, e.g. (obs_dim,) or (H, W, C).
    """

    def __init__(self, env, history_len: int, obs_shape: Tuple):
        super().__init__(env)
        self.history_len = history_len
        self.obs_shape = obs_shape

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        # Fill the entire history with the first observation so there are no
        # zero-padded "phantom" steps at the start of an episode.
        obs_history = jnp.tile(obs[None], [self.history_len] + [1] * len(self.obs_shape))
        act_history = jnp.zeros(self.history_len, dtype=jnp.int32)
        state = SequenceHistoryState(
            env_state=env_state,
            obs_history=obs_history,
            act_history=act_history,
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        # Shift left (drop oldest entry at index 0) and append the new value at -1.
        # act_history[i] records the action taken from obs_history[i], so we store
        # the current action before overwriting obs_history with the new observation.
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

    Each observation element is mapped to one of n_bins integer tokens using
    uniform binning between obs_min and obs_max.  Useful for feeding Craftax
    symbolic observations into a discrete diffusion model such as ReMDM.

    The returned observation dtype is int32 with values in [0, n_bins - 1].
    Dimensions that are already integer-valued (e.g. Craftax categorical features)
    work best when obs_min / obs_max are set to the true category bounds.

    Args:
        env:      Gymnax environment (or wrapper).
        n_bins:   Number of discrete bins per observation element.
        obs_min:  Per-element lower bound, shape matching the observation.
        obs_max:  Per-element upper bound, shape matching the observation.
    """

    def __init__(
        self,
        env,
        n_bins: int,
        obs_min: jnp.ndarray,
        obs_max: jnp.ndarray,
    ):
        super().__init__(env)
        self.n_bins = n_bins
        self.obs_min = obs_min
        self.obs_max = obs_max

    def _tokenize(self, obs: chex.Array) -> chex.Array:
        obs_clipped = jnp.clip(obs, self.obs_min, self.obs_max)
        # Map to [0, 1) then scale to [0, n_bins).
        normalized = (obs_clipped - self.obs_min) / (self.obs_max - self.obs_min + 1e-8)
        tokens = jnp.floor(normalized * self.n_bins).astype(jnp.int32)
        return jnp.clip(tokens, 0, self.n_bins - 1)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, state = self._env.reset(key, params)
        return self._tokenize(obs), state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key: chex.PRNGKey, state, action: Union[int, float], params=None):
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
    """Manages the plan / replan cycle for a discrete diffusion planner (e.g. ReMDM).

    Expected wrapper stack (inner → outer):
        env  →  SequenceHistoryWrapper  →  LogWrapper
             →  BatchEnvWrapper / OptimisticResetVecEnvWrapper
             →  PlannerWrapper

    The planner_apply_fn must have the signature:
        fn(rng, model_params, obs) -> jnp.ndarray  # [num_envs, plan_horizon] int32

    Args:
        env:               Batched Gymnax environment (already handles num_envs).
        num_envs:          Number of parallel environments.
        plan_horizon:      Total number of actions the diffusion model outputs.
        replan_every:      Steps to execute before requesting a new plan (≤ plan_horizon).
        planner_apply_fn:  Callable that invokes the diffusion model.
    """

    def __init__(
        self,
        env,
        num_envs: int,
        plan_horizon: int,
        replan_every: int,
        planner_apply_fn: Callable,
    ):
        super().__init__(env)
        assert replan_every <= plan_horizon, "replan_every must be <= plan_horizon"
        self.num_envs = num_envs
        self.plan_horizon = plan_horizon
        self.replan_every = replan_every
        self.planner_apply_fn = planner_apply_fn

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        current_plan = jnp.zeros((self.num_envs, self.plan_horizon), dtype=jnp.int32)
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
        env_params=None,
    ) -> Tuple:
        """Step the environment using the diffusion plan.

        Args:
            key:           PRNG key.
            state:         Current PlannerState.
            last_obs:      Most recent batched observation [num_envs, *obs_shape].
            model_params:  Parameters passed to planner_apply_fn.
            env_params:    Optional Gymnax environment params.

        Returns:
            obs, state, action, reward, done, info
        """
        key, plan_key, step_key = jax.random.split(key, 3)

        current_plan = jax.lax.cond(
            state.plan_step == 0,
            lambda operand: self.planner_apply_fn(*operand),
            lambda _: state.current_plan,
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
    last_obs: Any   # [*obs_shape]
    buf_obs: Any    # [max_size, *obs_shape]
    buf_act: Any    # [max_size]  int32
    buf_reward: Any # [max_size]  float32
    buf_done: Any   # [max_size]  bool
    buf_next_obs: Any  # [max_size, *obs_shape]
    write_idx: Any
    num_valid: Any


class OfflineTrajectoryWrapper(GymnaxWrapper):
    """Accumulates (obs, action, reward, done, next_obs) transitions into a
    fixed-size circular replay buffer stored inside the JAX state.

    The buffer overwrites the oldest entries once full.  Use sample_sequences()
    to draw contiguous subsequences for training a sequence model like ReMDM.

    Designed for a single environment; compose with BatchEnvWrapper *outside*
    this wrapper to collect from multiple envs simultaneously (each env carries
    its own independent buffer in the vmapped state).

    Args:
        env:       Single Gymnax environment (or wrapper).
        max_size:  Maximum number of transitions to store.
        obs_shape: Shape of a single observation, e.g. (obs_dim,).
    """

    def __init__(self, env, max_size: int, obs_shape: Tuple):
        super().__init__(env)
        self.max_size = max_size
        self.obs_shape = obs_shape

    def _empty_buffer(self, env_state: Any, first_obs: chex.Array) -> TrajectoryBufferState:
        return TrajectoryBufferState(
            env_state=env_state,
            last_obs=first_obs,
            buf_obs=jnp.zeros((self.max_size, *self.obs_shape), dtype=jnp.float32),
            buf_act=jnp.zeros(self.max_size, dtype=jnp.int32),
            buf_reward=jnp.zeros(self.max_size, dtype=jnp.float32),
            buf_done=jnp.zeros(self.max_size, dtype=jnp.bool_),
            buf_next_obs=jnp.zeros((self.max_size, *self.obs_shape), dtype=jnp.float32),
            write_idx=0,
            num_valid=0,
        )

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None) -> Tuple[chex.Array, TrajectoryBufferState]:
        obs, env_state = self._env.reset(key, params)
        state = self._empty_buffer(env_state, obs)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state: TrajectoryBufferState,
        action: Union[int, float],
        params=None,
    ) -> Tuple:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        idx = state.write_idx % self.max_size
        buf_obs      = state.buf_obs.at[idx].set(state.last_obs)
        buf_act      = state.buf_act.at[idx].set(action)
        buf_reward   = state.buf_reward.at[idx].set(reward)
        buf_done     = state.buf_done.at[idx].set(done)
        buf_next_obs = state.buf_next_obs.at[idx].set(obs)

        new_write_idx = state.write_idx + 1
        new_state = TrajectoryBufferState(
            env_state=env_state,
            last_obs=obs,
            buf_obs=buf_obs,
            buf_act=buf_act,
            buf_reward=buf_reward,
            buf_done=buf_done,
            buf_next_obs=buf_next_obs,
            write_idx=new_write_idx,
            num_valid=jnp.minimum(new_write_idx, self.max_size),
        )
        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def sample_sequences(
        self,
        rng: chex.PRNGKey,
        state: TrajectoryBufferState,
        n_samples: int,
        seq_len: int,
    ) -> Tuple:
        """Sample n_samples contiguous subsequences of length seq_len.

        Precondition: state.num_valid >= seq_len.

        Sequences may span episode boundaries; use the returned `done` array to
        mask out invalid cross-episode transitions during loss computation.

        Returns:
            obs       [n_samples, seq_len, *obs_shape]
            act       [n_samples, seq_len]
            reward    [n_samples, seq_len]
            done      [n_samples, seq_len]
            next_obs  [n_samples, seq_len, *obs_shape]
        """
        # Draw random start positions within the valid region of the buffer.
        max_start = jnp.maximum(state.num_valid - seq_len, 1)
        start_indices = jax.random.randint(
            rng, shape=(n_samples,), minval=0, maxval=max_start
        )

        def gather_seq(start_idx):
            indices = (start_idx + jnp.arange(seq_len)) % self.max_size
            return (
                state.buf_obs[indices],
                state.buf_act[indices],
                state.buf_reward[indices],
                state.buf_done[indices],
                state.buf_next_obs[indices],
            )

        return jax.vmap(gather_seq)(start_indices)
