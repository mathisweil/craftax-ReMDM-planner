"""Core training loop for RL fine-tuning ablations (JAX-compiled).

``make_run_ablation`` is a factory mirroring ``src/planners/train.make_train``:
it pre-computes all static setup outside the returned closure, which runs
entirely inside ``jax.lax.scan`` with no Python-level loops.

The returned closure is safe for ``jax.jit`` and ``jax.vmap``.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from experiments.rl_finetuning.ablations.losses import (
    LossContext,
    estimate_fisher_diagonal,
    make_loss_baseline,
    make_loss_t_curriculum_jit,
)
from experiments.rl_finetuning.ablations.optimizers import (
    apply_fn_with_lora,
    gradient_surgery,
    make_lora_params,
    make_optimizer_lora_only,
)
from experiments.rl_finetuning.ablations.registry import AblationSpec
from experiments.rl_finetuning.diagnostics.gradient import (
    compute_per_layer_grad_norms_jax,
    compute_surgery_metrics_jax,
    make_grad_alignment_fn,
)
from experiments.rl_finetuning.diagnostics.representation import (
    make_cka_fn,
    make_repr_drift_fn,
)
from experiments.rl_finetuning.diagnostics.timestep import make_t_analysis_fn

logger = logging.getLogger(__name__)

_EPS: float = 1e-5


@dataclass
class AblationHistory:
    """Typed training history for a single ablation run.

    All list fields are appended to at the corresponding logging frequency.
    JSON-serialisable via ``to_dict()`` / ``from_dict()``.
    """

    # Training loss (logged every 10 iterations)
    iters: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)

    # Environment score (online rollout; logged every 10 iterations)
    env_score_iters: list[int] = field(default_factory=list)
    env_score: list[float] = field(default_factory=list)

    # Eval score (logged every eval_every iterations)
    eval_iters: list[int] = field(default_factory=list)
    eval_score: list[float] = field(default_factory=list)

    # Gradient alignment (logged every grad_align_every iterations)
    grad_align_iters: list[int] = field(default_factory=list)
    grad_align: list[float] = field(default_factory=list)
    rl_grad_norm: list[float] = field(default_factory=list)
    bc_grad_norm: list[float] = field(default_factory=list)

    # Per-layer gradient norms (logged every per_layer_every iterations)
    per_layer_iters: list[int] = field(default_factory=list)
    per_layer_norms: list[dict[str, float]] = field(default_factory=list)

    # Representation drift (logged every repr_drift_every iterations)
    repr_drift_iters: list[int] = field(default_factory=list)
    repr_drift_kl: list[float] = field(default_factory=list)
    repr_drift_kl_low_t: list[float] = field(default_factory=list)
    repr_drift_kl_mid_t: list[float] = field(default_factory=list)
    repr_drift_kl_high_t: list[float] = field(default_factory=list)

    # CKA similarity (logged every cka_every iterations)
    cka_iters: list[int] = field(default_factory=list)
    cka_similarity: list[float] = field(default_factory=list)

    # t-distribution analysis (logged every t_analysis_every iterations)
    t_analysis_iters: list[int] = field(default_factory=list)
    norm_low_t: list[float] = field(default_factory=list)
    norm_high_t: list[float] = field(default_factory=list)
    lowhigh_cos: list[float] = field(default_factory=list)
    t_bin_norms: list[dict[str, float]] = field(default_factory=list)

    # Return / advantage distributions (logged every 10 iterations)
    win_rate: list[float] = field(default_factory=list)
    effective_batch_size: list[float] = field(default_factory=list)

    # Gradient surgery metrics (when gradient_surgery=True)
    surgery_iters: list[int] = field(default_factory=list)
    surgery_fraction: list[float] = field(default_factory=list)
    surgery_n_conflicting: list[int] = field(default_factory=list)

    # Per-achievement unlock rates (one entry per eval checkpoint).
    per_achievement_rates: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict.

        Returns:
            Dict with all list fields preserved.
        """
        return {k: list(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> AblationHistory:
        """Reconstruct from a dict (e.g., loaded from JSON).

        Args:
            d: Dict produced by ``to_dict()``.

        Returns:
            ``AblationHistory`` instance.
        """
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class ReplayBuffer(NamedTuple):
    """Fixed-size ring buffer for mixed replay, pre-allocated as JAX arrays.

    Fields:
        acts:      ``[buf_size, plan_horizon]`` action sequences.
        obs:       ``[buf_size, obs_dim]`` observations.
        valid:     ``[buf_size]`` validity flags.
        returns:   ``[buf_size]`` return values.
        write_idx: Next write position (wraps around).
        count:     Number of entries written so far (capped at buf_size).
    """

    acts: jax.Array
    obs: jax.Array
    valid: jax.Array
    returns: jax.Array
    write_idx: jax.Array
    count: jax.Array


class AblationCarry(NamedTuple):
    """Scan carry for the main training loop.

    Fields:
        state:       Flax TrainState.
        ema_params:  Exponential moving average of model parameters (for eval).
        env_state:   Gymnax environment state.
        obs:         ``[num_envs, obs_dim]`` current observations.
        done:        ``[num_envs]`` episode-done flags.
        hstate:      PPO recurrent hidden state.
        rng:         PRNG key.
        step_idx:    Current iteration (1-based).
        running_mean: EMA running mean for advantage normalisation.
        running_std:  EMA running std for advantage normalisation.
        replay_buf:  Fixed-size replay buffer (for mixed replay).
        reward_model_state: TrainState for the learned reward model (or None-like dummy).
    """

    state: TrainState
    ema_params: Any
    env_state: Any
    obs: jax.Array
    done: jax.Array
    hstate: Any
    rng: jax.Array
    step_idx: jax.Array
    running_mean: jax.Array
    running_std: jax.Array
    replay_buf: ReplayBuffer
    reward_model_state: TrainState


class StepMetrics(NamedTuple):
    """Per-step output from the training scan, all JAX arrays.

    Fields have shape ``[]`` (scalars) or small fixed-size arrays.
    Stacked over the scan, they become ``[num_updates, ...]``.
    """

    loss: jax.Array
    env_score: jax.Array
    win_rate: jax.Array
    eff_batch_size: jax.Array
    # Gradient alignment
    cos_sim: jax.Array
    rl_grad_norm: jax.Array
    bc_grad_norm: jax.Array
    # Representation drift
    kl_mean: jax.Array
    kl_low_t: jax.Array
    kl_mid_t: jax.Array
    kl_high_t: jax.Array
    # CKA
    cka: jax.Array
    # t-analysis
    t_bin_norms: jax.Array  # [n_bins]
    low_high_cos: jax.Array
    t_norm_low: jax.Array
    t_norm_high: jax.Array
    # Surgery
    surgery_frac: jax.Array
    surgery_n_conflict: jax.Array
    # Per-layer norms
    per_layer_norms: jax.Array  # [num_leaves]
    # Eval
    eval_score: jax.Array
    # Flags: which diagnostics actually ran this step
    did_eval: jax.Array
    did_grad_align: jax.Array
    did_repr_drift: jax.Array
    did_cka: jax.Array
    did_t_analysis: jax.Array
    did_per_layer: jax.Array
    did_surgery: jax.Array
    did_log: jax.Array


def _build_reward_model(
    obs_dim: int,
    rng: jax.Array,
    width: int = 64,
    depth: int = 2,
    lr: float = 1e-3,
) -> tuple[Any, TrainState]:
    """Build a simple MLP reward model and return its TrainState.

    Args:
        obs_dim: Input observation dimensionality.
        rng:     PRNG key for parameter initialisation.
        width:   Hidden layer width.
        depth:   Number of hidden layers.
        lr:      Learning rate.

    Returns:
        Tuple of (flax_module, initial_train_state).
    """
    import flax.linen as nn

    class RewardMLP(nn.Module):
        """Lightweight reward predictor."""

        width: int
        depth: int

        @nn.compact
        def __call__(self, x: jax.Array) -> jax.Array:
            """Forward pass: obs -> predicted return.

            Args:
                x: ``[B, obs_dim]`` observations.

            Returns:
                ``[B]`` predicted returns.
            """
            for _ in range(self.depth):
                x = nn.Dense(self.width)(x)
                x = nn.relu(x)
            return nn.Dense(1)(x).squeeze(-1)

    net = RewardMLP(width=width, depth=depth)
    dummy_params = net.init(rng, jnp.zeros((1, obs_dim)))
    tx = optax.adam(lr)
    rm_state = TrainState.create(
        apply_fn=net.apply,
        params=dummy_params,
        tx=tx,
    )
    return net, rm_state


def _reward_model_train_step(
    rm_state: TrainState,
    obs: jax.Array,
    targets: jax.Array,
) -> tuple[TrainState, jax.Array]:
    """One gradient step on MSE loss for the reward model.

    Args:
        rm_state: Current reward model TrainState.
        obs:      ``[B, obs_dim]`` observations.
        targets:  ``[B]`` return targets.

    Returns:
        Tuple of (updated_state, scalar_loss).
    """

    def loss_fn(params: Any) -> jax.Array:
        preds = rm_state.apply_fn(params, obs)
        return jnp.mean((preds - targets) ** 2)

    loss_val, grads = jax.value_and_grad(loss_fn)(rm_state.params)
    rm_state = rm_state.apply_gradients(grads=grads)
    return rm_state, loss_val


def _compute_advantages(
    flat_returns: jax.Array,
    floor: float,
    cap: float,
    wins_only: bool,
    win_thresh: float,
    use_running_stats: bool,
    ema_decay: float,
    running_mean: jax.Array,
    running_std: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute per-sample advantage weights (pure JAX, JIT-safe).

    Args:
        flat_returns:      ``[N]`` raw returns.
        floor:             Lower clip bound.
        cap:               Upper clip bound.
        wins_only:         Binary win mask mode.
        win_thresh:        Threshold for win detection.
        use_running_stats: Whether to use EMA normalisation.
        ema_decay:         EMA decay factor.
        running_mean:      Current EMA mean.
        running_std:       Current EMA std.

    Returns:
        Tuple of (advantages ``[N]``, updated_mean, updated_std).
    """
    clipped = jnp.clip(flat_returns, 0.0, None)
    batch_mean = jnp.mean(clipped)
    batch_std = jnp.std(clipped) + 1e-8

    # Branch 1: wins_only
    adv_wins = (flat_returns > win_thresh).astype(jnp.float32)

    # Branch 2: running stats
    new_mean = ema_decay * running_mean + (1.0 - ema_decay) * batch_mean
    new_std = ema_decay * running_std + (1.0 - ema_decay) * batch_std
    adv_running = jnp.clip((clipped - new_mean) / new_std + 1.0, floor, cap)

    # Branch 3: standard normalisation
    weights = clipped / (batch_mean + _EPS)
    adv_standard = jnp.clip(weights, floor, cap)

    # Select branch without Python control flow
    adv = jnp.where(
        wins_only,
        adv_wins,
        jnp.where(use_running_stats, adv_running, adv_standard),
    )
    out_mean = jnp.where(use_running_stats, new_mean, batch_mean)
    out_std = jnp.where(use_running_stats, new_std, batch_std)
    return adv, out_mean, out_std


def _effective_batch_size(advantages: jax.Array) -> jax.Array:
    """Compute effective batch size: (sum w)^2 / sum w^2.

    Args:
        advantages: ``[N]`` advantage weights.

    Returns:
        Effective batch size as a JAX scalar.
    """
    sum_w = jnp.sum(advantages)
    sum_w2 = jnp.sum(advantages**2)
    return sum_w**2 / jnp.maximum(sum_w2, 1e-10)


def build_rollout_fn(
    env: Any,
    env_params: Any,
    ppo: Any,
    config: dict,
    obs_dim: int,
) -> Callable:
    """Build a JIT-compiled rollout function.

    Args:
        env:        Wrapped Craftax environment.
        env_params: Environment parameters.
        ppo:        PPOAgent instance.
        config:     UPPERCASE config dict.
        obs_dim:    Observation dimensionality.

    Returns:
        JIT-compiled ``collect_rollout(env_state, obs, done, hstate, rng)``
        returning ``(env_state, obs, done, hstate, rng, flat_obs,
        flat_acts, flat_valid, flat_returns, env_score_dict)``.
    """
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    plan_horizon = config["PLAN_HORIZON"]
    collect_temperature = config.get("COLLECT_TEMPERATURE", 1.0)
    valid_per_rollout = num_steps - plan_horizon + 1

    from src.planners.env import Transition

    @jax.jit
    def collect_rollout(
        env_state: Any,
        obs: jax.Array,
        done: jax.Array,
        hstate: Any,
        rng: jax.Array,
    ) -> tuple:
        """Run one PPO rollout and extract sliding-window samples.

        Args:
            env_state: Gymnax environment state.
            obs:       ``[num_envs, obs_dim]`` current observations.
            done:      ``[num_envs]`` episode-done flags.
            hstate:    PPO recurrent hidden state.
            rng:       PRNG key.

        Returns:
            Updated carry + window arrays + env score dict.
        """

        def _env_step(carry: tuple, _: None) -> tuple:
            es, ob, dn, hs, r = carry
            r, act_rng, step_rng = jax.random.split(r, 3)
            action, new_hs = ppo.act(
                ob, dn, hs, act_rng, temperature=collect_temperature
            )
            new_obs, es, reward, new_done, info = env.step(
                step_rng, es, action, env_params
            )
            t = Transition(done=dn, action=action, reward=reward, obs=ob, info=info)
            return (es, new_obs, new_done, new_hs, r), t

        (env_state, obs, done, hstate, rng), traj = jax.lax.scan(
            _env_step,
            (env_state, obs, done, hstate, rng),
            None,
            num_steps,
        )

        def _window(t_idx: jax.Array) -> tuple:
            obs_t = traj.obs[t_idx]  # [E, obs_dim]
            acts = jax.lax.dynamic_slice(
                traj.action,
                (t_idx, 0),
                (plan_horizon, num_envs),
            )  # [H, E]
            dones = jax.lax.dynamic_slice(
                traj.done,
                (t_idx + 1, 0),
                (plan_horizon - 1, num_envs),
            )  # [H-1, E]
            valid = ~jnp.any(dones, axis=0)  # [E]
            rews = jax.lax.dynamic_slice(
                traj.reward,
                (t_idx, 0),
                (plan_horizon, num_envs),
            )
            return obs_t, jnp.swapaxes(acts, 0, 1), valid, jnp.sum(rews, axis=0)

        obs_w, act_w, valid_w, ret_w = jax.vmap(_window)(jnp.arange(valid_per_rollout))
        flat_obs = obs_w.reshape(-1, obs_dim)  # [N, obs_dim]
        flat_acts = act_w.reshape(-1, plan_horizon)  # [N, H]
        flat_valid = valid_w.reshape(-1)  # [N]
        flat_returns = ret_w.reshape(-1)  # [N]

        info_returned = traj.info["returned_episode"]
        env_score = jax.tree.map(
            lambda x: (x * info_returned).sum() / (info_returned.sum() + _EPS),
            traj.info,
        )
        return (
            env_state,
            obs,
            done,
            hstate,
            rng,
            flat_obs,
            flat_acts,
            flat_valid,
            flat_returns,
            env_score,
        )

    return collect_rollout


def build_eval_fn(
    env: Any,
    env_params: Any,
    apply_eval: Callable,
    config: dict,
) -> Callable:
    """Build a JIT-compiled eval function.

    Args:
        env:        Wrapped Craftax environment.
        env_params: Environment parameters.
        apply_eval: Eval apply fn (no dropout).
        config:     UPPERCASE config dict.

    Returns:
        JIT-compiled ``eval_policy(params, rng) -> info_dict``.
    """
    from src.diffusion.sampling import sample_plan
    from src.diffusion.schedules import SCHEDULE_MAP

    schedule_fn, _ = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    num_actions = config["NUM_ACTIONS"]
    n_cycles = config["EVAL_STEPS"] // config["EVAL_REPLAN"]

    @jax.jit
    def eval_policy(params: Any, rng: jax.Array) -> dict:
        """Evaluate model by running env with diffusion plans.

        Args:
            params: Model parameters.
            rng:    PRNG key.

        Returns:
            Dict of episode-weighted mean metrics.
        """
        rng, env_rng = jax.random.split(rng)
        val_obs, val_es = env.reset(env_rng, env_params)

        def _cycle(carry: tuple, _: None) -> tuple:
            es, vo, r = carry
            r, p_rng = jax.random.split(r)
            plan = sample_plan(
                apply_eval,
                params,
                p_rng,
                vo,
                num_actions,
                config["PLAN_HORIZON"],
                num_steps=config["VAL_DIFFUSION_STEPS"],
                schedule_fn=schedule_fn,
                remask_strategy=config["REMASK_STRATEGY"],
                eta=config["ETA"],
                use_loop=config["USE_LOOP"],
                t_on=config["T_ON"],
                t_off=config["T_OFF"],
                temperature=config["TEMPERATURE"],
                top_p=config["TOP_P"],
            )

            def _step(c: tuple, step_i: jax.Array) -> tuple:
                es_i, vo_i, r_i = c
                r_i, s_rng = jax.random.split(r_i)
                vo_next, es_next, _, _, info = env.step(
                    s_rng,
                    es_i,
                    plan[:, step_i],
                    env_params,
                )
                return (es_next, vo_next, r_i), info

            (es, vo, r), infos = jax.lax.scan(
                _step,
                (es, vo, r),
                jnp.arange(config["EVAL_REPLAN"]),
            )
            return (es, vo, r), infos

        _, cycle_infos = jax.lax.scan(_cycle, (val_es, val_obs, rng), None, n_cycles)
        infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), cycle_infos)
        ret = infos["returned_episode"]
        return jax.tree.map(lambda x: (x * ret).sum() / (ret.sum() + _EPS), infos)

    return eval_policy


def _init_replay_buffer(
    buf_size: int,
    plan_horizon: int,
    obs_dim: int,
) -> ReplayBuffer:
    """Create a zeroed ring buffer.

    Args:
        buf_size:     Maximum buffer capacity.
        plan_horizon: Action sequence length.
        obs_dim:      Observation dimensionality.

    Returns:
        Empty ``ReplayBuffer`` with pre-allocated arrays.
    """
    return ReplayBuffer(
        acts=jnp.zeros((buf_size, plan_horizon), dtype=jnp.int32),
        obs=jnp.zeros((buf_size, obs_dim), dtype=jnp.float32),
        valid=jnp.zeros(buf_size, dtype=jnp.bool_),
        returns=jnp.zeros(buf_size, dtype=jnp.float32),
        write_idx=jnp.array(0, dtype=jnp.int32),
        count=jnp.array(0, dtype=jnp.int32),
    )


def _push_to_buffer(
    buf: ReplayBuffer,
    new_acts: jax.Array,
    new_obs: jax.Array,
    new_valid: jax.Array,
    new_returns: jax.Array,
    n_new: int,
) -> ReplayBuffer:
    """Push ``n_new`` samples into the ring buffer.

    Args:
        buf:         Current replay buffer.
        new_acts:    ``[>=n_new, H]`` new action sequences.
        new_obs:     ``[>=n_new, obs_dim]`` new observations.
        new_valid:   ``[>=n_new]`` validity flags.
        new_returns: ``[>=n_new]`` returns.
        n_new:       Number of samples to push (static int).

    Returns:
        Updated ``ReplayBuffer``.
    """
    buf_size = buf.acts.shape[0]
    indices = (buf.write_idx + jnp.arange(n_new)) % buf_size
    acts = buf.acts.at[indices].set(new_acts[:n_new])
    obs = buf.obs.at[indices].set(new_obs[:n_new])
    valid = buf.valid.at[indices].set(new_valid[:n_new])
    returns = buf.returns.at[indices].set(new_returns[:n_new])
    new_write_idx = (buf.write_idx + n_new) % buf_size
    new_count = jnp.minimum(buf.count + n_new, buf_size)
    return ReplayBuffer(
        acts=acts,
        obs=obs,
        valid=valid,
        returns=returns,
        write_idx=new_write_idx,
        count=new_count,
    )


def _sample_from_buffer(
    buf: ReplayBuffer,
    rng: jax.Array,
    n_samples: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample from the ring buffer (with replacement).

    Args:
        buf:       Replay buffer.
        rng:       PRNG key.
        n_samples: Number of samples to draw (static int).

    Returns:
        Tuple of (acts, obs, valid, returns) each of size n_samples.
    """
    valid_count = jnp.maximum(buf.count, 1)
    indices = jax.random.randint(rng, (n_samples,), 0, valid_count)
    return buf.acts[indices], buf.obs[indices], buf.valid[indices], buf.returns[indices]


def make_run_ablation(
    spec: AblationSpec,
    config: dict,
    pretrained_params: Any,
    apply_train: Callable,
    apply_eval: Callable,
    env: Any,
    env_params: Any,
    ppo: Any,
    schedule_fn: Callable,
    schedule_deriv_fn: Callable,
    num_actions: int,
    obs_dim: int,
    fisher: Any | None = None,
) -> Callable:
    """Build a compiled ablation training closure.

    All static setup (model, optimizer, loss factories, diagnostic
    functions) happens here.  The returned ``run(rng)`` closure is safe
    for ``jax.jit`` and ``jax.vmap``.

    Args:
        spec:              Ablation specification from the registry.
        config:            UPPERCASE merged config dict.
        pretrained_params: Frozen pretrained model parameters.
        apply_train:       Training apply fn (with dropout).
        apply_eval:        Eval apply fn (no dropout).
        env:               Wrapped Craftax environment.
        env_params:        Environment parameters.
        ppo:               PPOAgent instance.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Size of the discrete action vocabulary.
        obs_dim:           Observation dimensionality.
        fisher:            Pre-computed Fisher diagonal (for EWC ablation).

    Returns:
        ``run(rng) -> (StepMetrics_stacked, final_carry)`` closure.
    """
    max_iter = config["MAX_ITER"]
    batch_size = config["BATCH_SIZE"]
    plan_horizon = config["PLAN_HORIZON"]
    num_envs = config["NUM_ENVS"]
    sigma_t = config.get("TRAIN_SIGMA", 0.0)

    # Diagnostic frequencies
    eval_every = config["EVAL_EVERY"]
    grad_align_every = config.get("GRAD_ALIGN_EVERY", 25)
    repr_drift_every = config.get("REPR_DRIFT_EVERY", 25)
    t_analysis_every = config.get("T_ANALYSIS_EVERY", 25)
    cka_every = config.get("CKA_EVERY", 50)
    per_layer_every = config.get("PER_LAYER_EVERY", 25)
    n_t_bins = config.get("T_ANALYSIS_N_BINS", 10)

    # Advantage params
    floor = config.get("RETURN_WEIGHT_FLOOR", 0.1)
    cap = config.get("RETURN_WEIGHT_CAP", 5.0)
    win_thresh = config.get("WIN_THRESHOLD", 0.1)
    ema_decay = config.get("RUNNING_STATS_EMA_DECAY", 0.99)

    # Mixed replay
    replay_buffer_size = config.get("MIXED_REPLAY_BUFFER_SIZE", 10000)
    mixed_replay_ratio = config.get("MIXED_REPLAY_RATIO", 0.25)
    n_replay_push = replay_buffer_size // 10  # new samples per iteration

    # Reward model
    rm_train_steps = config.get("REWARD_MODEL_TRAIN_STEPS", 50)
    rm_width = config.get("REWARD_MODEL_WIDTH", 64)
    rm_depth = config.get("REWARD_MODEL_DEPTH", 2)
    rm_lr = config.get("REWARD_MODEL_LR", 1e-3)

    # EMA decay for eval model
    ema_decay = config.get("EMA_DECAY", 0.999)

    # LoRA setup
    is_lora = spec.name == "lora"
    lora_rank = config.get("LORA_RANK", 8)
    lora_alpha = config.get("LORA_ALPHA", 16.0)

    # Build LoRA apply functions if needed
    if is_lora:

        def lora_apply_train(params_combined, obs, z_t, t, r=None):
            return apply_fn_with_lora(
                apply_train,
                params_combined["base"],
                params_combined["lora"],
                lora_alpha,
                lora_rank,
                obs,
                z_t,
                t,
                r,
            )

        def lora_apply_eval(params_combined, obs, z_t, t, r=None):
            return apply_fn_with_lora(
                apply_eval,
                params_combined["base"],
                params_combined["lora"],
                lora_alpha,
                lora_rank,
                obs,
                z_t,
                t,
                r,
            )
    else:
        lora_apply_train = apply_train
        lora_apply_eval = apply_eval

    # Active apply functions for this ablation
    active_apply_train = lora_apply_train if is_lora else apply_train
    active_apply_eval = lora_apply_eval if is_lora else apply_eval

    # Loss context
    ctx = LossContext(
        apply_fn=active_apply_train,
        ref_params=pretrained_params,
        schedule_fn=schedule_fn,
        schedule_deriv_fn=schedule_deriv_fn,
        num_actions=num_actions,
        config=config,
    )

    # Loss function — t_curriculum uses the JIT-compatible variant;
    # all others use the spec's factory.  Both are always defined so the
    # scan body (which branches on a Python bool) can reference either.
    extra_kwargs: dict = {}
    if spec.name == "ewc" and fisher is not None:
        extra_kwargs["fisher"] = fisher
    if spec.t_curriculum:
        # Use the JIT-compatible version that takes step_idx
        t_curriculum_loss_fn = make_loss_t_curriculum_jit(ctx)
        # Provide a baseline as the standard loss_fn (unused in scan body)
        loss_fn = make_loss_baseline(ctx)
    else:
        t_curriculum_loss_fn = None
        loss_fn = spec.loss_factory(
            ctx,
            **{k: v for k, v in extra_kwargs.items() if k in spec.extra_loss_kwargs},
        )

    # BC loss (for gradient surgery and alignment)
    bc_loss_fn = make_loss_baseline(ctx)

    # Rollout and eval
    collect_rollout = build_rollout_fn(env, env_params, ppo, config, obs_dim)
    config_with_eval = {**config, "NUM_ACTIONS": num_actions}
    eval_policy = build_eval_fn(
        env,
        env_params,
        active_apply_eval,
        config_with_eval,
    )

    # Diagnostic functions
    grad_align_fn = make_grad_alignment_fn(
        apply_train,
        schedule_fn,
        schedule_deriv_fn,
        num_actions,
        sigma_t,
    )
    repr_drift_fn = make_repr_drift_fn(apply_eval, schedule_fn, num_actions)
    cka_batch_size: int = min(
        int(config.get("CKA_BATCH_SIZE", 64)),
        int(batch_size),
    )

    cka_fn = make_cka_fn(
        apply_eval,
        schedule_fn,
        num_actions,
        cka_batch_size=cka_batch_size,
    )
    t_analysis_fn = make_t_analysis_fn(
        apply_train,
        schedule_fn,
        schedule_deriv_fn,
        num_actions,
        sigma_t,
        n_t_bins,
    )

    # Count param leaves for per-layer norms array size
    num_param_leaves = len(jax.tree.leaves(pretrained_params))

    # Spec flags as Python bools (static, not traced)
    use_action_diversity = spec.action_diversity_filter
    use_reward_filtering = spec.reward_filtering
    use_mixed_replay = spec.mixed_replay
    use_wins_only = spec.wins_only
    use_running_stats = spec.running_stats
    use_reward_model = spec.reward_model_weighting
    use_gradient_surgery = spec.gradient_surgery
    reward_filter_pct = config.get("REWARD_FILTER_PERCENTILE", 75)

    def run(rng: jax.Array) -> tuple[AblationCarry, StepMetrics]:
        """Execute the full ablation training loop.

        Args:
            rng: PRNG key (one per vmap replica).

        Returns:
            Tuple of (final_carry, stacked_metrics).
        """
        rng, init_rng, env_rng, lora_rng, rm_rng = jax.random.split(rng, 5)

        # Initialise parameters
        init_params = jax.tree.map(jnp.array, pretrained_params)
        if is_lora:
            lora_params = make_lora_params(init_params, lora_rank, lora_rng)
            init_params_combined = {"base": init_params, "lora": lora_params}
            optimizer = make_optimizer_lora_only(config, init_params, lora_params)
            state = TrainState.create(
                apply_fn=lora_apply_train,
                params=init_params_combined,
                tx=optimizer,
            )
        else:
            optimizer = spec.optimizer_factory(config, init_params)
            state = TrainState.create(
                apply_fn=active_apply_train,
                params=init_params,
                tx=optimizer,
            )

        # Environment init
        obs, env_state = env.reset(env_rng, env_params)
        done = jnp.zeros(num_envs, dtype=jnp.bool_)
        hstate = ppo.init_hidden(num_envs)

        # Replay buffer
        replay_buf = _init_replay_buffer(replay_buffer_size, plan_horizon, obs_dim)

        # Reward model
        if use_reward_model:
            _, rm_state = _build_reward_model(
                obs_dim, rm_rng, rm_width, rm_depth, rm_lr
            )
        else:
            # Dummy: needs same pytree structure for scan carry
            _, rm_state = _build_reward_model(obs_dim, rm_rng, 4, 1, 1e-4)

        # EMA params initialised as a copy of the initial params
        ema_params_init = jax.tree.map(jnp.array, state.params)

        carry_init = AblationCarry(
            state=state,
            ema_params=ema_params_init,
            env_state=env_state,
            obs=obs,
            done=done,
            hstate=hstate,
            rng=rng,
            step_idx=jnp.array(1, dtype=jnp.int32),
            running_mean=jnp.array(0.0),
            running_std=jnp.array(1.0),
            replay_buf=replay_buf,
            reward_model_state=rm_state,
        )

        def _update_step(
            carry: AblationCarry,
            _: None,
        ) -> tuple[AblationCarry, StepMetrics]:
            state = carry.state
            step_idx = carry.step_idx
            rng = carry.rng
            rm_state = carry.reward_model_state

            rng, rollout_rng, loss_rng, perm_rng = jax.random.split(rng, 4)
            rng, align_rng, drift_rng, t_rng, cka_rng, eval_rng = jax.random.split(
                rng, 6
            )

            # -- Collect rollout --
            (
                env_state_new,
                obs_new,
                done_new,
                hstate_new,
                rng,
                flat_obs,
                flat_acts,
                flat_valid,
                flat_returns,
                env_score_dict,
            ) = collect_rollout(
                carry.env_state,
                carry.obs,
                carry.done,
                carry.hstate,
                rollout_rng,
            )

            # -- Action diversity filter --
            if use_action_diversity:
                is_diverse = jax.vmap(lambda a: jnp.any(a != a[0]))(flat_acts)
                mask = is_diverse & flat_valid
            else:
                mask = flat_valid

            # -- Reward filtering --
            if use_reward_filtering:
                thresh = jnp.percentile(flat_returns, reward_filter_pct)
                mask = mask & (flat_returns > thresh)

            # Apply mask to validity
            flat_valid = mask

            # -- Replay buffer update --
            if use_mixed_replay:
                replay_buf_new = _push_to_buffer(
                    carry.replay_buf,
                    flat_acts,
                    flat_obs,
                    flat_valid,
                    flat_returns,
                    n_replay_push,
                )
            else:
                replay_buf_new = carry.replay_buf

            # -- Reward model training --
            if use_reward_model:

                def _rm_step(
                    rm_st: TrainState, _: None
                ) -> tuple[TrainState, jax.Array]:
                    rm_st, loss = _reward_model_train_step(
                        rm_st,
                        flat_obs[:batch_size],
                        flat_returns[:batch_size],
                    )
                    return rm_st, loss

                rm_state, _ = jax.lax.scan(_rm_step, rm_state, None, rm_train_steps)
                # Re-weight returns with reward model predictions
                rm_preds = rm_state.apply_fn(rm_state.params, flat_obs)
                flat_returns_for_adv = rm_preds
            else:
                flat_returns_for_adv = flat_returns

            # -- Compute advantages --
            advantages, new_mean, new_std = _compute_advantages(
                flat_returns_for_adv,
                floor,
                cap,
                wins_only=use_wins_only,
                win_thresh=win_thresh,
                use_running_stats=use_running_stats,
                ema_decay=ema_decay,
                running_mean=carry.running_mean,
                running_std=carry.running_std,
            )

            # -- Shuffle and batch --
            n_samples = flat_obs.shape[0]
            perm = jax.random.permutation(perm_rng, n_samples)
            flat_obs = flat_obs[perm]
            flat_acts = flat_acts[perm]
            flat_valid = flat_valid[perm]
            advantages = advantages[perm]

            obs_b = flat_obs[:batch_size]  # [B, obs_dim]
            act_b = flat_acts[:batch_size]  # [B, H]
            val_b = flat_valid[:batch_size]  # [B]
            adv_b = advantages[:batch_size]  # [B]

            # -- Mixed replay: blend in offline data --
            if use_mixed_replay:
                n_offline = max(1, int(batch_size * mixed_replay_ratio))
                n_online = batch_size - n_offline
                rng, buf_rng = jax.random.split(rng)
                buf_acts, buf_obs, buf_valid, buf_returns = _sample_from_buffer(
                    replay_buf_new,
                    buf_rng,
                    n_offline,
                )
                buf_adv, _, _ = _compute_advantages(
                    buf_returns,
                    floor,
                    cap,
                    wins_only=False,
                    win_thresh=win_thresh,
                    use_running_stats=False,
                    ema_decay=ema_decay,
                    running_mean=jnp.array(0.0),
                    running_std=jnp.array(1.0),
                )
                obs_b = jnp.concatenate([obs_b[:n_online], buf_obs], axis=0)
                act_b = jnp.concatenate([act_b[:n_online], buf_acts], axis=0)
                val_b = jnp.concatenate([val_b[:n_online], buf_valid], axis=0)
                adv_b = jnp.concatenate([adv_b[:n_online], buf_adv], axis=0)

            # -- Gradient step --
            if use_gradient_surgery:

                def rl_loss_fn(p: Any) -> jax.Array:
                    if spec.t_curriculum:
                        return t_curriculum_loss_fn(
                            p, act_b, obs_b, val_b, loss_rng, adv_b, step_idx
                        )
                    return loss_fn(p, act_b, obs_b, val_b, loss_rng, adv_b)

                def bc_loss_fn_call(p: Any) -> jax.Array:
                    return bc_loss_fn(
                        p, act_b, obs_b, val_b, loss_rng, jnp.ones(obs_b.shape[0])
                    )

                loss_val, g_rl = jax.value_and_grad(rl_loss_fn)(state.params)
                g_bc = jax.grad(bc_loss_fn_call)(state.params)
                g_rl_before = g_rl
                g_rl = gradient_surgery(g_rl, g_bc)
                state = state.apply_gradients(grads=g_rl)

                surgery_frac, surgery_n = compute_surgery_metrics_jax(g_rl_before, g_rl)
                grads_for_diag = g_rl
            else:

                def _loss_for_step(p: Any) -> jax.Array:
                    if spec.t_curriculum:
                        return t_curriculum_loss_fn(
                            p, act_b, obs_b, val_b, loss_rng, adv_b, step_idx
                        )
                    return loss_fn(p, act_b, obs_b, val_b, loss_rng, adv_b)

                loss_val, grads = jax.value_and_grad(_loss_for_step)(state.params)
                state = state.apply_gradients(grads=grads)
                grads_for_diag = grads
                surgery_frac = jnp.array(0.0)
                surgery_n = jnp.array(0, dtype=jnp.int32)

            # -- EMA update --
            ema_params_new = jax.tree.map(
                lambda ema, p: ema_decay * ema + (1.0 - ema_decay) * p,
                carry.ema_params,
                state.params,
            )

            # -- Win rate and effective batch size --
            win_rate_val = jnp.mean((flat_returns > win_thresh).astype(jnp.float32))
            eff_bs = _effective_batch_size(adv_b)
            env_score_val = env_score_dict["returned_episode_returns"]

            # -- Params for diagnostics (strip LoRA if needed) --
            if is_lora:
                params_diag = state.params["base"]
            else:
                params_diag = state.params

            # -- Gradient alignment (conditional) --
            def _do_grad_align() -> tuple[jax.Array, jax.Array, jax.Array]:
                return grad_align_fn(
                    params_diag,
                    pretrained_params,
                    act_b,
                    obs_b,
                    val_b,
                    align_rng,
                    adv_b,
                )

            def _skip_grad_align() -> tuple[jax.Array, jax.Array, jax.Array]:
                return jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

            cos_sim, rl_norm, bc_norm = jax.lax.cond(
                step_idx % grad_align_every == 0,
                _do_grad_align,
                _skip_grad_align,
            )

            # -- Representation drift (conditional) --
            def _do_repr_drift() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                return repr_drift_fn(
                    params_diag, pretrained_params, obs_b, act_b, drift_rng
                )

            def _skip_repr_drift() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                z = jnp.array(0.0)
                return z, z, z, z

            kl_mean, kl_low, kl_mid, kl_high = jax.lax.cond(
                step_idx % repr_drift_every == 0,
                _do_repr_drift,
                _skip_repr_drift,
            )

            # -- CKA (conditional) --
            def _do_cka() -> jax.Array:
                return cka_fn(params_diag, pretrained_params, obs_b, act_b, cka_rng)

            cka_val = jax.lax.cond(
                step_idx % cka_every == 0,
                _do_cka,
                lambda: jnp.array(0.0),
            )

            # -- t-analysis (conditional) --
            def _do_t_analysis() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                return t_analysis_fn(params_diag, act_b, obs_b, val_b, adv_b, t_rng)

            def _skip_t_analysis() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
                return (
                    jnp.zeros(n_t_bins),
                    jnp.array(0.0),
                    jnp.array(0.0),
                    jnp.array(0.0),
                )

            t_bins, lh_cos, t_low, t_high = jax.lax.cond(
                step_idx % t_analysis_every == 0,
                _do_t_analysis,
                _skip_t_analysis,
            )

            # -- Per-layer gradient norms (conditional) --
            def _do_per_layer() -> jax.Array:
                return compute_per_layer_grad_norms_jax(grads_for_diag)

            per_layer = jax.lax.cond(
                step_idx % per_layer_every == 0,
                _do_per_layer,
                lambda: jnp.zeros(num_param_leaves),
            )

            # -- Eval (conditional, uses EMA params) --
            def _do_eval() -> jax.Array:
                eval_info = eval_policy(ema_params_new, eval_rng)
                return eval_info["returned_episode_returns"]

            eval_score = jax.lax.cond(
                step_idx % eval_every == 0,
                _do_eval,
                lambda: jnp.array(0.0),
            )

            # -- Build metrics --
            metrics = StepMetrics(
                loss=loss_val,
                env_score=env_score_val,
                win_rate=win_rate_val,
                eff_batch_size=eff_bs,
                cos_sim=cos_sim,
                rl_grad_norm=rl_norm,
                bc_grad_norm=bc_norm,
                kl_mean=kl_mean,
                kl_low_t=kl_low,
                kl_mid_t=kl_mid,
                kl_high_t=kl_high,
                cka=cka_val,
                t_bin_norms=t_bins,
                low_high_cos=lh_cos,
                t_norm_low=t_low,
                t_norm_high=t_high,
                surgery_frac=surgery_frac,
                surgery_n_conflict=surgery_n,
                per_layer_norms=per_layer,
                eval_score=eval_score,
                did_eval=(step_idx % eval_every == 0).astype(jnp.float32),
                did_grad_align=(step_idx % grad_align_every == 0).astype(jnp.float32),
                did_repr_drift=(step_idx % repr_drift_every == 0).astype(jnp.float32),
                did_cka=(step_idx % cka_every == 0).astype(jnp.float32),
                did_t_analysis=(step_idx % t_analysis_every == 0).astype(jnp.float32),
                did_per_layer=(step_idx % per_layer_every == 0).astype(jnp.float32),
                did_surgery=jnp.array(1.0 if use_gradient_surgery else 0.0),
                did_log=(step_idx % 10 == 0).astype(jnp.float32),
            )

            new_carry = AblationCarry(
                state=state,
                ema_params=ema_params_new,
                env_state=env_state_new,
                obs=obs_new,
                done=done_new,
                hstate=hstate_new,
                rng=rng,
                step_idx=step_idx + 1,
                running_mean=new_mean,
                running_std=new_std,
                replay_buf=replay_buf_new,
                reward_model_state=rm_state,
            )
            return new_carry, metrics

        final_carry, all_metrics = jax.lax.scan(
            _update_step,
            carry_init,
            None,
            max_iter,
        )
        return final_carry, all_metrics

    return run


def metrics_to_history(
    all_metrics: StepMetrics,
    max_iter: int,
    config: dict,
) -> AblationHistory:
    """Convert stacked ``StepMetrics`` from scan to ``AblationHistory``.

    This runs on host (after JIT returns) and converts JAX arrays to
    Python lists/dicts for JSON serialisation and the analysis pipeline.

    Args:
        all_metrics: ``StepMetrics`` with leading ``[max_iter]`` dimension.
        max_iter:    Number of training iterations.
        config:      UPPERCASE config dict (for frequency settings).

    Returns:
        Populated ``AblationHistory``.
    """
    n_t_bins = config.get("T_ANALYSIS_N_BINS", 10)
    bin_edges = [i / n_t_bins for i in range(n_t_bins + 1)]

    history = AblationHistory()

    # Convert to host arrays
    loss = jax.device_get(all_metrics.loss)
    env_score = jax.device_get(all_metrics.env_score)
    did_log = jax.device_get(all_metrics.did_log)
    did_eval = jax.device_get(all_metrics.did_eval)
    did_ga = jax.device_get(all_metrics.did_grad_align)
    did_rd = jax.device_get(all_metrics.did_repr_drift)
    did_cka = jax.device_get(all_metrics.did_cka)
    did_ta = jax.device_get(all_metrics.did_t_analysis)
    did_pl = jax.device_get(all_metrics.did_per_layer)
    did_surg = jax.device_get(all_metrics.did_surgery)

    for i in range(max_iter):
        step = i + 1  # 1-based

        # Periodic logging (every 10 iters)
        if did_log[i] > 0.5:
            history.iters.append(step)
            history.loss.append(float(loss[i]))
            history.env_score_iters.append(step)
            history.env_score.append(float(env_score[i]))
            history.win_rate.append(float(jax.device_get(all_metrics.win_rate[i])))
            history.effective_batch_size.append(
                float(jax.device_get(all_metrics.eff_batch_size[i])),
            )

        # Eval
        if did_eval[i] > 0.5:
            history.eval_iters.append(step)
            history.eval_score.append(float(jax.device_get(all_metrics.eval_score[i])))
            # Achievement rates are not available from scan output
            # (variable-key dicts aren't JAX-compatible); populated separately
            history.per_achievement_rates.append({})

        # Gradient alignment
        if did_ga[i] > 0.5:
            history.grad_align_iters.append(step)
            history.grad_align.append(float(jax.device_get(all_metrics.cos_sim[i])))
            history.rl_grad_norm.append(
                float(jax.device_get(all_metrics.rl_grad_norm[i]))
            )
            history.bc_grad_norm.append(
                float(jax.device_get(all_metrics.bc_grad_norm[i]))
            )

        # Representation drift
        if did_rd[i] > 0.5:
            history.repr_drift_iters.append(step)
            history.repr_drift_kl.append(float(jax.device_get(all_metrics.kl_mean[i])))
            history.repr_drift_kl_low_t.append(
                float(jax.device_get(all_metrics.kl_low_t[i]))
            )
            history.repr_drift_kl_mid_t.append(
                float(jax.device_get(all_metrics.kl_mid_t[i]))
            )
            history.repr_drift_kl_high_t.append(
                float(jax.device_get(all_metrics.kl_high_t[i]))
            )

        # CKA
        if did_cka[i] > 0.5:
            history.cka_iters.append(step)
            history.cka_similarity.append(float(jax.device_get(all_metrics.cka[i])))

        # t-analysis
        if did_ta[i] > 0.5:
            history.t_analysis_iters.append(step)
            t_norms = jax.device_get(all_metrics.t_bin_norms[i])
            bin_dict = {}
            for j in range(n_t_bins):
                label = f"t_{bin_edges[j]:.1f}-{bin_edges[j + 1]:.1f}"
                bin_dict[label] = float(t_norms[j])
            history.t_bin_norms.append(bin_dict)
            history.norm_low_t.append(float(jax.device_get(all_metrics.t_norm_low[i])))
            history.norm_high_t.append(
                float(jax.device_get(all_metrics.t_norm_high[i]))
            )
            history.lowhigh_cos.append(
                float(jax.device_get(all_metrics.low_high_cos[i]))
            )

        # Per-layer norms
        if did_pl[i] > 0.5:
            history.per_layer_iters.append(step)
            norms = jax.device_get(all_metrics.per_layer_norms[i])
            # Convert to dict with leaf indices as keys
            history.per_layer_norms.append(
                {f"leaf_{j}": float(norms[j]) for j in range(len(norms))},
            )

        # Surgery
        if did_surg[i] > 0.5 and did_ga[i] > 0.5:
            history.surgery_iters.append(step)
            history.surgery_fraction.append(
                float(jax.device_get(all_metrics.surgery_frac[i])),
            )
            history.surgery_n_conflicting.append(
                int(jax.device_get(all_metrics.surgery_n_conflict[i])),
            )

    return history


def run_ablation(
    spec: AblationSpec,
    config: dict,
    pretrained_params: Any,
    apply_train: Callable,
    apply_eval: Callable,
    env: Any,
    env_params: Any,
    ppo: Any,
    schedule_fn: Callable,
    schedule_deriv_fn: Callable,
    num_actions: int,
    obs_dim: int,
    rng: jax.Array,
    wandb_run: Any = None,
    output_dir: Any = None,
) -> tuple[AblationHistory, float, Any]:
    """Run one complete ablation (Python-level wrapper around make_run_ablation).

    Handles EWC Fisher estimation (which requires rollouts before JIT),
    then delegates to the compiled ``make_run_ablation`` closure.

    Args:
        spec:              Ablation specification from the registry.
        config:            UPPERCASE merged config dict.
        pretrained_params: Frozen pretrained model parameters.
        apply_train:       Training apply fn (with dropout).
        apply_eval:        Eval apply fn (no dropout).
        env:               Wrapped Craftax environment.
        env_params:        Environment parameters.
        ppo:               PPOAgent instance.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        num_actions:       Size of the discrete action vocabulary.
        obs_dim:           Observation dimensionality.
        rng:               PRNG key.
        wandb_run:         Optional W&B run object for logging.
        output_dir:        Optional Path for per-iteration checkpoint saving.

    Returns:
        Tuple of ``(history, final_score, final_params)``.
    """
    logger.info("=" * 60)
    logger.info("ABLATION: %s  [Group %s]", spec.name, spec.group)
    logger.info("  %s", spec.description)
    logger.info("=" * 60)

    # EWC Fisher estimation (requires rollouts, done before JIT)
    fisher = None
    if spec.name == "ewc":
        n_fisher_batches = config.get("EWC_FISHER_BATCHES", 20)
        logger.info("  Estimating Fisher diagonal (%d batches)...", n_fisher_batches)
        rng, fisher_rng, env_rng = jax.random.split(rng, 3)
        obs_init, es_init = env.reset(env_rng, env_params)
        done_init = jnp.zeros(config["NUM_ENVS"], dtype=jnp.bool_)
        hstate_init = ppo.init_hidden(config["NUM_ENVS"])
        collect_fn = build_rollout_fn(env, env_params, ppo, config, obs_dim)

        batches = []
        es_f, obs_f, done_f, hs_f = es_init, obs_init, done_init, hstate_init
        for _ in range(n_fisher_batches):
            fisher_rng, rollout_rng = jax.random.split(fisher_rng)
            es_f, obs_f, done_f, hs_f, _, flat_obs, flat_acts, flat_valid, _, _ = (
                collect_fn(
                    es_f,
                    obs_f,
                    done_f,
                    hs_f,
                    rollout_rng,
                )
            )
            bs = min(flat_acts.shape[0], config["BATCH_SIZE"])
            batches.append((flat_acts[:bs], flat_obs[:bs], flat_valid[:bs]))

        fisher = estimate_fisher_diagonal(
            apply_train,
            pretrained_params,
            schedule_fn,
            schedule_deriv_fn,
            num_actions,
            batches,
            sigma_t=config.get("TRAIN_SIGMA", 0.0),
        )
        logger.info("  Fisher diagonal estimated.")

    # Build and JIT-compile the training closure
    run_fn = make_run_ablation(
        spec=spec,
        config=config,
        pretrained_params=pretrained_params,
        apply_train=apply_train,
        apply_eval=apply_eval,
        env=env,
        env_params=env_params,
        ppo=ppo,
        schedule_fn=schedule_fn,
        schedule_deriv_fn=schedule_deriv_fn,
        num_actions=num_actions,
        obs_dim=obs_dim,
        fisher=fisher,
    )

    logger.info("  Compiling training loop...")
    jitted_run = jax.jit(run_fn)

    logger.info("  Running %d iterations...", config["MAX_ITER"])
    final_carry, all_metrics = jitted_run(rng)

    # Convert scan outputs to AblationHistory
    history = metrics_to_history(all_metrics, config["MAX_ITER"], config)

    # Post-JIT: run final eval to extract achievement rates
    # (variable-key dicts can't live inside jax.lax.scan)
    is_lora = spec.name == "lora"
    if is_lora:

        def _lora_apply_eval(params_combined, obs, z_t, t, r=None):
            return apply_fn_with_lora(
                apply_eval,
                params_combined["base"],
                params_combined["lora"],
                config.get("LORA_ALPHA", 16.0),
                config.get("LORA_RANK", 8),
                obs,
                z_t,
                t,
                r,
            )

        active_eval = _lora_apply_eval
    else:
        active_eval = apply_eval

    config_with_eval = {**config, "NUM_ACTIONS": num_actions}
    eval_policy = build_eval_fn(env, env_params, active_eval, config_with_eval)

    final_info = eval_policy(final_carry.ema_params, final_carry.rng)

    # Flatten LoRA params into base weights so downstream consumers
    # (e.g. action distribution analysis) get standard flat params.
    if is_lora:
        ema = final_carry.ema_params
        alpha = config.get("LORA_ALPHA", 16.0)
        rank = config.get("LORA_RANK", 8)
        scale = alpha / max(rank, 1)
        lora_p = ema["lora"]

        def _inject(path: tuple, param: jnp.ndarray) -> jnp.ndarray:
            path_str = "/".join(
                str(k.key) if hasattr(k, "key") else str(k) for k in path
            )
            if path_str in lora_p:
                ab = lora_p[path_str]
                return param + scale * (ab["A"] @ ab["B"])
            return param

        final_params = jax.device_get(
            jax.tree_util.tree_map_with_path(_inject, ema["base"])
        )
    else:
        final_params = jax.device_get(final_carry.ema_params)
    final_score = float(final_info.get("returned_episode_returns", jnp.array(0.0)))

    # Extract per-achievement unlock rates from final eval
    final_ach = {
        k: float(v) / 100.0 for k, v in final_info.items() if "achievement" in k.lower()
    }
    # Overwrite the empty dicts in history with final eval achievements
    if history.per_achievement_rates:
        history.per_achievement_rates[-1] = final_ach

    logger.info("  [%s] FINAL score: %.4f", spec.name, final_score)

    # W&B logging (outside JIT)
    if wandb_run is not None:
        ns = f"ablations/{spec.name}"
        for i, step in enumerate(history.iters):
            wandb_run.log(
                {
                    f"{ns}/train_loss": history.loss[i],
                    f"{ns}/env_score": history.env_score[i],
                    "iteration": step,
                }
            )
        for i, step in enumerate(history.eval_iters):
            wandb_run.log(
                {
                    f"{ns}/eval_score": history.eval_score[i],
                    "iteration": step,
                }
            )

    return history, final_score, final_params
