"""Core training loop for RL fine-tuning ablations.

``run_ablation`` is the single entry point for running one ablation.
It handles all special cases (gradient surgery, LoRA, mixed replay, etc.)
and populates an ``AblationHistory`` at configurable diagnostic frequencies.

The outer loop is Python-level (flexible).
The gradient step is JIT-compiled via Flax TrainState.apply_gradients.
"""

from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from experiments.rl_finetuning.ablations.losses import (
    LossContext,
    estimate_fisher_diagonal,
    make_loss_bc_wins,
    make_loss_baseline,
    make_loss_t_curriculum,
)
from experiments.rl_finetuning.ablations.optimizers import (
    apply_fn_with_lora,
    gradient_surgery,
    make_lora_params,
    make_optimizer_lora_only,
)
from experiments.rl_finetuning.ablations.registry import AblationSpec
from experiments.rl_finetuning.diagnostics.gradient import (
    GradAlignResult,
    PerLayerGradNorms,
    compute_per_layer_grad_norms,
    compute_surgery_metrics,
    make_grad_alignment_fn,
)
from experiments.rl_finetuning.diagnostics.representation import (
    ReprDriftResult,
    make_cka_fn,
    make_repr_drift_fn,
)
from experiments.rl_finetuning.diagnostics.timestep import (
    TBinGradNorms,
    make_t_analysis_fn,
)

logger = logging.getLogger(__name__)

_EPS: float = 1e-5


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------


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

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict.

        Returns:
            Dict with all list fields preserved.
        """
        return {k: list(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "AblationHistory":
        """Reconstruct from a dict (e.g., loaded from JSON).

        Args:
            d: Dict produced by ``to_dict()``.

        Returns:
            ``AblationHistory`` instance.
        """
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# Reward model (lightweight MLP for Group D ablation)
# ---------------------------------------------------------------------------


class SimpleRewardModel:
    """Lightweight 2-layer MLP reward model trained on (obs, returns).

    Args:
        obs_dim:     Input observation dimensionality.
        width:       Hidden layer width.
        depth:       Number of hidden layers.
        lr:          Learning rate.
    """

    def __init__(self, obs_dim: int, width: int = 64, depth: int = 2, lr: float = 1e-3):
        import flax.linen as nn

        class MLP(nn.Module):
            width: int
            depth: int

            @nn.compact
            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                for _ in range(self.depth):
                    x = nn.Dense(self.width)(x)
                    x = nn.relu(x)
                return nn.Dense(1)(x).squeeze(-1)

        self._net = MLP(width=width, depth=depth)
        rng = jax.random.PRNGKey(0)
        dummy_params = self._net.init(rng, jnp.zeros((1, obs_dim)))
        tx = optax.adam(lr)
        self._state = TrainState.create(
            apply_fn=self._net.apply, params=dummy_params, tx=tx
        )

    def train_step(self, obs: jnp.ndarray, targets: jnp.ndarray) -> float:
        """One gradient step on MSE loss.

        Args:
            obs:     ``[B, obs_dim]`` observations.
            targets: ``[B]`` return targets.

        Returns:
            Scalar MSE loss.
        """
        def loss_fn(params):
            preds = self._state.apply_fn(params, obs)
            return jnp.mean((preds - targets) ** 2)

        loss_val, grads = jax.value_and_grad(loss_fn)(self._state.params)
        self._state = self._state.apply_gradients(grads=grads)
        return float(loss_val)

    def predict(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Predict returns for observations.

        Args:
            obs: ``[B, obs_dim]`` observations.

        Returns:
            ``[B]`` predicted returns.
        """
        return self._state.apply_fn(self._state.params, obs)


# ---------------------------------------------------------------------------
# Advantage computation
# ---------------------------------------------------------------------------


def compute_advantages(
    flat_returns: jnp.ndarray,
    config: dict,
    wins_only: bool = False,
    running_mean: float | None = None,
    running_std: float | None = None,
    reward_model: SimpleRewardModel | None = None,
    flat_obs: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, float, float]:
    """Compute per-sample advantage weights from window returns.

    Args:
        flat_returns:  ``[N]`` float32 raw returns.
        config:        UPPERCASE config dict.
        wins_only:     If True, return binary win mask instead of normalised weights.
        running_mean:  Optional running mean for normalisation (returned updated).
        running_std:   Optional running std for normalisation (returned updated).
        reward_model:  Optional trained reward model for re-weighting.
        flat_obs:      ``[N, obs_dim]`` observations (required when reward_model given).

    Returns:
        Tuple of (advantages ``[N]``, updated_running_mean, updated_running_std).
    """
    win_thresh = config.get("WIN_THRESHOLD", 0.1)
    floor = config.get("RETURN_WEIGHT_FLOOR", 0.1)
    cap = config.get("RETURN_WEIGHT_CAP", 5.0)
    ema_decay = config.get("RUNNING_STATS_EMA_DECAY", 0.99)

    if wins_only:
        adv = (flat_returns > win_thresh).astype(jnp.float32)
        return adv, running_mean or 0.0, running_std or 1.0

    # Reward model re-weighting
    if reward_model is not None and flat_obs is not None:
        flat_returns = reward_model.predict(flat_obs)

    clipped = jnp.clip(flat_returns, 0.0, None)

    if running_mean is not None and running_std is not None:
        # EMA-based normalisation
        batch_mean = float(jnp.mean(clipped))
        batch_std = float(jnp.std(clipped)) + 1e-8
        new_mean = ema_decay * running_mean + (1.0 - ema_decay) * batch_mean
        new_std = ema_decay * running_std + (1.0 - ema_decay) * batch_std
        adv = (clipped - new_mean) / new_std
        adv = jnp.clip(adv + 1.0, floor, cap)  # shift to positive range
        return adv, new_mean, new_std

    weights = clipped / (jnp.mean(clipped) + _EPS)
    adv = jnp.clip(weights, floor, cap)
    return adv, float(jnp.mean(clipped)), float(jnp.std(clipped))


def effective_batch_size(advantages: jnp.ndarray) -> float:
    """Compute effective batch size: (sum w_i)^2 / sum w_i^2.

    Low value means the gradient is dominated by a few high-weight samples.

    Args:
        advantages: ``[N]`` advantage weights.

    Returns:
        Effective batch size as a float.
    """
    sum_w = float(jnp.sum(advantages))
    sum_w2 = float(jnp.sum(advantages ** 2))
    return sum_w ** 2 / max(sum_w2, 1e-10)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


def build_rollout_fn(env, env_params, ppo, config: dict, obs_dim: int) -> Callable:
    """Build a JIT-compiled rollout function (matches craftax_ablations pattern).

    Args:
        env:       Wrapped Craftax environment.
        env_params: Environment parameters.
        ppo:       PPOAgent instance.
        config:    UPPERCASE config dict.
        obs_dim:   Observation dimensionality.

    Returns:
        JIT-compiled ``collect_rollout(env_state, obs, done, hstate, rng)``
        returning windows ``(env_state, obs, done, hstate, rng, flat_obs,
        flat_acts, flat_valid, flat_returns, env_score)``.
    """
    num_envs = config["NUM_ENVS"]
    num_steps = config["NUM_STEPS"]
    plan_horizon = config["PLAN_HORIZON"]
    collect_temp = config.get("COLLECT_TEMP", 1.0)
    valid_per_rollout = num_steps - plan_horizon + 1

    from src.planners.env import Transition

    @jax.jit
    def collect_rollout(env_state, obs, done, hstate, rng):
        """Run one PPO rollout and extract sliding-window samples.

        Args:
            env_state: Gymnax environment state.
            obs:       ``[num_envs, obs_dim]`` current observations.
            done:      ``[num_envs]`` episode-done flags.
            hstate:    PPO recurrent hidden state.
            rng:       PRNG key.

        Returns:
            Updated (env_state, obs, done, hstate, rng) plus window arrays.
        """
        def _env_step(carry, _):
            es, ob, dn, hs, rng = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            action, new_hs = ppo.act(ob, dn, hs, act_rng, temperature=collect_temp)
            new_obs, es, reward, new_done, info = env.step(step_rng, es, action, env_params)
            t = Transition(done=dn, action=action, reward=reward, obs=ob, info=info)
            return (es, new_obs, new_done, new_hs, rng), t

        (env_state, obs, done, hstate, rng), traj = jax.lax.scan(
            _env_step, (env_state, obs, done, hstate, rng), None, num_steps
        )

        def _window(t_idx):
            obs_t = traj.obs[t_idx]
            acts = jax.lax.dynamic_slice(traj.action, (t_idx, 0), (plan_horizon, num_envs))
            dones = jax.lax.dynamic_slice(traj.done, (t_idx + 1, 0), (plan_horizon - 1, num_envs))
            valid = ~jnp.any(dones, axis=0)
            rews = jax.lax.dynamic_slice(traj.reward, (t_idx, 0), (plan_horizon, num_envs))
            return obs_t, jnp.swapaxes(acts, 0, 1), valid, jnp.sum(rews, axis=0)

        obs_w, act_w, valid_w, ret_w = jax.vmap(_window)(jnp.arange(valid_per_rollout))
        flat_obs = obs_w.reshape(-1, obs_dim)
        flat_acts = act_w.reshape(-1, plan_horizon)
        flat_valid = valid_w.reshape(-1)
        flat_returns = ret_w.reshape(-1)

        info_returned = traj.info["returned_episode"]
        env_score = jax.tree.map(
            lambda x: (x * info_returned).sum() / (info_returned.sum() + _EPS),
            traj.info,
        )
        return env_state, obs, done, hstate, rng, flat_obs, flat_acts, flat_valid, flat_returns, env_score

    return collect_rollout


def build_eval_fn(env, env_params, apply_eval: Callable, config: dict) -> Callable:
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
    def eval_policy(params, rng):
        rng, env_rng = jax.random.split(rng)
        val_obs, val_es = env.reset(env_rng, env_params)

        def _cycle(carry, _):
            es, vo, rng = carry
            rng, p_rng = jax.random.split(rng)
            plan = sample_plan(
                apply_eval, params, p_rng, vo,
                num_actions, config["PLAN_HORIZON"],
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

            def _step(c, step_i):
                es_i, vo_i, r = c
                r, s_rng = jax.random.split(r)
                vo_next, es_next, _, _, info = env.step(s_rng, es_i, plan[:, step_i], env_params)
                return (es_next, vo_next, r), info

            (es, vo, rng), infos = jax.lax.scan(
                _step, (es, vo, rng), jnp.arange(config["EVAL_REPLAN"])
            )
            return (es, vo, rng), infos

        _, cycle_infos = jax.lax.scan(_cycle, (val_es, val_obs, rng), None, n_cycles)
        infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), cycle_infos)
        ret = infos["returned_episode"]
        return jax.tree.map(lambda x: (x * ret).sum() / (ret.sum() + _EPS), infos)

    return eval_policy


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------


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
    """Run one complete ablation training loop.

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

    history = AblationHistory()
    max_iter = config["MAX_ITER"]

    # ── LoRA setup ────────────────────────────────────────────────────────────
    is_lora = spec.name == "lora"
    lora_params = None
    lora_apply_train = apply_train
    lora_apply_eval = apply_eval
    lora_rank = config.get("LORA_RANK", 8)
    lora_alpha = config.get("LORA_ALPHA", 16.0)

    if is_lora:
        rng, lora_rng = jax.random.split(rng)
        lora_params = make_lora_params(pretrained_params, lora_rank, lora_rng)

        def lora_apply_train(params_combined, obs, z_t, t, r=None):
            return apply_fn_with_lora(
                apply_train, params_combined["base"], params_combined["lora"],
                lora_alpha, lora_rank, obs, z_t, t, r,
            )

        def lora_apply_eval(params_combined, obs, z_t, t, r=None):
            return apply_fn_with_lora(
                apply_eval, params_combined["base"], params_combined["lora"],
                lora_alpha, lora_rank, obs, z_t, t, r,
            )

    # ── Loss context ─────────────────────────────────────────────────────────
    ctx = LossContext(
        apply_fn=lora_apply_train if is_lora else apply_train,
        ref_params=pretrained_params,
        schedule_fn=schedule_fn,
        schedule_deriv_fn=schedule_deriv_fn,
        num_actions=num_actions,
        config=config,
    )

    # ── EWC: estimate Fisher before training ──────────────────────────────────
    extra_kwargs: dict = {}
    if spec.name == "ewc":
        logger.info("  Estimating Fisher diagonal (%d batches)...", config.get("EWC_FISHER_BATCHES", 20))
        rng, fisher_rng, env_rng = jax.random.split(rng, 3)
        obs_init, es_init = env.reset(env_rng, env_params)
        done_init = jnp.zeros(config["NUM_ENVS"], dtype=bool)
        hstate_init = ppo.init_hidden(config["NUM_ENVS"])
        collect_fn = build_rollout_fn(env, env_params, ppo, config, obs_dim)

        batches = []
        n_fisher_batches = config.get("EWC_FISHER_BATCHES", 20)
        es_f, obs_f, done_f, hs_f = es_init, obs_init, done_init, hstate_init
        for _ in range(n_fisher_batches):
            fisher_rng, rollout_rng = jax.random.split(fisher_rng)
            es_f, obs_f, done_f, hs_f, _, flat_obs, flat_acts, flat_valid, _, _ = collect_fn(
                es_f, obs_f, done_f, hs_f, rollout_rng
            )
            bs = min(flat_acts.shape[0], config["BATCH_SIZE"])
            batches.append((flat_acts[:bs], flat_obs[:bs], flat_valid[:bs]))

        fisher = estimate_fisher_diagonal(
            apply_train, pretrained_params, schedule_fn, schedule_deriv_fn,
            num_actions, batches, sigma_t=config.get("TRAIN_SIGMA", 0.0),
        )
        extra_kwargs["fisher"] = fisher
        logger.info("  Fisher diagonal estimated.")

    # ── t-curriculum: mutable iter counter ───────────────────────────────────
    current_iter_container: list[int] = [0]
    if spec.t_curriculum:
        ctx_curriculum = LossContext(
            apply_fn=apply_train, ref_params=pretrained_params,
            schedule_fn=schedule_fn, schedule_deriv_fn=schedule_deriv_fn,
            num_actions=num_actions, config=config,
        )
        loss_fn = make_loss_t_curriculum(ctx_curriculum, current_iter_container)
    else:
        loss_fn = spec.loss_factory(ctx, **{k: v for k, v in extra_kwargs.items()
                                            if k in spec.extra_loss_kwargs})

    # ── BC loss (for gradient surgery and grad alignment) ─────────────────────
    bc_loss_fn = make_loss_baseline(ctx)

    # ── Initialise parameters and TrainState ──────────────────────────────────
    init_params = jax.tree.map(jnp.array, pretrained_params)
    if is_lora:
        init_params_combined = {"base": init_params, "lora": lora_params}
        optimizer = make_optimizer_lora_only(config, init_params, lora_params)
        state = TrainState.create(
            apply_fn=lora_apply_train, params=init_params_combined, tx=optimizer
        )
    else:
        optimizer = spec.optimizer_factory(config, init_params)
        from src.planners.model import create_train_state, build_model
        net = build_model(config, num_actions)
        state = create_train_state(net, init_params, config.get("LR", 3e-4), config.get("MAX_GRAD_NORM", 1.0))
        # Replace tx with the spec-specific optimizer
        state = TrainState.create(apply_fn=state.apply_fn, params=init_params, tx=optimizer)

    # ── Diagnostic functions ──────────────────────────────────────────────────
    sigma_t = config.get("TRAIN_SIGMA", 0.0)
    grad_align_fn = make_grad_alignment_fn(
        apply_train, schedule_fn, schedule_deriv_fn, num_actions, sigma_t
    )
    repr_drift_fn = make_repr_drift_fn(apply_eval, schedule_fn, num_actions)
    cka_fn = make_cka_fn(apply_eval, schedule_fn, num_actions)
    t_analysis_fn = make_t_analysis_fn(
        apply_train, schedule_fn, schedule_deriv_fn, num_actions, sigma_t
    )

    grad_align_every = config.get("GRAD_ALIGN_EVERY", 25)
    repr_drift_every = config.get("REPR_DRIFT_EVERY", 25)
    t_analysis_every = config.get("T_ANALYSIS_EVERY", 25)
    cka_every = config.get("CKA_EVERY", 50)
    per_layer_every = config.get("PER_LAYER_EVERY", 25)

    # ── Eval and rollout functions ─────────────────────────────────────────────
    collect_rollout = build_rollout_fn(env, env_params, ppo, config, obs_dim)
    config_with_eval = {**config, "NUM_ACTIONS": num_actions}
    eval_policy = build_eval_fn(env, env_params, lora_apply_eval if is_lora else apply_eval, config_with_eval)

    # ── Mixed replay buffer ───────────────────────────────────────────────────
    replay_buffer: list[tuple] = []  # list of (acts, obs, valid, returns)
    replay_buffer_size = config.get("MIXED_REPLAY_BUFFER_SIZE", 10000)
    mixed_replay_ratio = config.get("MIXED_REPLAY_RATIO", 0.25)

    # ── Running stats for advantage normalisation ─────────────────────────────
    running_mean: float | None = None
    running_std: float | None = None
    if spec.running_stats:
        running_mean = 0.0
        running_std = 1.0

    # ── Reward model ──────────────────────────────────────────────────────────
    reward_model: SimpleRewardModel | None = None
    if spec.reward_model_weighting:
        reward_model = SimpleRewardModel(
            obs_dim,
            width=config.get("REWARD_MODEL_WIDTH", 64),
            depth=config.get("REWARD_MODEL_DEPTH", 2),
            lr=config.get("REWARD_MODEL_LR", 1e-3),
        )

    # ── Environment init ──────────────────────────────────────────────────────
    rng, env_rng = jax.random.split(rng)
    obs, env_state = env.reset(env_rng, env_params)
    done = jnp.zeros(config["NUM_ENVS"], dtype=bool)
    hstate = ppo.init_hidden(config["NUM_ENVS"])

    running_loss = running_score = n_log = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for iteration in range(1, max_iter + 1):
        current_iter_container[0] = iteration
        rng, rollout_rng, loss_rng, align_rng, drift_rng, t_rng, cka_rng = jax.random.split(rng, 7)

        # Collect rollout
        (env_state, obs, done, hstate, rng,
         flat_obs, flat_acts, flat_valid, flat_returns, env_score) = collect_rollout(
            env_state, obs, done, hstate, rollout_rng
        )

        # Action diversity filter
        if spec.action_diversity_filter:
            is_diverse = jax.vmap(lambda a: jnp.any(a != a[0]))(flat_acts)
            mask = is_diverse & flat_valid
        else:
            mask = flat_valid

        # Reward filtering (top percentile)
        if spec.reward_filtering:
            percentile = config.get("REWARD_FILTER_PERCENTILE", 75)
            thresh = float(jnp.percentile(flat_returns, percentile))
            mask = mask & (flat_returns > thresh)

        # Update replay buffer
        if spec.mixed_replay:
            n_new = min(int(flat_obs.shape[0]), replay_buffer_size // 10)
            replay_buffer.append((
                jax.device_get(flat_acts[:n_new]),
                jax.device_get(flat_obs[:n_new]),
                jax.device_get(flat_valid[:n_new]),
                jax.device_get(flat_returns[:n_new]),
            ))
            total_stored = sum(a.shape[0] for a, _, _, _ in replay_buffer)
            while total_stored > replay_buffer_size and len(replay_buffer) > 1:
                removed = replay_buffer.pop(0)
                total_stored -= removed[0].shape[0]

        # Compute advantages
        params_for_adv = state.params if not is_lora else state.params
        adv_returns = flat_returns
        if reward_model is not None:
            rm_train_steps = config.get("REWARD_MODEL_TRAIN_STEPS", 50)
            bs_rm = min(flat_obs.shape[0], config["BATCH_SIZE"])
            for _ in range(rm_train_steps):
                reward_model.train_step(flat_obs[:bs_rm], flat_returns[:bs_rm])

        advantages, running_mean, running_std = compute_advantages(
            flat_returns, config,
            wins_only=spec.wins_only,
            running_mean=running_mean,
            running_std=running_std,
            reward_model=reward_model if spec.reward_model_weighting else None,
            flat_obs=flat_obs if spec.reward_model_weighting else None,
        )

        # Shuffle
        n_samples = flat_obs.shape[0]
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n_samples)
        flat_obs = flat_obs[perm]
        flat_acts = flat_acts[perm]
        flat_valid = flat_valid[perm]
        advantages = advantages[perm]

        bs = config["BATCH_SIZE"]
        obs_b = flat_obs[:bs]
        act_b = flat_acts[:bs]
        val_b = flat_valid[:bs]
        adv_b = advantages[:bs]

        # Mixed replay: blend in offline data
        if spec.mixed_replay and replay_buffer:
            n_offline = max(1, int(bs * mixed_replay_ratio))
            n_online = bs - n_offline
            obs_b = obs_b[:n_online]
            act_b = act_b[:n_online]
            val_b = val_b[:n_online]
            adv_b = adv_b[:n_online]
            # Sample from replay buffer
            all_acts_buf = np.concatenate([a for a, _, _, _ in replay_buffer], axis=0)
            all_obs_buf = np.concatenate([o for _, o, _, _ in replay_buffer], axis=0)
            all_valid_buf = np.concatenate([v for _, _, v, _ in replay_buffer], axis=0)
            all_ret_buf = np.concatenate([r for _, _, _, r in replay_buffer], axis=0)
            n_buf = all_acts_buf.shape[0]
            idx = np.random.choice(n_buf, size=n_offline, replace=n_buf < n_offline)
            obs_offline = jnp.array(all_obs_buf[idx])
            act_offline = jnp.array(all_acts_buf[idx])
            val_offline = jnp.array(all_valid_buf[idx])
            adv_offline, _, _ = compute_advantages(
                jnp.array(all_ret_buf[idx]), config
            )
            obs_b = jnp.concatenate([obs_b, obs_offline], axis=0)
            act_b = jnp.concatenate([act_b, act_offline], axis=0)
            val_b = jnp.concatenate([val_b, val_offline], axis=0)
            adv_b = jnp.concatenate([adv_b, adv_offline], axis=0)

        # ── Gradient step ────────────────────────────────────────────────────
        if spec.gradient_surgery:
            # PCGrad: compute RL and BC gradients separately, then project
            def rl_loss_callable(p):
                return loss_fn(p, act_b, obs_b, val_b, loss_rng, adv_b)

            def bc_loss_callable(p):
                return bc_loss_fn(p, act_b, obs_b, val_b, loss_rng, jnp.ones(obs_b.shape[0]))

            loss_val, g_rl = jax.value_and_grad(rl_loss_callable)(state.params)
            g_bc = jax.grad(bc_loss_callable)(state.params)

            g_rl_before = g_rl
            g_rl = gradient_surgery(g_rl, g_bc)

            state = state.apply_gradients(grads=g_rl)

            if iteration % grad_align_every == 0:
                surgery_metrics = compute_surgery_metrics(g_rl_before, g_rl)
                history.surgery_iters.append(iteration)
                history.surgery_fraction.append(surgery_metrics.projected_mass_fraction)
                history.surgery_n_conflicting.append(surgery_metrics.n_conflicting_params)

        else:
            # Standard gradient step
            def _loss_for_step(p):
                return loss_fn(p, act_b, obs_b, val_b, loss_rng, adv_b)

            loss_val, grads = jax.value_and_grad(_loss_for_step)(state.params)
            state = state.apply_gradients(grads=grads)

            # Per-layer gradient norms
            if iteration % per_layer_every == 0:
                layer_norms = compute_per_layer_grad_norms(grads)
                history.per_layer_iters.append(iteration)
                history.per_layer_norms.append(layer_norms.layer_norms)

        running_loss += float(loss_val)
        running_score += float(env_score.get("returned_episode_returns", jnp.array(0.0)))
        n_log += 1

        # Win rate and effective batch size
        win_rate = float(jnp.mean(flat_returns > config.get("WIN_THRESHOLD", 0.1)))
        eff_bs = effective_batch_size(adv_b)

        # ── Gradient alignment ────────────────────────────────────────────────
        if iteration % grad_align_every == 0:
            params_for_diag = state.params if not is_lora else state.params["base"]
            result = grad_align_fn(
                params_for_diag, pretrained_params, act_b, obs_b, val_b, align_rng, adv_b
            )
            history.grad_align_iters.append(iteration)
            history.grad_align.append(result.cos_sim)
            history.rl_grad_norm.append(result.rl_grad_norm)
            history.bc_grad_norm.append(result.bc_grad_norm)
            logger.info(
                "  [%s] iter %d | grad_align=%+.4f | rl_norm=%.4f | bc_norm=%.4f",
                spec.name, iteration, result.cos_sim, result.rl_grad_norm, result.bc_grad_norm,
            )

        # ── Representation drift ──────────────────────────────────────────────
        if iteration % repr_drift_every == 0:
            params_for_diag = state.params if not is_lora else state.params["base"]
            drift: ReprDriftResult = repr_drift_fn(
                params_for_diag, pretrained_params, obs_b, act_b, drift_rng
            )
            history.repr_drift_iters.append(iteration)
            history.repr_drift_kl.append(drift.kl_mean)
            history.repr_drift_kl_low_t.append(drift.kl_low_t)
            history.repr_drift_kl_mid_t.append(drift.kl_mid_t)
            history.repr_drift_kl_high_t.append(drift.kl_high_t)
            logger.info(
                "  [%s] iter %d | repr_drift_kl=%.6f", spec.name, iteration, drift.kl_mean
            )

        # ── CKA similarity ────────────────────────────────────────────────────
        if iteration % cka_every == 0:
            params_for_diag = state.params if not is_lora else state.params["base"]
            cka_result = cka_fn(params_for_diag, pretrained_params, obs_b, act_b, cka_rng)
            history.cka_iters.append(iteration)
            history.cka_similarity.append(cka_result.cka)

        # ── t-distribution analysis ───────────────────────────────────────────
        if iteration % t_analysis_every == 0:
            params_for_diag = state.params if not is_lora else state.params["base"]
            t_result: TBinGradNorms = t_analysis_fn(
                params_for_diag, act_b, obs_b, val_b, adv_b, t_rng
            )
            history.t_analysis_iters.append(iteration)
            history.norm_low_t.append(t_result.norm_low_t)
            history.norm_high_t.append(t_result.norm_high_t)
            history.lowhigh_cos.append(t_result.low_high_cos)
            history.t_bin_norms.append(t_result.bin_norms)

        # ── Periodic logging ──────────────────────────────────────────────────
        if iteration % 10 == 0:
            ml = running_loss / max(n_log, 1)
            ms = running_score / max(n_log, 1)
            history.iters.append(iteration)
            history.loss.append(ml)
            history.env_score_iters.append(iteration)
            history.env_score.append(ms)
            history.win_rate.append(win_rate)
            history.effective_batch_size.append(eff_bs)

            wins_count = int(jnp.sum(flat_returns > config.get("WIN_THRESHOLD", 0.1)))
            logger.info(
                "  [%s] iter %d/%d | loss=%.4f | score=%.3f | wins=%d | win_rate=%.3f | eff_bs=%.1f",
                spec.name, iteration, max_iter, ml, ms, wins_count, win_rate, eff_bs,
            )

            if wandb_run is not None:
                ns = f"ablations/{spec.name}"
                wandb_run.log({
                    f"{ns}/train_loss": ml,
                    f"{ns}/env_score": ms,
                    f"{ns}/win_rate": win_rate,
                    f"{ns}/effective_batch_size": eff_bs,
                    "iteration": iteration,
                })
            running_loss = running_score = n_log = 0.0

        # ── Eval ──────────────────────────────────────────────────────────────
        if iteration % config["EVAL_EVERY"] == 0:
            rng, eval_rng = jax.random.split(rng)
            eval_params = state.params if not is_lora else state.params
            eval_info = eval_policy(eval_params, eval_rng)
            eval_score = float(eval_info.get("returned_episode_returns", jnp.array(0.0)))
            history.eval_iters.append(iteration)
            history.eval_score.append(eval_score)
            logger.info("  [%s] Eval score: %.4f", spec.name, eval_score)
            if wandb_run is not None:
                wandb_run.log({f"ablations/{spec.name}/eval_score": eval_score, "iteration": iteration})

    # ── Final eval ────────────────────────────────────────────────────────────
    rng, eval_rng = jax.random.split(rng)
    eval_params = state.params if not is_lora else state.params
    final_info = eval_policy(eval_params, eval_rng)
    final_score = float(final_info.get("returned_episode_returns", jnp.array(0.0)))
    logger.info("  [%s] FINAL score: %.4f", spec.name, final_score)

    # Return base params (strip LoRA wrapper if applicable)
    final_params = state.params

    return history, final_score, final_params
