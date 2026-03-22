"""Training loop: environment rollout -> diffusion window extraction -> gradient updates."""

from __future__ import annotations

import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from src.diffusion.loss import compute_loss
from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import SCHEDULE_MAP
from .data import (
    PPOAgent,
    Transition,
    build_ppo_network,
    load_ppo_params,
    make_env,
)
from .state import build_model, init_params, create_train_state, make_apply_fns
from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from .logging import make_wandb_callback


# ---------------------------------------------------------------------------
# Gradient step factory
# ---------------------------------------------------------------------------

def _action_stats(acts: jnp.ndarray, num_actions: int, valid: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Compute action-distribution entropy and unique-action fraction over valid windows.

    Args:
        acts:        [B, H] int32 action sequences.
        num_actions: Size of the real action vocabulary.
        valid:       [B] bool mask; invalid samples are excluded from counts.

    Returns:
        Dict with ``action_entropy`` and ``action_unique_frac``.
    """
    mask = jnp.broadcast_to(valid[:, None], acts.shape).reshape(-1)
    flat = jnp.where(mask, acts.reshape(-1), num_actions + 1)
    counts = jnp.bincount(flat, length=num_actions).astype(jnp.float32)
    probs = counts / jnp.maximum(counts.sum(), 1.0)
    entropy = -jnp.sum(probs * jnp.log(jnp.where(probs > 0, probs, 1.0)))
    return {
        "action_entropy": entropy,
        "action_unique_frac": jnp.sum(probs > 0).astype(jnp.float32) / num_actions,
    }


def _make_grad_step(apply_train, num_actions, schedule_fn, schedule_deriv_fn, sigma_t, label_smoothing):
    """Return a jittable function: (state, acts, obs, valid, rng, advantages) -> (state, metrics).

    Args:
        apply_train:       Model apply function with dropout enabled.
        num_actions:       Size of the action vocabulary.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        sigma_t:           ReMDM remasking strength during training.
        label_smoothing:   Cross-entropy label smoothing epsilon.

    Returns:
        A ``step`` function ready for use inside ``jax.lax.scan``.
    """

    def _loss_fn(params, acts, obs, valid, rng, advantages):
        return compute_loss(
            apply_train, params, rng, acts, obs, valid,
            num_actions, schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
            advantages=advantages,
        )

    def step(state, acts, obs, valid, rng, advantages):
        """Single gradient update step.

        Args:
            state:      Current ``TrainState``.
            acts:       [B, H] int32 action sequences.
            obs:        [B, obs_dim] float32 observations.
            valid:      [B] bool validity mask (episode-boundary filter).
            rng:        PRNG key for dropout and noise sampling.
            advantages: [B] float return weights applied per-sample before reduction.

        Returns:
            Updated ``TrainState`` and a metrics dict.
        """
        (loss, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            state.params, acts, obs, valid, rng, advantages,
        )
        state = state.apply_gradients(grads=grads)
        info["grad_norm"] = optax.tree.norm(grads)
        info.update(_action_stats(acts, num_actions, valid))
        return state, info

    return step


# ---------------------------------------------------------------------------
# make_train
# ---------------------------------------------------------------------------

def make_train(config: dict[str, Any]):
    """Build the offline diffusion training closure.

    All environment construction, model instantiation, and static pre-computation
    happen here (outside the returned ``train`` closure) so they are not repeated
    across ``jax.vmap`` replicas or JIT retraces.

    Args:
        config: Upper-cased hyperparameter dict (see ``configs/defaults.yaml``).

    Returns:
        A ``train(rng) -> dict`` closure that is safe to JIT and vmap.
    """
    num_steps = config["NUM_STEPS"]
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    val_interval = config.get("VAL_INTERVAL", 50)
    val_replan_every = config.get("VAL_REPLAN_EVERY", 4)
    val_steps = config.get("VAL_STEPS", 128)
    n_val_cycles = val_steps // val_replan_every
    valid_per_rollout = num_steps - plan_horizon + 1
    num_samples = num_envs * valid_per_rollout
    return_weight_cap = config.get("RETURN_WEIGHT_CAP", 5.0)

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // num_steps // num_envs
    assert num_samples % config["NUM_MINIBATCHES"] == 0, (
        f"{num_samples} samples not divisible by {config['NUM_MINIBATCHES']} minibatches"
    )
    config["MINIBATCH_SIZE"] = num_samples // config["NUM_MINIBATCHES"]

    # Environment
    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env, num_envs=num_envs,
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], num_envs),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=num_envs)

    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dim = obs_shape[0]

    # PPO collector
    model_type = config["PPO_MODEL_TYPE"]
    ppo_net = build_ppo_network(model_type, num_actions, config["LAYER_SIZE"], config)
    ppo_params = load_ppo_params(
        config["PPO_CHECKPOINT_PATH"], ppo_net, model_type, num_envs, obs_shape, config["LAYER_SIZE"],
    )
    ppo = PPOAgent(ppo_net, ppo_params, model_type, config["LAYER_SIZE"])

    # Noise schedule
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    # Diffusion model — pure Flax dataclass, no randomness, safe to build once.
    net = build_model(config, num_actions)
    apply_eval, apply_train = make_apply_fns(net)
    grad_step = _make_grad_step(
        apply_train, num_actions, schedule_fn, schedule_deriv_fn,
        config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
    )

    # Cosine LR decay over total gradient steps with optional linear warm-up.
    total_grad_steps = config["NUM_UPDATES"] * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    warmup_steps = config.get("LR_WARMUP_STEPS", 0)
    lr_schedule = (
        optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config["LR"],
            warmup_steps=warmup_steps,
            decay_steps=total_grad_steps,
            end_value=config["LR"] * 0.1,
        )
        if warmup_steps > 0
        else optax.cosine_decay_schedule(
            init_value=config["LR"],
            decay_steps=total_grad_steps,
            alpha=0.1,  # final LR = 10% of initial
        )
    )

    # W&B callback — one closure shared across vmap replicas (timing is per-call).
    _wandb_log = (
        make_wandb_callback(
            config,
            steps_per_update=num_steps * num_envs,
            val_interval=val_interval,
        )
        if config["USE_WANDB"] else None
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        """JIT/vmap-compatible training loop.

        Args:
            rng: JAX PRNG key (one per vmap replica).

        Returns:
            Dict with ``runner_state`` (final scan carry) and ``metrics`` (all update metrics).
        """
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = init_params(net, init_rng, obs_dim, plan_horizon)
        state = create_train_state(net, params, lr_schedule, config["MAX_GRAD_NORM"])

        obsv, env_state = env.reset(env_rng, env_params)
        init_hstate = ppo.init_hidden(num_envs)

        # ------------------------------------------------------------------
        # Validation: amortise sampling over val_replan_every env steps per plan.
        # ------------------------------------------------------------------
        def _validate(state, rng):
            rng, val_rng = jax.random.split(rng)
            val_obs, val_env_state = env.reset(val_rng, env_params)

            def _val_cycle(carry, _):
                vs, vo, rng = carry
                rng, p_rng = jax.random.split(rng)
                plan = sample_plan(
                    apply_eval, state.params, p_rng, vo,
                    num_actions, plan_horizon,
                    num_steps=config.get("VAL_DIFFUSION_STEPS", 50),
                    schedule_fn=schedule_fn, remask_strategy="cap", use_loop=True,
                )  # [num_envs, plan_horizon]

                def _exec_step(inner_carry, step_i):
                    vs_i, vo_i, r = inner_carry
                    r, s_rng = jax.random.split(r)
                    vo_next, vs_next, _, _, info = env.step(
                        s_rng, vs_i, plan[:, step_i], env_params,
                    )
                    return (vs_next, vo_next, r), info

                (vs, vo, rng), step_infos = jax.lax.scan(
                    _exec_step, (vs, vo, rng), jnp.arange(val_replan_every),
                )
                return (vs, vo, rng), step_infos

            _, cycle_infos = jax.lax.scan(
                _val_cycle, (val_env_state, val_obs, rng), None, n_val_cycles,
            )
            # cycle_infos: {key: [n_val_cycles, val_replan_every, num_envs, ...]}
            # Flatten to [val_steps, num_envs, ...] for episode-return aggregation.
            infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), cycle_infos)
            returned = infos["returned_episode"]
            metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8), infos,
            )
            return {f"val/{k}": v for k, v in metrics.items()}

        # ------------------------------------------------------------------
        # Update step
        # ------------------------------------------------------------------
        def _update_step(runner, _):
            state, env_state, last_obs, last_done, hstate, rng, step_idx = runner

            # --- Trajectory collection (state excluded from carry: not modified here) ---
            def _env_step(carry, _):
                es, obs, done, hs, rng = carry
                rng, act_rng, step_rng = jax.random.split(rng, 3)
                action, new_hs = ppo.act(
                    obs, done, hs, act_rng, temperature=config.get("COLLECT_TEMPERATURE", 1.0),
                )
                new_obs, es, reward, new_done, info = env.step(step_rng, es, action, env_params)
                t = Transition(done=done, action=action, reward=reward, obs=obs, info=info)
                return (es, new_obs, new_done, new_hs, rng), t

            (env_state, last_obs, last_done, hstate, rng), traj = jax.lax.scan(
                _env_step, (env_state, last_obs, last_done, hstate, rng), None, num_steps,
            )

            # --- Diffusion window extraction ---
            def _window(t_idx):
                obs_t = traj.obs[t_idx]
                acts = jax.lax.dynamic_slice(traj.action, (t_idx, 0), (plan_horizon, num_envs))
                # traj.done[t] marks a reset *before* step t, so traj.done[t_idx]
                # only tells us obs_t is an episode-start — it does NOT invalidate the
                # window. We check done flags strictly *inside* the action sequence.
                dones = jax.lax.dynamic_slice(
                    traj.done, (t_idx + 1, 0), (plan_horizon - 1, num_envs),
                )
                valid = ~jnp.any(dones, axis=0)

                rew_seq = jax.lax.dynamic_slice(traj.reward, (t_idx, 0), (plan_horizon, num_envs))
                window_return = jnp.sum(rew_seq, axis=0)  # [num_envs]

                return obs_t, jnp.swapaxes(acts, 0, 1), valid, window_return

            obs_w, act_w, valid_w, returns_w = jax.vmap(_window)(jnp.arange(valid_per_rollout))

            flat_obs = obs_w.reshape(-1, obs_dim)
            flat_acts = act_w.reshape(-1, plan_horizon)
            flat_valid = valid_w.reshape(-1)  # bool: episode-boundary filter

            # Return-weighted advantages: normalise by batch mean, clip to [0.1, cap].
            # Passed as per-sample multipliers into compute_loss *after* per-position
            # normalisation, so the weight correctly scales each sample's contribution.
            flat_returns = returns_w.reshape(-1)
            flat_returns_clipped = jnp.clip(flat_returns, 0.0, None)
            return_weights = flat_returns_clipped / (jnp.mean(flat_returns_clipped) + 1e-8)
            return_weights = jnp.clip(return_weights, 0.1, return_weight_cap)

            dataset = (flat_obs, flat_acts, flat_valid, return_weights)

            # --- Minibatch SGD over UPDATE_EPOCHS epochs ---
            def _epoch(epoch_state, _):
                state, ds, rng = epoch_state
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, num_samples)
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), ds)
                batches = jax.tree.map(
                    lambda x: x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:]), shuffled,
                )

                def _mb(carry, batch):
                    st, rng = carry
                    rng, loss_rng = jax.random.split(rng)
                    obs_b, act_b, val_b, adv_b = batch
                    st, metrics = grad_step(st, act_b, obs_b, val_b, loss_rng, adv_b)
                    return (st, rng), metrics

                (state, rng), metrics = jax.lax.scan(_mb, (state, rng), batches)
                return (state, ds, rng), metrics

            (state, _, rng), loss_info = jax.lax.scan(
                _epoch, (state, dataset, rng), None, config["UPDATE_EPOCHS"],
            )

            # --- Metrics ---
            metric = jax.tree.map(jnp.mean, loss_info)
            returned = traj.info["returned_episode"]
            env_metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8), traj.info,
            )
            metric.update(env_metrics)
            metric["valid_frac"] = jnp.mean(flat_valid.astype(jnp.float32))
            metric["mean_return_weight"] = jnp.mean(return_weights)

            # --- Periodic validation ---
            rng, val_rng = jax.random.split(rng)
            dummy = jax.tree.map(
                jnp.zeros_like, {f"val/{k}": v for k, v in env_metrics.items()},
            )
            val_metrics = jax.lax.cond(
                step_idx % val_interval == 0,
                lambda: _validate(state, val_rng),
                lambda: dummy,
            )
            metric.update(val_metrics)

            if _wandb_log is not None:
                jax.debug.callback(_wandb_log, metric, step_idx)

            runner = (state, env_state, last_obs, last_done, hstate, rng, step_idx + 1)
            return runner, metric

        rng, run_rng = jax.random.split(rng)
        runner_init = (
            state, env_state, obsv, jnp.zeros(num_envs, dtype=bool),
            init_hstate, run_rng, 0,
        )
        runner_final, metrics = jax.lax.scan(_update_step, runner_init, None, config["NUM_UPDATES"])
        return {"runner_state": runner_final, "metric": metrics}

    return train


# ---------------------------------------------------------------------------
# make_train_from_data — offline training from pre-collected .npz files
# ---------------------------------------------------------------------------

def make_train_from_data(config: dict[str, Any]):
    """Build an offline training closure from pre-collected trajectory files.

    Data is loaded and all valid windows are pre-computed at Python time
    (outside JIT).  The returned ``train(rng)`` closure runs a Python host
    loop — XLA only ever sees one minibatch at a time, so the full dataset
    never needs to fit in GPU VRAM.

    Args:
        config: Upper-cased hyperparameter dict.  Must contain
            ``OFFLINE_DATA_PATH``, ``PLAN_HORIZON``, ``BATCH_SIZE``,
            ``UPDATE_EPOCHS``, ``LR``, ``MAX_GRAD_NORM``, and either
            ``NUM_UPDATES`` or ``TOTAL_TIMESTEPS`` (used to derive it).

    Returns:
        A ``train(rng) -> dict`` closure that runs a Python host loop.
    """
    plan_horizon = config["PLAN_HORIZON"]
    val_interval = config.get("VAL_INTERVAL", 50)
    val_replan_every = config.get("VAL_REPLAN_EVERY", 4)
    val_steps = config.get("VAL_STEPS", 128)
    n_val_cycles = val_steps // val_replan_every
    return_weight_cap = config.get("RETURN_WEIGHT_CAP", 5.0)
    batch_size = config["BATCH_SIZE"]

    # -- Load data and pre-compute sliding windows (Python / NumPy time) -------
    data = np.load(config["OFFLINE_DATA_PATH"])
    obs_np    = data["obs"]      # [E, T, obs_dim]
    acts_np   = data["actions"]  # [E, T]
    dones_np  = data["dones"]    # [E, T]
    rewards_np = data["rewards"] if "rewards" in data else None

    E, T, obs_dim = obs_np.shape
    H = plan_horizon
    assert T >= H, f"Data length {T} < plan_horizon {H}; cannot form any windows."
    valid_per_env = T - H + 1
    total_samples = E * valid_per_env

    # obs at window start: [E, W, D]  (slice, not a copy)
    obs_w = obs_np[:, :valid_per_env, :]

    # action windows: [E, W, H]
    acts_w = np.lib.stride_tricks.sliding_window_view(acts_np, H, axis=1)

    # validity: no done flag strictly inside the window (positions t+1 .. t+H-1)
    if H > 1:
        dones_inner = np.lib.stride_tricks.sliding_window_view(
            dones_np[:, 1:], H - 1, axis=1,
        )  # [E, W, H-1]
        valid_w = ~np.any(dones_inner, axis=-1)  # [E, W]
    else:
        valid_w = np.ones((E, valid_per_env), dtype=bool)

    # window returns: uniform weight 1 when rewards were not collected
    if rewards_np is not None:
        rew_w = np.lib.stride_tricks.sliding_window_view(rewards_np, H, axis=1)
        returns_w = rew_w.sum(axis=-1).astype(np.float32)  # [E, W]
    else:
        returns_w = np.ones((E, valid_per_env), dtype=np.float32)

    # Flatten and normalise return weights — stay on CPU as numpy throughout
    obs_flat   = obs_w.reshape(-1, obs_dim).astype(np.float32)
    acts_flat  = acts_w.reshape(-1, H).astype(np.int32)
    valid_flat = valid_w.reshape(-1)
    clipped    = np.clip(returns_w.reshape(-1), 0.0, None)
    rw         = np.clip(clipped / (clipped.mean() + 1e-8), 0.1, return_weight_cap).astype(np.float32)

    # Pre-compute scalars for metric dicts (avoids per-step numpy reductions)
    _valid_frac         = float(valid_flat.mean())
    _mean_return_weight = float(rw.mean())

    # -- Environment (validation only) ----------------------------------------
    num_val_envs = config["NUM_ENVS"]
    env, env_params = make_env(config, num_val_envs)
    num_actions = env.action_space(env_params).n
    assert obs_dim == env.observation_space(env_params).shape[0], (
        f"Data obs_dim {obs_dim} != env obs_dim "
        f"{env.observation_space(env_params).shape[0]}"
    )

    # -- Schedule, model, optimizer -------------------------------------------
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    net = build_model(config, num_actions)
    apply_eval, apply_train = make_apply_fns(net)

    # JIT a single gradient step — XLA only ever traces one minibatch at a time.
    grad_step = jax.jit(_make_grad_step(
        apply_train, num_actions, schedule_fn, schedule_deriv_fn,
        config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
    ))

    if "NUM_UPDATES" not in config:
        config["NUM_UPDATES"] = max(
            1, config.get("TOTAL_TIMESTEPS", total_samples) // total_samples,
        )

    # LR schedule: total steps = NUM_UPDATES × UPDATE_EPOCHS (no minibatch axis)
    total_grad_steps = config["NUM_UPDATES"] * config["UPDATE_EPOCHS"]
    warmup_steps = config.get("LR_WARMUP_STEPS", 0)
    lr_schedule = (
        optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config["LR"],
            warmup_steps=warmup_steps,
            decay_steps=total_grad_steps,
            end_value=config["LR"] * 0.1,
        )
        if warmup_steps > 0
        else optax.cosine_decay_schedule(
            init_value=config["LR"], decay_steps=total_grad_steps, alpha=0.1,
        )
    )

    _wandb_log = (
        make_wandb_callback(
            config,
            steps_per_update=None,  # no env frames consumed in data-replay mode
            val_interval=val_interval,
        )
        if config.get("USE_WANDB") else None
    )

    # JIT'd validation closure — pure JAX, no Python data access
    def _validate_fn(state, rng):
        rng, val_rng = jax.random.split(rng)
        val_obs, val_env_state = env.reset(val_rng, env_params)

        def _val_cycle(carry, _):
            vs, vo, rng = carry
            rng, p_rng = jax.random.split(rng)
            plan = sample_plan(
                apply_eval, state.params, p_rng, vo,
                num_actions, plan_horizon,
                num_steps=config.get("VAL_DIFFUSION_STEPS", 50),
                schedule_fn=schedule_fn, remask_strategy="cap", use_loop=True,
            )  # [num_val_envs, plan_horizon]

            def _exec_step(inner_carry, step_i):
                vs_i, vo_i, r = inner_carry
                r, s_rng = jax.random.split(r)
                vo_next, vs_next, _, _, info = env.step(
                    s_rng, vs_i, plan[:, step_i], env_params,
                )
                return (vs_next, vo_next, r), info

            (vs, vo, rng), step_infos = jax.lax.scan(
                _exec_step, (vs, vo, rng), jnp.arange(val_replan_every),
            )
            return (vs, vo, rng), step_infos

        _, cycle_infos = jax.lax.scan(
            _val_cycle, (val_env_state, val_obs, rng), None, n_val_cycles,
        )
        infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), cycle_infos)
        returned = infos["returned_episode"]
        metrics = jax.tree.map(
            lambda x: (x * returned).sum() / (returned.sum() + 1e-8), infos,
        )
        return {f"val/{k}": v for k, v in metrics.items()}

    _validate_jit = jax.jit(_validate_fn)

    def train(rng: jax.Array) -> dict[str, Any]:
        """Python host-loop training from a fixed window dataset.

        The outer update loop runs in Python; XLA only sees one minibatch at
        a time.  The full dataset remains in CPU RAM throughout, so GPU VRAM
        usage is bounded by a single minibatch regardless of dataset size.

        Args:
            rng: JAX PRNG key.

        Returns:
            Dict with ``runner_state`` (final ``TrainState``) and empty ``metrics``.
        """
        seed = int(jax.random.randint(rng, (), 0, 2**31))
        np_rng = np.random.default_rng(seed)
        rng, init_rng = jax.random.split(rng)
        params = init_params(net, init_rng, obs_dim, plan_horizon)
        state = create_train_state(net, params, lr_schedule, config["MAX_GRAD_NORM"])

        for update_idx in range(config["NUM_UPDATES"]):
            epoch_metrics: list[dict[str, float]] = []

            for _ in range(config["UPDATE_EPOCHS"]):
                idx    = np_rng.integers(0, total_samples, size=batch_size)
                obs_b  = jnp.array(obs_flat[idx])
                acts_b = jnp.array(acts_flat[idx])
                val_b  = jnp.array(valid_flat[idx])
                adv_b  = jnp.array(rw[idx])
                rng, loss_rng = jax.random.split(rng)
                state, step_m = grad_step(state, acts_b, obs_b, val_b, loss_rng, adv_b)
                epoch_metrics.append(jax.device_get(step_m))

            metric: dict[str, float] = {
                k: float(np.mean([m[k] for m in epoch_metrics]))
                for k in epoch_metrics[0]
            }
            metric["valid_frac"]         = _valid_frac
            metric["mean_return_weight"] = _mean_return_weight

            if update_idx % val_interval == 0:
                rng, val_rng = jax.random.split(rng)
                val_m = jax.device_get(_validate_jit(state, val_rng))
                metric.update({k: float(v) for k, v in val_m.items()})

            if _wandb_log is not None:
                _wandb_log(metric, update_idx)

        return {"runner_state": state, "metrics": {}}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_offline_diffusion(config):
    """Configure, compile, and run offline diffusion training.

    Routes to ``make_train_from_data`` when ``OFFLINE_DATA_PATH`` is set,
    otherwise to ``make_train`` (live PPO rollout).

    Args:
        config: Lower-cased hyperparameter dict from ``defaults.yaml`` / CLI.
    """
    config = {k.upper(): v for k, v in config.items()}

    if config.get("USE_WANDB"):
        if config.get("OFFLINE_DATA_PATH"):
            data_mode = "offline-dataset"
        else:
            data_mode = "offline-ppo"

        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config.get("WANDB_ENTITY"),
            config=config,
            name=f"{config['ENV_NAME']}-OfflineDiffusion-{data_mode}",
        )
    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    if config.get("OFFLINE_DATA_PATH"):
        # Python host loop — not JIT/vmap'd at the outer level; dataset stays on CPU.
        train_fn = make_train_from_data(config)
        t0 = time.time()
        out = train_fn(rngs[0])
    else:
        train_fn = jax.jit(jax.vmap(make_train(config)))
        t0 = time.time()
        out = train_fn(rngs)
    elapsed = time.time() - t0

    num_updates = config.get("NUM_UPDATES", 0)
    print(f"Time: {elapsed:.1f}s  Updates: {num_updates}  "
          f"Grad-steps/s: {num_updates * config['UPDATE_EPOCHS'] / elapsed:.0f}")

    if config.get("USE_WANDB") and config.get("SAVE_POLICY"):
        if config.get("OFFLINE_DATA_PATH"):
            # Python-loop mode: runner_state is the TrainState directly.
            train_state = out["runner_state"]
        else:
            train_states = out["runner_state"][0]
            train_state = jax.tree.map(lambda x: x[0], train_states)
        path = os.path.join(wandb.run.dir, "policies")
        with ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(max_to_keep=1)) as mgr:
            mgr.save(num_updates, args=ocp.args.StandardSave(train_state))
        print(f"Saved policy to {path}")
