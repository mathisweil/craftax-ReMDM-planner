from __future__ import annotations

import pathlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax.training.train_state import TrainState

from src.models.remdm import (
    ScheduleFn,
    compute_loss,
    cosine_schedule,
    linear_schedule,
    sample_plan,
)
from Craftax_Baselines.wrappers import (
    AutoResetEnvWrapper,
    BatchEnvWrapper,
    LogWrapper,
)
from src.envs.wrappers import PlannerWrapper
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_checkpoint,
    _load_ppo_checkpoint,
    _save_model,
    _make_env_stack,
    _valid_window_mask,
    _sample_windows_from_chunk,
    _make_apply_fns,
)

SCHEDULE_MAP: Dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# =============================================================================
# Shared Helpers
# =============================================================================


def _global_grad_norm(grads) -> jnp.ndarray:
    """L2 norm across all gradient leaves."""
    leaves = jax.tree.leaves(grads)
    return jnp.sqrt(sum(jnp.sum(g ** 2) for g in leaves))


def _action_stats(act_batch: jnp.ndarray, num_actions: int) -> Dict[str, jnp.ndarray]:
    """Compute action distribution entropy and mode fraction over a batch of action sequences."""
    counts = jnp.zeros(num_actions, dtype=jnp.float32)
    flat = act_batch.reshape(-1)
    counts = jnp.bincount(flat, length=num_actions).astype(jnp.float32)
    probs = counts / jnp.maximum(counts.sum(), 1.0)
    log_probs = jnp.where(probs > 0, jnp.log(probs), 0.0)
    entropy = -jnp.sum(probs * log_probs)
    unique_frac = jnp.sum(probs > 0).astype(jnp.float32) / num_actions
    return {"action_entropy": entropy, "action_unique_frac": unique_frac}


def _make_grad_step(apply_train, num_actions: int, schedule_fn, sigma_t: float):
    """Return a pure function: (train_state, act_batch, obs_batch, rng) -> (train_state, info).

    `info` includes loss components + grad_norm + action stats.
    """

    def grad_step(train_state, act_batch, obs_batch, rng):
        def loss_fn(params):
            return compute_loss(
                apply_train, params, rng,
                act_batch, obs_batch, num_actions, schedule_fn,
                sigma_t=sigma_t,
            )

        (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
        info["grad_norm"] = _global_grad_norm(grads)
        info.update(_action_stats(act_batch, num_actions))
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, info

    return grad_step


def _wandb_log_callback(metric_dict: Dict[str, jnp.ndarray], step: jnp.ndarray, prefix: str) -> None:
    """Host-side wandb.log callback — safe to call from jax.debug.callback."""
    payload = {f"{prefix}/{k}": float(v) for k, v in metric_dict.items()}
    payload[f"{prefix}/step"] = int(step)
    wandb.log(payload, step=int(step))


def _maybe_log(config, metrics: Dict[str, jnp.ndarray], step: jnp.ndarray, prefix: str):
    """Conditionally emit a jax.debug.callback for wandb logging."""
    if config.get("USE_WANDB"):
        jax.debug.callback(_wandb_log_callback, metrics, step, prefix)


# =============================================================================
# Data Collection  (step 2a — saves trajectories to disk)
# =============================================================================


def collect_offline_data(config: Dict[str, Any]) -> None:
    """Roll out a pre-trained PPO checkpoint and save (obs, actions, dones) to disk."""
    assert config.get("PPO_CHECKPOINT_PATH"), (
        "--ppo_checkpoint_path is required for --mode collect.\n"
        "Train a PPO agent first with ppo_rnn.py or ppo_rnd.py."
    )

    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    num_envs: int = config["COLLECT_NUM_ENVS"]
    num_iters: int = config["COLLECT_NUM_STEPS"] // num_envs

    ppo_agent = _load_ppo_checkpoint(
        config["PPO_CHECKPOINT_PATH"], num_actions, obs_dim,
        config.get("LAYER_SIZE", 512),
        model_type=config.get("PPO_MODEL_TYPE"),
    )
    is_rnn = ppo_agent.model_type == "ppo_rnn"
    env_w, _ = _make_env_stack(config, num_envs)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, env_rng, collect_rng = jax.random.split(rng, 3)
    obs, env_state = env_w.reset(env_rng, env_params)
    done = jnp.zeros(num_envs, dtype=bool)
    hidden = ppo_agent.init_hidden(num_envs)

    def _step_fn(carry, _):
        rng, env_state, obs, done, hidden = carry
        rng, k1, k2 = jax.random.split(rng, 3)
        if is_rnn:
            pi, _, new_hidden = ppo_agent.apply(ppo_agent.params, obs, hidden=hidden, done=done)
        else:
            pi, _, _ = ppo_agent.apply(ppo_agent.params, obs)
            new_hidden = hidden
        action = pi.sample(seed=k1)
        obs_next, env_state, _, done_next, _ = env_w.step(k2, env_state, action, env_params)
        return (rng, env_state, obs_next, done_next, new_hidden), (obs, action, done)

    rollout_fn = jax.jit(lambda c: jax.lax.scan(_step_fn, c, None, length=num_iters))
    _, (obs_arr, act_arr, done_arr) = rollout_fn((collect_rng, env_state, obs, done, hidden))

    obs_arr, act_arr, done_arr = (
        np.array(obs_arr).transpose(1, 0, 2),
        np.array(act_arr).transpose(1, 0),
        np.array(done_arr).transpose(1, 0),
    )

    out_path = config["OFFLINE_DATA_PATH"]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs_arr, actions=act_arr, dones=done_arr)
    total = obs_arr.shape[0] * obs_arr.shape[1]
    print(f"Saved {obs_arr.shape[0]}x{obs_arr.shape[1]} transitions ({total:,} total) to '{out_path}'")


# =============================================================================
# Offline Training — from saved trajectories  (step 3)
# =============================================================================


def make_train_offline(
    config: Dict[str, Any],
    offline_data: Dict[str, np.ndarray],
) -> Tuple[Callable, Dict[str, jnp.ndarray]]:
    """Return ``(train_fn, data_arrays)`` for offline MDLM training.

    The data arrays are returned separately so they can be passed as explicit
    arguments to the JIT-compiled ``train_fn``, avoiding the capture of ~5GB
    of constants into the XLA computation.

    Usage::

        train_fn, data = make_train_offline(config, offline_data)
        train_jit = jax.jit(train_fn)
        outs = train_jit(rng, data)
    """
    raw_obs, raw_act, raw_done = offline_data["obs"], offline_data["actions"], offline_data["dones"]
    num_envs_data, traj_len, obs_dim = raw_obs.shape

    plan_horizon: int = config["PLAN_HORIZON"]
    batch_size: int = config["BATCH_SIZE"]
    num_actions: int = config["NUM_ACTIONS"]
    num_train_steps: int = config["NUM_TRAIN_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    valid = _valid_window_mask(raw_done, plan_horizon)
    env_indices, time_indices = np.where(valid)
    num_valid = len(env_indices)
    assert num_valid > 0, (
        f"No valid {plan_horizon}-step windows (traj_len={traj_len}). Check episode boundary density."
    )
    print(f"Offline data: {num_valid:,} valid {plan_horizon}-step windows from {num_envs_data}x{traj_len}")

    # Keep on device but pass explicitly — NOT closed over inside train().
    data_arrays = {
        "obs": jnp.array(raw_obs, dtype=jnp.float32),
        "act": jnp.array(raw_act, dtype=jnp.int32),
        "env_idx": jnp.array(env_indices, dtype=jnp.int32),
        "time_idx": jnp.array(time_indices, dtype=jnp.int32),
    }

    model = _build_model(config, num_actions)
    _, apply_train = _make_apply_fns(model)
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    def train(rng: jax.Array, data: Dict[str, jnp.ndarray]) -> Dict[str, Any]:
        obs_data, act_data = data["obs"], data["act"]
        env_idx_arr, time_idx_arr = data["env_idx"], data["time_idx"]

        rng, init_rng = jax.random.split(rng)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        def _train_step(carry, step_idx):
            train_state, rng = carry
            rng, sample_rng, loss_rng = jax.random.split(rng, 3)

            flat_idxs = jax.random.randint(sample_rng, (batch_size,), 0, num_valid)
            sel_env, sel_time = env_idx_arr[flat_idxs], time_idx_arr[flat_idxs]

            obs_batch = obs_data[sel_env, sel_time]
            act_batch = jax.vmap(
                lambda row, t: jax.lax.dynamic_slice(row, (t,), (plan_horizon,))
            )(act_data[sel_env], sel_time)

            train_state, info = grad_step(train_state, act_batch, obs_batch, loss_rng)
            _maybe_log(config, info, step_idx, "offline")
            # Only carry the scalar loss through the scan to avoid OOM from
            # accumulating full metric dicts across all training steps.
            return (train_state, rng), info["loss"]

        (train_state, _), scan_losses = jax.lax.scan(
            _train_step, (train_state, rng), jnp.arange(num_train_steps)
        )
        return {"train_state": train_state, "final_loss": scan_losses[-1]}

    return train, data_arrays


# =============================================================================
# Offline Training — live collection from PPO agent  (step 2b)
# =============================================================================


def _make_collect_chunk_fn(ppo_agent, env_w, env_params, collect_steps: int):
    """Return a unified JIT-compiled collection function for RNN or MLP PPO agents."""
    is_rnn = ppo_agent.model_type == "ppo_rnn"

    @jax.jit
    def _collect_chunk(rng, env_state, obs, done, hidden):
        def _step(carry, _):
            rng, env_state, obs, done, hidden = carry
            rng, k1, k2 = jax.random.split(rng, 3)
            if is_rnn:
                pi, _, new_hidden = ppo_agent.apply(ppo_agent.params, obs, hidden=hidden, done=done)
            else:
                pi, _, _ = ppo_agent.apply(ppo_agent.params, obs)
                new_hidden = hidden
            action = pi.sample(seed=k1)
            obs_next, env_state, _, done_next, _ = env_w.step(k2, env_state, action, env_params)
            return (rng, env_state, obs_next, done_next, new_hidden), (obs, action, done)

        (rng, env_state, obs, done, hidden), (all_obs, all_acts, all_dones) = jax.lax.scan(
            _step, (rng, env_state, obs, done, hidden), None, collect_steps
        )
        return rng, env_state, obs, done, hidden, all_obs, all_acts, all_dones

    return _collect_chunk


def make_train_offline_from_agent(
    config: Dict[str, Any],
    ppo_checkpoint_path: str,
) -> Callable[[jax.Array], Dict[str, Any]]:
    """Train the diffusion model using a PPO agent for live data collection (Python loop)."""
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions: int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    plan_horizon: int = config["PLAN_HORIZON"]
    batch_size: int = config["BATCH_SIZE"]
    num_train_steps: int = config["NUM_TRAIN_STEPS"]
    num_envs: int = config.get("COLLECT_NUM_ENVS", config["NUM_ENVS"])
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    collect_steps = max(plan_horizon * 2, (batch_size // num_envs + 2) * plan_horizon)

    ppo_agent = _load_ppo_checkpoint(
        ppo_checkpoint_path, num_actions, obs_dim,
        config.get("LAYER_SIZE", 512),
        model_type=config.get("PPO_MODEL_TYPE"),
    )
    env_w, _ = _make_env_stack(config, num_envs)
    model = _build_model(config, num_actions)
    _, apply_train = _make_apply_fns(model)
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    _collect_chunk = _make_collect_chunk_fn(ppo_agent, env_w, env_params, collect_steps)

    @jax.jit
    def _jit_grad_step(train_state, obs_batch, act_batch, rng):
        return grad_step(train_state, act_batch, obs_batch, rng)

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        obs, env_state = env_w.reset(env_rng, env_params)
        done = jnp.zeros(num_envs, dtype=bool)
        hidden = ppo_agent.init_hidden(num_envs)

        rng, np_seed_rng = jax.random.split(rng)
        np_rng = np.random.default_rng(int(jax.random.randint(np_seed_rng, (), 0, 2 ** 31 - 1)))

        use_wandb = config.get("USE_WANDB", False)
        t0 = time.time()

        for step in range(num_train_steps):
            rng, collect_rng, loss_rng = jax.random.split(rng, 3)
            _, env_state, obs, done, hidden, chunk_obs, chunk_acts, chunk_dones = (
                _collect_chunk(collect_rng, env_state, obs, done, hidden)
            )

            chunk_obs_np = np.array(chunk_obs).transpose(1, 0, 2)
            chunk_acts_np = np.array(chunk_acts).T
            chunk_dones_np = np.array(chunk_dones).T

            result = _sample_windows_from_chunk(
                chunk_obs_np, chunk_acts_np, chunk_dones_np,
                plan_horizon, batch_size, np_rng,
            )
            if result is None:
                continue

            obs_batch, act_batch = result
            train_state, info = _jit_grad_step(train_state, obs_batch, act_batch, loss_rng)

            if use_wandb:
                payload = {f"offline/{k}": float(v) for k, v in info.items()}
                payload["offline/step"] = step
                elapsed = time.time() - t0
                payload["offline/steps_per_second"] = (step + 1) / max(elapsed, 1e-6)
                wandb.log(payload, step=step)

            if (step + 1) % 1000 == 0:
                elapsed = time.time() - t0
                print(f"  [{step + 1:>6}/{num_train_steps}] loss={float(info['loss']):.4f}  elapsed={elapsed:.0f}s")

        return {"train_state": train_state}

    return train


# =============================================================================
# Online Training  (step 4)
# =============================================================================


def make_train_online(
    config: Dict[str, Any],
    init_params: Optional[Any] = None,
) -> Callable[[jax.Array], Dict[str, Any]]:
    """Return ``train(rng)`` for online fine-tuning with the diffusion model as policy."""
    assert config["NUM_STEPS"] % config["REPLAN_EVERY"] == 0, "NUM_STEPS must be divisible by REPLAN_EVERY"

    num_envs: int = config["NUM_ENVS"]
    plan_horizon: int = config["PLAN_HORIZON"]
    replan_every: int = config["REPLAN_EVERY"]
    num_updates: int = config["NUM_UPDATES"]
    update_epochs: int = config["UPDATE_EPOCHS"]
    num_minibatches: int = config["NUM_MINIBATCHES"]
    diffusion_steps: int = config["DIFFUSION_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy: str = config["REMASK_STRATEGY"]
    eta: float = config["ETA"]
    t_on: float = config.get("T_ON", 0.7)
    t_off: float = config.get("T_OFF", 0.3)
    use_loop: bool = config.get("USE_LOOP", False)
    temperature: float = config.get("TEMPERATURE", 1.0)
    top_p: Optional[float] = config.get("TOP_P", None)
    num_plan_cycles: int = config["NUM_STEPS"] // replan_every
    use_optimistic_resets: bool = config.get("USE_OPTIMISTIC_RESETS", False)

    num_actions: int = config["NUM_ACTIONS"]
    obs_dim: int = config["OBS_DIM"]

    env_w, env_params = _make_env_stack(
        config, num_envs,
        use_optimistic_resets=use_optimistic_resets,
        use_sequence_history=True,
    )
    model = _build_model(config, num_actions)
    apply_inference, apply_train = _make_apply_fns(model)
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    total_samples = num_plan_cycles * num_envs
    assert total_samples % num_minibatches == 0, (
        f"num_plan_cycles * num_envs ({total_samples}) must be divisible by NUM_MINIBATCHES ({num_minibatches})"
    )
    minibatch_size = total_samples // num_minibatches

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = init_params if init_params is not None else _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])
        obs, env_state = env_w.reset(env_rng, env_params)

        def _update_step(runner_state, update_step):
            train_state, env_state, obs, rng = runner_state

            # -- Collect: generate plans and execute --
            def _plan_and_execute(carry, _):
                env_state, obs, rng = carry
                rng, plan_rng = jax.random.split(rng)
                plan = sample_plan(
                    apply_inference, train_state.params, plan_rng, obs,
                    num_actions, plan_horizon, diffusion_steps, schedule_fn,
                    remask_strategy, eta, use_loop, t_on, t_off, temperature, top_p,
                )

                def _exec_step(carry, step_idx):
                    env_state, _, rng = carry
                    rng, step_rng = jax.random.split(rng)
                    obs_next, env_state, reward, done, info = env_w.step(
                        step_rng, env_state, plan[:, step_idx], env_params
                    )
                    return (env_state, obs_next, rng), (reward, done, info)

                (env_state, obs_next, rng), (rewards, dones, infos) = jax.lax.scan(
                    _exec_step, (env_state, obs, rng), jnp.arange(replan_every),
                )
                return (env_state, obs_next, rng), (obs, plan, rewards, dones, infos)

            (env_state, obs, rng), traj = jax.lax.scan(
                _plan_and_execute, (env_state, obs, rng), None, num_plan_cycles,
            )
            traj_obs, traj_plans, traj_rewards, traj_dones, all_infos = traj

            # -- Flatten collected data --
            flat_obs = traj_obs.reshape(total_samples, obs_dim)
            flat_plans = traj_plans.reshape(total_samples, plan_horizon)

            # -- Train on collected (obs, plan) pairs --
            def _update_epoch(carry, _):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, total_samples)
                obs_mbs = flat_obs[perm].reshape(num_minibatches, minibatch_size, obs_dim)
                plan_mbs = flat_plans[perm].reshape(num_minibatches, minibatch_size, plan_horizon)

                def _update_minibatch(ts_rng, idx_and_mb):
                    ts, rng = ts_rng
                    mb_idx, obs_mb, plan_mb = idx_and_mb
                    loss_rng = jax.random.fold_in(rng, mb_idx)
                    ts, info = grad_step(ts, plan_mb, obs_mb, loss_rng)
                    return (ts, rng), info

                (train_state, rng), infos = jax.lax.scan(
                    _update_minibatch,
                    (train_state, rng),
                    (jnp.arange(num_minibatches), obs_mbs, plan_mbs),
                )
                return (train_state, rng), infos

            (train_state, rng), epoch_infos = jax.lax.scan(
                _update_epoch, (train_state, rng), None, update_epochs
            )

            # -- Metrics --
            ep_returns = all_infos["returned_episode_returns"]
            ep_lengths = all_infos.get("returned_episode_lengths", jnp.zeros_like(ep_returns))
            ep_mask = all_infos["returned_episode"]
            n_completed = ep_mask.sum()
            safe_n = jnp.maximum(n_completed, 1)

            metric = {
                "diffusion_loss": epoch_infos["loss"].mean(),
                "mean_t": epoch_infos["mean_t"].mean(),
                "frac_masked": epoch_infos["frac_masked"].mean(),
                "grad_norm": epoch_infos["grad_norm"].mean(),
                "action_entropy": epoch_infos["action_entropy"].mean(),
                "action_unique_frac": epoch_infos["action_unique_frac"].mean(),
                "episode_return": jnp.where(n_completed > 0, (ep_returns * ep_mask).sum() / safe_n, jnp.nan),
                "episode_length": jnp.where(n_completed > 0, (ep_lengths * ep_mask).sum() / safe_n, jnp.nan),
                "num_completed_eps": n_completed,
                "mean_step_reward": traj_rewards.mean(),
                "reward_std": traj_rewards.std(),
                "plan_diversity": jax.vmap(
                    lambda p: jnp.sum(jnp.bincount(p, length=num_actions) > 0).astype(jnp.float32) / plan_horizon
                )(flat_plans).mean(),
            }

            _maybe_log(config, metric, update_step, "online")
            return (train_state, env_state, obs, rng), metric

        runner_state = (train_state, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# =============================================================================
# Entry Points
# =============================================================================


def run_collect(config: Dict[str, Any]) -> None:
    """Collect offline data from a PPO agent and save to disk."""
    collect_offline_data(config)


def run_offline(config: Dict[str, Any]) -> None:
    """Offline training dispatcher."""
    if config.get("USE_WANDB"):
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"remdm-offline-{config['ENV_NAME']}",
        )

    if config.get("PPO_CHECKPOINT_PATH"):
        print(f"Offline training: live collection from PPO ({config['PPO_CHECKPOINT_PATH']})")
        train_fn = make_train_offline_from_agent(config, config["PPO_CHECKPOINT_PATH"])
        rng = jax.random.PRNGKey(config["SEED"])
        t0 = time.time()
        outs = [train_fn(jax.random.fold_in(rng, i)) for i in range(config["NUM_REPEATS"])]
        elapsed = time.time() - t0
        print(f"Offline training time: {elapsed:.1f}s")
        if config["SAVE_POLICY"]:
            _save_model(outs[0]["train_state"], config, "diffusion_offline")
    else:
        assert config.get("OFFLINE_DATA_PATH"), (
            "Either --ppo_checkpoint_path or --offline_data_path must be provided for --mode offline."
        )
        print(f"Offline training: loading trajectories from '{config['OFFLINE_DATA_PATH']}'")

        env = make_craftax_env_from_name(config["ENV_NAME"], True)
        config["NUM_ACTIONS"] = env.action_space(env.default_params).n

        offline_data = dict(np.load(config["OFFLINE_DATA_PATH"]))
        if offline_data["obs"].ndim == 2:
            print("WARNING: flat data format — reshaping to [1, N, obs_dim].")
            for k in ("obs", "actions"):
                offline_data[k] = offline_data[k][np.newaxis]
            offline_data["dones"] = offline_data.get(
                "dones", np.zeros_like(offline_data["actions"], dtype=bool)
            )[np.newaxis]

        n_envs, n_steps, obs_dim = offline_data["obs"].shape
        print(f"Loaded {n_envs}x{n_steps} transitions (obs_dim={obs_dim}, num_actions={config['NUM_ACTIONS']})")

        rng = jax.random.PRNGKey(config["SEED"])
        train_fn, data_arrays = make_train_offline(config, offline_data)
        train_jit = jax.jit(train_fn)
        t0 = time.time()
        outs = [train_jit(jax.random.fold_in(rng, i), data_arrays) for i in range(config["NUM_REPEATS"])]
        elapsed = time.time() - t0
        print(f"Offline training time: {elapsed:.1f}s")

        if config.get("USE_WANDB"):
            wandb.log({"offline/final_loss": float(outs[0]["final_loss"])})

        if config["SAVE_POLICY"]:
            _save_model(outs[0]["train_state"], config, "diffusion_offline")


def run_online(config: Dict[str, Any]) -> None:
    """Online fine-tuning: diffusion model collects its own data."""
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = env.action_space(env.default_params).n
    config["OBS_DIM"] = env.observation_space(env.default_params).shape[0]

    if config.get("USE_WANDB"):
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"remdm-online-{config['ENV_NAME']}",
        )

    init_params: Optional[Any] = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        model = _build_model(config, config["NUM_ACTIONS"])
        init_params = _load_checkpoint(config, model, config["OBS_DIM"], config["OFFLINE_CHECKPOINT_PATH"])

    rng = jax.random.PRNGKey(config["SEED"])
    train_jit = jax.jit(make_train_online(config, init_params=init_params))

    t0 = time.time()
    outs = [train_jit(jax.random.fold_in(rng, i)) for i in range(config["NUM_REPEATS"])]
    elapsed = time.time() - t0

    total_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
    sps = total_steps / max(elapsed, 1e-6)
    print(f"Online training time: {elapsed:.1f}s | SPS: {sps:.0f}")

    if config.get("USE_WANDB"):
        wandb.log({"online/total_sps": sps, "online/total_time_s": elapsed})

    if config["SAVE_POLICY"]:
        _save_model(outs[0]["runner_state"][0], config, "diffusion_online")


# =============================================================================
# Inference  (step 5)
# =============================================================================


def run_inference(config: Dict[str, Any]) -> None:
    """Evaluate a trained diffusion planner using ``PlannerWrapper``."""
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions: int = env.action_space(env_params).n
    obs_dim: int = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs: int = config["NUM_ENVS"]
    plan_horizon: int = config["PLAN_HORIZON"]
    replan_every: int = config["REPLAN_EVERY"]
    diffusion_steps: int = config["DIFFUSION_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy: str = config["REMASK_STRATEGY"]
    eta: float = config["ETA"]
    t_on: float = config.get("T_ON", 0.7)
    t_off: float = config.get("T_OFF", 0.3)
    use_loop: bool = config.get("USE_LOOP", False)
    temperature: float = config.get("TEMPERATURE", 1.0)
    top_p: Optional[float] = config.get("TOP_P", None)
    eval_steps: int = config.get("EVAL_STEPS", 1000)

    model = _build_model(config, num_actions)
    apply_inference, _ = _make_apply_fns(model)

    assert config.get("CHECKPOINT_PATH"), "--checkpoint_path required for inference"
    model_params = _load_checkpoint(config, model, obs_dim, config["CHECKPOINT_PATH"])

    def planner_apply_fn(rng, model_params, obs):
        return sample_plan(
            apply_inference, model_params, rng, obs,
            num_actions, plan_horizon, diffusion_steps, schedule_fn,
            remask_strategy, eta, use_loop, t_on, t_off, temperature, top_p,
        )

    env_w = PlannerWrapper(
        BatchEnvWrapper(AutoResetEnvWrapper(LogWrapper(env)), num_envs=num_envs),
        num_envs=num_envs,
        plan_horizon=plan_horizon,
        replan_every=replan_every,
        planner_apply_fn=planner_apply_fn,
    )

    rng = jax.random.PRNGKey(config["SEED"])

    @jax.jit
    def _eval_loop(rng):
        rng, env_rng = jax.random.split(rng)
        obs, state = env_w.reset(env_rng, env_params)

        def _step(carry, _):
            obs, state, rng = carry
            rng, step_rng = jax.random.split(rng)
            obs, state, action, reward, done, info = env_w.step(
                step_rng, state, obs, model_params, env_params,
            )
            return (obs, state, rng), (reward, done, info)

        _, (rewards, dones, infos) = jax.lax.scan(_step, (obs, state, rng), None, eval_steps)
        return rewards, dones, infos

    t0 = time.time()
    rewards, dones, infos = _eval_loop(rng)
    elapsed = time.time() - t0

    ep_returns, ep_mask = infos["returned_episode_returns"], infos["returned_episode"]
    completed = ep_mask.sum()
    mean_return = jnp.where(completed > 0, (ep_returns * ep_mask).sum() / completed, jnp.nan)

    print(f"Eval time: {elapsed:.1f}s  ({eval_steps * num_envs} steps)")
    print(f"Completed episodes: {int(completed)} | Mean return: {float(mean_return):.2f} | Mean step reward: {float(rewards.mean()):.4f}")

    if config.get("USE_WANDB"):
        wandb.log({
            "eval/mean_return": float(mean_return),
            "eval/completed_episodes": int(completed),
            "eval/mean_step_reward": float(rewards.mean()),
            "eval/sps": eval_steps * num_envs / max(elapsed, 1e-6),
        })
