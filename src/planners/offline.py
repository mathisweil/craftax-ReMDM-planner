import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from .common import SCHEDULE_MAP, _make_grad_step
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_ppo_checkpoint,
    _make_env_stack,
    _valid_window_mask,
    _make_apply_fns,
    _make_periodic_ckpt_manager,
    _resolve_ckpt_dir,
)

def make_train_offline(
    config: dict[str, Any],
    offline_data: dict[str, np.ndarray],
) -> Callable:                          # <-- no longer returns data_arrays separately
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

    data_arrays = {
        "obs": jnp.array(raw_obs, dtype=jnp.float32),
        "act": jnp.array(raw_act, dtype=jnp.int32),
        "env_idx": jnp.array(env_indices, dtype=jnp.int32),
        "time_idx": jnp.array(time_indices, dtype=jnp.int32),
    }

    model = _build_model(config, num_actions)
    _, apply_train = _make_apply_fns(model)
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    train_steps_per_chunk = min(num_train_steps, 100)
    num_chunks = (num_train_steps + train_steps_per_chunk - 1) // train_steps_per_chunk

    @jax.jit
    def _train_chunk(train_state, rng, data):
        obs_data, act_data = data["obs"], data["act"]
        env_idx_arr, time_idx_arr = data["env_idx"], data["time_idx"]

        def _train_step(carry, step_idx):
            train_state, rng = carry
            rng, sample_rng, loss_rng = jax.random.split(rng, 3)

            flat_idxs = jax.random.randint(sample_rng, (batch_size,), 0, num_valid)
            sel_env, sel_time = env_idx_arr[flat_idxs], time_idx_arr[flat_idxs]

            obs_batch = obs_data[sel_env, sel_time]
            act_batch = jax.vmap(
                lambda e, t: jax.lax.dynamic_slice(act_data[e], (t,), (plan_horizon,))
            )(sel_env, sel_time)

            train_state, info = grad_step(train_state, act_batch, obs_batch, loss_rng)
            return (train_state, rng), info

        (train_state, rng), infos = jax.lax.scan(
            _train_step, (train_state, rng), jnp.arange(train_steps_per_chunk)
        )
        mean_infos = jax.tree.map(jnp.mean, infos)
        return train_state, rng, mean_infos

    def train(rng: jax.Array) -> dict[str, Any]:
        rng, init_rng = jax.random.split(rng)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        t0 = time.time()
        log_every = max(num_chunks // 20, 1)
        ckpt_dir = _resolve_ckpt_dir(config)

        with _make_periodic_ckpt_manager(config) as ckpt_mgr:
            for chunk_idx in range(num_chunks):
                rng, train_rng = jax.random.split(rng)
                train_state, _, chunk_metrics = _train_chunk(
                    train_state, train_rng, data_arrays
                )

                step = (chunk_idx + 1) * train_steps_per_chunk

                is_final = (chunk_idx == num_chunks - 1)
                saved = ckpt_mgr.save(
                    step,
                    args=ocp.args.StandardSave(train_state),
                    force=is_final,
                )
                if saved:
                    print(f"  Checkpoint saved at step {step} -> '{ckpt_dir}'")

                if config.get("USE_WANDB", False):
                    elapsed = time.time() - t0
                    log_data = {
                        "offline/step": step,
                        "offline/steps_per_second": step / max(elapsed, 1e-6),
                    }
                    metrics_host = jax.device_get(chunk_metrics)
                    for key, val in metrics_host.items():
                        log_data[f"offline/{key}"] = float(val)
                    wandb.log(log_data, step=step)

                if (chunk_idx + 1) % log_every == 0 or is_final:
                    elapsed = time.time() - t0
                    total_steps = num_chunks * train_steps_per_chunk
                    metrics_str = "  ".join(
                        f"{k}={float(v):.4f}" for k, v in chunk_metrics.items()
                    )
                    print(f"  [{step:>6}/{total_steps}]  {metrics_str}  elapsed={elapsed:.0f}s")

        return {"train_state": train_state, "final_loss": chunk_metrics["loss"]}

    return train

def _make_collect_chunk_fn(ppo_agent, env_w, env_params, collect_steps: int, config):
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

            temperature = config.get("COLLECT_TEMPERATURE", 2.0)
            noisy_logits = pi.logits / temperature
            action = jax.random.categorical(k1, noisy_logits)

            obs_next, env_state, _, done_next, _ = env_w.step(k2, env_state, action, env_params)
            return (rng, env_state, obs_next, done_next, new_hidden), (obs, action, done)

        (rng, env_state, obs, done, hidden), (all_obs, all_acts, all_dones) = jax.lax.scan(
            _step, (rng, env_state, obs, done, hidden), None, collect_steps
        )

        return rng, env_state, obs, done, hidden, all_obs, all_acts, all_dones

    return _collect_chunk

def make_train_offline_from_agent(
    config: dict[str, Any],
    ppo_checkpoint_path: str,
) -> Callable[[jax.Array], dict[str, Any]]:
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
    train_steps_per_chunk = min(num_train_steps, 100)
    num_chunks = (num_train_steps + train_steps_per_chunk - 1) // train_steps_per_chunk

    ppo_agent = _load_ppo_checkpoint(
        ppo_checkpoint_path, num_actions, obs_dim,
        config.get("LAYER_SIZE", 512),
        model_type=config.get("PPO_MODEL_TYPE"),
    )
    env_w, _ = _make_env_stack(config, num_envs)
    model = _build_model(config, num_actions)
    _, apply_train = _make_apply_fns(model)
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    _collect_chunk = _make_collect_chunk_fn(ppo_agent, env_w, env_params, collect_steps, config)
    max_valid_indices = num_envs * collect_steps

    @jax.jit
    def _train_chunk(train_state, rng, data, num_valid):
        obs_data, act_data = data["obs"], data["act"]
        env_idx_arr, time_idx_arr = data["env_idx"], data["time_idx"]

        def _train_step(carry, step_idx):
            train_state, rng = carry
            rng, sample_rng, loss_rng = jax.random.split(rng, 3)

            flat_idxs = jax.random.randint(sample_rng, (batch_size,), 0, num_valid)
            sel_env, sel_time = env_idx_arr[flat_idxs], time_idx_arr[flat_idxs]

            obs_batch = obs_data[sel_env, sel_time]
            act_batch = jax.vmap(
                lambda e, t: jax.lax.dynamic_slice(act_data[e], (t,), (plan_horizon,))
            )(sel_env, sel_time)

            train_state, info = grad_step(train_state, act_batch, obs_batch, loss_rng)
            return (train_state, rng), info

        (train_state, rng), infos = jax.lax.scan(
            _train_step, (train_state, rng), jnp.arange(train_steps_per_chunk)
        )
        mean_infos = jax.tree.map(jnp.mean, infos)
        return train_state, rng, mean_infos

    def train(rng: jax.Array) -> dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        obs, env_state = env_w.reset(env_rng, env_params)
        done = jnp.zeros(num_envs, dtype=bool)
        hidden = ppo_agent.init_hidden(num_envs)

        t0 = time.time()
        log_every = max(num_chunks // 20, 1)

        ckpt_dir = _resolve_ckpt_dir(config)

        padded_env = np.zeros(max_valid_indices, dtype=np.int32)
        padded_time = np.zeros(max_valid_indices, dtype=np.int32)

        with _make_periodic_ckpt_manager(config) as ckpt_mgr:

            for chunk_idx in range(num_chunks):
                rng, collect_rng = jax.random.split(rng)
                _, env_state, obs, done, hidden, chunk_obs, chunk_acts, chunk_dones = (
                    _collect_chunk(collect_rng, env_state, obs, done, hidden)
                )

                chunk_obs_np = np.asarray(chunk_obs).transpose(1, 0, 2)
                chunk_acts_np = np.asarray(chunk_acts).T
                chunk_dones_np = np.asarray(chunk_dones).T

                valid = _valid_window_mask(chunk_dones_np, plan_horizon)
                env_idxs, time_idxs = np.where(valid)
                if len(env_idxs) == 0:
                    continue

                num_valid = len(env_idxs)
                padded_env[:num_valid] = env_idxs
                padded_time[:num_valid] = time_idxs
                padded_env[num_valid:] = 0
                padded_time[num_valid:] = 0

                data = {
                    "obs": jnp.array(chunk_obs_np, dtype=jnp.float32),
                    "act": jnp.array(chunk_acts_np, dtype=jnp.int32),
                    "env_idx": jnp.array(padded_env),
                    "time_idx": jnp.array(padded_time),
                }

                rng, train_rng = jax.random.split(rng)

                train_state, _, chunk_metrics = _train_chunk(
                    train_state, train_rng, data, jnp.int32(num_valid)
                )

                step = (chunk_idx + 1) * train_steps_per_chunk

                is_final = (chunk_idx == num_chunks - 1)
                saved = ckpt_mgr.save(
                    step,
                    args=ocp.args.StandardSave(train_state),
                    force=is_final,
                )
                if saved:
                    print(f"  Checkpoint saved at step {step} -> '{ckpt_dir}'")
                # -------------------------

                if config["USE_WANDB"]:
                    elapsed = time.time() - t0
                    log_data = {
                        "offline/step": step,
                        "offline/steps_per_second": step / max(elapsed, 1e-6),
                    }
                    metrics_host = jax.device_get(chunk_metrics)
                    for key, val in metrics_host.items():
                        log_data[f"offline/{key}"] = float(val)
                    wandb.log(log_data, step=step)

                if (chunk_idx + 1) % log_every == 0 or chunk_idx == num_chunks - 1:
                    elapsed = time.time() - t0
                    total_steps = num_chunks * train_steps_per_chunk
                    metrics_str = "  ".join(
                        f"{k}={float(v):.4f}" for k, v in chunk_metrics.items()
                    )
                    print(f"  [{step:>6}/{total_steps}]  {metrics_str}  elapsed={elapsed:.0f}s")

        return {"train_state": train_state}

    return train

def run_offline(config: dict[str, Any]) -> None:
    if config["USE_WANDB"]:
        total_steps = config["NUM_TRAIN_STEPS"] * config["BATCH_SIZE"]
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"] + "-remdm-offline-" + str(int(total_steps // 1e6)) + "M",
        )

    if config.get("PPO_CHECKPOINT_PATH"):
        print(f"Offline training: live collection from PPO ({config['PPO_CHECKPOINT_PATH']})")
        train_fn = make_train_offline_from_agent(config, config["PPO_CHECKPOINT_PATH"])
        rng = jax.random.PRNGKey(config["SEED"])
        t0 = time.time()
        outs = [train_fn(jax.random.fold_in(rng, i)) for i in range(config["NUM_REPEATS"])]
        elapsed = time.time() - t0
        print(f"Offline training time: {elapsed:.1f}s")
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
        num_repeats = config["NUM_REPEATS"]
        t0 = time.time()
        if num_repeats > 1:
            rngs = jnp.stack([jax.random.fold_in(rng, i) for i in range(num_repeats)])
            first_out = jax.jit(jax.vmap(train_fn, in_axes=(0, None)))(rngs, data_arrays)
            first_out = jax.tree.map(lambda x: x[0], first_out)
        else:
            first_out = jax.jit(train_fn)(rng, data_arrays)
        elapsed = time.time() - t0
        print(f"Offline training time: {elapsed:.1f}s")

        if config.get("USE_WANDB"):
            wandb.log({"offline/final_loss": float(first_out["final_loss"])})