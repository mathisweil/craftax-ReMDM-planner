import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from src.models.remdm import sample_plan
from src.logz.batch_logging import create_log_dict, batch_log

from .common import SCHEDULE_MAP, _make_grad_step
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_checkpoint,
    _save_model,
    _make_env_stack,
    _make_apply_fns,
)

def make_train_online(
    config: dict[str, Any],
    init_params: Optional[Any] = None,
) -> Callable[[jax.Array], dict[str, Any]]:
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

    def train(rng: jax.Array) -> dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = init_params if init_params is not None else _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])
        obs, env_state = env_w.reset(env_rng, env_params)

        def _update_step(runner_state, update_step):
            train_state, env_state, obs, rng = runner_state

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

            flat_obs = traj_obs.reshape(total_samples, obs_dim)
            flat_plans = traj_plans.reshape(total_samples, plan_horizon)

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

            for k, v in all_infos.items():
                if "achievement" in k.lower():
                    metric[k] = jnp.where(
                        n_completed > 0,
                        (v * ep_mask).sum() / safe_n,
                        jnp.nan,
                    )

            if config.get("DEBUG") and config.get("USE_WANDB"):
                def _wandb_callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(_wandb_callback, metric, update_step)

            return (train_state, env_state, obs, rng), metric

        runner_state = (train_state, env_state, obs, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def run_online(config: dict[str, Any]) -> None:
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = env.action_space(env.default_params).n
    config["OBS_DIM"] = env.observation_space(env.default_params).shape[0]

    total_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]

    if config.get("USE_WANDB"):
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"] + "-remdm-online-" + str(int(total_steps // 1e6)) + "M",
        )

    init_params: Optional[Any] = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        model = _build_model(config, config["NUM_ACTIONS"])
        init_params = _load_checkpoint(config, model, config["OBS_DIM"], config["OFFLINE_CHECKPOINT_PATH"])

    rng = jax.random.PRNGKey(config["SEED"])
    train_fn = make_train_online(config, init_params=init_params)
    num_repeats = config["NUM_REPEATS"]

    t0 = time.time()
    if num_repeats > 1:
        rngs = jnp.stack([jax.random.fold_in(rng, i) for i in range(num_repeats)])
        first_out = jax.jit(jax.vmap(train_fn))(rngs)
        first_out = jax.tree.map(lambda x: x[0], first_out)
    else:
        first_out = jax.jit(train_fn)(rng)
    elapsed = time.time() - t0

    sps = total_steps / max(elapsed, 1e-6)
    print(f"Online training time: {elapsed:.1f}s | SPS: {sps:.0f}")

    if config.get("USE_WANDB") and not config.get("DEBUG"):
        metrics = first_out["metrics"]
        num_updates = config["NUM_UPDATES"]
        log_interval = max(num_updates // 100, 1)
        for i in range(0, num_updates, log_interval):
            payload = {f"online/{k}": float(v[i]) for k, v in metrics.items()}
            payload["online/step"] = int(i)
            wandb.log(payload, step=int(i))

    if config.get("USE_WANDB"):
        wandb.log({"online/total_sps": sps, "online/total_time_s": elapsed})

    if config["SAVE_POLICY"]:
        _save_model(first_out["runner_state"][0], config, "diffusion_online")