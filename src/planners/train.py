"""Training loop: environment rollout -> diffusion window extraction -> gradient updates."""

from __future__ import annotations

import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from src.diffusion.loss import compute_loss
from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import SCHEDULE_MAP
from src.models.denoiser import DenoisingTransformer
from .data import (
    PPOAgent,
    Transition,
    build_ppo_network,
    load_ppo_params,
)
from .state import init_params, create_train_state, make_apply_fns
from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.logz.batch_logging import create_log_dict, batch_log


# ---------------------------------------------------------------------------
# Gradient step factory
# ---------------------------------------------------------------------------

def _action_stats(acts: jnp.ndarray, num_actions: int, valid: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Action distribution entropy and coverage over valid trajectories."""
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
    """Return a jittable function: (state, acts, obs, valid, rng) -> (state, metrics)."""

    def _loss_fn(params, acts, obs, valid, rng):
        return compute_loss(
            apply_train, params, rng, acts, obs, valid,
            num_actions, schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
        )

    def step(state, acts, obs, valid, rng):
        (loss, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            state.params, acts, obs, valid, rng,
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
    num_steps = config["NUM_STEPS"]
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    valid_per_rollout = num_steps - plan_horizon + 1
    num_samples = num_envs * valid_per_rollout

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

    # Schedule
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    def train(rng: jax.Array) -> dict[str, Any]:
        # Diffusion model
        net = DenoisingTransformer(
            num_actions=num_actions, plan_horizon=plan_horizon,
            d_model=config["D_MODEL"], n_heads=config["N_HEADS"],
            n_layers=config["N_LAYERS"], d_ff=config["D_FF"],
            obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
            obs_encoder_width=config["OBS_ENCODER_WIDTH"],
            dropout_rate=config["DROPOUT_RATE"],
        )
        apply_eval, apply_train = make_apply_fns(net)
        grad_step = _make_grad_step(
            apply_train, num_actions, schedule_fn, schedule_deriv_fn,
            config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
        )

        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = init_params(net, init_rng, obs_dim, plan_horizon)
        state = create_train_state(net, params, config["LR"], config["MAX_GRAD_NORM"])

        obsv, env_state = env.reset(env_rng, env_params)
        init_hstate = ppo.init_hidden(num_envs)

        # ------------------------------------------------------------------
        # Validation
        # ------------------------------------------------------------------
        def _validate(state, rng):
            rng, val_rng = jax.random.split(rng)
            val_obs, val_env_state = env.reset(val_rng, env_params)

            def _val_step(carry, _):
                vs, vo, rng = carry
                rng, p_rng, s_rng = jax.random.split(rng, 3)
                plan = sample_plan(
                    apply_eval, state.params, p_rng, vo,
                    num_actions, plan_horizon,
                    num_steps=config.get("VAL_DIFFUSION_STEPS", 50),
                    schedule_fn=schedule_fn, remask_strategy="cap", use_loop=True,
                )
                vo, vs, _, _, info = env.step(s_rng, vs, plan[:, 0], env_params)
                return (vs, vo, rng), info

            _, infos = jax.lax.scan(_val_step, (val_env_state, val_obs, rng), None, 128)
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

            # Collect trajectories
            def _env_step(carry, _):
                st, es, obs, done, hs, rng = carry
                rng, act_rng, step_rng = jax.random.split(rng, 3)
                action, new_hs = ppo.act(
                    obs, done, hs, act_rng, temperature=config.get("COLLECT_TEMPERATURE", 2.0),
                )
                new_obs, es, reward, new_done, info = env.step(step_rng, es, action, env_params)
                t = Transition(done=done, action=action, reward=reward, obs=obs, info=info)
                return (st, es, new_obs, new_done, new_hs, rng), t

            (state, env_state, last_obs, last_done, hstate, rng), traj = jax.lax.scan(
                _env_step, (state, env_state, last_obs, last_done, hstate, rng), None, num_steps,
            )

            # Extract diffusion windows
            def _window(t_idx):
                obs_t = traj.obs[t_idx]
                acts = jax.lax.dynamic_slice(traj.action, (t_idx, 0), (plan_horizon, num_envs))
                dones = jax.lax.dynamic_slice(traj.done, (t_idx, 0), (plan_horizon, num_envs))
                valid = ~jnp.any(dones, axis=0)
                return obs_t, jnp.swapaxes(acts, 0, 1), valid

            obs_w, act_w, valid_w = jax.vmap(_window)(jnp.arange(valid_per_rollout))
            flat_obs = obs_w.reshape(-1, obs_dim)
            flat_acts = act_w.reshape(-1, plan_horizon)
            flat_valid = valid_w.reshape(-1)
            dataset = (flat_obs, flat_acts, flat_valid)

            # Minibatch training
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
                    obs_b, act_b, val_b = batch
                    st, metrics = grad_step(st, act_b, obs_b, val_b, loss_rng)
                    return (st, rng), metrics

                (state, rng), metrics = jax.lax.scan(_mb, (state, rng), batches)
                return (state, ds, rng), metrics

            (state, _, rng), loss_info = jax.lax.scan(
                _epoch, (state, dataset, rng), None, config["UPDATE_EPOCHS"],
            )

            # Metrics
            metric = jax.tree.map(jnp.mean, loss_info)
            returned = traj.info["returned_episode"]
            env_metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8), traj.info,
            )
            metric.update(env_metrics)

            # Periodic validation
            val_interval = config.get("VAL_INTERVAL", 50)
            rng, val_rng = jax.random.split(rng)
            dummy = jax.tree.map(jnp.zeros_like, {f"val/{k}": v for k, v in env_metrics.items()})
            val_metrics = jax.lax.cond(
                step_idx % val_interval == 0,
                lambda: _validate(state, val_rng),
                lambda: dummy,
            )
            metric.update(val_metrics)

            if config["DEBUG"] and config["USE_WANDB"]:
                def _log(m, s):
                    batch_log(s, create_log_dict(m, config), config)
                jax.debug.callback(_log, metric, step_idx)

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
# Entry point
# ---------------------------------------------------------------------------

def run_offline_diffusion(config):
    config = {k.upper(): v for k, v in config.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"], entity=config["WANDB_ENTITY"],
            config=config,
            name=f"{config['ENV_NAME']}-OfflineDiffusion-{int(config['TOTAL_TIMESTEPS'] // 1e6)}M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    # FIX: jit wraps vmap, not the reverse
    train_fn = jax.jit(jax.vmap(make_train(config)))

    t0 = time.time()
    out = train_fn(rngs)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s  SPS: {config['TOTAL_TIMESTEPS'] / elapsed:.0f}")

    if config["USE_WANDB"] and config["SAVE_POLICY"]:
        train_states = out["runner_state"][0]
        train_state = jax.tree.map(lambda x: x[0], train_states)
        path = os.path.join(wandb.run.dir, "policies")
        with ocp.CheckpointManager(path, options=ocp.CheckpointManagerOptions(max_to_keep=1)) as mgr:
            mgr.save(config["TOTAL_TIMESTEPS"], args=ocp.args.StandardSave(train_state))
        print(f"Saved policy to {path}")
