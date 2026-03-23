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

from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import SCHEDULE_MAP
from .common import make_grad_step
from .env import Transition, make_env
from .model import build_model, init_params, create_train_state, make_apply_fns
from .ppo import PPOAgent, build_ppo_network, load_ppo_params
from .logging import make_wandb_callback


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
    env, env_params = make_env(config, num_envs)

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
    grad_step = make_grad_step(
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
                    schedule_fn=schedule_fn,
                    remask_strategy=config.get("REMASK_STRATEGY", "rescale"),
                    eta=config.get("ETA", 0.5),
                    use_loop=config.get("USE_LOOP", True),
                    t_on=config.get("T_ON", 0.7),
                    t_off=config.get("T_OFF", 0.3),
                    temperature=config.get("TEMPERATURE", 0.5),
                    top_p=config.get("TOP_P", 0.95),
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
        return {"runner_state": runner_final, "metrics": metrics}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_offline_diffusion(config):
    """Configure, compile, and run offline diffusion training.

    Args:
        config: Mixed-case hyperparameter dict from ``defaults.yaml`` / CLI merge.
                Keys are upper-cased on entry.
    """
    config = {k.upper(): v for k, v in config.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"], entity=config["WANDB_ENTITY"],
            config=config,
            name=f"{config['ENV_NAME']}-OfflineDiffusion-{int(config['TOTAL_TIMESTEPS'] // 1e6)}M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

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
            mgr.save(int(config["TOTAL_TIMESTEPS"]), args=ocp.args.StandardSave(train_state))
        print(f"Saved policy to {path}")
