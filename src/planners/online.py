"""Online GRPO training: sample plan groups, simulate, advantage-weight, update."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax.constants import Achievement as FullCraftaxAchievements
from craftax.craftax_classic.constants import Achievement as ClassicAchievements
from flax import serialization
from flax.training.train_state import TrainState

from src.diffusion import sample_plan, SCHEDULE_MAP
from src.models.reward_models import get_reward_model
from .data import PPOAgent, load_ppo_agent, make_env
from .state import build_model, init_params, load_checkpoint, create_train_state, make_apply_fns
from .train import make_grad_step


# ---------------------------------------------------------------------------
# make_train_online
# ---------------------------------------------------------------------------

def make_train_online(
    config: dict[str, Any],
    pretrained_params: Optional[Any] = None,
    ppo_agent: Optional[PPOAgent] = None,
):
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    replan_every = config["REPLAN_EVERY"]
    num_updates = config["NUM_UPDATES"]
    update_epochs = config["UPDATE_EPOCHS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    diffusion_steps = config["DIFFUSION_STEPS"]
    group_size = config.get("GRPO_GROUP_SIZE", 4)
    num_actions = config["NUM_ACTIONS"]
    obs_dim = config["OBS_DIM"]
    intrinsic_coef = config.get("INTRINSIC_COEF", 0.05)

    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    env, env_params = make_env(config, num_envs)
    model = build_model(config, num_actions)
    apply_eval, apply_train = make_apply_fns(model)
    grad_step = make_grad_step(
        apply_train, num_actions, schedule_fn, schedule_deriv_fn,
        config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)

        # Diffusion model
        params = pretrained_params if pretrained_params is not None else init_params(model, init_rng, obs_dim, plan_horizon)
        train_state = create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        obs, env_state = env.reset(env_rng, env_params)

        # Reward model (for intrinsic reward)
        reward_model = get_reward_model(config.get("REWARD_MODEL_TYPE", "mlp"))
        rm_params = reward_model.init(init_rng, jnp.zeros((1, obs_dim)))
        reward_load_path = config.get("REWARD_LOAD_PATH")
        if reward_load_path and os.path.exists(reward_load_path):
            with open(reward_load_path, "rb") as f:
                rm_params = serialization.from_bytes(rm_params, f.read())
            print(f"Loaded reward model from {reward_load_path}")
        rm_state = TrainState.create(
            apply_fn=reward_model.apply, params=rm_params, tx=optax.adam(1e-4),
        )

        # ---------------------------------------------------------------
        # Update step
        # ---------------------------------------------------------------
        def _update_step(runner, step_idx):
            train_state, env_state, obs, rng, rm_state = runner

            ppo_prob = config.get("PPO_INIT_PROB", 0.1) * jnp.power(
                config.get("PPO_DECAY_RATE", 0.99), step_idx,
            )
            init_ppo_hstate = ppo_agent.init_hidden(num_envs) if ppo_agent is not None else jnp.zeros(1)

            # --- Plan, simulate, score ---
            def _plan_and_execute(carry, _):
                es, cur_obs, rng, ppo_hs = carry
                rng, plan_rng, sim_rng = jax.random.split(rng, 3)

                # Sample group_size plans at varying temperatures
                temps = jnp.linspace(0.5, 1.5, group_size)

                def _sample(r, temp):
                    return sample_plan(
                        apply_eval, train_state.params, r, cur_obs,
                        num_actions, plan_horizon, diffusion_steps, schedule_fn,
                        config.get("REMASK_STRATEGY", "cap"), config.get("ETA", 0.5),
                        config.get("USE_LOOP", False), config.get("T_ON", 0.7),
                        config.get("T_OFF", 0.3), temp, config.get("TOP_P", 0.95),
                    )

                plans = jax.vmap(_sample)(jax.random.split(plan_rng, group_size), temps)

                # Simulate each plan
                is_teacher = jnp.arange(group_size) == 0

                def _sim_plan(plan, is_t):
                    rng_flip, r_sim = jax.random.split(sim_rng)
                    use_ppo = jnp.logical_and(
                        jax.random.bernoulli(rng_flip, ppo_prob, shape=(num_envs,)), is_t,
                    )

                    def _sim_step(c, step_i):
                        st, o, r, hs = c
                        r, s_rng, ppo_rng = jax.random.split(r, 3)
                        diff_act = plan[:, step_i]

                        if ppo_agent is not None:
                            pi, new_hs = ppo_agent.get_pi(o, jnp.zeros(num_envs, dtype=bool), hs)
                            ppo_act = jax.random.categorical(ppo_rng, pi.logits).squeeze(0)
                            final_act = jnp.where(use_ppo, ppo_act, diff_act)
                        else:
                            final_act = diff_act
                            new_hs = hs

                        o_next, st, rew, done, info = env.step(s_rng, st, final_act, env_params)
                        return (st, o_next, r, new_hs), (o_next, rew, final_act, info)

                    final_c, (obs_traj, rew_traj, act_traj, infos) = jax.lax.scan(
                        _sim_step, (es, cur_obs, r_sim, ppo_hs), jnp.arange(replan_every),
                    )
                    return final_c, obs_traj, rew_traj, act_traj, infos

                carries, all_obs, all_rew, all_act, all_infos = jax.vmap(_sim_plan)(plans, is_teacher)

                # actions: [group, steps, envs] -> [group, envs, steps]
                all_act = jnp.transpose(all_act, (0, 2, 1))

                # Real universe = index 0
                es_next = jax.tree.map(lambda x: x[0], carries[0])
                obs_next = carries[1][0]
                rng_next = carries[2][0]
                ppo_hs_next = carries[3][0]
                real_infos = jax.tree.map(lambda x: x[0], all_infos)

                # Intrinsic rewards
                if intrinsic_coef > 0.0:
                    flat_obs = all_obs.reshape(-1, obs_dim)
                    intr = reward_model.apply(rm_state.params, flat_obs).reshape(group_size, replan_every, num_envs)
                    group_intr = jnp.sum(intr, axis=1)
                else:
                    group_intr = jnp.zeros((group_size, num_envs))

                group_base = jnp.sum(all_rew, axis=1)
                group_train_r = group_base + intrinsic_coef * group_intr

                return (es_next, obs_next, rng_next, ppo_hs_next), (
                    cur_obs, all_act, group_train_r, group_base, group_intr, real_infos,
                )

            n_cycles = config["NUM_STEPS"] // replan_every
            (env_state, obs, rng, _), traj = jax.lax.scan(
                _plan_and_execute, (env_state, obs, rng, init_ppo_hstate), None, n_cycles,
            )
            traj_obs, traj_plans, traj_train_r, traj_base_r, traj_intr_r, all_infos = traj

            # GRPO advantages (z-score across group)
            mean_r = jnp.mean(traj_train_r, axis=1, keepdims=True)
            std_r = jnp.std(traj_train_r, axis=1, keepdims=True) + 1e-8
            advantages = (traj_train_r - mean_r) / std_r
            awr_temp = config.get("AWR_TEMPERATURE", 2.0)
            adv_weights = jnp.clip(jnp.exp(advantages / awr_temp), 0.0, 20.0)

            # Flatten for training: tile obs across group dimension
            tiled_obs = jnp.tile(traj_obs[:, jnp.newaxis, :, :], (1, group_size, 1, 1))
            flat_obs = tiled_obs.reshape(-1, obs_dim)
            flat_plans = traj_plans.reshape(-1, plan_horizon)
            flat_adv = adv_weights.reshape(-1)
            flat_valid = jnp.ones(flat_obs.shape[0])
            total_samples = flat_obs.shape[0]

            # Minibatch SGD
            def _epoch(carry, _):
                ts, r = carry
                r, p_rng = jax.random.split(r)
                perm = jax.random.permutation(p_rng, total_samples)

                obs_mb = flat_obs[perm].reshape(num_minibatches, -1, obs_dim)
                plan_mb = flat_plans[perm].reshape(num_minibatches, -1, plan_horizon)
                adv_mb = flat_adv[perm].reshape(num_minibatches, -1)
                val_mb = flat_valid[perm].reshape(num_minibatches, -1)

                def _mb(ts_r, data):
                    ts, r = ts_r
                    idx, o, p, v, a = data
                    r_loss = jax.random.fold_in(r, idx)
                    ts, info = grad_step(ts, p, o, v, r_loss, advantages=a)
                    return (ts, r), info

                return jax.lax.scan(
                    _mb, (ts, r),
                    (jnp.arange(num_minibatches), obs_mb, plan_mb, val_mb, adv_mb),
                )

            (train_state, rng), epoch_infos = jax.lax.scan(
                _epoch, (train_state, rng), None, update_epochs,
            )

            # Co-train reward model
            if intrinsic_coef > 0.0:
                def _train_rm(s):
                    def _rm_loss(params):
                        preds = reward_model.apply(params, flat_obs)
                        if config.get("REWARD_MODEL_TYPE", "mlp") == "mlp":
                            return jnp.mean((preds - (-1.0)) ** 2)
                        return jnp.mean(preds)

                    loss, grads = jax.value_and_grad(_rm_loss)(s.params)
                    return s.apply_gradients(grads=grads)

                rm_state = jax.lax.cond(step_idx % 100 == 0, _train_rm, lambda s: s, rm_state)

            # Metrics
            ep_mask = all_infos["returned_episode"]
            n_done = jnp.maximum(ep_mask.sum(), 1)

            real_segment = traj_base_r + traj_intr_r
            real_mean = jnp.mean(real_segment, axis=1, keepdims=True)
            real_std = jnp.std(real_segment, axis=1, keepdims=True) + 1e-8
            raw_adv = (real_segment - real_mean) / real_std

            metrics = {
                "loss": epoch_infos["loss"].mean(),
                "ppo_prob": ppo_prob,
                "advantage_mean": raw_adv.mean(),
                "advantage_std": raw_adv.std(),
                "reward_mean": real_segment.mean(),
                "env_reward_mean": traj_base_r.mean(),
                "intrinsic_reward_mean": traj_intr_r.mean(),
                "grad_norm": epoch_infos["grad_norm"].mean(),
                "death_toll": ep_mask.sum(),
            }

            # Achievement tracking
            is_classic = "Classic" in config["ENV_NAME"]
            ach_cls = ClassicAchievements if is_classic else FullCraftaxAchievements
            for k, v in all_infos.items():
                kl = k.lower()
                if "achievement" in kl and "returned_episode" in kl:
                    name = k.split("_")[-1].title()
                    for e in ach_cls:
                        if e.name.lower() in kl:
                            name = e.name.replace("_", " ").title()
                            break
                    metrics[f"Achievements/{name}"] = (v * ep_mask).sum() / n_done

            jax.debug.print(
                "Update: {step} | Loss: {loss:.3f} | Score: {score:.2f} | Intr: {intr:.2f}",
                step=step_idx, loss=metrics["loss"],
                score=metrics["env_reward_mean"], intr=metrics["intrinsic_reward_mean"],
            )

            if config.get("USE_WANDB") and config.get("DEBUG", True):
                frames_per_update = num_envs * config["NUM_STEPS"]

                def _wandb_cb(mets, step):
                    try:
                        log = {"global_step": int(step) * frames_per_update}
                        for k, v in mets.items():
                            val = np.array(v).item()
                            if k.startswith("Achievements/"):
                                log[k] = val * 100.0
                            else:
                                log[f"online/{k}"] = val
                        wandb.log(log)
                    except Exception as e:
                        print(f"[WANDB ERROR at step {int(step)}]: {e}")

                jax.debug.callback(_wandb_cb, metrics, step_idx)

            return (train_state, env_state, obs, rng, rm_state), metrics

        # Host loop with periodic checkpointing
        runner = (train_state, env_state, obs, rng, rm_state)
        _jit_step = jax.jit(_update_step)
        all_metrics = []
        t0 = time.time()
        log_every = max(num_updates // 20, 1)

        ckpt_dir = config.get("CHECKPOINT_DIR", "checkpoints_online")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_opts = ocp.CheckpointManagerOptions(
            max_to_keep=config.get("MAX_CHECKPOINTS", 3),
            save_interval_steps=config.get("CHECKPOINT_INTERVAL", 500),
        )
        with ocp.CheckpointManager(ckpt_dir, options=ckpt_opts) as ckpt_mgr:
            for si in range(num_updates):
                runner, metrics = _jit_step(runner, jnp.int32(si))
                all_metrics.append(metrics)

                is_final = si == num_updates - 1
                saved = ckpt_mgr.save(si + 1, args=ocp.args.StandardSave(runner[0]), force=is_final)
                if saved:
                    print(f"  Checkpoint saved at step {si + 1}")

                if (si + 1) % log_every == 0 or is_final:
                    elapsed = time.time() - t0
                    loss_val = float(jax.device_get(metrics["loss"]))
                    print(f"  [{si + 1:>6}/{num_updates}]  loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

        stacked = jax.tree.map(lambda *xs: jnp.stack(xs), *all_metrics)
        return {"runner_state": runner, "metrics": stacked}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_online(config: dict[str, Any]) -> None:
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = int(env.action_space(env.default_params).n)
    config["OBS_DIM"] = int(env.observation_space(env.default_params).shape[0])

    ppo = None
    if config.get("PPO_CHECKPOINT_PATH"):
        print(f"Loading PPO teacher from {config['PPO_CHECKPOINT_PATH']}...")
        ppo = load_ppo_agent(
            config["PPO_CHECKPOINT_PATH"], config["NUM_ACTIONS"], config["OBS_DIM"],
            config.get("LAYER_SIZE", 512), config.get("PPO_MODEL_TYPE", "ppo_rnn"),
            config, num_envs=config["NUM_ENVS"],
        )

    pretrained = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        net = build_model(config, config["NUM_ACTIONS"])
        rng = jax.random.PRNGKey(0)
        pretrained = load_checkpoint(net, rng, config["OBS_DIM"], config["PLAN_HORIZON"], config["OFFLINE_CHECKPOINT_PATH"])

    if config.get("USE_WANDB"):
        wandb.init(
            project=config.get("WANDB_PROJECT", "craftax-remdm"),
            config=config, name=f"GRPO-{config['ENV_NAME']}",
        )
        wandb.define_metric("global_step")
        wandb.define_metric("online/*", step_metric="global_step")
        wandb.define_metric("Achievements/*", step_metric="global_step")

    train_fn = make_train_online(config, pretrained_params=pretrained, ppo_agent=ppo)
    print("Starting Online GRPO Training...")
    train_fn(jax.random.PRNGKey(config["SEED"]))
    print("Training complete.")
