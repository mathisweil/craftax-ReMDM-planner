"""Online GRPO training: sample plan groups, simulate, advantage-weight, update."""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb

from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import SCHEDULE_MAP

from .common import make_grad_step
from .env import make_env
from .model import build_model, init_params, load_checkpoint, create_train_state, make_apply_fns
from .ppo import PPOAgent, load_ppo_agent

from .logging import make_wandb_callback


# ---------------------------------------------------------------------------
# make_train_online  (aligned with make_train in train.py)
# ---------------------------------------------------------------------------

def make_train_online(config: dict[str, Any]):
    """Build the online train closure.  Mirrors make_train in train.py:
    - All env / model setup happens here (outside the returned closure).
    - The returned `train(rng)` is jit-friendly and uses jax.lax.scan
      for the outer update loop.
    """
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    replan_every = config["REPLAN_EVERY"]
    num_updates = config["NUM_UPDATES"]
    update_epochs = config["UPDATE_EPOCHS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    diffusion_steps = config["DIFFUSION_STEPS"]
    group_size = config.get("GRPO_GROUP_SIZE", 4)

    # Environment ----------------------------------------------------------
    env, env_params = make_env(config, num_envs)
    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dim = obs_shape[0]

    # PPO collector (optional) ---------------------------------------------
    ppo: Optional[PPOAgent] = None
    if config.get("PPO_CHECKPOINT_PATH"):
        ppo = load_ppo_agent(
            config["PPO_CHECKPOINT_PATH"], num_actions, obs_dim,
            config.get("LAYER_SIZE", 512),
            config.get("PPO_MODEL_TYPE", "ppo_rnn"),
            config, num_envs=num_envs,
        )

    # Schedule -------------------------------------------------------------
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    # Diffusion model / apply fns ------------------------------------------
    model = build_model(config, num_actions)
    apply_eval, apply_train = make_apply_fns(model)
    grad_step = make_grad_step(
        apply_train, num_actions, schedule_fn, schedule_deriv_fn,
        config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
    )

    # Pretrained checkpoint (host I/O — must happen before jit/vmap tracing)
    pretrained_params = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        _tmp_rng = jax.random.PRNGKey(0)
        pretrained_params = load_checkpoint(
            model, _tmp_rng, obs_dim, plan_horizon,
            config["OFFLINE_CHECKPOINT_PATH"],
        )

    # Samples per update
    n_cycles = config["NUM_STEPS"] // replan_every
    total_samples = n_cycles * group_size * num_envs
    assert total_samples % num_minibatches == 0, (
        f"{total_samples} samples not divisible by {num_minibatches} minibatches"
    )

    # -----------------------------------------------------------------------

    # No periodic validation in online mode; set interval beyond num_updates.
    _val_interval = num_updates + 1
    _wandb_log = (
        make_wandb_callback(
            config,
            steps_per_update=num_envs * config["NUM_STEPS"],
            val_interval=_val_interval,
            is_online=True,
        )
        if config.get("USE_WANDB") else None
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)

        # Use pretrained params if available, otherwise init from scratch
        if pretrained_params is not None:
            params = pretrained_params
        else:
            params = init_params(model, init_rng, obs_dim, plan_horizon)
        state = create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        obs, env_state = env.reset(env_rng, env_params)
        # Must be a concrete array (not None) so scan/vmap carries have
        # consistent pytree structure.
        init_ppo_hstate = (
            ppo.init_hidden(num_envs) if ppo is not None
            else jnp.zeros((num_envs, 1))
        )

        # --------------------------------------------------------------
        # _update_step  (matches train.py signature: (runner, _) -> ...)
        # --------------------------------------------------------------
        def _update_step(runner, _):
            state, env_state, obs, rng, step_idx = runner

            ppo_prob = config.get("PPO_INIT_PROB", 0.1) * jnp.power(
                config.get("PPO_DECAY_RATE", 0.99), step_idx,
            )
            ppo_hs = init_ppo_hstate

            # --- Plan, simulate, score  --------------------------------
            def _plan_and_execute(carry, _):
                es, cur_obs, rng, ppo_hs = carry
                rng, plan_rng, sim_base_rng = jax.random.split(rng, 3)

                # Sample group_size plans at varying temperatures
                temps = jnp.linspace(0.5, 1.5, group_size)

                def _sample(r, temp):
                    return sample_plan(
                        apply_eval, state.params, r, cur_obs,
                        num_actions, plan_horizon, diffusion_steps, schedule_fn,
                        config.get("REMASK_STRATEGY", "rescale"),
                        config.get("ETA", 0.5),
                        config.get("USE_LOOP", True),
                        config.get("T_ON", 0.7),
                        config.get("T_OFF", 0.3),
                        temp, config.get("TOP_P", 0.95),
                    )

                plans = jax.vmap(_sample)(
                    jax.random.split(plan_rng, group_size), temps,
                )  # [G, E, H]

                sim_rngs = jax.random.split(sim_base_rng, group_size)
                is_teacher = jnp.arange(group_size) == 0

                def _sim_plan(plan, sim_rng, is_t):
                    rng_flip, r_sim = jax.random.split(sim_rng)
                    use_ppo = jnp.logical_and(
                        jax.random.bernoulli(rng_flip, ppo_prob, shape=(num_envs,)),
                        is_t,
                    )

                    def _sim_step(c, step_i):
                        st, o, r, hs = c
                        r, s_rng, ppo_rng = jax.random.split(r, 3)
                        diff_act = plan[:, step_i]

                        if ppo is not None:
                            pi, new_hs = ppo.get_pi(
                                o, jnp.zeros(num_envs, dtype=bool), hs,
                            )
                            ppo_act = jax.random.categorical(ppo_rng, pi.logits).squeeze(0)
                            final_act = jnp.where(use_ppo, ppo_act, diff_act)
                        else:
                            final_act = diff_act
                            new_hs = hs

                        o_next, st, rew, done, info = env.step(
                            s_rng, st, final_act, env_params,
                        )
                        return (st, o_next, r, new_hs), (rew, final_act, info)

                    final_c, (rew_traj, act_traj, infos) = jax.lax.scan(
                        _sim_step,
                        (es, cur_obs, r_sim, ppo_hs),
                        jnp.arange(replan_every),
                    )
                    return final_c, rew_traj, act_traj, infos

                carries, all_rew, all_act, all_infos = jax.vmap(_sim_plan)(
                    plans, sim_rngs, is_teacher,
                )

                # actions: [G, steps, E] -> [G, E, steps]
                all_act = jnp.transpose(all_act, (0, 2, 1))

                # Real universe = index 0
                es_next = jax.tree.map(lambda x: x[0], carries[0])
                obs_next = carries[1][0]
                rng_next = carries[2][0]
                ppo_hs_next = carries[3][0]
                real_infos = jax.tree.map(lambda x: x[0], all_infos)

                group_reward = jnp.sum(all_rew, axis=1)  # [G, E]

                return (es_next, obs_next, rng_next, ppo_hs_next), (
                    cur_obs, all_act, group_reward, real_infos,
                )

            (env_state, obs, rng, _ppo_hs), traj = jax.lax.scan(
                _plan_and_execute,
                (env_state, obs, rng, ppo_hs),
                None,
                n_cycles,
            )
            traj_obs, traj_plans, traj_reward, all_infos = traj
            # traj_obs:   [C, E, obs_dim]
            # traj_plans: [C, G, E, H]
            # traj_reward: [C, G, E]

            # GRPO advantages (z-score across group dim)
            mean_r = jnp.mean(traj_reward, axis=1, keepdims=True)
            std_r = jnp.std(traj_reward, axis=1, keepdims=True) + 1e-8
            advantages = (traj_reward - mean_r) / std_r  # [C, G, E]

            # Flatten: tile obs across group dim, then flatten
            # traj_obs is [C, E, D] -> [C, G, E, D]
            tiled_obs = jnp.broadcast_to(
                traj_obs[:, jnp.newaxis, :, :],
                (n_cycles, group_size, num_envs, obs_dim),
            )
            # Pad plans to plan_horizon if replan_every < plan_horizon
            # traj_plans is [C, G, E, replan_every] but we need [.., plan_horizon]
            # (only the first replan_every actions are used per cycle)

            flat_obs = tiled_obs.reshape(-1, obs_dim)
            flat_plans = traj_plans.reshape(-1, plan_horizon)
            flat_adv = advantages.reshape(-1)
            flat_valid = jnp.ones(flat_obs.shape[0])

            # Minibatch SGD  (matches train.py _epoch / _mb pattern)
            dataset = (flat_obs, flat_plans, flat_valid, flat_adv)

            def _epoch(epoch_state, _):
                state, ds, rng = epoch_state
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, total_samples)
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), ds)
                batches = jax.tree.map(
                    lambda x: x.reshape(num_minibatches, -1, *x.shape[1:]),
                    shuffled,
                )

                def _mb(carry, batch):
                    st, rng = carry
                    rng, loss_rng = jax.random.split(rng)
                    obs_b, act_b, val_b, adv_b = batch
                    st, metrics = grad_step(st, act_b, obs_b, val_b, loss_rng, advantages=adv_b)
                    return (st, rng), metrics

                (state, rng), metrics = jax.lax.scan(_mb, (state, rng), batches)
                return (state, ds, rng), metrics

            (state, _, rng), loss_info = jax.lax.scan(
                _epoch, (state, dataset, rng), None, update_epochs,
            )

            # Metrics (matches train.py pattern)
            metric = jax.tree.map(jnp.mean, loss_info)
            returned = all_infos["returned_episode"]
            env_metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8),
                all_infos,
            )
            metric.update(env_metrics)
            metric["ppo_prob"] = ppo_prob
            metric["advantage_mean"] = jnp.mean(advantages)
            metric["advantage_std"] = jnp.std(advantages)
            metric["reward_mean"] = jnp.mean(traj_reward)

            if _wandb_log is not None:
                jax.debug.callback(_wandb_log, metric, step_idx)

            runner = (state, env_state, obs, rng, step_idx + 1)
            return runner, metric

        # Outer scan (matches train.py: single jax.lax.scan, no host loop)
        rng, run_rng = jax.random.split(rng)
        runner_init = (state, env_state, obs, run_rng, 0)
        runner_final, metrics = jax.lax.scan(
            _update_step, runner_init, None, num_updates,
        )
        return {"runner_state": runner_final, "metrics": metrics}

    return train


# ---------------------------------------------------------------------------
# Entry point  (aligned with run_offline_diffusion in train.py)
# ---------------------------------------------------------------------------

def run_online(config: dict[str, Any]) -> None:
    config = {k.upper(): v for k, v in config.items()}

    if config.get("USE_WANDB"):
        wandb.init(
            project=config.get("WANDB_PROJECT", "remdm-craftax"),
            config=config,
            name=f"GRPO-{config['ENV_NAME']}",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config.get("NUM_REPEATS", 1))

    train_fn = jax.jit(jax.vmap(make_train_online(config)))

    t0 = time.time()
    out = train_fn(rngs)
    elapsed = time.time() - t0

    total_frames = config["NUM_UPDATES"] * config["NUM_ENVS"] * config["NUM_STEPS"]
    print(f"Time: {elapsed:.1f}s  SPS: {total_frames / elapsed:.0f}")

    if config.get("USE_WANDB") and config.get("SAVE_POLICY"):
        train_states = out["runner_state"][0]
        train_state = jax.tree.map(lambda x: x[0], train_states)
        path = os.path.join(wandb.run.dir, "policies")
        with ocp.CheckpointManager(
            path, options=ocp.CheckpointManagerOptions(max_to_keep=1),
        ) as mgr:
            mgr.save(
                int(config["NUM_UPDATES"]),
                args=ocp.args.StandardSave(train_state),
            )
        print(f"Saved policy to {path}")