"""Online DAgger training: roll out learner, label with expert, aggregate, update."""

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
# make_train_dagger
# ---------------------------------------------------------------------------

def make_train_dagger(config: dict[str, Any]):
    """Build the DAgger train closure.

    Each update:
      1. Roll out the current diffusion policy (mixed with expert via β).
      2. At every visited state, query the expert for target actions.
      3. Train the diffusion model on (state, expert_action) pairs with BC loss.
      β decays over updates so the learner's own policy dominates rollouts.
    """
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    replan_every = config["REPLAN_EVERY"]
    num_updates = config["NUM_UPDATES"]
    update_epochs = config["UPDATE_EPOCHS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    diffusion_steps = config["DIFFUSION_STEPS"]

    # Environment ----------------------------------------------------------
    env, env_params = make_env(config, num_envs)
    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dim = obs_shape[0]

    # Expert (PPO) — required for DAgger -----------------------------------
    assert config.get("PPO_CHECKPOINT_PATH"), (
        "DAgger requires an expert policy; set PPO_CHECKPOINT_PATH."
    )
    ppo: PPOAgent = load_ppo_agent(
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

    # Pretrained checkpoint ------------------------------------------------
    pretrained_params = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        _tmp_rng = jax.random.PRNGKey(0)
        pretrained_params = load_checkpoint(
            model, _tmp_rng, obs_dim, plan_horizon,
            config["OFFLINE_CHECKPOINT_PATH"],
        )

    # Samples per update
    n_cycles = config["NUM_STEPS"] // replan_every
    total_samples = n_cycles * num_envs
    assert total_samples % num_minibatches == 0, (
        f"{total_samples} samples not divisible by {num_minibatches} minibatches"
    )

    # β schedule: probability of using expert for rollout actions
    beta_init = config.get("DAGGER_BETA_INIT", 1.0)
    beta_decay = config.get("DAGGER_BETA_DECAY", 0.95)

    # Wandb ----------------------------------------------------------------
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

        if pretrained_params is not None:
            params = pretrained_params
        else:
            params = init_params(model, init_rng, obs_dim, plan_horizon)
        state = create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])

        obs, env_state = env.reset(env_rng, env_params)
        init_ppo_hstate = ppo.init_hidden(num_envs)

        # --------------------------------------------------------------
        # _update_step
        # --------------------------------------------------------------
        def _update_step(runner, _):
            state, env_state, obs, rng, step_idx = runner

            # β decays each update: expert → learner
            beta = beta_init * jnp.power(beta_decay, step_idx)

            ppo_hs = init_ppo_hstate

            # --- Roll out with mixed policy, collect expert labels -----
            def _plan_and_execute(carry, _):
                es, cur_obs, rng, ppo_hs = carry
                rng, plan_rng, sim_rng = jax.random.split(rng, 3)

                # Sample a plan from the current diffusion policy
                learner_plan = sample_plan(
                    apply_eval, state.params, plan_rng, cur_obs,
                    num_actions, plan_horizon, diffusion_steps, schedule_fn,
                    config.get("REMASK_STRATEGY", "rescale"),
                    config.get("ETA", 0.5),
                    config.get("USE_LOOP", True),
                    config.get("T_ON", 0.7),
                    config.get("T_OFF", 0.3),
                    config.get("TEMPERATURE", 1.0),
                    config.get("TOP_P", 0.95),
                )  # [E, H]

                # Simulate replan_every steps, collecting expert labels
                def _sim_step(c, step_i):
                    st, o, r, hs = c
                    r, s_rng, mix_rng, ppo_rng = jax.random.split(r, 4)

                    # Expert action
                    pi, new_hs = ppo.get_pi(
                        o, jnp.zeros(num_envs, dtype=bool), hs,
                    )
                    expert_act = jax.random.categorical(ppo_rng, pi.logits).squeeze(0)

                    # Learner action from the plan
                    learner_act = learner_plan[:, step_i]

                    # Mixed execution: with prob β use expert, else learner
                    use_expert = jax.random.bernoulli(
                        mix_rng, beta, shape=(num_envs,),
                    )
                    exec_act = jnp.where(use_expert, expert_act, learner_act)

                    o_next, st, rew, done, info = env.step(
                        s_rng, st, exec_act, env_params,
                    )
                    return (st, o_next, r, new_hs), (o, expert_act, rew, info)

                final_c, (visited_obs, expert_acts, rews, infos) = jax.lax.scan(
                    _sim_step,
                    (es, cur_obs, sim_rng, ppo_hs),
                    jnp.arange(replan_every),
                )
                # visited_obs:  [steps, E, obs_dim]
                # expert_acts:  [steps, E]

                es_next, obs_next, rng_next, ppo_hs_next = final_c

                return (es_next, obs_next, rng_next, ppo_hs_next), (
                    visited_obs, expert_acts, rews, infos,
                )

            (env_state, obs, rng, _ppo_hs), traj = jax.lax.scan(
                _plan_and_execute,
                (env_state, obs, rng, ppo_hs),
                None,
                n_cycles,
            )
            traj_obs, traj_expert_acts, traj_rew, all_infos = traj
            # traj_obs:         [C, steps, E, obs_dim]
            # traj_expert_acts: [C, steps, E]
            # traj_rew:         [C, steps, E]

            # Build expert plan targets:
            # For each cycle, the expert actions over replan_every steps form
            # the target plan.  Pad to plan_horizon if needed.
            # traj_expert_acts: [C, steps, E] -> [C, E, steps]
            expert_plans = jnp.transpose(traj_expert_acts, (0, 2, 1))
            if replan_every < plan_horizon:
                pad_width = plan_horizon - replan_every
                # Pad with zeros (masked out during loss via valid mask)
                expert_plans = jnp.pad(
                    expert_plans,
                    ((0, 0), (0, 0), (0, pad_width)),
                    constant_values=0,
                )
            # expert_plans: [C, E, plan_horizon]

            # Obs at the start of each cycle (the conditioning observation)
            # traj_obs[:, 0, :, :] gives [C, E, obs_dim]
            cycle_obs = traj_obs[:, 0, :, :]

            # Valid mask: only first replan_every positions are real
            valid_per_step = jnp.concatenate([
                jnp.ones(replan_every),
                jnp.zeros(max(plan_horizon - replan_every, 0)),
            ])  # [plan_horizon]

            # Flatten across cycles and envs
            flat_obs = cycle_obs.reshape(-1, obs_dim)           # [C*E, D]
            flat_plans = expert_plans.reshape(-1, plan_horizon)  # [C*E, H]
            flat_valid = jnp.broadcast_to(
                valid_per_step, (flat_obs.shape[0], plan_horizon),
            )  # [C*E, H]

            # Minibatch SGD (standard BC, no advantage weighting)
            dataset = (flat_obs, flat_plans, flat_valid)

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
                    obs_b, act_b, val_b = batch
                    # Standard BC loss — no advantages
                    st, metrics = grad_step(st, act_b, obs_b, val_b, loss_rng)
                    return (st, rng), metrics

                (state, rng), metrics = jax.lax.scan(_mb, (state, rng), batches)
                return (state, ds, rng), metrics

            (state, _, rng), loss_info = jax.lax.scan(
                _epoch, (state, dataset, rng), None, update_epochs,
            )

            # Metrics
            metric = jax.tree.map(jnp.mean, loss_info)
            returned = all_infos["returned_episode"]
            env_metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8),
                all_infos,
            )
            metric.update(env_metrics)
            metric["beta"] = beta
            metric["reward_mean"] = jnp.mean(traj_rew)

            if _wandb_log is not None:
                jax.debug.callback(_wandb_log, metric, step_idx)

            runner = (state, env_state, obs, rng, step_idx + 1)
            return runner, metric

        # Outer scan
        rng, run_rng = jax.random.split(rng)
        runner_init = (state, env_state, obs, run_rng, 0)
        runner_final, metrics = jax.lax.scan(
            _update_step, runner_init, None, num_updates,
        )
        return {"runner_state": runner_final, "metrics": metrics}

    return train


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_online(config: dict[str, Any]) -> None:
    config = {k.upper(): v for k, v in config.items()}

    if config.get("USE_WANDB"):
        wandb.init(
            project=config.get("WANDB_PROJECT", "remdm-craftax"),
            config=config,
            name=f"DAgger-{config['ENV_NAME']}",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config.get("NUM_REPEATS", 1))

    train_fn = jax.jit(jax.vmap(make_train_dagger(config)))

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

        artifact = wandb.Artifact(
            name=f"{config['ENV_NAME']}-policy",
            type="model",
            metadata=config,
        )
        artifact.add_dir(path)
        wandb.log_artifact(artifact)

        print("Uploaded policy artifact to wandb")
