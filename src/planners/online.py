"""Online DAgger training: roll out learner, label with expert, aggregate, update.

Implements Algorithm 3.1 from Ross et al. (2011) — 'A Reduction of Imitation
Learning and Structured Prediction to No-Regret Online Learning'.

Each DAgger iteration:
  1. Roll out the mixed policy (beta * expert + (1-beta) * learner).
  2. At every visited state, query the expert for target actions.
  3. Aggregate (obs, expert_plan) pairs into a growing replay buffer.
  4. Train the diffusion model on the full buffer with BC loss (MDLM ELBO).
  beta decays exponentially so the learner's own policy dominates rollouts.
"""

from __future__ import annotations

import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from flax.training.train_state import TrainState

from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import SCHEDULE_MAP

from .common import (
    compile_and_run,
    dagger_sizing,
    extract_sliding_windows,
    format_timing,
    make_grad_step,
    make_validate,
    print_config_snapshot,
    resolve_num_updates,
    resolve_scaled_hyperparams,
)
from .env import make_env
from .logging import init_wandb, make_wandb_callback
from .model import (
    build_model,
    create_train_state,
    init_params,
    load_checkpoint,
    load_checkpoint_for_resume,
    make_apply_fns,
    save_checkpoint_metadata,
)
from .ppo import PPOAgent, load_ppo_agent


class DAggerCarry(NamedTuple):
    """Carry state for the outer DAgger update scan."""

    train_state: Any
    env_state: Any
    obs: jnp.ndarray  # [E, obs_dim]
    rng: jax.Array
    step_idx: int
    ppo_hs: Any  # [E, layer_size] or None (non-RNN)
    prev_done: jnp.ndarray  # [E] bool
    buf_obs: jnp.ndarray  # [max_buf, obs_dim]
    buf_plans: jnp.ndarray  # [max_buf, plan_horizon] int32
    buf_valid: jnp.ndarray  # [max_buf] float32
    buf_write_idx: int
    buf_fill: int
    best_params: Any  # copy of params with highest val return
    best_val_return: jnp.ndarray  # scalar, -inf initially


def expert_action(
    pi_logits: jax.Array, deterministic: bool, rng: jax.Array
) -> jax.Array:
    """Expert label for the visited state (spec-training §1.5).

    Deterministic argmax keeps the expert mapping s -> a* fixed,
    removing label noise from the aggregated dataset; the sampled
    variant draws from the expert policy.
    """
    if deterministic:
        return jnp.argmax(pi_logits, axis=-1)
    return jax.random.categorical(rng, pi_logits)


def mixed_execution_mask(
    rng: jax.Array, beta: jax.Array, num_envs: int
) -> jax.Array:
    """Per-step Bernoulli(beta) expert/learner gate.

    DAgger Alg 3.1 instantiation (spec-training §1.2): each env
    executes the expert action with probability beta, else the
    learner's planned action.
    """
    return jax.random.bernoulli(rng, beta, shape=(num_envs,))


def make_train_online_dagger(config: dict[str, Any]):
    """Build the DAgger train closure.

    All environment construction, model instantiation, and static
    pre-computation happen here (outside the returned ``train`` closure) so
    they are not repeated across ``jax.vmap`` replicas or JIT retraces.

    Args:
        config: Upper-cased hyperparameter dict (see ``configs/defaults.yaml``).

    Returns:
        A ``train(rng) -> dict`` closure that is safe to JIT and vmap.
    """
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    num_updates = config["NUM_UPDATES"]
    update_epochs = config["UPDATE_EPOCHS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    diffusion_steps = config["DIFFUSION_STEPS"]
    num_steps = config["NUM_STEPS"]

    # Validation config
    val_interval = config.get("VAL_INTERVAL", 50)
    val_replan_every = config.get("VAL_REPLAN_EVERY", 4)
    val_steps = config.get("VAL_STEPS", 128)
    n_val_cycles = val_steps // val_replan_every

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
        config["PPO_CHECKPOINT_PATH"],
        num_actions,
        obs_dim,
        config.get("LAYER_SIZE", 512),
        config.get("PPO_MODEL_TYPE", "ppo_rnn"),
        config,
        num_envs=num_envs,
    )

    # Schedule -------------------------------------------------------------
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    # Diffusion model / apply fns ------------------------------------------
    model = build_model(config, num_actions)
    apply_eval, apply_train = make_apply_fns(model)
    grad_step = make_grad_step(
        apply_train,
        num_actions,
        schedule_fn,
        schedule_deriv_fn,
        config.get("TRAIN_SIGMA", 0.0),
        config.get("LABEL_SMOOTHING", 0.0),
    )

    # Pretrained checkpoint ------------------------------------------------
    pretrained_params = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        _tmp_rng = jax.random.PRNGKey(0)
        pretrained_params = load_checkpoint(
            model,
            _tmp_rng,
            obs_dim,
            plan_horizon,
            config["OFFLINE_CHECKPOINT_PATH"],
        )

    # roll out num_steps env transitions across n_cycles plans, then
    # extract sliding windows the same way offline.py does — yields
    # W = num_steps - plan_horizon + 1 windows per env per update instead
    # of only one window per cycle (~16x denser for the default config).
    assert num_steps % plan_horizon == 0, (
        f"NUM_STEPS ({num_steps}) must be divisible by PLAN_HORIZON ({plan_horizon})"
    )
    assert num_steps >= plan_horizon, (
        f"NUM_STEPS ({num_steps}) must be >= PLAN_HORIZON ({plan_horizon})"
    )
    # sliding windows, buffer capacity and pass count.  Derived in
    # common.dagger_sizing so print_config_snapshot reports exactly what
    # runs here; the two used to disagree on n_train_passes.
    sizing = dagger_sizing(config, num_updates)
    n_cycles = sizing["n_cycles"]
    valid_per_rollout = sizing["valid_per_rollout"]
    samples_per_update = sizing["samples_per_update"]
    max_buffer_size = sizing["max_buffer_size"]
    n_train_passes = sizing["n_train_passes"]

    assert samples_per_update % num_minibatches == 0, (
        f"samples_per_update ({samples_per_update}) not divisible by"
        f" num_minibatches ({num_minibatches})"
    )
    assert samples_per_update <= max_buffer_size, (
        f"samples_per_update ({samples_per_update}) exceeds"
        f" max_buffer_size ({max_buffer_size}); raise DAGGER_BUFFER_MAX"
        f" or shrink NUM_ENVS / NUM_STEPS"
    )

    # deterministic-by-default expert.  Sampling from ``pi.logits``
    # injects label noise — two queries to the same state can return
    # different actions — which breaks DAgger's assumption of a fixed
    # expert mapping s -> a*.  Argmax keeps labels consistent.
    expert_deterministic = config.get("DAGGER_EXPERT_DETERMINISTIC", True)

    # Beta schedule: probability of using expert for rollout actions.
    # beta_i = beta_init * beta_decay^i  ->  0 as i -> inf.
    beta_init = config.get("DAGGER_BETA_INIT", 1.0)
    beta_decay = config.get("DAGGER_BETA_DECAY", 0.95)

    # cosine LR schedule matching offline training.  Stretched to
    # cover all gradient steps across train passes (B1).
    total_grad_steps = num_updates * n_train_passes * update_epochs * num_minibatches
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
            alpha=0.1,
        )
    )

    # Resume checkpoint (loaded outside JIT, captured by train closure) ------
    resume_step = config.get("RESUME_STEP") or 0
    resume_state = None
    if config.get("RESUME_CHECKPOINT_PATH"):
        resume_state = load_checkpoint_for_resume(
            model,
            jax.random.PRNGKey(0),
            obs_dim,
            plan_horizon,
            config["RESUME_CHECKPOINT_PATH"],
            lr_schedule,
            config["MAX_GRAD_NORM"],
        )
        target_opt_step = resume_step * n_train_passes * update_epochs * num_minibatches
        resume_state = resume_state.replace(step=target_opt_step)

    scan_length = num_updates - resume_step

    # W&B ------------------------------------------------------------------
    _wandb_log = (
        make_wandb_callback(
            config,
            steps_per_update=num_envs * num_steps,
            val_interval=val_interval,
            is_online=True,
        )
        if config.get("USE_WANDB")
        else None
    )

    def train(rng: jax.Array) -> dict[str, Any]:
        """JIT/vmap-compatible DAgger training loop.

        Args:
            rng: JAX PRNG key (one per vmap replica).

        Returns:
            Dict with ``runner_state`` (final DAggerCarry) and ``metrics``.
        """
        rng, init_rng, env_rng = jax.random.split(rng, 3)

        if resume_state is not None:
            state = resume_state
            params = resume_state.params
        elif pretrained_params is not None:
            params = pretrained_params
            state = create_train_state(
                model,
                params,
                lr_schedule,
                config["MAX_GRAD_NORM"],
            )
        else:
            params = init_params(model, init_rng, obs_dim, plan_horizon)
            state = create_train_state(
                model,
                params,
                lr_schedule,
                config["MAX_GRAD_NORM"],
            )

        obs, env_state = env.reset(env_rng, env_params)

        # Pre-allocate DAgger replay buffer
        buf_obs = jnp.zeros((max_buffer_size, obs_dim))
        buf_plans = jnp.zeros(
            (max_buffer_size, plan_horizon),
            dtype=jnp.int32,
        )
        buf_valid = jnp.zeros(max_buffer_size)

        # Shared validation closure (see common.py)
        _validate = make_validate(
            env,
            env_params,
            apply_eval,
            num_actions,
            plan_horizon,
            schedule_fn,
            config,
            val_replan_every,
            n_val_cycles,
        )

        def _update_step(carry: DAggerCarry, _):
            (
                state,
                env_state,
                obs,
                rng,
                step_idx,
                ppo_hs,
                prev_done,
                buf_obs,
                buf_plans,
                buf_valid,
                buf_write_idx,
                buf_fill,
                best_params,
                best_val_return,
            ) = carry

            # Beta decays each update: expert -> learner
            beta = beta_init * jnp.power(beta_decay, step_idx)

            # --- Roll out with mixed policy, collect expert labels -
            def _plan_and_execute(outer_carry, _):
                es, cur_obs, rng, hs, p_done = outer_carry
                rng, plan_rng, sim_rng = jax.random.split(rng, 3)

                # Learner plan from the current diffusion policy
                learner_plan = sample_plan(
                    apply_eval,
                    state.params,
                    plan_rng,
                    cur_obs,
                    num_actions,
                    plan_horizon,
                    diffusion_steps,
                    schedule_fn,
                    config.get("REMASK_STRATEGY", "rescale"),
                    config.get("ETA", 0.5),
                    config.get("USE_LOOP", True),
                    config.get("T_ON", 0.7),
                    config.get("T_OFF", 0.3),
                    config.get("TEMPERATURE", 1.0),
                    config.get("TOP_P", 0.95),
                )  # [E, H]

                # simulate plan_horizon steps, recording the visited
                # obs at every state alongside the expert action.  The
                # outer code then extracts sliding windows from the full
                # per-step trace, mirroring offline.py.
                def _sim_step(c, step_i):
                    st, o, r, hs, p_done = c
                    r, s_rng, mix_rng, ppo_rng = jax.random.split(
                        r,
                        4,
                    )

                    # Expert action with the correct done flag so the
                    # PPO RNN hidden state resets on episode boundaries.
                    pi, new_hs = ppo.get_pi(o, p_done, hs)
                    expert_act = expert_action(
                        pi.logits, expert_deterministic, ppo_rng
                    ).squeeze(0)

                    # Learner action from the plan
                    learner_act = learner_plan[:, step_i]

                    # Mixed execution: prob beta -> expert, else learner
                    use_expert = mixed_execution_mask(mix_rng, beta, num_envs)
                    exec_act = jnp.where(
                        use_expert,
                        expert_act,
                        learner_act,
                    )

                    o_next, st, rew, done, info = env.step(
                        s_rng,
                        st,
                        exec_act,
                        env_params,
                    )
                    # Yield the visited obs ``o`` (not ``o_next``) so the
                    # paired (obs_t, expert_act_t) is consistent.
                    return (st, o_next, r, new_hs, done), (
                        o,
                        expert_act,
                        rew,
                        done,
                        info,
                    )

                (
                    final_c,
                    (
                        obs_seq,
                        expert_acts,
                        rews,
                        dones,
                        infos,
                    ),
                ) = jax.lax.scan(
                    _sim_step,
                    (es, cur_obs, sim_rng, hs, p_done),
                    jnp.arange(plan_horizon),
                )
                # obs_seq:     [H, E, obs_dim]
                # expert_acts: [H, E]
                # dones:       [H, E]
                es_next, obs_next, _, hs_next, done_next = final_c

                return (es_next, obs_next, rng, hs_next, done_next), (
                    obs_seq,
                    expert_acts,
                    rews,
                    dones,
                    infos,
                )

            # ppo_hs and prev_done persist across cycles and
            # updates via the scan carry.
            (env_state, obs, rng, ppo_hs, prev_done), traj = jax.lax.scan(
                _plan_and_execute,
                (env_state, obs, rng, ppo_hs, prev_done),
                None,
                n_cycles,
            )
            # traj_obs:         [C, H, E, obs_dim]
            # traj_expert_acts: [C, H, E]
            # traj_dones:       [C, H, E]
            (
                traj_obs,
                traj_expert_acts,
                traj_rew,
                traj_dones,
                all_infos,
            ) = traj

            # concatenate cycles into one [T, E, ...] rollout.  Cycles
            # are contiguous in time so a flat reshape preserves order.
            T = num_steps
            obs_t = traj_obs.reshape(T, num_envs, obs_dim)  # [T, E, D]
            acts_t = traj_expert_acts.reshape(T, num_envs)  # [T, E]
            dones_t = traj_dones.reshape(T, num_envs)  # [T, E]

            # sliding-window extraction (shared helper, same semantics
            # as offline.py): window (t, e) is valid iff dones_t[t..t+H-2]
            # are all False; dones_t already marks post-action resets.
            obs_w, act_w, valid_w = extract_sliding_windows(
                obs_t, acts_t, dones_t, plan_horizon
            )
            # obs_w:   [W, E, D]
            # act_w:   [W, E, H]
            # valid_w: [W, E]
            flat_obs = obs_w.reshape(-1, obs_dim)
            flat_plans = act_w.reshape(-1, plan_horizon)
            flat_valid = valid_w.reshape(-1).astype(jnp.float32)

            # write new samples into circular replay buffer
            write_indices = (
                buf_write_idx + jnp.arange(samples_per_update)
            ) % max_buffer_size
            buf_obs = buf_obs.at[write_indices].set(flat_obs)
            buf_plans = buf_plans.at[write_indices].set(flat_plans)
            buf_valid = buf_valid.at[write_indices].set(flat_valid)
            buf_write_idx = (buf_write_idx + samples_per_update) % max_buffer_size
            buf_fill = jnp.minimum(
                buf_fill + samples_per_update,
                max_buffer_size,
            )

            # multi-pass training over the aggregated buffer.  Each
            # pass redraws a fresh sample of size ``samples_per_update``
            # from the filled portion of the buffer; the default is a
            # single pass, matching offline BC's per-update gradient
            # work exactly (see ``dagger_sizing``); raise
            # ``DAGGER_TRAIN_PASSES`` to trade that fairness for more
            # per-update buffer coverage.
            def _pass(pass_state, _):
                state, rng = pass_state
                rng, sample_rng = jax.random.split(rng)
                buf_indices = jax.random.randint(
                    sample_rng,
                    (samples_per_update,),
                    0,
                    buf_fill,
                )
                dataset = (
                    buf_obs[buf_indices],
                    buf_plans[buf_indices],
                    buf_valid[buf_indices],
                )

                def _epoch(epoch_state, _):
                    state, ds, rng = epoch_state
                    rng, perm_rng = jax.random.split(rng)
                    perm = jax.random.permutation(
                        perm_rng,
                        samples_per_update,
                    )
                    shuffled = jax.tree.map(
                        lambda x: jnp.take(x, perm, axis=0),
                        ds,
                    )
                    batches = jax.tree.map(
                        lambda x: x.reshape(
                            num_minibatches,
                            -1,
                            *x.shape[1:],
                        ),
                        shuffled,
                    )

                    def _mb(mb_carry, batch):
                        st, rng = mb_carry
                        rng, loss_rng = jax.random.split(rng)
                        obs_b, act_b, val_b = batch
                        adv_b = jnp.ones(act_b.shape[0])
                        st, metrics = grad_step(
                            st,
                            act_b,
                            obs_b,
                            val_b,
                            loss_rng,
                            advantages=adv_b,
                        )
                        return (st, rng), metrics

                    (state, rng), metrics = jax.lax.scan(
                        _mb,
                        (state, rng),
                        batches,
                    )
                    return (state, ds, rng), metrics

                (state, _, rng), loss_info = jax.lax.scan(
                    _epoch,
                    (state, dataset, rng),
                    None,
                    update_epochs,
                )
                return (state, rng), loss_info

            (state, rng), loss_info = jax.lax.scan(
                _pass,
                (state, rng),
                None,
                n_train_passes,
            )

            # --- Metrics ------------------------------------------
            metric = jax.tree.map(jnp.mean, loss_info)
            returned = all_infos["returned_episode"]
            env_metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8),
                all_infos,
            )
            metric.update(env_metrics)
            metric["beta"] = beta
            metric["reward_mean"] = jnp.mean(traj_rew)
            metric["buffer_fill"] = buf_fill.astype(jnp.float32)
            metric["valid_frac"] = jnp.mean(flat_valid)

            # --- Periodic validation ------------------------------
            rng, val_rng = jax.random.split(rng)
            dummy = jax.tree.map(
                jnp.zeros_like,
                {f"val/{k}": v for k, v in env_metrics.items()},
            )
            val_metrics = jax.lax.cond(
                step_idx % val_interval == 0,
                lambda: _validate(state, val_rng),
                lambda: dummy,
            )
            metric.update(val_metrics)

            # Best-model tracking: update when validation improves.  Under
            # cond so the full-tree select only runs on val steps.
            is_val_step = step_idx % val_interval == 0

            def _update_best():
                val_ret = val_metrics.get(
                    "val/returned_episode_returns",
                    jnp.array(-jnp.inf),
                )
                improved = val_ret > best_val_return
                new_best = jax.tree.map(
                    lambda b, c: jnp.where(improved, c, b),
                    best_params,
                    state.params,
                )
                return new_best, jnp.where(improved, val_ret, best_val_return)

            best_params, best_val_return = jax.lax.cond(
                is_val_step,
                _update_best,
                lambda: (best_params, best_val_return),
            )
            metric["best_val_return"] = best_val_return

            if _wandb_log is not None:
                jax.debug.callback(_wandb_log, metric, step_idx)

            new_carry = DAggerCarry(
                train_state=state,
                env_state=env_state,
                obs=obs,
                rng=rng,
                step_idx=step_idx + 1,
                ppo_hs=ppo_hs,
                prev_done=prev_done,
                buf_obs=buf_obs,
                buf_plans=buf_plans,
                buf_valid=buf_valid,
                buf_write_idx=buf_write_idx,
                buf_fill=buf_fill,
                best_params=best_params,
                best_val_return=best_val_return,
            )
            return new_carry, metric

        # --- Outer scan -------------------------------------------
        rng, run_rng = jax.random.split(rng)
        runner_init = DAggerCarry(
            train_state=state,
            env_state=env_state,
            obs=obs,
            rng=run_rng,
            step_idx=resume_step,
            ppo_hs=ppo.init_hidden(num_envs),
            prev_done=jnp.zeros(num_envs, dtype=bool),
            buf_obs=buf_obs,
            buf_plans=buf_plans,
            buf_valid=buf_valid,
            buf_write_idx=jnp.int32(0),
            buf_fill=jnp.int32(0),
            best_params=params,
            best_val_return=jnp.array(-jnp.inf),
        )
        runner_final, metrics = jax.lax.scan(
            _update_step,
            runner_init,
            None,
            scan_length,
        )
        return {
            "runner_state": runner_final,
            "metrics": metrics,
            "best_params": runner_final.best_params,
        }

    return train


def run_online(config: dict[str, Any]) -> dict[str, Any]:
    """Configure, compile, and run DAgger online training.

    Args:
        config: Mixed-case hyperparameter dict from ``defaults.yaml`` / CLI.

    Returns:
        The training output: ``runner_state``, ``metrics`` and ``best_params``,
        each with a leading ``NUM_REPEATS`` axis.  ``--mode online`` discards
        this; ``--mode smoke`` uses it to print a summary.
    """
    config = {k.upper(): v for k, v in config.items()}

    # ONLINE_TOTAL_TIMESTEPS (env frames) is the hardware-portable source of
    # truth: invariant under num_envs changes, so the same config trains the
    # same amount of environment experience on any GPU.  ONLINE_NUM_UPDATES is
    # kept as a legacy fallback for configs that prefer the update form.
    resolve_num_updates(config, "online")
    # Translate env-frame-denominated hyperparameters (LR_WARMUP_FRAMES,
    # VAL_INTERVAL_FRAMES, DAGGER_BETA_FINAL, DAGGER_BUFFER_CYCLES) into
    # their update-step legacy keys.  Must run AFTER resolve_num_updates
    # because DAGGER_BETA_FINAL needs NUM_UPDATES.
    resolve_scaled_hyperparams(config, "online")
    print_config_snapshot(config, "online")

    if config.get("USE_WANDB"):
        init_wandb(
            config,
            name=f"{config['ENV_NAME']}-Online-Diffusion-DAgger-{int(config['ONLINE_TOTAL_TIMESTEPS'] // 1e6)}M",
            resume_run_id=config.get("RESUME_WANDB_RUN_ID"),
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config.get("NUM_REPEATS", 1))

    train_fn = jax.jit(jax.vmap(make_train_online_dagger(config)))

    out, timing = compile_and_run(train_fn, rngs, config["ONLINE_TOTAL_TIMESTEPS"])
    print(format_timing(timing))

    if config.get("USE_WANDB") and config.get("SAVE_POLICY"):
        # Final checkpoint (last iteration params)
        train_states = out["runner_state"].train_state
        train_state = jax.tree.map(lambda x: x[0], train_states)
        path = os.path.join(wandb.run.dir, "policies")
        with ocp.CheckpointManager(
            path,
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        ) as mgr:
            mgr.save(
                int(config["NUM_UPDATES"]),
                args=ocp.args.StandardSave(train_state),
            )
        print(f"Saved final policy to {path}")

        num_updates = config["NUM_UPDATES"]
        save_checkpoint_metadata(
            path,
            mode="online",
            update_step=num_updates,
            total_gradient_steps=num_updates
            * config["UPDATE_EPOCHS"]
            * config["NUM_MINIBATCHES"],
            wandb_run_id=wandb.run.id if wandb.run else None,
            config=config,
        )

        artifact = wandb.Artifact(
            name=f"{config['ENV_NAME']}-policy",
            type="model",
            metadata=config,
        )
        artifact.add_dir(path)
        wandb.log_artifact(artifact)

        # Best checkpoint (highest validation return).
        # Wrap in a dummy TrainState so the Orbax structure matches
        # the final checkpoint — load_checkpoint expects TrainState.
        best_params = jax.tree.map(
            lambda x: x[0],
            out["best_params"],
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        best_state = TrainState.create(
            apply_fn=lambda *a: None,
            params=best_params,
            tx=tx,
        )
        best_path = os.path.join(wandb.run.dir, "policies_best")
        with ocp.CheckpointManager(
            best_path,
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        ) as mgr:
            mgr.save(0, args=ocp.args.StandardSave(best_state))
        print(f"Saved best policy to {best_path}")

        best_artifact = wandb.Artifact(
            name=f"{config['ENV_NAME']}-policy-best",
            type="model",
            metadata=config,
        )
        best_artifact.add_dir(best_path)
        wandb.log_artifact(best_artifact)

        print("Uploaded final + best policy artifacts to wandb")

    return out
