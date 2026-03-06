"""Training and inference scripts for the ReMDM discrete diffusion planner on Craftax.

Usage:
    # Collect offline trajectories from a trained PPO checkpoint
    python planners.py --mode collect \\
        --ppo_checkpoint_path /path/to/ckpt \\
        --offline_data_path trajectories.npz

    # Train the diffusion model offline on collected trajectories
    python planners.py --mode offline --offline_data_path trajectories.npz

    # Online fine-tuning (optionally loading an offline pre-trained model)
    python planners.py --mode online
    python planners.py --mode online --offline_checkpoint_path /path/to/offline_ckpt

    # Evaluate a trained model
    python planners.py --mode inference --checkpoint_path /path/to/ckpt

All defaults are loaded from configs/defaults.yaml.  Any argument can be
overridden on the command line, e.g.:
    python planners.py --mode online --lr 1e-4 --num_envs 64
"""

import argparse
import os
import pathlib
import sys
import time
import yaml

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from craftax.craftax_env import make_craftax_env_from_name
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from models.denoiser import DenoisingTransformer
from remdm import (
    STRATEGY_MAP,
    compute_loss,
    cosine_schedule,
    linear_schedule,
    sample_plan,
)
from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper, PlannerWrapper

SCHEDULE_MAP = {"cosine": cosine_schedule, "linear": linear_schedule}


# =============================================================================
# PPO config helper
# =============================================================================


def _build_ppo_config(config: dict) -> dict:
    """Build a PPO training config from the PPO_-prefixed keys in the master config.

    The returned dict uses the uppercase key names that ppo_rnn.make_train and
    ppo_rnd.make_train expect.  Wandb logging is disabled for inline PPO training
    to avoid conflicts with the diffusion training run.
    """
    return {
        "ENV_NAME": config["ENV_NAME"],
        "TOTAL_TIMESTEPS": config["PPO_TOTAL_TIMESTEPS"],
        "NUM_STEPS": config["PPO_NUM_STEPS"],
        "NUM_ENVS": config["PPO_NUM_ENVS"],
        "LR": config["PPO_LR"],
        "UPDATE_EPOCHS": config["PPO_UPDATE_EPOCHS"],
        "NUM_MINIBATCHES": config["PPO_NUM_MINIBATCHES"],
        "GAMMA": config["PPO_GAMMA"],
        "GAE_LAMBDA": config["PPO_GAE_LAMBDA"],
        "CLIP_EPS": config["PPO_CLIP_EPS"],
        "ENT_COEF": config["PPO_ENT_COEF"],
        "VF_COEF": config["PPO_VF_COEF"],
        "MAX_GRAD_NORM": config["MAX_GRAD_NORM"],
        "LAYER_SIZE": config["PPO_LAYER_SIZE"],
        "ANNEAL_LR": config["PPO_ANNEAL_LR"],
        "USE_OPTIMISTIC_RESETS": config["PPO_USE_OPTIMISTIC_RESETS"],
        "OPTIMISTIC_RESET_RATIO": config["PPO_OPTIMISTIC_RESET_RATIO"],
        # Disable wandb for inline PPO so it doesn't conflict with the main run
        "USE_WANDB": False,
        "DEBUG": False,
        "NUM_REPEATS": 1,
        "WANDB_PROJECT": config.get("WANDB_PROJECT", ""),
        "WANDB_ENTITY": config.get("WANDB_ENTITY", ""),
        # RND-specific keys (ignored by ppo_rnn)
        "USE_RND": config.get("PPO_USE_RND", True),
        "RND_LAYER_SIZE": config.get("PPO_RND_LAYER_SIZE", 256),
        "RND_OUTPUT_SIZE": config.get("PPO_RND_OUTPUT_SIZE", 512),
        "RND_LR": config.get("PPO_RND_LR", 3.0e-4),
        "RND_REWARD_COEFF": config.get("PPO_RND_REWARD_COEFF", 1.0),
        "RND_LOSS_COEFF": config.get("PPO_RND_LOSS_COEFF", 0.01),
        "RND_GAE_COEFF": config.get("PPO_RND_GAE_COEFF", 0.01),
        "RND_IS_EPISODIC": config.get("PPO_RND_IS_EPISODIC", False),
        "EXPLORATION_UPDATE_EPOCHS": config.get("PPO_EXPLORATION_UPDATE_EPOCHS", 1),
    }


# =============================================================================
# Data Collection
# =============================================================================


def collect_offline_data(config: dict) -> None:
    """Roll out a PPO policy and save (obs, actions, dones) to disk.

    Three variants controlled by PPO_VARIANT:
        "checkpoint" — restore a pre-trained ActorCritic from PPO_CHECKPOINT_PATH.
        "rnn"        — train a fresh PPO-RNN agent (ppo_rnn.make_train) then collect.
        "rnd"        — train a fresh PPO-RND agent (ppo_rnd.make_train) then collect.

    In all cases the rollout saves COLLECT_NUM_STEPS transitions from
    COLLECT_NUM_ENVS parallel environments to OFFLINE_DATA_PATH.
    """
    ppo_variant = config.get("PPO_VARIANT", "checkpoint")

    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    num_envs = config["COLLECT_NUM_ENVS"]

    env_w = LogWrapper(env)
    env_w = AutoResetEnvWrapper(env_w)
    env_w = BatchEnvWrapper(env_w, num_envs=num_envs)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, env_rng, collect_rng = jax.random.split(rng, 3)

    # ── Obtain trained policy ──────────────────────────────────────────────────
    if ppo_variant == "rnn":
        from ppo_rnn import ActorCriticRNN, ScannedRNN
        from ppo_rnn import make_train as _make_ppo

        ppo_config = _build_ppo_config(config)
        rng, ppo_rng = jax.random.split(rng)
        print("Training PPO-RNN agent for data collection...")
        out = jax.jit(_make_ppo(ppo_config))(ppo_rng)
        ppo_params = out["runner_state"][0].params
        network = ActorCriticRNN(num_actions, config=ppo_config)
        init_hstate = ScannedRNN.initialize_carry(num_envs, ppo_config["LAYER_SIZE"])
        print("PPO-RNN training complete.")

        @jax.jit
        def _step(rng, env_state, obs, done, hstate):
            rng, k1, k2 = jax.random.split(rng, 3)
            ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
            hstate, pi, _ = network.apply(ppo_params, hstate, ac_in)
            action = pi.sample(seed=k1).squeeze(0)
            obs_next, env_state, _, done_next, _ = env_w.step(
                k2, env_state, action, env_params
            )
            return rng, env_state, obs_next, action, done_next, hstate

    elif ppo_variant == "rnd":
        from models.rnd import ActorCriticRND
        from ppo_rnd import make_train as _make_ppo

        ppo_config = _build_ppo_config(config)
        rng, ppo_rng = jax.random.split(rng)
        print("Training PPO-RND agent for data collection...")
        out = jax.jit(_make_ppo(ppo_config))(ppo_rng)
        ppo_params = out["runner_state"][0].params
        network = ActorCriticRND(num_actions, ppo_config["LAYER_SIZE"])
        init_hstate = jnp.zeros(())  # unused, keeps rollout loop uniform
        print("PPO-RND training complete.")

        @jax.jit
        def _step(rng, env_state, obs, done, hstate):
            rng, k1, k2 = jax.random.split(rng, 3)
            pi, _, _ = network.apply(ppo_params, obs)
            action = pi.sample(seed=k1)
            obs_next, env_state, _, done_next, _ = env_w.step(
                k2, env_state, action, env_params
            )
            return rng, env_state, obs_next, action, done_next, hstate

    else:  # "checkpoint"
        from models.actor_critic import ActorCritic

        assert config.get("PPO_CHECKPOINT_PATH"), (
            "PPO_CHECKPOINT_PATH is required when ppo_variant='checkpoint'"
        )
        network = ActorCritic(num_actions, config["LAYER_SIZE"])
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, *env.observation_space(env_params).shape))
        tmp_tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-4, eps=1e-5))
        tmp_ts = TrainState.create(
            apply_fn=network.apply,
            params=network.init(init_rng, dummy_obs),
            tx=tmp_tx,
        )
        checkpointer = PyTreeCheckpointer()
        ckpt_mgr = CheckpointManager(
            config["PPO_CHECKPOINT_PATH"],
            checkpointer,
            CheckpointManagerOptions(max_to_keep=1, create=True),
        )
        tmp_ts = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=tmp_ts)
        ppo_params = tmp_ts.params
        init_hstate = jnp.zeros(())  # unused, keeps rollout loop uniform
        print(f"Restored PPO checkpoint (step={ckpt_mgr.latest_step()})")

        @jax.jit
        def _step(rng, env_state, obs, done, hstate):
            rng, k1, k2 = jax.random.split(rng, 3)
            pi, _ = network.apply(ppo_params, obs)
            action = pi.sample(seed=k1)
            obs_next, env_state, _, done_next, _ = env_w.step(
                k2, env_state, action, env_params
            )
            return rng, env_state, obs_next, action, done_next, hstate

    # ── Unified rollout ────────────────────────────────────────────────────────
    obs, env_state = env_w.reset(env_rng, env_params)
    done = jnp.zeros(num_envs, dtype=bool)
    hstate = init_hstate

    num_iters = config["COLLECT_NUM_STEPS"] // num_envs
    all_obs, all_actions, all_dones = [], [], []

    for i in range(num_iters):
        collect_rng, env_state, obs_next, action, done, hstate = _step(
            collect_rng, env_state, obs, done, hstate
        )
        all_obs.append(np.array(obs))
        all_actions.append(np.array(action))
        all_dones.append(np.array(done))
        obs = obs_next
        if (i + 1) % 500 == 0:
            print(f"  {(i + 1) * num_envs:,} / {config['COLLECT_NUM_STEPS']:,} steps")

    obs_arr = np.concatenate(all_obs, axis=0)       # [T, obs_dim]
    act_arr = np.concatenate(all_actions, axis=0)   # [T]
    done_arr = np.concatenate(all_dones, axis=0)    # [T]

    out_path = config["OFFLINE_DATA_PATH"]
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, obs=obs_arr, actions=act_arr, dones=done_arr)
    print(f"Saved {obs_arr.shape[0]:,} transitions to '{out_path}'")


# =============================================================================
# Offline Training
# =============================================================================


def make_train_offline(config: dict, offline_data: dict):
    """Return train(rng) for offline MDLM training on pre-collected trajectories.

    Args:
        config:       Configuration dict (all-uppercase keys).
        offline_data: Dict with 'obs' [N, obs_dim] float32 and 'actions' [N] int32.

    Returns:
        train: Callable[[PRNGKey], dict] — JIT-able training function.
    """
    obs_data = jnp.array(offline_data["obs"], dtype=jnp.float32)   # [N, obs_dim]
    act_data = jnp.array(offline_data["actions"], dtype=jnp.int32)  # [N]
    N, obs_dim = obs_data.shape

    plan_horizon = config["PLAN_HORIZON"]
    batch_size = config["BATCH_SIZE"]
    num_actions = config["NUM_ACTIONS"]
    num_train_steps = config["NUM_TRAIN_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    model = DenoisingTransformer(
        num_actions=num_actions,
        plan_horizon=plan_horizon,
        d_model=config["D_MODEL"],
        n_heads=config["N_HEADS"],
        n_layers=config["N_LAYERS"],
        d_ff=config["D_FF"],
        obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
        obs_encoder_width=config["OBS_ENCODER_WIDTH"],
        dropout_rate=config["DROPOUT_RATE"],
    )

    def train(rng):
        # Init model
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
        dummy_t = jnp.zeros((1,))
        params = model.init(init_rng, dummy_obs, dummy_act, dummy_t)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        def _train_step(carry, step_idx):
            train_state, rng = carry
            rng, sample_rng, loss_rng = jax.random.split(rng, 3)

            # Sample random starting positions in the flat trajectory buffer
            start_idxs = jax.random.randint(
                sample_rng, (batch_size,), 0, N - plan_horizon
            )
            obs_batch = obs_data[start_idxs]  # [B, obs_dim]
            # Gather contiguous plan_horizon-length action sequences
            act_batch = jax.vmap(
                lambda i: jax.lax.dynamic_slice(act_data, (i,), (plan_horizon,))
            )(start_idxs)  # [B, plan_horizon]

            def loss_fn(params):
                def _apply(p, obs, z_t, t):
                    return model.apply(p, obs, z_t, t, deterministic=False)

                return compute_loss(
                    _apply, params, loss_rng,
                    act_batch, obs_batch, num_actions, schedule_fn,
                )

            (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                train_state.params
            )
            train_state = train_state.apply_gradients(grads=grads)

            if config["DEBUG"] and config["USE_WANDB"]:
                def _log(info, step):
                    wandb.log(
                        {
                            "diffusion_loss": float(info["loss"]),
                            "mean_t": float(info["mean_t"]),
                            "frac_masked": float(info["frac_masked"]),
                        },
                        step=int(step),
                    )

                jax.debug.callback(_log, info, step_idx)

            return (train_state, rng), info

        (train_state, _), metrics = jax.lax.scan(
            _train_step, (train_state, rng), jnp.arange(num_train_steps)
        )
        return {"train_state": train_state, "metrics": metrics}

    return train


# =============================================================================
# Online Training
# =============================================================================


def make_train_online(config: dict, init_params=None):
    """Return train(rng) for online fine-tuning with the diffusion model as policy.

    At each update step:
      1. Generate plans via sample_plan and execute REPLAN_EVERY actions per plan.
      2. Collect (obs, plan) pairs as training data (self-imitation).
      3. Fine-tune the model on those pairs for UPDATE_EPOCHS passes.

    Args:
        config:      Configuration dict (all-uppercase keys).
        init_params: Optional pre-loaded model parameters (e.g. from offline checkpoint).

    Returns:
        train: Callable[[PRNGKey], dict] — JIT-able training function.
    """
    assert config["NUM_STEPS"] % config["REPLAN_EVERY"] == 0, (
        "NUM_STEPS must be divisible by REPLAN_EVERY"
    )

    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]

    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    replan_every = config["REPLAN_EVERY"]
    num_updates = config["NUM_UPDATES"]
    update_epochs = config["UPDATE_EPOCHS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    diffusion_steps = config["DIFFUSION_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy = config["REMASK_STRATEGY"]
    eta = config["ETA"]
    t_on = config.get("T_ON", 0.7)
    t_off = config.get("T_OFF", 0.3)
    num_plan_cycles = config["NUM_STEPS"] // replan_every

    env_w = LogWrapper(env)
    env_w = AutoResetEnvWrapper(env_w)
    env_w = BatchEnvWrapper(env_w, num_envs=num_envs)

    model = DenoisingTransformer(
        num_actions=num_actions,
        plan_horizon=plan_horizon,
        d_model=config["D_MODEL"],
        n_heads=config["N_HEADS"],
        n_layers=config["N_LAYERS"],
        d_ff=config["D_FF"],
        obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
        obs_encoder_width=config["OBS_ENCODER_WIDTH"],
        dropout_rate=config["DROPOUT_RATE"],
    )

    def _apply_inference(params, obs, z_t, t):
        return model.apply(params, obs, z_t, t)  # deterministic=True

    def _apply_train(params, obs, z_t, t):
        return model.apply(params, obs, z_t, t, deterministic=False)

    def train(rng):
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
        dummy_t = jnp.zeros((1,))
        params = model.init(init_rng, dummy_obs, dummy_act, dummy_t)

        if init_params is not None:
            params = init_params

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        obs, env_state = env_w.reset(env_rng, env_params)

        def _update_step(runner_state, _):
            train_state, env_state, obs, rng, update_step = runner_state

            # ── Collect: generate plans and execute them ─────────────────────
            def _plan_and_execute(carry, _):
                env_state, obs, rng = carry
                rng, plan_rng = jax.random.split(rng)

                # Generate a plan for the current observation
                plan = sample_plan(
                    _apply_inference,
                    train_state.params,
                    plan_rng,
                    obs,
                    num_actions,
                    plan_horizon,
                    diffusion_steps,
                    schedule_fn,
                    remask_strategy,
                    eta,
                    t_on,
                    t_off,
                )
                # plan: [num_envs, plan_horizon]

                # Execute replan_every actions from the plan
                def _exec_step(carry, step_idx):
                    env_state, _, rng = carry
                    action = plan[:, step_idx]
                    rng, step_rng = jax.random.split(rng)
                    obs_next, env_state, reward, done, info = env_w.step(
                        step_rng, env_state, action, env_params
                    )
                    return (env_state, obs_next, rng), (reward, done, info)

                (env_state, obs_next, rng), (rewards, dones, infos) = jax.lax.scan(
                    _exec_step,
                    (env_state, obs, rng),
                    jnp.arange(replan_every),
                )
                # rewards: [replan_every, num_envs]

                return (env_state, obs_next, rng), (obs, plan, rewards, dones, infos)

            (env_state, obs, rng), traj = jax.lax.scan(
                _plan_and_execute,
                (env_state, obs, rng),
                None,
                num_plan_cycles,
            )
            # traj_obs:    [num_plan_cycles, num_envs, obs_dim]
            # traj_plans:  [num_plan_cycles, num_envs, plan_horizon]
            # all_rewards: [num_plan_cycles, replan_every, num_envs]
            # all_dones:   [num_plan_cycles, replan_every, num_envs]
            # all_infos:   pytree of [num_plan_cycles, replan_every, num_envs, ...]
            traj_obs, traj_plans, *_, all_infos = traj

            # ── Train on collected (obs, plan) pairs ──────────────────────────
            total_samples = num_plan_cycles * num_envs
            minibatch_size = total_samples // num_minibatches
            flat_obs = traj_obs.reshape(total_samples, obs_dim)
            flat_plans = traj_plans.reshape(total_samples, plan_horizon)

            def _update_epoch(carry, _):
                train_state, rng = carry
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, total_samples)
                obs_shuf = flat_obs[perm]
                plan_shuf = flat_plans[perm]
                obs_mbs = obs_shuf.reshape(num_minibatches, minibatch_size, obs_dim)
                plan_mbs = plan_shuf.reshape(
                    num_minibatches, minibatch_size, plan_horizon
                )

                def _update_minibatch(ts_rng, idx_and_mb):
                    ts, rng = ts_rng
                    mb_idx, obs_mb, plan_mb = idx_and_mb
                    loss_rng = jax.random.fold_in(rng, mb_idx)

                    def loss_fn(params):
                        return compute_loss(
                            _apply_train, params, loss_rng,
                            plan_mb, obs_mb, num_actions, schedule_fn,
                        )

                    (_, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                        ts.params
                    )
                    ts = ts.apply_gradients(grads=grads)
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

            # ── Metrics ───────────────────────────────────────────────────────
            ep_returns = all_infos["returned_episode_returns"]  # [cycles, replan, E]
            ep_mask = all_infos["returned_episode"]             # [cycles, replan, E]
            mean_ep_return = jnp.where(
                ep_mask.sum() > 0,
                (ep_returns * ep_mask).sum() / ep_mask.sum(),
                jnp.nan,
            )

            metric = jax.tree.map(
                lambda x: (x * ep_mask).sum() / jnp.maximum(ep_mask.sum(), 1),
                all_infos,
            )
            metric["diffusion_loss"] = epoch_infos["loss"].mean()
            metric["mean_t"] = epoch_infos["mean_t"].mean()
            metric["frac_masked"] = epoch_infos["frac_masked"].mean()
            metric["episode_return"] = mean_ep_return

            if config["DEBUG"] and config["USE_WANDB"]:
                def _log(loss, mean_t, frac_masked, ep_return, update_step):
                    wandb.log(
                        {
                            "diffusion_loss": float(loss),
                            "mean_t": float(mean_t),
                            "frac_masked": float(frac_masked),
                            "episode_return": float(ep_return),
                        },
                        step=int(update_step),
                    )

                jax.debug.callback(
                    _log,
                    metric["diffusion_loss"],
                    metric["mean_t"],
                    metric["frac_masked"],
                    metric["episode_return"],
                    update_step,
                )

            runner_state = (train_state, env_state, obs, rng, update_step + 1)
            return runner_state, metric

        runner_state = (train_state, env_state, obs, rng, 0)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# =============================================================================
# Entry Points
# =============================================================================


def _save_model(train_state, config: dict, dir_name: str) -> None:
    """Save a TrainState checkpoint using orbax."""
    orbax_checkpointer = PyTreeCheckpointer()
    options = CheckpointManagerOptions(max_to_keep=1, create=True)
    path = os.path.join(wandb.run.dir, dir_name) if config.get("USE_WANDB") else dir_name
    ckpt_mgr = CheckpointManager(path, orbax_checkpointer, options)
    save_args = orbax_utils.save_args_from_target(train_state)
    ckpt_mgr.save(
        config.get("NUM_TRAIN_STEPS", config.get("NUM_UPDATES", 0)),
        train_state,
        save_kwargs={"save_args": save_args},
    )
    print(f"Saved model checkpoint to '{path}'")


def run_collect(config: dict) -> None:
    collect_offline_data(config)


def run_offline(config: dict) -> None:
    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"remdm-offline-{config['ENV_NAME']}",
        )

    offline_data = np.load(config["OFFLINE_DATA_PATH"])
    config["NUM_ACTIONS"] = int(offline_data["actions"].max()) + 1
    print(
        f"Loaded {offline_data['obs'].shape[0]:,} transitions "
        f"(obs_dim={offline_data['obs'].shape[1]}, "
        f"num_actions={config['NUM_ACTIONS']})"
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_fn = make_train_offline(config, offline_data)
    train_jit = jax.jit(train_fn)

    t0 = time.time()
    # Run each repeat sequentially to avoid closing over large arrays in vmap
    outs = [train_jit(rngs[i]) for i in range(config["NUM_REPEATS"])]
    t1 = time.time()
    print(f"Offline training time: {t1 - t0:.1f}s")

    if config["USE_WANDB"] and config["SAVE_POLICY"]:
        _save_model(outs[0]["train_state"], config, "diffusion_offline")


def run_online(config: dict) -> None:
    # Derive NUM_ACTIONS from the environment
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = env.action_space(env.default_params).n

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=f"remdm-online-{config['ENV_NAME']}",
        )

    # Load offline checkpoint OUTSIDE JIT (I/O cannot happen inside JIT)
    init_params = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        plan_horizon = config["PLAN_HORIZON"]
        num_actions = config["NUM_ACTIONS"]
        obs_dim = env.observation_space(env.default_params).shape[0]
        tmp_model = DenoisingTransformer(
            num_actions=num_actions,
            plan_horizon=plan_horizon,
            d_model=config["D_MODEL"],
            n_heads=config["N_HEADS"],
            n_layers=config["N_LAYERS"],
            d_ff=config["D_FF"],
            obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
            obs_encoder_width=config["OBS_ENCODER_WIDTH"],
            dropout_rate=config["DROPOUT_RATE"],
        )
        tmp_rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
        dummy_t = jnp.zeros((1,))
        tmp_params = tmp_model.init(tmp_rng, dummy_obs, dummy_act, dummy_t)
        tmp_tx = optax.adam(config["LR"])
        tmp_ts = TrainState.create(apply_fn=tmp_model.apply, params=tmp_params, tx=tmp_tx)
        checkpointer = PyTreeCheckpointer()
        ckpt_mgr = CheckpointManager(
            config["OFFLINE_CHECKPOINT_PATH"],
            checkpointer,
            CheckpointManagerOptions(max_to_keep=1, create=True),
        )
        tmp_ts = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=tmp_ts)
        init_params = tmp_ts.params
        print(f"Loaded offline checkpoint (step={ckpt_mgr.latest_step()})")

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_fn = make_train_online(config, init_params=init_params)
    train_jit = jax.jit(train_fn)
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print(f"Online training time: {t1 - t0:.1f}s")
    total_steps = config["NUM_UPDATES"] * config["NUM_STEPS"] * config["NUM_ENVS"]
    print(f"SPS: {total_steps / (t1 - t0):.0f}")

    if config["USE_WANDB"] and config["SAVE_POLICY"]:
        train_state = jax.tree.map(lambda x: x[0], out["runner_state"][0])
        _save_model(train_state, config, "diffusion_online")


# =============================================================================
# Inference (using PlannerWrapper)
# =============================================================================


def run_inference(config: dict) -> None:
    """Evaluate a trained diffusion planner using PlannerWrapper.

    Loads a checkpoint, wraps the environment with PlannerWrapper for
    automatic plan generation and execution, and runs for EVAL_STEPS steps.
    """
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    config["NUM_ACTIONS"] = num_actions

    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    replan_every = config["REPLAN_EVERY"]
    diffusion_steps = config["DIFFUSION_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    remask_strategy = config["REMASK_STRATEGY"]
    eta = config["ETA"]
    t_on = config.get("T_ON", 0.7)
    t_off = config.get("T_OFF", 0.3)
    eval_steps = config.get("EVAL_STEPS", 1000)

    model = DenoisingTransformer(
        num_actions=num_actions,
        plan_horizon=plan_horizon,
        d_model=config["D_MODEL"],
        n_heads=config["N_HEADS"],
        n_layers=config["N_LAYERS"],
        d_ff=config["D_FF"],
        obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
        obs_encoder_width=config["OBS_ENCODER_WIDTH"],
        dropout_rate=config["DROPOUT_RATE"],
    )

    # Init and load checkpoint
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_act = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
    dummy_t = jnp.zeros((1,))
    params = model.init(init_rng, dummy_obs, dummy_act, dummy_t)

    assert config.get("CHECKPOINT_PATH"), "--checkpoint_path required for inference"
    checkpointer = PyTreeCheckpointer()
    ckpt_mgr = CheckpointManager(
        config["CHECKPOINT_PATH"],
        checkpointer,
        CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    tmp_ts = TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adam(config["LR"]),
    )
    tmp_ts = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=tmp_ts)
    model_params = tmp_ts.params
    print(f"Loaded checkpoint (step={ckpt_mgr.latest_step()})")

    def _apply_inference(params, obs, z_t, t):
        return model.apply(params, obs, z_t, t)

    def planner_apply_fn(rng, model_params, obs):
        return sample_plan(
            _apply_inference, model_params, rng, obs,
            num_actions, plan_horizon, diffusion_steps, schedule_fn,
            remask_strategy, eta, t_on, t_off,
        )

    # Build wrapper stack: env → LogWrapper → AutoReset → Batch → Planner
    env_w = LogWrapper(env)
    env_w = AutoResetEnvWrapper(env_w)
    env_w = BatchEnvWrapper(env_w, num_envs=num_envs)
    env_w = PlannerWrapper(
        env_w,
        num_envs=num_envs,
        plan_horizon=plan_horizon,
        replan_every=replan_every,
        planner_apply_fn=planner_apply_fn,
    )

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

        (obs, state, rng), (rewards, dones, infos) = jax.lax.scan(
            _step, (obs, state, rng), None, eval_steps,
        )
        return rewards, dones, infos

    t0 = time.time()
    rewards, dones, infos = _eval_loop(rng)
    t1 = time.time()

    ep_returns = infos["returned_episode_returns"]
    ep_mask = infos["returned_episode"]
    completed = ep_mask.sum()
    mean_return = jnp.where(
        completed > 0,
        (ep_returns * ep_mask).sum() / completed,
        jnp.nan,
    )
    print(f"Eval time: {t1 - t0:.1f}s ({eval_steps * num_envs} steps)")
    print(f"Completed episodes: {int(completed)}")
    print(f"Mean episode return: {float(mean_return):.2f}")
    print(f"Mean step reward: {float(rewards.mean()):.4f}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # ── Load defaults from YAML ────────────────────────────────────────────────
    _src_dir = pathlib.Path(__file__).parent
    _default_cfg_path = _src_dir.parent / "configs" / "defaults.yaml"

    # Pre-parse just --config so we can load the right file before building the
    # full parser (avoids chicken-and-egg with set_defaults).
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=str(_default_cfg_path))
    _pre_args, _ = _pre.parse_known_args()

    with open(_pre_args.config) as _f:
        _yaml_defaults = yaml.safe_load(_f)
    # YAML nulls become None, large ints stay int — no conversion needed.

    # ── Build full parser (YAML values become argparse defaults) ───────────────
    parser = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default=str(_default_cfg_path),
        help="Path to a YAML config file (overridden by any explicit CLI flag).",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "offline", "online", "inference"],
        required=True,
        help="Mode: collect offline data, train offline, train online, or run inference.",
    )

    # Environment
    parser.add_argument("--env_name", type=str)

    # Diffusion model
    parser.add_argument("--plan_horizon", type=int)
    parser.add_argument("--diffusion_steps", type=int)
    parser.add_argument("--diffusion_schedule", type=str, choices=["cosine", "linear"])
    parser.add_argument(
        "--remask_strategy", type=str, choices=list(STRATEGY_MAP.keys()),
    )
    parser.add_argument("--eta", type=float)
    parser.add_argument("--t_on", type=float)
    parser.add_argument("--t_off", type=float)

    # Transformer architecture
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--obs_encoder_layers", type=int)
    parser.add_argument("--obs_encoder_width", type=int)
    parser.add_argument("--dropout_rate", type=float)

    # Optimisation
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--batch_size", type=int)

    # Offline training
    parser.add_argument("--offline_data_path", type=str)
    parser.add_argument("--num_train_steps", type=lambda x: int(float(x)))

    # Online training
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--num_updates", type=lambda x: int(float(x)))
    parser.add_argument("--replan_every", type=int)
    parser.add_argument("--update_epochs", type=int)
    parser.add_argument("--num_minibatches", type=int)
    parser.add_argument(
        "--offline_checkpoint_path", type=str,
        help="Path to an offline-trained checkpoint to warm-start online training.",
    )

    # Inference
    parser.add_argument(
        "--checkpoint_path", type=str,
        help="Path to a trained model checkpoint for inference.",
    )
    parser.add_argument("--eval_steps", type=int)

    # Data collection
    parser.add_argument(
        "--ppo_variant", type=str, choices=["rnn", "rnd", "checkpoint"],
        help="PPO variant for data collection: train rnn/rnd from scratch, or load a checkpoint.",
    )
    parser.add_argument("--collect_num_steps", type=lambda x: int(float(x)))
    parser.add_argument("--collect_num_envs", type=int)
    parser.add_argument("--layer_size", type=int)
    parser.add_argument(
        "--ppo_checkpoint_path", type=str,
        help="Path to a pre-trained PPO checkpoint (only used when --ppo_variant checkpoint).",
    )

    # PPO training (used when ppo_variant is rnn or rnd)
    parser.add_argument("--ppo_total_timesteps", type=lambda x: int(float(x)))
    parser.add_argument("--ppo_num_envs", type=int)
    parser.add_argument("--ppo_num_steps", type=int)
    parser.add_argument("--ppo_lr", type=float)
    parser.add_argument("--ppo_update_epochs", type=int)
    parser.add_argument("--ppo_num_minibatches", type=int)
    parser.add_argument("--ppo_gamma", type=float)
    parser.add_argument("--ppo_gae_lambda", type=float)
    parser.add_argument("--ppo_clip_eps", type=float)
    parser.add_argument("--ppo_ent_coef", type=float)
    parser.add_argument("--ppo_vf_coef", type=float)
    parser.add_argument("--ppo_layer_size", type=int)
    parser.add_argument("--ppo_anneal_lr", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ppo_use_optimistic_resets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ppo_optimistic_reset_ratio", type=int)

    # PPO-RND specific
    parser.add_argument("--ppo_use_rnd", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ppo_rnd_layer_size", type=int)
    parser.add_argument("--ppo_rnd_output_size", type=int)
    parser.add_argument("--ppo_rnd_lr", type=float)
    parser.add_argument("--ppo_rnd_reward_coeff", type=float)
    parser.add_argument("--ppo_rnd_loss_coeff", type=float)
    parser.add_argument("--ppo_rnd_gae_coeff", type=float)
    parser.add_argument("--ppo_rnd_is_episodic", action=argparse.BooleanOptionalAction)
    parser.add_argument("--ppo_exploration_update_epochs", type=int)

    # W&B / logging
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--save_policy", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction)

    # Inject YAML values as argparse defaults (CLI flags override them)
    parser.set_defaults(**_yaml_defaults)

    args, rest = parser.parse_known_args(sys.argv[1:])
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    config = {k.upper(): v for k, v in vars(args).items()}
    # Remove the config-file path itself from the downstream config dict
    config.pop("CONFIG", None)

    def _run():
        if config["MODE"] == "collect":
            if config.get("PPO_VARIANT", "checkpoint") == "checkpoint":
                assert config.get("PPO_CHECKPOINT_PATH"), (
                    "--ppo_checkpoint_path is required when --ppo_variant checkpoint"
                )
            run_collect(config)
        elif config["MODE"] == "offline":
            run_offline(config)
        elif config["MODE"] == "online":
            run_online(config)
        elif config["MODE"] == "inference":
            run_inference(config)

    if args.jit:
        _run()
    else:
        with jax.disable_jit():
            _run()
