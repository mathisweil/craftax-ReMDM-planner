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
"""

import argparse
import os
import sys
import time

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
from wrappers import AutoResetEnvWrapper, BatchEnvWrapper, LogWrapper

SCHEDULE_MAP = {"cosine": cosine_schedule, "linear": linear_schedule}


# =============================================================================
# Data Collection
# =============================================================================


def collect_offline_data(config: dict) -> None:
    """Roll out a trained PPO agent and save (obs, actions, dones) to disk.

    Restores the network from a W&B-style orbax checkpoint directory and
    runs it for COLLECT_NUM_STEPS steps across COLLECT_NUM_ENVS parallel
    environments.  The resulting flat arrays are written to OFFLINE_DATA_PATH.
    """
    from models.actor_critic import ActorCritic

    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_envs = config["COLLECT_NUM_ENVS"]

    env_w = LogWrapper(env)
    env_w = AutoResetEnvWrapper(env_w)
    env_w = BatchEnvWrapper(env_w, num_envs=num_envs)

    # Restore PPO actor-critic
    network = ActorCritic(env.action_space(env_params).n, config["LAYER_SIZE"])
    rng = jax.random.PRNGKey(config["SEED"])
    rng, init_rng, env_rng = jax.random.split(rng, 3)
    dummy_obs = jnp.zeros((1, *env.observation_space(env_params).shape))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(2e-4, eps=1e-5))
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network.init(init_rng, dummy_obs),
        tx=tx,
    )

    checkpointer = PyTreeCheckpointer()
    ckpt_mgr = CheckpointManager(
        config["PPO_CHECKPOINT_PATH"],
        checkpointer,
        CheckpointManagerOptions(max_to_keep=1, create=True),
    )
    train_state = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=train_state)  # type: ignore[assignment]
    print(f"Restored PPO checkpoint (step={ckpt_mgr.latest_step()})")

    obs, env_state = env_w.reset(env_rng, env_params)

    @jax.jit
    def _step(rng, env_state, obs):
        pi, _ = network.apply(train_state.params, obs)  # type: ignore[union-attr]
        rng, k1, k2 = jax.random.split(rng, 3)
        action = pi.sample(seed=k1)
        obs_next, env_state, _, done, _ = env_w.step(k2, env_state, action, env_params)
        return rng, env_state, obs_next, action, done

    num_iters = config["COLLECT_NUM_STEPS"] // num_envs
    all_obs, all_actions, all_dones = [], [], []

    for i in range(num_iters):
        rng, env_state, obs_next, action, done = _step(rng, env_state, obs)
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


def make_train_online(config: dict):
    """Return train(rng) for online fine-tuning with the diffusion model as policy.

    At each update step:
      1. Generate plans via sample_plan and execute REPLAN_EVERY actions per plan.
      2. Collect (obs, plan) pairs as training data (self-imitation).
      3. Fine-tune the model on those pairs for UPDATE_EPOCHS passes.

    Args:
        config: Configuration dict (all-uppercase keys).

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

        # Optionally warm-start from an offline checkpoint
        if config.get("OFFLINE_CHECKPOINT_PATH"):
            checkpointer = PyTreeCheckpointer()
            ckpt_mgr = CheckpointManager(
                config["OFFLINE_CHECKPOINT_PATH"],
                checkpointer,
                CheckpointManagerOptions(max_to_keep=1, create=True),
            )
            tmp_ts = TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=optax.adam(config["LR"]),
            )
            tmp_ts = ckpt_mgr.restore(ckpt_mgr.latest_step(), items=tmp_ts)  # type: ignore[assignment]
            params = tmp_ts.params  # type: ignore[union-attr]
            print(f"Loaded offline checkpoint (step={ckpt_mgr.latest_step()})")

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

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_fn = make_train_online(config)
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
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax"
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "offline", "online"],
        required=True,
        help="Training mode: collect offline data, train offline, or train online.",
    )

    # Environment
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")

    # Diffusion model
    parser.add_argument("--plan_horizon", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument(
        "--diffusion_schedule", type=str, choices=["cosine", "linear"], default="cosine"
    )
    parser.add_argument(
        "--remask_strategy",
        type=str,
        choices=list(STRATEGY_MAP.keys()),
        default="rescale",
    )
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--t_on", type=float, default=0.7)
    parser.add_argument("--t_off", type=float, default=0.3)

    # Transformer architecture
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--obs_encoder_layers", type=int, default=2)
    parser.add_argument("--obs_encoder_width", type=int, default=512)
    parser.add_argument("--dropout_rate", type=float, default=0.1)

    # Optimisation
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=256)

    # Offline training
    parser.add_argument("--offline_data_path", type=str, default="trajectories.npz")
    parser.add_argument(
        "--num_train_steps", type=lambda x: int(float(x)), default=100_000
    )

    # Online training
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=128)
    parser.add_argument(
        "--num_updates", type=lambda x: int(float(x)), default=1_000
    )
    parser.add_argument("--replan_every", type=int, default=8)
    parser.add_argument("--update_epochs", type=int, default=1)
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument(
        "--offline_checkpoint_path",
        type=str,
        default=None,
        help="Path to an offline-trained checkpoint to warm-start online training.",
    )

    # Data collection
    parser.add_argument("--ppo_checkpoint_path", type=str, default=None)
    parser.add_argument(
        "--collect_num_steps", type=lambda x: int(float(x)), default=1_000_000
    )
    parser.add_argument("--collect_num_envs", type=int, default=64)
    parser.add_argument("--layer_size", type=int, default=512)

    # W&B / logging
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--wandb_project", type=str, default="remdm-craftax")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)

    args, rest = parser.parse_known_args(sys.argv[1:])
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    config = {k.upper(): v for k, v in vars(args).items()}

    def _run():
        if config["MODE"] == "collect":
            assert config["PPO_CHECKPOINT_PATH"], "--ppo_checkpoint_path required"
            run_collect(config)
        elif config["MODE"] == "offline":
            run_offline(config)
        elif config["MODE"] == "online":
            run_online(config)

    if args.jit:
        _run()
    else:
        with jax.disable_jit():
            _run()
