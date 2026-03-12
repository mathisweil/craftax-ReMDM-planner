import argparse
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time

import orbax.checkpoint as ocp

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name

# Offline Discrete Diffusion Planner trained on PPO-collected trajectories.
# Follows the exact make_train / train / _update_step / _env_step / _update_epoch
# / _update_minbatch scaffold from ppo_rnn.py, replacing only the parts that
# differ: (1) data collection uses a frozen PPO agent, (2) the model being trained
# is a denoiser, (3) the loss is masked-diffusion cross-entropy instead of
# clipped surrogate + value + entropy.

# ─────────────────────────────────────────────────────────────────────────────
# PPO agent (frozen, for data collection only)
# Identical to ppo_rnn.py – we need these classes to load the checkpoint.
# ─────────────────────────────────────────────────────────────────────────────


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorCriticRNN(nn.Module):
    action_dim: int
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.config["LAYER_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion denoiser (the model being trained)
# Takes (obs, noisy_plan, diffusion_t) → logits over clean plan.
# ─────────────────────────────────────────────────────────────────────────────

def sinusoidal_timestep_embedding(timestep, dim):
    half_dim = dim // 2
    freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
    args = timestep[..., None].astype(jnp.float32) * freq
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class DiffusionDenoiser(nn.Module):
    action_dim: int
    plan_horizon: int
    config: Dict

    @nn.compact
    def __call__(self, obs, noisy_plan, diff_t):
        ls = self.config["LAYER_SIZE"]
        ped = self.config.get("PLAN_EMBED_DIM", 32)

        # Observation encoder
        obs_embed = nn.Dense(
            ls, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        obs_embed = nn.relu(obs_embed)

        # Plan token embedding (action_dim regular + 1 MASK token)
        pe = nn.Embed(num_embeddings=self.action_dim + 1, features=ped)(noisy_plan)
        pe = pe.reshape(*pe.shape[:-2], -1)

        # Timestep embedding
        te = sinusoidal_timestep_embedding(diff_t, ls)
        te = nn.relu(
            nn.Dense(ls, kernel_init=orthogonal(2), bias_init=constant(0.0))(te)
        )

        # Denoiser MLP (mirrors the 2-hidden-layer actor head from PPO)
        x = jnp.concatenate([obs_embed, pe, te], axis=-1)
        x = nn.Dense(
            ls, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            ls, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        logits = nn.Dense(
            self.plan_horizon * self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        logits = logits.reshape(
            *logits.shape[:-1], self.plan_horizon, self.action_dim
        )
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Masking schedules for forward diffusion
# ─────────────────────────────────────────────────────────────────────────────

def _linear_schedule(t, T):
    """Mask probability = t / T."""
    return t / T


def _cosine_schedule(t, T):
    """Cosine mask schedule (Nichol & Dhariwal style)."""
    return 1.0 - jnp.cos(0.5 * jnp.pi * t / T)


SCHEDULE_MAP = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
}


# ─────────────────────────────────────────────────────────────────────────────
# PPO checkpoint loader
# ─────────────────────────────────────────────────────────────────────────────

def _load_ppo_params(checkpoint_path, ppo_network, num_envs, obs_shape, layer_size):
    """Restore PPO parameters from an orbax checkpoint."""
    init_x = (
        jnp.zeros((1, num_envs, *obs_shape)),
        jnp.zeros((1, num_envs)),
    )
    init_hstate = ScannedRNN.initialize_carry(num_envs, layer_size)
    abstract_params = ppo_network.init(jax.random.PRNGKey(0), init_hstate, init_x)

    dummy_tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(1e-4)
    )

    abstract_state = TrainState.create(
        apply_fn=ppo_network.apply,
        params=abstract_params,
        tx=dummy_tx,
    )

    with ocp.CheckpointManager(checkpoint_path) as mgr:
        latest_step = mgr.latest_step()
        restored = mgr.restore(
            latest_step,
            args=ocp.args.PyTreeRestore(item=abstract_state, partial_restore=True)
        )

    print(f"Loaded PPO checkpoint from '{checkpoint_path}' (step {latest_step})")
    return restored.params


# ─────────────────────────────────────────────────────────────────────────────
# Transition (replaces PPO's version)
#   REMOVED: value, log_prob  (PPO-specific, not needed for offline diffusion)
#   KEPT:    done, action, reward, obs, info
# ─────────────────────────────────────────────────────────────────────────────

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


# ─────────────────────────────────────────────────────────────────────────────
# make_train  (mirrors ppo_rnn.py line-for-line)
# ─────────────────────────────────────────────────────────────────────────────

def make_train(config, ppo_checkpoint_path):
    # ── CONFIG ARITHMETIC (identical to PPO) ──
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # ── CREATE ENVIRONMENT (identical to PPO) ──
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    env = LogWrapper(env)

    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env,
            num_envs=config["NUM_ENVS"],
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    plan_horizon = config["PLAN_HORIZON"]
    num_diff_steps = config["NUM_DIFF_STEPS"]
    mask_token = num_actions  # MASK = action_dim (one past the valid range)
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    assert config["NUM_STEPS"] >= plan_horizon, (
        f"NUM_STEPS ({config['NUM_STEPS']}) must be >= PLAN_HORIZON ({plan_horizon})"
    )

    # ── LOAD FROZEN PPO AGENT (replaces PPO's "own network") ──
    ppo_config = {"LAYER_SIZE": config["LAYER_SIZE"]}
    ppo_network = ActorCriticRNN(num_actions, config=ppo_config)
    ppo_params = _load_ppo_params(
        ppo_checkpoint_path, ppo_network,
        config["NUM_ENVS"], obs_shape, config["LAYER_SIZE"],
    )

    # ── LR SCHEDULE (identical to PPO) ──
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # ────────────────────────────────────────────────────────────
        # INIT NETWORK  (replaces PPO's ActorCriticRNN init)
        # ────────────────────────────────────────────────────────────
        diffusion_net = DiffusionDenoiser(
            action_dim=num_actions,
            plan_horizon=plan_horizon,
            config=config,
        )
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((1, *obs_shape))
        init_plan = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
        init_t = jnp.zeros((1,), dtype=jnp.int32)
        network_params = diffusion_net.init(_rng, init_obs, init_plan, init_t)

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=diffusion_net.apply,
            params=network_params,
            tx=tx,
        )

        # ────────────────────────────────────────────────────────────
        # INIT ENV  (identical to PPO)
        # ────────────────────────────────────────────────────────────
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        ppo_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        # ────────────────────────────────────────────────────────────
        # TRAIN LOOP
        # ────────────────────────────────────────────────────────────
        def _update_step(runner_state, unused):
            # ── COLLECT TRAJECTORIES (using frozen PPO agent) ──
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    ppo_hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng = jax.random.split(rng)

                # SELECT ACTION  (frozen PPO → replaces PPO self-play)
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                ppo_hstate, pi, _ = ppo_network.apply(
                    ppo_params, ppo_hstate, ac_in
                )
                action = pi.sample(seed=_rng)
                action = action.squeeze(0)

                # STEP ENV  (identical to PPO)
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = Transition(
                    last_done, action, reward, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    ppo_hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # ── UNPACK (identical to PPO) ──
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                ppo_hstate,
                rng,
                update_step,
            ) = runner_state

            # ────────────────────────────────────────────────────────
            # UPDATE NETWORK
            #
            # PPO structure:  _update_epoch → _update_minbatch → _loss_fn
            # Diffusion:      same scaffold, different loss.
            #
            # Difference from PPO:  _update_minbatch carry includes rng
            # because the diffusion loss is stochastic (random t, random
            # masking).  PPO's loss is deterministic given the batch.
            # ────────────────────────────────────────────────────────
            def _update_epoch(update_state, unused):
                def _update_minbatch(carry, batch_info):
                    train_state, rng = carry
                    traj_batch = batch_info
                    rng, loss_rng = jax.random.split(rng)

                    def _loss_fn(params, traj_batch, rng):
                        T, B = traj_batch.obs.shape[:2]
                        max_start = T - plan_horizon

                        rng, start_rng, diff_rng, mask_rng = jax.random.split(rng, 4)

                        # ── SAMPLE PLAN WINDOWS ──
                        # One random start time per env in the minibatch
                        start_times = jax.random.randint(
                            start_rng, (B,), 0, max_start + 1
                        )

                        # Extract obs at the start of each window
                        obs_batch = traj_batch.obs[
                            start_times, jnp.arange(B)
                        ]  # (B, *obs_shape)

                        # Extract plan = action[start : start+H] per env
                        offsets = jnp.arange(plan_horizon)  # (H,)
                        act_idx = start_times[:, None] + offsets[None, :]  # (B, H)
                        plan_batch = traj_batch.action[
                            act_idx, jnp.arange(B)[:, None]
                        ]  # (B, H)

                        # ── VALIDITY MASK (no episode boundary in window) ──
                        # traj_batch.done[t, e] = True means a reset happened
                        # before step t.  A window is invalid if any reset
                        # occurs at steps start+1 … start+H-1.
                        if plan_horizon > 1:
                            done_offsets = jnp.arange(1, plan_horizon)
                            done_idx = start_times[:, None] + done_offsets[None, :]
                            done_in_window = traj_batch.done[
                                done_idx, jnp.arange(B)[:, None]
                            ]
                            valid = ~jnp.any(done_in_window, axis=-1)  # (B,)
                        else:
                            valid = jnp.ones(B, dtype=bool)

                        # ── FORWARD DIFFUSION: mask the plan ──
                        diff_t = jax.random.randint(
                            diff_rng, (B,), 1, num_diff_steps + 1
                        )
                        mask_prob = schedule_fn(diff_t, num_diff_steps)  # (B,)
                        mask_noise = jax.random.uniform(
                            mask_rng, (B, plan_horizon)
                        )
                        noisy_plan = jnp.where(
                            mask_noise < mask_prob[:, None],
                            mask_token,
                            plan_batch,
                        )

                        # ── PREDICT CLEAN PLAN ──
                        logits = diffusion_net.apply(
                            params, obs_batch, noisy_plan, diff_t
                        )  # (B, H, num_actions)

                        # ── CROSS-ENTROPY LOSS ──
                        plan_onehot = jax.nn.one_hot(plan_batch, num_actions)
                        ce = -jnp.sum(
                            plan_onehot
                            * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1,
                        )  # (B, H)
                        ce = ce.mean(axis=-1)  # (B,)

                        # Zero out invalid windows
                        loss = (
                            jnp.where(valid, ce, 0.0).sum()
                            / (valid.sum() + 1e-8)
                        )

                        return loss, (loss, valid.mean())

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, loss_rng
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return (train_state, rng), total_loss

                # ── SHUFFLE + MINIBATCH (identical to PPO) ──
                (
                    train_state,
                    traj_batch,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = traj_batch

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                rng, _mb_rng = jax.random.split(rng)
                (train_state, _), total_loss = jax.lax.scan(
                    _update_minbatch, (train_state, _mb_rng), minibatches
                )
                update_state = (
                    train_state,
                    traj_batch,
                    rng,
                )
                return update_state, total_loss

            # ── EPOCH SCAN (identical to PPO) ──
            update_state = (
                train_state,
                traj_batch,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            # ── METRIC AGGREGATION (identical to PPO) ──
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, loss_info, update_step):
                    to_log = create_log_dict(metric, config)
                    # Append diffusion loss metrics
                    last_epoch_loss = jax.tree.map(lambda x: x[-1], loss_info)
                    total, (denoising_loss, valid_frac) = last_epoch_loss
                    to_log["diffusion/denoising_loss"] = float(denoising_loss[-1])
                    to_log["diffusion/valid_window_frac"] = float(valid_frac[-1])
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, loss_info, update_step)

            # ── REPACK RUNNER STATE (identical to PPO) ──
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                ppo_hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        # ── INITIAL RUNNER STATE + MAIN SCAN (identical to PPO) ──
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            ppo_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


# ─────────────────────────────────────────────────────────────────────────────
# run_offline_diffusion  (mirrors run_ppo)
# ─────────────────────────────────────────────────────────────────────────────

def run_offline_diffusion(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}
    ppo_checkpoint_path = config["PPO_CHECKPOINT"]

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-OfflineDiffusion-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config, ppo_checkpoint_path))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)

            path = os.path.join(wandb.run.dir, dir_name)
            options = ocp.CheckpointManagerOptions(max_to_keep=1)

            with ocp.CheckpointManager(path, options=options) as checkpoint_manager:
                checkpoint_manager.save(
                    config["TOTAL_TIMESTEPS"],
                    args=ocp.args.StandardSave(train_state),
                )

            print(f"saved runner state to {path}")

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


# ─────────────────────────────────────────────────────────────────────────────
# __main__  (mirrors ppo_rnn.py, replacing PPO-only flags with diffusion ones)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # ── Diffusion-specific ──
    parser.add_argument("--plan_horizon", type=int, default=8)
    parser.add_argument("--num_diff_steps", type=int, default=16)
    parser.add_argument("--plan_embed_dim", type=int, default=32)
    parser.add_argument(
        "--diffusion_schedule", type=str, default="cosine",
        choices=list(SCHEDULE_MAP.keys()),
    )
    # ── PPO checkpoint for data collection ──
    parser.add_argument("--ppo_checkpoint", type=str, required=True)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_offline_diffusion(args)
    else:
        with jax.disable_jit():
            run_offline_diffusion(args)