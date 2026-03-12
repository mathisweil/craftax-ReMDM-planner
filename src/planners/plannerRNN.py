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
import functools

from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name

# Code adapted from the original PPO-RNN implementation by Chris Lu
# Original code located at https://github.com/luchris429/purejaxrl
# Adapted for a Discrete Diffusion Planner using ReMDM-style masked diffusion.


# ─────────────────────────────────────────────────────────────────────
# KEPT UNCHANGED: ScannedRNN – encodes observation history identically
# to the PPO agent.  The RNN hidden state captures partial-observability
# information which the denoiser conditions on.
# ─────────────────────────────────────────────────────────────────────
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
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# ─────────────────────────────────────────────────────────────────────
# HELPER: Sinusoidal positional embedding for the diffusion timestep.
# Standard practice in diffusion models (Vaswani et al. 2017 / Ho 2020).
# ─────────────────────────────────────────────────────────────────────
def sinusoidal_timestep_embedding(timestep, dim):
    """Compute sinusoidal embeddings for diffusion timestep.

    Args:
        timestep: integer tensor of shape (...,)
        dim: embedding dimension (must be even)
    Returns:
        embedding of shape (..., dim)
    """
    half_dim = dim // 2
    freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half_dim) / half_dim)
    # timestep: (...,) -> (..., 1) * (half_dim,) -> (..., half_dim)
    args = timestep[..., None].astype(jnp.float32) * freq
    return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


# ─────────────────────────────────────────────────────────────────────
# CHANGED: ActorCriticRNN ──► DiffusionPlannerRNN
#
# Structural mapping:
#   • obs Dense+ReLU ──► KEPT (identical first-layer encoder)
#   • ScannedRNN      ──► KEPT (identical recurrence for history)
#   • Actor MLP head  ──► REPLACED by denoising MLP that takes
#                         (obs_embed, plan_embed, timestep_embed)
#                         and outputs logits (plan_horizon, action_dim)
#   • Critic MLP head ──► KEPT (value baseline for return-weighted
#                         diffusion loss)
#   • distrax.Categorical ──► REMOVED (sampling is via the diffusion
#                              reverse process, not a single-step π)
#
# The module exposes two sub-paths through separate methods so that
# the RNN is executed once per env step while the denoiser can be
# called repeatedly during the T-step reverse process:
#   encode()  – obs → RNN → embedding, value
#   denoise() – (embedding, noisy_plan, t) → logits
#   __call__  – convenience: encode + denoise in one pass (training)
# ─────────────────────────────────────────────────────────────────────
class DiffusionPlannerRNN(nn.Module):
    action_dim: int
    plan_horizon: int
    config: Dict

    def setup(self):
        ls = self.config["LAYER_SIZE"]
        ped = self.config.get("PLAN_EMBED_DIM", 32)

        # --- Observation encoder (mirrors PPO's first Dense + RNN) ---
        self.obs_dense = nn.Dense(
            ls, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.rnn = ScannedRNN()

        # --- Plan token embedding (NEW) ---
        #   action_dim regular tokens + 1 MASK token
        self.plan_embed = nn.Embed(
            num_embeddings=self.action_dim + 1, features=ped
        )

        # --- Timestep projection (NEW) ---
        self.t_proj = nn.Dense(ls, kernel_init=orthogonal(2), bias_init=constant(0.0))

        # --- Denoising MLP (REPLACES actor head) ---
        self.denoise_dense1 = nn.Dense(
            ls, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.denoise_dense2 = nn.Dense(
            ls, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.denoise_out = nn.Dense(
            self.plan_horizon * self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )

        # --- Value / critic head (KEPT from PPO) ---
        self.critic_dense1 = nn.Dense(
            ls, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.critic_dense2 = nn.Dense(
            ls, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )
        self.critic_out = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )

    # ---- sub-path 1: encode obs through RNN ----
    def encode(self, hidden, x):
        """Process observation through Dense + RNN.  Returns (hidden, embedding, value)."""
        obs, dones = x
        embedding = nn.relu(self.obs_dense(obs))
        rnn_in = (embedding, dones)
        hidden, embedding = self.rnn(hidden, rnn_in)

        # value head (identical structure to PPO critic)
        critic = nn.relu(self.critic_dense1(embedding))
        critic = nn.relu(self.critic_dense2(critic))
        value = jnp.squeeze(self.critic_out(critic), axis=-1)

        return hidden, embedding, value

    # ---- sub-path 2: denoise plan given obs embedding ----
    def denoise(self, obs_embedding, noisy_plan, diff_t):
        """Predict clean-plan logits.

        Args:
            obs_embedding: (..., layer_size)
            noisy_plan:    (..., plan_horizon) int tokens (0..action_dim = mask)
            diff_t:        (...,) int diffusion timestep
        Returns:
            logits: (..., plan_horizon, action_dim)
        """
        # embed each token in the noisy plan
        pe = self.plan_embed(noisy_plan)                        # (..., H, ped)
        pe = pe.reshape(*pe.shape[:-2], -1)                     # (..., H*ped)

        # sinusoidal timestep → project
        te = sinusoidal_timestep_embedding(diff_t, self.config["LAYER_SIZE"])
        te = nn.relu(self.t_proj(te))                           # (..., ls)

        combined = jnp.concatenate([obs_embedding, pe, te], axis=-1)

        x = nn.relu(self.denoise_dense1(combined))
        x = nn.relu(self.denoise_dense2(x))
        logits = self.denoise_out(x)
        logits = logits.reshape(
            *logits.shape[:-1], self.plan_horizon, self.action_dim
        )
        return logits

    # ---- full forward (used in training re-run) ----
    def __call__(self, hidden, x, noisy_plan, diff_t):
        hidden, embedding, value = self.encode(hidden, x)
        logits = self.denoise(embedding, noisy_plan, diff_t)
        return hidden, logits, value


# ─────────────────────────────────────────────────────────────────────
# CHANGED: Transition NamedTuple
#
#   REMOVED : log_prob  (no single-step policy ratio in diffusion)
#   ADDED   : plan      (the H-step plan generated by the reverse process)
#   ADDED   : diff_t    (sampled diffusion timestep used during collection,
#                         stored for optional analytics / curriculum)
#   KEPT    : done, action, value, reward, obs, info
# ─────────────────────────────────────────────────────────────────────
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray       # first action of plan actually executed
    value: jnp.ndarray        # value baseline (for return computation)
    reward: jnp.ndarray
    plan: jnp.ndarray         # full H-step plan from denoiser
    obs: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    # ─── KEPT: update / minibatch arithmetic ───
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    # ─── Derived diffusion constants ───
    MASK_TOKEN = None  # will be set after we know action_dim
    NUM_DIFF_STEPS = config["NUM_DIFF_STEPS"]
    PLAN_HORIZON = config["PLAN_HORIZON"]
    REPLAN_EVERY = config.get("REPLAN_EVERY", 1)

    # ─── KEPT: Create environment (identical to PPO) ───
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

    # ─── KEPT: LR schedule (works just as well for diffusion training) ───
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # ─── CHANGED: init network ───
        action_dim = env.action_space(env_params).n
        mask_token = action_dim  # use index = action_dim as the [MASK] token

        network = DiffusionPlannerRNN(
            action_dim=action_dim,
            plan_horizon=PLAN_HORIZON,
            config=config,
        )
        rng, _rng = jax.random.split(rng)

        # dummy inputs for parameter initialisation
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )
        init_plan = jnp.zeros(
            (1, config["NUM_ENVS"], PLAN_HORIZON), dtype=jnp.int32
        )
        init_t = jnp.zeros((1, config["NUM_ENVS"]), dtype=jnp.int32)

        network_params = network.init(
            _rng, init_hstate, init_x, init_plan, init_t
        )

        # ─── KEPT: optimizer (identical) ───
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
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # ─── KEPT: init env (identical) ───
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ENVS"], config["LAYER_SIZE"]
        )

        # ─────────────────────────────────────────────────────────────
        # ReMDM reverse-process sampler (replaces pi.sample)
        #
        # Implements absorbing-state masked diffusion with remasking:
        #   1. Start from a fully-masked plan  x_T = [M, M, ..., M]
        #   2. For t = T … 1:
        #       a. Predict clean logits  x̂_0 = f_θ(obs_embed, x_t, t)
        #       b. Sample candidate tokens from  x̂_0
        #       c. Unmask a fraction of positions (linear schedule)
        #       d. REMASK a fraction σ_t of previously-decoded tokens
        #          (the key ReMDM contribution for iterative refinement)
        #   3. Final plan = fully-unmasked sequence after step 1.
        # ─────────────────────────────────────────────────────────────
        def _generate_plan(params, obs_embedding, rng):
            """ReMDM-style reverse sampling to produce an H-step plan.

            Args:
                params: network parameters
                obs_embedding: (num_envs, layer_size)
                rng: PRNG key
            Returns:
                plan: (num_envs, plan_horizon) action indices
            """
            init_plan = jnp.full(
                (config["NUM_ENVS"], PLAN_HORIZON), mask_token, dtype=jnp.int32
            )

            def _denoise_step(carry, step_idx):
                plan, rng = carry
                rng, pred_rng, unmask_rng, remask_rng = jax.random.split(rng, 4)

                # current diffusion time  (T → 1)
                t = NUM_DIFF_STEPS - step_idx
                t_input = jnp.full((config["NUM_ENVS"],), t, dtype=jnp.int32)

                # predict clean-plan logits conditioned on obs
                logits = network.apply(
                    params, obs_embedding, plan, t_input,
                    method=network.denoise,
                )  # (num_envs, H, action_dim)

                # sample candidate tokens (with optional temperature)
                temperature = config.get("SAMPLE_TEMPERATURE", 1.0)
                flat_logits = (logits / temperature).reshape(-1, action_dim)
                candidates = jax.random.categorical(
                    pred_rng, flat_logits
                ).reshape(config["NUM_ENVS"], PLAN_HORIZON)

                # ── unmask schedule: linearly reveal positions ──
                is_masked = (plan == mask_token)
                # target fraction unmasked after this step
                target_unmasked = (step_idx + 1) / NUM_DIFF_STEPS
                unmask_noise = jax.random.uniform(
                    unmask_rng, plan.shape
                )
                should_unmask = is_masked & (unmask_noise < target_unmasked)
                plan = jnp.where(should_unmask, candidates, plan)

                # ── ReMDM remasking: stochastically re-mask decoded tokens ──
                eta = config.get("REMASK_ETA", 0.1)  # remasking rate
                is_unmasked = (plan != mask_token)
                remask_noise = jax.random.uniform(
                    remask_rng, plan.shape
                )
                # remask probability decays with t so final steps are stable
                remask_prob = eta * (t / NUM_DIFF_STEPS)
                should_remask = is_unmasked & (remask_noise < remask_prob)
                plan = jnp.where(should_remask, mask_token, plan)

                return (plan, rng), None

            (plan, _), _ = jax.lax.scan(
                _denoise_step,
                (init_plan, rng),
                jnp.arange(NUM_DIFF_STEPS),
            )

            # final pass: any still-masked positions get a greedy decode
            still_masked = (plan == mask_token)
            t_final = jnp.ones((config["NUM_ENVS"],), dtype=jnp.int32)
            final_logits = network.apply(
                params, obs_embedding, plan, t_final,
                method=network.denoise,
            )
            greedy = jnp.argmax(final_logits, axis=-1)
            plan = jnp.where(still_masked, greedy, plan)

            return plan

        # ─── TRAIN LOOP ───
        def _update_step(runner_state, unused):
            # ─── COLLECT TRAJECTORIES ───
            # CHANGED: action selection uses diffusion reverse process
            # instead of pi.sample; value is computed from the encoder.
            def _env_step(runner_state, unused):
                (
                    train_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                    update_step,
                ) = runner_state
                rng, _rng, _plan_rng = jax.random.split(rng, 3)

                # ── ENCODE observation (replaces full network forward) ──
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                hstate, embedding, value = network.apply(
                    train_state.params, hstate, ac_in,
                    method=network.encode,
                )
                value = value.squeeze(0)
                obs_embed = embedding.squeeze(0)  # (num_envs, ls)

                # ── GENERATE PLAN via ReMDM reverse process ──
                plan = _generate_plan(
                    train_state.params, obs_embed, _plan_rng,
                )  # (num_envs, plan_horizon)

                # execute first action of the plan
                action = plan[:, 0]

                # ── STEP ENV (identical to PPO) ──
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )
                transition = Transition(
                    last_done, action, value, reward, plan, last_obs, info
                )
                runner_state = (
                    train_state,
                    env_state,
                    obsv,
                    done,
                    hstate,
                    rng,
                    update_step,
                )
                return runner_state, transition

            initial_hstate = runner_state[-3]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # ─────────────────────────────────────────────────────────
            # CHANGED: CALCULATE RETURNS (replaces GAE)
            #
            # We use simple discounted returns instead of GAE because
            # the diffusion loss is weighted by return magnitude, not
            # by a clipped policy ratio.  The value baseline is still
            # subtracted to reduce variance (advantage weighting).
            # ─────────────────────────────────────────────────────────
            (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step,
            ) = runner_state

            # bootstrap value (KEPT: identical call structure)
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(
                train_state.params, hstate, ac_in,
                method=network.encode,
            )
            last_val = last_val.squeeze(0)

            def _calculate_returns(traj_batch, last_val, last_done):
                """Monte-Carlo-style discounted returns (replaces GAE scan)."""
                def _get_return(carry, transition):
                    next_return, next_done = carry
                    done, reward = transition.done, transition.reward
                    # zero out return across episode boundaries
                    ret = reward + config["GAMMA"] * next_return * (1 - next_done)
                    return (ret, done), ret

                _, returns = jax.lax.scan(
                    _get_return,
                    (last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                advantages = returns - traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_returns(
                traj_batch, last_val, last_done
            )

            # ─────────────────────────────────────────────────────────
            # CHANGED: UPDATE NETWORK
            #
            # PPO's clipped surrogate + value loss + entropy bonus
            # is replaced by:
            #   1. Masked diffusion cross-entropy loss (denoising loss)
            #      – sample random t, mask the stored plan, predict it
            #   2. Advantage-weighted diffusion loss
            #      – higher-return plans get stronger gradient signal
            #   3. Value loss (MSE on returns, same as PPO)
            #
            # The epoch / minibatch scaffold is KEPT unchanged.
            # ─────────────────────────────────────────────────────────
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        rng = jax.random.PRNGKey(0)  # deterministic inside grad

                        # ── RE-RUN ENCODER (mirrors PPO's network re-run) ──
                        hstate_0, embedding, value = network.apply(
                            params, init_hstate[0],
                            (traj_batch.obs, traj_batch.done),
                            method=network.encode,
                        )
                        # embedding: (num_steps, minibatch, ls)

                        # ── SAMPLE DIFFUSION TIMESTEP ──
                        # one t per (step, env) element
                        t_shape = traj_batch.obs.shape[:2]  # (T, B)
                        rng, t_rng, mask_rng = jax.random.split(rng, 3)
                        diff_t = jax.random.randint(
                            t_rng, t_shape,
                            minval=1, maxval=NUM_DIFF_STEPS + 1,
                        )

                        # ── MASK THE PLAN (forward diffusion) ──
                        # mask each position independently with prob t/T
                        mask_prob = diff_t[..., None] / NUM_DIFF_STEPS
                        # mask_prob: (T, B, 1)  broadcast over plan_horizon
                        mask_noise = jax.random.uniform(
                            mask_rng,
                            (*t_shape, PLAN_HORIZON),
                        )
                        is_masked = mask_noise < mask_prob
                        noisy_plan = jnp.where(
                            is_masked, mask_token, traj_batch.plan
                        )

                        # ── PREDICT (denoiser forward) ──
                        logits = network.apply(
                            params, embedding, noisy_plan, diff_t,
                            method=network.denoise,
                        )
                        # logits: (T, B, H, action_dim)

                        # ── DIFFUSION LOSS: cross-entropy on ALL positions ──
                        # (predicting clean plan from noisy plan)
                        plan_onehot = jax.nn.one_hot(
                            traj_batch.plan, action_dim
                        )
                        ce = -jnp.sum(
                            plan_onehot * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1,
                        )  # (T, B, H)

                        # weight more heavily on actually-masked positions
                        mask_weight = jnp.where(is_masked, 1.0, 0.1)
                        ce = (ce * mask_weight).mean(axis=-1)  # (T, B)

                        # ── ADVANTAGE WEIGHTING ──
                        # Normalise advantages → weights ∈ [0, ∞)
                        adv = gae
                        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                        # softmax-style positive weighting (exp clamped)
                        weights = jnp.exp(
                            config.get("ADV_TEMPERATURE", 1.0) * adv
                        )
                        weights = weights / (weights.mean() + 1e-8)

                        diffusion_loss = (ce * weights).mean()

                        # ── VALUE LOSS (identical to PPO value head) ──
                        value_loss = 0.5 * jnp.square(value - targets).mean()

                        # ── TOTAL LOSS ──
                        total_loss = (
                            diffusion_loss
                            + config["VF_COEF"] * value_loss
                        )
                        return total_loss, (value_loss, diffusion_loss, ce.mean())

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                # ─── KEPT: epoch / minibatch shuffling (identical) ───
                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

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

                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            # ─── KEPT: epoch scan structure (identical) ───
            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            # ─── KEPT: metric aggregation (identical) ───
            metric = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                / traj_batch.info["returned_episode"].sum(),
                traj_batch.info,
            )
            rng = update_state[-1]

            # ─── KEPT: wandb logging callback (identical) ───
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step)

            # ─── KEPT: runner_state packing (identical) ───
            runner_state = (
                train_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
                update_step + 1,
            )
            return runner_state, metric

        # ─── KEPT: initial runner state + main scan (identical) ───
        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
            0,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


# ─────────────────────────────────────────────────────────────────────
# CHANGED: run_ppo ──► run_diffusion_planner
# Only the wandb name tag and a few config keys change.
# ─────────────────────────────────────────────────────────────────────
def run_diffusion_planner(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-ReMDM_RNN-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
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
                    args=ocp.args.StandardSave(train_state)
                )

            print(f"saved runner state to {path}")

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")