import os
import time
from typing import Any, NamedTuple

import wandb
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name

from src.models.remdm import sample_plan
from .common import SCHEDULE_MAP, _make_grad_step
from .utils import (
    _init_model_params,
    _create_train_state,
    _make_apply_fns,
)
from src.models.denoiser import DenoisingTransformer
from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.logz.batch_logging import create_log_dict, batch_log
from Craftax_Baselines.ppo_rnn import ActorCriticRNN
from Craftax_Baselines.ppo import ActorCritic
from Craftax_Baselines.ppo_rnd import ActorCriticRND

def _load_ppo_params(checkpoint_path, ppo_network, model_type, num_envs, obs_shape, layer_size=512):
    """Restore PPO parameters from an Orbax checkpoint across multiple architectures."""
    rng = jax.random.PRNGKey(0)
    model_type = model_type.lower()

    if model_type == "ppo_rnn":
        init_x = (
            jnp.zeros((1, num_envs, *obs_shape)),
            jnp.zeros((1, num_envs)),
        )
        init_hstate = jnp.zeros((num_envs, layer_size))
        abstract_params = ppo_network.init(rng, init_hstate, init_x)
    else:
        init_x = jnp.zeros((1, *obs_shape))
        abstract_params = ppo_network.init(rng, init_x)

    with ocp.CheckpointManager(checkpoint_path) as mgr:
        latest_step = mgr.latest_step()
        if latest_step is None:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        restored = mgr.restore(
            latest_step,
            args=ocp.args.PyTreeRestore(
                item={"params": abstract_params},
                partial_restore=True
            )
        )

    print(f"Loaded {model_type.upper()} checkpoint from '{checkpoint_path}' (step {latest_step})")
    return restored["params"]


class PPOAgentAdapter:
    def __init__(self, network, params, model_type: str, layer_size: int = 512):
        self.network = network
        self.params = params
        self.model_type = model_type.lower()
        self.layer_size = layer_size

    def init_hidden(self, batch_size: int):
        """Returns the initial hidden state for RNNs, or None for feedforward networks."""
        if self.model_type == "ppo_rnn":
            return jnp.zeros((batch_size, self.layer_size))
        return None

    def get_action_and_hidden(self, obs, done, hidden, rng, temperature=1.0):
        """Uniform forward pass across all architectures."""

        if self.model_type == "ppo_rnn":
            ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
            new_hidden, pi, _ = self.network.apply(self.params, hidden, ac_in)
        elif self.model_type == "ppo_rnd":
            pi, _value_e, _value_i = self.network.apply(self.params, obs)
            new_hidden = hidden
        else:
            pi, _value = self.network.apply(self.params, obs)
            new_hidden = hidden

        noisy_logits = pi.logits / temperature
        action = jax.random.categorical(rng, noisy_logits)

        if self.model_type == "ppo_rnn":
            action = action.squeeze(0)

        return action, new_hidden


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: dict


def make_train(config: dict[str, Any]):
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    valid_steps_per_rollout = config["NUM_STEPS"] - config["PLAN_HORIZON"] + 1
    num_samples = config["NUM_ENVS"] * valid_steps_per_rollout
    assert num_samples % config["NUM_MINIBATCHES"] == 0, (
        f"NUM_ENVS * valid_steps ({num_samples}) must be divisible by "
        f"NUM_MINIBATCHES ({config['NUM_MINIBATCHES']})"
    )
    config["MINIBATCH_SIZE"] = num_samples // config["NUM_MINIBATCHES"]

    # Create environment
    # Create environment
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"]
    )
    env_params = env.default_params

    # Wrap with some extra logging
    env = LogWrapper(env)

    # Wrap with a batcher, maybe using optimistic resets
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
    obs_dim = obs_shape[0]

    model_type = config["PPO_MODEL_TYPE"]
    if model_type == "ppo_rnn":
        network = ActorCriticRNN(env.action_space(env_params).n, config=config)
    elif model_type == "ppo_rnd":
        network = ActorCriticRND(num_actions, config["LAYER_SIZE"])
    else:
        network = ActorCritic(num_actions, config["LAYER_SIZE"])

    network_params = _load_ppo_params(config["PPO_CHECKPOINT_PATH"], network, model_type, config["NUM_ENVS"], obs_shape,
                                      config["LAYER_SIZE"])

    ppo_agent = PPOAgentAdapter(network, network_params, model_type, config["LAYER_SIZE"])

    def train(rng: jax.Array) -> dict[str, Any]:
        # INIT NETWORK & ENV
        diffusion_net = DenoisingTransformer(
            num_actions=env.action_space(env_params).n,
            plan_horizon=config["PLAN_HORIZON"],
            d_model=config["D_MODEL"],
            n_heads=config["N_HEADS"],
            n_layers=config["N_LAYERS"],
            d_ff=config["D_FF"],
            obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
            obs_encoder_width=config["OBS_ENCODER_WIDTH"],
            dropout_rate=config["DROPOUT_RATE"],
        )
        apply_eval, apply_train = _make_apply_fns(diffusion_net)
        schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
        grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

        rng, init_rng, env_rng = jax.random.split(rng, 3)
        params = _init_model_params(diffusion_net, init_rng, obs_dim, config["PLAN_HORIZON"])
        train_state = _create_train_state(diffusion_net, params, config["LR"], config["MAX_GRAD_NORM"])

        obsv, env_state = env.reset(env_rng, env_params)
        init_hstate = ppo_agent.init_hidden(config["NUM_ENVS"])

        def _run_validation(train_state, rng):
            # Setup a small number of dedicated eval envs
            rng, val_rng, init_rng = jax.random.split(rng, 3)

            # Reset eval envs
            val_obs, val_env_state = env.reset(val_rng, env_params)

            def _val_step(carry, _):
                val_env_state, val_obs, rng = carry
                rng, plan_rng, step_rng = jax.random.split(rng, 3)

                # Use your existing sample_plan logic from remdm.py
                # This generates a full sequence of actions [B, H]
                plan = sample_plan(
                    apply_eval, train_state.params, plan_rng, val_obs,
                    num_actions, config["PLAN_HORIZON"],
                    num_steps=config.get("VAL_DIFFUSION_STEPS", 50),
                    schedule_fn=schedule_fn,
                    remask_strategy="cap",  # Recommended for planning
                    use_loop=True  # Enable mistake-correction
                )

                # Execute the FIRST action of the plan (Receding Horizon Control)
                action = plan[:, 0]
                val_obs, val_env_state, _, _, info = env.step(
                    step_rng, val_env_state, action, env_params
                )
                return (val_env_state, val_obs, rng), info

            # Run for a fixed number of steps (e.g., a full episode length)
            _, val_infos = jax.lax.scan(_val_step, (val_env_state, val_obs, rng), None, 128)

            # Calculate mean achievements across validation episodes
            val_metrics = jax.tree.map(
                lambda x: (x * val_infos["returned_episode"]).sum()
                          / (val_infos["returned_episode"].sum() + 1e-8),
                val_infos
            )
            return {f"val/{k}": v for k, v in val_metrics.items()}

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
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

                rng, _rng, k1 = jax.random.split(rng, 3)

                # SELECT ACTION (Handled dynamically by the Adapter)
                action, new_hstate = ppo_agent.get_action_and_hidden(
                    obs=last_obs,
                    done=last_done,
                    hidden=hstate,
                    rng=k1,
                    temperature=config.get("COLLECT_TEMPERATURE", 2.0)
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = env.step(
                    _rng, env_state, action, env_params
                )

                # PAD TRANSITION
                transition = Transition(
                    done=last_done,
                    action=action,
                    reward=reward,
                    obs=last_obs,
                    info=info
                )

                runner_state = (
                    train_state, env_state, obsv, done, new_hstate, rng, update_step
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # PREPARE DIFFUSION WINDOWS
            train_state, env_state, last_obs, last_done, hstate, rng, update_step = runner_state
            plan_horizon = config["PLAN_HORIZON"]

            def get_window(t):
                obs_t = traj_batch.obs[t]
                act_window = jax.lax.dynamic_slice(traj_batch.action, (t, 0), (plan_horizon, config["NUM_ENVS"]))
                done_window = jax.lax.dynamic_slice(traj_batch.done, (t, 0), (plan_horizon, config["NUM_ENVS"]))
                
                valid = ~jnp.any(done_window, axis=0)
                return obs_t, jnp.swapaxes(act_window, 0, 1), valid

            obs_windows, act_windows, valid_masks = jax.vmap(get_window)(jnp.arange(valid_steps_per_rollout))
            
            flat_obs = obs_windows.reshape(-1, obs_dim)
            flat_acts = act_windows.reshape(-1, plan_horizon)
            flat_valid = valid_masks.reshape(-1)
            dataset = (flat_obs, flat_acts, flat_valid)
            num_samples = valid_steps_per_rollout * config["NUM_ENVS"]

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state_and_rng, batch_info):
                    train_state, rng = train_state_and_rng
                    rng, loss_rng = jax.random.split(rng)
                    obs_batch, act_batch, valid_batch = batch_info

                    train_state, metrics = grad_step(train_state, act_batch, obs_batch, valid_batch, loss_rng)
                    return (train_state, rng), metrics

                train_state, dataset, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                permutation = jax.random.permutation(_rng, num_samples)
                shuffled_dataset = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), dataset)

                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_dataset,
                )

                (train_state, rng), epoch_metrics = jax.lax.scan(_update_minbatch, (train_state, rng), minibatches)
                update_state = (train_state, dataset, rng)
                return update_state, epoch_metrics

            update_state = (train_state, dataset, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            
            train_state = update_state[0]
            rng = update_state[-1]
            metric = jax.tree.map(jnp.mean, loss_info)
            env_metrics = jax.tree.map(
                lambda x: (x * traj_batch.info["returned_episode"]).sum()
                          / (traj_batch.info["returned_episode"].sum() + 1e-8),
                traj_batch.info,
            )
            metric.update(env_metrics)

            val_interval = config.get("VAL_INTERVAL", 50)
            is_val_step = (update_step % val_interval == 0)

            dummy_val_metrics = jax.tree.map(jnp.zeros_like, {f"val/{k}": v for k, v in env_metrics.items()})

            # 2. Fix RNG hygiene by splitting before the conditional
            rng, cond_val_rng = jax.random.split(rng)

            val_metrics = jax.lax.cond(
                is_val_step,
                lambda: _run_validation(train_state, cond_val_rng),  # Pass isolated RNG
                lambda: dummy_val_metrics
            )
            metric.update(val_metrics)

            if config["DEBUG"] and config["USE_WANDB"]:
                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)
                jax.debug.callback(callback, metric, update_step)

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
        
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metric": metric}

    return train


def run_offline_diffusion(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

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

    train_fn = make_train(config)
    train_vmap = jax.jit(jax.vmap(train_fn))

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
