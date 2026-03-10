import time
from typing import Any, Callable, Optional, Dict

import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from src.models.reward_models import get_reward_model
from src.models.remdm import sample_plan
from src.logz.batch_logging import create_log_dict, batch_log

from .common import SCHEDULE_MAP, _make_grad_step
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_checkpoint,
    _save_model,
    _make_env_stack,
    _make_apply_fns,
)

def make_train_online(
    config: Dict[str, Any],
    init_params: Optional[Any] = None,
) -> Callable[[jax.Array], Dict[str, Any]]:
    
    # --- CONFIG SETUP ---
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    replan_every = config["REPLAN_EVERY"]
    num_updates = config["NUM_UPDATES"]
    update_epochs = config["UPDATE_EPOCHS"]
    num_minibatches = config["NUM_MINIBATCHES"]
    diffusion_steps = config["DIFFUSION_STEPS"]
    schedule_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    
    # GRPO specific: Number of plans to sample per state to calculate relative advantage
    # Usually 4-8 is a good group size for GRPO
    group_size = config.get("GRPO_GROUP_SIZE", 4) 
    
    num_actions = config["NUM_ACTIONS"]
    obs_dim = config["OBS_DIM"]

    # --- ENVIRONMENT & MODEL SETUP ---
    env_w, env_params = _make_env_stack(
        config, num_envs,
        use_optimistic_resets=config.get("USE_OPTIMISTIC_RESETS", False),
        use_sequence_history=True,
    )
    
    model = _build_model(config, num_actions)
    apply_inference, apply_train = _make_apply_fns(model)
    
    # Note: grad_step for GRPO is essentially a weighted Cross-Entropy loss 
    # where weights = advantages calculated from the group
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        
        # --- 1. INITIALIZE DIFFUSION MODEL ---
        params = init_params if init_params is not None else _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])
        
        # --- 2. INITIALIZE ENVIRONMENT (Must happen before runner_state is packed!) ---
        obs, env_state = env_w.reset(env_rng, env_params)

        # --- 3. INITIALIZE REWARD MODEL & CO-TRAINING STATE ---
        from flax.training import train_state as flax_train_state
        from flax import serialization
        import optax
        from src.models.reward_models import get_reward_model
        import os
        
        reward_model = get_reward_model(config.get("REWARD_MODEL_TYPE", "mlp"))
        rm_params = reward_model.init(init_rng, jnp.zeros((1, obs_dim)))
        
        reward_load_path = config.get("REWARD_LOAD_PATH")
        if reward_load_path and os.path.exists(reward_load_path):
            with open(reward_load_path, "rb") as f:
                rm_params = serialization.from_bytes(rm_params, f.read())
            # This print happens at JAX trace-time, which is perfectly safe
            print(f"Loaded Reward Model weights from {reward_load_path}")
            
        rm_tx = optax.adam(learning_rate=1e-4)
        rm_state = flax_train_state.TrainState.create(
            apply_fn=reward_model.apply, params=rm_params, tx=rm_tx
        )

        # --- 4. THE COMPILED INNER LOOP ---
        def _update_step(runner_state, update_step_idx):
            # Unpack the runner state (now including rm_state)
            train_state, env_state, obs, rng, rm_state = runner_state
            
            # Exponential PPO Injection Probability
            init_prob = config.get("PPO_INIT_PROB", 0.1)
            decay_rate = config.get("PPO_DECAY_RATE", 0.99)
            ppo_injection_prob = init_prob * jnp.power(decay_rate, update_step_idx)

            def _plan_and_execute(carry, _):
                e_state, current_obs, current_rng = carry
                current_rng, plan_rng, choice_rng, step_rng = jax.random.split(current_rng, 4)
                
                # Heterogeneous Temperatures for GRPO Group
                group_temps = jnp.linspace(0.5, 1.5, group_size)
                
                def _sample_single_plan(r, temp):
                    return sample_plan(
                        apply_inference, train_state.params, r, current_obs,
                        num_actions, plan_horizon, diffusion_steps, schedule_fn,
                        config["REMASK_STRATEGY"], config["ETA"], config.get("USE_LOOP", False),
                        config.get("T_ON", 0.7), config.get("T_OFF", 0.3), 
                        temp, config.get("TOP_P", None),
                    )
                
                group_plans = jax.vmap(_sample_single_plan)(
                    jax.random.split(plan_rng, group_size), 
                    group_temps
                )
                group_plans = jnp.transpose(group_plans, (1, 0, 2)) 

                # Execute the first plan in the environment
                executed_plan = group_plans[:, 0, :]

                def _exec_step(c, step_idx):
                    st, _, r = c
                    r, s_rng = jax.random.split(r)
                    o_next, st, reward, done, info = env_w.step(
                        s_rng, st, executed_plan[:, step_idx], env_params
                    )
                    return (st, o_next, r), (reward, done, info)

                (e_state, obs_next, current_rng), (rewards, dones, infos) = jax.lax.scan(
                    _exec_step, (e_state, current_obs, current_rng), jnp.arange(replan_every)
                )
                
                return (e_state, obs_next, current_rng), (current_obs, group_plans, rewards, dones, infos)

            # Collect Data
            num_plan_cycles = config["NUM_STEPS"] // replan_every
            (env_state, obs, rng), traj = jax.lax.scan(
                _plan_and_execute, (env_state, obs, rng), None, num_plan_cycles
            )
            traj_obs, traj_group_plans, traj_rewards, traj_dones, all_infos = traj
            
            
            # intrinsic_rewards shape: [num_plan_cycles, num_envs]
            intrinsic_rewards = reward_model.apply(rm_state.params, traj_obs)
            
            # Sum base rewards across BOTH cycles (axis 0) and micro-steps (axis 1)
            # Resulting shape: [num_envs]
            base_env_rewards = jnp.sum(traj_rewards, axis=(0, 1)) 
            
            # Sum intrinsic rewards across cycles (axis 0)
            # Resulting shape: [num_envs]
            total_intrinsic_rewards = jnp.sum(intrinsic_rewards, axis=0)
            
            # Total reward per environment for the rollout
            total_segment_rewards = base_env_rewards + total_intrinsic_rewards
            
            # Calculate GRPO Advantage (Shape: [num_envs])
            mean_r = jnp.mean(total_segment_rewards)
            std_r = jnp.std(total_segment_rewards) + 1e-8
            advantages = (total_segment_rewards - mean_r) / std_r
            
            # Flatten for training
            total_samples = num_plan_cycles * num_envs
            flat_obs = traj_obs.reshape(total_samples, obs_dim)
            flat_plans = traj_group_plans[:, :, 0, :].reshape(total_samples, plan_horizon)
            
            # Tile copies the [num_envs] array down to [num_plan_cycles, num_envs]
            adv_matrix = jnp.tile(advantages, (num_plan_cycles, 1)) 
            flat_advantages = adv_matrix.flatten() # Safely flatten to match flat_obs

            # Update Diffusion Agent
            def _update_epoch(carry, _):
                ts, r = carry
                r, p_rng = jax.random.split(r)
                perm = jax.random.permutation(p_rng, total_samples)
                
                obs_mbs = flat_obs[perm].reshape(num_minibatches, -1, obs_dim)
                plan_mbs = flat_plans[perm].reshape(num_minibatches, -1, plan_horizon)
                adv_mbs = flat_advantages[perm].reshape(num_minibatches, -1)

                def _update_minibatch(ts_r, data):
                    ts, r = ts_r
                    idx, o_mb, p_mb, a_mb = data
                    l_rng = jax.random.fold_in(r, idx)
                    ts, info = grad_step(ts, p_mb, o_mb, l_rng, advantages=a_mb)
                    return (ts, r), info

                return jax.lax.scan(_update_minibatch, (ts, r), (jnp.arange(num_minibatches), obs_mbs, plan_mbs, adv_mbs))

            (train_state, rng), epoch_infos = jax.lax.scan(_update_epoch, (train_state, rng), None, update_epochs)

            # --- CO-TRAINING REWARD MODEL ---
            def _train_reward_fn(state):
                def _loss_fn(params):
                    preds = reward_model.apply(params, flat_obs)
                    # Push recent states to -1.0 (they are now considered "known" baseline)
                    loss = jnp.mean((preds - (-1.0)) ** 2) 
                    return loss
                
                loss, grads = jax.value_and_grad(_loss_fn)(state.params)
                return state.apply_gradients(grads=grads)
            
            rm_state = jax.lax.cond(
                update_step_idx % 10 == 0,
                _train_reward_fn,  
                lambda s: s,       
                rm_state
            )

            # --- LOGGING ---
            metrics = {
                "loss": epoch_infos["loss"].mean(),
                "ppo_prob": ppo_injection_prob,
                "advantage_mean": advantages.mean(),
                "advantage_std": advantages.std(),
                "reward_std": std_r,
                "adv_abs_mean": jnp.mean(jnp.abs(advantages)),
                "reward_mean": mean_r,
                "intrinsic_reward_mean": intrinsic_rewards.mean(),
                "grad_norm": epoch_infos["grad_norm"].mean(),
            }
            
            ep_mask = all_infos["returned_episode"]
            n_done = jnp.maximum(ep_mask.sum(), 1)
            for k, v in all_infos.items():
                if "achievement" in k.lower():
                    metrics[f"Achievements/{k.split('_')[-1]}"] = (v * ep_mask).sum() / n_done

            jax.debug.print(
                "Update: {step} | Loss: {loss:.3f} | Adv Spread: {adv:.3f} | Score: {score:.2f} | Intrinsic: {intr:.2f} | PPO%: {ppo:.3f}",
                step=update_step_idx,
                loss=metrics["loss"],
                adv=metrics["adv_abs_mean"],
                score=metrics["reward_mean"],
                intr=metrics["intrinsic_reward_mean"],
                ppo=metrics["ppo_prob"]
            )

            if config.get("USE_WANDB") and config.get("DEBUG", True):
                def _wandb_callback(mets, step):
                    import numpy as np
                    try:
                        log_dict = {}
                        for k, v in mets.items():
                            val = np.array(v).item() 
                            clean_name = k.replace("returned_episode_achievements_", "").replace("_", " ").title()
                            if "Achievement" in k or "returned_episode" in k:
                                log_dict[f"Achievements/{clean_name}"] = val
                            else:
                                log_dict[f"online/{k}"] = val
                                
                        log_dict["online/step"] = int(step)
                        wandb.log(log_dict, step=int(step))
                    except Exception as e:
                        print(f"\n[WANDB ERROR at step {int(step)}]: {e}\n")

                jax.debug.callback(_wandb_callback, metrics, update_step_idx)
            
            return (train_state, env_state, obs, rng, rm_state), metrics

        # --- 5. PACK INITIAL RUNNER STATE & EXECUTE ---
        runner_state = (train_state, env_state, obs, rng, rm_state)
        
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, jnp.arange(num_updates))
        
        return {"runner_state": runner_state, "metrics": metrics}

    return train

def run_online(config: Dict[str, Any]) -> None:
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = int(env.action_space(env.default_params).n)
    config["OBS_DIM"] = int(env.observation_space(env.default_params).shape[0])

    init_params = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        model = _build_model(config, config["NUM_ACTIONS"])
        init_params = _load_checkpoint(config, model, config["OBS_DIM"], config["OFFLINE_CHECKPOINT_PATH"])
    
    # Standard setup and WandB init
    if config.get("USE_WANDB"):
        wandb.init(project=config["WANDB_PROJECT"], config=config, name=f"GRPO-{config['ENV_NAME']}")

    train_fn = make_train_online(config, init_params=init_params)
    
    print("Starting Online GRPO Training...")
    out = jax.jit(train_fn)(jax.random.PRNGKey(config["SEED"]))
    
    if config["SAVE_POLICY"]:
        _save_model(out["runner_state"][0], config, "diffusion_online_grpo")
    
    print("Training Complete.")