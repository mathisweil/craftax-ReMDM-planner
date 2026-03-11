import time
import os
from typing import Any, Callable, Optional, Dict

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import wandb
import optax
from flax.training import train_state as flax_train_state
from flax import serialization
from craftax.craftax_env import make_craftax_env_from_name

from src.models.remdm import sample_plan
from src.models.reward_models import get_reward_model

from .common import SCHEDULE_MAP, _make_grad_step
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_checkpoint,
    _save_model,
    _make_env_stack,
    _make_apply_fns,
    _make_periodic_ckpt_manager,
    _resolve_ckpt_dir,
    _load_ppo_checkpoint,
)

def make_train_online(
    config: Dict[str, Any],
    init_params: Optional[Any] = None,
    ppo_agent: Optional[Any] = None,
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
    
    grad_step = _make_grad_step(apply_train, num_actions, schedule_fn, config.get("TRAIN_SIGMA", 0.0))

    def train(rng: jax.Array) -> Dict[str, Any]:
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        
        # --- 1. INITIALIZE DIFFUSION MODEL ---
        params = init_params if init_params is not None else _init_model_params(model, init_rng, obs_dim, plan_horizon)
        train_state = _create_train_state(model, params, config["LR"], config["MAX_GRAD_NORM"])
        
        # --- 2. INITIALIZE ENVIRONMENT ---
        obs, env_state = env_w.reset(env_rng, env_params)

        # --- 3. INITIALIZE REWARD MODEL & CO-TRAINING STATE ---
        reward_model = get_reward_model(config.get("REWARD_MODEL_TYPE", "mlp"))
        rm_params = reward_model.init(init_rng, jnp.zeros((1, obs_dim)))
        
        reward_load_path = config.get("REWARD_LOAD_PATH")
        if reward_load_path and os.path.exists(reward_load_path):
            with open(reward_load_path, "rb") as f:
                rm_params = serialization.from_bytes(rm_params, f.read())
            print(f"Loaded Reward Model weights from {reward_load_path}")
            
        rm_tx = optax.adam(learning_rate=1e-4)
        rm_state = flax_train_state.TrainState.create(
            apply_fn=reward_model.apply, params=rm_params, tx=rm_tx
        )

        # --- 4. THE COMPILED INNER LOOP ---
        def _update_step(runner_state, update_step_idx):
            train_state, env_state, obs, rng, rm_state = runner_state
            
            # Exponential PPO Injection Probability
            init_prob = config.get("PPO_INIT_PROB", 0.1)
            decay_rate = config.get("PPO_DECAY_RATE", 0.99)
            ppo_injection_prob = init_prob * jnp.power(decay_rate, update_step_idx)

            # Initialize PPO hidden state for the rollout chunk
            if ppo_agent is not None:
                init_ppo_hstate = ppo_agent.init_hidden(num_envs)
            else:
                init_ppo_hstate = jnp.zeros(1)

            def _plan_and_execute(carry, _):
                e_state, current_obs, current_rng, ppo_hstate = carry
                current_rng, plan_rng, sim_rng = jax.random.split(current_rng, 3)
                
                # 1. Sample the group of plans (Shape: [group_size, num_envs, plan_horizon])
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
                    jax.random.split(plan_rng, group_size), group_temps
                )

                # 2. THE MULTIVERSE: Simulate all plans to get True GRPO rewards
                def _sim_plan(plan):
                    def _sim_step(st_r, step_idx):
                        st, r = st_r
                        r, s_rng = jax.random.split(r)
                        o_next, st, rew, done, info = env_w.step(s_rng, st, plan[:, step_idx], env_params)
                        return (st, r), (o_next, rew)
                    _, (obs_traj, rew_traj) = jax.lax.scan(_sim_step, (e_state, sim_rng), jnp.arange(replan_every))
                    return obs_traj, rew_traj
                    
                all_obs_traj, all_rew_traj = jax.vmap(_sim_plan)(group_plans)
                
                # Calculate Intrinsic Reward efficiently during simulation
                flat_sim_obs = all_obs_traj.reshape(-1, obs_dim)
                intr_rews = reward_model.apply(rm_state.params, flat_sim_obs).reshape(group_size, replan_every, num_envs)
                
                # Combine Group Rewards
                intrinsic_coef = config.get("INTRINSIC_COEF", 0.05)
                
                # Extract the raw, unscaled sums for logging
                group_base_rewards = jnp.sum(all_rew_traj, axis=1)
                group_intr_rewards = jnp.sum(intr_rews, axis=1)
                
                # The scaled version for actual training
                group_train_rewards = group_base_rewards + (intrinsic_coef * group_intr_rewards)

                # 3. REAL EXECUTION: Advance the real universe using Plan 0 (with PPO injection)
                executed_plan = group_plans[0] # Take the first plan to actually play the game

                def _exec_step(c, step_idx):
                    st, cur_obs, r, hstate = c
                    r, s_rng, ppo_rng, choice_rng = jax.random.split(r, 4)
                    
                    diff_action = executed_plan[:, step_idx]
                    
                    if ppo_agent is not None:
                        ppo_dones = jnp.zeros(num_envs, dtype=bool)
                        pi, _, new_hstate = ppo_agent.apply(ppo_agent.params, cur_obs, hidden=hstate, done=ppo_dones)
                        ppo_action = jax.random.categorical(ppo_rng, pi.logits).squeeze(0)
                        
                        use_ppo = jax.random.bernoulli(choice_rng, ppo_injection_prob, shape=(num_envs,))
                        final_action = jnp.where(use_ppo, ppo_action, diff_action)
                    else:
                        final_action = diff_action
                        new_hstate = hstate

                    o_next, st, reward, done, info = env_w.step(s_rng, st, final_action, env_params)
                    return (st, o_next, r, new_hstate), (reward, done, info)

                (e_state, obs_next, current_rng, ppo_hstate), (_, _, infos) = jax.lax.scan(
                    _exec_step, (e_state, current_obs, current_rng, ppo_hstate), jnp.arange(replan_every)
                )
                
                return (e_state, obs_next, current_rng, ppo_hstate), (
                    current_obs, group_plans, group_train_rewards, group_base_rewards, group_intr_rewards, infos
                )

            # --- DATA COLLECTION AND TRUE GRPO ADVANTAGE ---
            num_plan_cycles = config["NUM_STEPS"] // replan_every
            (env_state, obs, rng, _), traj = jax.lax.scan(
                _plan_and_execute, (env_state, obs, rng, init_ppo_hstate), None, num_plan_cycles
            )
            # Unpack the two new variables!
            traj_obs, traj_group_plans, traj_train_rewards, traj_base_rewards, traj_intr_rewards, all_infos = traj
            
            # --- 1. TRAINING ADVANTAGES (Scaled) ---
            train_mean_r = jnp.mean(traj_train_rewards, axis=1, keepdims=True)
            train_std_r = jnp.std(traj_train_rewards, axis=1, keepdims=True) + 1e-8
            train_advantages = (traj_train_rewards - train_mean_r) / train_std_r 
            
            temperature = config.get("AWR_TEMPERATURE", 2.0)
            positive_adv_weights = jnp.clip(jnp.exp(train_advantages / temperature), 0.0, 20.0)
            
            # --- 2. LOGGING ADVANTAGES (Unscaled real values) ---
            real_segment_rewards = traj_base_rewards + traj_intr_rewards
            real_mean_r = jnp.mean(real_segment_rewards, axis=1, keepdims=True)
            real_std_r = jnp.std(real_segment_rewards, axis=1, keepdims=True) + 1e-8
            real_raw_advantages = (real_segment_rewards - real_mean_r) / real_std_r
            
            # --- PREPARE DATA FOR DIFFUSION ---
            # Train on ALL 8 plans by copying the observation for each plan!
            tiled_obs = jnp.tile(traj_obs[:, jnp.newaxis, :, :], (1, group_size, 1, 1)) 
            
            flat_obs = tiled_obs.reshape(-1, obs_dim)
            flat_plans = traj_group_plans.reshape(-1, plan_horizon)
            flat_advantages = positive_adv_weights.reshape(-1)
            
            total_samples = flat_obs.shape[0]

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

            # --- CO-TRAINING REWARD MODEL (UNIVERSAL SWITCHER) ---
            def _train_reward_fn(state):
                def _loss_fn(params):
                    preds = reward_model.apply(params, flat_obs)
                    model_type = config.get("REWARD_MODEL_TYPE", "mlp")
                    
                    if model_type == "mlp":
                        return jnp.mean((preds - (-1.0)) ** 2)
                    elif model_type in ["rnd", "vision_rnd"]:
                        return jnp.mean(preds)
                    else:
                        return jnp.mean(preds)
                
                loss, grads = jax.value_and_grad(_loss_fn)(state.params)
                return state.apply_gradients(grads=grads)
            
            # Only update the reward model once every 100 GRPO updates to keep it stable
            rm_state = jax.lax.cond(
                update_step_idx % 100 == 0,
                _train_reward_fn,  
                lambda s: s,       
                rm_state
            )

            # --- LOGGING ---
            ep_mask = all_infos["returned_episode"]
            n_done = jnp.maximum(ep_mask.sum(), 1)

            metrics = {
                "loss": epoch_infos["loss"].mean(),
                "ppo_prob": ppo_injection_prob,
                "advantage_mean": real_raw_advantages.mean(),
                "advantage_std": real_raw_advantages.std(),
                "reward_std": real_std_r.mean(), 
                "adv_abs_mean": jnp.mean(jnp.abs(train_advantages)), # What the network actually felt
                "reward_mean": real_segment_rewards.mean(),
                "env_reward_mean": traj_base_rewards.mean(),
                "intrinsic_reward_mean": traj_intr_rewards.mean(),
                "grad_norm": epoch_infos["grad_norm"].mean(),
                "death_toll": ep_mask.sum(),
            }
            
            ep_mask = all_infos["returned_episode"]
            n_done = jnp.maximum(ep_mask.sum(), 1)
            for k, v in all_infos.items():
                if "achievement" in k.lower():
                    metrics[f"Achievements/{k.split('_')[-1]}"] = (v * ep_mask).sum() / n_done

            jax.debug.print(
                "Update: {step} | Loss: {loss:.3f} | Score: {score:.2f} | Intr: {intr:.2f} | PPO%: {ppo:.3f} | Deaths: {deaths}",
                step=update_step_idx,
                loss=metrics["loss"],
                score=metrics["env_reward_mean"],
                intr=metrics["intrinsic_reward_mean"],
                ppo=metrics["ppo_prob"],
                deaths=metrics["death_toll"]
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

        runner_state = (train_state, env_state, obs, rng, rm_state)

        _jit_update_step = jax.jit(_update_step)
        all_metrics = []
        t0 = time.time()
        log_every = max(num_updates // 20, 1)

        with _make_periodic_ckpt_manager(config, subdir="checkpoints_online") as ckpt_mgr:
            for step_idx in range(num_updates):
                runner_state, metrics = _jit_update_step(runner_state, jnp.int32(step_idx))
                all_metrics.append(metrics)

                is_final = (step_idx == num_updates - 1)
                saved = ckpt_mgr.save(
                    step_idx + 1,
                    args=ocp.args.StandardSave(runner_state[0]),  # train_state
                    force=is_final,
                )
                if saved:
                    ckpt_dir = _resolve_ckpt_dir(config, subdir="checkpoints_online")
                    print(f"  Checkpoint saved at step {step_idx + 1} -> '{ckpt_dir}'")

                if (step_idx + 1) % log_every == 0 or is_final:
                    elapsed = time.time() - t0
                    loss_val = float(jax.device_get(metrics["loss"]))
                    print(f"  [{step_idx + 1:>6}/{num_updates}]  loss={loss_val:.4f}  elapsed={elapsed:.0f}s")

        stacked_metrics = jax.tree.map(lambda *xs: jnp.stack(xs), *all_metrics)
        return {"runner_state": runner_state, "metrics": stacked_metrics}

    return train

def run_online(config: Dict[str, Any]) -> None:
    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    config["NUM_ACTIONS"] = int(env.action_space(env.default_params).n)
    config["OBS_DIM"] = int(env.observation_space(env.default_params).shape[0])

    # 1. Load Offline Diffusion Checkpoint
    init_params = None
    if config.get("OFFLINE_CHECKPOINT_PATH"):
        model = _build_model(config, config["NUM_ACTIONS"])
        init_params = _load_checkpoint(config, model, config["OBS_DIM"], config["OFFLINE_CHECKPOINT_PATH"])
    
    # 2. Load PPO Teacher
    ppo_agent = None
    if config.get("PPO_CHECKPOINT_PATH"):
        print(f"Loading PPO Teacher from {config['PPO_CHECKPOINT_PATH']}...")
        ppo_agent = _load_ppo_checkpoint(
            config["PPO_CHECKPOINT_PATH"],
            config["NUM_ACTIONS"],
            config["OBS_DIM"],
            config.get("LAYER_SIZE", 512),
            model_type=config.get("PPO_MODEL_TYPE", "ppo_rnn"),
        )
        print("PPO Teacher loaded successfully!")
    
    if config.get("USE_WANDB"):
        wandb.init(project=config["WANDB_PROJECT"], config=config, name=f"GRPO-{config['ENV_NAME']}")

    # Pass the PPO agent into the factory
    train_fn = make_train_online(config, init_params=init_params, ppo_agent=ppo_agent)
    
    print("Starting Online GRPO Training...")
    out = train_fn(jax.random.PRNGKey(config["SEED"]))
    
    if config["SAVE_POLICY"]:
        _save_model(out["runner_state"][0], config, "diffusion_online_grpo")
    
    print("Training Complete.")