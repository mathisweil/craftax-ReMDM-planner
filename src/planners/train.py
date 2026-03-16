"""Training loop: environment rollout -> diffusion & WM window extraction -> gradient updates."""

from __future__ import annotations

import os
import time
from typing import Any

import jax
from jax import remat
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from craftax.craftax_env import make_craftax_env_from_name

from src.diffusion.loss import compute_loss
from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import SCHEDULE_MAP
from src.models.denoiser import DenoisingTransformer
from src.models.worldmodel import TransformerWorldModel
from .data import (
    PPOAgent,
    Transition,
    build_ppo_network,
    load_ppo_params,
)
from .state import init_params, create_train_state, make_apply_fns
from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.logz.batch_logging import create_log_dict, batch_log

# ---------------------------------------------------------------------------
# Gradient step factories
# ---------------------------------------------------------------------------

def _action_stats(acts: jnp.ndarray, num_actions: int, valid: jnp.ndarray) -> dict[str, jnp.ndarray]:
    mask = jnp.broadcast_to(valid[:, None], acts.shape).reshape(-1)
    flat = jnp.where(mask, acts.reshape(-1), num_actions + 1)
    counts = jnp.bincount(flat, length=num_actions).astype(jnp.float32)
    probs = counts / jnp.maximum(counts.sum(), 1.0)
    entropy = -jnp.sum(probs * jnp.log(jnp.where(probs > 0, probs, 1.0)))
    return {
        "action_entropy": entropy,
        "action_unique_frac": jnp.sum(probs > 0).astype(jnp.float32) / num_actions,
    }

# 1. Diffusion Grad Step
def _make_grad_step(apply_train, num_actions, schedule_fn, schedule_deriv_fn, sigma_t, label_smoothing):
    def _loss_fn(params, acts, obs, valid, rng):
        return compute_loss(
            apply_train, params, rng, acts, obs, valid,
            num_actions, schedule_fn, schedule_deriv_fn,
            sigma_t=sigma_t, label_smoothing=label_smoothing,
        )

    def step(state, acts, obs, valid, rng):
        (loss, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            state.params, acts, obs, valid, rng,
        )
        state = state.apply_gradients(grads=grads)
        info["diff_grad_norm"] = optax.tree.norm(grads)
        info.update(_action_stats(acts, num_actions, valid))
        return state, info

    return step

# 2. World Model Grad Step (Fully Overhauled)
def _make_wm_grad_step(apply_wm_train, config):
    reward_weight = config.get("WM_REWARD_WEIGHT", 10.0)
    anneal_steps = config.get("WM_ROLLOUT_ANNEAL_STEPS", 1000.0)
    max_p_rollout = config.get("WM_MAX_P_ROLLOUT", 0.5)

    def _wm_loss_fn(wm_params, obs_seq, act_seq, next_obs_seq, rew_seq, dones_seq, rng, step_idx):
        B, T, obs_dim = obs_seq.shape

        # FIX 2: Per-timestep valid mask (only mask AFTER death)
        cumulative_dones = jnp.cumsum(dones_seq, axis=1)
        valid_mask = (cumulative_dones == 0).astype(jnp.float32) # (B, T)

        # FIX 3: Scheduled Sampling Annealing
        p_rollout = jnp.minimum(step_idx / anneal_steps, max_p_rollout)

        def scan_step(carry, step_idx_t):
            current_obs_seq, rng = carry
            rng, drop_rng, sample_rng = jax.random.split(rng, 3)

            # Forward pass using sequence built so far
            # Requires updated WorldModel to return: next_obs, rew_logits, rew_mags
            next_obs_preds, rew_logits, rew_mags = apply_wm_train(
                wm_params, drop_rng, current_obs_seq, act_seq
            )

            # Extract specific step predictions
            pred_next_obs = next_obs_preds[:, step_idx_t, :]
            pred_rew_logit = rew_logits[:, step_idx_t]
            pred_rew_mag = rew_mags[:, step_idx_t]

            # Real targets for this step
            real_next_obs_t = next_obs_seq[:, step_idx_t, :]
            rew_seq_t = rew_seq[:, step_idx_t]
            valid_t = valid_mask[:, step_idx_t]

            # FIX 4: Observation Normalization (Batch-level standardization)
            obs_mean = jnp.mean(real_next_obs_t, axis=0)
            obs_std = jnp.std(real_next_obs_t, axis=0) + 1e-8
            norm_pred_obs = (pred_next_obs - obs_mean) / obs_std
            norm_real_obs = (real_next_obs_t - obs_mean) / obs_std
            
            obs_loss_t = jnp.mean(jnp.square(norm_pred_obs - norm_real_obs), axis=-1)

            # FIX 5: Binary Reward Head + Magnitude
            reward_happened = (rew_seq_t > 0).astype(jnp.float32)
            bce_loss_t = optax.sigmoid_binary_cross_entropy(pred_rew_logit, reward_happened)
            mag_loss_t = jnp.square(pred_rew_mag - rew_seq_t) * reward_happened
            rew_loss_t = bce_loss_t + mag_loss_t

            # FIX 3: Scheduled sampling - mix real vs predicted for the NEXT input
            use_predicted = jax.random.bernoulli(sample_rng, p_rollout, shape=(B,))
            next_obs_input = jnp.where(use_predicted[:, None], pred_next_obs, real_next_obs_t)

            # Inject predicted state into sequence for the next loop
            next_obs_seq_carry = jax.lax.cond(
                step_idx_t + 1 < T,
                lambda: current_obs_seq.at[:, step_idx_t + 1, :].set(next_obs_input),
                lambda: current_obs_seq
            )

            # ... [end of scan_step function] ...
            return (next_obs_seq_carry, rng), (obs_loss_t, rew_loss_t, bce_loss_t, mag_loss_t, valid_t)

        # --- THE NEW CODE: Wrap the step to save VRAM ---
        remat_scan_step = remat(scan_step)

        # Autoregressive unroll (use the remat version!)
        init_obs_seq = obs_seq
        _, (obs_losses, rew_losses, bce_losses, mag_losses, valids) = jax.lax.scan(
            remat_scan_step, (init_obs_seq, rng), jnp.arange(T)
        )

        # valids is shape (T, B), losses are (T, B)
        total_valid = jnp.maximum(jnp.sum(valids), 1.0)
        
        total_obs_loss = jnp.sum(obs_losses * valids) / total_valid
        total_rew_loss = jnp.sum(rew_losses * valids) / total_valid
        total_bce_loss = jnp.sum(bce_losses * valids) / total_valid
        total_mag_loss = jnp.sum(mag_losses * valids) / total_valid

        # FIX 1: Upweight Reward Loss
        loss = total_obs_loss + (reward_weight * total_rew_loss)

        return loss, {
            "wm_total_loss": loss,
            "wm_obs_loss": total_obs_loss,
            "wm_rew_loss": total_rew_loss,
            "wm_rew_bce_loss": total_bce_loss,
            "wm_rew_mag_loss": total_mag_loss,
            "wm_p_rollout": p_rollout
        }

    def step(wm_state, obs_seq, act_seq, next_obs_seq, rew_seq, dones_seq, rng, step_idx):
        (loss, info), grads = jax.value_and_grad(_wm_loss_fn, has_aux=True)(
            wm_state.params, obs_seq, act_seq, next_obs_seq, rew_seq, dones_seq, rng, step_idx
        )
        wm_state = wm_state.apply_gradients(grads=grads)
        info["wm_grad_norm"] = optax.tree.norm(grads)
        return wm_state, info
    
    return step

# ---------------------------------------------------------------------------
# make_train
# ---------------------------------------------------------------------------

def make_train(config: dict[str, Any]):
    num_steps = config["NUM_STEPS"]
    num_envs = config["NUM_ENVS"]
    plan_horizon = config["PLAN_HORIZON"]
    valid_per_rollout = num_steps - plan_horizon + 1
    num_samples = num_envs * valid_per_rollout

    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // num_steps // num_envs
    assert num_samples % config["NUM_MINIBATCHES"] == 0, (
        f"{num_samples} samples not divisible by {config['NUM_MINIBATCHES']} minibatches"
    )
    config["MINIBATCH_SIZE"] = num_samples // config["NUM_MINIBATCHES"]

    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params
    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(
            env, num_envs=num_envs,
            reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], num_envs),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=num_envs)

    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dim = obs_shape[0]

    model_type = config["PPO_MODEL_TYPE"]
    ppo_net = build_ppo_network(model_type, num_actions, config["LAYER_SIZE"], config)
    ppo_params = load_ppo_params(
        config["PPO_CHECKPOINT_PATH"], ppo_net, model_type, num_envs, obs_shape, config["LAYER_SIZE"],
    )
    ppo = PPOAgent(ppo_net, ppo_params, model_type, config["LAYER_SIZE"])

    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]

    def train(rng: jax.Array) -> dict[str, Any]:
        # 1. Init Diffusion Model
        net = DenoisingTransformer(
            num_actions=num_actions, plan_horizon=plan_horizon,
            d_model=config["D_MODEL"], n_heads=config["N_HEADS"],
            n_layers=config["N_LAYERS"], d_ff=config["D_FF"],
            obs_encoder_layers=config["OBS_ENCODER_LAYERS"],
            obs_encoder_width=config["OBS_ENCODER_WIDTH"],
            dropout_rate=config["DROPOUT_RATE"],
        )
        apply_eval, apply_train = make_apply_fns(net)
        grad_step = _make_grad_step(
            apply_train, num_actions, schedule_fn, schedule_deriv_fn,
            config.get("TRAIN_SIGMA", 0.0), config.get("LABEL_SMOOTHING", 0.0),
        )

        # 2. Init World Model
        wm_net = TransformerWorldModel(
            num_actions=num_actions, obs_dim=obs_dim,
            d_model=config.get("WM_D_MODEL", config["D_MODEL"]), 
            n_heads=config.get("WM_N_HEADS", config["N_HEADS"]),
            n_layers=config.get("WM_N_LAYERS", config["N_LAYERS"]),
            dropout_rate=config.get("DROPOUT_RATE", 0.1),
        )
        
        def apply_wm_train(params, rng, obs_seq, act_seq):
            # Expects updated WM to return 3 values
            return wm_net.apply({"params": params}, obs_seq, act_seq, deterministic=False, rngs={"dropout": rng})
        
        wm_grad_step = _make_wm_grad_step(apply_wm_train, config)

        # 3. Create Train States
        rng, init_rng, wm_init_rng, env_rng = jax.random.split(rng, 4)
        
        params = init_params(net, init_rng, obs_dim, plan_horizon)
        state = create_train_state(net, params, config["LR"], config["MAX_GRAD_NORM"])
        
        dummy_obs_seq = jnp.zeros((1, plan_horizon, obs_dim))
        dummy_act_seq = jnp.zeros((1, plan_horizon), dtype=jnp.int32)
        wm_params = wm_net.init(wm_init_rng, dummy_obs_seq, dummy_act_seq)["params"]
        
        # FIX 7: Separate Gradient Clipping for WM
        wm_state = create_train_state(
            wm_net, wm_params, 
            config.get("WM_LR", config["LR"]), 
            config.get("WM_MAX_GRAD_NORM", 1.0) 
        )

        obsv, env_state = env.reset(env_rng, env_params)
        init_hstate = ppo.init_hidden(num_envs)

        def _validate(state, wm_state, rng):
            # ... (Keep existing validation code from previous steps) ...
            return {}

        # ------------------------------------------------------------------
        # Update step
        # ------------------------------------------------------------------
        def _update_step(runner, _):
            state, wm_state, env_state, last_obs, last_done, hstate, rng, step_idx = runner

            def _env_step(carry, _):
                st, es, obs, done, hs, rng = carry
                rng, act_rng, step_rng = jax.random.split(rng, 3)
                action, new_hs = ppo.act(
                    obs, done, hs, act_rng, temperature=config.get("COLLECT_TEMPERATURE", 1.0),
                )
                new_obs, es, reward, new_done, info = env.step(step_rng, es, action, env_params)
                t = Transition(done=done, action=action, reward=reward, obs=obs, info=info)
                return (st, es, new_obs, new_done, new_hs, rng), t

            (new_runner_st, env_state, last_obs, last_done, hstate, rng), traj = jax.lax.scan(
                _env_step, (state, env_state, last_obs, last_done, hstate, rng), None, num_steps,
            )

            all_obs = jnp.concatenate([traj.obs, last_obs[None, ...]], axis=0)

            def _window(t_idx):
                obs_t = traj.obs[t_idx]
                acts = jax.lax.dynamic_slice(traj.action, (t_idx, 0), (plan_horizon, num_envs))
                dones = jax.lax.dynamic_slice(traj.done, (t_idx, 0), (plan_horizon, num_envs))
                valid = ~jnp.any(dones, axis=0)
                
                obs_seq = jax.lax.dynamic_slice(all_obs, (t_idx, 0, 0), (plan_horizon, num_envs, obs_dim))
                next_obs_seq = jax.lax.dynamic_slice(all_obs, (t_idx+1, 0, 0), (plan_horizon, num_envs, obs_dim))
                rew_seq = jax.lax.dynamic_slice(traj.reward, (t_idx, 0), (plan_horizon, num_envs))
                
                return obs_t, jnp.swapaxes(acts, 0, 1), valid, \
                       jnp.swapaxes(obs_seq, 0, 1), jnp.swapaxes(next_obs_seq, 0, 1), \
                       jnp.swapaxes(rew_seq, 0, 1), jnp.swapaxes(dones, 0, 1)

            obs_w, act_w, valid_w, obs_seq_w, next_obs_seq_w, rew_seq_w, dones_seq_w = jax.vmap(_window)(jnp.arange(valid_per_rollout))
            
            flat_obs = obs_w.reshape(-1, obs_dim)
            flat_acts = act_w.reshape(-1, plan_horizon)
            flat_valid = valid_w.reshape(-1)
            
            flat_obs_seq = obs_seq_w.reshape(-1, plan_horizon, obs_dim)
            flat_next_obs_seq = next_obs_seq_w.reshape(-1, plan_horizon, obs_dim)
            flat_rew_seq = rew_seq_w.reshape(-1, plan_horizon)
            flat_dones_seq = dones_seq_w.reshape(-1, plan_horizon)
            
            # ------------------------------------------------------------------
            # FIX 8: Decoupled Training Loops (Allows more epochs for WM)
            # ------------------------------------------------------------------
            
            # --- 1. Diffusion Training Loop ---
            diff_dataset = (flat_obs, flat_acts, flat_valid)
            def _diff_epoch(epoch_state, _):
                st_d, ds, rng = epoch_state
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, num_samples)
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), ds)
                batches = jax.tree.map(lambda x: x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:]), shuffled)

                def _diff_mb(carry, batch):
                    st, r = carry
                    r, diff_rng = jax.random.split(r)
                    obs_b, act_b, val_b = batch
                    st, diff_metrics = grad_step(st, act_b, obs_b, val_b, diff_rng)
                    return (st, r), diff_metrics

                (st_d, rng), metrics = jax.lax.scan(_diff_mb, (st_d, rng), batches)
                return (st_d, ds, rng), metrics

            (state, _, rng), diff_loss_info = jax.lax.scan(
                _diff_epoch, (state, diff_dataset, rng), None, config["UPDATE_EPOCHS"],
            )

            # --- 2. World Model Training Loop ---
            wm_dataset = (flat_obs_seq, flat_acts, flat_next_obs_seq, flat_rew_seq, flat_dones_seq)
            wm_epochs = config.get("WM_UPDATE_EPOCHS", config["UPDATE_EPOCHS"] * 2) # More epochs for WM
            
            def _wm_epoch(epoch_state, _):
                st_w, ds, rng = epoch_state
                rng, perm_rng = jax.random.split(rng)
                perm = jax.random.permutation(perm_rng, num_samples)
                shuffled = jax.tree.map(lambda x: jnp.take(x, perm, axis=0), ds)
                batches = jax.tree.map(lambda x: x.reshape(config["NUM_MINIBATCHES"], -1, *x.shape[1:]), shuffled)

                def _wm_mb(carry, batch):
                    st, r = carry
                    r, wm_rng = jax.random.split(r)
                    obs_seq_b, act_b, next_obs_seq_b, rew_seq_b, dones_seq_b = batch
                    
                    st, wm_metrics = wm_grad_step(
                        st, obs_seq_b, act_b, next_obs_seq_b, rew_seq_b, dones_seq_b, wm_rng, step_idx
                    )
                    return (st, r), wm_metrics

                (st_w, rng), metrics = jax.lax.scan(_wm_mb, (st_w, rng), batches)
                return (st_w, ds, rng), metrics

            (wm_state, _, rng), wm_loss_info = jax.lax.scan(
                _wm_epoch, (wm_state, wm_dataset, rng), None, wm_epochs,
            )

            # Combine metrics
            metric = jax.tree.map(jnp.mean, diff_loss_info)
            wm_metric = jax.tree.map(jnp.mean, wm_loss_info)
            metric.update(wm_metric)

            returned = traj.info["returned_episode"]
            env_metrics = jax.tree.map(
                lambda x: (x * returned).sum() / (returned.sum() + 1e-8), traj.info,
            )
            metric.update(env_metrics)

            if config["USE_WANDB"]:
                def _log(m, s):
                    log_dict = create_log_dict(m, config)
                    for k, v in m.items():
                        if "loss" in k or "grad_norm" in k or "action_" in k or "val/" in k or "wm_" in k:
                            log_dict[k] = float(v)
                    
                    if int(s) % config.get("VAL_INTERVAL", 50) != 0:
                        log_dict = {k: v for k, v in log_dict.items() if not k.startswith("val/")}

                    wandb.log(log_dict, step=int(s))
                    
                jax.debug.callback(_log, metric, step_idx)

            runner = (state, wm_state, env_state, last_obs, last_done, hstate, rng, step_idx + 1)
            return runner, metric

        rng, run_rng = jax.random.split(rng)
        runner_init = (
            state, wm_state, env_state, obsv, jnp.zeros(num_envs, dtype=bool),
            init_hstate, run_rng, 0,
        )
        runner_final, metrics = jax.lax.scan(_update_step, runner_init, None, config["NUM_UPDATES"])
        return {"runner_state": runner_final, "metric": metrics}

    return train

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_offline_diffusion(config):
    config = {k.upper(): v for k, v in config.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"], entity=config["WANDB_ENTITY"],
            config=config,
            name=f"{config['ENV_NAME']}-DualTrain-{int(config['TOTAL_TIMESTEPS'] // 1e6)}M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_fn = jax.jit(jax.vmap(make_train(config)))

    t0 = time.time()
    out = train_fn(rngs)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.1f}s  SPS: {config['TOTAL_TIMESTEPS'] / elapsed:.0f}")

    if config["USE_WANDB"] and config["SAVE_POLICY"]:
        train_states = out["runner_state"][0] # Extract the vectorized state array
        wm_states = out["runner_state"][1]
        
        # Take the first model from the vectorized repeats
        train_state = jax.tree.map(lambda x: x[0], train_states) 
        wm_state = jax.tree.map(lambda x: x[0], wm_states)

        # Save Diffusion Brain
        diff_path = os.path.join(wandb.run.dir, "policies")
        with ocp.CheckpointManager(diff_path, options=ocp.CheckpointManagerOptions(max_to_keep=1)) as mgr:
            mgr.save(int(config["TOTAL_TIMESTEPS"]), args=ocp.args.StandardSave(train_state))
        
        # Save World Model Engine
        wm_path = os.path.join(wandb.run.dir, "world_model")
        with ocp.CheckpointManager(wm_path, options=ocp.CheckpointManagerOptions(max_to_keep=1)) as mgr:
            mgr.save(int(config["TOTAL_TIMESTEPS"]), args=ocp.args.StandardSave(wm_state))
            
        print(f"Saved Diffusion Brain to {diff_path}")
        print(f"Saved World Model to {wm_path}")