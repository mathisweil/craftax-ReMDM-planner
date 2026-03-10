import time
from typing import Any, Dict
import os
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training import train_state
from flax import serialization

from src.models.reward_models import get_reward_model

from .utils import _save_model

def run_train_reward(config: Dict[str, Any]) -> None:
    data_path = config["OFFLINE_DATA_PATH"]
    print(f"Loading offline trajectories from {data_path}...")
    
    # 1. LOAD THE .NPZ FILE
    try:
        data = np.load(data_path)
        print(f"Successfully loaded dataset! Keys found: {data.files}")
        
        # Extract the observations (check your collect script for the exact key name, 
        # it is usually "obs" or "observations")
        obs = data["obs"] 
        
    except Exception as e:
        raise RuntimeError(f"Failed to load data from {data_path}. Error: {e}")

    # 2. FLATTEN THE DATA
    # If the shape is [envs, steps, obs_dim], we smash the first two dimensions together
    if len(obs.shape) > 2:
        obs = obs.reshape(-1, obs.shape[-1])
        
    print(f"Total individual frames loaded: {obs.shape[0]}")
    print(f"Observation dimension: {obs.shape[1]}")

    # 3. SPLIT INTO POSITIVE AND NEGATIVE SAMPLES (Meier & Mujika method)
    # We will assume the agent gets further into the game/tech tree as the steps go on.
    # Therefore, we take the FIRST 80% of frames as Negative (-1) 
    # and the LAST 20% of frames as Positive (+1).
    
    split_idx = int(obs.shape[0] * 0.8)
    
    obs_neg = obs[:split_idx]  # Common / Early game states
    obs_pos = obs[split_idx:]  # Rare / Deep exploration states
    
    print(f"Negative Samples (Target -1.0): {obs_neg.shape[0]}")
    print(f"Positive Samples (Target +1.0): {obs_pos.shape[0]}")

    # BUILD MODEL FROM FACTORY
    rng = jax.random.PRNGKey(config.get("SEED", 42))
    rng, init_rng = jax.random.split(rng)
    
    model = get_reward_model(config.get("REWARD_MODEL_TYPE", "mlp"))
    obs_dim = obs.shape[-1]
    
    # INITIALIZE DUMMY PARAMS (JAX always requires this to get shapes)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = model.init(init_rng, dummy_obs)
    
    # LOAD WEIGHTS (If path is provided)
    load_path = config.get("REWARD_LOAD_PATH")
    if load_path and os.path.exists(load_path):
        print(f"Loading pre-trained reward weights from {load_path}...")
        with open(load_path, "rb") as f:
            # Inject the saved weights into the dummy parameters!
            params = serialization.from_bytes(params, f.read())

    # CREATE TRAIN STATE
    tx = optax.adam(learning_rate=config.get("REWARD_LR", 3e-4))
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # --- UNIVERSAL LOSS FUNCTION SWITCHER ---
    model_type = config.get("REWARD_MODEL_TYPE", "mlp")
    
    if model_type == "mlp":
        # 1. Standard Discriminator Math (Meier & Mujika)
        @jax.jit
        def train_step(state, batch_neg, batch_pos):
            def loss_fn(params):
                r_neg = model.apply(params, batch_neg)
                r_pos = model.apply(params, batch_pos)
                
                loss_neg = jnp.mean((r_neg - (-1.0)) ** 2)
                loss_pos = jnp.mean((r_pos - 1.0) ** 2)
                total_loss = loss_neg + loss_pos
                return total_loss, (loss_neg, loss_pos, r_neg.mean(), r_pos.mean())
                
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (l_n, l_p, mean_r_neg, mean_r_pos)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            
            metrics = {
                "reward_loss": loss, "loss_neg": l_n, "loss_pos": l_p,
                "pred_reward_neg": mean_r_neg, "pred_reward_pos": mean_r_pos 
            }
            return state, metrics

    else:
        # 2. RND & Vision RND Math
        # For RND, pre-training just means teaching the Predictor to mimic the Target 
        # on the offline dataset to establish a baseline. We just minimize the output directly!
        @jax.jit
        def train_step(state, batch_neg, batch_pos):
            def loss_fn(params):
                r_neg = model.apply(params, batch_neg)
                r_pos = model.apply(params, batch_pos)
                
                # The output IS the error. Just minimize it!
                loss_neg = jnp.mean(r_neg)
                loss_pos = jnp.mean(r_pos)
                total_loss = loss_neg + loss_pos
                return total_loss, (loss_neg, loss_pos, r_neg.mean(), r_pos.mean())
                
            grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
            (loss, (l_n, l_p, mean_r_neg, mean_r_pos)), grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
            
            metrics = {
                "reward_loss": loss, "loss_neg": l_n, "loss_pos": l_p,
                "pred_reward_neg": mean_r_neg, "pred_reward_pos": mean_r_pos 
            }
            return state, metrics

    # 5. TRAINING LOOP
    if config.get("USE_WANDB"):
        wandb.init(project=config.get("WANDB_PROJECT", "remdm-craftax"), name="Train-Neural-Reward")

    epochs = config.get("REWARD_EPOCHS", 10)
    batch_size = config.get("BATCH_SIZE", 256)
    
    print("Starting Reward Model Training...")
    for epoch in range(epochs):
        # Shuffle negative and positive datasets
        rng, shuffle_rng = jax.random.split(rng)
        perm_neg = jax.random.permutation(shuffle_rng, obs_neg.shape[0])
        perm_pos = jax.random.permutation(shuffle_rng, obs_pos.shape[0])
        
        obs_neg_shuffled = obs_neg[perm_neg]
        obs_pos_shuffled = obs_pos[perm_pos]
        
        # We step through the smaller of the two datasets
        num_batches = min(len(obs_neg), len(obs_pos)) // batch_size
        
        epoch_metrics = []
        for b in range(num_batches):
            b_neg = obs_neg_shuffled[b*batch_size : (b+1)*batch_size]
            b_pos = obs_pos_shuffled[b*batch_size : (b+1)*batch_size]
            
            state, mets = train_step(state, b_neg, b_pos)
            epoch_metrics.append(mets)
            
        # Log average metrics for the epoch
        avg_metrics = {k: np.mean([m[k] for m in epoch_metrics]) for k in epoch_metrics[0].keys()}
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_metrics['reward_loss']:.3f} | Pos Reward: {avg_metrics['pred_reward_pos']:.2f} | Neg Reward: {avg_metrics['pred_reward_neg']:.2f}")
        
        if config.get("USE_WANDB"):
            wandb.log({f"reward_model/{k}": v for k, v in avg_metrics.items()}, step=epoch)

    # 6. SAVE THE MODEL
    save_path = config.get("REWARD_SAVE_PATH", "checkpoints/reward_model.msgpack")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "wb") as f:
        # Serialize the parameters to a binary msgpack file
        f.write(serialization.to_bytes(state.params))
    print(f"Reward weights successfully saved to {save_path}!")