import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import jax

class DeterministicNeuralReward(nn.Module):
    hidden_dims: Sequence[int] = (512, 256, 128)
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        x = obs
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
        reward = nn.Dense(1)(x)
        return jnp.squeeze(reward, axis=-1)

class RNDReward(nn.Module):
    """
    Random Network Distillation.
    Reward is the MSE between a trained Predictor and a frozen Target network.
    """
    hidden_dims: Sequence[int] = (256, 128, 64)
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # --- TARGET NETWORK (Frozen) ---
        tx = obs
        for dim in self.hidden_dims:
            # Unique names prevent parameter sharing
            tx = nn.Dense(dim, name=f'target_dense_{dim}')(tx)
            tx = nn.relu(tx)
        
        # Completely freeze the target network from gradient updates
        target_emb = jax.lax.stop_gradient(tx)
        
        # --- PREDICTOR NETWORK (Trained) ---
        px = obs
        for dim in self.hidden_dims:
            px = nn.Dense(dim, name=f'pred_dense_{dim}')(px)
            px = nn.relu(px)
            
        # The Intrinsic Reward IS the prediction error!
        # Shape goes from [batch, 64] -> [batch]
        reward = jnp.mean((px - target_emb) ** 2, axis=-1)
        
        return reward

import jax.numpy as jnp
import flax.linen as nn
from jax.nn.initializers import orthogonal, lecun_normal
import jax

class VisionRNDReward(nn.Module):
    """
    RND with Orthogonal Initialization. Forces the Target network to be 
    highly chaotic so the Predictor actually has to work to minimize the loss!
    """
    internal_w: int = 9  
    internal_h: int = 9
    internal_c: int = 16 
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        
        def build_cnn(x, prefix):
            # THE FIX: Target network gets chaotic weights, Predictor gets stable weights
            w_init = orthogonal(1.414) if prefix == 'target' else lecun_normal()
            
            flat_size = self.internal_w * self.internal_h * self.internal_c
            x = nn.Dense(flat_size, kernel_init=w_init, name=f'{prefix}_proj')(x)
            x = nn.relu(x)
            
            x = x.reshape(x.shape[:-1] + (self.internal_w, self.internal_h, self.internal_c))
            
            x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', kernel_init=w_init, name=f'{prefix}_conv1')(x)
            x = nn.relu(x)
            x = nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding='VALID', kernel_init=w_init, name=f'{prefix}_conv2')(x)
            x = nn.relu(x)
            
            x = x.reshape(x.shape[:-3] + (-1,)) 
            x = nn.Dense(128, kernel_init=w_init, name=f'{prefix}_dense')(x)
            return x # Remember, no ReLU here!

        target_emb = jax.lax.stop_gradient(build_cnn(obs, 'target'))
        pred_emb = build_cnn(obs, 'pred')
            
        # Calculate MSE
        reward = jnp.mean((pred_emb - target_emb) ** 2, axis=-1)
        
        # THE SCALE FIX: Multiply the intrinsic reward so the agent actually feels it!
        return reward * 100.0

# Add it to your factory!
REWARD_MODELS = {
    "mlp": DeterministicNeuralReward,
    "rnd": RNDReward,
    "vision_rnd": VisionRNDReward,
}

def get_reward_model(model_type: str) -> nn.Module:
    if model_type not in REWARD_MODELS:
        raise ValueError(f"Unknown reward model type: {model_type}. Choose from {list(REWARD_MODELS.keys())}")
    return REWARD_MODELS[model_type]()