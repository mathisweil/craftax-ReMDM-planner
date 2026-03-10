import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence

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

# You will define your other two models here later!
# class CuriosityReward(nn.Module): ...
# class RNDReward(nn.Module): ...

# --- THE FACTORY ---
REWARD_MODELS = {
    "mlp": DeterministicNeuralReward,
    # "curiosity": CuriosityReward,
    # "rnd": RNDReward,
}

def get_reward_model(model_type: str) -> nn.Module:
    if model_type not in REWARD_MODELS:
        raise ValueError(f"Unknown reward model type: {model_type}. Choose from {list(REWARD_MODELS.keys())}")
    return REWARD_MODELS[model_type]()