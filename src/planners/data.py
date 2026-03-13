"""PPO agent adapter, environment construction, and trajectory data utilities."""

from __future__ import annotations
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name

from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.ppo_rnn import ActorCriticRNN
from Craftax_Baselines.ppo import ActorCritic
from Craftax_Baselines.ppo_rnd import ActorCriticRND


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: dict


# ---------------------------------------------------------------------------
# Environment construction
# ---------------------------------------------------------------------------

def make_env(config: dict, num_envs: int):
    """Build a wrapped Craftax environment. Returns (env, env_params)."""
    env = make_craftax_env_from_name(
        config["ENV_NAME"], not config.get("USE_OPTIMISTIC_RESETS", False),
    )
    env_params = env.default_params
    env = LogWrapper(env)
    if config.get("USE_OPTIMISTIC_RESETS", False):
        env = OptimisticResetVecEnvWrapper(
            env, num_envs=num_envs,
            reset_ratio=min(config.get("OPTIMISTIC_RESET_RATIO", num_envs), num_envs),
        )
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=num_envs)
    return env, env_params


# ---------------------------------------------------------------------------
# PPO checkpoint loading
# ---------------------------------------------------------------------------

def load_ppo_params(
    path: str,
    network: Any,
    model_type: str,
    num_envs: int,
    obs_shape: tuple,
    layer_size: int = 512,
) -> Any:
    """Restore PPO parameters from an Orbax checkpoint."""
    rng = jax.random.PRNGKey(0)

    if model_type == "ppo_rnn":
        init_x = (jnp.zeros((1, num_envs, *obs_shape)), jnp.zeros((1, num_envs)))
        abstract = network.init(rng, jnp.zeros((num_envs, layer_size)), init_x)
    else:
        abstract = network.init(rng, jnp.zeros((1, *obs_shape)))

    with ocp.CheckpointManager(path) as mgr:
        step = mgr.latest_step()
        if step is None:
            raise FileNotFoundError(f"No checkpoint at {path}")
        restored = mgr.restore(
            step,
            args=ocp.args.PyTreeRestore(item={"params": abstract}, partial_restore=True),
        )
    print(f"Loaded {model_type.upper()} checkpoint from '{path}' (step {step})")
    return restored["params"]


def build_ppo_network(model_type: str, num_actions: int, layer_size: int, config: dict):
    """Instantiate the correct PPO architecture."""
    model_type = model_type.lower()
    if model_type == "ppo_rnn":
        return ActorCriticRNN(num_actions, config=config)
    elif model_type == "ppo_rnd":
        return ActorCriticRND(num_actions, layer_size)
    return ActorCritic(num_actions, layer_size)


def load_ppo_agent(
    path: str, num_actions: int, obs_dim: int,
    layer_size: int, model_type: str, config: dict,
    num_envs: int = 1,
) -> "PPOAgent":
    """One-shot: build network, load params, return PPOAgent."""
    net = build_ppo_network(model_type, num_actions, layer_size, config)
    params = load_ppo_params(path, net, model_type, num_envs, (obs_dim,), layer_size)
    return PPOAgent(net, params, model_type, layer_size)


# ---------------------------------------------------------------------------
# PPO agent adapter
# ---------------------------------------------------------------------------

class PPOAgent:
    """Uniform interface over PPO-RNN / PPO / PPO-RND for action collection."""

    def __init__(self, network, params, model_type: str, layer_size: int = 512):
        self.network = network
        self.params = params
        self.model_type = model_type.lower()
        self.layer_size = layer_size

    def init_hidden(self, batch_size: int):
        if self.model_type == "ppo_rnn":
            return jnp.zeros((batch_size, self.layer_size))
        return None

    def act(self, obs, done, hidden, rng, temperature=1.0):
        """Returns (action, new_hidden)."""
        if self.model_type == "ppo_rnn":
            ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
            new_hidden, pi, _ = self.network.apply(self.params, hidden, ac_in)
        elif self.model_type == "ppo_rnd":
            pi, _, _ = self.network.apply(self.params, obs)
            new_hidden = hidden
        else:
            pi, _ = self.network.apply(self.params, obs)
            new_hidden = hidden

        action = jax.random.categorical(rng, pi.logits / temperature)
        if self.model_type == "ppo_rnn":
            action = action.squeeze(0)
        return action, new_hidden

    def get_pi(self, obs, done=None, hidden=None):
        """Return the policy distribution (for GRPO simulation)."""
        if self.model_type == "ppo_rnn":
            ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
            new_hidden, pi, _ = self.network.apply(self.params, hidden, ac_in)
            return pi, new_hidden
        elif self.model_type == "ppo_rnd":
            pi, _, _ = self.network.apply(self.params, obs)
            return pi, hidden
        else:
            pi, _ = self.network.apply(self.params, obs)
            return pi, hidden
