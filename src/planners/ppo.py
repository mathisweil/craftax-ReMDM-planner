"""PPO agent adapter and checkpoint loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from Craftax_Baselines.ppo import ActorCritic
from Craftax_Baselines.ppo_rnd import ActorCriticRND
from Craftax_Baselines.ppo_rnn import ActorCriticRNN


def load_ppo_params(
    path: str,
    network: Any,
    model_type: str,
    num_envs: int,
    obs_shape: tuple,
    layer_size: int = 512,
    seed: int = 0,
) -> Any:
    """Restore PPO parameters from an Orbax checkpoint.

    Args:
        path:        Path to the Orbax checkpoint directory.
        network:     Instantiated Flax network (used only for structure).
        model_type:  One of ``"ppo_rnn"``, ``"ppo_rnd"``, or ``"ppo"``.
        num_envs:    Number of parallel environments (affects RNN init shape).
        obs_shape:   Observation shape tuple.
        layer_size:  Hidden layer size (for RNN hidden state init).

    Returns:
        Restored parameter pytree.
    """
    path = str(Path(path).resolve())
    rng = jax.random.PRNGKey(seed)
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
            args=ocp.args.PyTreeRestore(
                item={"params": abstract}, partial_restore=True
            ),
        )
    print(f"Loaded {model_type.upper()} checkpoint from '{path}' (step {step})")
    return restored["params"]


def build_ppo_network(
    model_type: str, num_actions: int, layer_size: int, config: dict
) -> Any:
    """Instantiate the correct PPO architecture.

    Args:
        model_type:  One of ``"ppo_rnn"``, ``"ppo_rnd"``, or ``"ppo"``.
        num_actions: Size of the discrete action space.
        layer_size:  Hidden layer width.
        config:      Training config (forwarded to ``ActorCriticRNN``).

    Returns:
        Flax module instance.
    """
    model_type = model_type.lower()
    if model_type == "ppo_rnn":
        return ActorCriticRNN(num_actions, config=config)
    if model_type == "ppo_rnd":
        return ActorCriticRND(num_actions, layer_size)
    return ActorCritic(num_actions, layer_size)


def load_ppo_agent(
    path: str,
    num_actions: int,
    obs_dim: int,
    layer_size: int,
    model_type: str,
    config: dict,
    num_envs: int = 1,
) -> PPOAgent:
    """Build network, load params, and return a :class:`PPOAgent`.

    Args:
        path:        Path to the Orbax checkpoint directory.
        num_actions: Size of the discrete action space.
        obs_dim:     Observation vector dimensionality.
        layer_size:  Hidden layer width.
        model_type:  One of ``"ppo_rnn"``, ``"ppo_rnd"``, or ``"ppo"``.
        config:      Training config dict.
        num_envs:    Number of parallel environments.

    Returns:
        A fully initialised :class:`PPOAgent`.
    """
    net = build_ppo_network(model_type, num_actions, layer_size, config)
    params = load_ppo_params(
        path,
        net,
        model_type,
        num_envs,
        (obs_dim,),
        layer_size,
        seed=int(config.get("SEED") or 0),
    )
    return PPOAgent(net, params, model_type, layer_size)


class PPOAgent:
    """Uniform interface over PPO-RNN / PPO / PPO-RND for action collection.

    Args:
        network:    Flax actor-critic module.
        params:     Loaded parameter pytree.
        model_type: One of ``"ppo_rnn"``, ``"ppo_rnd"``, or ``"ppo"``.
        layer_size: Hidden layer width (used for RNN hidden-state shape).
    """

    def __init__(
        self, network: Any, params: Any, model_type: str, layer_size: int = 512
    ) -> None:
        self.network = network
        self.params = params
        self.model_type = model_type.lower()
        self.layer_size = layer_size

    def init_hidden(self, batch_size: int) -> jnp.ndarray | None:
        """Return a zero hidden state for RNN models, else ``None``."""
        if self.model_type == "ppo_rnn":
            return jnp.zeros((batch_size, self.layer_size))
        return None

    def act(
        self,
        obs: jnp.ndarray,
        done: jnp.ndarray,
        hidden: jnp.ndarray | None,
        rng: jax.Array,
        temperature: float = 1.0,
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """Sample an action.

        Args:
            obs:         Observation array ``[B, obs_dim]``.
            done:        Episode-done flags ``[B]``.
            hidden:      RNN hidden state (``None`` for non-RNN models).
            rng:         PRNG key.
            temperature: Softmax temperature for sampling.

        Returns:
            ``(action, new_hidden)`` tuple.
        """
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

    def get_pi(
        self,
        obs: jnp.ndarray,
        done: jnp.ndarray | None = None,
        hidden: jnp.ndarray | None = None,
    ) -> tuple[Any, jnp.ndarray | None]:
        """Return the policy distribution (used in DAgger expert labelling).

        Args:
            obs:    Observation array ``[B, obs_dim]``.
            done:   Episode-done flags ``[B]`` (required for RNN models).
            hidden: RNN hidden state.

        Returns:
            ``(pi, new_hidden)`` tuple.
        """
        if self.model_type == "ppo_rnn":
            ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
            new_hidden, pi, _ = self.network.apply(self.params, hidden, ac_in)
            return pi, new_hidden
        if self.model_type == "ppo_rnd":
            pi, _, _ = self.network.apply(self.params, obs)
            return pi, hidden
        pi, _ = self.network.apply(self.params, obs)
        return pi, hidden
