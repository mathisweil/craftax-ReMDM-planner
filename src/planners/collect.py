import pathlib
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from .utils import _load_ppo_checkpoint, _make_env_stack

def collect_offline_data(config: Dict[str, Any]) -> None:
    assert config.get("PPO_CHECKPOINT_PATH"), (
        "--ppo_checkpoint_path is required for --mode collect.\n"
        "Train a PPO agent first with ppo_rnn.py or ppo_rnd.py."
    )

    env = make_craftax_env_from_name(config["ENV_NAME"], True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    num_envs: int = config["COLLECT_NUM_ENVS"]
    num_iters: int = config["COLLECT_NUM_STEPS"] // num_envs

    ppo_agent = _load_ppo_checkpoint(
        config["PPO_CHECKPOINT_PATH"], num_actions, obs_dim,
        config.get("LAYER_SIZE", 512),
        model_type=config.get("PPO_MODEL_TYPE"),
    )
    is_rnn = ppo_agent.model_type == "ppo_rnn"
    env_w, _ = _make_env_stack(config, num_envs)

    rng = jax.random.PRNGKey(config["SEED"])
    rng, env_rng, collect_rng = jax.random.split(rng, 3)
    obs, env_state = env_w.reset(env_rng, env_params)
    done = jnp.zeros(num_envs, dtype=bool)
    hidden = ppo_agent.init_hidden(num_envs)

    def _step_fn(carry, _):
        rng, env_state, obs, done, hidden = carry
        rng, k1, k2 = jax.random.split(rng, 3)
        if is_rnn:
            pi, _, new_hidden = ppo_agent.apply(ppo_agent.params, obs, hidden=hidden, done=done)
        else:
            pi, _, _ = ppo_agent.apply(ppo_agent.params, obs)
            new_hidden = hidden
        action = pi.sample(seed=k1)
        obs_next, env_state, _, done_next, _ = env_w.step(k2, env_state, action, env_params)
        return (rng, env_state, obs_next, done_next, new_hidden), (obs, action, done)

    rollout_fn = jax.jit(lambda c: jax.lax.scan(_step_fn, c, None, length=num_iters))
    _, (obs_arr, act_arr, done_arr) = rollout_fn((collect_rng, env_state, obs, done, hidden))

    obs_arr, act_arr, done_arr = (
        np.array(obs_arr).transpose(1, 0, 2),
        np.array(act_arr).transpose(1, 0),
        np.array(done_arr).transpose(1, 0),
    )

    out_path = config["OFFLINE_DATA_PATH"]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, obs=obs_arr, actions=act_arr, dones=done_arr)
    total = obs_arr.shape[0] * obs_arr.shape[1]
    print(f"Saved {obs_arr.shape[0]}x{obs_arr.shape[1]} transitions ({total:,} total) to '{out_path}'")

def run_collect(config: Dict[str, Any]) -> None:
    collect_offline_data(config)