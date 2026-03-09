"""Tests for src/planners/collect.py."""

import os
import numpy as np
from unittest.mock import MagicMock, patch

from src.planners.collect import collect_offline_data


@patch("src.planners.collect.make_craftax_env_from_name")
@patch("src.planners.collect._load_ppo_checkpoint")
@patch("src.planners.collect._make_env_stack")
def test_collect_writes_npz(mock_make_env_stack, mock_load_ppo, mock_make_env, tmp_path):
    out_path = str(tmp_path / "test_data.npz")
    config = {
        "ENV_NAME": "Craftax-Symbolic-v1",
        "PPO_CHECKPOINT_PATH": "/fake",
        "COLLECT_NUM_ENVS": 2,
        "COLLECT_NUM_STEPS": 4,
        "SEED": 0,
        "OFFLINE_DATA_PATH": out_path,
    }

    mock_env = MagicMock()
    mock_env.default_params = {}
    mock_env.action_space.return_value.n = 5
    mock_env.observation_space.return_value.shape = (8,)
    mock_make_env.return_value = mock_env

    mock_ppo_agent = MagicMock()
    mock_ppo_agent.model_type = "ppo"
    mock_ppo_agent.init_hidden.return_value = None
    mock_load_ppo.return_value = mock_ppo_agent

    mock_env_w = MagicMock()
    mock_env_w.reset.return_value = (np.zeros((2, 8)), {})
    mock_make_env_stack.return_value = (mock_env_w, {})

    # Bypass JAX scan to simulate a completed rollout
    # Shapes output by scan: (steps, envs, ...)
    dummy_obs = np.zeros((2, 2, 8))
    dummy_acts = np.zeros((2, 2))
    dummy_dones = np.zeros((2, 2), dtype=bool)

    with patch("jax.jit", side_effect=lambda f: f), \
         patch("jax.lax.scan", return_value=(None, (dummy_obs, dummy_acts, dummy_dones))):
        collect_offline_data(config)

    assert os.path.exists(out_path)
    data = np.load(out_path)
    assert "obs" in data
    assert "actions" in data
    assert "dones" in data

    # Verify the function transposed the arrays correctly: (envs, steps, ...)
    assert data["obs"].shape == (2, 2, 8)
    assert data["actions"].shape == (2, 2)