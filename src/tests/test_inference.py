"""Tests for src/planners/inference.py."""

import numpy as np
from unittest.mock import MagicMock, patch

from src.planners.inference import run_inference


@patch("src.planners.inference.PlannerWrapper")
@patch("src.planners.inference.wandb")
@patch("src.planners.inference.make_craftax_env_from_name")
@patch("src.planners.inference._load_checkpoint")
def test_run_inference(mock_load_ckpt, mock_make_env, mock_wandb, mock_planner_wrapper, small_config):
    small_config.update({
        "ENV_NAME": "Craftax-Symbolic-v1",
        "NUM_ENVS": 2,
        "PLAN_HORIZON": 4,
        "REPLAN_EVERY": 2,
        "DIFFUSION_STEPS": 1,
        "DIFFUSION_SCHEDULE": "linear",
        "REMASK_STRATEGY": "autoregressive",
        "ETA": 1.0,
        "EVAL_STEPS": 5,
        "CHECKPOINT_PATH": "/fake.ckpt",
        "SEED": 0,
        "USE_WANDB": True,
    })

    # Safely mock the base environment and ensure env_params is hashable (None)
    mock_env = MagicMock()
    mock_env.default_params = None
    mock_env.action_space.return_value.n = 5
    mock_env.observation_space.return_value.shape = (8,)
    mock_make_env.return_value = mock_env

    mock_load_ckpt.return_value = {}

    # Mock the PlannerWrapper stack to bypass JAX environment compilation
    mock_env_w = MagicMock()
    mock_env_w.reset.return_value = (np.zeros((2, 8)), None)
    mock_planner_wrapper.return_value = mock_env_w

    dummy_rewards = np.zeros(5)
    dummy_dones = np.zeros((5, 2), dtype=bool)
    dummy_infos = {
        "returned_episode_returns": np.array([10.0, 20.0]),
        "returned_episode": np.array([True, True]),
        "achievement_wood": np.array([1.0, 0.0])
    }

    # Disable JIT tracing and scan loop execution
    with patch("jax.jit", side_effect=lambda f: f), \
         patch("jax.lax.scan", return_value=(None, (dummy_rewards, dummy_dones, dummy_infos))):
        run_inference(small_config)

    mock_wandb.log.assert_called_once()
    log_args = mock_wandb.log.call_args[0][0]

    assert "eval/mean_return" in log_args
    assert "eval/achievement_wood" in log_args
    assert log_args["eval/completed_episodes"] == 2