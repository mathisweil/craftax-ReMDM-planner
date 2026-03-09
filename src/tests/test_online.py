"""Tests for src/planners/online.py."""

from unittest.mock import MagicMock, patch

from src.planners.online import run_online, make_train_online


def test_make_train_online_returns_callable(small_config):
    small_config.update({
        "NUM_STEPS": 4,
        "REPLAN_EVERY": 2,
        "NUM_ENVS": 2,
        "NUM_UPDATES": 1,
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "DIFFUSION_STEPS": 1,
        "DIFFUSION_SCHEDULE": "linear",
        "REMASK_STRATEGY": "autoregressive",
        "ETA": 1.0,
        "NUM_ACTIONS": 5,
        "OBS_DIM": 8,
    })

    with patch("src.planners.online._make_env_stack") as mock_stack, \
            patch("src.planners.online._build_model") as mock_build:
        mock_env = MagicMock()
        mock_env.reset.return_value = (None, None)
        mock_stack.return_value = (mock_env, {})

        mock_build.return_value = MagicMock()

        train_fn = make_train_online(small_config)
        assert callable(train_fn)


@patch("src.planners.online.wandb")
@patch("src.planners.online._save_model")
@patch("src.planners.online.make_train_online")
@patch("src.planners.online.make_craftax_env_from_name")
def test_run_online_execution(mock_make_env, mock_make_train, mock_save, mock_wandb, small_config):
    small_config.update({
        "ENV_NAME": "Craftax-Symbolic-v1",
        "NUM_STEPS": 4,
        "NUM_ENVS": 2,
        "NUM_UPDATES": 2,
        "SEED": 0,
        "USE_WANDB": True,
        "SAVE_POLICY": True,
        "NUM_REPEATS": 1,
        "DEBUG": False,
    })

    mock_env = MagicMock()
    mock_env.default_params = {}
    mock_env.action_space.return_value.n = 5
    mock_env.observation_space.return_value.shape = (8,)
    mock_make_env.return_value = mock_env

    mock_train_fn = MagicMock()
    mock_train_fn.return_value = {
        "metrics": {"diffusion_loss": [0.5, 0.4]},
        "runner_state": (MagicMock(),)
    }
    mock_make_train.return_value = mock_train_fn

    with patch("jax.jit", side_effect=lambda f: f):
        run_online(small_config)

    mock_wandb.init.assert_called_once()
    mock_wandb.log.assert_called()
    mock_save.assert_called_once()