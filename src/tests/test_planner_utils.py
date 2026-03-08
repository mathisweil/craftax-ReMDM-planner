"""Tests for src/planners/utils.py — checkpoint I/O, PPOAgent, and data utilities."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax.training.train_state import TrainState

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Craftax_Baselines uses bare imports (e.g. ``from wrappers import ...``).
_baselines_dir = os.path.join(_project_root, "Craftax_Baselines")
if _baselines_dir not in sys.path:
    sys.path.insert(0, _baselines_dir)

from src.planners.utils import (
    PPOAgent,
    _build_ppo_network,
    _detect_ppo_model_type,
    _load_checkpoint,
    _load_ppo_checkpoint,
    _restore_train_state,
    _save_model,
    _valid_window_mask,
    _sample_windows_from_chunk,
)

# ── Shared constants ────────────────────────────────────────────────────────
NUM_ACTIONS = 5
OBS_DIM = 8
LAYER_SIZE = 64  # small for speed


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture()
def _ppo_ts():
    """Real PPO TrainState for the plain ActorCritic architecture."""
    from Craftax_Baselines.models.actor_critic import ActorCritic
    net = ActorCritic(NUM_ACTIONS, LAYER_SIZE)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4, eps=1e-5))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


@pytest.fixture()
def _rnd_ts():
    """Real PPO-RND TrainState."""
    from Craftax_Baselines.models.rnd import ActorCriticRND
    net = ActorCriticRND(NUM_ACTIONS, LAYER_SIZE)
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4, eps=1e-5))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


@pytest.fixture()
def _rnn_ts():
    """Real PPO-RNN TrainState."""
    from Craftax_Baselines.ppo_rnn import ActorCriticRNN, ScannedRNN
    net = ActorCriticRNN(NUM_ACTIONS, config={"LAYER_SIZE": LAYER_SIZE})
    h = ScannedRNN.initialize_carry(1, LAYER_SIZE)
    params = net.init(
        jax.random.PRNGKey(0), h,
        (jnp.zeros((1, 1, OBS_DIM)), jnp.zeros((1, 1))),
    )
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-4, eps=1e-5))
    return TrainState.create(apply_fn=net.apply, params=params, tx=tx)


@contextmanager
def _mock_ckpt_manager(restored_item: Any = None, latest_step: int | None = 42):
    """Patch ``_make_ckpt_manager`` with a mock that returns *restored_item*."""
    mgr = MagicMock()
    mgr.latest_step.return_value = latest_step
    mgr.restore.return_value = restored_item
    mgr.__enter__ = lambda s: s
    mgr.__exit__ = MagicMock(return_value=False)
    with patch("src.planners.utils._make_ckpt_manager", return_value=mgr) as _:
        yield mgr


# ═══════════════════════════════════════════════════════════════════════════
# _restore_train_state
# ═══════════════════════════════════════════════════════════════════════════

class TestRestoreTrainState:
    def test_raises_when_no_checkpoint(self):
        with _mock_ckpt_manager(latest_step=None):
            with pytest.raises(FileNotFoundError, match="No valid checkpoint"):
                _restore_train_state("/fake/path", MagicMock())

    def test_returns_restored_object(self, _ppo_ts):
        with _mock_ckpt_manager(restored_item=_ppo_ts) as mgr:
            result = _restore_train_state("/fake/path", MagicMock())
        assert result is _ppo_ts
        mgr.restore.assert_called_once()

    def test_passes_abstract_ts_to_restore(self, _ppo_ts):
        sentinel = SimpleNamespace(tag="abstract")
        with _mock_ckpt_manager(restored_item=_ppo_ts) as mgr:
            _restore_train_state("/fake/path", sentinel)
        call_kwargs = mgr.restore.call_args
        assert call_kwargs[0][0] == 42  # step
        assert call_kwargs[1]["args"].item is sentinel


# ═══════════════════════════════════════════════════════════════════════════
# _build_ppo_network — abstract params pytree (no opt_state)
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPpoNetwork:
    @pytest.mark.parametrize("model_type", ["ppo", "ppo_rnd", "ppo_rnn"])
    def test_returns_abstract_params_pytree(self, model_type):
        """abstract_params must be a raw params dict (not a TrainState)."""
        _, abstract_params = _build_ppo_network(
            model_type, NUM_ACTIONS, OBS_DIM, LAYER_SIZE
        )
        # All leaves are ShapeDtypeStructs from jax.eval_shape
        leaves = jax.tree.leaves(abstract_params)
        assert len(leaves) > 0
        for leaf in leaves:
            assert hasattr(leaf, "shape") and hasattr(leaf, "dtype")

    @pytest.mark.parametrize("model_type", ["ppo", "ppo_rnd", "ppo_rnn"])
    def test_abstract_params_is_not_train_state(self, model_type):
        """Must be a raw params dict — no opt_state, so partial_restore works."""
        _, abstract_params = _build_ppo_network(
            model_type, NUM_ACTIONS, OBS_DIM, LAYER_SIZE
        )
        assert not isinstance(abstract_params, TrainState)

    @pytest.mark.parametrize("model_type", ["ppo", "ppo_rnd", "ppo_rnn"])
    def test_abstract_params_has_flax_params_key(self, model_type):
        """Flax stores weights under a 'params' sub-key."""
        _, abstract_params = _build_ppo_network(
            model_type, NUM_ACTIONS, OBS_DIM, LAYER_SIZE
        )
        # abstract_params is {'params': {'Dense_0': ..., ...}}
        assert isinstance(abstract_params, dict) or hasattr(abstract_params, "keys")
        assert "params" in abstract_params

    @pytest.mark.parametrize("model_type", ["ppo", "ppo_rnd", "ppo_rnn"])
    def test_first_layer_shape_matches_obs_dim(self, model_type):
        """Dense_0 kernel must have obs_dim as first axis."""
        _, abstract_params = _build_ppo_network(
            model_type, NUM_ACTIONS, OBS_DIM, LAYER_SIZE
        )
        kernel = abstract_params["params"]["Dense_0"]["kernel"]
        assert kernel.shape[0] == OBS_DIM

    def test_ppo_network_type(self):
        from Craftax_Baselines.models.actor_critic import ActorCritic
        net, _ = _build_ppo_network("ppo", NUM_ACTIONS, OBS_DIM, LAYER_SIZE)
        assert isinstance(net, ActorCritic)

    def test_ppo_rnd_network_type(self):
        from Craftax_Baselines.models.rnd import ActorCriticRND
        net, _ = _build_ppo_network("ppo_rnd", NUM_ACTIONS, OBS_DIM, LAYER_SIZE)
        assert isinstance(net, ActorCriticRND)

    def test_ppo_rnn_network_type(self):
        from Craftax_Baselines.ppo_rnn import ActorCriticRNN
        net, _ = _build_ppo_network("ppo_rnn", NUM_ACTIONS, OBS_DIM, LAYER_SIZE)
        assert isinstance(net, ActorCriticRNN)


# ═══════════════════════════════════════════════════════════════════════════
# _detect_ppo_model_type
# ═══════════════════════════════════════════════════════════════════════════

class TestDetectPpoModelType:
    def test_detects_rnn_from_filesystem(self, tmp_path):
        (tmp_path / "1000" / "default" / "params" / "ScannedRNN_0").mkdir(parents=True)
        assert _detect_ppo_model_type(str(tmp_path)) == "ppo_rnn"

    def test_detects_rnd_from_filesystem(self, tmp_path):
        (tmp_path / "1000" / "default" / "params" / "Dense_8").mkdir(parents=True)
        assert _detect_ppo_model_type(str(tmp_path)) == "ppo_rnd"

    def test_detects_ppo_from_filesystem(self, tmp_path):
        (tmp_path / "1000" / "default" / "params" / "Dense_0").mkdir(parents=True)
        assert _detect_ppo_model_type(str(tmp_path)) == "ppo"

    def test_falls_back_to_path_rnn(self):
        assert _detect_ppo_model_type("/some/ppo_rnn_checkpoint") == "ppo_rnn"

    def test_falls_back_to_path_rnd(self):
        assert _detect_ppo_model_type("/some/ppo_rnd_checkpoint") == "ppo_rnd"

    def test_defaults_to_ppo(self):
        assert _detect_ppo_model_type("/nonexistent/plain") == "ppo"

    def test_os_error_graceful(self):
        assert _detect_ppo_model_type("/dev/null/nope") == "ppo"


# ═══════════════════════════════════════════════════════════════════════════
# _load_ppo_checkpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadPpoCheckpoint:
    @pytest.mark.parametrize("model_type", ["ppo", "ppo_rnd", "ppo_rnn"])
    def test_returns_ppo_agent(self, model_type, _ppo_ts, _rnd_ts, _rnn_ts):
        ts_map = {"ppo": _ppo_ts, "ppo_rnd": _rnd_ts, "ppo_rnn": _rnn_ts}
        ts = ts_map[model_type]
        # restore returns {"params": params_dict} — matching PyTreeRestore output
        with _mock_ckpt_manager(restored_item={"params": ts.params}):
            agent = _load_ppo_checkpoint(
                "/fake", NUM_ACTIONS, OBS_DIM, LAYER_SIZE,
                model_type=model_type,
            )
        assert isinstance(agent, PPOAgent)
        assert agent.model_type == model_type
        assert agent.layer_size == LAYER_SIZE
        # agent.params must equal the params we put in the mock
        leaves_agent = jax.tree.leaves(agent.params)
        leaves_ts = jax.tree.leaves(ts.params)
        assert len(leaves_agent) == len(leaves_ts)
        for a, b in zip(leaves_agent, leaves_ts):
            np.testing.assert_array_equal(a, b)

    def test_raises_when_no_checkpoint(self):
        with _mock_ckpt_manager(latest_step=None):
            with pytest.raises(FileNotFoundError):
                _load_ppo_checkpoint(
                    "/fake", NUM_ACTIONS, OBS_DIM, LAYER_SIZE, model_type="ppo"
                )

    def test_uses_partial_restore(self):
        """Verify PyTreeRestore is called with partial_restore=True."""
        import orbax.checkpoint as ocp
        from Craftax_Baselines.models.actor_critic import ActorCritic
        net = ActorCritic(NUM_ACTIONS, LAYER_SIZE)
        params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
        with _mock_ckpt_manager(restored_item={"params": params}) as mgr:
            _load_ppo_checkpoint("/fake", NUM_ACTIONS, OBS_DIM, LAYER_SIZE, model_type="ppo")
        restore_args = mgr.restore.call_args[1]["args"]
        assert isinstance(restore_args, ocp.args.PyTreeRestore)
        assert restore_args.partial_restore is True

    def test_restore_item_contains_only_params(self):
        """Item passed to PyTreeRestore must be {'params': ...} — no opt_state."""
        import orbax.checkpoint as ocp
        from Craftax_Baselines.models.actor_critic import ActorCritic
        net = ActorCritic(NUM_ACTIONS, LAYER_SIZE)
        params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS_DIM)))
        with _mock_ckpt_manager(restored_item={"params": params}) as mgr:
            _load_ppo_checkpoint("/fake", NUM_ACTIONS, OBS_DIM, LAYER_SIZE, model_type="ppo")
        restore_args = mgr.restore.call_args[1]["args"]
        assert set(restore_args.item.keys()) == {"params"}


# ═══════════════════════════════════════════════════════════════════════════
# PPOAgent.apply — unified interface
# ═══════════════════════════════════════════════════════════════════════════

class TestPPOAgentApply:
    def test_ppo_returns_pi_value_none(self, _ppo_ts):
        from Craftax_Baselines.models.actor_critic import ActorCritic
        net = ActorCritic(NUM_ACTIONS, LAYER_SIZE)
        agent = PPOAgent(net, _ppo_ts.params, "ppo", LAYER_SIZE)
        obs = jnp.ones((2, OBS_DIM))
        pi, value, hidden = agent.apply(_ppo_ts.params, obs)
        assert pi.logits.shape == (2, NUM_ACTIONS)
        assert value.shape == (2,)
        assert hidden is None

    def test_ppo_rnd_returns_extrinsic_value(self, _rnd_ts):
        from Craftax_Baselines.models.rnd import ActorCriticRND
        net = ActorCriticRND(NUM_ACTIONS, LAYER_SIZE)
        agent = PPOAgent(net, _rnd_ts.params, "ppo_rnd", LAYER_SIZE)
        obs = jnp.ones((2, OBS_DIM))
        pi, value, hidden = agent.apply(_rnd_ts.params, obs)
        assert pi.logits.shape == (2, NUM_ACTIONS)
        assert value.shape == (2,)
        assert hidden is None

    def test_ppo_rnn_returns_hidden(self, _rnn_ts):
        from Craftax_Baselines.ppo_rnn import ActorCriticRNN, ScannedRNN
        net = ActorCriticRNN(NUM_ACTIONS, config={"LAYER_SIZE": LAYER_SIZE})
        agent = PPOAgent(net, _rnn_ts.params, "ppo_rnn", LAYER_SIZE)
        h = ScannedRNN.initialize_carry(2, LAYER_SIZE)
        obs = jnp.ones((2, OBS_DIM))
        done = jnp.zeros((2,), dtype=bool)
        pi, value, new_h = agent.apply(_rnn_ts.params, obs, hidden=h, done=done)
        assert pi.logits.shape == (1, 2, NUM_ACTIONS)  # [T=1, B, A]
        assert value.shape == (2,)
        assert new_h is not None
        assert new_h.shape == h.shape

    def test_ppo_rnn_asserts_without_hidden(self, _rnn_ts):
        from Craftax_Baselines.ppo_rnn import ActorCriticRNN
        net = ActorCriticRNN(NUM_ACTIONS, config={"LAYER_SIZE": LAYER_SIZE})
        agent = PPOAgent(net, _rnn_ts.params, "ppo_rnn", LAYER_SIZE)
        with pytest.raises(AssertionError, match="hidden and done"):
            agent.apply(_rnn_ts.params, jnp.ones((2, OBS_DIM)))


class TestPPOAgentInitHidden:
    def test_mlp_returns_none(self):
        agent = PPOAgent(MagicMock(), {}, "ppo", LAYER_SIZE)
        assert agent.init_hidden(4) is None

    def test_rnd_returns_none(self):
        agent = PPOAgent(MagicMock(), {}, "ppo_rnd", LAYER_SIZE)
        assert agent.init_hidden(4) is None

    def test_rnn_returns_array(self):
        from Craftax_Baselines.ppo_rnn import ActorCriticRNN
        net = ActorCriticRNN(NUM_ACTIONS, config={"LAYER_SIZE": LAYER_SIZE})
        agent = PPOAgent(net, {}, "ppo_rnn", LAYER_SIZE)
        h = agent.init_hidden(4)
        assert h is not None
        assert h.shape == (4, LAYER_SIZE)


# ═══════════════════════════════════════════════════════════════════════════
# _load_checkpoint (diffusion model)
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadCheckpoint:
    def test_returns_params_dict(self, small_config):
        from src.planners.utils import _build_model, _init_model_params, _create_train_state
        model = _build_model(small_config, small_config["NUM_ACTIONS"])
        params = _init_model_params(
            model, jax.random.PRNGKey(0),
            small_config["OBS_DIM"], small_config["PLAN_HORIZON"],
        )
        ts = _create_train_state(
            model, params, small_config["LR"], small_config["MAX_GRAD_NORM"],
        )
        with _mock_ckpt_manager(restored_item=ts):
            loaded = _load_checkpoint(small_config, model, small_config["OBS_DIM"], "/fake")

        loaded_leaves = jax.tree.leaves(loaded)
        param_leaves = jax.tree.leaves(params)
        assert len(loaded_leaves) == len(param_leaves)
        for a, b in zip(loaded_leaves, param_leaves):
            assert a.shape == b.shape
            assert a.dtype == b.dtype

    def test_raises_on_missing_checkpoint(self, small_config):
        from src.planners.utils import _build_model
        model = _build_model(small_config, small_config["NUM_ACTIONS"])
        with _mock_ckpt_manager(latest_step=None):
            with pytest.raises(FileNotFoundError):
                _load_checkpoint(small_config, model, small_config["OBS_DIM"], "/fake")


# ═══════════════════════════════════════════════════════════════════════════
# _save_model
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveModel:
    def test_calls_save_and_wait(self, _ppo_ts):
        config = {"USE_WANDB": False, "NUM_TRAIN_STEPS": 100}
        with _mock_ckpt_manager() as mgr:
            _save_model(_ppo_ts, config, "test_dir")
        mgr.save.assert_called_once()
        mgr.wait_until_finished.assert_called_once()

    def test_step_from_config(self, _ppo_ts):
        config = {"USE_WANDB": False, "NUM_TRAIN_STEPS": 999}
        with _mock_ckpt_manager() as mgr:
            _save_model(_ppo_ts, config, "test_dir")
        assert mgr.save.call_args[0][0] == 999

    def test_fallback_step_key(self, _ppo_ts):
        config = {"USE_WANDB": False, "NUM_UPDATES": 500}
        with _mock_ckpt_manager() as mgr:
            _save_model(_ppo_ts, config, "test_dir")
        assert mgr.save.call_args[0][0] == 500


# ═══════════════════════════════════════════════════════════════════════════
# _valid_window_mask
# ═══════════════════════════════════════════════════════════════════════════

class TestValidWindowMask:
    def test_no_dones_all_valid(self):
        dones = np.zeros((2, 10), dtype=bool)
        valid = _valid_window_mask(dones, plan_horizon=3)
        assert valid[:, :-2].all()
        assert not valid[:, -2:].any()

    def test_done_blocks_window(self):
        dones = np.zeros((1, 10), dtype=bool)
        dones[0, 4] = True
        valid = _valid_window_mask(dones, plan_horizon=3)
        assert not valid[0, 3]
        assert not valid[0, 4]
        assert valid[0, 2]
        assert valid[0, 5]

    def test_plan_horizon_one(self):
        dones = np.zeros((1, 5), dtype=bool)
        dones[0, 2] = True
        assert _valid_window_mask(dones, plan_horizon=1).all()

    def test_all_dones_nothing_valid(self):
        assert not _valid_window_mask(np.ones((1, 5), dtype=bool), 2).any()

    def test_multiple_envs(self):
        dones = np.zeros((3, 8), dtype=bool)
        dones[1, 3] = True
        valid = _valid_window_mask(dones, plan_horizon=2)
        assert valid[0, :-1].all()
        assert valid[2, :-1].all()
        assert not valid[1, 3]


# ═══════════════════════════════════════════════════════════════════════════
# _sample_windows_from_chunk
# ═══════════════════════════════════════════════════════════════════════════

class TestSampleWindowsFromChunk:
    def test_returns_correct_shapes(self):
        obs = np.random.randn(2, 20, 4).astype(np.float32)
        acts = np.random.randint(0, 5, (2, 20)).astype(np.int32)
        dones = np.zeros((2, 20), dtype=bool)
        result = _sample_windows_from_chunk(obs, acts, dones, 3, 8, np.random.default_rng(0))
        assert result is not None
        assert result[0].shape == (8, 4)
        assert result[1].shape == (8, 3)

    def test_returns_none_when_no_valid(self):
        result = _sample_windows_from_chunk(
            np.zeros((1, 5, 4), np.float32),
            np.zeros((1, 5), np.int32),
            np.ones((1, 5), dtype=bool),
            3, 4, np.random.default_rng(0),
        )
        assert result is None

    def test_replacement_when_fewer_valid(self):
        dones = np.zeros((1, 5), dtype=bool)
        dones[0, 2:] = True
        result = _sample_windows_from_chunk(
            np.zeros((1, 5, 4), np.float32),
            np.zeros((1, 5), np.int32),
            dones, 3, 4, np.random.default_rng(0),
        )
        assert result is not None
        assert result[0].shape == (4, 4)

    def test_act_batch_contiguous(self):
        result = _sample_windows_from_chunk(
            np.arange(20).reshape(1, 5, 4).astype(np.float32),
            np.arange(5).reshape(1, 5).astype(np.int32),
            np.zeros((1, 5), dtype=bool),
            3, 2, np.random.default_rng(42),
        )
        assert result is not None
        for row in np.array(result[1]):
            assert np.all(np.diff(row) == 1)
