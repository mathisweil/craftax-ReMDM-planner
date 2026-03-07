"""Tests for src/planners/utils.py — offline data utilities and model helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.models.denoiser import DenoisingTransformer

# We import the functions under test directly since they live in a file
# that also imports Craftax. We use the utils module path.
# Note: the planners.py file imports from `utils` (relative), so we import
# from the actual module path.
import importlib
import sys
import os

# Add project root to path so `utils` can be imported as planners.py does
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# =============================================================================
# _valid_window_mask
# =============================================================================

# Import only the pure utility functions that don't require Craftax
from src.planners.utils import _valid_window_mask, _sample_windows_from_chunk


class TestValidWindowMask:
    def test_no_dones_all_valid(self):
        dones = np.zeros((2, 10), dtype=bool)
        valid = _valid_window_mask(dones, plan_horizon=3)
        # Last 2 positions should be False (not enough room)
        assert valid[:, :-2].all()
        assert not valid[:, -2:].any()

    def test_done_blocks_window(self):
        dones = np.zeros((1, 10), dtype=bool)
        dones[0, 4] = True  # done at step 4
        valid = _valid_window_mask(dones, plan_horizon=3)
        # Positions 3, 4 should be invalid (window would include done at 4)
        assert not valid[0, 3]
        assert not valid[0, 4]
        # Position 2 should be valid: window [2,3,4) -> checks dones at 2,3 = False
        # Actually checks dones[2], dones[3] for a horizon of 3 starting at 2
        # dones[2]=F, dones[3]=F -> but we need to check if done appears in [t, t+H-1)
        # window [2, 3, 4]: dones[3]=F, dones[4]=T -> invalid
        # Let me re-check the logic
        # _valid_window_mask checks dones[t:t+plan_horizon-1] all False
        # For t=2, H=3: checks dones[2], dones[3] (offsets 0,1,2 but H-1=2 offsets)
        # offset 0: dones[2]=F, offset 1: dones[3]=F -> valid[0,2] could be True
        # Actually the shifted array at offset 1 checks dones[3] which is False
        # At offset 2 (not checked since range is plan_horizon-1=2, so offsets 0,1)
        # So valid[0,2] should be True
        assert valid[0, 2]
        # Position 5 should be valid (done is behind us)
        assert valid[0, 5]

    def test_plan_horizon_one(self):
        dones = np.zeros((1, 5), dtype=bool)
        dones[0, 2] = True
        valid = _valid_window_mask(dones, plan_horizon=1)
        # With H=1, no done checking needed (range(0) loop)
        # All positions should be valid (no tail trimming with H=1)
        assert valid[0, 0]
        assert valid[0, 1]
        # dones[2]=True, but with H=1 the loop range(0) does nothing
        # Actually offset range is range(plan_horizon - 1) = range(0) = empty
        # So no shifted checks, valid stays all True, and plan_horizon <= 1 means
        # no tail trimming either.
        assert valid.all()

    def test_all_dones_nothing_valid(self):
        dones = np.ones((1, 5), dtype=bool)
        valid = _valid_window_mask(dones, plan_horizon=2)
        assert not valid.any()

    def test_multiple_envs(self):
        dones = np.zeros((3, 8), dtype=bool)
        dones[1, 3] = True
        valid = _valid_window_mask(dones, plan_horizon=2)
        # env 0 and 2 should be all valid except last position
        assert valid[0, :-1].all()
        assert valid[2, :-1].all()
        # env 1 should have position 3 invalid
        assert not valid[1, 3]


# =============================================================================
# _sample_windows_from_chunk
# =============================================================================


class TestSampleWindowsFromChunk:
    def test_returns_correct_shapes(self):
        n_envs, T, obs_dim = 2, 20, 4
        plan_horizon, batch_size = 3, 8
        chunk_obs = np.random.randn(n_envs, T, obs_dim).astype(np.float32)
        chunk_acts = np.random.randint(0, 5, (n_envs, T)).astype(np.int32)
        chunk_dones = np.zeros((n_envs, T), dtype=bool)
        np_rng = np.random.default_rng(0)

        result = _sample_windows_from_chunk(
            chunk_obs, chunk_acts, chunk_dones, plan_horizon, batch_size, np_rng
        )
        assert result is not None
        obs_batch, act_batch = result
        assert obs_batch.shape == (batch_size, obs_dim)
        assert act_batch.shape == (batch_size, plan_horizon)

    def test_returns_none_when_no_valid(self):
        # All dones -> no valid windows
        chunk_obs = np.random.randn(1, 5, 4).astype(np.float32)
        chunk_acts = np.zeros((1, 5), dtype=np.int32)
        chunk_dones = np.ones((1, 5), dtype=bool)
        np_rng = np.random.default_rng(0)

        result = _sample_windows_from_chunk(
            chunk_obs, chunk_acts, chunk_dones, 3, 4, np_rng
        )
        assert result is None

    def test_replacement_when_fewer_valid(self):
        # Only 1 valid window but request batch_size=4 -> should use replacement
        chunk_obs = np.random.randn(1, 5, 4).astype(np.float32)
        chunk_acts = np.zeros((1, 5), dtype=np.int32)
        chunk_dones = np.zeros((1, 5), dtype=bool)
        chunk_dones[0, 2:] = True  # Only positions 0,1 no-done, but need H=3
        # Actually with H=3: valid positions need dones[t:t+2] all False
        # t=0: dones[0]=F, dones[1]=F -> valid
        # t=1: dones[1]=F, dones[2]=T -> invalid
        # So only 1 valid window
        np_rng = np.random.default_rng(0)
        result = _sample_windows_from_chunk(
            chunk_obs, chunk_acts, chunk_dones, 3, 4, np_rng
        )
        assert result is not None
        obs_batch, act_batch = result
        assert obs_batch.shape == (4, 4)  # batch_size=4, with replacement

    def test_act_batch_correct_horizon(self):
        chunk_obs = np.arange(20).reshape(1, 5, 4).astype(np.float32)
        chunk_acts = np.arange(5).reshape(1, 5).astype(np.int32)
        chunk_dones = np.zeros((1, 5), dtype=bool)
        np_rng = np.random.default_rng(42)

        result = _sample_windows_from_chunk(
            chunk_obs, chunk_acts, chunk_dones, 3, 2, np_rng
        )
        assert result is not None
        _, act_batch = result
        # Each action window should be contiguous
        for i in range(act_batch.shape[0]):
            diffs = np.diff(np.array(act_batch[i]))
            assert np.all(diffs == 1)
