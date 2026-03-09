"""Tests for main.py — CLI configuration, imports, and mode validation."""

from __future__ import annotations

import pytest


class TestMainImports:
    def test_schedule_map_keys(self):
        from main import SCHEDULE_MAP
        assert "cosine" in SCHEDULE_MAP
        assert "linear" in SCHEDULE_MAP

    def test_strategy_map_imported(self):
        from src.models.remdm import STRATEGY_MAP
        assert isinstance(STRATEGY_MAP, dict)
        assert len(STRATEGY_MAP) == 3

    def test_run_functions_importable(self):
        from src.planners.planners import run_collect, run_offline, run_online, run_inference
        assert callable(run_collect)
        assert callable(run_offline)
        assert callable(run_online)
        assert callable(run_inference)


class TestScheduleMapConsistency:
    """Verify that SCHEDULE_MAP in main.py and planners.py agree."""

    def test_same_keys(self):
        from main import SCHEDULE_MAP as main_map
        from src.planners.planners import SCHEDULE_MAP as planner_map
        assert set(main_map.keys()) == set(planner_map.keys())

    def test_same_functions(self):
        from main import SCHEDULE_MAP as main_map
        from src.planners.planners import SCHEDULE_MAP as planner_map
        for key in main_map:
            assert main_map[key] is planner_map[key]
