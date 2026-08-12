"""Config layering: preset resolution, cluster parity, delta-only presets."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import main
from experiments.rl_finetuning.run_ablations import (
    _apply_fast_overrides,
    _load_ablation_config,
    _to_upper,
)

_ROOT = Path(__file__).resolve().parents[1]
_CONFIGS = _ROOT / "configs"
_DEFAULTS = _CONFIGS / "defaults.yaml"
_ABL_CONFIGS = _ROOT / "experiments" / "rl_finetuning" / "configs"
_ABL_DEFAULT = _ABL_CONFIGS / "ablations_default.yaml"

_DELTA_EXEMPT = {"defaults.yaml"}


# ---------------------------------------------------------------------------
# build_config wiring
# ---------------------------------------------------------------------------


def _build(argv: list[str]) -> dict:
    import sys

    saved = sys.argv
    sys.argv = ["main.py", *argv]
    try:
        return main.build_config()
    finally:
        sys.argv = saved


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        ("configs/final_craftax_ucl.yaml", {"NUM_ENVS": 448, "SEED": 42}),
        ("configs/final_craftax_qmul.yaml", {"NUM_ENVS": 64, "SEED": 43}),
        ("configs/final_classic_ucl.yaml", {"NUM_ENVS": 512, "SEED": 42}),
        ("configs/final_classic_qmul.yaml", {"NUM_ENVS": 96, "SEED": 43}),
    ],
)
def test_preset_resolves_over_the_defaults(preset, expected) -> None:
    config = _build(["--mode", "online", "--config", preset])
    for key, value in expected.items():
        assert config[key] == value
    # Comes from defaults.yaml, which holds the recipe, not from the preset.
    assert config["D_MODEL"] == 384
    assert config["N_LAYERS"] == 6


@pytest.mark.parametrize("family", ["classic", "craftax"])
def test_cluster_siblings_differ_only_in_num_envs_and_seed(family) -> None:
    """No inheritance links the pair, so this is the only thing holding them
    together. For craftax it matters most: eleven keys are duplicated verbatim
    across the two files and would otherwise drift apart silently."""
    ucl = _build(["--mode", "online", "--config", f"configs/final_{family}_ucl.yaml"])
    qmul = _build(["--mode", "online", "--config", f"configs/final_{family}_qmul.yaml"])
    differing = {k for k in ucl.keys() & qmul.keys() if ucl[k] != qmul[k]}
    assert differing == {"NUM_ENVS", "SEED"}, differing


# ---------------------------------------------------------------------------
# ablation configs layer onto ablations_default.yaml
# ---------------------------------------------------------------------------


def test_none_path_returns_empty() -> None:
    assert _load_ablation_config(None) == {}


def test_base_loads_as_itself() -> None:
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    assert _load_ablation_config(str(_ABL_DEFAULT)) == base


def test_machine_config_inherits_the_full_base_key_set() -> None:
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    merged = _load_ablation_config(
        str(_ABL_CONFIGS / "ablations_final_craftax_ucl.yaml")
    )
    assert set(merged) == set(base)
    assert merged["d_model"] == 384  # own delta
    assert merged["lr"] == base["lr"]  # inherited


def test_ablation_suite_sees_the_main_defaults() -> None:
    """run_ablations layers configs/defaults.yaml under the ablations config.

    Keys that live only there would otherwise be absent from the suite's
    config entirely. jax_compilation_cache_dir is the one with teeth: the
    suite reads it to enable the persistent XLA cache, so dropping
    defaults.yaml from that merge would silently disable caching rather than
    fail, and every (ablation, seed) would recompile from scratch.
    """
    defaults = yaml.safe_load(_DEFAULTS.read_text())
    abl = yaml.safe_load(_ABL_DEFAULT.read_text())
    assert "jax_compilation_cache_dir" in defaults
    assert "jax_compilation_cache_dir" not in abl

    merged = {**defaults, **_load_ablation_config(str(_ABL_DEFAULT))}
    assert "jax_compilation_cache_dir" in merged


# ---------------------------------------------------------------------------
# the fast overlay is applied raw, not layered
# ---------------------------------------------------------------------------


def test_fast_overlay_preserves_machine_config_deltas() -> None:
    """Layered like a config, the base would drag these back to its own values."""
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    path = _ABL_CONFIGS / "ablations_final_craftax_ucl.yaml"
    own = yaml.safe_load(path.read_text())
    fast = yaml.safe_load((_ABL_CONFIGS / "ablations_fast.yaml").read_text())

    merged = _apply_fast_overrides(_to_upper(_load_ablation_config(str(path))))

    for key, value in own.items():
        if key in fast:
            continue
        assert merged[key.upper()] == value, key
        assert value != base[key], f"{key} is not a delta any more"


def test_fast_overlay_shrinks_only_its_own_keys() -> None:
    path = _ABL_CONFIGS / "ablations_final_craftax_ucl.yaml"
    before = _to_upper(_load_ablation_config(str(path)))
    after = _apply_fast_overrides(before)
    fast = yaml.safe_load((_ABL_CONFIGS / "ablations_fast.yaml").read_text())

    changed = {k for k in before if before[k] != after[k]}
    assert changed <= {k.upper() for k in fast}


# ---------------------------------------------------------------------------
# delta-only invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "preset",
    sorted(p.name for p in _CONFIGS.glob("*.yaml") if p.name not in _DELTA_EXEMPT),
)
def test_preset_restates_no_inherited_value(preset) -> None:
    """A preset may only hold keys whose value differs from defaults.yaml.

    Since defaults.yaml is the Classic recipe rather than a neutral baseline,
    a preset that predates it keeps explicit pins, which are genuine deltas and
    pass. What this catches is the reverse: a preset restating a value it would
    inherit anyway, which then silently stops tracking the recipe.
    """
    defaults = yaml.safe_load(_DEFAULTS.read_text())
    raw = yaml.safe_load((_CONFIGS / preset).read_text()) or {}
    restated = {k: v for k, v in raw.items() if k in defaults and defaults[k] == v}
    assert not restated, f"{preset} restates inherited values: {restated}"


@pytest.mark.parametrize(
    "preset",
    sorted(
        p.name
        for p in _ABL_CONFIGS.glob("*.yaml")
        if p.name not in {"ablations_default.yaml", "ablations_fast.yaml"}
    ),
)
def test_ablation_config_restates_no_inherited_value(preset) -> None:
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    raw = yaml.safe_load((_ABL_CONFIGS / preset).read_text()) or {}
    restated = {k: v for k, v in raw.items() if k in base and base[k] == v}
    assert not restated, f"{preset} restates inherited values: {restated}"
