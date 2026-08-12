"""Config inheritance: extends resolution, override rules, delta-only presets."""

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


def _write(directory: Path, name: str, body: str) -> Path:
    path = directory / name
    path.write_text(body)
    return path


# ---------------------------------------------------------------------------
# extends resolution (main.py)
# ---------------------------------------------------------------------------


def test_chain_is_base_first(tmp_path) -> None:
    _write(tmp_path, "base.yaml", "lr: 1.0\n")
    child = _write(tmp_path, "child.yaml", "extends: base.yaml\nlr: 2.0\n")
    chain = main._load_config_chain(child)
    assert [p.name for p, _ in chain] == ["base.yaml", "child.yaml"]


def test_three_deep_chain_child_wins(tmp_path) -> None:
    _write(tmp_path, "g.yaml", "num_envs: 1\nlr: 9.0\nnum_steps: 3\n")
    _write(tmp_path, "m.yaml", "extends: g.yaml\nnum_envs: 2\nlr: 8.0\n")
    _write(tmp_path, "c.yaml", "extends: m.yaml\nnum_envs: 3\n")

    merged: dict = {}
    for _, raw in main._load_config_chain(tmp_path / "c.yaml"):
        merged.update({k: v for k, v in raw.items() if k != "extends"})
    assert merged == {"num_envs": 3, "lr": 8.0, "num_steps": 3}


def test_no_extends_yields_single_entry(tmp_path) -> None:
    cfg = _write(tmp_path, "solo.yaml", "lr: 1.0\n")
    assert len(main._load_config_chain(cfg)) == 1


def test_self_cycle_raises_naming_the_loop(tmp_path) -> None:
    cfg = _write(tmp_path, "loop.yaml", "extends: loop.yaml\n")
    with pytest.raises(ValueError, match="loop.yaml"):
        main._load_config_chain(cfg)


def test_two_file_cycle_raises_naming_both(tmp_path) -> None:
    _write(tmp_path, "a.yaml", "extends: b.yaml\n")
    _write(tmp_path, "b.yaml", "extends: a.yaml\n")
    with pytest.raises(ValueError) as excinfo:
        main._load_config_chain(tmp_path / "a.yaml")
    assert "a.yaml" in str(excinfo.value)
    assert "b.yaml" in str(excinfo.value)


def test_missing_base_raises_naming_both_files(tmp_path) -> None:
    cfg = _write(tmp_path, "orphan.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        main._load_config_chain(cfg)
    assert "nope.yaml" in str(excinfo.value)
    assert "orphan.yaml" in str(excinfo.value)


def test_extends_resolves_relative_to_the_declaring_file(tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    _write(tmp_path, "base.yaml", "lr: 1.0\n")
    cfg = _write(tmp_path / "sub", "child.yaml", "extends: ../base.yaml\n")
    assert [p.name for p, _ in main._load_config_chain(cfg)] == [
        "base.yaml",
        "child.yaml",
    ]


def test_absolute_extends_is_accepted(tmp_path) -> None:
    cfg = _write(tmp_path, "abs.yaml", f"extends: {_DEFAULTS}\nnum_envs: 42\n")
    chain = main._load_config_chain(cfg)
    assert [p.name for p, _ in chain] == ["defaults.yaml", "abs.yaml"]


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
def test_preset_resolves_through_its_chain(preset, expected) -> None:
    config = _build(["--mode", "online", "--config", preset])
    for key, value in expected.items():
        assert config[key] == value
    # Inherited from the QMUL base rather than restated in the UCL file.
    assert config["D_MODEL"] == 384
    assert config["N_LAYERS"] == 6


def test_extends_never_reaches_the_config() -> None:
    config = _build(["--mode", "online", "--config", "configs/final_craftax_ucl.yaml"])
    assert not any(k.lower() == "extends" for k in config)


def test_extends_rejected_as_cli_override() -> None:
    with pytest.raises(ValueError, match="extends"):
        _build(["--mode", "smoke", "--override", "extends=defaults.yaml"])


def test_ucl_and_qmul_siblings_differ_only_in_num_envs_and_seed() -> None:
    """The invariant the extends link exists to enforce."""
    for family in ("craftax", "classic"):
        ucl = _build(
            ["--mode", "online", "--config", f"configs/final_{family}_ucl.yaml"]
        )
        qmul = _build(
            ["--mode", "online", "--config", f"configs/final_{family}_qmul.yaml"]
        )
        differing = {k for k in ucl.keys() & qmul.keys() if ucl[k] != qmul[k]}
        assert differing == {"NUM_ENVS", "SEED"}, differing


# ---------------------------------------------------------------------------
# extends resolution (run_ablations.py)
# ---------------------------------------------------------------------------


def test_none_path_returns_empty() -> None:
    assert _load_ablation_config(None) == {}


def test_empty_extends_opts_out(tmp_path) -> None:
    cfg = _write(tmp_path, "child.yaml", "extends:\nbatch_size: 7\n")
    assert _load_ablation_config(str(cfg)) == {"batch_size": 7}


def test_base_does_not_self_extend() -> None:
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


def test_ablation_self_cycle_raises(tmp_path) -> None:
    cfg = _write(tmp_path, "loop.yaml", "extends: loop.yaml\n")
    with pytest.raises(ValueError, match="loop.yaml"):
        _load_ablation_config(str(cfg))


def test_ablation_missing_base_raises_naming_both_files(tmp_path) -> None:
    cfg = _write(tmp_path, "orphan.yaml", "extends: nope.yaml\n")
    with pytest.raises(FileNotFoundError) as excinfo:
        _load_ablation_config(str(cfg))
    assert "nope.yaml" in str(excinfo.value)
    assert "orphan.yaml" in str(excinfo.value)


def test_ablation_suite_sees_the_main_defaults() -> None:
    """run_ablations layers configs/defaults.yaml under the ablations config.

    Keys that live only there would otherwise be absent from the suite's
    config entirely. jax_compilation_cache_dir is the one with teeth: the
    suite reads it to enable the persistent XLA cache, so dropping
    defaults.yaml from the chain would silently disable caching rather than
    fail, and every (ablation, seed) would recompile from scratch.
    """
    defaults = yaml.safe_load(_DEFAULTS.read_text())
    abl = yaml.safe_load(_ABL_DEFAULT.read_text())
    assert "jax_compilation_cache_dir" in defaults
    assert "jax_compilation_cache_dir" not in abl

    merged = {**defaults, **_load_ablation_config(str(_ABL_DEFAULT))}
    assert "jax_compilation_cache_dir" in merged


# ---------------------------------------------------------------------------
# the fast overlay stays outside the chain
# ---------------------------------------------------------------------------


def test_fast_overlay_carries_no_extends() -> None:
    """It is applied raw, so an extends key would leak into the namespace."""
    raw = yaml.safe_load((_ABL_CONFIGS / "ablations_fast.yaml").read_text())
    assert "extends" not in raw


def test_fast_overlay_preserves_machine_config_deltas() -> None:
    """Through the extends chain the base would drag these back to its own values."""
    base = yaml.safe_load(_ABL_DEFAULT.read_text())
    path = _ABL_CONFIGS / "ablations_final_craftax_ucl.yaml"
    own = yaml.safe_load(path.read_text())
    fast = yaml.safe_load((_ABL_CONFIGS / "ablations_fast.yaml").read_text())

    merged = _apply_fast_overrides(_to_upper(_load_ablation_config(str(path))))

    for key, value in own.items():
        if key == "extends" or key in fast:
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


def _inherited_values(path: Path, defaults: dict) -> dict:
    """Values *path* would see from its ancestors, excluding itself."""
    inherited = dict(defaults)
    for source, raw in main._load_config_chain(path):
        if source.resolve() == path.resolve():
            continue
        inherited.update({k: v for k, v in raw.items() if k != "extends"})
    return inherited


@pytest.mark.parametrize(
    "preset",
    sorted(p.name for p in _CONFIGS.glob("*.yaml") if p.name not in _DELTA_EXEMPT),
)
def test_preset_restates_no_inherited_value(preset) -> None:
    defaults = yaml.safe_load(_DEFAULTS.read_text())
    path = _CONFIGS / preset
    raw = yaml.safe_load(path.read_text()) or {}
    inherited = _inherited_values(path, defaults)
    restated = {
        k: v
        for k, v in raw.items()
        if k != "extends" and k in inherited and inherited[k] == v
    }
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
    restated = {
        k: v for k, v in raw.items() if k != "extends" and k in base and base[k] == v
    }
    assert not restated, f"{preset} restates inherited values: {restated}"
