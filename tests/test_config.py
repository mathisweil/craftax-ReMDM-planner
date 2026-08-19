"""Config layering: preset resolution, cluster parity, delta-only presets."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest
import yaml

import main
from experiments.rl_finetuning.run_ablations import (
    _RESULT_AFFECTING,
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
        ("configs/final_craftax_classic_ucl.yaml", {"NUM_ENVS": 512, "SEED": 42}),
        ("configs/final_craftax_classic_qmul.yaml", {"NUM_ENVS": 96, "SEED": 43}),
    ],
)
def test_preset_resolves_over_the_defaults(preset, expected) -> None:
    config = _build(["--mode", "online", "--config", preset])
    for key, value in expected.items():
        assert config[key] == value
    # Comes from defaults.yaml, which holds the recipe, not from the preset.
    assert config["D_MODEL"] == 384
    assert config["N_LAYERS"] == 6


@pytest.mark.parametrize("family", ["craftax_classic", "craftax"])
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


# ---------------------------------------------------------------------------
# cross-machine poolability of ablation runs
#
# `run_ablations.py --merge` averages seeds of the same ablation across
# results.json files, so pooling two machine configs is only sound when they
# agree on everything that changes the trained model or the measured score.
# Each family's UCL config is its reference (the minihack twin applies the
# same policy to its single family).
# ---------------------------------------------------------------------------

# The result-affecting key set is declared once, in production, as
# ``run_ablations._RESULT_AFFECTING``: the same set that classifies these
# configs is the one ``--merge`` enforces on the configs a run recorded.

#: Reference config per family.
_REFERENCE_CONFIG = {
    "classic": "ablations_final_craftax_classic_ucl.yaml",
    "craftax": "ablations_final_craftax_ucl.yaml",
}

#: Configs whose runs may be merged with their family's reference.
_POOLABLE = set(_REFERENCE_CONFIG.values())

#: Configs that must NOT be merged with the reference, mapped to the
#: result-affecting keys on which they are known to diverge.
_NOT_POOLABLE = {
    "ablations_final_craftax_classic_qmul.yaml": frozenset(
        {
            "num_envs",  # 64 vs 192: less rollout diversity per iteration
            "batch_size",  # 256 vs 1024: ~4x per-update SNR
            "eval_steps",  # 512 vs 1024: noisier score
            "mixed_replay_buffer_size",  # 10000 vs 20000
        }
    ),
    "ablations_final_craftax_qmul.yaml": frozenset(
        {
            "num_envs",  # 64 vs 128
            "batch_size",  # 512 vs 1024
            "eval_steps",  # 512 vs 1024
        }
    ),
}


def _family(name: str) -> str:
    return "classic" if "classic" in name else "craftax"


def _machine_configs() -> list[str]:
    return sorted(p.name for p in _ABL_CONFIGS.glob("ablations_final_*.yaml"))


def _divergence(name: str) -> set[str]:
    reference = _load_ablation_config(
        str(_ABL_CONFIGS / _REFERENCE_CONFIG[_family(name)])
    )
    candidate = _load_ablation_config(str(_ABL_CONFIGS / name))
    return {
        k
        for k in _RESULT_AFFECTING
        if candidate.get(k, "<absent>") != reference.get(k, "<absent>")
    }


def test_every_machine_config_is_classified() -> None:
    """A new machine config must be declared poolable or not, never silently."""
    unclassified = set(_machine_configs()) - _POOLABLE - set(_NOT_POOLABLE)
    assert not unclassified, (
        f"Unclassified ablation machine config(s): {sorted(unclassified)}. "
        "Add to _POOLABLE (and align result-affecting keys with the family "
        "reference) or to _NOT_POOLABLE with the diverging keys."
    )


@pytest.mark.parametrize("name", sorted(_POOLABLE))
def test_poolable_config_matches_reference(name) -> None:
    diverged = _divergence(name)
    assert not diverged, (
        f"{name} is declared poolable with {_REFERENCE_CONFIG[_family(name)]} "
        f"but diverges on result-affecting key(s): {sorted(diverged)}. Align "
        "them or move it to _NOT_POOLABLE."
    )


@pytest.mark.parametrize("name", sorted(_NOT_POOLABLE))
def test_not_poolable_divergence_is_recorded(name) -> None:
    """Catches drift in both directions: new divergence, and silent alignment."""
    expected = _NOT_POOLABLE[name]
    actual = _divergence(name)
    assert actual == set(expected), (
        f"{name} divergence from {_REFERENCE_CONFIG[_family(name)]} has "
        f"changed. Recorded: {sorted(expected)}; actual: {sorted(actual)}. "
        "Update _NOT_POOLABLE, or move it to _POOLABLE if now aligned."
    )


# ---------------------------------------------------------------------------
# --merge enforces the policy above on the configs a run actually recorded.
# The classification tests read the shipped YAML; these read results.json,
# which is what an operator merges and where `main` records the UPPERCASE
# `merged` dict, so the guard has to answer to both casings.
# ---------------------------------------------------------------------------


def _results_file(tmp_path: Path, name: str, config: dict, scores: list[float]):
    """Write a minimal results.json recording *config*.

    Args:
        tmp_path: Directory to write into.
        name:     File name.
        config:   The config to record, exactly as given.
        scores:   Per-seed scores for the single ablation in the file.

    Returns:
        The path, as a string.
    """
    payload = {
        "pretrained_score": 0.5,
        "pretrained_ach_rates": {},
        "config": config,
        "ablations": {
            "baseline_rl": {
                "score": sum(scores) / len(scores),
                "score_std": 0.0,
                "all_scores": scores,
                "history": {},
            }
        },
    }
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def _recorded(name: str) -> dict:
    """The shipped machine config *name* as `main` would record it."""
    return _to_upper(_load_ablation_config(str(_ABL_CONFIGS / name)))


@pytest.mark.parametrize("name", sorted(_NOT_POOLABLE))
def test_merge_refuses_a_not_poolable_pair(tmp_path, name) -> None:
    """The refusal names every recorded diverging key, with both values."""
    from experiments.rl_finetuning.run_ablations import _merge_result_files

    reference = _REFERENCE_CONFIG[_family(name)]
    paths = [
        _results_file(tmp_path, "ref.json", _recorded(reference), [1.0]),
        _results_file(tmp_path, "cand.json", _recorded(name), [2.0]),
    ]
    with pytest.raises(ValueError) as excinfo:
        _merge_result_files(paths)

    message = str(excinfo.value)
    assert "not poolable" in message
    for key in _NOT_POOLABLE[name]:
        assert key in message, f"{key} is a recorded divergence but unnamed"


def test_merge_refuses_a_cross_family_pair(tmp_path) -> None:
    """Classic and Craftax runs are never poolable: different env, among
    other things. Neither family's config is the other's reference, so
    nothing above would have caught this."""
    from experiments.rl_finetuning.run_ablations import _merge_result_files

    paths = [
        _results_file(
            tmp_path,
            "classic.json",
            _recorded("ablations_final_craftax_classic_ucl.yaml"),
            [1.0],
        ),
        _results_file(
            tmp_path,
            "craftax.json",
            _recorded("ablations_final_craftax_ucl.yaml"),
            [2.0],
        ),
    ]
    with pytest.raises(ValueError, match="not poolable"):
        _merge_result_files(paths)


def test_merge_refuses_a_file_that_records_no_config(tmp_path) -> None:
    """A distinct refusal: absent is not equal, so it is never merged on trust."""
    from experiments.rl_finetuning.run_ablations import _merge_result_files

    paths = [
        _results_file(
            tmp_path, "ref.json", _recorded("ablations_final_craftax_ucl.yaml"), [1.0]
        ),
        _results_file(tmp_path, "bare.json", {}, [2.0]),
    ]
    with pytest.raises(ValueError, match="records no config"):
        _merge_result_files(paths)


def test_a_merged_config_is_one_input_file_and_the_merge_is_recorded(tmp_path) -> None:
    """The config recorded beside merged results is one input file's, whole,
    and which one is written next to it (spec-ablations §1.3).

    Merging the configs key by key produced a config that matched no input
    file. The poolability guard forbids the result-affecting keys from
    diverging, so the chimera can only form out of the rest -- worker
    counts, output paths, the W&B run name -- but those are what tell a
    reader which machine produced the numbers, and a per-key blend names a
    run that never happened.

    Derivation: two poolable files differing only in the W&B project, a key
    the guard does not police. The blend takes the second file's value while
    every other key comes from the first; the fix takes the first file's
    config entire, and records that it did.
    """
    from experiments.rl_finetuning.run_ablations import (
        _merge_result_files,
        _results_to_json,
    )

    config_a = _recorded("ablations_final_craftax_ucl.yaml")
    config_a["WANDB_PROJECT"] = "run-on-ucl"
    config_b = dict(config_a)
    config_b["WANDB_PROJECT"] = "run-on-qmul"
    paths = [
        _results_file(tmp_path, "a.json", config_a, [1.0, 2.0]),
        _results_file(tmp_path, "b.json", config_b, [3.0]),
    ]
    _, _, _, merged_config = _merge_result_files(paths)

    # The whole of the first file's config, not a key-by-key blend.
    assert merged_config == config_a
    assert merged_config["WANDB_PROJECT"] == "run-on-ucl"
    assert _merge_result_files(paths[::-1])[3] == config_b

    # And the merge itself is on the record.
    provenance = {"inputs": paths, "config_from": paths[0]}
    payload = json.loads(
        _results_to_json({}, 0.5, merged_config, {}, provenance).decode()
    )
    assert payload["merge_provenance"] == provenance
    # A single run carries no provenance block, so its presence marks a merge.
    assert "merge_provenance" not in json.loads(
        _results_to_json({}, 0.5, merged_config, {}).decode()
    )


def test_merge_still_pools_two_runs_of_the_same_config(tmp_path) -> None:
    """The guard is a refusal on wrong input only: a poolable pair merges to
    the same values it did before the guard existed."""
    from experiments.rl_finetuning.run_ablations import _merge_result_files

    config = _recorded("ablations_final_craftax_ucl.yaml")
    paths = [
        _results_file(tmp_path, "a.json", config, [1.0, 2.0]),
        _results_file(tmp_path, "b.json", config, [3.0]),
    ]
    merged, pretrained, _, merged_config = _merge_result_files(paths)

    assert merged["baseline_rl"]["all_scores"] == [1.0, 2.0, 3.0]
    assert merged["baseline_rl"]["score"] == pytest.approx(2.0)
    assert merged["baseline_rl"]["score_std"] == pytest.approx(math.sqrt(2 / 3))
    assert pretrained == pytest.approx(0.5)
    assert merged_config == config


_SHIPPED_CONFIGS = (_CONFIGS / "defaults.yaml", _ABL_DEFAULT)
_SOURCE_DIRS = ("src", "experiments", "scripts")


def _production_sources() -> list[Path]:
    """Every production module: the entry point and the source packages."""
    found = [_ROOT / "main.py"]
    for name in _SOURCE_DIRS:
        found += sorted((_ROOT / name).rglob("*.py"))
    return found


def _normalise(key: str) -> str:
    """Config keys are lower case in YAML and upper case in code here."""
    return key.upper()


# Every key read from a config object that no shipped YAML declares, with the
# reason it is not declared. Anything not listed here fails the test.
_NOT_FROM_A_CONFIG_FILE = frozenset(
    {
        # Derived at load by resolve_num_updates / resolve_scaled_hyperparams
        # from the env-frame and update-cycle forms that ARE declared. Setting
        # a derived form directly is a documented hazard, not a config key
        # (README §Configuration, CLAUDE.md).
        "NUM_UPDATES",
        "LR_WARMUP_STEPS",
        "VAL_INTERVAL",
        "DAGGER_BETA_DECAY",
        "DAGGER_BUFFER_MAX",
        "MINIBATCH_SIZE",
        # Read off the environment's action space, not off a file.
        "NUM_ACTIONS",
        # Injected by the CLI: run mode, JIT toggle and the four artefact
        # paths, none of which has a default worth shipping.
        "MODE",
        "JIT",
        "CHECKPOINT_PATH",
        "PPO_CHECKPOINT_PATH",
        "OFFLINE_CHECKPOINT_PATH",
        "OFFLINE_DATA_PATH",
        "INFERENCE_OUTPUT",
        # Read from the PPO expert artefact's own W&B config snapshot in
        # hf_upload.py, which declares it; it is not a key of this repo.
        "TOTAL_TIMESTEPS",
    }
)


# ---------------------------------------------------------------------------
# Publishing: discovery layout and the published config surface
# ---------------------------------------------------------------------------


def _hf_upload():
    spec = importlib.util.spec_from_file_location(
        "hf_upload", _ROOT / "scripts" / "hf_upload.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _plant(directory: Path) -> None:
    """Write the marker craftax discovery looks for, under a step directory."""
    step = directory / "1000"
    step.mkdir(parents=True, exist_ok=True)
    (step / "_CHECKPOINT_METADATA").write_text("{}")


def test_publish_discovery_is_at_the_released_layout_and_skips_downloads(tmp_path):
    """Checkpoint discovery finds the released layout and never the Hub
    download copies under `checkpoints/hf/` (both repos' README, Publishing).

    The two repos failed this in opposite directions. craftax used a
    fixed-depth glob, which happened to exclude `checkpoints/hf/` because a
    download sits one level deeper than a real checkpoint -- true by
    arithmetic, not by intent, and silent if the layout ever changed depth.
    minihack used a recursive `rglob("*.pth")`, which on its live tree found
    ten directories, **none of them at the released layout**: seven raw
    `dagger_<timestamp>/` run directories, the repository root with two loose
    files, and two artefacts already published, which a publish would have
    pushed back up into a nested `checkpoints/hf/checkpoints/...` tree.

    Both now discover at the released layout and both skip `checkpoints/hf/`
    explicitly, so the exclusion holds at any depth.
    """
    hf = _hf_upload()
    monkey_ckpts = tmp_path / "checkpoints"
    released = monkey_ckpts / "online" / "Some-Model-100M"
    download = monkey_ckpts / "hf" / "checkpoints" / "online" / "Some-Model-100M"
    stray = monkey_ckpts / "dagger_20260101_000000_abcd"
    for d in (released, download, stray):
        d.mkdir(parents=True)
        _plant(d)
    # A download parked at the released depth is still a download.
    shallow_download = monkey_ckpts / "hf" / "Some-Model-100M"
    shallow_download.mkdir(parents=True)
    _plant(shallow_download)

    hf.CKPTS = monkey_ckpts
    found = hf.discover_checkpoints()

    assert set(found) == {released}, sorted(str(p) for p in found)
    assert not any(hf._is_download_copy(p) for p in found)
    assert hf._is_download_copy(download)
    assert hf._is_download_copy(shallow_download)



def test_the_published_config_drops_the_same_environment_keys_in_both_repos():
    """The published config keeps the recipe and drops provenance, by one
    declaration that is byte-identical across the repos (both READMEs).

    Neither repo dropped `use_wandb`: it starts with neither `wandb_` nor
    `hub_`, so minihack's prefix rule missed it, and craftax removed only the
    nested `_wandb` blob, leaving `USE_WANDB`, `WANDB_ENTITY` and
    `WANDB_PROJECT` in the released `config.yaml`. No credential was exposed
    either way -- those live in `_wandb` and `wandb-metadata.json`, both
    already removed -- but the published surface advertised an account and a
    project that are nothing to do with the recipe, and the two repos
    advertised different ones.

    Keys are compared lower-cased because craftax records them UPPERCASE and
    minihack lower-case.
    """
    hf = _hf_upload()

    for key in ("_wandb", "use_wandb", "USE_WANDB", "wandb_project",
                "WANDB_ENTITY", "hub_repo_id", "HUB_TOKEN"):
        assert hf.is_environment_key(key), key

    for key in ("lr", "LR", "batch_size", "NUM_ENVS", "noise_schedule",
                "use_amp", "USE_AMP", "hubris", "wandbish"):
        assert not hf.is_environment_key(key), key

# ---------------------------------------------------------------------------
# Config-key reachability (the class F-1 belongs to)
# ---------------------------------------------------------------------------
# F-1: `hf_upload.py::selection` read `checkpoint_every` and `max_iterations`
# long after both were renamed out of the config. `.get()` returned None, so
# every published DAgger selection.json recorded a null and nothing failed.
# The class is "a key the code reads that no shipped config declares", and the
# only way to keep it closed is to re-derive the set on every run.
#
# `_CONFIG_NAMES` are the identifiers a merged config is bound to across the
# production sources. Attributes and subscripts on those names are config
# reads; leading-underscore names are not -- they are the convention for a
# value the code stamps onto the namespace at run time.
_CONFIG_NAMES = frozenset({"config", "cfg", "merged", "conf"})
_NOT_CONFIG_ATTRS = frozenset(
    {"get", "items", "keys", "values", "update", "copy", "pop", "setdefault"}
)


def _is_config_ref(node: ast.AST) -> bool:
    """Is *node* a reference to a merged config?

    Two shapes reach one and both are production: a bare name (``cfg.KEY``)
    and an attribute chain ending in a config name (``self.cfg.KEY``,
    ``ctx.cfg.KEY``). Matching only the first made the scan blind to a fifth
    of the ablation recipe's readers (sweep S0-3, gate F-7).
    """
    if isinstance(node, ast.Name):
        return node.id in _CONFIG_NAMES
    if isinstance(node, ast.Attribute):
        return node.attr in _CONFIG_NAMES
    return False


class _ConfigKeyScanner(ast.NodeVisitor):
    """Collect every config key a module reads, with its call site."""

    def __init__(self) -> None:
        self.keys: dict[str, set[str]] = {}
        self.path = "?"

    def _record(self, key: str, node: ast.AST) -> None:
        if not key.startswith("_"):
            self.keys.setdefault(key, set()).add(f"{self.path}:{node.lineno}")

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and _is_config_ref(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            self._record(node.args[0].value, node)
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and _is_config_ref(node.args[0])
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            self._record(node.args[1].value, node)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            _is_config_ref(node.value)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
        ):
            self._record(node.slice.value, node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            _is_config_ref(node.value)
            and node.attr not in _NOT_CONFIG_ATTRS
        ):
            self._record(node.attr, node)
        self.generic_visit(node)


def _config_keys_read() -> dict[str, set[str]]:
    scanner = _ConfigKeyScanner()
    for source in _production_sources():
        scanner.path = str(source.relative_to(_ROOT))
        scanner.visit(ast.parse(source.read_text()))
    return scanner.keys


def _declared_keys() -> set[str]:
    declared: set[str] = set()
    for path in _SHIPPED_CONFIGS:
        declared |= set(yaml.safe_load(path.read_text()) or {})
    return declared


def test_every_config_key_the_code_reads_is_declared_in_a_shipped_config():
    """No production source may read a key no shipped config declares.

    Every unresolved key must be named in `_NOT_FROM_A_CONFIG_FILE` with the
    reason it is not declared -- derived at load, injected by the CLI, or read
    from a foreign config. That list is the point of the test: an entry is a
    visible decision, whereas F-1 was a silent one.
    """
    read = _config_keys_read()
    declared = {_normalise(key) for key in _declared_keys()}

    unresolved = {
        key: sorted(sites)
        for key, sites in read.items()
        if _normalise(key) not in declared and key not in _NOT_FROM_A_CONFIG_FILE
    }

    assert not unresolved, (
        "config keys read by production code that no shipped config declares "
        f"and no exemption covers: {unresolved}"
    )


def test_the_reachability_scan_reaches_the_code_it_claims_to_cover():
    """Guard the scanner itself.

    A scan that silently matched nothing would pass the test above forever.
    These floors pin that the production sources are found, that a healthy
    number of keys is recovered, and that the exemption list has not grown
    into an allowlist for everything.
    """
    read = _config_keys_read()

    assert len(_production_sources()) >= 20
    assert len(read) >= 80
    assert len(_declared_keys()) >= 80
    assert len(_NOT_FROM_A_CONFIG_FILE) <= len(read) // 4
    stale = sorted(_NOT_FROM_A_CONFIG_FILE - set(read))
    assert not stale, f"exemptions for keys nothing reads any more: {stale}"


# ---------------------------------------------------------------------------
# Ablation descriptions (sweep S0-2)
#
# `AblationSpec.description` is what `--list` prints and what every run logs,
# and nothing read it: reverting either the F-2 or the F-3 description fix
# left the whole suite green. The table below is the shared canon, character
# for character with the sibling repo -- which is also what caught two
# typographic divergences between them (`baseline_rl`'s dash and
# `kl_penalty`'s "vs"), aligned to ASCII in the same commit.
#
# It is deliberately NOT compared with the `experiments/README.md` table:
# those cells are short labels in a different register ("Soft KL constraint
# vs pretrained") and the registry carries the mechanism sentence. 24 of 25
# differ by design; only `mixed_replay` coincides, because `a2d8231` gave
# the registry and the README the same self-replay sentence -- superseding
# F-2, which had reached the coincidence by copying the older README wording.
# ---------------------------------------------------------------------------

_EXPECTED_DESCRIPTIONS = {
    "action_diversity": (
        "Baseline ELBO with degenerate (all-same-action) plans discarded"
    ),
    "advantage_clip": (
        "Baseline ELBO with PPO-style advantage clipping to [1-eps, 1+eps]"
    ),
    "attention_only": (
        "Baseline ELBO updating only attention weights (Q/K/V/O); FFN frozen"
    ),
    "baseline_rl": (
        "Return-weighted ELBO -- no modifications"
    ),
    "bc_wins": (
        "Uniform ELBO on win windows only (no advantage weighting)"
    ),
    "entropy_bonus": (
        "Baseline ELBO minus entropy bonus (encourages action diversity)"
    ),
    "ewc": (
        "ELBO + Elastic Weight Consolidation (Fisher diagonal regularisation)"
    ),
    "ffn_only": (
        "Baseline ELBO updating only FFN layers; attention frozen"
    ),
    "frozen_backbone": (
        "Baseline ELBO training the action head and token embeddings (backbone frozen)"
    ),
    "gradient_surgery": (
        "PCGrad: RL gradient projected to remove conflict with BC gradient"
    ),
    "head_only": (
        "Baseline ELBO updating only the final linear projection"
    ),
    "kl_penalty": (
        "Return-weighted ELBO + soft KL penalty vs pretrained"
    ),
    "layer_ablation_top1": (
        "Baseline ELBO updating only the top-1 transformer block + head"
    ),
    "layer_ablation_top2": (
        "Baseline ELBO updating only the top-2 transformer blocks + head"
    ),
    "layer_ablation_top3": (
        "Baseline ELBO updating only the top-3 transformer blocks + head"
    ),
    "llrd": (
        "Baseline ELBO with Layer-wise Learning Rate Decay"
    ),
    "lora": (
        "Baseline ELBO with LoRA adaptation (rank-r attention projections only)"
    ),
    "low_t": (
        "Return-weighted ELBO restricted to low-t (fine-detail) regime"
    ),
    "mixed_replay": (
        "Self-replay: the run's own past online windows resampled into each batch"
    ),
    "normalized_adv": (
        "Baseline ELBO with (A - mean) / (std + eps) advantage normalisation"
    ),
    "reward_filtering": (
        "Baseline ELBO trained only on top-75th-percentile return windows"
    ),
    "reward_model": (
        "Baseline ELBO with advantages re-weighted by a learned MLP reward model"
    ),
    "running_stats": (
        "Baseline ELBO with EMA running mean/std for advantage normalisation"
    ),
    "t_curriculum": (
        "ELBO with t range annealed from high-t to low-t over training"
    ),
    "trust_region_kl": (
        "Baseline ELBO + hard KL trust region via quadratic barrier"
    ),
}


def test_every_registered_ablation_has_its_pinned_description():
    """Character for character, and the same table in the sibling repo."""
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    assert set(REGISTRY) == set(_EXPECTED_DESCRIPTIONS), (
        f"registry/table mismatch: only in registry "
        f"{sorted(set(REGISTRY) - set(_EXPECTED_DESCRIPTIONS))}, only in "
        f"table {sorted(set(_EXPECTED_DESCRIPTIONS) - set(REGISTRY))}"
    )
    wrong = {
        name: (spec.description, _EXPECTED_DESCRIPTIONS[name])
        for name, spec in REGISTRY.items()
        if spec.description != _EXPECTED_DESCRIPTIONS[name]
    }
    assert not wrong, f"description drift: {wrong}"


def test_every_ablation_names_a_hypothesis_and_is_listed_in_the_readme():
    """The two other strings a run surfaces. The README check is by name
    only, for the register reason in the comment above."""
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    readme = (_ROOT / "experiments" / "README.md").read_text()
    for name, spec in sorted(REGISTRY.items()):
        assert spec.hypothesis.strip(), f"{name} has no hypothesis"
        assert f"`{name}`" in readme, f"{name} is in no README table"


def test_the_reachability_scan_sees_config_reads_through_an_attribute():
    """`self.cfg.KEY` and `ctx.cfg.KEY` are production shapes; the scanner
    matched only a bare `cfg.KEY` and was blind to both (sweep S0-3, gate
    F-7). All four access forms are covered, and an unrelated attribute
    chain must still be ignored."""
    scanner = _ConfigKeyScanner()
    scanner.path = "<synthetic>"
    scanner.visit(
        ast.parse(
            "cfg.plain_name\n"
            "self.cfg.via_self\n"
            "ctx.cfg.via_ctx\n"
            "self.cfg.get('via_self_get', 0)\n"
            "self.cfg['via_self_subscript']\n"
            "getattr(self.cfg, 'via_self_getattr', 0)\n"
            "unrelated.attr.not_a_config\n"
        )
    )
    assert set(scanner.keys) == {
        "plain_name",
        "via_self",
        "via_ctx",
        "via_self_get",
        "via_self_subscript",
        "via_self_getattr",
    }
