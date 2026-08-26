"""``--emit-tex-macros``: the generated-number path into the manuscript.

The paper workspace requires every reported number to reach the manuscript
through a macro, so it is traceable to the run that produced it. That only
holds if the emitted file really is nothing but definitions and no name is
ever defined twice -- a collision would file one condition's number under a
name that reads as another's.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import ROOT, import_or_skip

tables = import_or_skip("experiments.rl_finetuning.analysis.tables")
run_ablations = import_or_skip("experiments.rl_finetuning.run_ablations")

CRAFTAX_RESULTS = (
    ROOT / "results/experiments/rl_finetuning/outputs/craftax_classic_ablations"
    "/results.json"
)

DEFINITION = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{[^{}]*\}$")


def macro_names(text: str) -> list[str]:
    """Every macro defined in *text*, asserting the file holds nothing else.

    Comment and blank lines carry provenance, not content; every other line
    must be a single ``\\newcommand``.
    """
    names = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("%"):
            continue
        match = DEFINITION.match(line)
        assert match, f"line {lineno} is not a macro definition: {line!r}"
        names.append(match.group(1))
    return names


@pytest.fixture(scope="module")
def suite() -> tuple[dict, float, dict]:
    """The shipped Craftax Classic results, as the analysis path loads them."""
    if not CRAFTAX_RESULTS.exists():
        pytest.skip(f"no results.json at {CRAFTAX_RESULTS}")
    results, pretrained, _ach, config = run_ablations._results_from_json(
        str(CRAFTAX_RESULTS)
    )
    return results, pretrained, config


@pytest.fixture(scope="module")
def emitted(suite, tmp_path_factory) -> tuple[Path, str]:
    """``results.tex`` written from the shipped Craftax Classic results."""
    results, pretrained, config = suite
    out = tmp_path_factory.mktemp("tex") / "results.tex"
    tables.write_tex_macros(results, pretrained, out, config=config)
    return out, out.read_text()


def test_file_is_definitions_only(emitted):
    _path, text = emitted
    assert macro_names(text), "no macros emitted"
    assert "\\def" not in text, "must emit \\newcommand, never \\def"


def test_macro_names_are_unique(emitted):
    _path, text = emitted
    names = macro_names(text)
    duplicates = {n for n in names if names.count(n) > 1}
    assert not duplicates, f"macro defined more than once: {sorted(duplicates)}"


def test_macro_names_are_legal_control_sequences(emitted):
    _path, text = emitted
    for name in macro_names(text):
        assert name.isalpha(), f"{name!r} is not letters-only"


def test_headline_quantities_are_covered(emitted):
    """The quantities the manuscript reports must all have a macro."""
    _path, text = emitted
    names = set(macro_names(text))
    required = {
        "rwPretrainedScore",
        "rwPooledSeedSd",
        "rwScoreBaselineRl",
        "rwScoreSdBaselineRl",
        "rwEssBaselineRl",
        "rwCvABaselineRl",
        *(f"rwGroupMean{g}" for g in "ABCD"),
    }
    assert required <= names, f"missing macros: {sorted(required - names)}"


def test_digits_are_spelled_out():
    """Condition names carrying digits must mangle to letters-only."""
    assert tables._macro_name("layer_ablation_top1") == "LayerAblationTopOne"
    assert tables._macro_name("layer_ablation_top2") == "LayerAblationTopTwo"
    assert tables._macro_name("advantage_clip") == "AdvantageClip"


def test_mangling_is_collision_free_over_the_registry():
    registry = import_or_skip("experiments.rl_finetuning.ablations.registry")
    tags = [tables._macro_name(name) for name in registry.REGISTRY]
    assert len(set(tags)) == len(tags), "two conditions mangle to one macro name"


def test_collision_raises_rather_than_overwriting(suite, tmp_path):
    """The lossy mangling can collide; that must fail loudly.

    ``top1`` and ``top_one`` are distinct conditions that both become
    ``TopOne``. Silently overwriting would publish one under the other's name.
    """
    results, pretrained, config = suite
    entry = next(iter(results.values()))
    colliding = {"layer_ablation_top1": entry, "layer_ablation_top_one": entry}
    with pytest.raises(ValueError, match="collision"):
        tables.write_tex_macros(
            colliding, pretrained, tmp_path / "results.tex", config=config
        )


def test_merge_emits_macros(suite, tmp_path):
    """``--merge --emit-tex-macros`` must write the same file the run does."""
    out = tmp_path / "merged"
    proc = subprocess.run(
        [
            sys.executable,
            "experiments/rl_finetuning/run_ablations.py",
            "--merge",
            str(CRAFTAX_RESULTS),
            "--output-dir",
            str(out),
            "--emit-tex-macros",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env={
            **{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
            "JAX_PLATFORMS": "cpu",
            "WANDB_MODE": "disabled",
            "MPLBACKEND": "Agg",
        },
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    emitted = out / "tables" / "results.tex"
    assert emitted.exists(), f"--merge did not write {emitted}"
    names = macro_names(emitted.read_text())
    assert len(set(names)) == len(names)
    assert "rwPretrainedScore" in names
