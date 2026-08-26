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


GDELTA_AGGREGATE = (
    ROOT / "experiments/rl_finetuning/outputs/gdelta_verification"
    "/gdelta_aggregate.json"
)


@pytest.fixture(scope="module")
def gdelta_agg() -> dict:
    """The shipped three-seed aggregate, or skip."""
    if not GDELTA_AGGREGATE.exists():
        pytest.skip(f"no gdelta aggregate at {GDELTA_AGGREGATE}")
    import orjson

    return orjson.loads(GDELTA_AGGREGATE.read_bytes())


@pytest.fixture(scope="module")
def emitted_with_gdelta(suite, gdelta_agg, tmp_path_factory) -> str:
    results, pretrained, config = suite
    out = tmp_path_factory.mktemp("tex_gd") / "results.tex"
    tables.write_tex_macros(
        results, pretrained, out, config=config, gdelta=gdelta_agg
    )
    return out.read_text()


def test_gdelta_macros_are_emitted_and_tagged_by_ablation(emitted_with_gdelta):
    """Measured quantities appear, tagged by ablation name not variant name."""
    names = set(macro_names(emitted_with_gdelta))
    required = {
        "rwGdeltaNSeeds",
        "rwGdeltaBcSelfCos",
        "rwGdeltaRandomCosSd",
        "rwGdeltaEqFourResidual",
        # baseline_clipped_ratio tags as BaselineRl, matching rwScoreBaselineRl.
        "rwGdeltaCvABaselineRl",
        "rwGdeltaRatioBaselineRl",
        "rwGdeltaAbarBaselineRl",
        "rwGdeltaCosAdvantageClip",
        "rwGdeltaRatioShufBcWins",
    }
    assert required <= names, f"missing macros: {sorted(required - names)}"
    assert not any(n.endswith("BaselineClippedRatio") for n in names)


def macro_values(text: str) -> dict[str, str]:
    """Every macro in *text* as a name -> value mapping."""
    out = {}
    for line in text.splitlines():
        match = DEFINITION.match(line)
        if match:
            out[match.group(1)] = line.split("}{", 1)[1].rstrip("}")
    return out


def test_gdelta_cv_a_does_not_displace_the_ess_derived_one(
    emitted, emitted_with_gdelta
):
    """The two CV_A macros are different quantities and must both survive.

    ``rwCvABaselineRl`` recovers CV_A from the ESS logged during training;
    ``rwGdeltaCvABaselineRl`` is measured on the measurement batches at the
    pretrained checkpoint. The manuscript quotes both, and they do not agree.
    """
    _path, plain = emitted
    before = macro_values(plain)
    after = macro_values(emitted_with_gdelta)

    assert {"rwCvABaselineRl", "rwGdeltaCvABaselineRl"} <= set(after)
    # Adding the measurement must not perturb a macro that was already defined.
    assert {k: after[k] for k in before} == before


def test_gdelta_macros_omitted_when_not_measured(emitted):
    """A run with no measurement emits no gdelta macros at all."""
    _path, text = emitted
    assert not [n for n in macro_names(text) if n.startswith("rwGdelta")]


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
