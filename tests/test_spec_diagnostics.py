"""Diagnostics closed-form spec tests (step 8).

Sources: research/spec-ablations.md §3 (CKA per Kornblith 2019
eqs (4)-(5); PCGrad surgery metrics per Yu 2020; ESS per Kish 1965).
Expected values are hand-computed in the docstrings. The minihack twin
file carries the same CKA/ESS/surgery assertions plus its repo-specific
JS/permutation/bootstrap/merge diagnostics.
"""

from __future__ import annotations

import re

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import _effective_batch_size
from experiments.rl_finetuning.analysis.report import (
    _HYPOTHESIS_GROUPS,
    _score_hypothesis,
)
from experiments.rl_finetuning.analysis.tables import (
    baseline_rl_score_of,
    metric_scale,
    verdict,
)
from experiments.rl_finetuning.diagnostics.gradient import (
    compute_surgery_metrics_jax,
)
from experiments.rl_finetuning.diagnostics.representation import _linear_cka


def test_linear_cka_is_one_for_identical_and_corr_squared_for_1d():
    """Linear CKA (Kornblith 2019 eqs (4)-(5), centred HSIC form):
    CKA(X, X) = 1, and for 1-D features CKA equals the squared Pearson
    correlation.

    Derivation of the 1-D case: with centred x~, y~ the centred Gram
    matrices are rank one, HSIC(x,y) = (x~.y~)^2 and the normalisation
    is |x~|^2 |y~|^2, so CKA = cos^2(angle) = corr^2. For x=[1,2,3,4],
    y=[1,3,2,4]: x~.y~ = 4, |x~|^2 = |y~|^2 = 5 -> CKA = (4/5)^2 = 0.64.
    """
    x = jnp.array([[1.0, 0.5], [2.0, -1.0], [-0.5, 0.25], [0.0, 3.0]])
    assert float(_linear_cka(x, x)) == pytest.approx(1.0, abs=1e-5)

    x1 = jnp.array([[1.0], [2.0], [3.0], [4.0]])
    y1 = jnp.array([[1.0], [3.0], [2.0], [4.0]])
    assert float(_linear_cka(x1, y1)) == pytest.approx(0.64, abs=1e-5)


def test_linear_cka_is_invariant_to_scaling_and_orthogonal_maps():
    """CKA is invariant to isotropic scaling and orthogonal
    transformations (Kornblith 2019 §2.3): CKA(X, c X Q) = 1."""
    x = jnp.array([[1.0, 0.5], [2.0, -1.0], [-0.5, 0.25], [0.0, 3.0]])
    theta = 0.3
    q = jnp.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )
    assert float(_linear_cka(x, 2.5 * (x @ q))) == pytest.approx(1.0, abs=1e-5)


def test_effective_sample_size_closed_form():
    """ESS = (sum w)^2 / sum w^2 (Kish 1965; spec-ablations §3.3).

    Derivation: w=[1,1,2] -> 16/6 = 2.6667; uniform weights give N.
    """
    assert float(_effective_batch_size(jnp.array([1.0, 1.0, 2.0]))) == pytest.approx(
        16 / 6, rel=1e-6
    )
    assert float(_effective_batch_size(jnp.ones(7))) == pytest.approx(7.0, rel=1e-6)


def test_surgery_metrics_measure_removed_gradient_mass():
    """PCGrad surgery metrics: fraction = removed squared mass /
    total before; a leaf counts as conflicting iff the projection
    removed mass from it (spec-ablations §3.2).

    Derivation: leaf a [2,0]->[1,0] (mass 4->1), leaf b [3,4] unchanged
    (mass 25): fraction = (29-26)/29 = 3/29; n_conflicting = 1.
    """
    before = {"a": jnp.array([2.0, 0.0]), "b": jnp.array([3.0, 4.0])}
    after = {"a": jnp.array([1.0, 0.0]), "b": jnp.array([3.0, 4.0])}
    frac, n_conf = compute_surgery_metrics_jax(before, after)
    assert float(frac) == pytest.approx(3 / 29, rel=1e-5)
    assert int(n_conf) == 1


# ---------------------------------------------------------------------------
# Ablation-suite verdict rule (shared with the sibling repo, character for
# character; PARITY open question resolved 2026-08-17)
# ---------------------------------------------------------------------------


def test_verdict_labels_against_baseline_rl_at_metric_scale():
    """Labels are taken against `baseline_rl`, with thresholds that are
    fractions of the metric's magnitude: IMPROVEMENT above +5%, COLLAPSE
    below -10%, NEUTRAL between.

    Derivation at scale 10 (`baseline_rl` 10.0, pretrained 8.0, the order
    of magnitude of a Craftax episode-weighted mean return): the
    improvement bar is +0.5 and the collapse bar -1.0, so 10.6 improves,
    10.4 does not, 9.1 holds and 8.9 collapses.

    The last case is the one the absolute rule got wrong. Constructed to
    the recorded shape: an arm sitting 1.911 below `baseline_rl` read
    IMPROVEMENT under the old craftax rule, because +0.089 against
    pretrained cleared an absolute +0.05 bar.
    """
    assert verdict(10.6, 10.0, 8.0) == "IMPROVEMENT"
    assert verdict(10.4, 10.0, 8.0) == "NEUTRAL"
    assert verdict(9.1, 10.0, 8.0) == "NEUTRAL"
    assert verdict(8.9, 10.0, 8.0) == "COLLAPSE"
    assert verdict(10.0 - 1.911, 10.0, 8.0) == "COLLAPSE"


def test_verdict_reduces_to_the_absolute_rule_at_a_metric_scale_of_one():
    """At scale 1.0 the fractions are the absolute +0.05 / -0.10 they
    replace, and both comparisons are strict.

    This is the anchor for a bounded metric: a MiniHack win rate lives in
    [0, 1], so the rule that governed it is unchanged in form. With
    `baseline_rl` 0.0 and pretrained 1.0 the scale is exactly 1.0 and the
    delta is the score itself, so the boundaries are exact in float.
    """
    assert verdict(0.05, 0.0, 1.0) == "NEUTRAL"
    assert verdict(0.06, 0.0, 1.0) == "IMPROVEMENT"
    assert verdict(-0.10, 0.0, 1.0) == "NEUTRAL"
    assert verdict(-0.11, 0.0, 1.0) == "COLLAPSE"


def test_verdict_scale_is_the_larger_reference_and_one_is_required():
    """The scale is the larger reference score in absolute value, so a
    `baseline_rl` near zero cannot shrink the threshold to nothing; with
    both references at zero there is no scale and no label is defensible.

    Derivation: `baseline_rl` 0.0 with pretrained 8.0 gives scale 8.0, so
    the bars are +0.4 and -0.8, not +0.0 and -0.0.
    """
    assert metric_scale(0.0, 8.0) == 8.0
    assert metric_scale(10.0, 8.0) == 10.0
    assert metric_scale(-3.0, 1.0) == 3.0

    assert verdict(0.39, 0.0, 8.0) == "NEUTRAL"
    assert verdict(0.41, 0.0, 8.0) == "IMPROVEMENT"
    assert verdict(-0.79, 0.0, 8.0) == "NEUTRAL"
    assert verdict(-0.81, 0.0, 8.0) == "COLLAPSE"

    assert verdict(0.0, 0.0, 0.0) == "NEUTRAL"
    assert verdict(1.0, 0.0, 0.0) == "NEUTRAL"


def test_the_reference_arm_falls_back_to_the_pretrained_score():
    """A suite run without `baseline_rl` has no reference arm, so the
    pretrained score stands in and every delta is measured from it."""
    assert baseline_rl_score_of({"baseline_rl": {"score": 0.7}}, 0.5) == 0.7
    assert baseline_rl_score_of({"kl_penalty": {"score": 0.6}}, 0.5) == 0.5


# ---------------------------------------------------------------------------
# Hypothesis attribution: the evidence set and the recommendation must agree
# (shared with the sibling repo, character for character; S7-9, decided
# 2026-08-18)
# ---------------------------------------------------------------------------

# The six groups and their evidence sets, pinned to the same literal in both
# repos. `analysis/report.py` carried no test at all until now, and the two
# dicts drifting apart would put different numbers under one heading in a
# cross-repo table. Update both files together or not at all.
_EXPECTED_EVIDENCE_SETS = {
    "Catastrophic Forgetting": [
        "ewc", "frozen_backbone", "head_only", "kl_penalty", "llrd", "lora",
    ],
    "Gradient Conflict": ["gradient_surgery", "kl_penalty", "low_t"],
    "Signal Sparsity": [
        "bc_wins", "reward_filtering", "reward_model", "running_stats",
    ],
    "Distributional Shift": ["action_diversity", "mixed_replay"],
    "Mode Collapse": ["advantage_clip", "entropy_bonus", "normalized_adv"],
    "t-Bias": ["low_t", "t_curriculum"],
}


def _named_in(text: str, arm: str) -> bool:
    """Does *text* name *arm* by its registry name, in prose?

    Registry keys are snake_case and the recommendations write them as prose,
    so the separator is relaxed to space, underscore or hyphen: `low_t`
    appears as "low-t" and `entropy_bonus` as "entropy bonus". The word
    bounds are what keep this honest -- a bare substring test would find
    `ewc` inside any word containing those letters, and matching on the
    relaxed separator alone would miss the hyphenated forms entirely.
    """
    pattern = r"\b" + r"[ _\-]".join(re.escape(part) for part in arm.split("_")) + r"\b"
    return re.search(pattern, text, re.IGNORECASE) is not None


def test_every_arm_a_recommendation_names_is_in_its_own_evidence_set():
    """A hypothesis may not recommend an intervention whose ablation it
    excludes from the evidence that scores it.

    `Catastrophic Forgetting` recommended LoRA -- "or use LoRA to restrict
    the parameter update space" -- while omitting the `lora` arm from its
    `supporting_ablations`, in both repos identically. Not cosmetic:
    `_score_hypothesis` computes `evidence_score = n_supporting /
    max(n_tested, 1)` over that list, so the omission changes the ranking
    `diagnosis.md` and the hypothesis-verdict tables print. Author decision
    2026-08-18: drift, not scoping.

    Only recommendations that name a registered arm are constrained. That
    eight of the 25 arms are cited by no hypothesis at all is a separate,
    deliberately open question and is not asserted here.
    """
    offenders = {
        name: sorted(
            arm
            for arm in REGISTRY
            if _named_in(info["recommendation"], arm)
            and arm not in info["supporting_ablations"]
        )
        for name, info in _HYPOTHESIS_GROUPS.items()
    }
    offenders = {name: arms for name, arms in offenders.items() if arms}

    assert not offenders, (
        "hypotheses recommending an intervention whose arm they leave out of "
        f"their own evidence set: {offenders}"
    )


def test_the_hypothesis_evidence_sets_are_the_pinned_shared_ones():
    """The groups and their membership are identical across the two repos.

    Nothing else pins `_HYPOTHESIS_GROUPS`, and it is the input to every
    number in `diagnosis.md`'s hypothesis ranking, so silent drift here is
    invisible until two repos disagree in one table.
    """
    actual = {
        name: sorted(info["supporting_ablations"])
        for name, info in _HYPOTHESIS_GROUPS.items()
    }

    assert actual == _EXPECTED_EVIDENCE_SETS


def test_the_evidence_score_is_the_raw_quotient_in_both_repos():
    """`evidence_score` is the unrounded fraction; rounding is for display.

    minihack returned `round(evidence, 3)` and craftax the raw quotient, so
    the same inputs gave 0.3330 and 0.3333 under one field name. Every
    consumer already formats at the point of use -- `:.0%` in the report
    tables and `int(score * 5)` for the star rating -- so rounding inside the
    scorer bought nothing and cost cross-repo agreement.
    """
    results = {
        "baseline_rl": {"score": 10.0},
        "kl_penalty": {"score": 10.5},
        "ewc": {"score": 10.5},
        "llrd": {"score": 9.0},
        "lora": {"score": 9.0},
        "frozen_backbone": {"score": 9.0},
        "head_only": {"score": 9.0},
    }
    scored = _score_hypothesis(
        "Catastrophic Forgetting",
        _HYPOTHESIS_GROUPS["Catastrophic Forgetting"],
        results,
        8.0,
    )

    assert scored["n_supporting"] == 2
    assert scored["n_tested"] == 6
    assert scored["evidence_score"] == 2 / 6


def test_an_unregistered_supporting_arm_is_an_error():
    """A typo'd or retired arm name must not be scored as a smaller sample.

    `_score_hypothesis` skips arms absent from `results`, which is correct for
    a run that did not include them -- and indistinguishable from a name that
    can never appear. Left unguarded, renaming an arm silently lowers
    `n_tested` and moves every evidence score that cites it.
    """
    broken = dict(_HYPOTHESIS_GROUPS["Catastrophic Forgetting"])
    broken["supporting_ablations"] = ["ewc", "not_an_ablation"]

    with pytest.raises(KeyError, match="not_an_ablation"):
        _score_hypothesis("Catastrophic Forgetting", broken, {}, 0.0)
