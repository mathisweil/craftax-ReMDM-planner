"""Diagnostics closed-form spec tests (step 8).

Sources: research/spec-ablations.md §3 (CKA per Kornblith 2019
eqs (4)-(5); PCGrad surgery metrics per Yu 2020; ESS per Kish 1965;
§3.4 exact permutation test + bootstrap CI).
Expected values are hand-computed in the docstrings. The minihack twin
file carries the same CKA/ESS/surgery and significance assertions --
`write_significance_test` is byte-identical across the repos -- plus its
repo-specific JS/action-distribution/merge diagnostics.
"""

from __future__ import annotations

import math
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
    write_significance_test,
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


def test_action_entropy_is_reported_in_nats():
    """Action-distribution entropy is in nats (spec-ablations §3.5; the same
    unit as minihack's `compute_entropy` since its matched commit).

    Both repos reported "entropy" under one label with the unit stated
    nowhere: craftax natural log, minihack log base 2, a factor of
    1/ln 2 = 1.442695 apart. Canon is nats, which is what the NELBO and
    cross-entropy figures throughout both suites already use.

    Derivation: eight actions drawn 4/2/1/1 out of 8 give probabilities
    [1/2, 1/4, 1/8, 1/8] over the four used and zero elsewhere, and an
    entropy of (1/2)ln2 + (1/4)ln4 + 2*(1/8)ln8 = 1.75 ln 2 = 1.2130075656
    nats, which is 1.75 bits. The epsilon the implementation adds inside
    the log costs less than 1e-9 of that.
    """
    from experiments.rl_finetuning.analysis.action_distribution import _compute_metrics

    actions = np.array([[0, 0, 0, 0, 1, 1, 2, 3]], dtype=np.int32)
    rewards = np.zeros_like(actions, dtype=np.float32)
    dones = np.zeros_like(actions, dtype=bool)
    metrics = _compute_metrics(actions, rewards, dones, 8, 0.0)
    assert metrics.entropy == pytest.approx(1.75 * math.log(2), abs=1e-6)
    assert metrics.entropy == pytest.approx(1.2130075656, abs=1e-6)


def test_the_significance_test_states_its_floor_and_corrects_for_selection(tmp_path):
    """The significance test is exact over all C(n_a+n_b, n_b) relabellings,
    reports the floor that enumeration imposes, and draws its null
    distribution over every candidate arm rather than over the one it picked
    (spec-ablations §3.4; both repos' experiments/README tables).

    Derivation, floor: every relabelling's complement negates each mean
    difference and so ties the statistic, which makes the count at least two
    -- p >= 2/C(6,3) = 0.100 at three seeds a side, for any data whatsoever.
    Baseline [0,0,0] against [1e6,1e6,1e6] therefore reports p = 0.100, and
    0.100 has to be reported as the floor rather than left to read as
    marginal significance.

    Derivation, selection: baseline [0,1,2,3] against [4,5,6,7] has an
    observed difference of 4, which only the two extreme partitions of the
    70 relabellings reach -- p = 2/70 = 0.029 while that arm is the only
    candidate. The null arm [-6,-2,2,6] scores no better than baseline but
    is spread widely enough that its own relabellings reach a statistic of 4
    another twelve times, and it is a candidate the maximum must range over,
    so p becomes 14/70 = 0.200. Selecting the arm from the same scores and
    then testing it uncorrected reports 0.029 either way.
    """
    write_significance_test(
        {
            "baseline_rl": {"all_scores": [0.0, 0.0, 0.0]},
            "kl_penalty": {"all_scores": [1e6, 1e6, 1e6]},
        },
        tmp_path,
    )
    text = (tmp_path / "significance_test.txt").read_text()
    assert "20 relabellings" in text
    assert "p = 0.100" in text
    assert "minimum attainable p at 3 baseline and 3 condition seeds: 0.100" in text
    assert "AT the floor" in text

    alone = tmp_path / "alone"
    write_significance_test(
        {
            "baseline_rl": {"all_scores": [0.0, 1.0, 2.0, 3.0]},
            "kl_penalty": {"all_scores": [4.0, 5.0, 6.0, 7.0]},
        },
        alone,
    )
    text = (alone / "significance_test.txt").read_text()
    assert "1 candidate arm " in text
    assert "p = 0.029" in text
    ci_line = next(line for line in text.splitlines() if "bootstrap" in line)
    assert float(ci_line.split("[")[1].split(",")[0]) > 0.0

    with_null_arm = tmp_path / "with_null_arm"
    write_significance_test(
        {
            "baseline_rl": {"all_scores": [0.0, 1.0, 2.0, 3.0]},
            "kl_penalty": {"all_scores": [4.0, 5.0, 6.0, 7.0]},
            "ewc": {"all_scores": [-6.0, -2.0, 2.0, 6.0]},
        },
        with_null_arm,
    )
    text = (with_null_arm / "significance_test.txt").read_text()
    assert "2 candidate arms" in text
    assert "p = 0.200" in text


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
