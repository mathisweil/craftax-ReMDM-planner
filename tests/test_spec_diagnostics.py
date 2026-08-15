"""Diagnostics closed-form spec tests (step 8).

Sources: research/spec-ablations.md §3 (CKA per Kornblith 2019
eqs (4)-(5); PCGrad surgery metrics per Yu 2020; ESS per Kish 1965).
Expected values are hand-computed in the docstrings. The minihack twin
file carries the same CKA/ESS/surgery assertions plus its repo-specific
JS/permutation/bootstrap/merge diagnostics.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from experiments.rl_finetuning.ablations.training import _effective_batch_size
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
