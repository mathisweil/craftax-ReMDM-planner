"""Spec tests for the DAgger/BC training pipeline seams (step 9).

The window-extraction, expert-labelling and policy-mixing logic used to
live inline in the jitted runner closures (step-8 seam list); they are
now module-level pure functions pinned here against spec-training §1.2,
§1.5 and §2.2 / Amendment 6. Expected values are hand-derived in the
docstrings. The minihack DAgger variant is repo-specific (PARITY
"Method pipeline"); its own pipeline tests live in its
tests/test_spec_training.py.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.planners.common import extract_sliding_windows
from src.planners.online import expert_action, mixed_execution_mask


def _arrays(T=6, E=2, D=3):
    obs = jnp.arange(T * E * D, dtype=jnp.float32).reshape(T, E, D)
    acts = jnp.arange(T * E, dtype=jnp.int32).reshape(T, E)
    rewards = jnp.arange(T * E, dtype=jnp.float32).reshape(T, E)
    return obs, acts, rewards


def test_sliding_windows_validity_and_boundary_on_last_action():
    """Window t is valid iff no episode boundary falls strictly inside
    its action sequence: dones_after[t..t+H-2] all False; a boundary on
    the LAST action is allowed (spec-training §2.2: windows valid iff
    dones inside the sequence are all False; Amendment 6).

    Derivation (T=6, H=3, env 0 boundary after action 2, env 1 none):
    windows check dones_after rows t..t+1, so env 0 gives
    [True, False, False, True] for t=0..3 (t=0 sees rows 0-1: clean -
    the boundary sits on its last action; t=1 sees rows 1-2 and t=2
    rows 2-3: both hit row 2). Env 1 is all-valid. Returns are H-step
    reward sums: window t of env e sums rewards rows t..t+2.
    """
    obs, acts, rewards = _arrays()
    dones_after = jnp.zeros((6, 2), dtype=bool).at[2, 0].set(True)
    obs_w, acts_w, valid_w, returns_w = extract_sliding_windows(
        obs, acts, dones_after, plan_horizon=3, rewards_t=rewards
    )
    assert obs_w.shape == (4, 2, 3) and acts_w.shape == (4, 2, 3)
    assert np.asarray(valid_w)[:, 0].tolist() == [True, False, False, True]
    assert np.asarray(valid_w)[:, 1].tolist() == [True] * 4
    # window 0, env 0: rewards rows 0,1,2 of column 0 = 0 + 2 + 4
    assert float(returns_w[0, 0]) == 6.0
    # actions of window 1, env 1: rows 1,2,3 of column 1 = 3, 5, 7
    assert np.asarray(acts_w)[1, 1].tolist() == [3, 5, 7]


def test_sliding_windows_offline_pre_step_shift_is_equivalent():
    """The offline runner's pre-step done convention (traj.done[i] =
    reset BEFORE step i) converts to the helper's post-action convention
    by dropping the first row: done[i+1] marks the reset after action i
    (spec-training Amendment 6: the two runners' window-validity
    conventions are equivalent under the shift; the padded last row is
    never read).

    Derivation: pre-step dones with a reset before step 3 of env 0 mean
    the boundary is after action 2 - the same scenario as the direct
    test above, so the same validity pattern must come out.
    """
    obs, acts, _ = _arrays()
    pre_step = jnp.zeros((6, 2), dtype=bool).at[3, 0].set(True)
    dones_after = jnp.concatenate(
        [pre_step[1:], jnp.zeros((1, 2), dtype=bool)], axis=0
    )
    _, _, valid_w = extract_sliding_windows(obs, acts, dones_after, plan_horizon=3)
    assert np.asarray(valid_w)[:, 0].tolist() == [True, False, False, True]


def test_expert_action_is_argmax_when_deterministic():
    """dagger_expert_deterministic=true labels visited states with the
    expert's argmax action, keeping the mapping s -> a* fixed
    (spec-training §1.5; craftax defaults.yaml documented intent)."""
    logits = jnp.array([[[0.1, 2.0, -1.0], [3.0, 0.0, 1.0]]])
    got = expert_action(logits, True, jax.random.PRNGKey(0))
    assert np.asarray(got).tolist() == [[1, 0]]


def test_expert_action_samples_the_policy_when_stochastic():
    """The non-deterministic variant samples the expert policy: label
    frequencies follow softmax(logits) (DAgger's pi* labels, Ross 2011
    Alg 3.1). Derivation: logits [0, 1] give p1 = e/(1+e) = 0.7311;
    8192 draws, sigma = sqrt(0.7311*0.2689/8192) = 0.0049; bound 0.02 =
    4.1 sigma.
    """
    logits = jnp.broadcast_to(jnp.array([0.0, 1.0]), (8192, 2))
    got = np.asarray(expert_action(logits, False, jax.random.PRNGKey(1)))
    assert abs(got.mean() - np.e / (1 + np.e)) < 0.02


def test_mixed_execution_mask_matches_beta():
    """Per-step Bernoulli(beta) expert/learner mixing (DAgger Alg 3.1;
    spec-training §1.2: per-step, not per-trajectory). Derivation:
    beta = 0.7, 20000 envs: sigma = sqrt(0.7*0.3/20000) = 0.0032;
    bound 0.015 = 4.6 sigma. beta=1 and beta=0 are exact.
    """
    mask = np.asarray(mixed_execution_mask(jax.random.PRNGKey(2), 0.7, 20000))
    assert abs(mask.mean() - 0.7) < 0.015
    assert np.asarray(
        mixed_execution_mask(jax.random.PRNGKey(3), 1.0, 64)
    ).all()
    assert not np.asarray(
        mixed_execution_mask(jax.random.PRNGKey(4), 0.0, 64)
    ).any()
