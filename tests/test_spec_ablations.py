"""Per-ablation behavioural spec tests (step 8).

One deterministic behavioural test per ablation mechanism of
research/spec-ablations.md §2, with expected values from the pinned
sources (SPG, Jaques 2017, Kirkpatrick 2017, Sun 2019, Hu 2021,
Yu 2020, Kim 2025) or derivations written in the docstrings. The
group-C trainable-set tests reuse the step-7 reproduction method
(verification/2026-08-15-executable-baseline.md §3): apply the
registry optimizer to all-ones gradients and classify parameters by
non-zero update. xfail(strict=True) marks canonical-vs-implemented
disagreements from the defect register or the step-8 findings list.

The minihack twin file carries the same mechanisms in its framework.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import SEED

from experiments.rl_finetuning.ablations.losses import (
    LossContext,
    _ewc_penalty,
    make_loss_advantage_clip,
    make_loss_baseline,
    make_loss_bc_wins,
    make_loss_entropy_bonus,
    make_loss_ewc,
    make_loss_kl_penalty,
    make_loss_low_t,
    make_loss_normalized_adv,
    make_loss_t_curriculum,
    make_loss_t_curriculum_jit,
    make_loss_trust_region_kl,
)
from experiments.rl_finetuning.ablations.optimizers import (
    gradient_surgery,
    make_lora_params,
    merge_lora_into_base,
)
from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import (
    _build_reward_model,
    _compute_advantages,
    _init_replay_buffer,
    _push_to_buffer,
    _reward_model_train_step,
)
from src.diffusion.schedules import SCHEDULE_MAP
from src.planners.model import build_model, init_params

V, H, OBS = 4, 4, 8
B = 4
TINY = {
    "D_MODEL": 16, "N_HEADS": 2, "N_LAYERS": 2, "D_FF": 16,
    "OBS_ENCODER_LAYERS": 1, "OBS_ENCODER_WIDTH": 16, "PLAN_HORIZON": H,
}
LINEAR = SCHEDULE_MAP["linear"]
COSINE = SCHEDULE_MAP["cosine"]


def _uniform_ctx(config=None, horizon=H, vocab=V):
    """LossContext with a uniform-logits stub model (params ignored)."""

    def apply_fn(params, obs, z, t, rng):
        return jnp.zeros((obs.shape[0], horizon, vocab))

    return LossContext(
        apply_fn=apply_fn, ref_params=None, schedule_fn=LINEAR[0],
        schedule_deriv_fn=LINEAR[1], num_actions=vocab,
        config=config or {},
    )


def _logit_ctx(ref_logits, config=None, horizon=H):
    """LossContext whose stub model broadcasts `params` as the logits.

    ref_params holds the reference distribution's logits, so
    KL(current || pretrained) has the closed form used in the tests.
    """

    def apply_fn(params, obs, z, t, rng):
        return jnp.broadcast_to(params, (obs.shape[0], horizon, params.shape[-1]))

    return LossContext(
        apply_fn=apply_fn, ref_params=jnp.asarray(ref_logits),
        schedule_fn=LINEAR[0], schedule_deriv_fn=LINEAR[1],
        num_actions=ref_logits.shape[-1], config=config or {},
    )


def _batch(b=B, horizon=H, obs_dim=OBS, vocab=V, key=1):
    k = jax.random.PRNGKey(key)
    acts = jax.random.randint(k, (b, horizon), 0, vocab)
    obs = jnp.zeros((b, obs_dim))
    return acts, obs, jnp.ones(b)


# ---------------------------------------------------------------------------
# baseline_rl and the advantage pipeline (spec-ablations §2 baseline row)
# ---------------------------------------------------------------------------


def test_compute_advantages_standard_branch_closed_form():
    """weight = clip(max(R,0)/(mean(max(R,0))+eps), 0.1, 5.0).

    Source: spec-ablations §2 baseline_rl effective params (SPG eq (5)
    positive branch with the pinned deviations). Derivation: returns
    [0,1,2,3] -> clipped mean 1.5 -> raw weights [0, 2/3, 4/3, 2] ->
    floor lifts the first to 0.1.
    """
    adv, mean, std = _compute_advantages(
        jnp.array([0.0, 1.0, 2.0, 3.0]), 0.1, 5.0, wins_only=False,
        win_thresh=0.5, use_running_stats=False, ema_decay=0.99,
        running_mean=jnp.zeros(()), running_std=jnp.ones(()),
    )
    assert np.allclose(np.asarray(adv), [0.1, 2 / 3, 4 / 3, 2.0], atol=1e-4)
    assert float(mean) == pytest.approx(1.5, abs=1e-6)
    assert float(std) == pytest.approx(math.sqrt(1.25), abs=1e-5)


def test_compute_advantages_running_stats_branch_closed_form():
    """running_stats: EMA of batch mean/std, adv = clip((w-mu)/sigma + 1,
    0.1, 5.0).

    Source: spec-ablations §2 running_stats row. Derivation with
    ema_decay d=0.5, prior mean 0 / std 1, batch [0,1,2,3]:
    new_mean = 0.5*0 + 0.5*1.5 = 0.75;
    new_std = 0.5*1 + 0.5*(sqrt(1.25)) = 1.0590;
    adv_i = clip((w_i - 0.75)/1.0590 + 1, 0.1, 5) =
    [0.2918, 1.2361, 2.1804, 3.1246].
    """
    adv, mean, std = _compute_advantages(
        jnp.array([0.0, 1.0, 2.0, 3.0]), 0.1, 5.0, wins_only=False,
        win_thresh=0.5, use_running_stats=True, ema_decay=0.5,
        running_mean=jnp.zeros(()), running_std=jnp.ones(()),
    )
    new_std = 0.5 * 1.0 + 0.5 * math.sqrt(1.25)
    expected = np.clip((np.array([0, 1, 2, 3.0]) - 0.75) / new_std + 1.0, 0.1, 5.0)
    assert np.allclose(np.asarray(adv), expected, atol=1e-4)
    assert float(mean) == pytest.approx(0.75, abs=1e-6)
    assert float(std) == pytest.approx(new_std, abs=1e-5)


def test_baseline_loss_is_linear_in_the_advantages():
    """The advantage weight is a per-sample multiplier on the ELBO
    (SPG eq (5) positive branch: A * L_ELBO), so scaling every
    advantage by 2 exactly doubles the loss under the same RNG."""
    ctx = _uniform_ctx()
    loss_fn = make_loss_baseline(ctx)
    acts, obs, valid = _batch()
    rng = jax.random.PRNGKey(SEED)
    adv = jnp.array([0.5, 1.0, 1.5, 2.0])
    l1 = float(loss_fn(None, acts, obs, valid, rng, adv))
    l2 = float(loss_fn(None, acts, obs, valid, rng, 2.0 * adv))
    assert l2 == pytest.approx(2.0 * l1, rel=1e-6)


# ---------------------------------------------------------------------------
# bc_wins (defect §8.5)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.5: make_loss_bc_wins discards the advantages "
        "argument (losses.py:338-339) and no caller pre-filters, so the "
        "win mask never reaches the loss - bc_wins trains a uniform ELBO "
        "on all windows"
    ),
)
def test_bc_wins_loss_is_zero_on_an_all_losing_batch():
    """Canonical bc_wins ('Uniform ELBO on win windows', win = return >
    win_threshold, spec-ablations §2): a batch with no winning window
    carries no training signal, so the loss must be 0.

    The win mask [0,0,0,0] is produced by the pipeline's own
    _compute_advantages(wins_only=True) on an all-losing batch
    (returns below win_thresh=0.5), exactly as in the training loop.
    """
    ctx = _uniform_ctx()
    loss_fn = make_loss_bc_wins(ctx)
    acts, obs, valid = _batch()
    win_mask, _, _ = _compute_advantages(
        jnp.array([0.0, 0.1, 0.2, 0.3]), 0.1, 5.0, wins_only=True,
        win_thresh=0.5, use_running_stats=False, ema_decay=0.99,
        running_mean=jnp.zeros(()), running_std=jnp.ones(()),
    )
    assert np.allclose(np.asarray(win_mask), 0.0)
    loss = float(loss_fn(None, acts, obs, valid, jax.random.PRNGKey(SEED), win_mask))
    assert loss == 0.0


# ---------------------------------------------------------------------------
# advantage_clip / normalized_adv (spec-ablations §2)
# ---------------------------------------------------------------------------


def test_advantage_clip_clips_the_weight_to_the_documented_band():
    """advantage_clip clips the return-weight itself to [1-eps, 1+eps]
    (project-specific; PPO's ratio clip rejected as source per
    SOURCES.md). With eps=0.2 the ablation loss equals the baseline
    loss on manually clipped advantages, under the same RNG."""
    ctx = _uniform_ctx({"ADV_CLIP_EPS": 0.2})
    acts, obs, valid = _batch()
    rng = jax.random.PRNGKey(SEED)
    adv = jnp.array([10.0, 0.0, 1.0, 1.1])
    got = float(make_loss_advantage_clip(ctx)(None, acts, obs, valid, rng, adv))
    want = float(
        make_loss_baseline(ctx)(None, acts, obs, valid, rng, jnp.clip(adv, 0.8, 1.2))
    )
    assert got == pytest.approx(want, abs=0.0)


def test_normalized_adv_standardises_over_the_batch():
    """normalized_adv applies (A - mean)/(std + 1e-8) over the batch
    (What Matters C67; spec-ablations §2). Equals the baseline loss on
    manually standardised advantages under the same RNG."""
    ctx = _uniform_ctx()
    acts, obs, valid = _batch()
    rng = jax.random.PRNGKey(SEED)
    adv = jnp.array([10.0, 0.0, 1.0, 1.1])
    norm = (adv - adv.mean()) / (adv.std() + 1e-8)
    got = float(make_loss_normalized_adv(ctx)(None, acts, obs, valid, rng, adv))
    want = float(make_loss_baseline(ctx)(None, acts, obs, valid, rng, norm))
    assert got == pytest.approx(want, abs=0.0)


# ---------------------------------------------------------------------------
# kl_penalty / trust_region_kl (Jaques 2017 family)
# ---------------------------------------------------------------------------

_P_LOGITS = jnp.log(jnp.array([0.7, 0.1, 0.1, 0.1]))
_Q_LOGITS = jnp.zeros(4)  # uniform reference
# KL(p || uniform) = sum p ln(4p) = 0.7 ln 2.8 + 0.3 ln 0.4
_KL_PQ = 0.7 * math.log(2.8) + 0.3 * math.log(0.4)
_P1_LOGITS = jnp.log(jnp.array([0.55, 0.15, 0.15, 0.15]))
_KL_P1Q = 0.55 * math.log(2.2) + 0.45 * math.log(0.6)

_KL_H = 512  # with H=512, P(a row draws no masked position) = E[alpha^H]
# ~ 1/513 per row under t~U(eps,1) linear, so all 4 rows almost surely
# contribute the constant per-position KL and the masked average is exact.


def test_kl_penalty_adds_coef_times_the_closed_form_kl():
    """kl_penalty adds kl_coef * KL(current || pretrained) on masked
    positions (Jaques 2017 eqs (2)-(4); spec-ablations §2 kl_penalty).

    Derivation: constant per-position logits make the masked-position
    KL equal KL(p||q) = 0.7 ln 2.8 + 0.3 ln 0.4 = 0.445846 in every row
    with at least one mask. Loss difference between kl_coef 0.3 and 0.1
    under identical RNG isolates 0.2 * KL.
    """
    acts, _, valid = _batch(horizon=_KL_H)
    obs = jnp.zeros((B, OBS))
    rng = jax.random.PRNGKey(SEED)
    losses = {}
    for coef in (0.1, 0.3):
        ctx = _logit_ctx(_Q_LOGITS, {"KL_COEF": coef}, horizon=_KL_H)
        losses[coef] = float(
            make_loss_kl_penalty(ctx)(_P_LOGITS, acts, obs, valid, rng, jnp.ones(B))
        )
    got_kl = (losses[0.3] - losses[0.1]) / 0.2
    assert got_kl == pytest.approx(_KL_PQ, rel=1e-3)


def test_kl_penalty_is_zero_when_current_equals_pretrained():
    """KL(p||p) = 0: with params == ref the penalty vanishes, so the
    kl_penalty loss equals its own RL term (replicated with the
    factory's deterministic RNG split)."""
    from experiments.rl_finetuning.ablations.losses import _core_loss

    ctx = _logit_ctx(_Q_LOGITS, {"KL_COEF": 0.1}, horizon=_KL_H)
    acts, _, valid = _batch(horizon=_KL_H)
    obs = jnp.zeros((B, OBS))
    rng = jax.random.PRNGKey(SEED)
    core_rng, _ = jax.random.split(rng)
    got = float(
        make_loss_kl_penalty(ctx)(_Q_LOGITS, acts, obs, valid, rng, jnp.ones(B))
    )
    want = float(
        _core_loss(ctx, _Q_LOGITS, core_rng, acts, obs, valid, jnp.ones(B))
    )
    assert got == pytest.approx(want, abs=1e-6)


def test_trust_region_barrier_is_zero_below_and_quadratic_above():
    """trust_region_kl adds a quadratic barrier c*max(KL-delta,0)^2
    (spec-ablations §2 trust_region row; delta=0.05).

    Below the threshold (params == ref, KL=0) the barrier is exactly 0.
    Above it, the barrier for two KL levels K0=0.445846 and K1=0.203781
    (closed forms as in the KL test) satisfies the quadratic ratio
    ((K0-delta)/(K1-delta))^2 = (0.395846/0.153781)^2 = 6.6262 - this
    pins the barrier's form without pinning the project-specific c.
    """
    from experiments.rl_finetuning.ablations.losses import _core_loss

    acts, _, valid = _batch(horizon=_KL_H)
    obs = jnp.zeros((B, OBS))
    rng = jax.random.PRNGKey(SEED)
    core_rng, _ = jax.random.split(rng)

    def barrier(cur_logits):
        ctx = _logit_ctx(_Q_LOGITS, {"TRUST_REGION_KL": 0.05}, horizon=_KL_H)
        total = float(
            make_loss_trust_region_kl(ctx)(cur_logits, acts, obs, valid, rng, jnp.ones(B))
        )
        rl = float(_core_loss(ctx, cur_logits, core_rng, acts, obs, valid, jnp.ones(B)))
        return total - rl

    assert barrier(_Q_LOGITS) == pytest.approx(0.0, abs=1e-6)
    b0, b1 = barrier(_P_LOGITS), barrier(_P1_LOGITS)
    assert b0 > 0 and b1 > 0
    want_ratio = ((_KL_PQ - 0.05) / (_KL_P1Q - 0.05)) ** 2
    assert b0 / b1 == pytest.approx(want_ratio, rel=2e-2)


# ---------------------------------------------------------------------------
# ewc (Kirkpatrick 2017 eq (3), lambda-reparameterised per SOURCES.md)
# ---------------------------------------------------------------------------


def test_ewc_penalty_closed_form_and_factory_scaling():
    """EWC adds lambda * sum_i F_i (theta_i - theta*_i)^2.

    Source: Kirkpatrick 2017 eq (3); the repo folds the paper's 1/2
    into lambda (documented reparameterisation, spec-ablations §2).
    Derivation: F={a:[1,2]}, theta={a:[3,5]}, theta*={a:[1,1]} ->
    penalty = 1*(2^2) + 2*(4^2) = 36; with ewc_lambda=100 the factory
    loss exceeds the same-RNG core loss by exactly 3600.
    """
    fisher = {"a": jnp.array([1.0, 2.0])}
    theta = {"a": jnp.array([3.0, 5.0])}
    ref = {"a": jnp.array([1.0, 1.0])}
    assert float(_ewc_penalty(fisher, theta, ref)) == pytest.approx(36.0)

    from experiments.rl_finetuning.ablations.losses import _core_loss

    def apply_fn(params, obs, z, t, rng):
        return jnp.zeros((obs.shape[0], H, V))

    ctx = LossContext(
        apply_fn=apply_fn, ref_params=ref, schedule_fn=LINEAR[0],
        schedule_deriv_fn=LINEAR[1], num_actions=V,
        config={"EWC_LAMBDA": 100.0},
    )
    acts, obs, valid = _batch()
    rng = jax.random.PRNGKey(SEED)
    got = float(make_loss_ewc(ctx, fisher)(theta, acts, obs, valid, rng, jnp.ones(B)))
    rl = float(_core_loss(ctx, theta, rng, acts, obs, valid, jnp.ones(B)))
    assert got - rl == pytest.approx(3600.0, rel=1e-5)


# ---------------------------------------------------------------------------
# entropy_bonus (standard tier, cf. Mnih 2016)
# ---------------------------------------------------------------------------


def test_entropy_bonus_subtracts_coef_times_the_closed_form_entropy():
    """entropy_bonus subtracts entropy_coef * H(p_theta) on masked
    positions (spec-ablations §2). Derivation: constant per-position
    p=[0.7,0.1,0.1,0.1] gives H = -(0.7 ln 0.7 + 0.3 ln 0.1) = 0.940448
    (the masked average is globally normalised, so it is exact whenever
    any position is masked). The coefficient difference 0.03-0.01
    isolates -0.02 * H.
    """
    acts, obs, valid = _batch()
    rng = jax.random.PRNGKey(SEED)
    losses = {}
    for coef in (0.01, 0.03):
        ctx = _logit_ctx(_Q_LOGITS, {"ENTROPY_COEF": coef})
        losses[coef] = float(
            make_loss_entropy_bonus(ctx)(_P_LOGITS, acts, obs, valid, rng, jnp.ones(B))
        )
    entropy = -(0.7 * math.log(0.7) + 0.3 * math.log(0.1))
    assert (losses[0.01] - losses[0.03]) / 0.02 == pytest.approx(entropy, rel=1e-4)


# ---------------------------------------------------------------------------
# low_t / t_curriculum (Kim 2025; spec-ablations §2)
# ---------------------------------------------------------------------------


def _recording_ctx(config, records):
    def apply_fn(params, obs, z, t, rng):
        records.append(np.asarray(t))
        return jnp.zeros((obs.shape[0], H, V))

    return LossContext(
        apply_fn=apply_fn, ref_params=None, schedule_fn=LINEAR[0],
        schedule_deriv_fn=LINEAR[1], num_actions=V, config=config,
    )


def test_low_t_restricts_sampling_to_the_low_noise_regime():
    """low_t trains only on t in [eps, t_max_low=0.2]
    (spec-ablations §2 low_t row)."""
    records: list[np.ndarray] = []
    ctx = _recording_ctx({"T_MAX_LOW": 0.2}, records)
    acts, obs, valid = _batch(b=64)
    make_loss_low_t(ctx)(None, acts, obs, jnp.ones(64), jax.random.PRNGKey(0), jnp.ones(64))
    t = np.concatenate(records)
    assert t.min() >= 1e-5 - 1e-9 and t.max() <= 0.2 + 1e-6


@pytest.mark.parametrize("variant", ["list", "jit"])
def test_t_curriculum_anneals_high_noise_to_low_noise(variant):
    """t_curriculum anneals the t window from [0.8, 1.0] to
    [eps, 0.2] linearly over 200 iterations, high-noise (easy) first
    (Kim 2025, simplified linear anneal per SOURCES.md; params
    t_start=0.8, t_end=0.2, steps=200, spec-ablations §1.6).

    Expected windows: iter 0 -> [0.8, 1.0]; iter 100 (frac 0.5) ->
    [0.4, 0.6]; iter >= 200 -> [eps, 0.2].
    """
    config = {"T_CURRICULUM_START": 0.8, "T_CURRICULUM_END": 0.2,
              "T_CURRICULUM_STEPS": 200}
    acts, obs, _ = _batch(b=64)
    for it, (lo, hi) in [(0, (0.8, 1.0)), (100, (0.4, 0.6)), (200, (1e-5, 0.2))]:
        records: list[np.ndarray] = []
        ctx = _recording_ctx(config, records)
        if variant == "list":
            fn = make_loss_t_curriculum(ctx, [it])
            fn(None, acts, obs, jnp.ones(64), jax.random.PRNGKey(0), jnp.ones(64))
        else:
            fn = make_loss_t_curriculum_jit(ctx)
            fn(None, acts, obs, jnp.ones(64), jax.random.PRNGKey(0), jnp.ones(64),
               jnp.array(it))
        t = np.concatenate(records)
        assert t.min() >= lo - 1e-6, (it, t.min(), lo)
        assert t.max() <= hi + 1e-6, (it, t.max(), hi)


# ---------------------------------------------------------------------------
# llrd (Sun 2019: eta_{k-1} = xi * eta_k, top-down from the head)
# ---------------------------------------------------------------------------


def test_llrd_learning_rates_decay_geometrically_from_the_head():
    """LLRD gives the head base_lr and each layer at depth d from the
    top base_lr * decay^d; the observation encoder sits below the
    lowest block (Sun 2019; spec-ablations §2 llrd row, decay 0.9).

    Method: one AdamW step on all-ones gradients with weight decay 0
    and the norm clip disabled; the first-step AdamW update magnitude
    is lr * ghat/(sqrt(vhat)+eps) = lr/(1+1e-5) uniformly, so update
    ratios equal LR ratios. With N_LAYERS=2:
    head(Dense_4)=base; block_1=base*0.9; block_0=base*0.81;
    everything else 0.9^3.
    """
    model = build_model(TINY, V)
    params = init_params(model, jax.random.PRNGKey(SEED), OBS, H)
    config = {**TINY, "LR": 1e-3, "LLRD_DECAY": 0.9, "N_LAYERS": 2,
              "WEIGHT_DECAY": 0.0, "MAX_GRAD_NORM": 1e9}
    tx = REGISTRY["llrd"].optimizer_factory(config, params)
    state = tx.init(params)
    updates, _ = tx.update(jax.tree.map(jnp.ones_like, params), state, params)
    flat = jax.tree_util.tree_flatten_with_path(updates)[0]

    def group_of(path_str: str) -> str:
        if "TransformerBlock_0" in path_str:
            return "block_0"
        if "TransformerBlock_1" in path_str:
            return "block_1"
        if "Dense_4" in path_str:
            return "head"
        return "obs_enc"

    mags: dict[str, set[float]] = {}
    for path, leaf in flat:
        p = "/".join(str(k.key) for k in path)
        mags.setdefault(group_of(p), set()).add(round(float(jnp.abs(leaf).max()), 10))

    base = max(mags["head"])
    expected = {"head": 1.0, "block_1": 0.9, "block_0": 0.81, "obs_enc": 0.9**3}
    for group, rel in expected.items():
        got = max(mags[group])
        assert got / base == pytest.approx(rel, rel=1e-4), (group, got / base)


# ---------------------------------------------------------------------------
# lora (Hu 2021 eq (3): h = W0 x + (alpha/r) B A x, B zero-initialised)
# ---------------------------------------------------------------------------


def test_lora_delta_is_zero_at_init_low_rank_and_scaled_by_alpha_over_r():
    """LoRA per Hu 2021 eq (3)/§4: B=0 at init (delta exactly zero),
    the delta is (alpha/r)*A@B with rank <= r, on attention kernels.

    The delta expectation is recomputed with NumPy from the A/B factors,
    independently of merge_lora_into_base.
    """
    rank, alpha = 8, 16.0
    model = build_model(TINY, V)
    params = init_params(model, jax.random.PRNGKey(SEED), OBS, H)
    lora = make_lora_params(params, rank, jax.random.PRNGKey(7))
    assert lora, "no attention kernels matched"
    for ab in lora.values():
        assert np.allclose(np.asarray(ab["B"]), 0.0)

    merged0 = merge_lora_into_base(params, lora, alpha, rank)
    for (pa, la), (_pb, lb) in zip(
        jax.tree_util.tree_flatten_with_path(params)[0],
        jax.tree_util.tree_flatten_with_path(merged0)[0],
        strict=True,
    ):
        assert np.array_equal(np.asarray(la), np.asarray(lb)), pa

    perturbed = {
        k: {"A": ab["A"], "B": jnp.ones_like(ab["B"])} for k, ab in lora.items()
    }
    merged = merge_lora_into_base(params, perturbed, alpha, rank)
    flat_base = {
        "/".join(str(k.key) for k in path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    }
    flat_merged = {
        "/".join(str(k.key) for k in path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(merged)[0]
    }
    for path_str, ab in perturbed.items():
        a = np.asarray(ab["A"])
        bmat = np.asarray(ab["B"])
        want = (alpha / rank) * (a @ bmat)
        got = (
            np.asarray(flat_merged[path_str]) - np.asarray(flat_base[path_str])
        ).reshape(want.shape)
        assert np.allclose(got, want, atol=1e-5), path_str
        assert np.linalg.matrix_rank(got) <= rank


# ---------------------------------------------------------------------------
# gradient_surgery (Yu 2020 Alg 1, one-sided per SOURCES.md)
# ---------------------------------------------------------------------------


def test_pcgrad_projection_closed_form_and_one_sidedness():
    """PCGrad: if g_rl . g_bc < 0, g_rl <- g_rl - (g_rl.g_bc/|g_bc|^2) g_bc;
    otherwise unchanged (Yu 2020 Alg 1; one-sided variant per SOURCES.md).

    Derivation: g_rl=[1,0], g_bc=[-1,1]: dot=-1, |g_bc|^2=2 ->
    projected = [1,0] - (-1/2)[-1,1] = [0.5, 0.5], orthogonal to g_bc.
    Non-conflicting g_bc=[1,1] leaves g_rl untouched.
    """
    g_rl = {"w": jnp.array([1.0, 0.0])}
    out = gradient_surgery(g_rl, {"w": jnp.array([-1.0, 1.0])})
    assert np.allclose(np.asarray(out["w"]), [0.5, 0.5], atol=1e-6)
    assert float(out["w"] @ jnp.array([-1.0, 1.0])) == pytest.approx(0.0, abs=1e-6)
    out2 = gradient_surgery(g_rl, {"w": jnp.array([1.0, 1.0])})
    assert np.allclose(np.asarray(out2["w"]), [1.0, 0.0])


# ---------------------------------------------------------------------------
# mixed_replay (self-replay ring buffer; spec-ablations §2 + §5 res. 12)
# ---------------------------------------------------------------------------


def test_mixed_replay_ring_buffer_holds_the_runs_own_windows():
    """mixed_replay's buffer holds the run's own rollout windows
    (self-replay; the 'offline data' one-liner is recorded stale by
    traceability §5 res. 12) with ring-wrap FIFO semantics and a
    first-n push cap.

    Derivation: capacity 4, push 3 rows with returns [0,1,2] (write idx
    0..2), then 3 with [3,4,5] (indices 3,0,1) -> buffer returns
    [4,5,2,3]; count caps at 4. A push with n_new=2 takes only the
    first 2 rows of its batch.
    """
    buf = _init_replay_buffer(4, H, OBS)

    def rows(vals):
        n = len(vals)
        return (
            jnp.tile(jnp.arange(H, dtype=jnp.int32), (n, 1)),
            jnp.zeros((n, OBS)),
            jnp.ones(n, dtype=bool),
            jnp.array(vals, dtype=jnp.float32),
        )

    a, o, v, r = rows([0.0, 1.0, 2.0])
    buf = _push_to_buffer(buf, a, o, v, r, 3)
    a, o, v, r = rows([3.0, 4.0, 5.0])
    buf = _push_to_buffer(buf, a, o, v, r, 3)
    assert np.allclose(np.asarray(buf.returns), [4.0, 5.0, 2.0, 3.0])
    assert int(buf.count) == 4

    buf2 = _init_replay_buffer(4, H, OBS)
    a, o, v, r = rows([7.0, 8.0, 9.0])
    buf2 = _push_to_buffer(buf2, a, o, v, r, 2)
    assert int(buf2.count) == 2
    assert np.allclose(np.asarray(buf2.returns[:2]), [7.0, 8.0])


# ---------------------------------------------------------------------------
# reward_model (spec-ablations §2: MLP obs -> return, MSE)
# ---------------------------------------------------------------------------


def test_reward_model_learns_a_linear_return_map():
    """The reward model is an MLP regressor on (obs -> return) trained
    with MSE (spec-ablations §2 reward_model row: width 64, depth 2).
    50 gradient steps on a fixed linear target must cut the MSE by more
    than half (deterministic under the fixed seed).
    """
    _, rm_state = _build_reward_model(
        OBS, jax.random.PRNGKey(SEED), width=64, depth=2, lr=1e-3
    )
    k = jax.random.PRNGKey(3)
    obs = jax.random.normal(k, (64, OBS))
    targets = obs[:, 0] * 2.0 + 1.0
    _, loss0 = _reward_model_train_step(rm_state, obs, targets)
    state = rm_state
    for _ in range(50):
        state, loss = _reward_model_train_step(state, obs, targets)
    assert float(loss) < 0.5 * float(loss0)


# ---------------------------------------------------------------------------
# running_stats effective decay (defect §8.4) - step-7 closure method
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.4: training.py:721 reads RUNNING_STATS_EMA_DECAY "
        "and training.py:735 shadows the same variable with EMA_DECAY, so "
        "the running-stats advantage EMA runs at the eval-weights decay "
        "(0.999), not the configured 0.99"
    ),
)
def test_running_stats_uses_the_configured_decay():
    """The running_stats training closure must consume
    RUNNING_STATS_EMA_DECAY (spec-ablations §1.6: 0.99), not the
    eval-EMA decay. Reuses the step-7 reproduction: build the shipped
    make_run_ablation closure with the two decays set to distinct
    sentinels and read the ema_decay cell it captured.
    """
    from experiments.rl_finetuning.ablations.training import make_run_ablation
    from src.planners.model import make_apply_fns

    config = {
        **TINY, "NUM_ACTIONS": V, "MAX_ITER": 1, "NUM_ENVS": 2,
        "NUM_STEPS": 8, "BATCH_SIZE": 3, "EVAL_EVERY": 1, "USE_WANDB": False,
        "SEED": 0, "RUNNING_STATS_EMA_DECAY": 0.111, "EMA_DECAY": 0.999,
    }
    model = build_model(config, V)
    params = init_params(model, jax.random.PRNGKey(0), OBS, H)
    apply_eval, apply_train = make_apply_fns(model)
    run = make_run_ablation(
        spec=REGISTRY["running_stats"], config=config, pretrained_params=params,
        apply_train=apply_train, apply_eval=apply_eval, env=None, env_params=None,
        ppo=None, schedule_fn=COSINE[0], schedule_deriv_fn=COSINE[1],
        num_actions=V, obs_dim=OBS,
    )
    cells = dict(zip(run.__code__.co_freevars, run.__closure__ or (), strict=False))
    assert "ema_decay" in cells, "closure extraction failed"
    assert cells["ema_decay"].cell_contents == pytest.approx(0.111)


# ---------------------------------------------------------------------------
# Group C: trainable-parameter sets (defects §8.1/§8.2 + step-8 finding)
# step-7 reproduction method: ones-grads through the registry optimizer
# ---------------------------------------------------------------------------


def _trainable_modules(name: str) -> frozenset[str]:
    model = build_model(TINY, V)
    params = init_params(model, jax.random.PRNGKey(SEED), OBS, H)
    tx = REGISTRY[name].optimizer_factory({**TINY, "WEIGHT_DECAY": 0.0}, params)
    state = tx.init(params)
    updates, _ = tx.update(jax.tree.map(jnp.ones_like, params), state, params)
    flat = jax.tree_util.tree_flatten_with_path(updates)[0]
    modules = set()
    for path, leaf in flat:
        p = "/".join(str(k.key) for k in path)
        if bool(jnp.any(leaf != 0)):
            modules.add(p.rsplit("/", 1)[0])
    return frozenset(modules)


def _all_modules() -> frozenset[str]:
    model = build_model(TINY, V)
    params = init_params(model, jax.random.PRNGKey(SEED), OBS, H)
    flat = jax.tree_util.tree_flatten_with_path(params)[0]
    return frozenset(
        "/".join(str(k.key) for k in path).rsplit("/", 1)[0] for path, _ in flat
    )


# Tiny-arch module map (OBS_ENCODER_LAYERS=1, N_LAYERS=2): Dense_0 obs
# encoder, Dense_1 obs projection, Dense_2/Dense_3 time embedding,
# Dense_4 action head, Embed_0 action embedding, LayerNorm_0/1 top level.
_HEAD = frozenset({"params/Dense_4"})


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.2: frozen_backbone trains the obs projection, "
        "time-embedding Denses and the action embedding besides the head "
        "(registry.py:98-107)"
    ),
)
def test_frozen_backbone_trains_only_the_output_head():
    """Docs: 'Only train the output head' (spec-ablations §2)."""
    assert _trainable_modules("frozen_backbone") == _HEAD


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.2: head_only leaves the time-embedding Denses "
        "trainable besides the head (registry.py:110-122)"
    ),
)
def test_head_only_trains_only_the_final_projection():
    """Docs: 'Only train the final linear projection' (spec-ablations §2)."""
    assert _trainable_modules("head_only") == _HEAD


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.2: attention_only additionally trains the obs "
        "projection, time-embedding Denses, action embedding and the head "
        "(registry.py:125-131)"
    ),
)
def test_attention_only_trains_only_the_attention_projections():
    """Docs: 'Only train attention weights (Q/K/V/O)' (spec-ablations §2)."""
    expected = frozenset(
        m for m in _all_modules() if "MultiHeadDotProductAttention_" in m
    )
    assert _trainable_modules("attention_only") == expected


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.1: ffn_only's bare Dense_0/Dense_1 fragments also "
        "match the FFN Denses inside every TransformerBlock, freezing the "
        "very layers it claims to train (registry.py:134-143)"
    ),
)
def test_ffn_only_trains_only_the_ffn_layers():
    """Docs: 'Only train FFN layers' (spec-ablations §2). The FFN is
    the two Dense layers inside each TransformerBlock."""
    expected = frozenset(
        m
        for m in _all_modules()
        if "TransformerBlock_" in m and ("/Dense_0" in m or "/Dense_1" in m)
    )
    assert _trainable_modules("ffn_only") == expected


@pytest.mark.xfail(
    strict=True,
    reason=(
        "step-8 finding: the layer-ablation probes also train the obs "
        "projection, time-embedding Denses and top-level LayerNorms "
        "(registry.py:146-164 freezes only Dense_0/Dense_1/SinusoidalPosEmbed_/"
        "Embed_ outside the kept blocks) - same family as §8.2 but not in "
        "the register"
    ),
)
@pytest.mark.parametrize("top_n", [1, 2])
def test_layer_ablation_trains_only_the_top_blocks_and_head(top_n):
    """Docs: 'Train only the top-k transformer block(s) (+ head)'
    (spec-ablations §2 layer_ablation row). With N_LAYERS=2 the top-1
    set is TransformerBlock_1 + head; top-2 adds TransformerBlock_0.
    """
    kept = {f"params/TransformerBlock_{i}" for i in range(2 - top_n, 2)}
    expected = frozenset(
        m for m in _all_modules() if any(m.startswith(k) for k in kept)
    ) | _HEAD
    assert _trainable_modules(f"layer_ablation_top{top_n}") == expected


# ---------------------------------------------------------------------------
# Suite loss estimator (cross-repo twin; NELBO per spec-method §3.1/§3.4)
# ---------------------------------------------------------------------------


def test_suite_loss_uses_the_nelbo_weight_and_per_token_normalisation():
    """The suite's per-sample loss is the NELBO estimator
    w(t) * sum_masked(CE) / H (spec-method §3.1/§3.4; spec-ablations §2
    baseline row: 'return-weighted ELBO').

    Derivation (cosine schedule, uniform logits, t ~ U(eps, 0.2) via the
    low_t factory, unit advantages): E[per-sample] = E_t[w(t) ln V
    (1-alpha)/1] = ln V * E_t[-alpha'(t)] = ln V * (1 - cos(0.1 pi))/0.2
    = 0.244715 ln V. Statistical: B=16384 rows; per-row second moment
    is ~0.62 (ln V)^2 (w ~ 2/t and Bin(H=8, 1-alpha) masking), giving
    sigma ~ 0.0061 ln V; bound 0.03 ln V ~ 4.9 sigma. The minihack twin
    asserts the same value and xfails: its suite loss drops w(t) and
    normalises by the realised masked count (step-8 finding).
    """
    b = 16384

    def apply_fn(params, obs, z, t, rng):
        return jnp.zeros((obs.shape[0], H, V))

    ctx = LossContext(
        apply_fn=apply_fn, ref_params=None, schedule_fn=COSINE[0],
        schedule_deriv_fn=COSINE[1], num_actions=V, config={"T_MAX_LOW": 0.2},
    )
    k = jax.random.PRNGKey(2)
    acts = jax.random.randint(k, (b, H), 0, V)
    obs = jnp.zeros((b, 1))
    loss = float(
        make_loss_low_t(ctx)(None, acts, obs, jnp.ones(b), jax.random.PRNGKey(0),
                             jnp.ones(b))
    )
    expected = math.log(V) * (1 - math.cos(0.1 * math.pi)) / 0.2
    assert abs(loss - expected) < 0.03 * math.log(V)
