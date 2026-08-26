"""Tests for the g_delta measurement behind the paper's gradient decomposition.

Checkpoint-free: the algebra is exercised on a tiny model and a synthetic
batch. The expensive part of ``analysis/gdelta.py`` is the checkpoint restore
and the on-policy rollout, and neither is what could be wrong.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tests.conftest import NUM_ACTIONS, OBS_DIM, PLAN_HORIZON, SEED, import_or_skip

gd = import_or_skip("experiments.rl_finetuning.analysis.gdelta")

WEIGHT_BATCH = 32


@pytest.fixture(scope="module")
def gd_config(real_ablations_config: dict) -> dict:
    return {
        "ADV_CLIP_EPS": real_ablations_config.get("ADV_CLIP_EPS", 0.2),
        "WIN_THRESHOLD": real_ablations_config.get("WIN_THRESHOLD", 0.5),
        "RETURN_WEIGHT_FLOOR": real_ablations_config.get("RETURN_WEIGHT_FLOOR", 0.1),
        "RETURN_WEIGHT_CAP": real_ablations_config.get("RETURN_WEIGHT_CAP", 5.0),
    }


@pytest.fixture(scope="module")
def returns() -> jnp.ndarray:
    """Sparse-reward returns: most windows earn nothing, a few earn a lot."""
    keys = jax.random.split(jax.random.PRNGKey(SEED), 2)
    win = jax.random.bernoulli(keys[0], 0.25, (WEIGHT_BATCH,))
    return jnp.where(win, jax.random.uniform(keys[1], (WEIGHT_BATCH,), minval=1.0,
                                             maxval=6.0), 0.0)


@pytest.fixture(scope="module")
def advantages(returns, gd_config):
    training = import_or_skip("experiments.rl_finetuning.ablations.training")
    adv, _, _ = training._compute_advantages(
        returns,
        gd_config["RETURN_WEIGHT_FLOOR"],
        gd_config["RETURN_WEIGHT_CAP"],
        wins_only=False,
        win_thresh=gd_config["WIN_THRESHOLD"],
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=jnp.array(0.0),
        running_std=jnp.array(1.0),
    )
    return adv


@pytest.fixture(scope="module")
def variants(advantages, returns, gd_config):
    return gd.build_variants(advantages, returns, gd_config, WEIGHT_BATCH)


# ---------------------------------------------------------------------------
# The weight vectors are the ones the trainer applies
# ---------------------------------------------------------------------------


def test_registry_still_matches_the_assumed_variants(variants):
    gd.verify_registry()
    assert set(variants) == set(gd.REGISTRY_RULES)


@pytest.mark.parametrize("variant", ["advantage_clip", "normalized_adv", "bc_wins"])
def test_variant_matches_the_loss_the_trainer_runs(
    variant, variants, advantages, returns, gd_config, monkeypatch,
    tiny_config, apply_fns, params, schedules
):
    """Each reconstructed weight vector equals the one its loss factory applies.

    ``_core_loss`` is the single point where every factory hands its transformed
    weights to ``compute_loss``, so capturing its argument reads the trainer's
    own vector rather than a re-implementation of it.
    """
    losses = import_or_skip("experiments.rl_finetuning.ablations.losses")
    registry = import_or_skip("experiments.rl_finetuning.ablations.registry")

    captured = {}

    def capture(ctx, p, rng, acts, obs, valid, adv, *a, **k):
        captured["w"] = adv
        return jnp.array(0.0)

    monkeypatch.setattr(losses, "_core_loss", capture)

    _, apply_train = apply_fns
    schedule_fn, schedule_deriv_fn = schedules
    ctx = losses.LossContext(
        apply_fn=apply_train,
        ref_params=params,
        schedule_fn=schedule_fn,
        schedule_deriv_fn=schedule_deriv_fn,
        num_actions=NUM_ACTIONS,
        config={**tiny_config, **gd_config},
    )
    name, _, wins_only = gd.REGISTRY_RULES[variant]
    loss_fn = registry.REGISTRY[name].loss_factory(ctx)

    # The trainer hands the loss whatever _compute_advantages returned under
    # this ablation's wins_only flag, not the baseline vector.
    training = import_or_skip("experiments.rl_finetuning.ablations.training")
    trainer_adv, _, _ = training._compute_advantages(
        returns,
        gd_config["RETURN_WEIGHT_FLOOR"],
        gd_config["RETURN_WEIGHT_CAP"],
        wins_only=wins_only,
        win_thresh=gd_config["WIN_THRESHOLD"],
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=jnp.array(0.0),
        running_std=jnp.array(1.0),
    )
    dummy = jnp.zeros((WEIGHT_BATCH, PLAN_HORIZON), dtype=jnp.int32)
    loss_fn(params, dummy, jnp.zeros((WEIGHT_BATCH, OBS_DIM)),
            jnp.ones((WEIGHT_BATCH,), dtype=bool),
            jax.random.PRNGKey(SEED), trainer_adv)

    assert jnp.allclose(captured["w"], variants[variant], atol=1e-5)


# ---------------------------------------------------------------------------
# delta and (A1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "variant", ["baseline_clipped_ratio", "advantage_clip", "bc_wins"]
)
def test_delta_is_zero_mean_where_a1_holds(variant, variants):
    delta, abar, a1_holds = gd.centred_delta(variants[variant])
    assert a1_holds
    assert abar > 0.0
    assert float(jnp.abs(jnp.mean(delta))) < 1e-5


def test_normalized_adv_is_flagged_as_a1_violating(variants):
    delta, abar, a1_holds = gd.centred_delta(variants["normalized_adv"])
    assert not a1_holds
    assert abs(abar) < 1e-5
    # The fallback centres rather than dividing by a vanishing mean.
    assert bool(jnp.all(jnp.isfinite(delta)))


def test_cv_a_and_ess_agree(variants):
    """ESS/B == 1 / (1 + CV_A^2) exactly, which is Eq. 5 of the paper."""
    for name in ["baseline_clipped_ratio", "advantage_clip", "bc_wins"]:
        delta, _, _ = gd.centred_delta(variants[name])
        cv_a = float(jnp.sqrt(jnp.mean(delta ** 2)))
        assert gd.effective_sample_size(variants[name]) == pytest.approx(
            1.0 / (1.0 + cv_a ** 2), rel=1e-4
        )


# ---------------------------------------------------------------------------
# The decomposition itself
# ---------------------------------------------------------------------------


def test_decomposition_identity(variants, tiny_config, apply_fns, params, schedules):
    """grad L_RW == Abar * (grad L_BC + g_delta) under a shared (z_t, t) draw."""
    compute_loss = import_or_skip("src.diffusion.loss").compute_loss
    _, apply_train = apply_fns
    schedule_fn, schedule_deriv_fn = schedules

    keys = jax.random.split(jax.random.PRNGKey(SEED + 1), 3)
    acts = jax.random.randint(keys[0], (WEIGHT_BATCH, PLAN_HORIZON), 0, NUM_ACTIONS)
    obs = jax.random.normal(keys[1], (WEIGHT_BATCH, OBS_DIM))
    valid = jnp.ones((WEIGHT_BATCH,), dtype=bool)
    key = keys[2]

    def gradient(weights):
        def loss(p):
            value, _ = compute_loss(
                apply_train, p, key, acts, obs, valid, NUM_ACTIONS,
                schedule_fn, schedule_deriv_fn,
                sigma_t=tiny_config.get("TRAIN_SIGMA", 0.0),
                label_smoothing=tiny_config.get("LABEL_SMOOTHING", 0.0),
                advantages=weights,
            )
            return value
        grads = jax.tree.leaves(jax.grad(loss)(params))
        return jnp.concatenate([x.ravel() for x in grads])

    weights = variants["baseline_clipped_ratio"]
    delta, abar, a1_holds = gd.centred_delta(weights)
    assert a1_holds

    g_bc = gradient(None)
    g_delta = gradient(delta)
    g_rw = gradient(weights)

    residual = float(jnp.linalg.norm(g_rw - abar * (g_bc + g_delta))
                     / jnp.linalg.norm(g_rw))
    assert residual < 1e-4


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _fake_seed_blob(seed: int, ratio: float, cos: float, abar: float) -> dict:
    return {
        "aggregate": False, "seed": seed, "n_draws": 8, "n_params": 100,
        "random_cos_sd": 0.1, "batch": 32, "bc_self_cos_mean": 0.9,
        "bc_self_cos_std": 0.01, "eq4_residual_max": 1e-5 * (seed + 1),
        "variants": {
            "baseline_clipped_ratio": {
                "cv_a": 1.0, "abar": abar, "abar_ratio_to_baseline": 1.0,
                "ess_fraction": 0.5, "a1_violated": False,
                "ratio_mean": ratio, "ratio_std_draws": 0.5,
                "cos_mean": cos, "cos_std_draws": 0.5,
                "ratio_shuffled_mean": 0.4, "ratio_shuffled_std": 0.01,
                "cos_shuffled_mean": 0.0, "cos_shuffled_std": 0.01,
            }
        },
    }


def test_aggregate_reports_dispersion_across_seeds():
    blobs = [
        _fake_seed_blob(seed, ratio, cos, abar)
        for seed, (ratio, cos, abar) in enumerate(
            [(0.40, 0.00, 0.30), (0.50, 0.02, 0.32), (0.60, 0.04, 0.34)]
        )
    ]

    agg = gd.aggregate(blobs)
    rec = agg["variants"]["baseline_clipped_ratio"]

    assert agg["n_seeds"] == 3
    assert agg["seeds"] == [0, 1, 2]
    assert rec["ratio_mean"] == pytest.approx(0.50)
    # Across seeds, not the 0.5 each seed reported across its own draws.
    assert rec["ratio_std_seeds"] == pytest.approx(0.0816, abs=1e-3)
    assert rec["cos_std_seeds"] == pytest.approx(0.0163, abs=1e-3)
    assert rec["abar_mean"] == pytest.approx(0.32)
    assert agg["eq4_residual_max"] == pytest.approx(3e-5)


def test_aggregate_refuses_an_aggregate_input():
    with pytest.raises(ValueError, match="already an aggregate"):
        gd.aggregate([{"aggregate": True, "variants": {}}])
