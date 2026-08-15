"""Value-level pins for the canonical scientific recipe (step 8).

Sources: research/spec-method.md §7 (method parameters), research/
spec-config.md §2/§4/§5.2 (PRIMARY/LEGACY resolution, documented
resolved quantities, Full-Craftax 11-key delta), research/traceability.md
§3 (budget arithmetic). Every expected value is transcribed from those
pinned loci or derived in the docstring — never read back from the
resolvers (the existing documented-quantities test in test_smoke_src.py
does that and is recorded as SELF-ORACLE in the step-8 audit).

xfail(strict=True) marks assertions of canonical behaviour that the
defect register says the implementation currently violates; they must
start failing loudly the moment step 9 fixes the defect.
"""

from __future__ import annotations

import math

import pytest
import yaml

from conftest import ROOT, load_config

# ---------------------------------------------------------------------------
# Method recipe values (spec-method §7, craftax column; anchors
# defaults.yaml:44-60)
# ---------------------------------------------------------------------------

_CLASSIC_METHOD_PINS = {
    "DIFFUSION_SCHEDULE": "cosine",
    "DIFFUSION_STEPS": 15,
    "DIFFUSION_STEPS_EVAL": 10,
    "REMASK_STRATEGY": "rescale",
    "ETA": 0.5,
    "USE_LOOP": True,
    "T_ON": 0.7,
    "T_OFF": 0.3,
    "TEMPERATURE": 0.5,
    "TOP_P": 0.95,
    "PLAN_HORIZON": 32,
    "TRAIN_SIGMA": 0.0,
    "LABEL_SMOOTHING": 0.0,
    "D_MODEL": 384,
    "N_HEADS": 8,
    "N_LAYERS": 6,
    "D_FF": 768,
    "OBS_ENCODER_WIDTH": 768,
}


def test_classic_recipe_method_values():
    """defaults.yaml IS the Classic paper recipe; these are its
    scientific method parameters per spec-method §7 (craftax column)."""
    config = load_config("configs/defaults.yaml")
    for key, expected in _CLASSIC_METHOD_PINS.items():
        assert config[key] == expected, f"{key}: {config[key]} != {expected}"


# Full Craftax departs from Classic on 11 keys duplicated verbatim in
# both cluster siblings (spec-config §5.2; final_craftax_ucl.yaml:6-9).
_FULL_CRAFTAX_DELTA = {
    "ENV_NAME": "Craftax-Symbolic-v1",
    "DIFFUSION_STEPS": 25,
    "TEMPERATURE": 0.3,
    "LR": 5e-4,
    "LR_WARMUP_FRAMES": 7.86432e7,
    "OFFLINE_TOTAL_TIMESTEPS": 1.0e8,
    "ONLINE_TOTAL_TIMESTEPS": 1.0e8,
    "DAGGER_BETA_FINAL": 0.385,
    "DAGGER_BUFFER_CYCLES": 0.76294,
    "VAL_DIFFUSION_STEPS": 25,
    "VAL_REPLAN_EVERY": 2,
}


@pytest.mark.parametrize("preset", ["final_craftax_ucl", "final_craftax_qmul"])
def test_full_craftax_recipe_deltas(preset):
    """The Full-Craftax recipe's 11 documented departure keys carry the
    spec-config §5.2 values in both cluster siblings."""
    raw = {
        k.upper(): v
        for k, v in yaml.safe_load(
            (ROOT / f"configs/{preset}.yaml").read_text()
        ).items()
    }
    for key, expected in _FULL_CRAFTAX_DELTA.items():
        got = raw[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            # PyYAML 1.1 leaves "7.86432e7"-style scientific notation as a
            # string (the caveat documented at src/planners/common.py:197);
            # the loaders float() it, so compare numerically here too.
            got = float(got)
            assert got == pytest.approx(expected), (
                f"{preset}: {key}: {got} != {expected}"
            )
        else:
            assert got == expected, f"{preset}: {key}: {got!r} != {expected!r}"


# ---------------------------------------------------------------------------
# Budget arithmetic: the resolver implements the documented formula
# (independent hand arithmetic, unlike the SELF-ORACLE derivations table)
# ---------------------------------------------------------------------------

_FINALS = [
    "final_classic_ucl", "final_classic_qmul",
    "final_craftax_ucl", "final_craftax_qmul",
]


def _resolved(preset):
    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {
        **load_config("configs/defaults.yaml"),
        **load_config(f"configs/{preset}.yaml"),
    }
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")
    return config


@pytest.mark.parametrize("preset", _FINALS)
def test_num_updates_formula_and_resnap(preset):
    """NUM_UPDATES = max(1, floor(online_total_timesteps / (num_envs *
    num_steps))) and the budget key is re-snapped to NUM_UPDATES * fpu.

    Source: spec-config §4 shared formula; §Amendments 1 (max(1,.) floor
    and re-snap). The expected value is computed here from the raw YAML
    numbers, independently of the resolver.
    """
    raw_defaults = load_config("configs/defaults.yaml")
    raw = {**raw_defaults, **load_config(f"configs/{preset}.yaml")}
    fpu = int(raw["NUM_ENVS"]) * int(raw["NUM_STEPS"])
    expected_updates = max(1, int(float(raw["ONLINE_TOTAL_TIMESTEPS"])) // fpu)

    config = _resolved(preset)
    assert int(config["NUM_UPDATES"]) == expected_updates
    assert int(config["ONLINE_TOTAL_TIMESTEPS"]) == expected_updates * fpu


@pytest.mark.parametrize(
    ("preset", "n_documented", "decay_documented"),
    [
        ("final_classic_qmul", 8138, 0.9998689),
        ("final_classic_ucl", 1525, 0.9993005),
    ],
)
def test_classic_beta_decay_resolves_to_documented(preset, n_documented, decay_documented):
    """beta decay = (beta_final / beta_init)^(1/N) at the documented N.

    Source: spec-config §2 (resolution rule) and §4 (documented
    quantities: 8138 updates / decay 0.9998689 on QMUL, 1525 / 0.9993005
    on UCL). Hand derivation: 0.344^(1/8138) = exp(-1.067114/8138) =
    0.99986888; 0.344^(1/1525) = exp(-1.067114/1525) = 0.99930049.
    """
    config = _resolved(preset)
    assert int(config["NUM_UPDATES"]) == n_documented
    assert math.isclose(
        float(config["DAGGER_BETA_DECAY"]), decay_documented, abs_tol=5e-8
    )
    # cross-check the closed form written above
    assert math.isclose(
        float(config["DAGGER_BETA_DECAY"]),
        0.344 ** (1.0 / n_documented),
        abs_tol=1e-12,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.8: final_craftax_* ship online_total_timesteps=1e8 "
        "but the documented quantities (5231 / 36621 updates, decays "
        "0.9998175 / 0.9999739) are consistent only with a 3e8 budget; "
        "which is canonical is a step-9 decision"
    ),
)
@pytest.mark.parametrize(
    ("preset", "n_documented", "decay_documented"),
    [
        ("final_craftax_ucl", 5231, 0.9998175),
        ("final_craftax_qmul", 36_621, 0.9999739),
    ],
)
def test_full_craftax_beta_decay_resolves_to_documented(
    preset, n_documented, decay_documented
):
    """Canonical (documented) Full-Craftax resolutions per spec-config §4.

    Hand derivation: 0.385^(1/5231) = exp(-0.954512/5231) = 0.99981754;
    0.385^(1/36621) = exp(-0.954512/36621) = 0.99997394.
    """
    config = _resolved(preset)
    assert int(config["NUM_UPDATES"]) == n_documented
    assert math.isclose(
        float(config["DAGGER_BETA_DECAY"]), decay_documented, abs_tol=5e-8
    )


@pytest.mark.parametrize(
    ("preset", "warmup_steps", "buffer_max"),
    [
        ("final_classic_ucl", 1600, 125_000),
        ("final_classic_qmul", 8533, 23_438),
        ("final_craftax_ucl", 1371, 43_750),
        ("final_craftax_qmul", 9600, 6_250),
    ],
)
def test_budget_independent_quantities(preset, warmup_steps, buffer_max):
    """Warmup-step and buffer resolutions (budget-independent).

    Source: spec-config §4 documented quantities. Hand derivations:
    warmup = floor(lr_warmup_frames / fpu): 1.048576e8/65536 = 1600,
    1.048576e8/12288 = 8533.3 -> 8533; 7.86432e7/57344 = 1371.4 -> 1371;
    7.86432e7/8192 = 9600. buffer = round(cycles * fpu):
    1.90735*65536 = 125000.1 -> 125000; 1.90735*12288 = 23437.5 -> 23438;
    0.76294*57344 = 43750.0 -> 43750; 0.76294*8192 = 6250.0 -> 6250.
    (The *unit* of the warmup count is defective — see the xfail below —
    but these are the documented resolutions of the shipped formula.)
    """
    config = _resolved(preset)
    assert int(config["LR_WARMUP_STEPS"]) == warmup_steps
    assert int(config["DAGGER_BUFFER_MAX"]) == buffer_max


# ---------------------------------------------------------------------------
# Warmup units (defect §8.11) and the short-budget crash floor (finding N1)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.11 (author decision 2026-08-15): warmup is "
        "environment-frame denominated, but the implemented conversion "
        "frames//fpu is an update count consumed by optax as gradient "
        "steps - warmup covers 1/64 of the intended frames"
    ),
)
@pytest.mark.parametrize("preset", _FINALS)
def test_warmup_step_count_covers_the_configured_frames(preset):
    """The resolved warmup count, read in the unit optax consumes it
    (gradient steps), must cover lr_warmup_frames environment frames.

    Source: spec-training §5 warmup row (author decision: frame-
    denominated intent, deterministic conversion via the effective
    geometry). Derivation: one optimiser update consumes fpu frames and
    performs update_epochs * num_minibatches * n_train_passes gradient
    steps, so frames per gradient step = fpu / geometry, and a
    frame-faithful warmup count satisfies
    |steps * fpu / geometry - lr_warmup_frames| < fpu (one update of
    slack, rounding-agnostic). The shipped frames//fpu count is 64x too
    small for every final config.
    """
    config = _resolved(preset)
    fpu = int(config["NUM_ENVS"]) * int(config["NUM_STEPS"])
    geometry = (
        int(config["UPDATE_EPOCHS"])
        * int(config["NUM_MINIBATCHES"])
        * int(config.get("N_TRAIN_PASSES") or config.get("DAGGER_TRAIN_PASSES") or 1)
    )
    frames_covered = int(config["LR_WARMUP_STEPS"]) * fpu / geometry
    assert abs(frames_covered - float(config["LR_WARMUP_FRAMES"])) < fpu


@pytest.mark.xfail(
    strict=True,
    reason=(
        "step-7 finding N1 (verification/2026-08-15-executable-baseline.md): "
        "a budget below the warmup floor crashes inside optax "
        "(decay_steps <= 0) instead of failing fast with a clear message; "
        "step 9 adds a guard alongside the §8.11 unit fix"
    ),
)
def test_short_budget_fails_fast_when_warmup_exceeds_it():
    """A budget smaller than the warmup must be rejected with an
    informative error at config resolution, not crash deep in optax.

    Source: step-7 finding N1 classification (correctness defect: missing
    guard, symptom of §8.11). Reproduction: final_craftax_ucl with
    online_total_timesteps = one update (57344 frames) resolves
    LR_WARMUP_STEPS=1371 > 64 total gradient steps and later dies with
    optax's `decay_steps=-1243`.
    """
    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {
        **load_config("configs/defaults.yaml"),
        **load_config("configs/final_craftax_ucl.yaml"),
    }
    config["ONLINE_TOTAL_TIMESTEPS"] = 448 * 128  # one update
    with pytest.raises(ValueError, match="[Ww]armup"):
        resolve_num_updates(config, "online")
        resolve_scaled_hyperparams(config, "online")
