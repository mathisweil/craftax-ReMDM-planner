"""Config and ablation-suite mechanics spec tests (step 8).

Sources: research/spec-config.md §1.2 (override typing), §6.1
(checkpoint/config coupling), research/spec-ablations.md §1.1 (unknown
key = KeyError in the suite chain) and §Amendments 4/7, research/
traceability.md §8.6/§8.7. The minihack twin file pins the same
override-typing table (its validation is already covered by its own
test_config.py).
"""

from __future__ import annotations

import jax
import orbax.checkpoint as ocp
import pytest
from conftest import NUM_ACTIONS, OBS_DIM, PLAN_HORIZON, SEED, TINY_ARCH

from main import _cast_override, _validate_keys

# ---------------------------------------------------------------------------
# --override typing (spec-config §1.2: values cast to the key's type,
# "a typo is an error, not a silent no-op")
# ---------------------------------------------------------------------------


def test_override_values_cast_to_the_key_type():
    """Casting table per spec-config §1.2 (README.md:213).

    The sibling repo implements the same table but raises TypeError
    where this repo raises ValueError (PARITY 'Config-loader
    semantics'); each twin asserts its own documented exception type.
    """
    assert _cast_override("k", "0.3", 0.5) == 0.3
    assert _cast_override("k", "1e-4", 0.5) == pytest.approx(1e-4)
    assert _cast_override("k", "7", 3) == 7
    assert _cast_override("k", "7.0", 3) == 7
    assert _cast_override("k", "false", True) is False
    assert _cast_override("k", "text", "old") == "text"
    assert _cast_override("k", "null", 3) is None
    assert _cast_override("k", "3", None) == 3

    with pytest.raises(ValueError):
        _cast_override("k", "not_a_number", 3)
    with pytest.raises(ValueError):
        _cast_override("k", "2.5", 3)  # non-integral float for an int key
    with pytest.raises(ValueError):
        _cast_override("k", "maybe", True)


def test_unknown_override_key_is_an_error():
    """Unknown --override keys raise instead of silently no-oping
    (spec-config §1.2)."""
    with pytest.raises(ValueError, match="Unknown config key"):
        _validate_keys(["not_a_real_key"], {"a", "b"}, "--override")


# ---------------------------------------------------------------------------
# Ablation-suite unknown keys (spec-ablations §1.1; defect §8.6)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.6: the craftax ablation suite merges unknown "
        "config keys silently (run_ablations.py:102-146); the canonical "
        "chain semantics (spec-ablations §1.1) require KeyError, as the "
        "minihack suite already implements"
    ),
)
def test_ablation_suite_rejects_unknown_config_keys(tmp_path):
    """An unknown key in an ablation config must raise KeyError
    (spec-ablations §1.1: 'unknown key = KeyError')."""
    from experiments.rl_finetuning.run_ablations import _load_ablation_config

    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("definitely_not_a_key_xyz: 1\nmax_iter: 1\n")
    with pytest.raises(KeyError):
        _load_ablation_config(str(bogus))


# ---------------------------------------------------------------------------
# Dead config key layer_ablation_top_n (defect §8.7)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "traceability §8.7: layer_ablation_top_n is a dead key - the "
        "registry's top-n comes from the factory closure, so the config "
        "value has no effect (step 9 either wires the key or removes it; "
        "delete this test if the key is removed)"
    ),
)
def test_layer_ablation_top_n_key_controls_the_trainable_depth():
    """Setting layer_ablation_top_n must change which transformer
    blocks a layer-ablation optimizer trains (the key is documented in
    the suite config; spec-ablations §2 layer_ablation row).

    Method: apply layer_ablation_top1's optimizer to all-ones gradients
    under LAYER_ABLATION_TOP_N = 1 vs 3 and compare the sets of
    parameters that receive a non-zero update.
    """
    import jax.numpy as jnp

    from experiments.rl_finetuning.ablations.registry import REGISTRY
    from src.planners.model import build_model, init_params

    config = {**TINY_ARCH, "N_LAYERS": 4, "WEIGHT_DECAY": 0.0}
    model = build_model({**config, "PLAN_HORIZON": PLAN_HORIZON}, NUM_ACTIONS)
    params = init_params(model, jax.random.PRNGKey(SEED), OBS_DIM, PLAN_HORIZON)
    grads = jax.tree.map(jnp.ones_like, params)

    def trainable_paths(top_n_value: int) -> frozenset[str]:
        tx = REGISTRY["layer_ablation_top1"].optimizer_factory(
            {**config, "LAYER_ABLATION_TOP_N": top_n_value}, params
        )
        state = tx.init(params)
        updates, _ = tx.update(grads, state, params)
        flat = jax.tree_util.tree_flatten_with_path(updates)[0]
        return frozenset(
            "/".join(str(k.key) for k in path)
            for path, leaf in flat
            if bool(jnp.any(leaf != 0))
        )

    assert trainable_paths(1) != trainable_paths(3), (
        "layer_ablation_top_n had no effect on the trainable set"
    )


# ---------------------------------------------------------------------------
# Checkpoint/config coupling (spec-config §6.1)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "step-8 finding: load_checkpoint silently restores a checkpoint "
        "whose architecture disagrees with the config (PyTree restore "
        "without an abstract target adopts the saved shapes); traceability "
        "§5 res. 19 recorded 'Orbax raises on mismatch', which does not "
        "hold on this path - the failure surfaces only later at apply time"
    ),
)
def test_checkpoint_restore_with_mismatched_config_fails_loudly(tmp_path):
    """Restoring a checkpoint under a config with a different
    architecture must raise, not silently load (spec-config §6.1 'the
    model is built from the config: match the config to the checkpoint';
    traceability §5 res. 19: Orbax raises on mismatch, no pre-check).
    """
    from src.planners.model import build_model, init_params, load_checkpoint

    model_a = build_model({**TINY_ARCH, "PLAN_HORIZON": PLAN_HORIZON}, NUM_ACTIONS)
    params_a = init_params(model_a, jax.random.PRNGKey(SEED), OBS_DIM, PLAN_HORIZON)
    ckpt = tmp_path / "ckpt"
    with ocp.CheckpointManager(str(ckpt)) as mgr:
        mgr.save(0, args=ocp.args.PyTreeSave({"params": params_a}))
        mgr.wait_until_finished()

    arch_b = {**TINY_ARCH, "D_MODEL": TINY_ARCH["D_MODEL"] * 2, "PLAN_HORIZON": PLAN_HORIZON}
    model_b = build_model(arch_b, NUM_ACTIONS)
    with pytest.raises(Exception):  # noqa: B017 - the canonical contract is
        # only "fails rather than silently loads"; the concrete type is
        # Orbax-internal and unpinned (traceability §5 res. 19)
        load_checkpoint(
            model_b, jax.random.PRNGKey(SEED + 1), OBS_DIM, PLAN_HORIZON, str(ckpt)
        )


# (The smoke preset's PRIMARY-key null pins are already covered by
# test_smoke_src.py::test_smoke_budget_resolves_to_a_short_run.)
