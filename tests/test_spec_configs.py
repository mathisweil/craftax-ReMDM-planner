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
from tests.conftest import (
    NUM_ACTIONS,
    OBS_DIM,
    PLAN_HORIZON,
    SEED,
    TINY_ARCH,
    load_config,
)

from src.config import cast_override, validate_keys

# ---------------------------------------------------------------------------
# --override typing (spec-config §1.2: values cast to the key's type,
# "a typo is an error, not a silent no-op")
# ---------------------------------------------------------------------------


def test_override_values_cast_to_the_key_type():
    """Casting table per spec-config §1.2 (README.md:213).

    Both repos implement the same table with the same error types
    (PARITY 'Config-loader semantics', aligned at step 10).
    """
    assert cast_override("k", "0.3", 0.5) == 0.3
    assert cast_override("k", "1e-4", 0.5) == pytest.approx(1e-4)
    assert cast_override("k", "7", 3) == 7
    assert cast_override("k", "7.0", 3) == 7
    assert cast_override("k", "false", True) is False
    assert cast_override("k", "text", "old") == "text"
    assert cast_override("k", "null", 3) is None
    assert cast_override("k", "3", None) == 3

    with pytest.raises(TypeError):
        cast_override("k", "not_a_number", 3)
    with pytest.raises(TypeError):
        cast_override("k", "2.5", 3)  # non-integral float for an int key
    with pytest.raises(TypeError):
        cast_override("k", "maybe", True)


def test_unknown_override_key_is_an_error():
    """Unknown --override keys raise instead of silently no-oping
    (spec-config §1.2)."""
    with pytest.raises(KeyError, match="Unknown config key"):
        validate_keys(["not_a_real_key"], {"a", "b"}, "--override")


# ---------------------------------------------------------------------------
# Ablation-suite unknown keys (spec-ablations §1.1; defect §8.6)
# ---------------------------------------------------------------------------


def test_ablation_suite_rejects_unknown_config_keys(tmp_path):
    """An unknown key in an ablation config must raise KeyError
    (spec-ablations §1.1: 'unknown key = KeyError'; was defect §8.6)."""
    from experiments.rl_finetuning.run_ablations import _load_ablation_config

    bogus = tmp_path / "bogus.yaml"
    bogus.write_text("definitely_not_a_key_xyz: 1\nmax_iter: 1\n")
    with pytest.raises(KeyError):
        _load_ablation_config(str(bogus))


# ---------------------------------------------------------------------------
# Checkpoint/config coupling (spec-config §6.1)
# ---------------------------------------------------------------------------


def test_checkpoint_restore_with_mismatched_config_fails_loudly(tmp_path):
    """Restoring a checkpoint under a config with a different
    architecture must raise, not silently load (spec-config §6.1 'the
    model is built from the config: match the config to the checkpoint';
    was step-8 finding S8-1: the mismatch restored silently and failed
    only at apply time).
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
    with pytest.raises(ValueError, match="does not match the model"):
        load_checkpoint(
            model_b, jax.random.PRNGKey(SEED + 1), OBS_DIM, PLAN_HORIZON, str(ckpt)
        )


# (The smoke preset's frame-denominated sizing keys are already covered by
# test_smoke_src.py::test_smoke_budget_resolves_to_a_short_run.)


# ---------------------------------------------------------------------------
# Checkpoint persistence without W&B (spec-config §6.2 amendment 2)
# ---------------------------------------------------------------------------


def test_checkpoints_are_written_when_wandb_is_off():
    """save_policy alone decides whether a run keeps its weights: with
    W&B off the checkpoints go under checkpoint_dir instead of being
    discarded (PARITY 'Checkpoint selection and persistence'; the
    minihack twin saves unconditionally)."""
    from src.planners.common import checkpoint_root

    root = checkpoint_root(
        {"CHECKPOINT_DIR": "ckpts"}, "online", "Env-Online-Diffusion-DAgger-100M"
    )
    assert root == "ckpts/online/Env-Online-Diffusion-DAgger-100M"
    # Default when the key is absent.
    assert checkpoint_root({}, "offline", "run").startswith("checkpoints/offline/")


# ---------------------------------------------------------------------------
# W&B naming (spec-config §6.5: the config keys govern; the "remdm-*"
# literals were dead fallbacks)
# ---------------------------------------------------------------------------


def test_wandb_names_come_from_the_config_not_a_literal():
    """Training and the ablation suite both take project and entity from
    the config, and the shipped names are the canonical ones. The
    minihack twin pins the same rule in its test_config.py."""
    import yaml

    from tests.conftest import ROOT, load_config

    config = load_config("configs/defaults.yaml")
    assert config["WANDB_PROJECT"] == "craftax-ReMDM-planner"

    abl = yaml.safe_load(
        (ROOT / "experiments/rl_finetuning/configs/ablations_default.yaml").read_text()
    )
    assert abl["wandb_project"] == "craftax-ReMDM-planner-ablations"
    assert abl["wandb_entity"] == config["WANDB_ENTITY"]

    for rel in ("src/planners/logging.py", "experiments/rl_finetuning/run_ablations.py"):
        assert "remdm-craftax" not in (ROOT / rel).read_text(), rel


def test_wandb_init_takes_project_and_entity_from_the_config():
    """init_wandb passes the config's project and entity through to
    wandb.init rather than defaulting them."""
    import inspect

    from src.planners.logging import init_wandb

    src = inspect.getsource(init_wandb)
    assert '"project": config["WANDB_PROJECT"]' in src
    assert '"entity": config.get("WANDB_ENTITY")' in src


def test_suite_wandb_init_takes_project_and_entity_from_the_config():
    """The ablation suite's wandb.init does the same, from the merged
    suite config."""
    import inspect

    from experiments.rl_finetuning.run_ablations import main

    src = inspect.getsource(main)
    assert 'project=merged.get("WANDB_PROJECT")' in src or 'merged["WANDB_PROJECT"]' in src
    assert 'entity=merged.get("WANDB_ENTITY")' in src


def test_inference_defaults_to_the_sampler_the_paper_evaluates_with() -> None:
    """--mode inference must use the harness sampler, not inpainting.

    Every published number comes from build_eval_fn, which calls
    sample_plan with no locked prefix and replans every EVAL_REPLAN steps.
    run_inference previously used sample_plan_inpainting, which locks each
    executed action as a fixed prefix and scores far lower on the same
    weights, so the README sent readers to a different agent than the one
    the paper reports. Inpainting stays reachable as an explicit ablation.
    """
    import inspect

    from src.planners import inference

    config = load_config("configs/defaults.yaml")
    assert config["INFERENCE_SAMPLER"] == "sample_plan"
    assert config["EVAL_REPLAN"] == 8

    src = inspect.getsource(inference.run_inference)
    assert "sample_plan(" in src, "the default path must call sample_plan"
    assert "sample_plan_inpainting(" in src, (
        "the inpainting ablation must stay reachable"
    )
