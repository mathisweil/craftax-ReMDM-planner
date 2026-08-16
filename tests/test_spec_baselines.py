"""Supervision-source and released-artefact spec tests (step 8).

Sources: research/spec-training.md §3.1 (PPO expert), research/
spec-config.md §6.2/§6.4 (checkpoint metadata and published artefacts),
step-7 findings N7 (no expert/env pre-check) and N3 (offline
step-counter unit) as classified in
verification/2026-08-15-executable-baseline.md §9.

The minihack twin file covers its in-repo SB3/DT baselines
(spec-training §6.1); the PPO expert is craftax-only (PARITY
"Supervision").
"""

from __future__ import annotations

import inspect
import json

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from tests.conftest import ROOT, load_config

from src.planners.common import resolve_num_updates

_HF_OFFLINE = (
    ROOT
    / "checkpoints/hf/checkpoints/offline/Craftax-Classic-Symbolic-v1-Offline-Diffusion-BC-100M"
)


# ---------------------------------------------------------------------------
# N7: expert/env compatibility pre-check
# ---------------------------------------------------------------------------


def test_ppo_expert_load_rejects_a_mismatched_obs_dimension(tmp_path):
    """Loading expert parameters whose observation dimensionality
    disagrees with the target environment must raise a clear ValueError
    naming the dimensions, before any JIT tracing (spec-config §6.1:
    match the config to the checkpoint; was step-7 finding N7).

    Method: save a tiny 'ppo' (MLP ActorCritic) checkpoint initialised
    at obs dim 8, then load it declaring obs dim 16.
    """
    from src.planners.ppo import build_ppo_network, load_ppo_params

    net = build_ppo_network("ppo", num_actions=5, layer_size=16, config={})
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((1, 8)))
    ckpt = tmp_path / "expert"
    with ocp.CheckpointManager(str(ckpt)) as mgr:
        mgr.save(0, args=ocp.args.PyTreeSave({"params": params}))
        mgr.wait_until_finished()

    with pytest.raises(ValueError, match="obs"):
        load_ppo_params(str(ckpt), net, "ppo", num_envs=1, obs_shape=(16,),
                        layer_size=16)


# ---------------------------------------------------------------------------
# N3: offline checkpoint step-counter unit (released artefact)
# ---------------------------------------------------------------------------

_needs_artefact = pytest.mark.skipif(
    not (_HF_OFFLINE / "resume_metadata.json").exists(),
    reason="released HF checkpoints not downloaded to checkpoints/hf/",
)


@_needs_artefact
def test_released_offline_metadata_is_recipe_consistent():
    """The released offline BC checkpoint's own metadata is internally
    consistent with the Classic recipe at 512 envs (spec-config §4):
    1e8 frames // (512*128) = 1525 updates; gradient steps =
    1525 * update_epochs(8) * num_minibatches(8) = 97,600; the
    re-snapped budget is 1525 * 65,536 = 99,942,400 frames.
    """
    meta = json.loads((_HF_OFFLINE / "resume_metadata.json").read_text())
    assert meta["mode"] == "offline"
    assert meta["update_step"] == 1525
    assert meta["total_gradient_steps_completed"] == 1525 * 64
    snap = meta["config_snapshot"]
    assert int(snap["OFFLINE_TOTAL_TIMESTEPS"]) == 1525 * 512 * 128
    assert snap["NUM_ENVS"] == 512 and snap["NUM_STEPS"] == 128


@_needs_artefact
def test_released_offline_step_dir_is_the_documented_historical_exception():
    """The released Classic offline checkpoint's step directory reads
    1000000000, from a convention that predates the frame-denominated
    one (step-7 finding N3).

    Author decision 2026-08-16: the artefact stays as published and is
    historical/noncanonical; new artefacts use the canonical unit,
    which for offline is the resolved env-frame budget
    (offline.py: ``mgr.save(int(config["OFFLINE_TOTAL_TIMESTEPS"]))``,
    99,942,400 for the Classic recipe at 512 envs). This test fails if
    the artefact is silently renamed, or if the README's historical
    note stops explaining the two numbers.
    """
    step_dirs = sorted(
        int(p.name) for p in _HF_OFFLINE.iterdir() if p.name.isdigit()
    )
    assert step_dirs == [1_000_000_000]

    note = (ROOT / "README.md").read_text()
    assert "Historical note" in note
    assert "1000000000" in note and "99,942,400" in note


def test_new_offline_checkpoints_use_the_frame_denominated_step():
    """The canonical unit for a new offline checkpoint is the resolved
    env-frame budget, not an update count or a PPO-style timestep.

    Pinned at the call site so a change of unit has to change this
    test: the offline runner saves at ``int(OFFLINE_TOTAL_TIMESTEPS)``
    after the resolver has re-snapped it to ``NUM_UPDATES * fpu``.
    """
    from src.planners import offline

    source = inspect.getsource(offline.run_offline_diffusion)
    assert 'mgr.save(\n                int(config["OFFLINE_TOTAL_TIMESTEPS"]),' in source

    config = {
        **load_config("configs/defaults.yaml"),
        **load_config("configs/final_classic_ucl.yaml"),
    }
    resolve_num_updates(config, "offline")
    assert int(config["OFFLINE_TOTAL_TIMESTEPS"]) == 99_942_400
