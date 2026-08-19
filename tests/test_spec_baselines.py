"""Supervision-source and released-artefact spec tests (step 8).

Sources: research/spec-training.md §3.1 (PPO expert), research/
spec-config.md §6.1/§6.2/§6.4 (checkpoint metadata and published
artefacts), which record both the missing expert/env pre-check and the
offline step-counter unit that the released checkpoint was renamed to.

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

from src.planners.common import resolve_num_updates
from tests.conftest import ROOT, load_config

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
def test_released_offline_step_dir_uses_the_frame_denominated_unit():
    """The released Classic offline checkpoint's step directory is the
    resolved env-frame budget, the same unit a new one would save at
    (offline.py: ``mgr.save(int(config["OFFLINE_TOTAL_TIMESTEPS"]))``).

    It read 1000000000, from a convention that predates the
    frame-denominated one (step-7 finding N3), and was renamed on the
    Hub to 99,942,400 = 1525 updates x 512 envs x 128 steps, which is
    what this run's own resume_metadata.json records. The contents are
    unchanged and the restored parameters are identical; the step
    number appears in no file inside the checkpoint.
    """
    step_dirs = sorted(
        int(p.name) for p in _HF_OFFLINE.iterdir() if p.name.isdigit()
    )
    assert step_dirs == [99_942_400]

    meta = json.loads((_HF_OFFLINE / "resume_metadata.json").read_text())
    assert int(meta["config_snapshot"]["OFFLINE_TOTAL_TIMESTEPS"]) == step_dirs[0]


def test_new_online_checkpoints_use_the_frame_denominated_step():
    """Both online checkpoints are named in env frames, the same unit the
    offline runner and the released artefacts use.

    craftax had **three** step-directory conventions across three call
    sites: `offline.py` saved at the resolved frame budget, `online.py`
    saved the final policy at `NUM_UPDATES`, and it saved the
    best-by-validation policy at a fixed `0` sentinel. Only the first
    matched the released artefact `.../DAgger-100M/100000000/`, and an
    update count is not invariant under `num_envs` -- the same run on 96
    and 512 envs produced two different directory names for the same
    experience.

    Pinned at both call sites so a change of unit has to change this test.
    """
    from src.planners import online

    source = inspect.getsource(online.run_online)
    assert 'mgr.save(\n                int(config["ONLINE_TOTAL_TIMESTEPS"]),' in source
    assert "mgr.save(best_frames, args=ocp.args.StandardSave(best_state))" in source
    assert "mgr.save(0, args=" not in source

    config = {
        **load_config("configs/defaults.yaml"),
        **load_config("configs/final_classic_ucl.yaml"),
    }
    resolve_num_updates(config, "online")
    fpu = int(config["NUM_ENVS"]) * int(config["NUM_STEPS"])
    # The resolver re-snaps the budget, so the name is the exact frame count.
    assert int(config["ONLINE_TOTAL_TIMESTEPS"]) == int(config["NUM_UPDATES"]) * fpu


def test_the_best_policy_step_is_the_frame_count_it_was_captured_at():
    """`policies_best` is named for the frames trained when the best
    validation score was seen, not for the end of the run.

    `best_step_idx` is the 0-based update index the best parameters were
    captured at, so the run had completed `best_step_idx + 1` updates and
    `(best_step_idx + 1) * fpu` frames. A run whose validation never
    improved leaves the index at its -1 initial value and so saves at frame
    0 -- the honest number there, and the only case that still looks like
    the old sentinel.

    The index is carried through the training scan purely to name the
    checkpoint: it enters no loss, no parameter update and no RNG draw.
    Verified rather than asserted -- on CPU, where XLA is deterministic, a
    smoke run before and after the carry was added restores **bit-identical**
    parameters for both `policies` and `policies_best` (max |difference|
    exactly 0.0, same value hash), with only the directory names changing,
    6 -> 384 and 0 -> 64.
    """
    from src.planners.online import DAggerCarry, run_online

    # The field exists and is carried, not recomputed at save time.
    assert "best_step_idx" in DAggerCarry._fields

    source = inspect.getsource(run_online)
    assert "best_frames = (best_step_idx + 1) * frames_per_update" in source
    assert (
        'frames_per_update = int(config["NUM_ENVS"]) * int(config["NUM_STEPS"])'
        in source
    )

    # -1 is the "no validation improved" initial value, which names frame 0.
    train_source = inspect.getsource(
        __import__("src.planners.online", fromlist=["x"]).make_train_online_dagger
    )
    assert "best_step_idx=jnp.int32(-1)" in train_source


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
