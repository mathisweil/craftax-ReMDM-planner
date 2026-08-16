"""Smoke tests for the src/ diffusion planner: build, forward, train, save, sample.

Proves the pipeline runs end to end. Asserts nothing about result quality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from tests.conftest import (
    BATCH,
    NUM_ACTIONS,
    OBS_DIM,
    PLAN_HORIZON,
    ROOT,
    SEED,
    SRC_MODULES,
    import_or_skip,
    load_config,
)


def _finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(x)))


# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", SRC_MODULES)
def test_src_module_imports(module_name: str) -> None:
    assert import_or_skip(module_name) is not None


def test_module_discovery_found_the_package() -> None:
    assert "src.models.denoiser" in SRC_MODULES
    assert "src.diffusion.sampling" in SRC_MODULES


# ---------------------------------------------------------------------------
# 2. Instantiation from the real config
# ---------------------------------------------------------------------------


def test_model_builds_from_real_config(real_config, craftax_env) -> None:
    """Full-size model from the shipped configs/defaults.yaml at real env dims."""
    from src.planners.model import build_model, init_params

    model = build_model(real_config, craftax_env["num_actions"])
    assert model.d_model == real_config["D_MODEL"]
    assert model.n_layers == real_config["N_LAYERS"]
    assert model.plan_horizon == real_config["PLAN_HORIZON"]

    real_params = init_params(
        model, jax.random.PRNGKey(SEED),
        craftax_env["obs_dim"], real_config["PLAN_HORIZON"],
    )
    leaves = jax.tree_util.tree_leaves(real_params)
    assert leaves, "real config produced an empty parameter tree"
    assert all(_finite(leaf) for leaf in leaves)


def test_tiny_model_params_are_finite(params) -> None:
    leaves = jax.tree_util.tree_leaves(params)
    assert leaves
    assert all(_finite(leaf) for leaf in leaves)


# ---------------------------------------------------------------------------
# 3. Forward pass
# ---------------------------------------------------------------------------


def test_forward_pass_shape_dtype_and_no_nans(apply_fns, params, batch) -> None:
    apply_eval, _ = apply_fns
    logits = apply_eval(params, batch["obs"], batch["acts"], batch["timestep"])

    assert logits.shape == (BATCH, PLAN_HORIZON, NUM_ACTIONS)
    assert logits.dtype == jnp.float32
    assert _finite(logits)


def test_forward_pass_accepts_mask_tokens(apply_fns, params, batch) -> None:
    """MASK id == num_actions must be a valid input token."""
    apply_eval, _ = apply_fns
    masked = jnp.full_like(batch["acts"], NUM_ACTIONS)
    logits = apply_eval(params, batch["obs"], masked, batch["timestep"])

    assert logits.shape == (BATCH, PLAN_HORIZON, NUM_ACTIONS)
    assert _finite(logits)


def test_forward_pass_is_deterministic(apply_fns, params, batch) -> None:
    apply_eval, _ = apply_fns
    first = apply_eval(params, batch["obs"], batch["acts"], batch["timestep"])
    second = apply_eval(params, batch["obs"], batch["acts"], batch["timestep"])
    assert jnp.array_equal(first, second)


def test_train_apply_runs_with_dropout(apply_fns, params, batch) -> None:
    _, apply_train = apply_fns
    logits = apply_train(
        params, batch["obs"], batch["acts"], batch["timestep"],
        jax.random.PRNGKey(SEED),
    )
    assert logits.shape == (BATCH, PLAN_HORIZON, NUM_ACTIONS)
    assert _finite(logits)


# ---------------------------------------------------------------------------
# 4. Loss and one training step
# ---------------------------------------------------------------------------


def test_compute_loss_is_finite(apply_fns, params, batch, schedules) -> None:
    from src.diffusion.loss import compute_loss

    _, apply_train = apply_fns
    schedule_fn, schedule_deriv_fn = schedules
    loss, info = compute_loss(
        apply_train, params, jax.random.PRNGKey(SEED),
        batch["acts"], batch["obs"], batch["valid"],
        NUM_ACTIONS, schedule_fn, schedule_deriv_fn,
    )

    assert loss.shape == ()
    assert _finite(loss)
    assert all(_finite(v) for v in info.values())


def test_adamw_at_zero_decay_matches_adam(model, apply_fns, params, batch, schedules) -> None:
    """Core training moved from optax.adam to optax.adamw with an
    explicit weight_decay=0.0 (author decision 2026-08-16). AdamW's
    decay is decoupled and additive, so at 0.0 the two updates are the
    same to the last bit - this is the equivalence guard for that
    change.

    Only the optimiser-state *structure* differs (adamw's chain carries
    an extra EmptyState), which is why a checkpoint saved by the old
    chain cannot be resumed into the new one.
    """
    import optax
    from flax.training.train_state import TrainState

    from src.planners.common import make_grad_step
    from src.planners.model import create_train_state

    _, apply_train = apply_fns
    schedule_fn, schedule_deriv_fn = schedules
    step_fn = make_grad_step(
        apply_train, NUM_ACTIONS, schedule_fn, schedule_deriv_fn, 0.0, 0.0
    )

    adamw_state = create_train_state(model, params, 1e-3, 1.0, weight_decay=0.0)
    adam_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3, eps=1e-5)),
    )

    args = (batch["acts"], batch["obs"], batch["valid"],
            jax.random.PRNGKey(SEED), batch["advantages"])
    for _ in range(3):
        adamw_state, _ = step_fn(adamw_state, *args)
        adam_state, _ = step_fn(adam_state, *args)

    for a, b in zip(
        jax.tree_util.tree_leaves(adamw_state.params),
        jax.tree_util.tree_leaves(adam_state.params),
    ):
        assert jnp.array_equal(a, b)


def test_one_grad_step_runs_and_updates_params(
    model, apply_fns, params, batch, schedules, tiny_config,
) -> None:
    from src.planners.common import make_grad_step
    from src.planners.model import create_train_state

    _, apply_train = apply_fns
    schedule_fn, schedule_deriv_fn = schedules
    state = create_train_state(model, params, tiny_config["LR"], 1.0)

    step = make_grad_step(
        apply_train, NUM_ACTIONS, schedule_fn, schedule_deriv_fn,
        sigma_t=tiny_config["TRAIN_SIGMA"],
        label_smoothing=tiny_config["LABEL_SMOOTHING"],
    )
    new_state, metrics = step(
        state, batch["acts"], batch["obs"], batch["valid"],
        jax.random.PRNGKey(SEED), batch["advantages"],
    )

    assert _finite(metrics["loss"])
    assert _finite(metrics["grad_norm"])
    assert int(new_state.step) == int(state.step) + 1
    assert all(_finite(leaf) for leaf in jax.tree_util.tree_leaves(new_state.params))

    changed = any(
        not jnp.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(state.params),
            jax.tree_util.tree_leaves(new_state.params),
        )
    )
    assert changed, "one gradient step left every parameter untouched"


# ---------------------------------------------------------------------------
# 5. Save / reload
# ---------------------------------------------------------------------------


def test_checkpoint_roundtrip_preserves_output(
    model, apply_fns, params, batch, tmp_path,
) -> None:
    from src.planners.model import load_checkpoint

    apply_eval, _ = apply_fns
    before = apply_eval(params, batch["obs"], batch["acts"], batch["timestep"])

    ckpt_dir = tmp_path / "ckpt"
    with ocp.CheckpointManager(str(ckpt_dir)) as mgr:
        mgr.save(0, args=ocp.args.PyTreeSave({"params": params}))
        mgr.wait_until_finished()

    restored = load_checkpoint(
        model, jax.random.PRNGKey(SEED + 1), OBS_DIM, PLAN_HORIZON, str(ckpt_dir),
    )
    after = apply_eval(restored, batch["obs"], batch["acts"], batch["timestep"])

    assert jnp.array_equal(before, after), "reloaded model produced different output"


def test_checkpoint_metadata_roundtrip(tmp_path, tiny_config) -> None:
    from src.planners.model import load_checkpoint_metadata, save_checkpoint_metadata

    assert load_checkpoint_metadata(str(tmp_path)) is None

    save_checkpoint_metadata(
        str(tmp_path), mode="offline",
        update_step=np.int64(7), total_gradient_steps=np.int64(70),
        wandb_run_id=None, config=tiny_config,
    )
    meta = load_checkpoint_metadata(str(tmp_path))

    assert meta["mode"] == "offline"
    assert meta["update_step"] == 7


def test_missing_checkpoint_raises(model, tmp_path) -> None:
    from src.planners.model import load_checkpoint

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            model, jax.random.PRNGKey(SEED), OBS_DIM, PLAN_HORIZON,
            str(tmp_path / "empty"),
        )


# ---------------------------------------------------------------------------
# 6. Diffusion internals and sampling
# ---------------------------------------------------------------------------


def test_schedule_endpoints_and_derivatives() -> None:
    from src.diffusion.schedules import SCHEDULE_MAP

    for name, (alpha, alpha_dot) in SCHEDULE_MAP.items():
        t = jnp.array([0.0, 0.5, 1.0])
        a, ad = alpha(t), alpha_dot(t)
        assert _finite(a) and _finite(ad), name
        assert np.isclose(float(a[0]), 1.0, atol=1e-6), f"{name}: alpha(0) != 1"
        assert np.isclose(float(a[-1]), 0.0, atol=1e-6), f"{name}: alpha(1) != 0"
        assert bool(jnp.all(ad <= 0)), f"{name}: alpha is not non-increasing"


def test_forward_process_masks_and_preserves(batch) -> None:
    from src.diffusion.forward import forward_process

    keep_all = forward_process(
        jax.random.PRNGKey(SEED), batch["acts"], jnp.ones((BATCH,)), NUM_ACTIONS,
    )
    mask_all = forward_process(
        jax.random.PRNGKey(SEED), batch["acts"], jnp.zeros((BATCH,)), NUM_ACTIONS,
    )

    assert jnp.array_equal(keep_all, batch["acts"])
    assert bool(jnp.all(mask_all == NUM_ACTIONS))
    assert mask_all.dtype == batch["acts"].dtype


# Each combination compiles its own scan, so cover the three remasking
# strategies through the three-phase loop; the non-loop path is exercised by
# the environment rollouts below, which run with USE_LOOP disabled.
@pytest.mark.parametrize("remask_strategy", ["rescale", "cap", "conf"])
@pytest.mark.parametrize("use_loop", [True])
def test_sample_plan_runs(
    apply_fns, params, batch, schedules, remask_strategy, use_loop,
) -> None:
    from src.diffusion.sampling import sample_plan

    apply_eval, _ = apply_fns
    schedule_fn, _ = schedules
    plan = sample_plan(
        apply_eval, params, jax.random.PRNGKey(SEED), batch["obs"],
        NUM_ACTIONS, PLAN_HORIZON, num_steps=3, schedule_fn=schedule_fn,
        remask_strategy=remask_strategy, eta=0.5, use_loop=use_loop,
        t_on=0.7, t_off=0.3, temperature=0.5, top_p=0.95,
    )

    assert plan.shape == (BATCH, PLAN_HORIZON)
    assert jnp.issubdtype(plan.dtype, jnp.integer)
    assert bool(jnp.all(plan >= 0)) and bool(jnp.all(plan < NUM_ACTIONS)), (
        "sampled plan contains a MASK or out-of-vocabulary action"
    )


def test_sample_plan_rejects_unknown_strategy(apply_fns, params, batch, schedules) -> None:
    from src.diffusion.sampling import sample_plan

    apply_eval, _ = apply_fns
    schedule_fn, _ = schedules
    with pytest.raises(ValueError):
        sample_plan(
            apply_eval, params, jax.random.PRNGKey(SEED), batch["obs"],
            NUM_ACTIONS, PLAN_HORIZON, num_steps=2, schedule_fn=schedule_fn,
            remask_strategy="not-a-strategy",
        )


def test_sample_plan_inpainting_locks_history(
    apply_fns, params, batch, schedules,
) -> None:
    from src.diffusion.sampling import sample_plan_inpainting

    apply_eval, _ = apply_fns
    schedule_fn, _ = schedules
    hist_len = jnp.full((BATCH,), 2, dtype=jnp.int32)
    history = jnp.zeros((BATCH, PLAN_HORIZON), dtype=jnp.int32)

    plan = sample_plan_inpainting(
        apply_eval, params, jax.random.PRNGKey(SEED), batch["obs"],
        history, hist_len, NUM_ACTIONS, PLAN_HORIZON,
        diffusion_steps=3, schedule_fn=schedule_fn,
        remask_strategy="rescale", eta=0.5,
        use_loop=False, t_on=0.7, t_off=0.3,
        temperature=0.5, top_p=0.95,
    )

    assert plan.shape == (BATCH, PLAN_HORIZON)
    assert jnp.issubdtype(plan.dtype, jnp.integer)
    assert jnp.array_equal(plan[:, :2], history[:, :2]), "historical prefix was overwritten"


# ---------------------------------------------------------------------------
# 7. Config resolution used by the runners
# ---------------------------------------------------------------------------


def test_resolve_num_updates_from_frame_budget(real_config) -> None:
    from src.planners.common import resolve_num_updates

    config = {**real_config, "NUM_ENVS": 8, "NUM_STEPS": 4, "OFFLINE_TOTAL_TIMESTEPS": 320}
    resolve_num_updates(config, "offline")

    assert config["NUM_UPDATES"] == 10
    assert config["OFFLINE_TOTAL_TIMESTEPS"] == 320

    resolve_num_updates(config, "offline")  # idempotent
    assert config["NUM_UPDATES"] == 10


def test_resolve_scaled_hyperparams(real_config) -> None:
    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {
        **real_config, "NUM_ENVS": 8, "NUM_STEPS": 4,
        "UPDATE_EPOCHS": 2, "NUM_MINIBATCHES": 3,
        "ONLINE_TOTAL_TIMESTEPS": 320, "LR_WARMUP_FRAMES": 64,
        "VAL_INTERVAL_FRAMES": 320, "DAGGER_BETA_FINAL": 0.1,
        "DAGGER_BUFFER_CYCLES": 2,
    }
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    # 64 frames // 32 fpu = 2 updates x (2 epochs * 3 minibatches) = 12
    # gradient steps (frame-denominated warmup, author decision 2026-08-15)
    assert config["LR_WARMUP_STEPS"] == 12
    assert config["VAL_INTERVAL"] == 10
    assert config["DAGGER_BUFFER_MAX"] == 64
    assert 0.0 < config["DAGGER_BETA_DECAY"] < 1.0


def test_resolve_num_updates_rejects_unknown_mode(real_config) -> None:
    from src.planners.common import resolve_num_updates

    with pytest.raises(ValueError):
        resolve_num_updates({**real_config}, "nonsense")


def test_dagger_sizing_defaults_to_one_train_pass(real_config) -> None:
    """The runner's default is 1 pass, keeping DAgger's per-update gradient
    work equal to offline BC's."""
    from src.planners.common import dagger_sizing

    config = {**real_config, "NUM_ENVS": 8, "NUM_STEPS": 8, "PLAN_HORIZON": 4,
              "DAGGER_BUFFER_MAX": 1_000_000, "DAGGER_TRAIN_PASSES": None}
    sizing = dagger_sizing(config, num_updates=10)

    assert sizing["n_train_passes"] == 1
    assert sizing["valid_per_rollout"] == 5
    assert sizing["samples_per_update"] == 40
    assert sizing["n_cycles"] == 2
    # Capped by the run length, not by DAGGER_BUFFER_MAX.
    assert sizing["max_buffer_size"] == 400

    assert dagger_sizing({**config, "DAGGER_TRAIN_PASSES": 4}, 10)["n_train_passes"] == 4


# Derived quantities the four shipped final configs' comments quote,
# HAND-DERIVED from the canonical 1e8-frame budget and 1,638,400-frame
# warmup (author decisions 2026-08-15, final; the earlier version of this
# table read the values back from the resolvers - flagged SELF-ORACLE by
# the step-8 audit). Arithmetic, per config (fpu = num_envs * 128;
# geometry = update_epochs 8 * num_minibatches 8 = 64):
#   NUM_UPDATES = 1e8 // fpu; LR_WARMUP_STEPS = (1_638_400 // fpu) * 64;
#   DAGGER_BUFFER_MAX = round(0.76294 or 1.90735 cycles * fpu).
#   classic_ucl  (fpu 65_536): 1525;  25*64 = 1600;  round(1.90735*65536) = 125_000
#   classic_qmul (fpu 12_288): 8138; 133*64 = 8512;  round(1.90735*12288) = 23_438
#   craftax_ucl  (fpu 57_344): 1743;  28*64 = 1792;  round(0.76294*57344) = 43_750
#   craftax_qmul (fpu  8_192): 12_207; 200*64 = 12_800; round(0.76294*8192) = 6_250
FINAL_CONFIG_DERIVATIONS = {
    "configs/final_classic_ucl.yaml": {
        "NUM_ENVS": 512,
        "NUM_UPDATES": 1525,
        "LR_WARMUP_STEPS": 1600,
        "DAGGER_BUFFER_MAX": 125_000,
    },
    "configs/final_craftax_ucl.yaml": {
        "NUM_ENVS": 448,
        "NUM_UPDATES": 1743,
        "LR_WARMUP_STEPS": 1792,
        "DAGGER_BUFFER_MAX": 43_750,
    },
    "configs/final_classic_qmul.yaml": {
        "NUM_ENVS": 96,
        "NUM_UPDATES": 8138,
        "LR_WARMUP_STEPS": 8512,
        "DAGGER_BUFFER_MAX": 23_438,
    },
    "configs/final_craftax_qmul.yaml": {
        "NUM_ENVS": 64,
        "NUM_UPDATES": 12_207,
        "LR_WARMUP_STEPS": 12_800,
        "DAGGER_BUFFER_MAX": 6_250,
    },
}


@pytest.mark.parametrize(
    "config_path", sorted(FINAL_CONFIG_DERIVATIONS)
)
def test_final_configs_resolve_to_their_documented_quantities(config_path: str) -> None:
    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {**load_config("configs/defaults.yaml"), **load_config(config_path)}
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    for key, expected in FINAL_CONFIG_DERIVATIONS[config_path].items():
        assert int(config[key]) == expected, (
            f"{config_path}: {key} resolves to {config[key]}, not {expected}. "
            "Update the config's comments in the same change."
        )


@pytest.mark.parametrize("config_path", sorted(FINAL_CONFIG_DERIVATIONS))
def test_lr_warmup_is_shorter_than_the_cosine_horizon(config_path: str) -> None:
    """The resolved warmup is a gradient-step count (author decision
    2026-08-15: frames convert through the effective geometry), so it
    must sit strictly below decay_steps = num_updates * update_epochs *
    num_minibatches, and the frames it spans must reproduce
    lr_warmup_frames to within one optimiser update of slack."""
    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {**load_config("configs/defaults.yaml"), **load_config(config_path)}
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    geometry = int(config["UPDATE_EPOCHS"]) * int(config["NUM_MINIBATCHES"])
    grad_steps = int(config["NUM_UPDATES"]) * geometry
    warmup = int(config["LR_WARMUP_STEPS"])
    assert warmup < grad_steps
    fpu = int(config["NUM_ENVS"]) * int(config["NUM_STEPS"])
    frames_covered = warmup * fpu / geometry
    assert abs(frames_covered - float(config["LR_WARMUP_FRAMES"])) < fpu


def test_snapshot_minibatch_matches_what_the_runners_use(real_config, capsys) -> None:
    """Regression: print_config_snapshot derived the minibatch from fpu.

    Both modes train on sliding windows, so a rollout of num_steps transitions
    yields num_steps - plan_horizon + 1 windows per environment. offline.py:68
    sets MINIBATCH_SIZE from that, and the DAgger training scan reshapes a
    dataset of the same size. Deriving from fpu overstated the printed
    minibatch by num_steps / valid_per_rollout.
    """
    from src.planners.common import (
        dagger_sizing,
        print_config_snapshot,
        resolve_num_updates,
        resolve_scaled_hyperparams,
    )

    config = {**real_config}
    resolve_num_updates(config, "offline")
    resolve_scaled_hyperparams(config, "offline")

    sizing = dagger_sizing(config, config["NUM_UPDATES"])
    expected = sizing["samples_per_update"] // config["NUM_MINIBATCHES"]
    fpu_derived = (
        config["NUM_STEPS"] * config["NUM_ENVS"] // config["NUM_MINIBATCHES"]
    )
    assert expected != fpu_derived, "pick a config where the two disagree"

    print_config_snapshot(config, "offline")
    out = capsys.readouterr().out

    assert f"minibatch={expected}" in out
    assert f"minibatch={fpu_derived}" not in out
    assert f"samples_per_update  = {sizing['samples_per_update']:,}" in out


@pytest.mark.parametrize("config_path", sorted(FINAL_CONFIG_DERIVATIONS))
def test_offline_and_dagger_stay_compute_matched(config_path: str, capsys) -> None:
    """The BC baseline's whole purpose: same updates, same gradient steps.

    common.py:22 resolves NUM_UPDATES for both modes and dagger_sizing documents
    DAGGER_TRAIN_PASSES=1 as the thing that keeps the per-update gradient work
    equal. Both final config pairs set the two frame budgets equal.
    """
    from src.planners.common import (
        print_config_snapshot,
        resolve_num_updates,
        resolve_scaled_hyperparams,
    )

    base = {**load_config("configs/defaults.yaml"), **load_config(config_path)}
    assert int(float(base["OFFLINE_TOTAL_TIMESTEPS"])) == int(
        float(base["ONLINE_TOTAL_TIMESTEPS"])
    ), "the two budgets must match or the baseline is not compute-matched"

    snapshots = {}
    for mode in ("offline", "online"):
        config = {**base}
        resolve_num_updates(config, mode)
        resolve_scaled_hyperparams(config, mode)
        print_config_snapshot(config, mode)
        out = capsys.readouterr().out
        grad_steps = (
            config["NUM_UPDATES"]
            * config["UPDATE_EPOCHS"]
            * config["NUM_MINIBATCHES"]
        )
        snapshots[mode] = (config["NUM_UPDATES"], grad_steps)
        assert "total_grad_steps" in out
        assert f"= {grad_steps:,}" in out

    assert snapshots["offline"] == snapshots["online"], (
        f"{config_path}: offline {snapshots['offline']} vs "
        f"online {snapshots['online']}"
    )


def test_compile_and_run_separates_compile_from_execute() -> None:
    """Regression: the runners timed ``out = train_fn(rngs)`` with no block.

    JAX dispatch is asynchronous, so that call returns once compilation is done
    and the work is enqueued. The reported SPS therefore divided total frames
    by a duration that excluded nearly all of the execution.
    """
    from src.planners.common import compile_and_run

    @jax.jit
    def train_fn(x):
        def body(c, _):
            return jnp.tanh(c @ c) * 1.0001, None

        out, _ = jax.lax.scan(body, x, None, 200)
        return out

    x = jnp.eye(64, dtype=jnp.float32) * 0.5

    out, timing = compile_and_run(train_fn, x, total_frames=1000)

    assert _finite(out)
    assert set(timing) == {
        "compile_s",
        "execute_s",
        "total_s",
        "sps_execute",
        "sps_total",
    }
    assert timing["compile_s"] > 0.0
    assert timing["execute_s"] > 0.0
    # The execute leg is blocked, so it is real time, not dispatch time.
    assert timing["total_s"] == pytest.approx(
        timing["compile_s"] + timing["execute_s"]
    )
    assert timing["sps_total"] < timing["sps_execute"]


def test_format_timing_reports_both_legs() -> None:
    from src.planners.common import format_timing

    text = format_timing(
        {
            "compile_s": 52.0,
            "execute_s": 3600.0,
            "total_s": 3652.0,
            "sps_execute": 27_000.0,
            "sps_total": 26_600.0,
        }
    )
    assert "Compile: 52.0s" in text
    assert "Execute: 3600.0s" in text
    assert "27000 (execute)" in text
    assert "26600 (including compile)" in text


def test_online_runner_no_longer_reports_one_fused_time() -> None:
    """The old shape printed a single ``Time: ...s  SPS: ...`` line."""
    source = (ROOT / "src" / "planners" / "online.py").read_text()
    assert "compile_and_run" in source
    assert 'f"Time: {elapsed' not in source


def test_snapshot_reports_the_gradient_steps_that_actually_run(real_config, capsys) -> None:
    """Regression: print_config_snapshot used to derive n_train_passes as
    ``buffer_max // samples_per_update`` while the runner used 1, overstating
    total_grad_steps by 2x on defaults.yaml and 23x on classic_exp_c."""
    from src.planners.common import (
        dagger_sizing,
        print_config_snapshot,
        resolve_num_updates,
        resolve_scaled_hyperparams,
    )

    config = {**real_config}
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")
    print_config_snapshot(config, "online")
    out = capsys.readouterr().out

    sizing = dagger_sizing(config, config["NUM_UPDATES"])
    expected = (
        config["NUM_UPDATES"] * sizing["n_train_passes"]
        * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
    )
    assert f"total_grad_steps         = {expected:,}" in out, out

    # The stale formula on defaults.yaml; assert we are not printing it.
    stale_passes = max(
        1, int(config["DAGGER_BUFFER_MAX"]) // sizing["samples_per_update"]
    )
    assert stale_passes > sizing["n_train_passes"], (
        "defaults.yaml no longer exercises the divergence; pick another config"
    )
    stale = expected * stale_passes
    assert f"{stale:,}" not in out


def test_validate_config_requires_checkpoints() -> None:
    import main as main_module

    with pytest.raises(ValueError):
        main_module.validate_config({"MODE": "offline"})
    with pytest.raises(ValueError):
        main_module.validate_config({"MODE": "inference"})

    main_module.validate_config({"MODE": "offline", "PPO_CHECKPOINT_PATH": "x"})
    main_module.validate_config({"MODE": "inference", "CHECKPOINT_PATH": "x"})


def test_compilation_cache_is_opt_in_and_creates_its_directory(tmp_path) -> None:
    import main as main_module

    assert main_module.configure_compilation_cache({}) is None
    assert main_module.configure_compilation_cache({"JAX_COMPILATION_CACHE_DIR": None}) is None

    target = tmp_path / "nested" / "jax-cache"
    try:
        resolved = main_module.configure_compilation_cache(
            {"JAX_COMPILATION_CACHE_DIR": str(target)}
        )
        assert resolved == str(target)
        assert target.is_dir()
        assert jax.config.jax_compilation_cache_dir == str(target)
    finally:
        # Session-wide config; leave it as the rest of the suite expects.
        jax.config.update("jax_compilation_cache_dir", None)


def test_defaults_config_declares_the_compilation_cache_key(real_config: dict) -> None:
    """main.py reads it, so defaults.yaml must declare it or --override rejects it."""
    assert "JAX_COMPILATION_CACHE_DIR" in real_config
    assert real_config["JAX_COMPILATION_CACHE_DIR"] is None, (
        "the shipped default must be off: the right path is machine-specific"
    )


def test_dispatch_table_covers_every_mode() -> None:
    import main as main_module

    modes = {"collect", "offline", "online", "inference", "smoke"}
    assert set(main_module.DISPATCH) == modes
    assert all(callable(fn) for fn in main_module.DISPATCH.values())

    parser = main_module._build_parser("configs/defaults.yaml")
    choices = next(a.choices for a in parser._actions if a.dest == "mode")
    assert set(choices) == modes, "--mode choices and DISPATCH disagree"


# ---------------------------------------------------------------------------
# 8. End to end against the real environment
# ---------------------------------------------------------------------------


def test_env_reset(craftax_env) -> None:
    """Env stepping is covered by the rollout below; a separate step() call here
    would cost another few seconds of tracing for no extra coverage."""
    env, env_params = craftax_env["env"], craftax_env["env_params"]

    obs, _ = env.reset(jax.random.PRNGKey(SEED), env_params)

    assert obs.shape == (craftax_env["num_envs"], craftax_env["obs_dim"])
    assert _finite(obs)


def test_plan_and_act_in_env(craftax_env, tiny_config, schedules) -> None:
    """Plan with the diffusion model, execute the plan in Craftax, log metrics."""
    from src.planners.common import make_validate
    from src.planners.model import (
        build_model,
        create_train_state,
        init_params,
        make_apply_fns,
    )

    num_actions, obs_dim = craftax_env["num_actions"], craftax_env["obs_dim"]
    config = {
        **tiny_config, "NUM_ACTIONS": num_actions,
        # Minimal loop: the sampler's own variants are covered above.
        "VAL_DIFFUSION_STEPS": 1, "USE_LOOP": False,
    }

    model = build_model(config, num_actions)
    real_dim_params = init_params(
        model, jax.random.PRNGKey(SEED), obs_dim, PLAN_HORIZON,
    )
    apply_eval, _ = make_apply_fns(model)
    state = create_train_state(model, real_dim_params, config["LR"], 1.0)

    validate = make_validate(
        craftax_env["env"], craftax_env["env_params"], apply_eval,
        num_actions, PLAN_HORIZON, schedules[0], config,
        val_replan_every=1, n_val_cycles=1,
    )
    metrics = validate(state, jax.random.PRNGKey(SEED))

    assert metrics, "validation rollout produced no metrics"
    assert all(k.startswith("val/") for k in metrics)
    assert all(_finite(v) for v in metrics.values())


# ---------------------------------------------------------------------------
# 9. Entry point scripts in minimal smoke mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    ["main --help", "count_params --help", "eval_ppo_expert --help",
     "hf_upload --help", "hf_upload_demo --help"],
)
def test_entry_point_help(entry_point_runs, label: str) -> None:
    """--help proves the script's full import chain and parser are intact."""
    result = entry_point_runs[label]
    assert result.returncode == 0, f"{label} failed:\n{result.stderr[-2000:]}"
    assert "usage" in result.stdout.lower()


def test_count_params_script_runs(entry_point_runs) -> None:
    """count_params.py is the only entry point that runs a real job without a checkpoint."""
    result = entry_point_runs["count_params run"]
    assert result.returncode == 0, result.stderr[-2000:]
    assert "params" in result.stdout


def test_main_rejects_missing_mode(entry_point_runs) -> None:
    result = entry_point_runs["main no-mode"]
    assert result.returncode != 0
    assert "--mode" in result.stderr


def test_smoke_mode_trains_end_to_end(entry_point_runs) -> None:
    """`main.py --mode smoke` is the full pipeline: rollout, DAgger, validation."""
    result = entry_point_runs["main --mode smoke"]
    assert result.returncode == 0, result.stderr[-3000:]
    assert "SMOKE TEST SUMMARY" in result.stdout
    assert "all metrics finite   = True" in result.stdout


def test_smoke_config_overlays_defaults() -> None:
    """configs/smoke.yaml is overrides-only; main.build_config layers it on top."""
    from tests.conftest import load_config

    defaults = load_config("configs/defaults.yaml")
    smoke = load_config("configs/smoke.yaml")

    import main as main_module

    assert smoke, "smoke.yaml is empty"

    # A key that is neither in defaults.yaml nor a CLI-backed config key
    # would be rejected by build_config's key validation.
    cli_keys = {k.upper() for k in main_module._CLI_CONFIG_KEYS}
    unknown = set(smoke) - set(defaults) - cli_keys
    assert not unknown, f"smoke.yaml sets keys nothing reads: {unknown}"

    merged = {**defaults, **smoke}
    # The invariants make_train_dagger asserts on derived values.
    num_envs, num_steps = merged["NUM_ENVS"], merged["NUM_STEPS"]
    plan_horizon, num_minibatches = merged["PLAN_HORIZON"], merged["NUM_MINIBATCHES"]

    assert num_steps >= plan_horizon
    assert num_steps % plan_horizon == 0
    samples_per_update = num_envs * (num_steps - plan_horizon + 1)
    assert samples_per_update % num_minibatches == 0
    # dagger_buffer_max is derived from the cycle-denominated key at load.
    buffer_max = round(merged["DAGGER_BUFFER_CYCLES"] * num_envs * num_steps)
    assert samples_per_update <= buffer_max


def test_smoke_budget_resolves_to_a_short_run() -> None:
    """Every smoke-sizing key must survive resolution: the frame-denominated
    keys are rescaled to the smoke rollout width, so a stale one silently
    restores a full-size run."""
    from tests.conftest import load_config

    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {**load_config("configs/defaults.yaml"), **load_config("configs/smoke.yaml")}
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    assert config["NUM_UPDATES"] <= 10, config["NUM_UPDATES"]
    assert config["VAL_INTERVAL"] <= config["NUM_UPDATES"], (
        "no validation rollout would run"
    )
    samples_per_update = config["NUM_ENVS"] * (
        config["NUM_STEPS"] - config["PLAN_HORIZON"] + 1
    )
    assert config["DAGGER_BUFFER_MAX"] >= samples_per_update
    assert config["LR_WARMUP_STEPS"] == 0, "warmup would eat the whole smoke run"
