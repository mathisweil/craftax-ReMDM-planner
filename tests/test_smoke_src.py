"""Smoke tests for the src/ diffusion planner: build, forward, train, save, sample.

Proves the pipeline runs end to end. Asserts nothing about result quality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from conftest import (
    BATCH,
    NUM_ACTIONS,
    OBS_DIM,
    PLAN_HORIZON,
    SEED,
    SRC_MODULES,
    import_or_skip,
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
        "ONLINE_TOTAL_TIMESTEPS": 320, "LR_WARMUP_FRAMES": 64,
        "VAL_INTERVAL_FRAMES": 320, "DAGGER_BETA_FINAL": 0.1,
        "DAGGER_BUFFER_CYCLES": 2,
    }
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    assert config["LR_WARMUP_STEPS"] == 2
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
    from conftest import load_config

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
    assert samples_per_update <= merged["DAGGER_BUFFER_MAX"]


def test_smoke_budget_resolves_to_a_short_run() -> None:
    """Shrinking the frame budget alone would not shrink the run: check the
    derived update count and the keys that would silently override it."""
    from conftest import load_config

    from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams

    config = {**load_config("configs/defaults.yaml"), **load_config("configs/smoke.yaml")}
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    assert config["NUM_UPDATES"] <= 10, config["NUM_UPDATES"]
    assert config["VAL_INTERVAL"] <= config["NUM_UPDATES"], (
        "no validation rollout would run"
    )
    # PRIMARY env-frame keys silently override the update-step values above.
    for primary in ("VAL_INTERVAL_FRAMES", "DAGGER_BUFFER_CYCLES",
                    "DAGGER_BETA_FINAL", "LR_WARMUP_FRAMES"):
        assert config[primary] is None, f"{primary} would override the smoke sizing"
