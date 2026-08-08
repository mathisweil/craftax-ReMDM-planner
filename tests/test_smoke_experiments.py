"""Smoke tests for the experiments/rl_finetuning ablation pipeline.

Covers the second training pipeline: every registered ablation's loss and
optimizer, the LoRA path, the diagnostics, and the analysis/reporting stage.
Asserts nothing about result quality.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import pytest

from conftest import (
    BATCH,
    EXPERIMENT_MODULES,
    NUM_ACTIONS,
    PLAN_HORIZON,
    SEED,
    import_or_skip,
)


def _finite(x) -> bool:
    return bool(jnp.all(jnp.isfinite(x)))


# ---------------------------------------------------------------------------
# Fixtures local to the ablation pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def abl_config(real_ablations_config, tiny_config) -> dict:
    """Real ablations config with the architecture and loop sizes shrunk."""
    return {
        **real_ablations_config,
        **{k: tiny_config[k] for k in ("D_MODEL", "N_HEADS", "N_LAYERS", "D_FF",
                                       "OBS_ENCODER_LAYERS", "OBS_ENCODER_WIDTH")},
        "PLAN_HORIZON": PLAN_HORIZON,
        "NUM_ACTIONS": NUM_ACTIONS,
        "MAX_ITER": 2,
        "NUM_ENVS": 2,
        "NUM_STEPS": 8,
        "BATCH_SIZE": BATCH,
        "EVAL_EVERY": 1,
        # Smallest loop the eval fn accepts: one plan, one env step, one
        # denoising step. The multi-phase sampler is covered in test_smoke_src.
        "EVAL_STEPS": 1,
        "EVAL_REPLAN": 1,
        "VAL_DIFFUSION_STEPS": 1,
        "USE_LOOP": False,
        "USE_WANDB": False,
        "SEED": SEED,
    }


@pytest.fixture(scope="module")
def loss_ctx(abl_config, apply_fns, params, schedules):
    from experiments.rl_finetuning.ablations.losses import LossContext

    _, apply_train = apply_fns
    return LossContext(
        apply_fn=apply_train,
        ref_params=params,
        schedule_fn=schedules[0],
        schedule_deriv_fn=schedules[1],
        num_actions=NUM_ACTIONS,
        config=abl_config,
    )


@pytest.fixture(scope="module")
def fisher(loss_ctx, params, batch):
    """Fisher diagonal for the EWC ablation.

    Called under jax.jit: the function loops in Python over batches and takes
    grads eagerly, which costs ~5.4s traced op by op versus ~0.8s compiled.
    """
    from experiments.rl_finetuning.ablations.losses import estimate_fisher_diagonal

    return jax.jit(
        lambda p: estimate_fisher_diagonal(
            loss_ctx.apply_fn, p, loss_ctx.schedule_fn, loss_ctx.schedule_deriv_fn,
            NUM_ACTIONS, batches=[(batch["acts"], batch["obs"], batch["valid"])],
        )
    )(params)


def _build_loss_fn(spec, ctx, fisher_tree):
    """Instantiate a spec's loss exactly as ablations/training.py does."""
    from experiments.rl_finetuning.ablations.losses import make_loss_t_curriculum_jit

    if spec.t_curriculum:
        return make_loss_t_curriculum_jit(ctx), True

    extras = {"fisher": fisher_tree} if spec.name == "ewc" else {}
    return spec.loss_factory(
        ctx, **{k: v for k, v in extras.items() if k in spec.extra_loss_kwargs},
    ), False


def _synthetic_history():
    """An AblationHistory with one entry in every field the analysis reads."""
    from experiments.rl_finetuning.ablations.training import AblationHistory

    return AblationHistory(
        iters=[0, 1], loss=[1.0, 0.9],
        env_score_iters=[0, 1], env_score=[0.1, 0.2],
        eval_iters=[0, 1], eval_score=[0.1, 0.2],
        grad_align_iters=[0, 1], grad_align=[0.5, 0.4],
        rl_grad_norm=[1.0, 1.1], bc_grad_norm=[1.0, 0.9],
        per_layer_iters=[0, 1],
        per_layer_norms=[{"layer_0": 0.5}, {"layer_0": 0.6}],
        repr_drift_iters=[0, 1], repr_drift_kl=[0.01, 0.02],
        repr_drift_kl_low_t=[0.01, 0.02], repr_drift_kl_mid_t=[0.01, 0.02],
        repr_drift_kl_high_t=[0.01, 0.02],
        cka_iters=[0, 1], cka_similarity=[0.99, 0.98],
        t_analysis_iters=[0, 1], norm_low_t=[0.3, 0.4], norm_high_t=[0.6, 0.7],
        lowhigh_cos=[0.2, 0.1],
        t_bin_norms=[{"0": 0.1, "1": 0.2}, {"0": 0.15, "1": 0.25}],
        win_rate=[0.4, 0.5], effective_batch_size=[8.0, 9.0],
        surgery_iters=[0, 1], surgery_fraction=[0.1, 0.2],
        surgery_n_conflicting=[1, 2],
        per_achievement_rates=[{"collect_wood": 0.5}, {"collect_wood": 0.4}],
    )


@pytest.fixture(scope="module")
def synthetic_results():
    return {
        "baseline_rl": {
            "history": _synthetic_history(), "score": 0.2, "score_std": 0.01,
            "all_scores": [0.19, 0.21], "seeds": [0, 1],
        },
        "kl_penalty": {
            "history": _synthetic_history(), "score": 0.3, "score_std": 0.02,
            "all_scores": [0.28, 0.32], "seeds": [0, 1],
        },
    }


# ---------------------------------------------------------------------------
# 1. Imports and registry
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("module_name", EXPERIMENT_MODULES)
def test_experiment_module_imports(module_name: str) -> None:
    assert import_or_skip(module_name) is not None


def test_registry_is_well_formed() -> None:
    from experiments.rl_finetuning.ablations.registry import REGISTRY, AblationSpec

    assert REGISTRY, "ablation registry is empty"
    for name, spec in REGISTRY.items():
        assert isinstance(spec, AblationSpec)
        assert spec.name == name, f"registry key {name!r} != spec.name {spec.name!r}"
        assert spec.group in {"Baseline", "A", "B", "C", "D"}, spec.group
        assert callable(spec.loss_factory)
        assert callable(spec.optimizer_factory)


# ---------------------------------------------------------------------------
# 2. Every ablation: loss builds, evaluates finite, and yields finite grads
# ---------------------------------------------------------------------------


def _registry_names() -> list[str]:
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    return sorted(REGISTRY)


@pytest.mark.parametrize("ablation_name", _registry_names())
def test_ablation_loss_and_gradients_are_finite(
    ablation_name, loss_ctx, fisher, params, batch,
) -> None:
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    spec = REGISTRY[ablation_name]
    loss_fn, takes_step = _build_loss_fn(spec, loss_ctx, fisher)
    args = (batch["acts"], batch["obs"], batch["valid"],
            jax.random.PRNGKey(SEED), batch["advantages"])
    call = (
        (lambda p: loss_fn(p, *args, jnp.array(1))) if takes_step
        else (lambda p: loss_fn(p, *args))
    )

    loss, grads = jax.value_and_grad(call)(params)

    assert loss.shape == ()
    assert _finite(loss), f"{ablation_name}: non-finite loss"
    assert all(_finite(g) for g in jax.tree_util.tree_leaves(grads)), (
        f"{ablation_name}: non-finite gradient"
    )


@pytest.mark.parametrize("ablation_name", _registry_names())
def test_ablation_optimizer_takes_one_step(ablation_name, abl_config, params) -> None:
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    spec = REGISTRY[ablation_name]
    tx = spec.optimizer_factory(abl_config, params)
    opt_state = tx.init(params)

    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.01, params)
    updates, _ = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    assert all(_finite(p) for p in jax.tree_util.tree_leaves(new_params))


# ---------------------------------------------------------------------------
# 3. One fine-tuning step, end to end
# ---------------------------------------------------------------------------


def test_one_finetuning_step(loss_ctx, abl_config, params, batch) -> None:
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    spec = REGISTRY["baseline_rl"]
    loss_fn, _ = _build_loss_fn(spec, loss_ctx, None)
    tx = spec.optimizer_factory(abl_config, params)
    opt_state = tx.init(params)

    loss, grads = jax.value_and_grad(loss_fn)(
        params, batch["acts"], batch["obs"], batch["valid"],
        jax.random.PRNGKey(SEED), batch["advantages"],
    )
    updates, _ = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    assert _finite(loss)
    assert all(_finite(p) for p in jax.tree_util.tree_leaves(new_params))
    assert any(
        not jnp.array_equal(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(params),
                        jax.tree_util.tree_leaves(new_params))
    ), "fine-tuning step left every parameter untouched"


# ---------------------------------------------------------------------------
# 4. LoRA path and gradient surgery
# ---------------------------------------------------------------------------


def test_lora_targets_the_attention_kernels(abl_config, params) -> None:
    """Attention kernels are 3-D; targeting only 2-D matched nothing at all."""
    from experiments.rl_finetuning.ablations.optimizers import make_lora_params

    lora = make_lora_params(params, abl_config["LORA_RANK"], jax.random.PRNGKey(SEED))

    assert lora, "no LoRA targets found in the parameter tree"
    assert all("MultiHeadDotProductAttention" in path for path in lora)
    assert any(path.endswith("out/kernel") for path in lora), "out projection missed"
    assert any(path.endswith("query/kernel") for path in lora), "query projection missed"

    # Every factorisation must reconstruct its kernel's exact shape.
    shapes = {
        "/".join(str(k.key) for k in path): leaf.shape
        for path, leaf in jax.tree_util.tree_flatten_with_path(params)[0]
    }
    for path, ab in lora.items():
        assert ab["A"].shape[1] == abl_config["LORA_RANK"]
        assert ab["B"].shape[0] == abl_config["LORA_RANK"]
        delta = ab["A"] @ ab["B"]
        assert delta.size == int(jnp.prod(jnp.array(shapes[path]))), path
        assert float(jnp.max(jnp.abs(delta))) == 0.0, "B must be zero-initialised"


def test_lora_apply_and_optimizer_run(abl_config, apply_fns, params, batch) -> None:
    """LoRA injection and masked optimizer, on Dense kernels so targets exist."""
    from experiments.rl_finetuning.ablations.optimizers import (
        apply_fn_with_lora,
        make_lora_params,
        make_optimizer_lora_only,
    )

    apply_eval, _ = apply_fns
    rank, alpha = abl_config["LORA_RANK"], abl_config["LORA_ALPHA"]
    lora = make_lora_params(
        params, rank, jax.random.PRNGKey(SEED), path_fragment="Dense",
    )
    assert lora, "no 2-D Dense kernels to attach LoRA to"

    logits = apply_fn_with_lora(
        apply_eval, params, lora, alpha, rank,
        batch["obs"], batch["acts"], batch["timestep"],
    )
    assert logits.shape == (BATCH, PLAN_HORIZON, NUM_ACTIONS)
    assert _finite(logits)

    baseline = apply_eval(params, batch["obs"], batch["acts"], batch["timestep"])
    assert jnp.allclose(logits, baseline, atol=1e-5), (
        "zero-initialised LoRA must leave the base model unchanged"
    )

    combined = {"base": params, "lora": lora}
    tx = make_optimizer_lora_only(abl_config, params, lora)
    opt_state = tx.init(combined)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.01, combined)
    updates, _ = tx.update(grads, opt_state, combined)
    updated = optax.apply_updates(combined, updates)

    assert all(_finite(p) for p in jax.tree_util.tree_leaves(updated["lora"]))
    assert all(_finite(p) for p in jax.tree_util.tree_leaves(updated["base"]))


def test_frozen_paths_receive_no_update(abl_config, params) -> None:
    """Regression: optax.masked passes NON-selected leaves through unchanged
    rather than zeroing them, so masking in the trainable params handed every
    'frozen' parameter its raw clipped gradient. Now multi_transform +
    set_to_zero."""
    from experiments.rl_finetuning.ablations.optimizers import make_optimizer_frozen_paths

    fragments = ["TransformerBlock"]
    tx = make_optimizer_frozen_paths(abl_config, params, fragments)
    opt_state = tx.init(params)
    grads = jax.tree.map(lambda p: jnp.full_like(p, 0.5), params)
    updates, _ = tx.update(grads, opt_state, params)

    frozen = [
        leaf for path, leaf in jax.tree_util.tree_flatten_with_path(updates)[0]
        if any(f in "/".join(str(k.key) for k in path) for f in fragments)
    ]
    assert frozen, "test selected no frozen leaves"
    assert all(float(jnp.max(jnp.abs(u))) == 0.0 for u in frozen), (
        "frozen parameters received a non-zero update"
    )


def test_lora_optimizer_freezes_the_base(abl_config, apply_fns, params) -> None:
    from experiments.rl_finetuning.ablations.optimizers import (
        make_lora_params,
        make_optimizer_lora_only,
    )

    lora = make_lora_params(
        params, abl_config["LORA_RANK"], jax.random.PRNGKey(SEED), path_fragment="Dense",
    )
    combined = {"base": params, "lora": lora}
    tx = make_optimizer_lora_only(abl_config, params, lora)
    opt_state = tx.init(combined)
    grads = jax.tree.map(lambda p: jnp.ones_like(p) * 0.01, combined)
    updates, _ = tx.update(grads, opt_state, combined)
    updated = optax.apply_updates(combined, updates)

    assert all(
        jnp.array_equal(a, b)
        for a, b in zip(jax.tree_util.tree_leaves(combined["base"]),
                        jax.tree_util.tree_leaves(updated["base"]))
    ), "LoRA optimizer must not update the frozen base"


def test_gradient_surgery_projects_conflicts(params) -> None:
    from experiments.rl_finetuning.ablations.optimizers import gradient_surgery

    g_rl = jax.tree.map(lambda p: jnp.ones_like(p), params)
    g_bc = jax.tree.map(lambda p: -jnp.ones_like(p), params)
    projected = gradient_surgery(g_rl, g_bc)

    assert all(_finite(g) for g in jax.tree_util.tree_leaves(projected))
    for proj, ref in zip(jax.tree_util.tree_leaves(projected),
                         jax.tree_util.tree_leaves(g_bc)):
        assert float(jnp.sum(proj * ref)) <= 1e-4, "conflict was not removed"


# ---------------------------------------------------------------------------
# 5. Diagnostics
# ---------------------------------------------------------------------------


def test_gradient_diagnostics(apply_fns, params, batch, schedules) -> None:
    from experiments.rl_finetuning.diagnostics.gradient import (
        compute_per_layer_grad_norms_jax,
        compute_surgery_metrics_jax,
        make_grad_alignment_fn,
    )

    _, apply_train = apply_fns
    align = make_grad_alignment_fn(apply_train, schedules[0], schedules[1], NUM_ACTIONS)
    cos_sim, rl_norm, bc_norm = align(
        params, params, batch["acts"], batch["obs"], batch["valid"],
        jax.random.PRNGKey(SEED), batch["advantages"],
    )
    assert all(_finite(v) for v in (cos_sim, rl_norm, bc_norm))

    grads = jax.tree.map(jnp.ones_like, params)
    norms = compute_per_layer_grad_norms_jax(grads)
    assert norms.size > 0 and _finite(norms)

    frac, n_conflict = compute_surgery_metrics_jax(grads, jax.tree.map(jnp.negative, grads))
    assert _finite(frac) and _finite(n_conflict)


def test_representation_diagnostics(apply_fns, params, batch, schedules, abl_config) -> None:
    from experiments.rl_finetuning.diagnostics.representation import make_cka_fn, make_repr_drift_fn

    apply_eval, _ = apply_fns
    drift = make_repr_drift_fn(apply_eval, schedules[0], NUM_ACTIONS)
    kls = drift(params, params, batch["obs"], batch["acts"], jax.random.PRNGKey(SEED))
    assert len(kls) == 4
    assert all(_finite(v) for v in kls)

    # cka_batch_size must not exceed the batch: the fn slices obs to that size.
    cka = make_cka_fn(apply_eval, schedules[0], NUM_ACTIONS, cka_batch_size=BATCH)
    value = cka(params, params, batch["obs"], batch["acts"], jax.random.PRNGKey(SEED))
    assert _finite(value)


def test_timestep_diagnostics(apply_fns, params, batch, schedules) -> None:
    from experiments.rl_finetuning.diagnostics.timestep import make_t_analysis_fn

    _, apply_train = apply_fns
    analyse = make_t_analysis_fn(
        apply_train, schedules[0], schedules[1], NUM_ACTIONS, n_bins=3,
    )
    bin_norms, low_high_cos, norm_low, norm_high = analyse(
        params, batch["acts"], batch["obs"], batch["valid"],
        batch["advantages"], jax.random.PRNGKey(SEED),
    )

    assert bin_norms.shape == (3,)
    assert all(_finite(v) for v in (bin_norms, low_high_cos, norm_low, norm_high))


# ---------------------------------------------------------------------------
# 6. History, results I/O and analysis outputs
# ---------------------------------------------------------------------------


def test_history_dict_roundtrip() -> None:
    from experiments.rl_finetuning.ablations.training import AblationHistory

    history = _synthetic_history()
    restored = AblationHistory.from_dict(history.to_dict())

    assert restored.to_dict() == history.to_dict()
    assert AblationHistory.from_dict({"unknown_field": [1]}).iters == []


def test_results_json_roundtrip_and_merge(tmp_path, synthetic_results, abl_config) -> None:
    from experiments.rl_finetuning.run_ablations import (
        _merge_result_files,
        _results_from_json,
        _results_to_json,
    )

    payload = _results_to_json(synthetic_results, 0.15, abl_config, {"collect_wood": 0.5})
    path_a = tmp_path / "results_a.json"
    path_b = tmp_path / "results_b.json"
    path_a.write_bytes(payload)
    path_b.write_bytes(payload)

    results, pretrained, ach_rates, config = _results_from_json(str(path_a))
    assert set(results) == set(synthetic_results)
    assert pretrained == pytest.approx(0.15)
    assert ach_rates["collect_wood"] == pytest.approx(0.5)
    assert config["ENV_NAME"] == abl_config["ENV_NAME"]

    merged, merged_pretrained, _, _ = _merge_result_files([str(path_a), str(path_b)])
    assert len(merged["baseline_rl"]["all_scores"]) == 4
    assert merged_pretrained == pytest.approx(0.15)

    with pytest.raises(ValueError):
        _merge_result_files([])


def test_analysis_stage_writes_tables_plots_and_report(tmp_path, synthetic_results) -> None:
    from experiments.rl_finetuning.analysis.plots import generate_all_plots
    from experiments.rl_finetuning.analysis.report import generate_diagnosis_report
    from experiments.rl_finetuning.analysis.tables import generate_summary_tables

    ach_rates = {"collect_wood": 0.5}
    tables = generate_summary_tables(synthetic_results, 0.15, tmp_path, ach_rates)
    generate_all_plots(synthetic_results, 0.15, tmp_path, ach_rates)
    report_path = generate_diagnosis_report(synthetic_results, 0.15, tables, tmp_path)

    assert tables and "main_results" in tables
    assert list((tmp_path / "tables").glob("*")), "no tables written"
    assert list((tmp_path / "figures").glob("*.png")), "no figures written"
    assert Path(report_path).exists()


def test_fast_overrides_shrink_the_loop(real_ablations_config) -> None:
    from experiments.rl_finetuning.run_ablations import _apply_fast_overrides

    fast = _apply_fast_overrides(real_ablations_config)

    assert fast["MAX_ITER"] < real_ablations_config["MAX_ITER"]
    assert fast["NUM_ENVS"] < real_ablations_config["NUM_ENVS"]
    assert fast["ENV_NAME"] == real_ablations_config["ENV_NAME"]


# ---------------------------------------------------------------------------
# 7. Eval loop against the real environment
# ---------------------------------------------------------------------------


def test_eval_fn_runs_against_env(craftax_env, abl_config, schedules) -> None:
    from experiments.rl_finetuning.ablations.training import build_eval_fn
    from src.planners.model import build_model, init_params, make_apply_fns

    num_actions, obs_dim = craftax_env["num_actions"], craftax_env["obs_dim"]
    config = {**abl_config, "NUM_ACTIONS": num_actions}

    model = build_model(config, num_actions)
    real_dim_params = init_params(model, jax.random.PRNGKey(SEED), obs_dim, PLAN_HORIZON)
    apply_eval, _ = make_apply_fns(model)

    eval_fn = build_eval_fn(
        craftax_env["env"], craftax_env["env_params"], apply_eval, config,
    )
    info = eval_fn(real_dim_params, jax.random.PRNGKey(SEED))

    assert info, "eval produced no metrics"
    assert "returned_episode_returns" in info
    assert all(_finite(v) for v in info.values())


# ---------------------------------------------------------------------------
# 8. Entry point in minimal smoke mode
# ---------------------------------------------------------------------------


def test_run_ablations_help(entry_point_runs) -> None:
    result = entry_point_runs["run_ablations --help"]
    assert result.returncode == 0, result.stderr[-2000:]
    assert "usage" in result.stdout.lower()


def test_run_ablations_list(entry_point_runs) -> None:
    """--list exercises the whole experiments import chain and the registry."""
    from experiments.rl_finetuning.ablations.registry import REGISTRY

    result = entry_point_runs["run_ablations --list"]
    assert result.returncode == 0, result.stderr[-2000:]
    for name in REGISTRY:
        assert name in result.stdout


def test_run_ablations_requires_checkpoints(entry_point_runs) -> None:
    result = entry_point_runs["run_ablations no-checkpoint"]
    assert result.returncode != 0
    assert "checkpoint_path" in result.stderr
