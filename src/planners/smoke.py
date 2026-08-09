"""Smoke mode: a fast end-to-end sanity check of the DAgger pipeline.

Delegates the entire training loop to :func:`src.planners.online.run_online`
under the shrunken ``configs/smoke.yaml`` overrides, then prints the final
training and validation metrics.  Nothing here reimplements training.

The mode is self-contained: when no expert checkpoint is supplied it
generates a randomly initialised one, so ``--mode smoke`` runs on a clean
clone with no downloads.  A random expert produces meaningless actions by
design; the mode proves the pipeline executes, never that it learns.
"""

from __future__ import annotations

import shutil
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from craftax.craftax_env import make_craftax_env_from_name

from .common import resolve_num_updates, resolve_scaled_hyperparams
from .online import run_online
from .ppo import build_ppo_network

_BAR = "=" * 72


# ---------------------------------------------------------------------------
# Synthetic expert
# ---------------------------------------------------------------------------


def _write_random_expert(
    config: dict[str, Any],
    num_actions: int,
    obs_shape: tuple,
    directory: str,
) -> None:
    """Save a randomly initialised PPO checkpoint that ``load_ppo_agent`` can read.

    The parameter tree and Orbax layout match what
    :func:`src.planners.ppo.load_ppo_params` restores, so the expert loads
    through the normal code path.

    Args:
        config:      Upper-cased config dict.
        num_actions: Size of the discrete action space.
        obs_shape:   Observation shape tuple.
        directory:   Destination checkpoint directory.
    """
    model_type = config.get("PPO_MODEL_TYPE", "ppo_rnn")
    layer_size = config.get("LAYER_SIZE", 512)
    num_envs = config["NUM_ENVS"]

    network = build_ppo_network(model_type, num_actions, layer_size, config)
    rng = jax.random.PRNGKey(int(config.get("SEED") or 0))

    if model_type == "ppo_rnn":
        init_x = (jnp.zeros((1, num_envs, *obs_shape)), jnp.zeros((1, num_envs)))
        params = network.init(rng, jnp.zeros((num_envs, layer_size)), init_x)
    else:
        params = network.init(rng, jnp.zeros((1, *obs_shape)))

    with ocp.CheckpointManager(directory) as mgr:
        mgr.save(0, args=ocp.args.PyTreeSave({"params": params}))
        mgr.wait_until_finished()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _first_repeat(metrics: dict[str, Any]) -> dict[str, np.ndarray]:
    """Drop the vmapped repeat axis, keeping the first seed.

    Args:
        metrics: Metric dict with leading axes ``[num_repeats, num_updates]``.

    Returns:
        Metric dict with arrays of shape ``[num_updates]``.
    """
    return {k: np.asarray(v)[0] for k, v in metrics.items()}


def _last_validated_step(config: dict[str, Any], num_updates: int) -> int | None:
    """Index of the last update whose validation rollout ran.

    Determined from the schedule rather than from the metric values: the
    rollout is episode-weighted, so a validation that completes no episode
    reports exact zeros and is indistinguishable from one that never ran.

    Args:
        config:      Upper-cased config dict (post-resolver).
        num_updates: Number of updates the run performed.

    Returns:
        Index into the metric arrays, or ``None`` if no validation ran.
    """
    val_interval = int(config.get("VAL_INTERVAL", 50))
    start = int(config.get("RESUME_STEP") or 0)
    validated = [i for i in range(num_updates) if (start + i) % val_interval == 0]
    return validated[-1] if validated else None


def _print_summary(metrics: dict[str, Any], config: dict[str, Any]) -> None:
    """Print final training and validation metrics from a DAgger run.

    Args:
        metrics: Metric dict returned by ``run_online``.
        config:  Upper-cased config dict (post-resolver).
    """
    m = _first_repeat(metrics)
    last = -1

    print(f"\n{_BAR}\n  SMOKE TEST SUMMARY\n{_BAR}")
    print(f"  updates completed     : {config['NUM_UPDATES']}")
    print(f"  env frames            : {config['ONLINE_TOTAL_TIMESTEPS']:,}")

    print("  -- Training (final update) --")
    for key, label in (
        ("loss", "loss"),
        ("unweighted_loss", "unweighted loss"),
        ("accuracy", "action accuracy"),
        ("grad_norm", "grad norm"),
        ("action_entropy", "action entropy"),
        ("beta", "dagger beta"),
        ("buffer_fill", "buffer fill"),
    ):
        if key in m:
            print(f"    {label:<20} = {float(m[key][last]):.4f}")

    print("  -- Environment (final update) --")
    if "reward_mean" in m:
        print(f"    {'mean step reward':<20} = {float(m['reward_mean'][last]):.4f}")
    if "returned_episode_returns" in m:
        print(f"    {'episode return':<20} = {float(m['returned_episode_returns'][last]):.3f}")
    if "returned_episode_lengths" in m:
        print(f"    {'episode length':<20} = {float(m['returned_episode_lengths'][last]):.1f}")

    achievements = {
        k: float(v[last]) for k, v in m.items()
        if "achievement" in k.lower() and not k.startswith("val/")
    }
    if achievements:
        # Craftax reports unlock rates as percentages.
        total = sum(achievements.values()) / 100.0
        unlocked = [k for k, v in achievements.items() if v > 0.0]
        print(f"    {'achievements':<20} = {total:.3f} over {len(achievements)} tracked")
        print(f"    {'unlocked':<20} = {len(unlocked)}")

    val_idx = _last_validated_step(config, len(m["loss"]))
    print("  -- Validation --")
    if val_idx is None:
        print("    no validation rollout ran (lower val_interval)")
    else:
        print(f"    {'ran at update':<20} = {val_idx}")
        for key, label in (
            ("val/returned_episode_returns", "episode return"),
            ("val/returned_episode_lengths", "episode length"),
        ):
            if key in m:
                print(f"    {label:<20} = {float(m[key][val_idx]):.3f}")

    finite = all(
        np.all(np.isfinite(np.asarray(v))) for v in m.values()
    )
    print(f"  -- Health --\n    {'all metrics finite':<20} = {finite}")
    print(
        "\n  Episode-weighted metrics (returns, lengths, achievements) read\n"
        "  0.000 until an episode terminates. A smoke run is far shorter than\n"
        "  a Craftax episode, so zeros here are expected, not a failure."
    )
    print(f"{_BAR}\n")

    if not finite:
        raise ValueError("Smoke test produced non-finite metrics")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_smoke(config: dict[str, Any]) -> None:
    """Run a shrunken DAgger job end to end and print its metrics.

    Args:
        config: Upper-cased config dict, normally ``configs/smoke.yaml``
                overlaid on ``configs/defaults.yaml`` by ``main.build_config``.

    Raises:
        ValueError: If any returned metric is non-finite.
    """
    config = {k.upper(): v for k, v in config.items()}

    # Idempotent, and run_online repeats them; done here so the summary can
    # report the resolved budget.
    resolve_num_updates(config, "online")
    resolve_scaled_hyperparams(config, "online")

    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    tmp_expert: str | None = None
    if not config.get("PPO_CHECKPOINT_PATH"):
        tmp_expert = tempfile.mkdtemp(prefix="remdm-smoke-expert-")
        _write_random_expert(config, num_actions, obs_shape, tmp_expert)
        config["PPO_CHECKPOINT_PATH"] = tmp_expert
        print(
            f"\n{_BAR}\n"
            "  No expert checkpoint given: using a RANDOMLY INITIALISED PPO expert.\n"
            "  This checks that the pipeline runs; the resulting numbers are\n"
            "  meaningless. Pass --ppo-checkpoint to use a trained expert.\n"
            f"{_BAR}"
        )
    else:
        print(f"\nSmoke test using expert: {config['PPO_CHECKPOINT_PATH']}")

    try:
        out = run_online(config)
    finally:
        if tmp_expert is not None:
            shutil.rmtree(tmp_expert, ignore_errors=True)

    _print_summary(out["metrics"], config)
    _smoke_inference_leg(out, config, num_actions, obs_shape)


def _smoke_inference_leg(out, config, num_actions, obs_shape) -> None:
    """Exercise the corrected inference-time sampler on the trained params.

    FIX-2 (ADJUDICATION B-3) rebuilt ``sample_plan_inpainting`` on the
    conforming ReMDM Algorithm 1 core; without this leg a smoke run would
    silently bypass the inference code path (it trains and validates via
    ``run_online`` only). Fails loudly on MASK leakage or a violated
    prefix lock.
    """
    import jax
    import jax.numpy as jnp

    from src.diffusion.sampling import sample_plan_inpainting
    from src.diffusion.schedules import SCHEDULE_MAP
    from src.planners.model import build_model, make_apply_fns

    params = jax.tree.map(lambda x: x[0], out["runner_state"].train_state.params)
    model = build_model(config, num_actions)
    apply_eval, _ = make_apply_fns(model)
    schedule_fn, _ = SCHEDULE_MAP[config.get("DIFFUSION_SCHEDULE", "cosine")]

    plan_horizon = int(config["PLAN_HORIZON"])
    obs = jnp.zeros((2, obs_shape[0]))
    history = jnp.zeros((2, plan_horizon), dtype=jnp.int32)
    hist_len = jnp.array([2, 0], dtype=jnp.int32)

    plan = sample_plan_inpainting(
        apply_eval, params, jax.random.PRNGKey(int(config["SEED"])), obs,
        history, hist_len, num_actions, plan_horizon,
        int(config.get("DIFFUSION_STEPS_EVAL", 10)), schedule_fn,
        config.get("REMASK_STRATEGY", "rescale"), config.get("ETA", 0.5),
        config.get("USE_LOOP", True), config.get("T_ON", 0.7),
        config.get("T_OFF", 0.3),
        config.get("TEMPERATURE", 0.5), config.get("TOP_P", 0.95),
    )
    assert plan.shape == (2, plan_horizon)
    assert bool(jnp.all((plan >= 0) & (plan < num_actions))), (
        "inference sampler emitted MASK or out-of-vocabulary tokens"
    )
    assert bool(jnp.all(plan[0, :2] == history[0, :2])), "prefix lock violated"
    print("  inference sampler (prefix-locked ReMDM Alg 1): OK")
