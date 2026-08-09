"""Shared gradient-step factory, validation rollout, and action diagnostics.

Both :mod:`src.planners.offline` and :mod:`src.planners.online` use identical
gradient update and validation logic.  Centralising it here eliminates
duplication.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import optax

from src.diffusion.loss import compute_loss
from src.diffusion.sampling import sample_plan
from src.diffusion.schedules import ScheduleFn


def resolve_num_updates(config: dict[str, Any], mode: str) -> None:
    """Resolve ``NUM_UPDATES`` from env-frame-denominated config keys.

    Mutates ``config`` in place.  After this call the runners can read
    ``NUM_UPDATES`` (and ``OFFLINE_TOTAL_TIMESTEPS`` /
    ``ONLINE_TOTAL_TIMESTEPS`` depending on mode) without worrying about
    whether the user specified the env-frame or update-count form.

    Resolution priority:

    ===========  ============================================================
    Mode         Priority (highest first)
    ===========  ============================================================
    ``offline``  ``OFFLINE_TOTAL_TIMESTEPS``  >  ``OFFLINE_NUM_UPDATES``
    ``online``   ``ONLINE_TOTAL_TIMESTEPS``   >  ``ONLINE_NUM_UPDATES``
    ===========  ============================================================

    Env-frame keys are preferred because they are invariant under
    ``num_envs`` changes — the same value yields the same total environment
    experience regardless of hardware sizing, which makes cross-hardware
    fairness studies (e.g. UCL 4096-env vs QMUL 96-env) trivially fair
    without manual scaling.

    The function is idempotent: calling it twice with the same config has
    the same effect as calling it once.

    Args:
        config: Upper-cased config dict.  Must contain ``NUM_STEPS`` and
                ``NUM_ENVS``.
        mode:   Either ``"offline"`` or ``"online"``.

    Raises:
        ValueError: If neither the env-frame nor the update-count form is
                    set for the given mode, or if ``mode`` is unknown.
    """
    frames_per_update = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])

    if mode == "offline":
        ts_key, nu_key = "OFFLINE_TOTAL_TIMESTEPS", "OFFLINE_NUM_UPDATES"
    elif mode == "online":
        ts_key, nu_key = "ONLINE_TOTAL_TIMESTEPS", "ONLINE_NUM_UPDATES"
    else:
        raise ValueError(f"Unknown mode: {mode!r}; expected 'offline' or 'online'.")

    ts = config.get(ts_key)
    nu = config.get(nu_key)
    # float() first to accept YAML scientific notation parsed as string
    # (PyYAML 1.1 only auto-coerces "3.0e+8", not "3e8" or "3.0e8").
    if ts is not None:
        num_updates = max(1, int(float(ts)) // frames_per_update)
    elif nu:
        num_updates = int(float(nu))
    else:
        raise ValueError(
            f"{mode.capitalize()} mode requires either "
            f"{ts_key.lower()!r} (env frames, preferred) or "
            f"{nu_key.lower()!r} to be set."
        )

    config["NUM_UPDATES"] = num_updates
    # Re-snap so downstream consumers (run names, SPS, checkpoint IDs)
    # see the exact integer multiple actually trained.
    config[ts_key] = num_updates * frames_per_update


def resolve_scaled_hyperparams(config: dict[str, Any], mode: str) -> None:
    """Resolve env-frame-denominated hyperparameters into update-step form.

    Mutates ``config`` in place.  PRIMARY (env-frame) keys override LEGACY
    (update-step) keys when set, mirroring the
    :func:`resolve_num_updates` pattern.  When the PRIMARY key is ``None``
    the LEGACY value passes through unchanged, preserving full
    backward compatibility with configs that predate this resolver.

    Resolution table
    ================

    +-----------------------------+------------------------+----------+
    | PRIMARY (env-frame)         | LEGACY (update-step)   | Mode     |
    +=============================+========================+==========+
    | ``LR_WARMUP_FRAMES``        | ``LR_WARMUP_STEPS``    | both     |
    +-----------------------------+------------------------+----------+
    | ``VAL_INTERVAL_FRAMES``     | ``VAL_INTERVAL``       | both     |
    +-----------------------------+------------------------+----------+
    | ``DAGGER_BETA_FINAL``       | ``DAGGER_BETA_DECAY``  | online   |
    +-----------------------------+------------------------+----------+
    | ``DAGGER_BUFFER_CYCLES``    | ``DAGGER_BUFFER_MAX``  | online   |
    +-----------------------------+------------------------+----------+

    Why env-frame units
    -------------------
    Env-frame values are invariant under ``num_envs`` changes, so the
    same config trains the same effective experiment on any GPU.  The
    update-step legacy keys had to be hand-derived per hardware tier,
    which was both error-prone and obscured the conceptual quantity
    (e.g. *final beta*, not *per-update decay constant*).

    The conversion for ``DAGGER_BETA_FINAL`` requires ``NUM_UPDATES``,
    so this function MUST be called after :func:`resolve_num_updates`
    when resolving online mode.

    Idempotent: calling this twice is equivalent to calling it once.

    Args:
        config: Upper-cased config dict.  Must contain ``NUM_STEPS`` and
                ``NUM_ENVS``.
        mode:   Either ``"offline"`` or ``"online"``.

    Raises:
        ValueError: If ``DAGGER_BETA_FINAL`` is set in online mode but
                    ``NUM_UPDATES`` has not been resolved yet.
    """
    fpu = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])

    # float() first to accept YAML scientific notation parsed as string
    # (PyYAML 1.1 only auto-coerces "3.0e+8", not "3e8" or "3.0e8").
    warmup_frames = config.get("LR_WARMUP_FRAMES")
    if warmup_frames is not None:
        config["LR_WARMUP_STEPS"] = int(float(warmup_frames)) // fpu

    val_frames = config.get("VAL_INTERVAL_FRAMES")
    if val_frames is not None:
        config["VAL_INTERVAL"] = max(1, int(float(val_frames)) // fpu)

    if mode != "online":
        return

    beta_final = config.get("DAGGER_BETA_FINAL")
    if beta_final is not None:
        num_updates = config.get("NUM_UPDATES")
        if num_updates is None:
            raise ValueError(
                "DAGGER_BETA_FINAL requires NUM_UPDATES to be resolved "
                "first; call resolve_num_updates() before "
                "resolve_scaled_hyperparams()."
            )
        beta_init = float(config.get("DAGGER_BETA_INIT", 1.0))
        # final = init * decay^N  =>  decay = (final / init) ** (1 / N)
        config["DAGGER_BETA_DECAY"] = (float(beta_final) / beta_init) ** (
            1.0 / int(num_updates)
        )

    buffer_cycles = config.get("DAGGER_BUFFER_CYCLES")
    if buffer_cycles is not None:
        config["DAGGER_BUFFER_MAX"] = max(1, int(round(float(buffer_cycles) * fpu)))


def dagger_sizing(config: dict[str, Any], num_updates: int) -> dict[str, int]:
    """Derive the DAgger loop sizes.

    Single source of truth for :func:`src.planners.online.make_train_dagger`
    and :func:`print_config_snapshot`.  The two used to derive
    ``n_train_passes`` independently and disagreed: the snapshot reported
    ``dagger_buffer_max // samples_per_update`` while the runner used 1,
    overstating ``total_grad_steps`` by 2x on ``defaults.yaml`` and both
    ``final_classic_*`` configs, and by 23x on
    ``classic_exp_c_full_recipe.yaml``.

    Sliding windows (B3): a rollout of ``num_steps`` transitions yields
    ``num_steps - plan_horizon + 1`` windows per environment, not one window
    per plan cycle.

    Buffer capacity (B1): the theoretical maximum is
    ``num_updates * samples_per_update``; ``DAGGER_BUFFER_MAX`` caps it to
    stay inside GPU memory.

    Training passes (B1): each pass redraws a fresh sample of size
    ``samples_per_update`` from the filled portion of the buffer.  The
    default of 1 makes DAgger do exactly the same gradient work per update
    as offline BC (``update_epochs * num_minibatches`` gradient steps over
    ``samples_per_update`` samples), which keeps the compute comparison
    fair.  After the B3 sliding-window fix ``samples_per_update`` is already
    ~16x the legacy cycle-only count, so one pass yields ~34% buffer
    coverage and cumulative coverage across a few updates approaches 100%.
    Set ``DAGGER_TRAIN_PASSES`` above 1 to trade BC fairness for higher
    per-update buffer coverage.

    Args:
        config:      Upper-cased config dict.  Must contain ``NUM_ENVS``,
                     ``NUM_STEPS`` and ``PLAN_HORIZON``.
        num_updates: Resolved update count from :func:`resolve_num_updates`.

    Returns:
        Dict with ``n_cycles``, ``valid_per_rollout``, ``samples_per_update``,
        ``max_buffer_size`` and ``n_train_passes``.
    """
    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    plan_horizon = int(config["PLAN_HORIZON"])

    valid_per_rollout = num_steps - plan_horizon + 1
    samples_per_update = num_envs * valid_per_rollout

    return {
        "n_cycles": num_steps // plan_horizon,
        "valid_per_rollout": valid_per_rollout,
        "samples_per_update": samples_per_update,
        "max_buffer_size": min(
            num_updates * samples_per_update,
            int(config.get("DAGGER_BUFFER_MAX", 100_000)),
        ),
        "n_train_passes": int(config.get("DAGGER_TRAIN_PASSES") or 1),
    }


def print_config_snapshot(config: dict[str, Any], mode: str) -> None:
    """Print a structured banner of training-critical hyperparameters.

    Surfaces fairness-critical, schedule, and architecture parameters at
    the start of every offline/online run so cross-hardware comparisons
    can be sanity-checked at a glance.  Must be called AFTER
    :func:`resolve_num_updates` and :func:`resolve_scaled_hyperparams`
    so the printed values reflect what training will actually use.

    Args:
        config: Upper-cased config dict (post-resolver).
        mode:   Either ``"offline"`` or ``"online"``.
    """
    fpu = int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])
    num_updates = int(config["NUM_UPDATES"])
    minibatch = fpu // int(config["NUM_MINIBATCHES"])
    ts_key = f"{mode.upper()}_TOTAL_TIMESTEPS"
    total_frames = int(config[ts_key])

    bar = "=" * 72
    title = f"{mode.upper()} training — config snapshot"
    print(f"\n{bar}\n  {title}\n{bar}")
    print(f"  env_name              : {config['ENV_NAME']}")
    print(f"  seed                  : {config['SEED']}")

    print("  -- Rollout / hardware --")
    print(f"    num_envs            = {config['NUM_ENVS']}")
    print(f"    num_steps           = {config['NUM_STEPS']}")
    print(f"    fpu (envs*steps)    = {fpu}")
    print(
        f"    num_minibatches     = {config['NUM_MINIBATCHES']}  (minibatch={minibatch})"
    )
    print(f"    update_epochs       = {config['UPDATE_EPOCHS']}")
    print(f"    num_repeats         = {config.get('NUM_REPEATS', 1)}")

    print("  -- Schedule --")
    print(
        f"    {ts_key.lower():<24} = {total_frames:,}  (~{total_frames / 1e6:.1f}M frames)"
    )
    print(f"    {'num_updates':<24} = {num_updates:,}")
    warmup = int(config.get("LR_WARMUP_STEPS", 0))
    print(f"    {'lr':<24} = {float(config['LR']):.2e}")
    print(
        f"    {'lr_warmup_steps':<24} = {warmup}  (~{warmup * fpu / 1e6:.2f}M frames)"
    )
    print(f"    {'max_grad_norm':<24} = {config.get('MAX_GRAD_NORM', 1.0)}")

    if mode == "online":
        beta_init = float(config.get("DAGGER_BETA_INIT", 1.0))
        beta_decay = float(config["DAGGER_BETA_DECAY"])
        final_beta = beta_init * beta_decay**num_updates
        # Derived exactly as the runner derives them; see dagger_sizing.
        sizing = dagger_sizing(config, num_updates)
        samples_per_update = sizing["samples_per_update"]
        buffer_max = sizing["max_buffer_size"]
        n_passes = sizing["n_train_passes"]
        cycles = buffer_max / fpu
        configured_max = int(config["DAGGER_BUFFER_MAX"])
        cap_note = (
            ""
            if buffer_max >= configured_max
            else f"  (capped from {configured_max:,} by num_updates)"
        )
        expert_det = bool(config.get("DAGGER_EXPERT_DETERMINISTIC", True))
        total_grad_steps = (
            num_updates
            * n_passes
            * int(config["UPDATE_EPOCHS"])
            * int(config["NUM_MINIBATCHES"])
        )
        passes_tag = (
            "default" if config.get("DAGGER_TRAIN_PASSES") is None else "override"
        )
        print("  -- DAgger --")
        print(f"    {'dagger_beta_init':<24} = {beta_init}")
        print(f"    {'dagger_beta_decay':<24} = {beta_decay:.10f}")
        print(f"    {'final beta':<24} = {final_beta:.4f}  (init * decay^N)")
        print(
            f"    {'buffer size (effective)':<24} = {buffer_max:,}  (~{cycles:.2f} update cycles){cap_note}"
        )
        print(f"    {'samples_per_update':<24} = {samples_per_update:,}")
        print(f"    {'dagger_train_passes':<24} = {n_passes}  ({passes_tag})")
        print(f"    {'dagger_expert_determ':<24} = {expert_det}")
        print(f"    {'total_grad_steps':<24} = {total_grad_steps:,}")
    else:
        total_grad_steps = (
            num_updates * int(config["UPDATE_EPOCHS"]) * int(config["NUM_MINIBATCHES"])
        )
        print(f"    {'total_grad_steps':<24} = {total_grad_steps:,}")

    val_int = int(config.get("VAL_INTERVAL", 0))  # display-only; fallback 0 != defaults.yaml (50)
    print("  -- Validation --")
    print(
        f"    val_interval        = {val_int} updates  (~{val_int * fpu / 1e6:.2f}M frames)"
    )
    print(f"    val_diffusion_steps = {config.get('VAL_DIFFUSION_STEPS')}")
    print(f"    val_replan_every    = {config.get('VAL_REPLAN_EVERY')}")
    print(f"    val_steps           = {config.get('VAL_STEPS')}")

    print("  -- Diffusion model --")
    print(
        f"    d_model/n_heads/n_layers/d_ff = "
        f"{config['D_MODEL']}/{config['N_HEADS']}/{config['N_LAYERS']}/{config['D_FF']}"
    )
    print(f"    plan_horizon        = {config['PLAN_HORIZON']}")
    print(f"    diffusion_steps     = {config['DIFFUSION_STEPS']}")
    print(
        f"    remask_strategy     = {config.get('REMASK_STRATEGY')}  eta={config.get('ETA')}"
    )
    print(
        f"    sampling: temp={config.get('TEMPERATURE')}  top_p={config.get('TOP_P')}  "
        f"loop={config.get('USE_LOOP')}  t_on/t_off={config.get('T_ON')}/{config.get('T_OFF')}"
    )
    print(f"{bar}\n", flush=True)


def _action_stats(
    acts: jnp.ndarray,
    num_actions: int,
    valid: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    """Compute action-distribution entropy and unique-action fraction over valid windows.

    Args:
        acts:        ``[B, H]`` int32 action sequences.
        num_actions: Size of the real action vocabulary.
        valid:       ``[B]`` bool mask; invalid samples are excluded from counts.

    Returns:
        Dict with ``action_entropy`` and ``action_unique_frac``.
    """
    mask = jnp.broadcast_to(valid[:, None], acts.shape).reshape(-1)
    flat = jnp.where(mask, acts.reshape(-1), num_actions + 1)
    counts = jnp.bincount(flat, length=num_actions).astype(jnp.float32)
    probs = counts / jnp.maximum(counts.sum(), 1.0)
    entropy = -jnp.sum(probs * jnp.log(jnp.where(probs > 0, probs, 1.0)))
    return {
        "action_entropy": entropy,
        "action_unique_frac": jnp.sum(probs > 0).astype(jnp.float32) / num_actions,
    }


def make_grad_step(
    apply_train: Callable,
    num_actions: int,
    schedule_fn: ScheduleFn,
    schedule_deriv_fn: ScheduleFn,
    sigma_t: float,
    label_smoothing: float,
) -> Callable:
    """Return a jittable gradient update function.

    Args:
        apply_train:       Model apply function with dropout enabled.
        num_actions:       Size of the action vocabulary.
        schedule_fn:       alpha(t) noise schedule.
        schedule_deriv_fn: d(alpha)/dt analytic derivative.
        sigma_t:           ReMDM remasking strength during training.
        label_smoothing:   Cross-entropy label smoothing epsilon.

    Returns:
        A ``step(state, acts, obs, valid, rng, advantages) -> (state, metrics)``
        function ready for use inside ``jax.lax.scan``.
    """

    def _loss_fn(
        params: Any,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        rng: jax.Array,
        advantages: jnp.ndarray,
    ) -> tuple[jnp.ndarray, dict]:
        return compute_loss(
            apply_train,
            params,
            rng,
            acts,
            obs,
            valid,
            num_actions,
            schedule_fn,
            schedule_deriv_fn,
            sigma_t=sigma_t,
            label_smoothing=label_smoothing,
            advantages=advantages,
        )

    def step(
        state: Any,
        acts: jnp.ndarray,
        obs: jnp.ndarray,
        valid: jnp.ndarray,
        rng: jax.Array,
        advantages: jnp.ndarray,
    ) -> tuple[Any, dict]:
        """Single gradient update step.

        Args:
            state:      Current ``TrainState``.
            acts:       ``[B, H]`` int32 action sequences.
            obs:        ``[B, obs_dim]`` float32 observations.
            valid:      ``[B]`` bool validity mask (episode-boundary filter).
            rng:        PRNG key for dropout and noise sampling.
            advantages: ``[B]`` float per-sample weights applied before loss reduction.

        Returns:
            Updated ``TrainState`` and a metrics dict.
        """
        (_, info), grads = jax.value_and_grad(_loss_fn, has_aux=True)(
            state.params,
            acts,
            obs,
            valid,
            rng,
            advantages,
        )
        state = state.apply_gradients(grads=grads)
        info["grad_norm"] = optax.tree.norm(grads)
        info.update(_action_stats(acts, num_actions, valid))
        return state, info

    return step


def make_validate(
    env: Any,
    env_params: Any,
    apply_eval: Callable,
    num_actions: int,
    plan_horizon: int,
    schedule_fn: ScheduleFn,
    config: dict[str, Any],
    val_replan_every: int,
    n_val_cycles: int,
) -> Callable:
    """Return a ``validate(state, rng) -> dict`` closure for periodic eval.

    The closure runs a held-out rollout using the diffusion model's current
    parameters and returns metrics under the ``val/`` namespace.

    Args:
        env:              Batched Gymnax environment.
        env_params:       Gymnax environment params.
        apply_eval:       Model apply function (eval mode, no dropout).
        num_actions:      Size of the action vocabulary.
        plan_horizon:     Action plan length H.
        schedule_fn:      alpha(t) noise schedule.
        config:           Training config dict (read-only).
        val_replan_every: Env steps executed per diffusion plan during validation.
        n_val_cycles:     Number of plan-execute cycles per validation rollout.

    Returns:
        A ``validate(state, rng) -> {str: jnp.ndarray}`` closure.
    """

    def validate(state: Any, rng: jax.Array) -> dict[str, jnp.ndarray]:
        """Run a validation rollout and return ``val/`` metrics.

        Args:
            state: Current ``TrainState`` (only ``.params`` is used).
            rng:   PRNG key.

        Returns:
            Dict with ``val/`` prefixed metric keys.
        """
        rng, val_rng = jax.random.split(rng)
        val_obs, val_env_state = env.reset(val_rng, env_params)

        def _val_cycle(carry, _):
            vs, vo, rng = carry
            rng, p_rng = jax.random.split(rng)
            plan = sample_plan(
                apply_eval,
                state.params,
                p_rng,
                vo,
                num_actions,
                plan_horizon,
                num_steps=config.get("VAL_DIFFUSION_STEPS", 50),
                schedule_fn=schedule_fn,
                remask_strategy=config.get("REMASK_STRATEGY", "rescale"),
                eta=config.get("ETA", 0.5),
                use_loop=config.get("USE_LOOP", True),
                t_on=config.get("T_ON", 0.7),
                t_off=config.get("T_OFF", 0.3),
                temperature=config.get("TEMPERATURE", 0.5),
                top_p=config.get("TOP_P", 0.95),
            )  # [num_envs, plan_horizon]

            def _exec_step(inner_carry, step_i):
                vs_i, vo_i, r = inner_carry
                r, s_rng = jax.random.split(r)
                vo_next, vs_next, _, _, info = env.step(
                    s_rng,
                    vs_i,
                    plan[:, step_i],
                    env_params,
                )
                return (vs_next, vo_next, r), info

            (vs, vo, rng), step_infos = jax.lax.scan(
                _exec_step,
                (vs, vo, rng),
                jnp.arange(val_replan_every),
            )
            return (vs, vo, rng), step_infos

        _, cycle_infos = jax.lax.scan(
            _val_cycle,
            (val_env_state, val_obs, rng),
            None,
            n_val_cycles,
        )
        infos = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            cycle_infos,
        )
        returned = infos["returned_episode"]
        metrics = jax.tree.map(
            lambda x: (x * returned).sum() / (returned.sum() + 1e-8),
            infos,
        )
        return {f"val/{k}": v for k, v in metrics.items()}

    return validate
