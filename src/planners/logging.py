"""Centralised W&B logging utilities for ReMDM training loops.

Design principles
-----------------
* ``build_log_dict`` is a pure function — no side effects, fully testable.
* ``make_wandb_callback`` is a factory that returns a closure suitable for
  ``jax.debug.callback``.  All timing state is local to the closure; there is
  no module-level global state.
* Both ``train.py`` and ``online.py`` import the same symbols, keeping all
  metric naming and aggregation logic in one place.

Metric namespacing
------------------
``diffusion/``  — ELBO loss, accuracy, and noise-level diagnostics from ``compute_loss``.
``train/``      — data quality, action distribution, throughput.
``env/``        — episode returns and per-achievement unlock rates (training envs).
``val/``        — same as ``env/`` but from the held-out validation rollout
                  (only emitted when ``step_idx % val_interval == 0``).
``dagger/``     — DAgger-specific metrics (online training only).
"""

from __future__ import annotations

import time
from typing import Any

import wandb


def init_wandb(
    config: dict[str, Any],
    name: str,
    *,
    resume_run_id: str | None = None,
) -> None:
    """Initialise a W&B run, optionally resuming an existing one.

    Args:
        config:        Training config dict (used for ``project``, ``entity``,
                       and logged as run config).
        name:          Human-readable run name.
        resume_run_id: If provided, attaches to an existing W&B run via
                       ``wandb.init(id=..., resume="must")``.  The run must
                       already exist.
    """
    kwargs: dict[str, Any] = {
        "project": config["WANDB_PROJECT"],  # config governs (spec-config §6.5)
        "entity": config.get("WANDB_ENTITY"),
        "config": config,
    }
    if resume_run_id is not None:
        kwargs["id"] = resume_run_id
        kwargs["resume"] = "must"
    else:
        kwargs["name"] = name

    wandb.init(**kwargs)


# Keys emitted by ``src.diffusion.loss.compute_loss`` info dict.
_DIFFUSION_KEYS: tuple[str, ...] = (
    "loss",
    "unweighted_loss",
    "accuracy",
    "acc_t_low",
    "acc_t_mid",
    "acc_t_high",
    "frac_masked",
    "mean_t",
    "grad_norm",
)

# Keys added locally by training loops.
_TRAIN_KEYS: tuple[str, ...] = (
    "action_entropy",
    "action_unique_frac",
    "valid_frac",
    "mean_return_weight",
)

# Keys specific to online DAgger training.
_DAGGER_KEYS: tuple[str, ...] = (
    "beta",
    "reward_mean",
    "buffer_fill",
    "valid_frac",
    "best_val_return",
)


def build_log_dict(
    metric: dict[str, Any],
    step_idx: int,
    val_interval: int,
    *,
    is_online: bool = False,
    sps: float | None = None,
) -> dict[str, float]:
    """Build a flat W&B-ready log dict from a merged training metric dict.

    Args:
        metric:       Merged metric dict from the current update step.
        step_idx:     Integer update step index.
        val_interval: How often (in steps) validation runs occur.
        is_online:    If ``True``, emit DAgger-specific keys under ``dagger/``.
        sps:          Pre-computed steps-per-second; omitted when ``None``.

    Returns:
        Flat ``{str: float}`` dict suitable for ``wandb.log``.
    """
    log: dict[str, float] = {}
    is_val_step = step_idx % val_interval == 0

    for k in _DIFFUSION_KEYS:
        if k in metric:
            log[f"diffusion/{k}"] = float(metric[k])

    for k in _TRAIN_KEYS:
        if k in metric:
            log[f"train/{k}"] = float(metric[k])

    if is_online:
        for k in _DAGGER_KEYS:
            if k in metric:
                log[f"dagger/{k}"] = float(metric[k])

    if "returned_episode_returns" in metric:
        log["env/episode_return"] = float(metric["returned_episode_returns"])
    if "returned_episode_lengths" in metric:
        log["env/episode_length"] = float(metric["returned_episode_lengths"])
    # Cross-repo x-axis alias: same key as the minihack repo's env-step
    # counter, so curves can share an env-frame axis.
    # Counts learner rollout frames only; expert labels are free here,
    # whereas minihack charges oracle steps to its budget.
    if "timestep" in metric:
        log["train/env_steps"] = float(metric["timestep"])

    # Per-achievement breakdown + aggregate score (Craftax reports as %, divide by 100).
    achieve_total = 0.0
    for k, v in metric.items():
        if "achievement" in k.lower() and not k.startswith("val/"):
            log[f"env/achieve/{k}"] = float(v)
            achieve_total += float(v) / 100.0
    log["env/achievements"] = achieve_total

    # Validation metrics — only emitted on val steps to avoid polluting charts with zeros.
    if is_val_step:
        val_achieve_total = 0.0
        for k, v in metric.items():
            if not k.startswith("val/"):
                continue
            inner = k[4:]  # strip leading "val/"
            if "achievement" in inner.lower():
                log[f"val/achieve/{inner}"] = float(v)
                val_achieve_total += float(v) / 100.0
            elif inner == "returned_episode_returns":
                log["val/episode_return"] = float(v)
            elif inner == "returned_episode_lengths":
                log["val/episode_length"] = float(v)
        log["val/achievements"] = val_achieve_total

    if sps is not None:
        # Validation runs inside the timed region on val steps, so the raw
        # rate there measures training + validation, not training slowdown.
        log["train/sps_incl_val" if is_val_step else "train/sps"] = sps

    return log


def make_wandb_callback(
    config: dict[str, Any],
    *,
    steps_per_update: int | None,
    val_interval: int,
    is_online: bool = False,
) -> Any:
    """Return a host-side logging closure for ``jax.debug.callback``.

    The closure tracks wall-clock time between successive calls to compute
    steps-per-second.  All state is local to the closure; there is no
    module-level mutable state.

    SPS is not reported on ``step_idx == 0`` (JIT compilation overhead) or
    when ``steps_per_update`` is ``None`` (e.g. data-replay mode where no
    environment frames are consumed).

    Args:
        config:           Training config dict (read-only; only consulted for
                          ``USE_WANDB`` — callers are expected to guard).
        steps_per_update: Environment frames consumed per update step.  Pass
                          ``None`` to disable ``train/sps`` logging entirely
                          (e.g. when training from pre-collected data files).
        val_interval:     Frequency (in steps) at which validation runs occur.
        is_online:        If ``True``, emit DAgger keys under ``dagger/``.

    Returns:
        A callable ``log_fn(metric, step_idx) -> None`` for
        ``jax.debug.callback``.
    """
    _t: list[float] = [time.time()]

    def log_fn(metric: dict[str, Any], step_idx: int) -> None:
        now = time.time()
        dt = now - _t[0]
        _t[0] = now

        sps: float | None = (
            steps_per_update / dt
            if steps_per_update is not None and int(step_idx) > 0 and dt > 1e-6
            else None
        )
        log = build_log_dict(
            metric,
            int(step_idx),
            val_interval,
            is_online=is_online,
            sps=sps,
        )
        wandb.log(log, step=int(step_idx))

    return log_fn
