from __future__ import annotations

import argparse
import pathlib
from typing import Any

import sys; sys.path.append(str(pathlib.Path(__file__).resolve().parent / "Craftax_Baselines"))

import os

# Must be set BEFORE `import jax` — JAX/XLA reads logging config at import time.
# Suppresses XLA Triton autotuner noise (rejected kernel configs logged at ERROR).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import numpy as np
import yaml

from src.planners.collect import run_collect
from src.planners.common import resolve_num_updates, resolve_scaled_hyperparams
from src.planners.model import load_checkpoint_metadata, resolve_checkpoint_path
from src.planners.offline import run_offline_diffusion
from src.planners.online import run_online
from src.planners.inference import run_inference
from src.planners.smoke import run_smoke

REMASK_STRATEGIES = ["rescale", "cap", "conf"]
DIFFUSION_SCHEDULES = ["cosine", "linear"]
PPO_TYPES = ["ppo", "ppo_rnn", "ppo_rnd"]

# Config keys set by run-level CLI flags rather than defaults.yaml. They are
# also legal in config files (smoke.yaml sets ppo_checkpoint_path: null so the
# smoke mode runs with a random expert on a clean clone).
_CLI_CONFIG_KEYS = {
    "ppo_checkpoint_path",
    "checkpoint_path",
    "offline_checkpoint_path",
    "offline_data_path",
    "inference_output",
}


# =============================================================================
# Parser
# =============================================================================

def _build_parser(default_cfg_path: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax",
    )

    p.add_argument(
        "--config", default=default_cfg_path,
        help="Experiment config, deep-merged onto configs/defaults.yaml",
    )
    p.add_argument(
        "--mode", required=True,
        choices=["collect", "offline", "online", "inference", "smoke"],
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Run seed (overrides the config value)",
    )
    p.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--override", action="append", default=[], metavar="KEY=VALUE",
        help=(
            "Config override, repeatable. Keys are validated against "
            "configs/defaults.yaml; unknown keys are an error."
        ),
    )

    p.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Planner checkpoint: evaluated by --mode inference, warm-starts "
            "--mode online/smoke. Accepts wandb: references."
        ),
    )
    p.add_argument(
        "--ppo-checkpoint", type=str, default=None,
        help=(
            "PPO expert checkpoint, required by collect/offline/online. "
            "Accepts wandb: references."
        ),
    )
    p.add_argument(
        "--data", type=str, default=None,
        help="Output .npz path for --mode collect",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Optional JSON path for machine-readable inference results (C-006).",
    )

    p.add_argument(
        "--resume", type=str, default=None,
        help=(
            "Checkpoint to resume a completed offline/online run from. "
            "Accepts wandb: references."
        ),
    )
    p.add_argument("--resume-step", type=int, default=None)
    p.add_argument("--resume-wandb-run-id", type=str, default=None)

    return p


# =============================================================================
# Overrides
# =============================================================================

def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for item in pairs:
        if "=" not in item:
            raise ValueError(f"--override expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        overrides[key] = value
    return overrides


def _validate_keys(keys, allowed: set[str], source: str) -> None:
    """Reject unknown config keys instead of silently ignoring them."""
    unknown = sorted(set(keys) - allowed)
    if unknown:
        raise ValueError(
            f"Unknown config key(s) {unknown} in {source}. "
            "Valid keys are defined in configs/defaults.yaml."
        )


def _cast_override(key: str, raw: str, current) -> object:
    """Cast a CLI override string to the type of the current config value."""
    if isinstance(current, str):
        return raw

    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw

    if current is None or value is None:
        return value

    # YAML 1.1 reads '1e-4' as a string; accept scientific notation for
    # numeric keys.
    if (
        isinstance(current, (int, float))
        and not isinstance(current, bool)
        and isinstance(value, str)
    ):
        try:
            value = float(value)
        except ValueError:
            pass

    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ValueError(f"'{key}' expects a boolean, got '{raw}'")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' expects an integer, got '{raw}'")
        if isinstance(value, float):
            if not value.is_integer():
                raise ValueError(f"'{key}' expects an integer, got '{raw}'")
            value = int(value)
        return value
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"'{key}' expects a number, got '{raw}'")
        return float(value)
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ValueError(f"'{key}' expects a list, got '{raw}'")
        return value
    return value


# =============================================================================
# Config
# =============================================================================

def build_config() -> dict[str, Any]:
    default_cfg = str(pathlib.Path(__file__).parent / "configs" / "defaults.yaml")
    smoke_cfg = str(pathlib.Path(__file__).parent / "configs" / "smoke.yaml")

    parser = _build_parser(default_cfg)
    args = parser.parse_args()

    with open(default_cfg) as f:
        yaml_cfg: dict[str, Any] = yaml.safe_load(f) or {}
    allowed = set(yaml_cfg) | _CLI_CONFIG_KEYS

    # Smoke mode overlays configs/smoke.yaml on the defaults.  Only when the
    # user did not name their own --config, so an explicit config always wins.
    # Exactly two layers: defaults.yaml, then the named preset.  A preset never
    # inherits from another preset.
    overlay_path = None
    if args.mode == "smoke" and args.config == default_cfg:
        overlay_path = smoke_cfg
    elif args.config != default_cfg:
        overlay_path = args.config

    if overlay_path is not None:
        with open(overlay_path) as f:
            overlay = yaml.safe_load(f) or {}
        _validate_keys(overlay, allowed, str(overlay_path))
        yaml_cfg.update(overlay)

    overrides = _parse_overrides(args.override)
    _validate_keys(overrides, allowed, "--override")
    for key, raw in overrides.items():
        yaml_cfg[key] = _cast_override(key, raw, yaml_cfg.get(key))

    config: dict[str, Any] = {k.upper(): v for k, v in yaml_cfg.items()}

    # Run-level flags override config values.
    config["MODE"] = args.mode
    config["JIT"] = args.jit
    if args.seed is not None:
        config["SEED"] = args.seed
    if args.ppo_checkpoint is not None:
        config["PPO_CHECKPOINT_PATH"] = args.ppo_checkpoint
    if args.data is not None:
        config["OFFLINE_DATA_PATH"] = args.data
    if args.output is not None:
        config["INFERENCE_OUTPUT"] = args.output
    if args.resume is not None:
        config["RESUME_CHECKPOINT_PATH"] = args.resume
    if args.resume_step is not None:
        config["RESUME_STEP"] = args.resume_step
    if args.resume_wandb_run_id is not None:
        config["RESUME_WANDB_RUN_ID"] = args.resume_wandb_run_id

    if args.checkpoint is not None:
        if args.mode == "inference":
            config["CHECKPOINT_PATH"] = args.checkpoint
        elif args.mode in {"online", "smoke"}:
            config["OFFLINE_CHECKPOINT_PATH"] = args.checkpoint
        else:
            raise ValueError(
                "--checkpoint is only used by --mode inference (weights to "
                "evaluate) and --mode online/smoke (warm start); got "
                f"--mode {args.mode}"
            )

    if config.get("SEED") is None:
        config["SEED"] = np.random.randint(2**31)

    return config


# =============================================================================
# W&B artifact resolution
# =============================================================================

_CHECKPOINT_PATH_KEYS = (
    "CHECKPOINT_PATH",
    "OFFLINE_CHECKPOINT_PATH",
    "PPO_CHECKPOINT_PATH",
    "RESUME_CHECKPOINT_PATH",
)


def _resolve_wandb_paths(config: dict[str, Any]) -> None:
    """Download W&B artifacts for any checkpoint path prefixed with ``wandb:``."""
    download_dir = config.get("WANDB_DOWNLOAD_DIR")
    for key in _CHECKPOINT_PATH_KEYS:
        val = config.get(key)
        if val and isinstance(val, str) and val.startswith("wandb:"):
            config[key] = resolve_checkpoint_path(val, download_dir)


# =============================================================================
# Resume resolution
# =============================================================================

def _resolve_resume(config: dict[str, Any]) -> None:
    """Read checkpoint metadata sidecar and fill missing resume params.

    Modifies *config* in-place.  If ``RESUME_CHECKPOINT_PATH`` is not set
    this is a no-op.

    Args:
        config: Upper-cased config dict.

    Raises:
        ValueError: If ``resume_step`` cannot be determined or is invalid.
    """
    resume_path = config.get("RESUME_CHECKPOINT_PATH")
    if not resume_path:
        return

    mode = config["MODE"]
    if mode not in {"offline", "online"}:
        raise ValueError(
            f"--resume is only supported for offline/online modes, "
            f"got '{mode}'"
        )

    # Attempt to read metadata sidecar for auto-population.
    metadata = load_checkpoint_metadata(resume_path)

    if config.get("RESUME_STEP") is None and metadata is not None:
        config["RESUME_STEP"] = metadata["update_step"]
        print(f"Auto-detected resume_step={config['RESUME_STEP']} from checkpoint metadata")

    if config.get("RESUME_WANDB_RUN_ID") is None and metadata is not None:
        wandb_id = metadata.get("wandb_run_id")
        if wandb_id:
            config["RESUME_WANDB_RUN_ID"] = wandb_id
            print(f"Auto-detected resume_wandb_run_id={wandb_id} from checkpoint metadata")

    if config.get("RESUME_STEP") is None:
        raise ValueError(
            "Cannot determine resume_step: no metadata sidecar found at "
            f"'{resume_path}'. Provide --resume-step explicitly."
        )

    resume_step = config["RESUME_STEP"]

    # Resolve NUM_UPDATES and scaled hyperparams via the shared helpers so
    # resume validation matches whatever the runner will compute.  Both are
    # idempotent — the runner re-runs them.
    resolve_num_updates(config, mode)
    resolve_scaled_hyperparams(config, mode)
    num_updates = config["NUM_UPDATES"]

    if resume_step >= num_updates:
        bump_key = (
            "offline_total_timesteps (or offline_num_updates)"
            if mode == "offline"
            else "online_total_timesteps (or online_num_updates)"
        )
        raise ValueError(
            f"resume_step ({resume_step}) >= num_updates ({num_updates}). "
            f"Increase --override {bump_key} to extend training."
        )


# =============================================================================
# Validation
# =============================================================================

def _check_choice(config: dict[str, Any], key: str, choices: list[str]) -> None:
    value = config.get(key)
    if value is not None and value not in choices:
        raise ValueError(f"{key.lower()} must be one of {choices}, got '{value}'")


def validate_config(config: dict[str, Any]) -> None:
    """Validate required config keys for the selected mode.

    Args:
        config: Upper-cased config dict.

    Raises:
        ValueError: If a required key is missing or an enum value is invalid.
    """
    mode = config["MODE"]

    if mode in {"collect", "offline", "online"} and not config.get("PPO_CHECKPOINT_PATH"):
        raise ValueError("--ppo-checkpoint required for this mode")

    if mode == "inference" and not config.get("CHECKPOINT_PATH"):
        raise ValueError("--checkpoint required for inference mode")

    _check_choice(config, "REMASK_STRATEGY", REMASK_STRATEGIES)
    _check_choice(config, "DIFFUSION_SCHEDULE", DIFFUSION_SCHEDULES)
    _check_choice(config, "PPO_MODEL_TYPE", PPO_TYPES)


# =============================================================================
# Compilation cache
# =============================================================================

def configure_compilation_cache(config: dict[str, Any]) -> str | None:
    """Enable JAX's persistent compilation cache when a directory is configured.

    Compiling the online DAgger training graph takes ~52 s on the 4070 Ti and
    the full Craftax graph considerably longer.  Every seed launched as its own
    process, every resumed run and every entry in the RL fine-tuning ablation
    suite currently repeats that compilation from scratch.  The cache is keyed
    on the lowered HLO, so a hit is bit-identical to a miss: this changes no
    numerics.

    Must be called before the first compilation, i.e. before dispatch.

    Args:
        config: Upper-cased config dict.

    Returns:
        The resolved cache directory, or ``None`` when caching is disabled.
    """
    cache_dir = config.get("JAX_COMPILATION_CACHE_DIR")
    if not cache_dir:
        return None

    path = pathlib.Path(str(cache_dir)).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(path))
    # -1 caches every executable regardless of size; the default skips small
    # ones, which here means skipping nothing useful and complicating the
    # hit-rate story.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    print(f"JAX persistent compilation cache: {path}")
    return str(path)


# =============================================================================
# Execution
# =============================================================================

DISPATCH = {
    "collect": run_collect,
    "offline": run_offline_diffusion,
    "online": run_online,
    "inference": run_inference,
    "smoke": run_smoke,
}


def run(config: dict[str, Any]) -> None:
    """Resolve paths, validate, and dispatch to the selected mode.

    Args:
        config: Upper-cased config dict.
    """
    _resolve_wandb_paths(config)
    validate_config(config)
    configure_compilation_cache(config)
    _resolve_resume(config)

    mode = config["MODE"]

    if config.get("JIT", True):
        DISPATCH[mode](config)
    else:
        with jax.disable_jit():
            DISPATCH[mode](config)


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    backend = jax.default_backend()
    print(f"JAX backend: {backend} | Devices: {jax.devices()}")

    if backend != "gpu":
        import warnings
        warnings.warn(f"JAX is using '{backend}', not GPU. uv sync --extra cuda")

    config = build_config()
    run(config)


if __name__ == "__main__":
    main()
