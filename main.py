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

REMASK_STRATEGIES = ["rescale", "cap", "conf"]
DIFFUSION_SCHEDULES = ["cosine", "linear"]
PPO_TYPES = ["ppo", "ppo_rnn", "ppo_rnd"]


# =============================================================================
# Parser
# =============================================================================

def _build_parser(default_cfg_path: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--config", default=default_cfg_path)

    p.add_argument(
        "--mode", required=True,
        choices=["collect", "offline", "online", "inference"],
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)

    # Paths
    p.add_argument("--ppo_checkpoint_path", type=str, default=None)
    p.add_argument("--offline_data_path", type=str, default=None)
    p.add_argument("--offline_checkpoint_path", type=str, default=None)
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--checkpoint_dir", type=str, default=None)
    p.add_argument("--inference_output", type=str, default=None,
                   help="Optional JSON path for machine-readable inference results (C-006).")

    # Environment
    p.add_argument("--env_name", type=str, default=None)
    p.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--optimistic_reset_ratio", type=int, default=None)

    # Architecture
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--n_heads", type=int, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--obs_encoder_layers", type=int, default=None)
    p.add_argument("--obs_encoder_width", type=int, default=None)
    p.add_argument("--dropout_rate", type=float, default=None)

    # Diffusion
    p.add_argument("--plan_horizon", type=int, default=None)
    p.add_argument("--diffusion_schedule", type=str, choices=DIFFUSION_SCHEDULES, default=None)
    p.add_argument("--diffusion_steps", type=int, default=None)
    p.add_argument("--diffusion_steps_eval", type=int, default=None)
    p.add_argument("--train_sigma", type=float, default=None)
    p.add_argument("--label_smoothing", type=float, default=None)
    p.add_argument("--remask_strategy", type=str, choices=REMASK_STRATEGIES, default=None)
    p.add_argument("--eta", type=float, default=None)
    p.add_argument("--use_loop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--t_on", type=float, default=None)
    p.add_argument("--t_off", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top_p", type=float, default=None)

    # Optimisation
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_grad_norm", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)

    # Offline training
    p.add_argument("--offline_total_timesteps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--offline_num_updates", type=int, default=None)
    p.add_argument("--num_envs", type=int, default=None)
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--num_minibatches", type=int, default=None)
    p.add_argument("--update_epochs", type=int, default=None)
    p.add_argument("--num_repeats", type=int, default=None)
    p.add_argument("--collect_temperature", type=float, default=None)
    p.add_argument("--val_interval", type=int, default=None)
    p.add_argument("--val_diffusion_steps", type=int, default=None)
    p.add_argument("--val_replan_every", type=int, default=None)
    p.add_argument("--val_steps", type=int, default=None)
    p.add_argument("--return_weight_cap", type=float, default=None)
    p.add_argument("--lr_warmup_steps", type=int, default=None)
    p.add_argument("--lr_warmup_frames", type=lambda x: int(float(x)), default=None)
    p.add_argument("--val_interval_frames", type=lambda x: int(float(x)), default=None)

    # Online DAgger
    p.add_argument("--online_num_updates", type=lambda x: int(float(x)), default=None)
    p.add_argument("--online_total_timesteps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--dagger_beta_init", type=float, default=None)
    p.add_argument("--dagger_beta_decay", type=float, default=None)
    p.add_argument("--dagger_beta_final", type=float, default=None)
    p.add_argument("--dagger_buffer_max", type=int, default=None)
    p.add_argument("--dagger_buffer_cycles", type=float, default=None)

    # Data collection
    p.add_argument("--collect_num_steps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--collect_num_envs", type=int, default=None)
    p.add_argument("--ppo_model_type", type=str, choices=PPO_TYPES, default=None)
    p.add_argument("--layer_size", type=int, default=None)

    # Inference
    p.add_argument("--eval_steps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--eval_num_envs", type=int, default=None)

    # Checkpointing
    p.add_argument("--save_policy", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--checkpoint_interval", type=int, default=None)
    p.add_argument("--max_checkpoints", type=int, default=None)

    # Resume
    p.add_argument("--resume_checkpoint_path", type=str, default=None)
    p.add_argument("--resume_wandb_run_id", type=str, default=None)
    p.add_argument("--resume_step", type=int, default=None)

    # Logging
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_download_dir", type=str, default=None)

    return p


# =============================================================================
# Config
# =============================================================================

def build_config() -> dict[str, Any]:
    default_cfg = str(pathlib.Path(__file__).parent / "configs" / "defaults.yaml")

    # Pre-parse config path
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_cfg)
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config) as f:
        yaml_cfg = yaml.safe_load(f) or {}

    parser = _build_parser(default_cfg)
    args, rest = parser.parse_known_args()
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    config: dict[str, Any] = {k.upper(): v for k, v in yaml_cfg.items()}
    cli = {k.upper(): v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(cli)

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
            f"--resume_checkpoint_path is only supported for offline/online "
            f"modes, got '{mode}'"
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
            f"'{resume_path}'. Provide --resume_step explicitly."
        )

    resume_step = config["RESUME_STEP"]

    # Resolve NUM_UPDATES and scaled hyperparams via the shared helpers so
    # resume validation matches whatever the runner will compute.  Both are
    # idempotent — the runner re-runs them.
    resolve_num_updates(config, mode)
    resolve_scaled_hyperparams(config, mode)
    num_updates = config["NUM_UPDATES"]

    if resume_step >= num_updates:
        bump_flag = (
            "--offline_total_timesteps (or --offline_num_updates)"
            if mode == "offline"
            else "--online_total_timesteps (or --online_num_updates)"
        )
        raise ValueError(
            f"resume_step ({resume_step}) >= num_updates ({num_updates}). "
            f"Increase {bump_flag} to extend training."
        )


# =============================================================================
# Validation
# =============================================================================

def validate_config(config: dict[str, Any]) -> None:
    """Validate required config keys for the selected mode.

    Args:
        config: Upper-cased config dict.

    Raises:
        ValueError: If a required key is missing.
    """
    mode = config["MODE"]

    if mode in {"collect", "offline", "online"} and not config.get("PPO_CHECKPOINT_PATH"):
        raise ValueError("--ppo_checkpoint_path required for this mode")

    if mode == "inference" and not config.get("CHECKPOINT_PATH"):
        raise ValueError("--checkpoint_path required for inference mode")


# =============================================================================
# Execution
# =============================================================================

DISPATCH = {
    "collect": run_collect,
    "offline": run_offline_diffusion,
    "online": run_online,
    "inference": run_inference,
}


def run(config: dict[str, Any]) -> None:
    """Resolve paths, validate, and dispatch to the selected mode.

    Args:
        config: Upper-cased config dict.
    """
    _resolve_wandb_paths(config)
    validate_config(config)
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