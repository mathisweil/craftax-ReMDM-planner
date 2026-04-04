from __future__ import annotations

import argparse
import pathlib
from typing import Any

import sys; sys.path.append(str(pathlib.Path(__file__).resolve().parent / "Craftax_Baselines"))

import jax
import numpy as np
import yaml

from src.planners.collect import run_collect
from src.planners.model import resolve_checkpoint_path
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
    p.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=None)
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

    # Online DAgger
    p.add_argument("--num_updates", type=lambda x: int(float(x)), default=None)
    p.add_argument("--replan_every", type=int, default=None)
    p.add_argument("--dagger_beta_init", type=float, default=None)
    p.add_argument("--dagger_beta_decay", type=float, default=None)
    p.add_argument("--dagger_buffer_max", type=int, default=None)

    # Data collection
    p.add_argument("--collect_num_steps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--collect_num_envs", type=int, default=None)
    p.add_argument("--ppo_model_type", type=str, choices=PPO_TYPES, default=None)
    p.add_argument("--layer_size", type=int, default=None)

    # Inference
    p.add_argument("--eval_steps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--eval_num_envs", type=int, default=None)

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
)


def _resolve_wandb_paths(config: dict[str, Any]) -> None:
    """Download W&B artifacts for any checkpoint path prefixed with ``wandb:``."""
    download_dir = config.get("WANDB_DOWNLOAD_DIR")
    for key in _CHECKPOINT_PATH_KEYS:
        val = config.get(key)
        if val and isinstance(val, str) and val.startswith("wandb:"):
            config[key] = resolve_checkpoint_path(val, download_dir)


# =============================================================================
# Validation
# =============================================================================

def validate_config(config: dict[str, Any]) -> None:
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
    _resolve_wandb_paths(config)
    validate_config(config)

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