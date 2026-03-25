"""Unified entrypoint for the ReMDM discrete diffusion planner on Craftax.

Modes
-----
collect       Collect offline trajectories from a trained PPO agent.
offline       Train diffusion model from live PPO rollouts.
online        GRPO fine-tuning with environment interaction.
inference     Evaluate a trained diffusion planner via MPC.

Usage
-----
    python main.py --mode offline --ppo_checkpoint_path /path/to/ppo
    python main.py --mode online --offline_checkpoint_path /path/to/ckpt
    python main.py --mode inference --checkpoint_path /path/to/ckpt
    python main.py --mode collect --ppo_checkpoint_path /path/to/ppo

All defaults are loaded from configs/defaults.yaml.  Any value can be
overridden on the command line (e.g. --lr 1e-4 --num_envs 64).
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import sys; sys.path.append(str(pathlib.Path(__file__).resolve().parent / "Craftax_Baselines"))

import jax
import numpy as np
import yaml

from src.planners.collect import run_collect
from src.planners.train import run_offline_diffusion
from src.planners.online import run_online
from src.planners.inference import run_inference

REMASK_STRATEGIES = ["rescale", "cap", "conf"]
DIFFUSION_SCHEDULES = ["cosine", "linear"]
PPO_TYPES = ["ppo", "ppo_rnn", "ppo_rnd"]


def _build_parser(default_cfg_path: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config file
    p.add_argument("--config", default=default_cfg_path)

    # Mode
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

    # Online GRPO
    p.add_argument("--num_updates", type=lambda x: int(float(x)), default=None)
    p.add_argument("--replan_every", type=int, default=None)
    p.add_argument("--grpo_group_size", type=int, default=None)
    p.add_argument("--ppo_init_prob", type=float, default=None)
    p.add_argument("--ppo_decay_rate", type=float, default=None)

    # Data collection / PPO
    p.add_argument("--collect_num_steps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--collect_num_envs", type=int, default=None)
    p.add_argument("--ppo_model_type", type=str, choices=PPO_TYPES, default=None)
    p.add_argument("--layer_size", type=int, default=None)

    # Inference
    p.add_argument("--eval_steps", type=lambda x: int(float(x)), default=None)
    p.add_argument("--eval_num_envs", type=int, default=None)

    # Checkpointing
    p.add_argument("--save_policy", action=argparse.BooleanOptionalAction, default=None)

    # Logging
    p.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)

    return p


def _validate(mode: str, config: dict[str, Any]) -> None:
    """Check that required keys are present for the selected mode."""
    if mode == "collect":
        assert config.get("PPO_CHECKPOINT_PATH"), "--ppo_checkpoint_path required for --mode collect"
    elif mode == "offline":
        assert config.get("PPO_CHECKPOINT_PATH"), (
            "--mode offline requires --ppo_checkpoint_path"
        )
    elif mode == "inference":
        assert config.get("CHECKPOINT_PATH"), "--checkpoint_path required for --mode inference"


DISPATCH = {
    "collect": run_collect,
    "offline": run_offline_diffusion,
    "online": run_online,
    "inference": run_inference,
}


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    backend = jax.default_backend()
    print(f"JAX backend: {backend} | Devices: {jax.devices()}")
    if backend != "gpu":
        import warnings
        warnings.warn(f"JAX is using '{backend}', not GPU. pip install jax[cuda12]")

    # Two-pass parsing: load YAML first, then override with CLI args
    default_cfg = str(pathlib.Path(__file__).parent / "configs" / "defaults.yaml")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=default_cfg)
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config) as f:
        yaml_cfg = yaml.safe_load(f) or {}

    parser = _build_parser(default_cfg)
    args, rest = parser.parse_known_args()
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    # Merge: YAML base (uppercased) -> CLI overrides (non-None only)
    config: dict[str, Any] = {k.upper(): v for k, v in yaml_cfg.items()}
    cli = {k.upper(): v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(cli)

    if config.get("SEED") is None:
        config["SEED"] = np.random.randint(2**31)

    mode = config["MODE"]
    _validate(mode, config)

    run = lambda: DISPATCH[mode](config)

    if config.get("JIT", True):
        run()
    else:
        with jax.disable_jit():
            run()
