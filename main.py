"""Training and inference scripts for the ReMDM discrete diffusion planner on Craftax.

Workflow
--------
Step 1  — Train a PPO agent (separate script):
            python ppo_rnn.py  --env_name Craftax-Symbolic-v1 --save_policy
            python ppo_rnd.py  --env_name Craftax-Symbolic-v1 --save_policy

Step 2a — Collect trajectories from the PPO checkpoint to disk:
            python main.py --mode collect \\
                --ppo_checkpoint_path /path/to/ppo_ckpt \\
                --offline_data_path trajectories.npz

Step 2b — (Alternative) Train diffusion model directly from the PPO checkpoint,
          without saving trajectories to disk:
            python main.py --mode offline \\
                --ppo_checkpoint_path /path/to/ppo_ckpt

Step 3  — (Optional) Train offline diffusion model from saved trajectories:
            python main.py --mode offline \\
                --offline_data_path trajectories.npz

Step 4  — Online fine-tuning (optionally warm-starting from offline checkpoint):
            python main.py --mode online
            python main.py --mode online \\
                --offline_checkpoint_path /path/to/offline_ckpt

Step 5  — Evaluate:
            python main.py --mode inference \\
                --checkpoint_path /path/to/ckpt

All defaults are loaded from configs/defaults.yaml.  Any argument can be
overridden on the command line, e.g.:
    python main.py --mode online --lr 1e-4 --num_envs 64
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import jax
import numpy as np
import yaml

from src.models.remdm import (
    ScheduleFn,
    STRATEGY_MAP,
    cosine_schedule,
    linear_schedule,
)

from src.planners import (
    run_collect,
    run_offline,
    run_online,
    run_inference,
)
from src.planners.train_reward import run_train_reward


SCHEDULE_MAP: dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    # --- JAX backend check ---
    backend = jax.default_backend()
    print(f"JAX backend: {backend} | Devices: {jax.devices()}")
    if backend != "gpu":
        import warnings
        warnings.warn(f"JAX is using '{backend}', not GPU. pip install jax[cuda12]")

    # --- Two-pass parsing: config file first, then CLI overrides ---
    _default_cfg = pathlib.Path(__file__).parent / "configs" / "defaults.yaml"
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(_default_cfg))
    pre_args, _ = pre.parse_known_args()

    with open(pre_args.config) as f:
        yaml_cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=str(_default_cfg))

    # -- Operational (CLI only, no YAML equivalent) --
    parser.add_argument("--mode", required=True,
                        choices=["collect", "offline", "online", "inference", "train_reward"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)

    # -- Paths (CLI only, run-specific) --
    parser.add_argument("--ppo_checkpoint_path", type=str, default=None)
    parser.add_argument("--offline_data_path", type=str, default=None)
    parser.add_argument("--offline_checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--reward_load_path", type=str, default=None)
    parser.add_argument("--reward_save_path", type=str, default=None)

    # -- Hyperparameter overrides (default=None, YAML is source of truth) --
    # Environment
    parser.add_argument("--env_name", type=str, default=None)
    # Diffusion
    parser.add_argument("--plan_horizon", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--diffusion_schedule", type=str, choices=["cosine", "linear"], default=None)
    parser.add_argument("--remask_strategy", type=str, choices=list(STRATEGY_MAP.keys()), default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--t_on", type=float, default=None)
    parser.add_argument("--t_off", type=float, default=None)
    parser.add_argument("--top_p", type=int, default=None)
    parser.add_argument("--use_loop", action=argparse.BooleanOptionalAction, default=None)
    # Architecture
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--obs_encoder_layers", type=int, default=None)
    parser.add_argument("--obs_encoder_width", type=int, default=None)
    parser.add_argument("--dropout_rate", type=float, default=None)
    # Optimisation
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    # Training
    parser.add_argument("--num_train_steps", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_updates", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--replan_every", type=int, default=None)
    parser.add_argument("--update_epochs", type=int, default=None)
    parser.add_argument("--num_minibatches", type=int, default=None)
    parser.add_argument("--grpo_group_size", type=int, default=None)
    parser.add_argument("--ppo_init_prob", type=float, default=None)
    parser.add_argument("--ppo_decay_rate", type=float, default=None)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=None)
    parser.add_argument("--layer_size", type=int, default=None)
    parser.add_argument("--collect_num_steps", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--collect_num_envs", type=int, default=None)
    parser.add_argument("--ppo_model_type", type=str, choices=["ppo", "ppo_rnn", "ppo_rnd"], default=None)
    parser.add_argument("--train_sigma", type=float, default=None)
    parser.add_argument("--collect_temperature", type=float, default=None)
    # Inference
    parser.add_argument("--eval_steps", type=float, default=10000)
    # Reward model
    parser.add_argument("--reward_model_type", type=str, choices=["mlp", "rnd", "vision_rnd"], default=None)
    parser.add_argument("--reward_epochs", type=int, default=None)
    parser.add_argument("--reward_lr", type=float, default=None)
    parser.add_argument("--reward_model_path", type=str, default=None)
    # Logging
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_policy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--num_repeats", type=int, default=None)
    parser.add_argument("--ckpt_every_steps", type=lambda x: int(float(x)), default=None)
    parser.add_argument("--ckpt_max_to_keep", type=int, default=None)

    args, rest = parser.parse_known_args(sys.argv[1:])
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    # --- Merge: YAML base, then CLI overrides (None = not set, don't override) ---
    config: dict[str, Any] = {k.upper(): v for k, v in yaml_cfg.items()}
    cli_overrides = {k.upper(): v for k, v in vars(args).items() if v is not None and k != "config"}
    config.update(cli_overrides)

    if config.get("SEED") is None:
        config["SEED"] = np.random.randint(2**31)

    # --- Validate required-by-mode args ---
    mode = config["MODE"]
    if mode == "collect":
        assert config.get("PPO_CHECKPOINT_PATH"), "--ppo_checkpoint_path required for --mode collect"
    elif mode == "offline":
        assert config.get("PPO_CHECKPOINT_PATH") or config.get("OFFLINE_DATA_PATH"), \
            "--mode offline requires --ppo_checkpoint_path or --offline_data_path"
    elif mode == "inference":
        assert config.get("CHECKPOINT_PATH"), "--checkpoint_path required for --mode inference"
    elif mode == "train_reward":
        assert config.get("OFFLINE_DATA_PATH"), "--offline_data_path required for --mode train_reward"

    dispatch = {
        "collect": run_collect,
        "offline": run_offline,
        "online": run_online,
        "inference": run_inference,
        "train_reward": run_train_reward,
    }

    run = lambda: dispatch[mode](config)

    if config["JIT"]:
        run()
    else:
        with jax.disable_jit():
            run()