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
from typing import Any, Dict

import jax
import numpy as np
import yaml

from src.models.remdm import (
    ScheduleFn,
    STRATEGY_MAP,
    cosine_schedule,
    linear_schedule,
)

from src.planners.planners import (
    run_collect,
    run_offline,
    run_online,
    run_inference,
)

SCHEDULE_MAP: Dict[str, ScheduleFn] = {
    "cosine": cosine_schedule,
    "linear": linear_schedule,
}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend} | Devices: {devices}")
    if backend != "gpu":
        import warnings
        warnings.warn(
            f"JAX is using '{backend}', not GPU. "
            "Install jaxlib with CUDA support: pip install jax[cuda12]"
        )

    _src_dir = pathlib.Path(__file__).parent
    _default_cfg_path = _src_dir / "configs" / "defaults.yaml"

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--config", type=str, default=str(_default_cfg_path))
    _pre_args, _ = _pre.parse_known_args()

    with open(_pre_args.config) as _f:
        _yaml_defaults = yaml.safe_load(_f)

    parser = argparse.ArgumentParser(
        description="ReMDM discrete diffusion planner for Craftax",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=str(_default_cfg_path),
        help="Path to a YAML config file (overridden by any explicit CLI flag).",
    )

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "offline", "online", "inference"],
        required=True,
        help=(
            "collect: save PPO trajectories to disk. "
            "offline: train diffusion model (from .npz or live from PPO agent). "
            "online: fine-tune with diffusion self-rollout. "
            "inference: evaluate."
        ),
    )

    # Environment
    parser.add_argument("--env_name", type=str)

    # Diffusion model
    parser.add_argument("--plan_horizon", type=int)
    parser.add_argument("--diffusion_steps", type=int)
    parser.add_argument("--diffusion_schedule", type=str, choices=["cosine", "linear"])
    parser.add_argument("--remask_strategy", type=str, choices=list(STRATEGY_MAP.keys()))
    parser.add_argument("--eta", type=float)
    parser.add_argument("--t_on", type=float)
    parser.add_argument("--t_off", type=float)

    # Transformer architecture
    parser.add_argument("--d_model", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--d_ff", type=int)
    parser.add_argument("--obs_encoder_layers", type=int)
    parser.add_argument("--obs_encoder_width", type=int)
    parser.add_argument("--dropout_rate", type=float)

    # Optimisation
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--batch_size", type=int)

    # Offline training
    parser.add_argument(
        "--offline_data_path", type=str,
        help="Path to .npz trajectories (--mode offline file-based).",
    )
    parser.add_argument("--num_train_steps", type=lambda x: int(float(x)))

    # Online training
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--num_updates", type=lambda x: int(float(x)))
    parser.add_argument("--replan_every", type=int)
    parser.add_argument("--update_epochs", type=int)
    parser.add_argument("--num_minibatches", type=int)
    parser.add_argument(
        "--offline_checkpoint_path", type=str,
        help="Path to an offline-trained checkpoint to warm-start online training.",
    )
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction)
    parser.add_argument("--optimistic_reset_ratio", type=int)

    # Inference
    parser.add_argument(
        "--checkpoint_path", type=str,
        help="Path to a trained model checkpoint for inference.",
    )
    parser.add_argument("--eval_steps", type=int)

    # Data collection / agent-based offline training
    parser.add_argument(
        "--ppo_checkpoint_path", type=str,
        help=(
            "Path to a pre-trained PPO (ActorCritic) checkpoint. "
            "Used by --mode collect (save trajectories to disk) and "
            "--mode offline (train directly from agent without saving)."
        ),
    )
    parser.add_argument(
        "--ppo_model_type", type=str,
        choices=["ppo", "ppo_rnn", "ppo_rnd"],
        help=(
            "Override the PPO model architecture used for checkpoint loading. "
            "When null (default), the architecture is auto-detected from the "
            "checkpoint directory contents. Valid values: ppo, ppo_rnn, ppo_rnd."
        ),
    )
    parser.add_argument(
        "--collect_num_steps", type=lambda x: int(float(x)),
        help="Total env steps to collect (--mode collect).",
    )
    parser.add_argument("--collect_num_envs", type=int)
    parser.add_argument(
        "--layer_size", type=int,
        help="Hidden layer width of the ActorCritic PPO network.",
    )

    # W&B / logging
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--save_policy", action=argparse.BooleanOptionalAction)
    parser.add_argument("--num_repeats", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction)

    parser.set_defaults(**_yaml_defaults)

    args, rest = parser.parse_known_args(sys.argv[1:])
    if rest:
        raise ValueError(f"Unknown arguments: {rest}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    config: Dict[str, Any] = {k.upper(): v for k, v in vars(args).items()}
    config.pop("CONFIG", None)

    def _run() -> None:
        if config["MODE"] == "collect":
            assert config.get("PPO_CHECKPOINT_PATH"), (
                "--ppo_checkpoint_path is required for --mode collect.\n"
                "Train a PPO agent first:  python ppo_rnn.py  or  python ppo_rnd.py"
            )
            run_collect(config)
        elif config["MODE"] == "offline":
            assert config.get("PPO_CHECKPOINT_PATH") or config.get("OFFLINE_DATA_PATH"), (
                "--mode offline requires either --ppo_checkpoint_path "
                "(live agent collection) or --offline_data_path (load from .npz)."
            )
            run_offline(config)
        elif config["MODE"] == "online":
            run_online(config)
        elif config["MODE"] == "inference":
            assert config.get("CHECKPOINT_PATH"), (
                "--checkpoint_path is required for --mode inference."
            )
            run_inference(config)

    if args.jit:
        _run()
    else:
        with jax.disable_jit():
            _run()