import argparse
import sys

import jax
import numpy as np

from src.planners.offline2 import run_offline_diffusion, _cosine_schedule, _linear_schedule


SCHEDULE_MAP = {
    "linear": _linear_schedule,
    "cosine": _cosine_schedule,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_policy", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument(
        "--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # ── Diffusion-specific ──
    parser.add_argument("--plan_horizon", type=int, default=8)
    parser.add_argument("--num_diff_steps", type=int, default=16)
    parser.add_argument("--plan_embed_dim", type=int, default=32)
    parser.add_argument(
        "--diffusion_schedule", type=str, default="cosine",
        choices=list(SCHEDULE_MAP.keys()),
    )
    # ── PPO checkpoint for data collection ──
    parser.add_argument("--ppo_checkpoint", type=str, required=True)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_offline_diffusion(args)
    else:
        with jax.disable_jit():
            run_offline_diffusion(args)