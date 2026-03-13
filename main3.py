if __name__ == "__main__":
    import argparse
    import sys
    import numpy as np
    import jax

    parser = argparse.ArgumentParser()

    # Environment & System
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)

    # Logging
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--wandb_entity", type=str, required=True)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_policy", action="store_true")

    # Offline Training Parameters
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--update_epochs", type=int, default=4)

    # PPO Adapter / Data Collection
    parser.add_argument("--ppo_checkpoint_path", type=str, required=True)
    parser.add_argument("--ppo_model_type", type=str, choices=["ppo", "ppo_rnd", "ppo_rnn"], default="ppo")
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--collect_temperature", type=float, default=2.0)

    # Diffusion Architecture
    parser.add_argument("--plan_horizon", type=int, default=16)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--obs_encoder_layers", type=int, default=2)
    parser.add_argument("--obs_encoder_width", type=int, default=256)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--diffusion_schedule", type=str, default="linear")
    parser.add_argument("--train_sigma", type=float, default=0.0)

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2 ** 31)

    if args.jit:
        run_offline_diffusion(args)
    else:
        with jax.disable_jit():
            run_offline_diffusion(args)