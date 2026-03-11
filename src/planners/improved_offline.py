import argparse
import os
import sys

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time

import orbax.checkpoint as ocp

import wandb
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import functools

from Craftax_Baselines.wrappers import (
    LogWrapper,
    OptimisticResetVecEnvWrapper,
    BatchEnvWrapper,
    AutoResetEnvWrapper,
)
from Craftax_Baselines.logz.batch_logging import create_log_dict, batch_log

from craftax.craftax_env import make_craftax_env_from_name

from src.planners.offline import make_train_offline
from .common import SCHEDULE_MAP, _make_grad_step
from .utils import (
    _build_model,
    _init_model_params,
    _create_train_state,
    _load_ppo_checkpoint,
    _make_env_stack,
    _valid_window_mask,
    _make_apply_fns,
    _make_periodic_ckpt_manager,
    _resolve_ckpt_dir,
)

def run_offline(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-OFFLINE-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M",
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    num_checkpoints = config.get("NUM_CHECKPOINTS", 10)
    total_updates = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    updates_per_checkpoint = total_updates // num_checkpoints
    config["NUM_UPDATES"] = updates_per_checkpoint

    ckpt_manager = None
    if config["SAVE_POLICY"] and config["USE_WANDB"]:
        path = os.path.join(wandb.run.dir, "policies")
        ckpt_manager = ocp.CheckpointManager(
            path,
            options=ocp.CheckpointManagerOptions(max_to_keep=3),
        )

    train_jit = jax.jit(make_train_offline(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()

    out = train_vmap(rngs)
    for seg in range(1, num_checkpoints):
        if ckpt_manager is not None:
            step = seg * updates_per_checkpoint
            train_states = out["runner_state"][0]
            train_state = jax.tree.map(lambda x: x[0], train_states)

            ckpt_manager.save(
                step,
                args=ocp.args.StandardSave(train_state),
            )
            ckpt_manager.wait_until_finished()
            print(f"Checkpoint saved at update {step}/{total_updates}")

        out = train_vmap(out["runner_state"])

    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS:", config["TOTAL_TIMESTEPS"] / (t1 - t0))

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)

            path = os.path.join(wandb.run.dir, dir_name)
            with ocp.CheckpointManager(
                    path,
                    options=ocp.CheckpointManagerOptions(max_to_keep=1),
            ) as checkpoint_manager:
                checkpoint_manager.save(
                    config["TOTAL_TIMESTEPS"],
                    args=ocp.args.StandardSave(train_state),
                )
                checkpoint_manager.wait_until_finished()

            print(f"saved runner state to {path}")

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1024,
    )
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument(
        "--anneal_lr", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=np.random.randint(2**31))
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

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)

    if args.jit:
        run_offline(args)
    else:
        with jax.disable_jit():
            run_offline(args)
