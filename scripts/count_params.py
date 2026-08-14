#!/usr/bin/env python3
"""Exact parameter counts for committed model configs.

Instantiates each config through the repo's own build path (build_model +
init_params) and prints name, architecture fields and the exact parameter
count. With --verify-checkpoint, additionally restores an Orbax checkpoint
via load_checkpoint and asserts its parameter count equals the config's.

Usage:
  uv run python scripts/count_params.py
  uv run python scripts/count_params.py --verify-checkpoint \
      checkpoints/offline/Craftax-Classic-Symbolic-v1-OfflineDiffusion-BC-100M/100000000 \
      --config configs/final_classic_ucl.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jax
import jax.numpy as jnp  # noqa: F401  (kept for parity with repo imports)
import yaml

from craftax.craftax_env import make_craftax_env_from_name
from src.planners.model import build_model, init_params, load_checkpoint

DEFAULT_CONFIGS = [
    "configs/classic_exp_d_100K_model.yaml",
    "configs/classic_exp_d_250K_model.yaml",
    "configs/classic_exp_d_850K_model.yaml",
    "configs/classic_exp_d_3M_model.yaml",
    "configs/craftax_exp_d_500K_model.yaml",
    "configs/craftax_exp_d_1M_model.yaml",
    "configs/craftax_exp_d_3M_model.yaml",
    "configs/craftax_exp_d_7M_model.yaml",
    "configs/final_classic_ucl.yaml",
    "experiments/rl_finetuning/configs/ablations_final_craftax_ucl.yaml",
]

_ENV_CACHE: dict[str, tuple] = {}


def _env_dims(env_name: str) -> tuple[int, int]:
    if env_name not in _ENV_CACHE:
        env = make_craftax_env_from_name(env_name, auto_reset=True)
        env_params = env.default_params
        _ENV_CACHE[env_name] = (
            env.action_space(env_params).n,
            env.observation_space(env_params).shape[0],
        )
    return _ENV_CACHE[env_name]


def count_config(cfg_path: str) -> dict:
    raw = yaml.safe_load(open(cfg_path))
    cfg = {k.upper(): v for k, v in raw.items()}
    env_name = cfg.get("ENV_NAME", "Craftax-Classic-Symbolic-v1")
    num_actions, obs_dim = _env_dims(env_name)
    model = build_model(cfg, num_actions)
    params = init_params(
        model, jax.random.PRNGKey(0), obs_dim, int(cfg.get("PLAN_HORIZON", 32))
    )
    n = int(sum(int(x.size) for x in jax.tree_util.tree_leaves(params)))
    return {
        "config": cfg_path,
        "env_name": env_name,
        "d_model": cfg.get("D_MODEL"),
        "n_heads": cfg.get("N_HEADS"),
        "n_layers": cfg.get("N_LAYERS"),
        "d_ff": cfg.get("D_FF"),
        "obs_dim": obs_dim,
        "params": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    ap.add_argument("--verify-checkpoint", type=str, default=None,
                    help="Orbax checkpoint dir to restore and count against the config.")
    ap.add_argument("--config", type=str, default=None,
                    help="Config matching --verify-checkpoint.")
    a = ap.parse_args()

    print(f"{'config':52s} {'arch (d/h/l/ff)':>18s} {'obs':>6s} {'params':>12s}")
    rows = []
    for c in a.configs:
        r = count_config(c)
        rows.append(r)
        arch = f"{r['d_model']}/{r['n_heads']}/{r['n_layers']}/{r['d_ff']}"
        print(f"{r['config']:52s} {arch:>18s} {r['obs_dim']:>6d} {r['params']:>12,d}")

    if a.verify_checkpoint:
        if not a.config:
            raise SystemExit("--config is required with --verify-checkpoint")
        r = count_config(a.config)
        raw = yaml.safe_load(open(a.config))
        cfg = {k.upper(): v for k, v in raw.items()}
        num_actions, obs_dim = _env_dims(cfg.get("ENV_NAME", "Craftax-Classic-Symbolic-v1"))
        model = build_model(cfg, num_actions)
        restored = load_checkpoint(
            model, jax.random.PRNGKey(0), obs_dim,
            int(cfg.get("PLAN_HORIZON", 32)), a.verify_checkpoint,
        )
        n_ckpt = int(sum(int(x.size) for x in jax.tree_util.tree_leaves(restored)))
        print(f"\ncheckpoint {a.verify_checkpoint}: {n_ckpt:,d} params")
        print(f"config     {a.config}: {r['params']:,d} params")
        if n_ckpt == r["params"]:
            print("PARAM COUNT GATE: PASS (checkpoint parameter count equals config count)")
        else:
            print("PARAM COUNT GATE: FAIL (counts differ)")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
