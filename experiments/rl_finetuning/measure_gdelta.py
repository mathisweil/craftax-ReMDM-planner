#!/usr/bin/env python3
"""Measure the return term g_delta of the return-weighted ELBO decomposition.

Implements the measurement behind Eq. 8 and Table 6 of the paper. Writing the
per-window weight as A_i and the batch mean as Abar, the training gradient
decomposes exactly as

    grad L_RW  =  Abar * ( grad L_BC  +  g_delta ),
    g_delta    =  (1/B) sum_i delta_i grad l_i,     delta_i = A_i/Abar - 1,

so g_delta carries the entire directional contribution of the return. This
script loads a pretrained checkpoint, collects one on-policy batch from it,
and evaluates grad L_BC, g_delta and grad L_RW on that batch at those
parameters under a shared (z_t, t) draw, so the only difference between the
three is the weight vector. It repeats for every weight transform the ablation
suite uses, and reports two references the cosine column needs:

  * the random-direction null, cos ~ N(0, 1/sqrt(D)) for D parameters;
  * cos(grad L_BC, grad L_BC) across two independent noise draws, which is the
    value a direction attains when it *is* the imitation direction.

No training and no optimiser step occur. Runs on CPU in a few minutes.

Usage, from the repository root:

    python experiments/rl_finetuning/measure_gdelta.py \
        --ckpt path/to/pretrained_checkpoint \
        --config path/to/results.json \
        --out gdelta_seed0.json --seed 0

`--config` accepts the `results.json` emitted by `run_ablations.py` (the script
reads its "config" entry) or a plain JSON dict of the same keys.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from experiments.rl_finetuning.ablations.training import (  # noqa: E402
    _compute_advantages,
    build_rollout_fn,
)
from src.diffusion.loss import compute_loss  # noqa: E402
from src.diffusion.schedules import SCHEDULE_MAP  # noqa: E402
from src.planners.env import make_env  # noqa: E402
from src.planners.model import (  # noqa: E402
    _validate_restored_tree,
    abstract_params,
    build_model,
    make_apply_fns,
)


def load_config(path: str) -> dict:
    """Read an ablation config, accepting a results.json or a bare dict."""
    blob = json.load(open(path))
    return dict(blob["config"] if "config" in blob else blob)


def restore_params(net, rng, obs_dim, plan_horizon, ckpt: str):
    """Restore Orbax parameters with an explicit single-device sharding.

    ``model.load_checkpoint`` builds its restore target with ``jax.eval_shape``,
    whose leaves carry no sharding. Orbax >= 0.12 rejects that when the
    checkpoint was written under a different device topology, which is the
    common case for a checkpoint trained on GPU and inspected on CPU. Attaching
    a concrete sharding here is the only difference from the library path.
    """
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    abstract = abstract_params(net, rng, obs_dim, plan_horizon)
    restore_args = jax.tree.map(
        lambda x: ocp.ArrayRestoreArgs(sharding=sharding, dtype=x.dtype), abstract
    )
    with ocp.CheckpointManager(str(Path(ckpt).resolve())) as mgr:
        step = mgr.latest_step()
        restored = mgr.restore(
            step,
            args=ocp.args.PyTreeRestore(
                item={"params": abstract},
                restore_args={"params": restore_args},
                partial_restore=True,
            ),
        )
    params = restored["params"]
    _validate_restored_tree(params, abstract, ckpt)
    return params, step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="Orbax checkpoint directory")
    ap.add_argument("--config", required=True, help="results.json or config JSON")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=None,
                    help="override NUM_ENVS; default is the config value")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override BATCH_SIZE; default is the config value")
    ap.add_argument("--n-draws", type=int, default=8,
                    help="independent (z_t, t) draws to average over")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.num_envs is not None:
        cfg["NUM_ENVS"] = args.num_envs
    if args.batch_size is not None:
        cfg["BATCH_SIZE"] = args.batch_size

    env, env_params = make_env(cfg, cfg["NUM_ENVS"])
    num_actions = env.action_space(env_params).n
    obs_dim = env.observation_space(env_params).shape[0]
    cfg["NUM_ACTIONS"] = num_actions
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[cfg.get("DIFFUSION_SCHEDULE", "cosine")]

    net = build_model(cfg, num_actions)
    apply_eval, apply_train = make_apply_fns(net)

    rng = jax.random.PRNGKey(args.seed)
    rng, key = jax.random.split(rng)
    params, step = restore_params(net, key, obs_dim, cfg["PLAN_HORIZON"], args.ckpt)
    n_params = sum(int(np.prod(x.shape)) for x in jax.tree.leaves(params))
    random_cos_sd = 1.0 / np.sqrt(n_params)
    print(f"checkpoint step {step}, D = {n_params/1e6:.2f}M, "
          f"random-cosine null sd = {random_cos_sd:.2e}", flush=True)

    # ---- one on-policy batch from the pretrained policy ----
    collect = build_rollout_fn(env, env_params, apply_eval, cfg, obs_dim)
    rng, key = jax.random.split(rng)
    obs0, state0 = env.reset(key, env_params)
    done0 = jnp.zeros(cfg["NUM_ENVS"], dtype=bool)
    rng, key = jax.random.split(rng)
    _, _, _, _, f_obs, f_acts, f_valid, f_ret, _ = collect(
        params, state0, obs0, done0, key
    )
    adv, _, _ = _compute_advantages(
        f_ret,
        cfg["RETURN_WEIGHT_FLOOR"],
        cfg["RETURN_WEIGHT_CAP"],
        wins_only=False,
        win_thresh=cfg["WIN_THRESHOLD"],
        use_running_stats=False,
        ema_decay=0.99,
        running_mean=jnp.array(0.0),
        running_std=jnp.array(1.0),
    )

    batch = min(cfg["BATCH_SIZE"], f_obs.shape[0])
    rng, key = jax.random.split(rng)
    idx = jax.random.permutation(key, f_obs.shape[0])[:batch]
    obs_b, act_b, val_b = f_obs[idx], f_acts[idx], f_valid[idx]
    adv_b, ret_b = adv[idx], f_ret[idx]
    print(f"batch {batch} windows, win rate {float((ret_b > cfg['WIN_THRESHOLD']).mean()):.3f}",
          flush=True)

    # ---- the weight transforms the suite uses ----
    eps = cfg["ADV_CLIP_EPS"]
    win = (ret_b > cfg["WIN_THRESHOLD"]).astype(jnp.float32)
    n_win = win.sum()
    variants = {
        "baseline_clipped_ratio": adv_b,
        "advantage_clip": jnp.clip(adv_b, 1.0 - eps, 1.0 + eps),
        "normalized_adv": (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8),
        "bc_wins": win * jnp.where(n_win > 0, batch / jnp.maximum(n_win, 1.0), 0.0),
    }

    flatten = lambda tree: jnp.concatenate([x.ravel() for x in jax.tree.leaves(tree)])

    def gradient(weights, key):
        def loss(p):
            value, _ = compute_loss(
                apply_train, p, key, act_b, obs_b, val_b, num_actions,
                schedule_fn, schedule_deriv_fn,
                sigma_t=cfg.get("TRAIN_SIGMA", 0.0),
                label_smoothing=cfg.get("LABEL_SMOOTHING", 0.0),
                advantages=weights,
            )
            return value
        return flatten(jax.grad(loss)(params))

    acc = {name: {"ratio": [], "cos": [], "cv": None} for name in variants}
    bc_self, residuals = [], []
    for draw in range(args.n_draws):
        rng, key = jax.random.split(rng)
        g_bc = gradient(None, key)
        norm_bc = float(jnp.linalg.norm(g_bc))

        rng, key2 = jax.random.split(rng)
        g_bc2 = gradient(None, key2)  # same objective, independent noise draw
        bc_self.append(
            float(jnp.dot(g_bc, g_bc2) / (norm_bc * jnp.linalg.norm(g_bc2) + 1e-12))
        )

        for name, weights in variants.items():
            wbar = jnp.mean(weights)
            # delta is undefined when the mean weight vanishes, which is what
            # mean-centring does; assumption (A1) of the paper fails there.
            delta = weights / wbar - 1.0 if abs(float(wbar)) > 1e-8 else weights - wbar
            g_delta = gradient(delta, key)
            norm_delta = float(jnp.linalg.norm(g_delta))
            acc[name]["ratio"].append(norm_delta / norm_bc)
            acc[name]["cos"].append(
                float(jnp.dot(g_delta, g_bc) / (norm_delta * norm_bc + 1e-12))
            )
            if acc[name]["cv"] is None:
                acc[name]["cv"] = float(jnp.sqrt(jnp.mean(delta ** 2)))
            if name == "baseline_clipped_ratio":
                # Eq. 4 identity check: grad L_RW == Abar * (grad L_BC + g_delta)
                g_rw = gradient(weights, key)
                residuals.append(float(
                    jnp.linalg.norm(g_rw - wbar * (g_bc + g_delta))
                    / (jnp.linalg.norm(g_rw) + 1e-12)
                ))
        print(f"  draw {draw + 1}/{args.n_draws}", flush=True)

    out = {
        "checkpoint_step": int(step),
        "n_params": int(n_params),
        "random_cos_sd": float(random_cos_sd),
        "batch": int(batch),
        "seed": args.seed,
        "n_draws": args.n_draws,
        "bc_self_cos_mean": float(np.mean(bc_self)),
        "bc_self_cos_std": float(np.std(bc_self)),
        "eq4_residual_max": float(np.max(residuals)),
        "variants": {},
    }
    print(f"\ncos(grad L_BC, grad L_BC) across draws = "
          f"{np.mean(bc_self):.3f} +/- {np.std(bc_self):.3f}   [same-objective reference]")
    print(f"random-direction null: cos ~ N(0, {random_cos_sd:.2e})")
    print(f"Eq. 4 identity, max relative residual = {np.max(residuals):.2e}\n")
    print(f"{'weight transform':26s} {'CV_A':>9s} {'||g_d||/||g_bc||':>19s} {'cos':>16s}")
    for name, rec in acc.items():
        ratio, cos = np.array(rec["ratio"]), np.array(rec["cos"])
        out["variants"][name] = {
            "cv_a": rec["cv"],
            "ratio_mean": float(ratio.mean()), "ratio_std": float(ratio.std()),
            "cos_mean": float(cos.mean()), "cos_std": float(cos.std()),
        }
        if rec["cv"] > 1e3:
            print(f"{name:26s} {'undefined: mean weight ~ 0, (A1) fails':>45s}")
        else:
            print(f"{name:26s} {rec['cv']:9.3f} "
                  f"{ratio.mean():12.3f} +/-{ratio.std():.3f} "
                  f"{cos.mean():+11.3f} +/-{cos.std():.3f}")

    json.dump(out, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
