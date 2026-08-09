"""Capture golden-output fingerprints for every trained checkpoint.

Writes parity/reference/*. Run once per approved baseline; parity/check.py
then verifies the working tree against these references. Never re-run this
to make a failing check pass.

Usage:
    uv run python parity/capture.py [--force]

Fingerprints:
    1. forward_<name>.npz/.json  logits on a fixed synthetic batch + param
                                 checksum, for the 4 diffusion checkpoints
    2. eval_<name>.json          32-step 4-env inference metrics, fixed seed
                                 (src loader monkeypatched with the CPU-safe
                                 numpy restore; see fingerprint_lib)
    3. train_online.json         6-update DAgger run (smoke config, random
                                 expert, seed 0): metric arrays + checksums
    4. schema_<name>.txt         orbax key structure, shapes, dtypes
                                 (all 6 checkpoints incl. the PPO experts)
    5. tolerances.json           derived from an observed variability probe
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "Craftax_Baselines"))   # as main.py does
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_MODE", "disabled")

from parity.fingerprint_lib import (  # noqa: E402
    PROJECT_ROOT, REFERENCE_DIR, array_stats, flatten_tree, git_commit,
    latest_step, load_json, numpy_restore, save_json, sha256_array,
    sha256_dir, sha256_file, sha256_tree,
)

DIFFUSION_CHECKPOINTS = [
    ("classic_bc", "checkpoints/offline/Craftax-Classic-Symbolic-v1-OfflineDiffusion-BC-100M"),
    ("craftax_bc", "checkpoints/offline/Craftax-Symbolic-v1-OfflineDiffusion-BC-100M"),
    ("classic_dagger", "checkpoints/online/Craftax-Classic-Symbolic-v1-OnlineDiffusion-DAgger-100M"),
    ("craftax_dagger", "checkpoints/online/Craftax-Symbolic-v1-OnlineDiffusion-DAgger-100M"),
]
PPO_CHECKPOINTS = [
    ("classic_ppo", "checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M"),
    ("craftax_ppo", "checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M"),
]

NUM_ACTIONS = {"Craftax-Classic-Symbolic-v1": 17, "Craftax-Symbolic-v1": 43}
EVAL_SEED = 1234
EVAL_STEPS = 32
EVAL_NUM_ENVS = 4
TRAIN_SEED = 0


def snapshot_config(ckpt_dir: str) -> dict:
    """Upper-cased config snapshot from the checkpoint's metadata sidecar."""
    meta = load_json(PROJECT_ROOT / ckpt_dir / "resume_metadata.json")
    return {str(k).upper(): v for k, v in meta["config_snapshot"].items()}


def restored_params(ckpt_dir: str) -> dict:
    """TrainState.params ({'params': {...}}) as numpy from a checkpoint."""
    return numpy_restore(PROJECT_ROOT / ckpt_dir)["params"]


def fixed_batch(obs_dim: int, num_actions: int, plan_horizon: int) -> dict:
    rs = np.random.RandomState(0)
    obs = rs.uniform(-1.0, 1.0, size=(4, obs_dim)).astype(np.float32)
    # vocab includes the MASK token (= num_actions)
    actions = rs.randint(0, num_actions + 1, size=(4, plan_horizon)).astype(np.int32)
    t = np.array([0.1, 0.4, 0.7, 0.95], dtype=np.float32)
    return {"obs": obs, "actions": actions, "t": t}


def capture_forward(name: str, ckpt_dir: str) -> None:
    import jax
    from src.planners.model import build_model

    config = snapshot_config(ckpt_dir)
    num_actions = NUM_ACTIONS[config["ENV_NAME"]]
    variables = restored_params(ckpt_dir)
    params_flat = flatten_tree(variables)
    obs_dim = variables["params"]["Dense_0"]["kernel"].shape[0]

    model = build_model(config, num_actions)
    batch = fixed_batch(obs_dim, num_actions, int(config["PLAN_HORIZON"]))
    variables_jax = jax.tree.map(lambda x: np.asarray(x), variables)
    logits = np.asarray(
        model.apply(
            variables_jax, batch["obs"], batch["actions"], batch["t"],
            deterministic=True,
        )
    )
    assert logits.shape == (4, int(config["PLAN_HORIZON"]), num_actions), logits.shape

    np.savez(REFERENCE_DIR / f"forward_{name}.npz", **batch, logits=logits)
    save_json(REFERENCE_DIR / f"forward_{name}.json", {
        "git_commit": git_commit(),
        "checkpoint": ckpt_dir,
        "checkpoint_sha256": sha256_dir(PROJECT_ROOT / ckpt_dir),
        "obs_dim": int(obs_dim),
        "num_actions": num_actions,
        "n_param_elements": int(sum(v.size for v in params_flat.values())),
        "param_checksum": sha256_tree(params_flat),
        "logits": {**array_stats(logits), "sha256": sha256_array(logits)},
    })
    print(f"forward_{name}: obs_dim={obs_dim}, "
          f"{sum(v.size for v in params_flat.values()):,} param elements")


def run_eval(ckpt_dir: str) -> dict:
    """Run the real inference path with the CPU-safe loader patched in."""
    import jax
    import src.planners.inference as inference_mod
    from src.planners.inference import run_inference

    config = snapshot_config(ckpt_dir)
    config.update({
        "MODE": "inference",
        "SEED": EVAL_SEED,
        "USE_WANDB": False,
        "EVAL_STEPS": EVAL_STEPS,
        "EVAL_NUM_ENVS": EVAL_NUM_ENVS,
        "CHECKPOINT_PATH": str(PROJECT_ROOT / ckpt_dir),
        "JIT": True,
    })
    tmp = Path(tempfile.mkdtemp(prefix="parity-eval-"))
    config["INFERENCE_OUTPUT"] = str(tmp / "eval.json")

    original = inference_mod.load_checkpoint

    def cpu_safe_load(model, rng, obs_dim, plan_horizon, path):
        variables = numpy_restore(path)["params"]
        return jax.tree.map(lambda x: np.asarray(x), variables)

    inference_mod.load_checkpoint = cpu_safe_load
    try:
        run_inference(config)
    finally:
        inference_mod.load_checkpoint = original

    results = json.loads((tmp / "eval.json").read_text())
    shutil.rmtree(tmp, ignore_errors=True)
    return {
        k: v for k, v in results.items()
        if isinstance(v, (int, float)) and k not in ("seed",)
    }


def run_train() -> dict:
    """Short DAgger run: smoke config, random expert, fixed seed."""
    import jax
    import yaml
    from craftax.craftax_env import make_craftax_env_from_name
    from src.planners.online import run_online
    from src.planners.smoke import _write_random_expert

    with open(PROJECT_ROOT / "configs" / "defaults.yaml") as f:
        cfg = yaml.safe_load(f)
    with open(PROJECT_ROOT / "configs" / "smoke.yaml") as f:
        cfg.update(yaml.safe_load(f))
    config = {k.upper(): v for k, v in cfg.items()}
    config.update({
        "MODE": "online", "SEED": TRAIN_SEED,
        "USE_WANDB": False, "SAVE_POLICY": False, "JIT": True,
    })

    env = make_craftax_env_from_name(config["ENV_NAME"], auto_reset=True)
    env_params = env.default_params
    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape

    tmp_expert = tempfile.mkdtemp(prefix="parity-expert-")
    try:
        _write_random_expert(config, num_actions, obs_shape, tmp_expert)
        config["PPO_CHECKPOINT_PATH"] = tmp_expert
        out = run_online(config)
    finally:
        shutil.rmtree(tmp_expert, ignore_errors=True)

    metrics = {
        k: np.asarray(v).astype(np.float64).ravel()
        for k, v in out["metrics"].items()
        if np.asarray(v).dtype.kind in "fiu"
    }
    params = flatten_tree(
        jax.tree.map(lambda x: np.asarray(x)[0], out["runner_state"].train_state.params)
    )
    return {
        "metrics": {k: [float(x) for x in v] for k, v in metrics.items()},
        "param_checksum": sha256_tree(params),
        "param_stats": array_stats(
            np.concatenate([p.ravel() for p in params.values()])
        ),
    }


def schema_text(ckpt_dir: str) -> str:
    import orbax.checkpoint as ocp

    step = latest_step(PROJECT_ROOT / ckpt_dir)
    step_dir = PROJECT_ROOT / ckpt_dir / str(step) / "default"
    meta = ocp.PyTreeCheckpointer().metadata(str(step_dir))
    flat = flatten_tree_meta(meta.item_metadata.tree)
    lines = [f"latest_step: {step}"]
    for f in sorted(p.name for p in (PROJECT_ROOT / ckpt_dir).iterdir() if p.is_file()):
        lines.append(f"sidecar: {f}")
    lines += [f"{k}: {v}" for k, v in sorted(flat.items())]
    return "\n".join(lines) + "\n"


def flatten_tree_meta(tree, prefix: str = "") -> dict[str, str]:
    flat: dict[str, str] = {}
    if isinstance(tree, dict):
        for k in sorted(tree, key=str):
            flat.update(flatten_tree_meta(tree[k], f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            flat.update(flatten_tree_meta(v, f"{prefix}.{i}" if prefix else str(i)))
    elif tree is None:
        flat[prefix] = "None"
    else:
        shape = getattr(tree, "shape", None)
        dtype = getattr(tree, "dtype", None)
        flat[prefix] = f"{tuple(shape) if shape is not None else '?'} {dtype}"
    return flat


def capture_schema(name: str, ckpt_dir: str) -> None:
    text = schema_text(ckpt_dir)
    (REFERENCE_DIR / f"schema_{name}.txt").write_text(text)
    print(f"schema_{name}: {len(text.splitlines())} entries")


def main() -> None:
    force = "--force" in sys.argv
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    marker = REFERENCE_DIR / "MANIFEST.json"
    if marker.exists() and not force:
        sys.exit("References already exist; pass --force to overwrite.")

    for name, ckpt in DIFFUSION_CHECKPOINTS:
        capture_forward(name, ckpt)
        capture_schema(name, ckpt)
    for name, ckpt in PPO_CHECKPOINTS:
        capture_schema(name, ckpt)

    evals: dict[str, dict] = {}
    for name, ckpt in DIFFUSION_CHECKPOINTS:
        evals[name] = run_eval(ckpt)
        print(f"eval_{name}: {evals[name]}")
    probe = run_eval(DIFFUSION_CHECKPOINTS[2][1])   # classic_dagger rerun
    eval_delta = max(
        abs(float(evals["classic_dagger"][k]) - float(probe[k]))
        for k in evals["classic_dagger"]
    )
    print(f"eval variability probe (classic_dagger rerun): max delta={eval_delta:.3g}")
    for name in evals:
        save_json(REFERENCE_DIR / f"eval_{name}.json", {
            "git_commit": git_commit(),
            "eval_steps": EVAL_STEPS, "eval_num_envs": EVAL_NUM_ENVS,
            "seed": EVAL_SEED, "metrics": evals[name],
        })

    t1 = run_train()
    t2 = run_train()
    train_delta = max(
        (max(abs(a - b) for a, b in zip(t1["metrics"][k], t2["metrics"][k]))
         if t1["metrics"][k] else 0.0)
        for k in t1["metrics"]
    )
    bit_repro = t1["param_checksum"] == t2["param_checksum"]
    print(f"train variability probe: max metric delta={train_delta:.3g}, "
          f"param checksums identical={bit_repro}")
    save_json(REFERENCE_DIR / "train_online.json", {
        "git_commit": git_commit(),
        "seed": TRAIN_SEED,
        **t1,
    })

    tolerances = {
        "forward_atol": 1e-6,
        "eval_atol": max(4 * eval_delta, 1e-9),
        "train_atol": max(4 * train_delta, 1e-9),
        "train_bit_reproducible": bool(bit_repro),
        "eval_observed_delta": eval_delta,
        "train_observed_delta": train_delta,
    }
    save_json(REFERENCE_DIR / "tolerances.json", tolerances)

    save_json(marker, {
        "git_commit": git_commit(),
        "checkpoints": {
            n: sha256_dir(PROJECT_ROOT / c)
            for n, c in DIFFUSION_CHECKPOINTS + PPO_CHECKPOINTS
        },
    })
    print("capture complete")


if __name__ == "__main__":
    main()
