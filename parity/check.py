"""Verify the working tree against the stored parity references.

Usage:
    uv run python parity/check.py [--fast]

--fast skips the evaluation and short-training fingerprints (minutes of JIT
compilation) and checks only forward passes and checkpoint schemas.

Exit code 0 = all green. Never fix a red check by re-capturing references.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_MODE", "disabled")

from parity.capture import (  # noqa: E402
    DIFFUSION_CHECKPOINTS, NUM_ACTIONS, PPO_CHECKPOINTS, fixed_batch,
    restored_params, run_eval, run_train, schema_text, snapshot_config,
)
from parity.fingerprint_lib import (  # noqa: E402
    PROJECT_ROOT, REFERENCE_DIR, Report, compare_array, compare_scalar,
    flatten_tree, load_json, sha256_dir, sha256_tree,
)


def check_forward(report: Report, name: str, ckpt_dir: str, atol: float) -> None:
    import jax
    from src.planners.model import build_model

    ref = np.load(REFERENCE_DIR / f"forward_{name}.npz")
    meta = load_json(REFERENCE_DIR / f"forward_{name}.json")

    if sha256_dir(PROJECT_ROOT / ckpt_dir) != meta["checkpoint_sha256"]:
        report.add("FAIL", f"forward_{name}/checkpoint-dir",
                   "checkpoint directory changed on disk")
        return

    config = snapshot_config(ckpt_dir)
    num_actions = NUM_ACTIONS[config["ENV_NAME"]]
    variables = restored_params(ckpt_dir)

    checksum = sha256_tree(flatten_tree(variables))
    status = "PASS" if checksum == meta["param_checksum"] else "FAIL"
    report.add(status, f"forward_{name}/param_checksum")

    batch = fixed_batch(meta["obs_dim"], num_actions, int(config["PLAN_HORIZON"]))
    for key in ("obs", "actions", "t"):
        compare_array(report, f"forward_{name}/input_{key}",
                      batch[key], ref[key], atol=0)

    model = build_model(config, num_actions)
    variables_jax = jax.tree.map(lambda x: np.asarray(x), variables)
    logits = np.asarray(
        model.apply(variables_jax, batch["obs"], batch["actions"], batch["t"],
                    deterministic=True)
    )
    compare_array(report, f"forward_{name}/logits", logits, ref["logits"], atol)


def check_schema(report: Report, name: str, ckpt_dir: str) -> None:
    want = (REFERENCE_DIR / f"schema_{name}.txt").read_text()
    got = schema_text(ckpt_dir)
    if got == want:
        report.add("PASS", f"schema_{name}")
    else:
        report.add("FAIL", f"schema_{name}", "key structure changed")


def main() -> None:
    fast = "--fast" in sys.argv
    report = Report()
    tol = load_json(REFERENCE_DIR / "tolerances.json")

    for name, ckpt in DIFFUSION_CHECKPOINTS:
        check_forward(report, name, ckpt, tol["forward_atol"])
        check_schema(report, name, ckpt)
    for name, ckpt in PPO_CHECKPOINTS:
        check_schema(report, name, ckpt)

    if not fast:
        for name, ckpt in DIFFUSION_CHECKPOINTS:
            want = load_json(REFERENCE_DIR / f"eval_{name}.json")["metrics"]
            got = run_eval(ckpt)
            for k in sorted(want):
                compare_scalar(report, f"eval_{name}/{k}",
                               got[k], want[k], tol["eval_atol"])

        want = load_json(REFERENCE_DIR / "train_online.json")
        got = run_train()
        for k in sorted(want["metrics"]):
            compare_array(report, f"train_online/{k}",
                          np.array(got["metrics"][k]),
                          np.array(want["metrics"][k]), tol["train_atol"])
        if tol["train_bit_reproducible"]:
            status = "PASS" if got["param_checksum"] == want["param_checksum"] else "FAIL"
            report.add(status, "train_online/param_checksum")
        else:
            compare_scalar(report, "train_online/param_mean",
                           got["param_stats"]["mean"],
                           want["param_stats"]["mean"], tol["train_atol"])

    sys.exit(report.summary())


if __name__ == "__main__":
    main()
