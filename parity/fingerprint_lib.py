"""Shared helpers for the parity regression harness.

Golden-output fingerprints protect the trained checkpoints: any code change
that alters what the repo computes must show up as a diff against the stored
references. See parity/capture.py (writes references) and parity/check.py
(compares against them).

The released checkpoints were saved on GPU; orbax refuses to restore them
through the src/ loader on a CPU-only machine (sharding topology mismatch).
``numpy_restore`` below restores them as host numpy arrays instead, which
needs no sharding and works everywhere. It lives here, not in src/, so the
production loader stays untouched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

PARITY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PARITY_DIR.parent
REFERENCE_DIR = PARITY_DIR / "reference"


def git_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True,
    )
    return out.stdout.strip() or "unknown"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | Path) -> str:
    """Order-stable checksum over every file in a checkpoint directory."""
    root = Path(path)
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(root)).encode())
        h.update(sha256_file(f).encode())
    return h.hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_tree(named_arrays: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for name in sorted(named_arrays):
        h.update(name.encode())
        h.update(sha256_array(named_arrays[name]).encode())
    return h.hexdigest()


def array_stats(arr: np.ndarray) -> dict:
    a = np.asarray(arr, dtype=np.float64)
    return {
        "shape": list(np.shape(arr)),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def save_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_json(path: str | Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def latest_step(ckpt_dir: str | Path) -> int:
    import orbax.checkpoint as ocp
    with ocp.CheckpointManager(str(Path(ckpt_dir).resolve())) as mgr:
        step = mgr.latest_step()
    if step is None:
        raise FileNotFoundError(f"No checkpoint step under {ckpt_dir}")
    return step


def numpy_restore(ckpt_dir: str | Path) -> dict:
    """Restore an orbax checkpoint as plain numpy arrays (CPU-safe)."""
    import jax
    import orbax.checkpoint as ocp

    step = latest_step(ckpt_dir)
    step_dir = Path(ckpt_dir).resolve() / str(step) / "default"
    ckptr = ocp.PyTreeCheckpointer()
    meta = ckptr.metadata(str(step_dir))
    tree = meta.item_metadata.tree
    restore_args = jax.tree.map(
        lambda m: ocp.RestoreArgs(restore_type=np.ndarray), tree
    )
    return ckptr.restore(
        str(step_dir), args=ocp.args.PyTreeRestore(restore_args=restore_args)
    )


def flatten_tree(tree, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten a nested dict/list pytree of arrays into {dotted.path: array}."""
    flat: dict[str, np.ndarray] = {}
    if isinstance(tree, dict):
        for k in sorted(tree, key=str):
            flat.update(flatten_tree(tree[k], f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            flat.update(flatten_tree(v, f"{prefix}.{i}" if prefix else str(i)))
    elif tree is None:
        pass
    else:
        flat[prefix] = np.asarray(tree)
    return flat


class Report:
    """Collects PASS/FAIL lines and renders a summary."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.lines.append((status, name, detail))
        print(f"[{status:4}] {name}" + (f"  {detail}" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.lines if s == "FAIL")

    def summary(self) -> int:
        n = len(self.lines)
        print(f"\n{n - self.failed}/{n} checks passed")
        return 1 if self.failed else 0


def compare_scalar(report: Report, name: str, got, want, atol: float) -> None:
    delta = abs(float(got) - float(want))
    if delta <= atol:
        report.add("PASS", name, f"delta={delta:.3g} (atol={atol:.3g})")
    else:
        report.add(
            "FAIL", name,
            f"got={got!r} want={want!r} delta={delta:.3g} atol={atol:.3g}",
        )


def compare_array(report: Report, name: str, got, want, atol: float) -> None:
    got = np.asarray(got)
    want = np.asarray(want)
    if got.shape != want.shape:
        report.add("FAIL", name, f"shape {got.shape} != {want.shape}")
        return
    delta = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) if got.size else 0.0
    if delta <= atol:
        report.add("PASS", name, f"max|d|={delta:.3g} (atol={atol:.3g})")
    else:
        report.add("FAIL", name, f"max|d|={delta:.3g} atol={atol:.3g}")
