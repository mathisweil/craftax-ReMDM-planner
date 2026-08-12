"""Upload the trained Craftax ReMDM checkpoints to the Hugging Face Hub.

Discovers every Orbax checkpoint under ``checkpoints/``, stages it with the
repo-relative layout preserved, drops wandb environment metadata (which carries
the author's email, hostname and local paths), generates a model card from the
checkpoints' own config snapshots, and uploads.

    HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py \\
        --repo-id MathisW78/remdm-craftax-checkpoints [--dry-run] [--private]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CKPTS = ROOT / "checkpoints"

PAPER = (
    "The Double Intractability of Reinforcement Learning "
    "for Discrete Diffusion Planners"
)
CODE_URL = "https://github.com/mathisweil/craftax-ReMDM-planner"

ROLES = {
    "offline": "Diffusion planner (offline BC)",
    "online": "Diffusion planner (online DAgger)",
    "ppo_agents": "PPO-RNN expert",
}

# wandb-metadata.json is pure environment provenance (email, host, git remote,
# absolute paths) and is never needed to restore a checkpoint.
COPY_IGNORE = shutil.ignore_patterns(
    ".DS_Store", "__pycache__", "*.pyc", "wandb-metadata.json",
)
HUB_IGNORE = ["**/.DS_Store", "**/__pycache__/**", "**/wandb-metadata.json"]


def discover() -> dict[Path, list[int]]:
    """Map each checkpoint directory to its saved step numbers."""
    models: dict[Path, list[int]] = {}
    for marker in sorted(CKPTS.glob("*/*/*/_CHECKPOINT_METADATA")):
        models.setdefault(marker.parent.parent, []).append(int(marker.parent.name))
    return models


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def describe(model_dir: Path, steps: list[int]) -> dict[str, str]:
    """Pull env name and training detail out of a checkpoint's own metadata."""
    resume = model_dir / "resume_metadata.json"
    if resume.exists():
        meta = json.loads(resume.read_text())
        cfg = meta["config_snapshot"]
        detail = f"{meta['total_gradient_steps_completed']:,} grad steps"
        arch = (
            f"{cfg['N_LAYERS']}L, d_model {cfg['D_MODEL']}, "
            f"{cfg['N_HEADS']} heads, horizon {cfg['PLAN_HORIZON']}"
        )
    else:
        raw = yaml.safe_load((model_dir / "config.yaml").read_text())
        cfg = {k: v["value"] for k, v in raw.items() if k != "_wandb"}
        detail = f"{float(cfg['TOTAL_TIMESTEPS']):.0e} frames"
        arch = f"RNN, layer size {cfg['LAYER_SIZE']}"
    return {
        "path": str(model_dir.relative_to(ROOT)),
        "role": ROLES[model_dir.parent.name],
        "env": cfg["ENV_NAME"],
        "arch": arch,
        "step": str(max(steps)),
        "detail": detail,
        "size": f"{dir_size_mb(model_dir):.0f} MB",
    }


def strip_wandb_block(config_yaml: Path) -> None:
    """Remove the ``_wandb`` blob (email, local paths) from a staged config."""
    raw = yaml.safe_load(config_yaml.read_text())
    raw.pop("_wandb", None)
    config_yaml.write_text(yaml.safe_dump(raw, sort_keys=True))


def scrub_abs_paths(resume_json: Path) -> None:
    """Shorten absolute cluster paths in a staged config snapshot."""
    meta = json.loads(resume_json.read_text())
    cfg = meta.get("config_snapshot", {})
    for key, value in cfg.items():
        if isinstance(value, str) and value.startswith("/"):
            cfg[key] = "/".join(Path(value).parts[-2:])
    resume_json.write_text(json.dumps(meta, indent=2))


def stage(staging: Path, models: dict[Path, list[int]]) -> list[dict[str, str]]:
    """Copy checkpoints, LICENSE and model card into ``staging``."""
    rows = []
    for model_dir, steps in models.items():
        target = staging / model_dir.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(model_dir, target, ignore=COPY_IGNORE)
        if (target / "config.yaml").exists():
            strip_wandb_block(target / "config.yaml")
        if (target / "resume_metadata.json").exists():
            scrub_abs_paths(target / "resume_metadata.json")
        rows.append(describe(model_dir, steps))

    shutil.copy2(ROOT / "LICENSE", staging / "LICENSE")
    return rows


def model_card(repo_id: str, rows: list[dict[str, str]]) -> str:
    header = (
        "| Path | Role | Environment | Architecture | Selected at | Training | Size |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    table = header + "".join(
        f"| `{r['path']}` | {r['role']} | `{r['env']}` | {r['arch']} | "
        f"{r['step']} | {r['detail']} | {r['size']} |\n"
        for r in sorted(rows, key=lambda r: r["path"])
    )
    return f"""---
license: mit
pipeline_tag: reinforcement-learning
tags:
- reinforcement-learning
- planning
- discrete-diffusion
- remdm
- craftax
- jax
- flax
- orbax
---

# ReMDM Planner: Craftax checkpoints

Trained weights accompanying *{PAPER}*: a remasking discrete diffusion model
(ReMDM) used as an action-sequence planner in
[Craftax](https://github.com/MichaelTMatthews/Craftax), together with the
PPO-RNN experts that supervise it.

Code, configs and evaluation harness: {CODE_URL}

## Contents

{table}
Each diffusion checkpoint ships a `resume_metadata.json` holding the full
config snapshot it was trained under; each PPO expert ships `config.yaml` and
`wandb-summary.json` (final training metrics).

Weights are [Orbax](https://orbax.readthedocs.io) checkpoint directories
(OCDBT format), not `safetensors` — the models are Flax modules restored via
`orbax.checkpoint`, and the paths above mirror the source repository so a
snapshot can be dropped straight into a working copy.

## Download

```python
from huggingface_hub import snapshot_download

# everything (~{sum(float(r['size'].split()[0]) for r in rows):.0f} MB)
snapshot_download(repo_id="{repo_id}", local_dir=".")

# a single model
snapshot_download(
    repo_id="{repo_id}",
    local_dir=".",
    allow_patterns="checkpoints/online/Craftax-Classic-*/**",
)
```

## Use

From a clone of the code repository, after downloading into it:

```bash
uv run python main.py --mode inference \\
    --checkpoint checkpoints/online/Craftax-Classic-Symbolic-v1-OnlineDiffusion-DAgger-100M
```

Programmatic loading uses `src.planners.model.load_checkpoint` for the
diffusion planners and `src.planners.ppo.load_ppo_agent` for the experts; both
take the checkpoint directory path and restore the latest step. Architecture
arguments should be read from the checkpoint's own `resume_metadata.json`
rather than hardcoded.

## Training

The diffusion planners are bidirectional transformers that denoise a masked
action plan conditioned on the symbolic observation, trained either by offline
behaviour cloning on PPO rollouts or by online DAgger against the PPO expert.
Model size and horizon differ per run (see the table); the PPO-RNN experts are
the Craftax baselines. Exact hyperparameters for every run, including the
remasking strategy, schedule and sampling settings, are in the per-checkpoint
metadata files listed above, which are the authoritative record.

Directory names encode the environment and the total environment timesteps the
run was trained for. `Latest step` is whatever each run used as its Orbax step
counter: environment frames for most runs, and update steps for the full
Craftax DAgger run, whose 1,743 updates cover the same ~100M timesteps.

## Limitations

These are research artefacts tied to specific Craftax versions and symbolic
observation encodings; they are not general-purpose agents and will not
transfer to other environments or to pixel observations. Evaluation results and
their variance are reported in the paper.

## Citation

```bibtex
@inproceedings{{remdm-craftax-planner,
  title  = {{{PAPER}}},
  author = {{Weil, Mathis}},
  year   = {{2026}},
  note   = {{NeurIPS 2026 Workshop: Beyond Next-Token Prediction}}
}}
```

## License

MIT, see `LICENSE`.
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True, help="e.g. MathisW78/remdm-craftax-checkpoints")
    p.add_argument("--private", action="store_true", help="create the repo private")
    p.add_argument("--dry-run", action="store_true", help="stage and print, do not upload")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        print("HF_TOKEN is not set.", file=sys.stderr)
        return 1

    models = discover()
    if not models:
        print(f"No Orbax checkpoints found under {CKPTS}.", file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory(prefix="remdm-ckpt-") as tmp:
        staging = Path(tmp)
        rows = stage(staging, models)
        (staging / "README.md").write_text(model_card(args.repo_id, rows))

        files = [f for f in staging.rglob("*") if f.is_file()]
        print(f"Staged {len(models)} checkpoints, {len(files)} files, "
              f"{dir_size_mb(staging):.0f} MB")
        for r in sorted(rows, key=lambda r: r["path"]):
            print(f"  {r['path']:<70} {r['size']:>8}")

        if args.dry_run:
            print(f"Dry run; staged tree left nowhere. Card:\n\n{model_card(args.repo_id, rows)}")
            return 0

        if not args.yes:
            visibility = "private" if args.private else "public"
            if input(f"Upload to {args.repo_id} ({visibility})? [y/N] ").strip().lower() not in {"y", "yes"}:
                print("Aborted.")
                return 0

        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staging),
            repo_type="model",
            ignore_patterns=HUB_IGNORE,
            commit_message="Upload Craftax ReMDM planner and PPO expert checkpoints",
        )
        print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
