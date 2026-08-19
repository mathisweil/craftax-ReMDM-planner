"""Upload the trained Craftax ReMDM checkpoints and results to the Hugging Face Hub.

Discovers every Orbax checkpoint under ``checkpoints/``, every ablation run
under ``experiments/rl_finetuning/outputs/`` and every ``--mode inference``
result under ``results/inference/``, stages them with the repo-relative layout
preserved, drops wandb environment metadata (which carries the author's email,
hostname and local paths), generates a model card from the checkpoints' own
config snapshots, and uploads.

    HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py \\
        --repo-id mathisweil/remdm-craftax-checkpoints \\
        [--inference-results PATH ...] [--dry-run] [--private]
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
RUNS = ROOT / "experiments" / "rl_finetuning" / "outputs"
INFERENCE = ROOT / "results" / "inference"

PAPER = (
    "The Double Intractability of Reinforcement Learning "
    "for Discrete Diffusion Planners"
)
CODE_URL = "https://github.com/mathisweil/craftax-ReMDM-planner"
ENV_NAME = "Craftax"

ROLES = {
    "offline": "Diffusion planner (offline BC)",
    "online": "Diffusion planner (online DAgger)",
    "ppo_agents": "PPO-RNN expert",
}

# An ablation run is published as its own summary plus tables and figures; the
# raw per-iteration logs stay in the code repository.
RUN_FILES = ("results.json", "diagnosis.md")
RUN_DIRS = ("tables", "figures")

# wandb-metadata.json is pure environment provenance (email, host, git remote,
# absolute paths) and is never needed to restore a checkpoint.
COPY_IGNORE = shutil.ignore_patterns(
    ".DS_Store", "__pycache__", "*.pyc", "wandb-metadata.json",
)
HUB_IGNORE = ["**/.DS_Store", "**/__pycache__/**", "**/wandb-metadata.json"]


# =============================================================================
# Discovery
# =============================================================================

# `checkpoints/hf/` is where a Hub *download* lands. Publishing from it would
# re-upload already-published artefacts into a nested `checkpoints/hf/...` tree
# on the Hub, so it is never a publish source in either repo.
HF_DOWNLOAD_DIR = "hf"


def _is_download_copy(path: Path) -> bool:
    """True for anything under ``checkpoints/hf/``, wherever it sits."""
    return HF_DOWNLOAD_DIR in path.relative_to(CKPTS).parts


def discover_checkpoints() -> dict[Path, list[int]]:
    """Map each checkpoint directory to its saved step numbers.

    Discovery is at the **released layout**, ``checkpoints/<role>/<name>/<step>/``
    — the layout the Hub repo mirrors, which is why `--dry-run` shows the tree a
    publish would create. A training run writes elsewhere, so its `policies`
    directory has to be copied into place first; the README documents that and
    it is a real requirement, not an accident of this glob.

    Anything under ``checkpoints/hf/`` is skipped as a download copy. The fixed
    depth already excluded it here, one level deeper than a real checkpoint, but
    only by arithmetic — the exclusion is now stated, so it survives a layout
    with a different depth.

    Measured on this repo's live tree: 4 checkpoints discovered, 4 download
    copies under ``checkpoints/hf/`` skipped.

    Returns:
        ``{checkpoint directory: [step numbers]}``.
    """
    models: dict[Path, list[int]] = {}
    for marker in sorted(CKPTS.glob("*/*/*/_CHECKPOINT_METADATA")):
        if _is_download_copy(marker):
            continue
        models.setdefault(marker.parent.parent, []).append(int(marker.parent.name))
    return models


def discover_runs() -> list[Path]:
    """Every ablation output directory holding a ``results.json``."""
    if not RUNS.is_dir():
        return []
    return sorted(d for d in RUNS.iterdir() if (d / "results.json").is_file())


def discover_inference(extra: list[str]) -> list[Path]:
    """Inference result JSONs: the default directory plus any given paths."""
    found: list[Path] = []
    for source in [INFERENCE, *(Path(p) for p in extra)]:
        if source.is_dir():
            found.extend(sorted(source.glob("*.json")))
        elif source.is_file():
            found.append(source)
        elif source != INFERENCE:
            print(f"No inference results at {source}.", file=sys.stderr)
    return list(dict.fromkeys(found))


# =============================================================================
# Helpers
# =============================================================================

def dir_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / 1_048_576
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576


def human_size(path: Path) -> str:
    mb = dir_size_mb(path)
    return f"{mb:.0f} MB" if mb >= 1 else f"{max(mb * 1024, 1):.0f} KB"


def plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def shorten_paths(value):
    """Shorten absolute cluster paths anywhere in a staged JSON document."""
    if isinstance(value, dict):
        return {k: shorten_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten_paths(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return "/".join(Path(value).parts[-2:])
    return value


def strip_wandb_block(config_yaml: Path) -> None:
    """Remove the ``_wandb`` blob (email, local paths) from a staged config."""
    raw = yaml.safe_load(config_yaml.read_text())
    raw.pop("_wandb", None)
    config_yaml.write_text(yaml.safe_dump(raw, sort_keys=True))


def scrub_abs_paths(resume_json: Path) -> None:
    """Shorten absolute cluster paths in a staged config snapshot."""
    meta = json.loads(resume_json.read_text())
    meta["config_snapshot"] = shorten_paths(meta.get("config_snapshot", {}))
    resume_json.write_text(json.dumps(meta, indent=2))


# =============================================================================
# Description
# =============================================================================

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
        "role": ROLES.get(model_dir.parent.name, model_dir.parent.name),
        "env": cfg["ENV_NAME"],
        "arch": arch,
        "step": f"{max(steps):,}",
        "detail": detail,
        "size": human_size(model_dir),
    }


def describe_run(run: Path, staged: Path) -> dict[str, str]:
    """Summarise what an ablation run contributes to the release."""
    counts = [
        f"{len(list((staged / d).glob('*')))} {d}"
        for d in RUN_DIRS if (staged / d).is_dir()
    ]
    files = [f for f in RUN_FILES if (staged / f).is_file()]
    return {
        "run": run.name,
        "path": str(run.relative_to(ROOT)),
        "contents": ", ".join([*(f"`{f}`" for f in files), *counts]),
        "size": human_size(staged),
    }


def describe_inference(name: str, payload: dict) -> dict[str, str]:
    """Summarise one ``--mode inference`` result JSON."""
    metrics = payload.get("metrics", payload)
    score = metrics.get("mean_score")
    envs, steps = metrics.get("eval_num_envs"), metrics.get("eval_steps")
    return {
        "file": name,
        "env": payload.get("env_name", "-"),
        "episodes": f"{envs} envs x {steps} steps" if envs and steps else "-",
        "metric": (
            f"mean score {score:.2f}" if isinstance(score, int | float) else "-"
        ),
    }


# =============================================================================
# Staging
# =============================================================================

def stage_checkpoints(staging: Path, models: dict[Path, list[int]]) -> list[dict[str, str]]:
    """Copy each checkpoint directory, scrubbing its provenance metadata."""
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
    return rows


def stage_runs(staging: Path, runs: list[Path]) -> list[dict[str, str]]:
    """Copy each ablation run's summary, tables and figures."""
    rows = []
    for run in runs:
        target = staging / run.relative_to(ROOT)
        target.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            if (run / name).is_file():
                shutil.copy2(run / name, target / name)
        for name in RUN_DIRS:
            if (run / name).is_dir():
                shutil.copytree(run / name, target / name, ignore=COPY_IGNORE)
        rows.append(describe_run(run, target))
    return rows


def stage_inference(staging: Path, files: list[Path]) -> list[dict[str, str]]:
    """Copy each inference result JSON into ``results/inference/``."""
    target_dir = staging / INFERENCE.relative_to(ROOT)
    rows: list[dict[str, str]] = []
    for src in files:
        try:
            payload = json.loads(src.read_text())
        except json.JSONDecodeError:
            print(f"Skipping unreadable inference result {src}.", file=sys.stderr)
            continue
        name = src.name
        if any(r["file"] == name for r in rows):
            name = f"{src.parent.name}-{src.name}"
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / name).write_text(
            json.dumps(shorten_paths(payload), indent=2) + "\n",
        )
        row = describe_inference(name, payload)
        row["size"] = human_size(target_dir / name)
        rows.append(row)
    return rows


def stage(
    staging: Path,
    models: dict[Path, list[int]],
    runs: list[Path],
    inference: list[Path],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Stage checkpoints, results and LICENSE; the card is written by the caller."""
    rows = stage_checkpoints(staging, models)
    run_rows = stage_runs(staging, runs)
    inf_rows = stage_inference(staging, inference)
    shutil.copy2(ROOT / "LICENSE", staging / "LICENSE")
    return rows, run_rows, inf_rows


# =============================================================================
# Model card
# =============================================================================

def table(headers: list[str], lines: list[str]) -> str:
    sep = "|".join(["---"] * len(headers))
    return f"| {' | '.join(headers)} |\n|{sep}|\n" + "".join(f"| {ln} |\n" for ln in lines)


def checkpoint_table(rows: list[dict[str, str]]) -> str:
    return table(
        ["Path", "Role", "Environment", "Architecture", "Selected at", "Training", "Size"],
        [
            f"`{r['path']}` | {r['role']} | `{r['env']}` | {r['arch']} | "
            f"{r['step']} | {r['detail']} | {r['size']}"
            for r in sorted(rows, key=lambda r: r["path"])
        ],
    )


def results_section(run_rows: list[dict[str, str]], inf_rows: list[dict[str, str]]) -> str:
    """Ablation and inference tables; empty when the release carries neither."""
    parts = []
    if run_rows:
        parts.append(
            "RL fine-tuning ablation runs, as produced by "
            "`experiments/rl_finetuning/run_ablations.py`. Each run ships its "
            "`results.json` summary, the `diagnosis.md` write-up, and the "
            "tables (`.csv` and `.tex`) and figures generated from it.\n\n"
            + table(
                ["Run", "Contents", "Size"],
                [
                    f"`{r['path']}` | {r['contents']} | {r['size']}"
                    for r in sorted(run_rows, key=lambda r: r["run"])
                ],
            ),
        )
    if inf_rows:
        parts.append(
            "Evaluation results produced by `main.py --mode inference` on the "
            "checkpoints above, under `results/inference/`.\n\n"
            + table(
                ["File", "Environment", "Evaluation", "Headline metric", "Size"],
                [
                    f"`{r['file']}` | `{r['env']}` | {r['episodes']} | "
                    f"{r['metric']} | {r['size']}"
                    for r in sorted(inf_rows, key=lambda r: r["file"])
                ],
            ),
        )
    return "## Results\n\n" + "\n".join(parts) if parts else ""


def featured(rows: list[dict[str, str]]) -> dict[str, str]:
    """The checkpoint the download and usage examples are written against."""
    planners = [r for r in rows if "planner" in r["role"].lower()]
    return sorted(planners or rows, key=lambda r: r["path"])[0]


def model_card(
    repo_id: str,
    rows: list[dict[str, str]],
    run_rows: list[dict[str, str]],
    inf_rows: list[dict[str, str]],
    total_mb: float,
) -> str:
    example = featured(rows)
    return f"""---
license: mit
library_name: jax
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

# ReMDM Planner: {ENV_NAME} checkpoints

Trained weights accompanying *{PAPER}*: a remasking discrete diffusion model
(ReMDM) used as an action-sequence planner in
[Craftax](https://github.com/MichaelTMatthews/Craftax), together with the
PPO-RNN experts that supervise it, and the results reported in the paper.

Code, configs and evaluation harness: {CODE_URL}

## Contents

{checkpoint_table(rows)}
Each diffusion checkpoint ships a `resume_metadata.json` holding the full
config snapshot it was trained under; each PPO expert ships `config.yaml` and
`wandb-summary.json` (final training metrics).

Weights are [Orbax](https://orbax.readthedocs.io) checkpoint directories
(OCDBT format), not `safetensors` — the models are Flax modules restored via
`orbax.checkpoint`, and the paths above mirror the source repository so a
snapshot can be dropped straight into a working copy.

{results_section(run_rows, inf_rows)}
## Download

```python
from huggingface_hub import snapshot_download

# everything (~{total_mb:.0f} MB)
snapshot_download(repo_id="{repo_id}", local_dir=".")

# a single model
snapshot_download(
    repo_id="{repo_id}",
    local_dir=".",
    allow_patterns="{example['path']}/**",
)
```

## Use

From a clone of the code repository, after downloading into it:

```bash
uv run python main.py --mode inference \\
    --checkpoint {example['path']} \\
    --output results/inference/eval.json
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
run was trained for. `Selected at` is whatever each run used as its Orbax step
counter, which is environment frames for the runs published here.

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


# =============================================================================
# Entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True, help="e.g. mathisweil/remdm-craftax-checkpoints")
    p.add_argument(
        "--inference-results", nargs="*", default=[], metavar="PATH",
        help=f"extra --mode inference JSONs or directories, on top of "
             f"{INFERENCE.relative_to(ROOT)}/",
    )
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

    models = discover_checkpoints()
    if not models:
        print(
            f"No Orbax checkpoints found under {CKPTS}.\n"
            "Discovery expects the released layout, "
            "checkpoints/<role>/<name>/<step>/, and skips checkpoints/hf/ "
            "because that is where Hub downloads land. Copy a run's "
            "`wandb.run.dir/policies` directory to "
            "checkpoints/{offline,online}/<name> first.",
            file=sys.stderr,
        )
        return 1
    runs = discover_runs()
    inference = discover_inference(args.inference_results)

    with tempfile.TemporaryDirectory(prefix="remdm-craftax-") as tmp:
        staging = Path(tmp)
        rows, run_rows, inf_rows = stage(staging, models, runs, inference)
        total_mb = dir_size_mb(staging)
        card = model_card(args.repo_id, rows, run_rows, inf_rows, total_mb)
        (staging / "README.md").write_text(card)

        files = [f for f in staging.rglob("*") if f.is_file()]
        print(f"Staged {plural(len(rows), 'checkpoint')}, "
              f"{plural(len(run_rows), 'ablation run')}, "
              f"{plural(len(inf_rows), 'inference result')}, "
              f"{plural(len(files), 'file')}, {total_mb:.0f} MB")
        for r in sorted(rows, key=lambda r: r["path"]):
            print(f"  {r['path']:<70} {r['size']:>8}")
        for r in sorted(run_rows, key=lambda r: r["run"]):
            print(f"  {r['path']:<70} {r['size']:>8}  {r['contents']}")
        for r in sorted(inf_rows, key=lambda r: r["file"]):
            print(f"  results/inference/{r['file']:<52} {r['size']:>8}  {r['metric']}")
        if not run_rows:
            print(f"Warning: no ablation runs with a results.json under {RUNS}.",
                  file=sys.stderr)
        if not inf_rows:
            print("Warning: no inference results; produce them with "
                  "`main.py --mode inference --output "
                  f"{INFERENCE.relative_to(ROOT)}/<name>.json`.", file=sys.stderr)

        if args.dry_run:
            print(f"Dry run; staged tree left nowhere. Card:\n\n{card}")
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
            commit_message="Upload Craftax ReMDM planner checkpoints and results",
        )
        print(f"Done: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
