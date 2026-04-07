"""Stage and upload the COMP0258 demo bundle to a public HuggingFace repo.

The downstream consumer is ``demo.ipynb`` (Cell 1), which calls
``snapshot_download(repo_id=HF_REPO_ID, local_dir="remdm-craftax")`` and then
imports from ``src/`` and reads ``checkpoint/``,
``experiments/.../analysis/{figures,tables}/``, and ``configs/``.

What this script uploads:

1. Source tree              -> ``src/``, ``Craftax_Baselines/``, ``configs/``
2. DAgger checkpoint        -> ``checkpoint/`` (renamed from ``artifacts/...``)
3. Pre-computed ablation    -> ``experiments/rl_finetuning/outputs/
                                 craftax_classic_final/analysis/``
4. Notebook + project meta  -> ``demo.ipynb``, ``pyproject.toml``, ``README.md``

What this script DOES NOT upload (filtered via ``ignore_patterns`` in
:func:`huggingface_hub.HfApi.upload_folder`):

- ``wandb/``, ``outputs/`` (other than the one ablation analysis dir)
- ``__pycache__``, ``*.pyc``, ``.venv``, ``uv.lock``
- ``tmp/``, ``checkpoints/`` (the legacy offline-BC checkpoints)
- All ``.npz`` rollout dumps
- Any ``.env`` / ``.git*`` files

Run it with::

    HF_TOKEN=hf_xxx uv run python scripts/upload_demo_bundle.py \\
        --repo-id  your-username/remdm-craftax-demo \\
        --private  false

The script first stages everything into a temporary directory, prints the
final tree size, asks for confirmation (unless ``--yes``), then uploads.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Project-relative paths.  All sources are referenced relative to PROJECT_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CHECKPOINT_SRC = (
    PROJECT_ROOT
    / "artifacts"
    / "Craftax-Classic-Symbolic-v1-policy-best-v4"
)

ABLATION_SRC = (
    PROJECT_ROOT
    / "experiments"
    / "rl_finetuning"
    / "outputs"
    / "craftax_classic_final"
    / "analysis"
)

# (source_relative_to_root, destination_relative_to_staging)
SOURCE_DIRS: list[tuple[Path, Path]] = [
    (PROJECT_ROOT / "src",               Path("src")),
    (PROJECT_ROOT / "Craftax_Baselines", Path("Craftax_Baselines")),
    (PROJECT_ROOT / "configs",           Path("configs")),
    (CHECKPOINT_SRC,                     Path("checkpoint")),
    (ABLATION_SRC,                       Path(
        "experiments/rl_finetuning/outputs/craftax_classic_final/analysis"
    )),
]

SOURCE_FILES: list[tuple[Path, Path]] = [
    (PROJECT_ROOT / "demo.ipynb",     Path("demo.ipynb")),
    (PROJECT_ROOT / "pyproject.toml", Path("pyproject.toml")),
    (PROJECT_ROOT / "README.md",      Path("README.md")),
]

# Patterns excluded from every directory copy.
COPY_IGNORE = shutil.ignore_patterns(
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "wandb",
    "outputs",
    "*.npz",
    "*.log",
    ".env",
    ".git",
    ".DS_Store",
)

# Extra defensive ignore list passed to ``HfApi.upload_folder``.  Hub-side
# filter applied AFTER staging in case anything slipped past ``shutil``.
HUB_IGNORE_PATTERNS = [
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.ipynb_checkpoints/**",
    "**/wandb/**",
    "**/.DS_Store",
    "**/.env",
]


def stage_bundle(staging_dir: Path) -> None:
    """Copy all source dirs and files into ``staging_dir`` with filtering.

    Args:
        staging_dir: Empty directory to populate.

    Raises:
        FileNotFoundError: If a required source path is missing.
    """
    for src, dst in SOURCE_DIRS:
        if not src.exists():
            raise FileNotFoundError(f"Required source dir missing: {src}")
        target = staging_dir / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Copying %s -> %s", src.relative_to(PROJECT_ROOT), dst)
        shutil.copytree(src, target, ignore=COPY_IGNORE, dirs_exist_ok=False)

    for src, dst in SOURCE_FILES:
        if not src.exists():
            raise FileNotFoundError(f"Required source file missing: {src}")
        target = staging_dir / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Copying %s -> %s", src.relative_to(PROJECT_ROOT), dst)
        shutil.copy2(src, target)


def directory_size_mb(path: Path) -> float:
    """Return total size of ``path`` (recursive) in megabytes."""
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / 1_048_576.0


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo id, e.g. 'your-username/remdm-craftax-demo'",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private (default: public, required by demo.ipynb)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage the bundle and print sizes without uploading",
    )
    return parser.parse_args()


def main() -> int:
    """Stage and upload the demo bundle.

    Returns:
        Process exit code (0 on success, non-zero on failure).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    )
    args = parse_args()

    token = os.environ.get("HF_TOKEN")
    if not args.dry_run and not token:
        logger.error("HF_TOKEN environment variable is not set.")
        return 1

    with tempfile.TemporaryDirectory(prefix="remdm-bundle-") as tmp:
        staging_dir = Path(tmp)
        try:
            stage_bundle(staging_dir)
        except FileNotFoundError as exc:
            logger.error("Staging failed: %s", exc)
            return 1

        size_mb = directory_size_mb(staging_dir)
        n_files = sum(1 for _ in staging_dir.rglob("*") if _.is_file())
        logger.info(
            "Staged bundle: %d files, %.1f MB at %s", n_files, size_mb, staging_dir,
        )

        if args.dry_run:
            logger.info("Dry run complete; nothing uploaded.")
            return 0

        if not args.yes:
            answer = input(
                f"Upload {n_files} files ({size_mb:.1f} MB) to "
                f"{args.repo_id} ({'private' if args.private else 'public'})? [y/N] "
            )
            if answer.strip().lower() not in {"y", "yes"}:
                logger.info("Aborted by user.")
                return 0

        # Lazy import so ``--dry-run`` works without huggingface_hub installed.
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=args.repo_id,
            folder_path=str(staging_dir),
            repo_type="model",
            ignore_patterns=HUB_IGNORE_PATTERNS,
            commit_message="Upload COMP0258 demo bundle (code + checkpoint + ablation assets)",
        )
        logger.info("Upload complete: https://huggingface.co/%s", args.repo_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
