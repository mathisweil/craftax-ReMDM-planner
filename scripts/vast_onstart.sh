#!/bin/bash
# vast.ai provisioning script for craftax-ReMDM-planner.
#
# Host this at a raw URL (GitHub raw or a gist) and set the template env var
# PROVISIONING_SCRIPT to that URL.
#
# It only prepares the box: clone the repo, link the pre-baked venv, expose the
# pre-baked checkpoints. It never starts a run. Everything it does is fast
# because the dependencies are already in the image.
#
# Optional template env vars:
#   BRANCH           git branch to check out (default: main)
#   REPO_URL         override the clone URL (e.g. your fork or an SSH URL)
#   PULL_CHECKPOINTS set to 1 to fetch the released checkpoints from the Hub

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mathisweil/craftax-ReMDM-planner.git}"
REPO_DIR="${REPO_DIR:-/workspace/craftax-ReMDM-planner}"
BRANCH="${BRANCH:-main}"

export UV_PROJECT_ENVIRONMENT=/opt/venv
export UV_LINK_MODE=copy
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

# 1. Code (submodules matter: Craftax_Baselines is one).
if [ ! -d "$REPO_DIR/.git" ]; then
    git clone --recurse-submodules --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
else
    git -C "$REPO_DIR" fetch --all --prune
    git -C "$REPO_DIR" checkout "$BRANCH"
    git -C "$REPO_DIR" pull --ff-only
    git -C "$REPO_DIR" submodule update --init --recursive
fi
cd "$REPO_DIR"

# 2. Dependencies are already in /opt/venv, so this only installs the project
#    itself: seconds, not minutes. Warn first if the lockfile has drifted from
#    the one the image was built with, because that silently turns this line
#    into a multi-GB download.
if ! cmp -s uv.lock /opt/venv/baked-uv.lock; then
    echo "WARNING: uv.lock differs from the image's baked lockfile."
    echo "         This sync will re-download dependencies. Rebuild the image."
fi
uv sync --frozen --extra cuda13

# 3. Checkpoints. Symlink them if they were baked into the image, otherwise
#    fetch from the Hub (~470 MB, under a minute).
if [ -d /opt/checkpoints/checkpoints ] && [ ! -e "$REPO_DIR/checkpoints" ]; then
    ln -s /opt/checkpoints/checkpoints "$REPO_DIR/checkpoints"
elif [ -n "${PULL_CHECKPOINTS:-}" ] && [ ! -d "$REPO_DIR/checkpoints" ]; then
    uv run hf download mathisweil/remdm-craftax-checkpoints \
        --include "checkpoints/**" --local-dir "$REPO_DIR"
fi

# 4. Interactive SSH shells land in the repo with the right env.
{
    echo "export UV_PROJECT_ENVIRONMENT=/opt/venv"
    echo "export HF_HOME=$HF_HOME"
    echo "cd $REPO_DIR"
} >> /root/.bashrc

# 5. Confirm JAX sees the GPU before you spend rental time on it.
uv run python -c "import jax; print('jax', jax.__version__, jax.devices())"

echo "ready: $REPO_DIR"
