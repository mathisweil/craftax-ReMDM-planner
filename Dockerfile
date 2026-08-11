# craftax-ReMDM-planner: vast.ai image.
#
# Build and push (amd64 is mandatory: vast GPU hosts are x86, an arm64 image
# from an Apple Silicon Mac will not run):
#
#   docker buildx build --platform linux/amd64 \
#       -t mathisweil/remdm-craftax:cuda13 \
#       -t mathisweil/remdm-craftax:latest --push .
#
# Bakes only the slow part: the jax[cuda13] wheel set (~3 GB of downloads). The
# repo is cloned at boot by scripts/vast_onstart.sh, so code changes never need
# a rebuild. Rebuild only when uv.lock changes.
#
# Nothing here starts training. You SSH in and run commands yourself.

# Extending a vastai/* image keeps SSH, Jupyter and the instance portal working,
# and vast hosts already cache these layers, so only your own layers get pulled.
# Current tags: https://hub.docker.com/r/vastai/base-image/tags
ARG BASE_IMAGE=vastai/base-image:cuda-13.3.1-auto
FROM ${BASE_IMAGE}

# NOTE: PATH is deliberately NOT modified. The vast base image runs its
# supervisor, Jupyter and portal scripts against /venv/main, and shadowing bare
# `python` / `pip` globally can break them. Everything below addresses
# /opt/venv explicitly; interactive shells get it via the on-start script.
ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1

# uv. Pinned so a rebuild months from now is reproducible; bump deliberately.
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /usr/local/bin/

# Tools the on-start script needs. The base image normally has these, but an
# explicit install costs ~nothing and turns a confusing boot failure into a
# build failure.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git git-lfs tmux rsync \
    && rm -rf /var/lib/apt/lists/*

# --- the expensive layer: dependencies only -------------------------------
# README.md is copied because pyproject.toml references it.
# --no-install-project keeps this layer keyed on the lockfile alone, so editing
# source code never invalidates it.
WORKDIR /opt/build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra cuda13 --no-install-project

# Fail the build now rather than after a 10 GB pull. Reports CPU-only devices
# at build time, which is expected: there is no GPU in the builder.
RUN /opt/venv/bin/python -c "import jax; print('jax', jax.__version__)"

# Record the lockfile the venv was built from. The on-start script compares it
# against the cloned repo's lockfile and warns loudly on drift, which otherwise
# shows up as a silent 3 GB download at boot.
RUN cp uv.lock /opt/venv/baked-uv.lock

WORKDIR /workspace

# --- optional: bake the released checkpoints ------------------------------
# Left out on purpose. 470 MB on the Hub unpacks into a large small-file layer,
# which is the slowest kind to pull and a likely cause of stalled image pulls.
# Fetching them at boot takes under a minute. If you do want them baked, use an
# isolated env so the pinned huggingface_hub in /opt/venv is not disturbed:
#
# RUN HF_HOME=/tmp/hf uvx --from "huggingface_hub[cli]" \
#         hf download mathisweil/remdm-craftax-checkpoints \
#         --include "checkpoints/**" --local-dir /opt/checkpoints \
#     && rm -rf /tmp/hf