# craftax-ReMDM-planner: vast.ai image.
#
#   docker build -t <dockerhub-user>/remdm-craftax:cuda13 .
#   docker push  <dockerhub-user>/remdm-craftax:cuda13
#
# Bakes the slow part: the jax[cuda13] wheel set (~3 GB of downloads) and,
# optionally, the released checkpoints. The repo itself is cloned at boot, so
# code changes never need an image rebuild. Rebuild only when uv.lock changes.
#
# Nothing here starts training. You SSH in and run commands yourself.

# Extending a vastai/* image keeps SSH, Jupyter and the instance portal working,
# and vast hosts already cache these layers, so only your own layers get pulled.
# Pick a current tag from https://hub.docker.com/r/vastai/base-image/tags .
# The CUDA version in the tag barely matters: jax[cuda13] ships its own CUDA and
# cuDNN as pip wheels. What matters is the host driver (580+ for CUDA 13).
ARG BASE_IMAGE=vastai/base-image:cuda-13.3.1-auto
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH

# uv. Pin the tag once you are happy with a version.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# --- the expensive layer: dependencies only -------------------------------
# README.md is copied because pyproject.toml references it.
WORKDIR /opt/build
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --extra cuda --no-install-project \
    && uv cache clean

# Fails the build early if the wheel set is broken. Reports CPU-only devices at
# build time, which is expected.
RUN python -c "import jax; print('jax', jax.__version__)"

# --- optional: bake the released checkpoints (~470 MB) --------------------
# Saves the Hub download on every boot. Comment out if you only ever train from
# scratch, or narrow the --include glob to the one checkpoint you use.
RUN uv pip install --python /opt/venv/bin/python --no-cache "huggingface_hub[cli]" \
    && HF_HOME=/tmp/hf hf download mathisweil/remdm-craftax-checkpoints \
        --include "checkpoints/**" --local-dir /opt/checkpoints \
    && rm -rf /tmp/hf

WORKDIR /workspace
