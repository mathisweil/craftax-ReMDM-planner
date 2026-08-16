"""GPU-gated agreement and restore tests (step 8).

The CPU-forced test harness (conftest sets JAX_PLATFORMS=cpu at import)
cannot host a CUDA backend in-process, so each check runs the same
deterministic computation in subprocesses with JAX_PLATFORMS set per
backend and compares the outputs. Skipped cleanly when no CUDA device
or no released checkpoints are present.

Sources: spec-method §2/§3 mathematics must be backend-invariant up to
float32 reassociation (JAX PRNG streams are bit-identical across
backends); spec-training §3.1 released experts (GPU restore verified
dynamically in step 7; CPU restore is a documented failure).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from tests.conftest import ROOT

_GPU = (
    os.environ.get("CUDA_VISIBLE_DEVICES") != ""
    and shutil.which("nvidia-smi") is not None
    and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
)
requires_gpu = pytest.mark.skipif(not _GPU, reason="no CUDA device visible")

_EXPERT = (
    ROOT / "checkpoints/hf/checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M"
)

_AGREEMENT_SCRIPT = r"""
import json, sys
import jax, jax.numpy as jnp
from src.diffusion.schedules import SCHEDULE_MAP
from src.diffusion.loss import compute_loss
from src.diffusion.sampling import sample_plan
from src.planners.model import build_model, init_params, make_apply_fns

V, H, OBS, B = 5, 8, 12, 32
cfg = {"D_MODEL": 32, "N_HEADS": 2, "N_LAYERS": 2, "D_FF": 32,
       "OBS_ENCODER_LAYERS": 1, "OBS_ENCODER_WIDTH": 32, "PLAN_HORIZON": H}
model = build_model(cfg, V)
params = init_params(model, jax.random.PRNGKey(0), OBS, H)
apply_eval, apply_train = make_apply_fns(model)
fn, deriv = SCHEDULE_MAP["cosine"]
k = jax.random.PRNGKey(1)
x0 = jax.random.randint(k, (B, H), 0, V)
obs = jax.random.normal(jax.random.PRNGKey(2), (B, OBS))
loss, _ = compute_loss(apply_train, params, jax.random.PRNGKey(3), x0, obs,
                       jnp.ones(B), V, fn, deriv, t_min=0.5, t_max=0.5)
plan = sample_plan(apply_eval, params, jax.random.PRNGKey(4), obs, V, H,
                   num_steps=6, schedule_fn=fn, remask_strategy="rescale",
                   eta=0.5, use_loop=False, temperature=0.0, top_p=None)
print(json.dumps({
    "backend": jax.devices()[0].platform,
    "loss": float(loss),
    "plan": [int(t) for t in jax.numpy.ravel(plan)],
}))
"""


def _run_backend(script: str, platform: str) -> dict:
    env = {
        **os.environ,
        "JAX_PLATFORMS": platform,
        "WANDB_MODE": "disabled",
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "Craftax_Baselines")]),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


@requires_gpu
def test_loss_and_sampler_agree_between_cuda_and_cpu():
    """The loss at pinned t and a deterministic (argmax) ReMDM plan
    agree across CPU and CUDA backends.

    JAX threefry PRNG streams are bit-identical across backends, so the
    forward masking and Bernoulli draws coincide; only float32
    reassociation (and TF32-class matmul rounding on GPU) separates the
    two. Bounds: loss within 5e-3 relative; plan token agreement above
    99% (argmax flips require near-exact logit ties).
    """
    gpu = _run_backend(_AGREEMENT_SCRIPT, "cuda")
    cpu = _run_backend(_AGREEMENT_SCRIPT, "cpu")
    assert gpu["backend"] == "gpu" and cpu["backend"] == "cpu"
    assert gpu["loss"] == pytest.approx(cpu["loss"], rel=5e-3)
    agree = np.mean(np.array(gpu["plan"]) == np.array(cpu["plan"]))
    assert agree > 0.99, f"plan agreement {agree}"


_RESTORE_SCRIPT = r"""
import json
import jax
from tests.conftest import load_config
from src.planners.env import make_env
from src.planners.ppo import build_ppo_network, load_ppo_params

config = load_config("configs/defaults.yaml")
env, env_params = make_env(config, 1)
num_actions = int(env.action_space(env_params).n)
obs_dim = int(env.observation_space(env_params).shape[0])
net = build_ppo_network("ppo_rnn", num_actions, 512, config)
params = load_ppo_params(
    "checkpoints/hf/checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M",
    net, "ppo_rnn", num_envs=1, obs_shape=(obs_dim,), layer_size=512,
)
n_leaves = len(jax.tree.leaves(params))
print(json.dumps({"backend": jax.devices()[0].platform, "n_leaves": n_leaves,
                  "obs_dim": obs_dim}))
"""


@requires_gpu
@pytest.mark.skipif(not _EXPERT.exists(), reason="released expert not downloaded")
def test_released_ppo_expert_restores_on_gpu():
    """The released Classic PPO expert (1e9 frames) restores on the GPU
    backend against the Classic environment geometry (spec-training
    §3.1; step-7 live-service check - CPU restore is the documented
    failure mode, so this runs CUDA-only)."""
    env = {
        **os.environ,
        "JAX_PLATFORMS": "cuda",
        "WANDB_MODE": "disabled",
        "PYTHONPATH": os.pathsep.join(
            [str(ROOT), str(ROOT / "Craftax_Baselines"), str(ROOT / "tests")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", _RESTORE_SCRIPT],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    out = json.loads(result.stdout.strip().splitlines()[-1])
    assert out["backend"] == "gpu"
    assert out["n_leaves"] > 0
    assert out["obs_dim"] == 1345  # Classic symbolic observation size
