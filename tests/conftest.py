"""Shared fixtures for the smoke suite.

CPU only, fixed seed, tiny synthetic data, no network and no real checkpoints.
Environment guards are set at import time, before jax/matplotlib/wandb load.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("WANDB_MODE", "disabled")
# Headless plotting: analysis modules import pyplot at module scope.
os.environ.setdefault("MPLBACKEND", "Agg")
# Stable location inside the OS temp dir, not the repo and not the user's home.
# A fresh directory per run would make matplotlib rebuild its font cache (~12s).
_MPL_CACHE = Path(tempfile.gettempdir()) / "remdm-smoke-mplconfig"
_MPL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

for _path in (ROOT, ROOT / "Craftax_Baselines"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import jax
import jax.numpy as jnp
import yaml

SEED = 0
NUM_ACTIONS = 5
OBS_DIM = 8
PLAN_HORIZON = 4
BATCH = 3

TINY_ARCH = {
    "D_MODEL": 16,
    "N_HEADS": 2,
    "N_LAYERS": 1,
    "D_FF": 16,
    "OBS_ENCODER_LAYERS": 1,
    "OBS_ENCODER_WIDTH": 16,
    "PLAN_HORIZON": PLAN_HORIZON,
}


def _discover_modules(package_dir: str) -> list[str]:
    """Dotted module names for every .py file under *package_dir*."""
    modules = []
    for path in sorted((ROOT / package_dir).rglob("*.py")):
        parts = list(path.relative_to(ROOT).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        if parts:
            modules.append(".".join(parts))
    return modules


SRC_MODULES = _discover_modules("src")
EXPERIMENT_MODULES = _discover_modules("experiments")


def import_or_skip(module_name: str):
    """Import a repo module, skipping if only an optional dependency is absent.

    The Craftax_Baselines git submodule is not present in a bare clone, and
    several modules import from it. That is a missing optional checkout, not a
    defect, so those tests skip with a clear message instead of erroring.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        missing = (exc.name or "").split(".")[0]
        if missing in {"Craftax_Baselines", "craftax"}:
            pytest.skip(
                f"{module_name} needs {missing!r}, which is not available. "
                "Run 'git submodule update --init' and 'uv sync'."
            )
        raise


def load_config(relative_path: str) -> dict:
    """Load a repo YAML config and upper-case its keys, as the runners do."""
    with open(ROOT / relative_path) as f:
        return {k.upper(): v for k, v in (yaml.safe_load(f) or {}).items()}


def run_entry_point(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a repo entry point in a subprocess with the smoke-test guards applied."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "Craftax_Baselines")]),
        "JAX_PLATFORMS": "cpu",
        "WANDB_MODE": "disabled",
    }
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

# label -> argv. Every one is a no-side-effect smoke invocation: it must not
# read a real checkpoint, hit the network, or write outside a temp directory.
ENTRY_POINTS = {
    "main --help": ["main.py", "--help"],
    "main no-mode": ["main.py"],
    "count_params --help": ["scripts/count_params.py", "--help"],
    "count_params run": ["scripts/count_params.py", "--configs", "configs/defaults.yaml"],
    "eval_ppo_expert --help": ["scripts/eval_ppo_expert.py", "--help"],
    "hf_upload --help": ["scripts/hf_upload.py", "--help"],
    "hf_upload_demo --help": ["scripts/hf_upload_demo.py", "--help"],
    "run_ablations --help": ["experiments/rl_finetuning/run_ablations.py", "--help"],
    "run_ablations --list": ["experiments/rl_finetuning/run_ablations.py", "--list"],
    "run_ablations no-checkpoint": [
        "experiments/rl_finetuning/run_ablations.py", "--ablations", "baseline_rl",
        # run_ablations creates its output dir before validating arguments.
        "--output_dir", "<TMP>",
    ],
}


_entry_point_futures: dict[str, "Future"] = {}
_entry_point_pool: ThreadPoolExecutor | None = None
_entry_point_dir: str | None = None


def pytest_collection_modifyitems(config, items) -> None:
    """Start the entry-point subprocesses as soon as we know they are needed.

    Each costs several seconds of interpreter and JAX import. Launching them
    here lets them run concurrently with the in-process tests instead of
    blocking the suite when the fixture is first requested.
    """
    global _entry_point_pool, _entry_point_dir

    if not any("entry_point_runs" in item.fixturenames for item in items):
        return

    _entry_point_dir = tempfile.mkdtemp(prefix="remdm-smoke-entry-")
    _entry_point_pool = ThreadPoolExecutor(max_workers=len(ENTRY_POINTS))
    for label, args in ENTRY_POINTS.items():
        argv = [str(Path(_entry_point_dir) / "run") if a == "<TMP>" else a for a in args]
        _entry_point_futures[label] = _entry_point_pool.submit(run_entry_point, argv)


def pytest_sessionfinish(session, exitstatus) -> None:
    if _entry_point_pool is not None:
        _entry_point_pool.shutdown(wait=True)
    if _entry_point_dir is not None:
        shutil.rmtree(_entry_point_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def entry_point_runs() -> dict[str, subprocess.CompletedProcess]:
    """Results of every entry point's smoke invocation, awaited on first use."""
    return {label: future.result() for label, future in _entry_point_futures.items()}


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def real_config() -> dict:
    """The unmodified shipped config, exactly as ``main.py`` loads it."""
    return load_config("configs/defaults.yaml")


@pytest.fixture(scope="session")
def real_ablations_config() -> dict:
    """The unmodified shipped ablations config."""
    return load_config("experiments/rl_finetuning/configs/ablations_default.yaml")


@pytest.fixture(scope="session")
def tiny_config(real_config: dict) -> dict:
    """Real config with the architecture shrunk to smoke-test size."""
    return {
        **real_config,
        **TINY_ARCH,
        "NUM_ACTIONS": NUM_ACTIONS,
        "USE_WANDB": False,
        "SEED": SEED,
    }


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model(tiny_config: dict):
    from src.planners.model import build_model

    return build_model(tiny_config, NUM_ACTIONS)


@pytest.fixture(scope="session")
def params(model):
    from src.planners.model import init_params

    return init_params(model, jax.random.PRNGKey(SEED), OBS_DIM, PLAN_HORIZON)


@pytest.fixture(scope="session")
def apply_fns(model):
    """``(apply_eval, apply_train)`` closures for the tiny model."""
    from src.planners.model import make_apply_fns

    return make_apply_fns(model)


@pytest.fixture(scope="session")
def schedules(tiny_config: dict):
    """``(schedule_fn, schedule_deriv_fn)`` for the config's schedule."""
    from src.diffusion.schedules import SCHEDULE_MAP

    return SCHEDULE_MAP[tiny_config["DIFFUSION_SCHEDULE"]]


@pytest.fixture(scope="session")
def batch() -> dict:
    """Tiny synthetic training batch. Deterministic under the fixed seed."""
    keys = jax.random.split(jax.random.PRNGKey(SEED), 3)
    return {
        "obs": jax.random.normal(keys[0], (BATCH, OBS_DIM)),
        "acts": jax.random.randint(keys[1], (BATCH, PLAN_HORIZON), 0, NUM_ACTIONS),
        "valid": jnp.ones((BATCH,), dtype=bool),
        "advantages": jnp.ones((BATCH,)),
        "timestep": jnp.full((BATCH,), 0.5),
    }


# ---------------------------------------------------------------------------
# Environment fixture (optional dependency)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def craftax_env(real_config: dict):
    """Two-env Craftax instance plus its dims.

    Skips when craftax or the Craftax_Baselines submodule is unavailable.
    """
    pytest.importorskip("craftax", reason="craftax not installed")
    if not (ROOT / "Craftax_Baselines" / "wrappers.py").exists():
        pytest.skip("Craftax_Baselines submodule not checked out (git submodule update --init)")

    from src.planners.env import make_env

    env, env_params = make_env(real_config, num_envs=2)
    return {
        "env": env,
        "env_params": env_params,
        "num_actions": int(env.action_space(env_params).n),
        "obs_dim": int(env.observation_space(env_params).shape[0]),
        "num_envs": 2,
    }
