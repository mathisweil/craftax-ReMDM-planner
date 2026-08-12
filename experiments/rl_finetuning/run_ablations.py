"""Entry point for the ReMDM RL Fine-Tuning Ablation Suite.

Usage::

    # Run all ablations
    python experiments/rl_finetuning/run_ablations.py \\
        --config configs/defaults.yaml \\
        --ablations-config experiments/rl_finetuning/configs/ablations_default.yaml \\
        --checkpoint /path/to/pretrained \\
        --ppo-checkpoint /path/to/ppo \\
        --all

    # Run specific ablations
    python experiments/rl_finetuning/run_ablations.py \\
        --ablations kl_penalty ewc lora gradient_surgery \\
        --checkpoint /path/to/pretrained \\
        --ppo-checkpoint /path/to/ppo

    # Fast smoke test.  --fast overlays configs/ablations_fast.yaml on top of
    # whichever ablations config is in use; do not also pass it as
    # --ablations-config, which would make it the base instead of the overlay.
    python experiments/rl_finetuning/run_ablations.py \\
        --ablations baseline_rl kl_penalty --fast

    # Analysis only (load existing JSON results)
    python experiments/rl_finetuning/run_ablations.py \\
        --analyze-only \\
        --results-path experiments/rl_finetuning/outputs/run_xyz/results.json

    # Merge multi-GPU results
    python experiments/rl_finetuning/run_ablations.py \\
        --merge outputs/gpu0/results.json outputs/gpu1/results.json \\
        --output-dir outputs/merged/
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Suppress XLA/Triton compiler C++ logs before JAX import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax
import jax.numpy as jnp
import numpy as np
import orjson
import yaml

# Add project root to sys.path so src/ imports work regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CONFIG_DIR = _PROJECT_ROOT / "experiments" / "rl_finetuning" / "configs"
_DEFAULT_ABLATIONS_CONFIG = _CONFIG_DIR / "ablations_default.yaml"
_FAST_ABLATIONS_CONFIG = _CONFIG_DIR / "ablations_fast.yaml"

# Add Craftax_Baselines to sys.path so its internal modules (like logz) can be found
_CRAFTAX_BASELINES = _PROJECT_ROOT / "Craftax_Baselines"
if str(_CRAFTAX_BASELINES) not in sys.path:
    sys.path.insert(0, str(_CRAFTAX_BASELINES))

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import AblationHistory, run_ablation
from experiments.rl_finetuning.analysis.action_distribution import (
    run_action_distribution_analysis,
)
from experiments.rl_finetuning.analysis.plots import generate_all_plots
from experiments.rl_finetuning.analysis.report import generate_diagnosis_report
from experiments.rl_finetuning.analysis.tables import generate_summary_tables

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str | None) -> dict:
    """Load a YAML file into a dict, returning empty dict if path is None.

    Args:
        path: File path string or None.

    Returns:
        Parsed YAML dict.
    """
    if path is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_ablation_config(path: str | None) -> dict:
    """Load an ablations config, layered on ``ablations_default.yaml``.

    Machine-specific configs (e.g. ``ablations_final_craftax_ucl.yaml``) carry
    only the keys they change; everything else comes from the base. Exactly two
    layers, and a config never inherits from another config.

    Args:
        path: File path or None.

    Returns:
        Merged config dict.
    """
    if path is None:
        return {}

    resolved = Path(path).expanduser().resolve()
    raw = _load_yaml(str(resolved))

    if resolved == _DEFAULT_ABLATIONS_CONFIG.resolve():
        return raw

    merged = dict(_load_yaml(str(_DEFAULT_ABLATIONS_CONFIG)))
    merged.update(raw)
    logger.info(
        "Ablation config: %s -> %s",
        _DEFAULT_ABLATIONS_CONFIG.name,
        resolved.name,
    )
    return merged


def _merge_configs(*dicts: dict) -> dict:
    """Merge config dicts left-to-right (later dicts override earlier).

    Args:
        *dicts: Config dicts to merge.

    Returns:
        Merged dict.
    """
    merged: dict = {}
    for d in dicts:
        merged.update(d)
    return merged


def _to_upper(config: dict) -> dict:
    """Convert all keys to UPPERCASE for compatibility with src/ conventions.

    Args:
        config: Lowercase-keyed YAML config dict.

    Returns:
        UPPERCASE-keyed config dict.
    """
    return {k.upper(): v for k, v in config.items()}


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    p = argparse.ArgumentParser(
        description="ReMDM RL Fine-Tuning Ablation Suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Config files
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to main pipeline config (configs/defaults.yaml). Optional.",
    )
    p.add_argument(
        "--ablations-config",
        type=str,
        default=str(_DEFAULT_ABLATIONS_CONFIG),
        help=(
            "Ablations-specific config. Layered on top of "
            "ablations_default.yaml, which supplies every key it omits."
        ),
    )

    # Checkpoints
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the pretrained diffusion checkpoint (offline, DAgger or online).",
    )
    p.add_argument(
        "--ppo-checkpoint",
        type=str,
        default=None,
        help="Path to PPO checkpoint used for rollout collection.",
    )

    # Ablation selection
    p.add_argument("--all", action="store_true", help="Run all registered ablations.")
    p.add_argument(
        "--ablations",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Names of specific ablations to run. Run --list to see options.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List all registered ablation names and exit.",
    )

    # Special modes
    p.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Smoke-test overlay: apply configs/ablations_fast.yaml "
            "(max_iter=50, num_envs=16, eval_every=10 and 12 more) last."
        ),
    )
    p.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip training; load results.json and regenerate plots/tables/report.",
    )
    p.add_argument(
        "--results-path",
        type=str,
        default=None,
        help="Path to results.json for --analyze-only mode.",
    )
    p.add_argument(
        "--merge",
        nargs="+",
        metavar="PATH",
        help="Merge multiple results.json files from multi-GPU runs, "
        "recompute statistics, and regenerate analysis.",
    )

    # Output
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Root output directory. Defaults to experiments/rl_finetuning/outputs/{run_id}/.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run identifier. Defaults to run_{timestamp}.",
    )

    # Multi-seed
    p.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="Number of random seeds per ablation (overrides config).",
    )

    # W&B
    p.add_argument(
        "--use-wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable W&B logging.",
    )
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--wandb-entity", type=str, default=None)

    # Config overrides (passed directly to merged config)
    p.add_argument("--max-iter", type=int, default=None)
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)

    return p


def _apply_fast_overrides(config: dict) -> dict:
    """Overlay configs/ablations_fast.yaml on a config dict.

    The fast config is read raw, deliberately NOT through
    ``_load_ablation_config``: it is an overlay applied on top of whichever
    ablations config is in use, so it must contribute only its own keys and
    never drag ablations_default.yaml back over a machine-specific config.

    Args:
        config: UPPERCASE config dict.

    Returns:
        Config dict with fast overrides applied.
    """
    fast_cfg = _to_upper(_load_yaml(str(_FAST_ABLATIONS_CONFIG)))
    return {**config, **fast_cfg}


def _apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply CLI argument overrides to the merged config.

    Args:
        config: UPPERCASE config dict.
        args:   Parsed CLI arguments.

    Returns:
        Config dict with CLI overrides applied.
    """
    overrides = {
        "MAX_ITER": args.max_iter,
        "NUM_ENVS": args.num_envs,
        "BATCH_SIZE": args.batch_size,
        "EVAL_EVERY": args.eval_every,
        "LR": args.lr,
        "SEED": args.seed,
        "NUM_SEEDS": args.num_seeds,
        "USE_WANDB": args.use_wandb,
        "WANDB_PROJECT": args.wandb_project,
        "WANDB_ENTITY": args.wandb_entity,
        "CHECKPOINT_PATH": args.checkpoint,
        "PPO_CHECKPOINT_PATH": args.ppo_checkpoint,
    }
    return {
        k: v if v is not None else config.get(k)
        for k, v in {
            **config,
            **{k: v for k, v in overrides.items() if v is not None},
        }.items()
    }


def _history_finals(history: AblationHistory) -> dict:
    """C-002 (F-024/F-035): final logged value per history field for one seed.

    Captures numeric finals, dict finals (e.g. per-achievement rates) and
    numeric-list finals, so per-seed evaluation endpoints survive the merge
    instead of only the first seed's history.
    """
    finals: dict = {}
    for k, v in history.to_dict().items():
        if isinstance(v, list) and v:
            last = v[-1]
            if isinstance(last, (int, float, dict)) or (
                isinstance(last, list)
                and all(isinstance(x, (int, float)) for x in last)
            ):
                finals[k] = last
    return finals


def _results_to_json(
    results: dict[str, dict],
    pretrained_score: float,
    config: dict,
    pretrained_ach_rates: dict[str, float] | None = None,
) -> bytes:
    """Serialise all results to orjson bytes.

    Args:
        results:              Dict mapping name -> {"history": AblationHistory, "score": float}.
        pretrained_score:     Pretrained baseline score.
        config:               Merged config dict.
        pretrained_ach_rates: Per-achievement unlock rates for the pretrained baseline.

    Returns:
        UTF-8 JSON bytes.
    """
    serialisable = {
        "pretrained_score": pretrained_score,
        "pretrained_ach_rates": pretrained_ach_rates or {},
        "config": {
            k: v
            for k, v in config.items()
            if isinstance(v, (str, int, float, bool, type(None)))
        },
        "ablations": {
            name: {
                "score": res["score"],
                "score_std": res.get("score_std", 0.0),
                "all_scores": res.get("all_scores", [res["score"]]),
                "history": res["history"].to_dict(),
                # C-001 (D7) / C-002: seeds, wall clock and per-seed finals when present
                **{
                    k: res[k]
                    for k in ("base_seed", "seeds", "wall_clock_s", "per_seed_finals")
                    if k in res
                },
            }
            for name, res in results.items()
        },
    }
    return orjson.dumps(serialisable, option=orjson.OPT_INDENT_2)


def _results_from_json(path: str) -> tuple[dict, float, dict[str, float], dict]:
    """Load results from a JSON file produced by ``_results_to_json``.

    Args:
        path: Path to the JSON file.

    Returns:
        Tuple of (results_dict, pretrained_score, pretrained_ach_rates, config).
    """
    with open(path, "rb") as f:
        data = orjson.loads(f.read())

    pretrained_score = data["pretrained_score"]
    pretrained_ach_rates: dict[str, float] = data.get("pretrained_ach_rates", {})
    config = data.get("config", {})
    results = {}
    for name, res_data in data["ablations"].items():
        results[name] = {
            "score": res_data["score"],
            "score_std": res_data.get("score_std", 0.0),
            "all_scores": res_data.get("all_scores", [res_data["score"]]),
            "history": AblationHistory.from_dict(res_data["history"]),
        }
        for _k in (
            "base_seed",
            "seeds",
            "wall_clock_s",
            "per_seed_finals",
        ):  # C-001/C-002
            if _k in res_data:
                results[name][_k] = res_data[_k]
    return results, pretrained_score, pretrained_ach_rates, config


def _merge_result_files(
    paths: list[str],
) -> tuple[dict[str, dict], float, dict[str, float], dict]:
    """Merge multiple results.json files from multi-GPU runs.

    For ablations appearing in multiple files, per-seed scores are
    concatenated and mean/std recomputed over the union.  The first
    seed's history is kept for plots.

    Args:
        paths: List of paths to results.json files.

    Returns:
        Tuple of (merged_results, pretrained_score, pretrained_ach_rates, config).

    Raises:
        FileNotFoundError: If any path does not exist.
        ValueError:        If no valid results files are provided.
    """
    if not paths:
        raise ValueError("No results paths provided for --merge")

    merged_results: dict[str, dict] = {}
    pretrained_scores: list[float] = []
    merged_ach_rates: dict[str, float] = {}
    merged_config: dict = {}

    for p in paths:
        results, pt_score, ach_rates, config = _results_from_json(p)
        pretrained_scores.append(pt_score)
        if ach_rates:
            merged_ach_rates.update(ach_rates)
        if config:
            merged_config.update(config)

        for name, res in results.items():
            if name not in merged_results:
                merged_results[name] = {
                    "history": res["history"],
                    "all_scores": list(res["all_scores"]),
                    "score": res["score"],
                    "score_std": res.get("score_std", 0.0),
                    # C-001 (D7) / C-002: carry seed, wall-clock and per-seed final records
                    **{
                        k: list(res[k])
                        for k in ("seeds", "wall_clock_s", "per_seed_finals")
                        if k in res
                    },
                    **({"base_seed": res["base_seed"]} if "base_seed" in res else {}),
                }
            else:
                merged_results[name]["all_scores"].extend(res["all_scores"])
                for _k in ("seeds", "wall_clock_s", "per_seed_finals"):  # C-001/C-002
                    if _k in res:
                        merged_results[name].setdefault(_k, []).extend(res[_k])

    # Recompute mean/std over the union of seeds
    for res in merged_results.values():
        scores = res["all_scores"]
        res["score"] = float(np.mean(scores))
        res["score_std"] = float(np.std(scores))

    pretrained_score = float(np.mean(pretrained_scores))
    logger.info(
        "Merged %d files: %d ablations, pretrained=%.4f",
        len(paths),
        len(merged_results),
        pretrained_score,
    )
    return merged_results, pretrained_score, merged_ach_rates, merged_config


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ablation suite.

    Args:
        argv: Optional argument list (uses sys.argv if None).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # List ablations and exit
    if args.list:
        print("Registered ablations:")
        for name, spec in sorted(REGISTRY.items()):
            print(f"  [{spec.group}] {name:30s} — {spec.description}")
        return

    run_id = args.run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (_PROJECT_ROOT / "experiments" / "rl_finetuning" / "outputs" / run_id)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", output_dir)

    if args.merge:
        logger.info("Merging %d results files...", len(args.merge))
        results, pretrained_score, pretrained_ach_rates, config = _merge_result_files(
            args.merge
        )
        if args.ablations:
            results = {k: v for k, v in results.items() if k in args.ablations}
            logger.info(
                "Filtered to %d ablations: %s", len(results), list(results.keys())
            )
        # Write merged results
        ach_rates_arg = pretrained_ach_rates if pretrained_ach_rates else None
        merged_path = output_dir / "results.json"
        merged_path.write_bytes(
            _results_to_json(results, pretrained_score, config, ach_rates_arg)
        )
        logger.info("Wrote merged results to %s", merged_path)
        # Regenerate analysis
        tables = generate_summary_tables(
            results, pretrained_score, output_dir, ach_rates_arg
        )
        generate_all_plots(results, pretrained_score, output_dir, ach_rates_arg)
        generate_diagnosis_report(results, pretrained_score, tables, output_dir)
        logger.info("Merge complete. Outputs in %s", output_dir)
        return

    if args.analyze_only:
        if not args.results_path:
            parser.error("--analyze-only requires --results-path")
        logger.info("Loading results from %s", args.results_path)
        results, pretrained_score, pretrained_ach_rates, config = _results_from_json(
            args.results_path
        )
        logger.info("Loaded %d ablation results.", len(results))
        if args.ablations:
            results = {k: v for k, v in results.items() if k in args.ablations}
            logger.info(
                "Filtered to %d ablations: %s", len(results), list(results.keys())
            )
        ach_rates_arg = pretrained_ach_rates if pretrained_ach_rates else None
        tables = generate_summary_tables(
            results, pretrained_score, output_dir, ach_rates_arg
        )
        generate_all_plots(results, pretrained_score, output_dir, ach_rates_arg)
        generate_diagnosis_report(results, pretrained_score, tables, output_dir)
        logger.info("Analysis complete. Outputs in %s", output_dir)
        return

    # Precedence: defaults.yaml -> ablations_default.yaml -> machine config
    # -> ablations_fast.yaml (--fast only) -> CLI flags.  The first three are
    # layered here; `_load_ablation_config` puts ablations_default.yaml under
    # the named config, so a machine config need only carry its own deltas.
    main_cfg = _load_yaml(
        args.config or str(_PROJECT_ROOT / "configs" / "defaults.yaml")
    )
    abl_cfg = _load_ablation_config(args.ablations_config)
    merged = _to_upper(_merge_configs(main_cfg, abl_cfg))

    if args.fast:
        merged = _apply_fast_overrides(merged)

    merged = _apply_cli_overrides(merged, args)

    # Validate required paths
    if not merged.get("CHECKPOINT_PATH"):
        parser.error("--checkpoint is required for training mode.")
    if not merged.get("PPO_CHECKPOINT_PATH"):
        parser.error("--ppo-checkpoint is required for training mode.")

    # Enable the persistent XLA cache before the first compilation.  The suite
    # is the workload it helps most: run_ablation builds a fresh jax.jit
    # closure per (ablation, seed), so nothing is reused in-process, yet the
    # graph is identical across seeds of one ablation -- only the PRNG key
    # differs, and that is a runtime argument. Seeds 2..N are cache hits, as
    # are reruns and the per-GPU processes of the --merge workflow.
    # Off unless jax_compilation_cache_dir is set; see configs/defaults.yaml.
    from main import configure_compilation_cache

    configure_compilation_cache(merged)

    # Resolve wandb: artifact paths before any checkpoint loading
    from src.planners.model import resolve_checkpoint_path

    download_dir = merged.get("WANDB_DOWNLOAD_DIR")
    for key in ("CHECKPOINT_PATH", "PPO_CHECKPOINT_PATH"):
        val = merged.get(key)
        if val and isinstance(val, str) and val.startswith("wandb:"):
            merged[key] = resolve_checkpoint_path(val, download_dir)

    if args.all:
        selected_names = list(REGISTRY.keys())
    elif args.ablations:
        unknown = [n for n in args.ablations if n not in REGISTRY]
        if unknown:
            parser.error(f"Unknown ablation(s): {unknown}. Use --list to see options.")
        selected_names = args.ablations
    else:
        parser.error("Specify --all or --ablations NAME [NAME ...]")

    logger.info("Selected ablations (%d): %s", len(selected_names), selected_names)

    from src.diffusion.schedules import SCHEDULE_MAP
    from src.planners.env import make_env
    from src.planners.model import build_model, load_checkpoint, make_apply_fns
    from src.planners.ppo import PPOAgent, build_ppo_network, load_ppo_params

    env, env_params = make_env(merged, merged["NUM_ENVS"])
    num_actions = env.action_space(env_params).n
    obs_shape = env.observation_space(env_params).shape
    obs_dim = obs_shape[0]
    merged["NUM_ACTIONS"] = num_actions

    model_type = merged.get("PPO_MODEL_TYPE", "ppo_rnn")
    layer_size = merged.get("LAYER_SIZE", 512)
    ppo_net = build_ppo_network(model_type, num_actions, layer_size, merged)
    ppo_params = load_ppo_params(
        merged["PPO_CHECKPOINT_PATH"],
        ppo_net,
        model_type,
        merged["NUM_ENVS"],
        obs_shape,
        layer_size,
    )
    ppo = PPOAgent(ppo_net, ppo_params, model_type, layer_size)

    schedule_name = merged.get("DIFFUSION_SCHEDULE", "cosine")
    schedule_fn, schedule_deriv_fn = SCHEDULE_MAP[schedule_name]

    net = build_model(merged, num_actions)
    apply_eval, apply_train = make_apply_fns(net)

    # Load pretrained checkpoint
    seed = merged.get("SEED") or 0
    rng = jax.random.PRNGKey(seed)
    rng, ckpt_rng = jax.random.split(rng)
    pretrained_params = load_checkpoint(
        net,
        ckpt_rng,
        obs_dim,
        merged["PLAN_HORIZON"],
        merged["CHECKPOINT_PATH"],
    )

    # Evaluate pretrained baseline
    logger.info("Evaluating pretrained model (no fine-tuning)...")
    from experiments.rl_finetuning.ablations.training import build_eval_fn

    merged_with_actions = {**merged, "NUM_ACTIONS": num_actions}
    eval_fn = build_eval_fn(env, env_params, apply_eval, merged_with_actions)
    rng, eval_rng = jax.random.split(rng)
    pretrained_info = eval_fn(pretrained_params, eval_rng)
    pretrained_score = float(
        pretrained_info.get("returned_episode_returns", jnp.array(0.0))
    )
    logger.info("Pretrained baseline score: %.4f", pretrained_score)
    # Extract per-achievement unlock rates from pretrained eval (Craftax reports percentages).
    pretrained_ach_rates: dict[str, float] = {
        k: float(v) / 100.0
        for k, v in pretrained_info.items()
        if "achievement" in k.lower()
    }
    logger.info("Pretrained achievements tracked: %d", len(pretrained_ach_rates))

    wandb_run = None
    if merged.get("USE_WANDB"):
        try:
            import wandb

            wandb_run = wandb.init(
                project=merged.get("WANDB_PROJECT", "remdm-craftax-ablations"),
                entity=merged.get("WANDB_ENTITY"),
                name=run_id,
                config={
                    k: v
                    for k, v in merged.items()
                    if isinstance(v, (str, int, float, bool))
                },
                tags=["ablations"] + selected_names,
            )
        except ImportError:
            logger.warning("wandb not installed; skipping W&B logging.")

    num_seeds = merged.get("NUM_SEEDS", 1)
    results: dict[str, dict] = {}

    for abl_name in selected_names:
        spec = REGISTRY[abl_name]
        seed_scores: list[float] = []
        seed_histories: list[AblationHistory] = []
        seeds_used: list[int] = []
        seed_times: list[float] = []
        first_seed_params: Any = None

        for seed_idx in range(num_seeds):
            abl_seed = (
                seed + seed_idx
            )  # C-001 (D7): literal seed set base+idx (default 0, 1, 2)
            abl_rng = jax.random.PRNGKey(abl_seed)
            seeds_used.append(abl_seed)
            logger.info("Running %s (seed %d/%d)...", abl_name, seed_idx + 1, num_seeds)

            _t0 = time.monotonic()
            history, final_score, final_params = run_ablation(
                spec=spec,
                config=merged,
                pretrained_params=pretrained_params,
                apply_train=apply_train,
                apply_eval=apply_eval,
                env=env,
                env_params=env_params,
                ppo=ppo,
                schedule_fn=schedule_fn,
                schedule_deriv_fn=schedule_deriv_fn,
                num_actions=num_actions,
                obs_dim=obs_dim,
                rng=abl_rng,
                wandb_run=wandb_run,
                output_dir=output_dir,
            )
            seed_times.append(
                round(time.monotonic() - _t0, 1)
            )  # C-001: per-seed wall clock
            seed_scores.append(final_score)
            seed_histories.append(history)
            if seed_idx == 0:
                first_seed_params = final_params

        # Aggregate over seeds (use first seed's history for plots; report mean score)
        mean_score = float(np.mean(seed_scores))
        std_score = float(np.std(seed_scores))
        logger.info(
            "%s: score = %.4f ± %.4f (seeds=%d)",
            abl_name,
            mean_score,
            std_score,
            num_seeds,
        )

        results[abl_name] = {
            "history": seed_histories[0],  # primary history for plots
            "score": mean_score,
            "score_std": std_score,
            "all_scores": seed_scores,
            "base_seed": seed,  # C-001 (D7)
            "seeds": seeds_used,  # C-001 (D7)
            "wall_clock_s": seed_times,  # C-001: per-seed wall clock
            "per_seed_finals": [_history_finals(h) for h in seed_histories],  # C-002
            "final_params": first_seed_params,  # in-memory only, not serialised
        }

        # Written after every ablation so a crash mid-run doesn't lose
        # already-completed results.  The file is valid JSON at all times
        # and is directly loadable by --analyze-only --results-path.
        results_path = output_dir / "results.json"
        results_path.write_bytes(
            _results_to_json(results, pretrained_score, merged, pretrained_ach_rates)
        )
        logger.info(
            "Saved partial results (%d/%d ablations) to %s",
            len(results),
            len(selected_names),
            results_path,
        )

    # (params not saved here to keep the results JSON small;
    #  enable via save_checkpoints in the ablations config if needed)

    logger.info("Running action distribution analysis...")
    rng, ad_rng = jax.random.split(rng)
    run_action_distribution_analysis(
        results,
        pretrained_params,
        apply_eval,
        env,
        env_params,
        merged,
        ad_rng,
        output_dir,
    )

    logger.info("Generating plots and tables...")
    ach_rates_arg = pretrained_ach_rates if pretrained_ach_rates else None
    tables = generate_summary_tables(
        results, pretrained_score, output_dir, ach_rates_arg
    )
    generate_all_plots(results, pretrained_score, output_dir, ach_rates_arg)
    report_path = generate_diagnosis_report(
        results, pretrained_score, tables, output_dir
    )

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("=" * 60)
    logger.info("Ablation suite complete.")
    logger.info("  Results:  %s", results_path)
    logger.info("  Report:   %s", report_path)
    logger.info("  Outputs:  %s", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
