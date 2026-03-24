"""Standalone report generation script for ReMDM ablation results.

Loads saved results from a results directory, regenerates all figures and
tables deterministically, and prints a human-readable verdict per method.

Usage
-----
    python notebooks/generate_report.py --results_dir notebooks/ablation_results

The results directory must contain:
    results.json     — serialised results_dict from the ablation notebook.
    histories.json   — serialised all_histories dict (per-method training logs).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow imports from the project root.
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import orjson

from src.ablations.visualisations import (
    plot_training_dynamics,
    plot_summary_bars,
    plot_scatter_diagnostics,
    plot_t_bin_heatmap,
    plot_failure_mode_map,
    plot_achievement_bars,
    plot_per_method_deep_dive,
    make_summary_table,
    make_correlation_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    """Load a JSON file using orjson.

    Args:
        path: Absolute or relative path to a JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    resolved = Path(path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Results file not found: {resolved}")
    with open(resolved, "rb") as fh:
        return orjson.loads(fh.read())


def _print_verdict(results_dict: dict) -> None:
    """Print a human-readable summary verdict for each method.

    Args:
        results_dict: ``{method_name: result_dict}``.
    """
    print("\n" + "=" * 60)
    print("ABLATION STUDY — VERDICT SUMMARY")
    print("=" * 60)

    methods_sorted = sorted(
        results_dict.items(),
        key=lambda kv: kv[1].get("final_score", 0.0),
        reverse=True,
    )

    for method, res in methods_sorted:
        score = res.get("final_score", float("nan"))
        pre = res.get("pretrained_score", float("nan"))
        delta = score - pre if (score == score and pre == pre) else float("nan")
        drift = res.get("final_drift", float("nan"))
        align = res.get("final_grad_align", float("nan"))
        std = res.get("score_std", float("nan"))

        # Simple rule-based verdict.
        if delta == delta and delta > pre * 0.05:
            verdict = "IMPROVED"
        elif drift == drift and drift > 5.0 and align == align and align < 0.0:
            verdict = "CATASTROPHIC_FORGETTING"
        elif align == align and align < -0.2:
            verdict = "GRADIENT_CONFLICT"
        else:
            verdict = "NO_CHANGE"

        std_str = f"±{std:.4f}" if std == std else ""
        delta_str = f"{delta:+.4f}" if delta == delta else "N/A"
        print(
            f"  {method:<22}  score={score:.4f}{std_str:<12}  "
            f"Δ={delta_str:<10}  drift={drift:.3f}  "
            f"align={align:.3f}  [{verdict}]"
        )

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(results_dir: str) -> None:
    """Load results, regenerate all figures and tables, and print verdicts.

    Args:
        results_dir: Path to the directory containing ``results.json`` and
                     ``histories.json``.
    """
    results_dir = str(Path(results_dir).resolve())
    figures_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Load serialised data.
    results_path = os.path.join(results_dir, "results.json")
    histories_path = os.path.join(results_dir, "histories.json")

    logger.info("Loading results from: %s", results_path)
    results_dict = _load_json(results_path)

    logger.info("Loading histories from: %s", histories_path)
    all_histories = _load_json(histories_path)

    pretrained_score = float(results_dict.get("_pretrained_score", 0.0))
    # Remove meta-keys that are not method results.
    method_results = {
        k: v for k, v in results_dict.items()
        if not k.startswith("_") and isinstance(v, dict)
    }

    logger.info("Found %d methods: %s", len(method_results), list(method_results.keys()))

    # ── Figures ──────────────────────────────────────────────────────────────

    logger.info("Generating training dynamics panel…")
    plot_training_dynamics(all_histories, pretrained_score, save_dir=figures_dir)

    logger.info("Generating summary bars…")
    plot_summary_bars(method_results, save_dir=figures_dir)

    logger.info("Generating scatter diagnostics…")
    plot_scatter_diagnostics(method_results, save_dir=figures_dir)

    logger.info("Generating per-t-bin heatmaps…")
    plot_t_bin_heatmap(all_histories, save_dir=figures_dir)

    logger.info("Generating failure mode map…")
    plot_failure_mode_map(method_results, save_dir=figures_dir)

    logger.info("Generating achievement bars…")
    plot_achievement_bars(all_histories, save_dir=figures_dir)

    # Deep-dive panels for all methods.
    for method, history in all_histories.items():
        logger.info("Generating deep-dive for: %s", method)
        plot_per_method_deep_dive(method, history, save_dir=figures_dir)

    # ── Tables ───────────────────────────────────────────────────────────────

    logger.info("Building summary table…")
    latex_str = make_summary_table(
        method_results, save_dir=results_dir, latex=True
    )

    logger.info("Building correlation table…")
    corr_df = make_correlation_table(method_results)
    if len(corr_df) > 0:
        corr_path = os.path.join(tables_dir, "correlations.csv")
        corr_df.write_csv(corr_path)
        logger.info("Saved correlation table: %s", corr_path)
        print("\nTop diagnostic correlates with final eval score:")
        print(corr_df.head(10))

    # ── Verdict ──────────────────────────────────────────────────────────────
    _print_verdict(method_results)

    logger.info(
        "Report complete. Figures saved to: %s  Tables saved to: %s",
        figures_dir, tables_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate all ablation figures and tables from saved results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="notebooks/ablation_results",
        help="Path to the directory containing results.json and histories.json.",
    )
    args = parser.parse_args()
    main(args.results_dir)
