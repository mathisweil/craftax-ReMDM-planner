"""All matplotlib figure generators for the ablation analysis suite.

``generate_all_plots`` is the single entry point: it accepts a results dict
(ablation_name -> {history, score}) and an output directory, and writes all
PNG files at 150 DPI.

Style conventions:
- Font sizes: title=13, axis labels=11, ticks=9, legend=9
- Grid: alpha=0.3, linestyle="--"
- Pretrained baseline shown as dashed horizontal line in comparison plots
- Group colours: Baseline=grey, A=blue, B=orange, C=green, D=red
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import AblationHistory

matplotlib.use("Agg")  # non-interactive backend for headless rendering
logger = logging.getLogger(__name__)

_DPI = 150
_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
}

_GROUP_COLORS = {
    "Baseline": "#9E9E9E",
    "A": "#2196F3",
    "B": "#FF9800",
    "C": "#4CAF50",
    "D": "#F44336",
}


def _group_color(name: str) -> str:
    """Return plot colour for an ablation by its group label.

    Args:
        name: Ablation name key (looked up in REGISTRY).

    Returns:
        Hex colour string.
    """
    spec = REGISTRY.get(name)
    group = spec.group if spec else "Baseline"
    return _GROUP_COLORS.get(group, "#9E9E9E")


def _ema(values: list[float], alpha: float = 0.3) -> list[float]:
    """Exponential moving average smoothing for a list of scalars.

    Args:
        values: Raw scalar values.
        alpha:  Smoothing factor (0=no smoothing, 1=no memory).

    Returns:
        Smoothed list of same length.
    """
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close it.

    Args:
        fig:  Matplotlib figure.
        path: Output file path (must have .png extension).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# 3.1 Per-ablation training curves
# ---------------------------------------------------------------------------


def plot_ablation_curves(
    name: str,
    history: AblationHistory,
    pretrained_score: float,
    output_dir: Path,
) -> None:
    """Generate a 2x3 grid of per-ablation training curves.

    Plots: eval score, training loss, env score, gradient norm, KL drift,
    grad alignment — all vs iteration.

    Args:
        name:             Ablation name.
        history:          Training history for this ablation.
        pretrained_score: Pretrained baseline score (dashed horizontal line).
        output_dir:       Directory to save the figure.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Training Curves: {name}", fontsize=14, fontweight="bold")
        color = _group_color(name)

        # Eval score
        ax = axes[0, 0]
        if history.eval_iters:
            ax.plot(history.eval_iters, history.eval_score, color=color, linewidth=1.5, label="eval")
        ax.axhline(pretrained_score, linestyle="--", color="black", alpha=0.6, label="pretrained")
        ax.set_title("Eval Score vs Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")
        ax.legend()

        # Training loss
        ax = axes[0, 1]
        if history.iters:
            ax.plot(history.iters, history.loss, color=color, alpha=0.4, linewidth=0.8, label="raw")
            ax.plot(history.iters, _ema(history.loss), color=color, linewidth=1.5, label="EMA")
        ax.set_title("Training Loss vs Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend()

        # Env score (online)
        ax = axes[0, 2]
        if history.env_score_iters:
            ax.plot(history.env_score_iters, history.env_score, color=color, linewidth=1.5)
        ax.axhline(pretrained_score, linestyle="--", color="black", alpha=0.6)
        ax.set_title("Online Env Score vs Iteration")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Score")

        # KL drift
        ax = axes[1, 0]
        if history.repr_drift_iters:
            ax.plot(history.repr_drift_iters, history.repr_drift_kl, color=color, linewidth=1.5, label="mean")
            if history.repr_drift_kl_low_t:
                ax.plot(history.repr_drift_iters, history.repr_drift_kl_low_t,
                        color=color, alpha=0.5, linestyle=":", label="low-t")
            if history.repr_drift_kl_high_t:
                ax.plot(history.repr_drift_iters, history.repr_drift_kl_high_t,
                        color=color, alpha=0.5, linestyle="-.", label="high-t")
        ax.set_title("KL Drift from Pretrained")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL Divergence")
        ax.legend()

        # Gradient alignment
        ax = axes[1, 1]
        if history.grad_align_iters:
            ax.plot(history.grad_align_iters, history.grad_align, color=color, linewidth=1.5)
            ax.axhline(0, linestyle="--", color="black", alpha=0.4)
        ax.set_title("Gradient Alignment (cos sim vs BC)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-1.1, 1.1)

        # Gradient norms
        ax = axes[1, 2]
        if history.grad_align_iters:
            ax.plot(history.grad_align_iters, history.rl_grad_norm, color=color, linewidth=1.5, label="RL grad")
            ax.plot(history.grad_align_iters, history.bc_grad_norm, color=color, linestyle="--", linewidth=1.2, label="BC grad")
        ax.set_title("Gradient Norms")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("L2 Norm")
        ax.legend()

        fig.tight_layout()
        _save(fig, output_dir / f"curves_{name}.png")


# ---------------------------------------------------------------------------
# 3.2 Aggregate comparison plots
# ---------------------------------------------------------------------------


def plot_final_score_comparison(
    results: dict[str, dict],
    pretrained_score: float,
    output_dir: Path,
) -> None:
    """Bar chart of final scores for all ablations, coloured by group.

    Args:
        results:          Dict mapping ablation name -> {"score": float, ...}.
        pretrained_score: Pretrained baseline score.
        output_dir:       Output directory.
    """
    names = ["pretrained"] + list(results.keys())
    scores = [pretrained_score] + [results[n]["score"] for n in results]
    colors = ["black"] + [_group_color(n) for n in results]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
        bars = ax.bar(range(len(names)), scores, color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(pretrained_score, linestyle="--", color="black", alpha=0.6, label="pretrained")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title("Final Score Comparison Across Ablations")
        ax.set_ylabel("Final Eval Score")

        # Group legend
        handles = [
            plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.8)
            for c in _GROUP_COLORS.values()
        ]
        ax.legend(handles, list(_GROUP_COLORS.keys()), title="Group", loc="upper right")
        fig.tight_layout()
        _save(fig, output_dir / "final_score_comparison.png")


def plot_eval_scores_over_training(
    results: dict[str, dict],
    pretrained_score: float,
    output_dir: Path,
) -> None:
    """All ablation eval scores overlaid on the same axes.

    Args:
        results:          Dict mapping name -> {"history": AblationHistory, ...}.
        pretrained_score: Pretrained baseline score (dashed horizontal).
        output_dir:       Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(pretrained_score, linestyle="--", color="black", linewidth=1.5, label="pretrained")

        for name, res in results.items():
            history: AblationHistory = res["history"]
            if history.eval_iters:
                ax.plot(history.eval_iters, history.eval_score,
                        color=_group_color(name), linewidth=1.2, alpha=0.8, label=name)

        ax.set_title("Eval Score vs Iteration — All Ablations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Eval Score")
        ax.legend(loc="lower right", ncol=2)
        fig.tight_layout()
        _save(fig, output_dir / "eval_scores_over_training.png")


def plot_score_delta(
    results: dict[str, dict],
    pretrained_score: float,
    baseline_rl_score: float,
    output_dir: Path,
) -> None:
    """Bar chart of score improvement over baseline-RL, sorted by delta.

    Args:
        results:            Dict mapping name -> {"score": float}.
        pretrained_score:   Pretrained baseline score.
        baseline_rl_score:  Baseline RL score (reference for delta).
        output_dir:         Output directory.
    """
    deltas = {n: results[n]["score"] - baseline_rl_score for n in results}
    sorted_items = sorted(deltas.items(), key=lambda x: x[1], reverse=True)
    names, vals = zip(*sorted_items) if sorted_items else ([], [])

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
        colors = [_group_color(n) for n in names]
        ax.bar(range(len(names)), vals, color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(0, linestyle="-", color="black", alpha=0.5, linewidth=1.0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_title("Score Improvement Over Baseline-RL")
        ax.set_ylabel("Delta Score vs Baseline-RL")
        fig.tight_layout()
        _save(fig, output_dir / "score_delta_over_baseline_rl.png")


# ---------------------------------------------------------------------------
# 3.3 Gradient analysis plots
# ---------------------------------------------------------------------------


def plot_gradient_alignment(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """All ablations' gradient alignment overlaid.

    Args:
        results:    Dict mapping name -> {"history": AblationHistory}.
        output_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(0, linestyle="--", color="black", alpha=0.4)

        for name, res in results.items():
            history: AblationHistory = res["history"]
            if history.grad_align_iters:
                ax.plot(history.grad_align_iters, history.grad_align,
                        color=_group_color(name), linewidth=1.2, alpha=0.8, label=name)

        ax.set_title("Gradient Alignment (RL vs BC cosine similarity) — All Ablations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-1.1, 1.1)
        ax.legend(loc="lower right", ncol=2)
        fig.tight_layout()
        _save(fig, output_dir / "gradient_alignment.png")


def plot_per_layer_gradient_heatmap(
    name: str,
    history: AblationHistory,
    output_dir: Path,
) -> None:
    """Heatmap of per-layer gradient norms over training iterations.

    Args:
        name:       Ablation name.
        history:    Training history.
        output_dir: Output directory.
    """
    if not history.per_layer_norms:
        return

    # Build 2D array: rows = layers, cols = iterations
    all_keys = sorted({k for d in history.per_layer_norms for k in d})
    matrix = np.array([[d.get(k, 0.0) for k in all_keys] for d in history.per_layer_norms]).T
    iters = history.per_layer_iters or list(range(len(history.per_layer_norms)))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, max(4, len(all_keys) * 0.5)))
        im = ax.imshow(matrix, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xticks(range(len(iters)))
        ax.set_xticklabels([str(i) for i in iters], rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(all_keys)))
        ax.set_yticklabels(all_keys, fontsize=7)
        ax.set_title(f"Per-Layer Gradient Norms: {name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Layer")
        plt.colorbar(im, ax=ax, label="L2 Norm")
        fig.tight_layout()
        _save(fig, output_dir / f"per_layer_grad_heatmap_{name}.png")


def plot_gradient_conflict_map(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Binary heatmap: for each (ablation, iteration), is grad_align < 0?

    Args:
        results:    Dict mapping name -> {"history": AblationHistory}.
        output_dir: Output directory.
    """
    names = [n for n, res in results.items() if res["history"].grad_align_iters]
    if not names:
        return

    max_len = max(len(results[n]["history"].grad_align) for n in names)
    matrix = np.ones((len(names), max_len)) * np.nan
    for i, n in enumerate(names):
        aligns = results[n]["history"].grad_align
        matrix[i, :len(aligns)] = [1.0 if a < 0 else 0.0 for a in aligns]

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, max(3, len(names) * 0.5)))
        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Gradient Conflict Map (red = cos_sim < 0 = conflicting)")
        ax.set_xlabel("Diagnostic step index")
        plt.colorbar(im, ax=ax, label="Conflict (1=yes, 0=no)")
        fig.tight_layout()
        _save(fig, output_dir / "gradient_conflict_map.png")


# ---------------------------------------------------------------------------
# 3.4 Representation drift
# ---------------------------------------------------------------------------


def plot_representation_drift(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """KL drift from pretrained for all ablations, overlaid.

    Args:
        results:    Dict mapping name -> {"history": AblationHistory}.
        output_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        for name, res in results.items():
            history: AblationHistory = res["history"]
            if history.repr_drift_iters:
                ax.plot(history.repr_drift_iters, history.repr_drift_kl,
                        color=_group_color(name), linewidth=1.2, alpha=0.8, label=name)
        ax.set_title("KL Divergence from Pretrained — All Ablations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("KL Divergence")
        ax.legend(loc="upper left", ncol=2)
        fig.tight_layout()
        _save(fig, output_dir / "representation_drift.png")


def plot_cka_similarity(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """CKA similarity with pretrained representations, all ablations overlaid.

    Args:
        results:    Dict mapping name -> {"history": AblationHistory}.
        output_dir: Output directory.
    """
    has_data = any(res["history"].cka_iters for res in results.values())
    if not has_data:
        return

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.axhline(1.0, linestyle="--", color="black", alpha=0.4, label="identical reps")
        for name, res in results.items():
            history: AblationHistory = res["history"]
            if history.cka_iters:
                ax.plot(history.cka_iters, history.cka_similarity,
                        color=_group_color(name), linewidth=1.2, alpha=0.8, label=name)
        ax.set_title("CKA Similarity with Pretrained Representations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("CKA (0=random, 1=identical)")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="lower left", ncol=2)
        fig.tight_layout()
        _save(fig, output_dir / "cka_similarity.png")


# ---------------------------------------------------------------------------
# 3.5 Timestep analysis
# ---------------------------------------------------------------------------


def plot_t_analysis(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """High-t vs low-t gradient norm ratio for all ablations.

    Args:
        results:    Dict mapping name -> {"history": AblationHistory}.
        output_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, res in results.items():
            history: AblationHistory = res["history"]
            if not history.t_analysis_iters:
                continue
            color = _group_color(name)
            axes[0].plot(history.t_analysis_iters, history.norm_high_t,
                         color=color, linewidth=1.2, alpha=0.8, label=name, linestyle="-")
            axes[0].plot(history.t_analysis_iters, history.norm_low_t,
                         color=color, linewidth=0.8, alpha=0.5, linestyle="--")
            axes[1].plot(history.t_analysis_iters, history.lowhigh_cos,
                         color=color, linewidth=1.2, alpha=0.8, label=name)

        axes[0].set_title("High-t (solid) vs Low-t (dashed) Gradient Norms")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("L2 Norm")
        axes[0].legend(loc="upper right", ncol=2)

        axes[1].axhline(0, linestyle="--", color="black", alpha=0.4)
        axes[1].set_title("Low-t / High-t Gradient Cosine Similarity")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_ylim(-1.1, 1.1)
        axes[1].legend(loc="lower right", ncol=2)

        fig.tight_layout()
        _save(fig, output_dir / "t_distribution_analysis.png")


def plot_t_bin_grad_norms(
    name: str,
    history: AblationHistory,
    output_dir: Path,
) -> None:
    """Per-t-bin gradient norms over training for a single ablation.

    Args:
        name:       Ablation name.
        history:    Training history.
        output_dir: Output directory.
    """
    if not history.t_bin_norms:
        return

    bins = sorted({k for d in history.t_bin_norms for k in d})
    iters = history.t_analysis_iters or list(range(len(history.t_bin_norms)))
    cmap = plt.cm.get_cmap("plasma", len(bins))

    with plt.rc_context(_STYLE):
        fig, ax = plt.subplots(figsize=(12, 5))
        for j, bin_key in enumerate(bins):
            vals = [d.get(bin_key, 0.0) for d in history.t_bin_norms]
            ax.plot(iters, vals, color=cmap(j), linewidth=1.2, alpha=0.8, label=bin_key)

        ax.set_title(f"Per-t-Bin Gradient Norms: {name}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("L2 Norm")
        ax.legend(loc="upper right", ncol=3, fontsize=7)
        fig.tight_layout()
        _save(fig, output_dir / f"t_bin_grad_norms_{name}.png")


# ---------------------------------------------------------------------------
# 3.6 Return / advantage distributions
# ---------------------------------------------------------------------------


def plot_return_distributions(
    results: dict[str, dict],
    output_dir: Path,
) -> None:
    """Win rate and effective batch size over training for all ablations.

    Args:
        results:    Dict mapping name -> {"history": AblationHistory}.
        output_dir: Output directory.
    """
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for name, res in results.items():
            history: AblationHistory = res["history"]
            color = _group_color(name)
            if history.win_rate and history.iters:
                axes[0].plot(history.iters, history.win_rate,
                             color=color, linewidth=1.2, alpha=0.8, label=name)
            if history.effective_batch_size and history.iters:
                axes[1].plot(history.iters, history.effective_batch_size,
                             color=color, linewidth=1.2, alpha=0.8, label=name)

        axes[0].set_title("Win Rate Over Training")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Win Rate")
        axes[0].legend(ncol=2)

        axes[1].set_title("Effective Batch Size Over Training")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Effective Batch Size")
        axes[1].legend(ncol=2)

        fig.tight_layout()
        _save(fig, output_dir / "win_rate_and_effective_batch_size.png")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_all_plots(
    results: dict[str, dict],
    pretrained_score: float,
    output_dir: Path,
) -> None:
    """Generate all analysis figures and save to output_dir/figures/.

    Args:
        results:          Dict mapping ablation_name -> {
                              "history": AblationHistory,
                              "score": float
                          }.
        pretrained_score: Pretrained model eval score (no fine-tuning).
        output_dir:       Root output directory; figures go in output_dir/figures/.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    baseline_rl_score = results.get("baseline_rl", {}).get("score", pretrained_score)

    logger.info("Generating per-ablation training curves...")
    for name, res in results.items():
        plot_ablation_curves(name, res["history"], pretrained_score, fig_dir)
        plot_per_layer_gradient_heatmap(name, res["history"], fig_dir)
        plot_t_bin_grad_norms(name, res["history"], fig_dir)

    logger.info("Generating aggregate comparison plots...")
    plot_final_score_comparison(results, pretrained_score, fig_dir)
    plot_eval_scores_over_training(results, pretrained_score, fig_dir)
    plot_score_delta(results, pretrained_score, baseline_rl_score, fig_dir)

    logger.info("Generating gradient analysis plots...")
    plot_gradient_alignment(results, fig_dir)
    plot_gradient_conflict_map(results, fig_dir)

    logger.info("Generating representation drift plots...")
    plot_representation_drift(results, fig_dir)
    plot_cka_similarity(results, fig_dir)

    logger.info("Generating timestep analysis plots...")
    plot_t_analysis(results, fig_dir)

    logger.info("Generating return / advantage plots...")
    plot_return_distributions(results, fig_dir)

    logger.info("All plots saved to %s", fig_dir)
