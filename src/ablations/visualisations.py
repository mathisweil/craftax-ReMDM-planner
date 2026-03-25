"""Plotting functions and summary tables for ablation experiment results.

All plotting functions:
- Use ``matplotlib`` only (no seaborn dependency).
- Accept an optional ``save_dir`` string; when provided, the figure is saved
  as ``{save_dir}/{name}.pdf`` (vector) and ``{save_dir}/{name}.png`` (raster).
- Return the ``Figure`` object so notebooks can display it inline.

Summary table functions use ``polars`` for all DataFrame operations.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import polars as pl
import scipy.stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

PALETTE: dict[str, str] = {
    "baseline":         "#4C72B0",
    "kl_penalty":       "#DD8452",
    "frozen_backbone":  "#55A868",
    "bc_wins":          "#C44E52",
    "low_t":            "#8172B2",
    "ewc":              "#937860",
    "mixed_replay":     "#DA8BC3",
    "t_curriculum":     "#8C8C8C",
    "entropy_reg":      "#CCB974",
    "token_advantage":  "#64B5CD",
    "trust_region":     "#1B7837",
    "pretrained":       "#000000",
    "_default":         "#999999",
}

# Craftax Classic achievement names (17 total).
CRAFTAX_ACHIEVEMENTS: list[str] = [
    "collect_wood",
    "place_table",
    "eat_plant",
    "defeat_zombie",
    "collect_sapling",
    "collect_drink",
    "make_wood_pickaxe",
    "make_wood_sword",
    "place_plant",
    "defeat_skeleton",
    "make_stone_pickaxe",
    "make_stone_sword",
    "wake_up",
    "place_stone",
    "place_furnace",
    "collect_coal",
    "make_iron_sword",
    "make_iron_pickaxe",
    "eat_cow",
    "collect_stone",
    "collect_iron",
    "collect_diamond",
]


def _method_color(name: str) -> str:
    """Return a consistent colour for a method name.

    Args:
        name: Method identifier string.

    Returns:
        Hex colour string.
    """
    for key in PALETTE:
        if key in name.lower():
            return PALETTE[key]
    return PALETTE["_default"]


def _save_figure(fig: plt.Figure, name: str, save_dir: Optional[str]) -> None:
    """Save a figure to both PDF and PNG formats.

    Args:
        fig:      Matplotlib figure object.
        name:     Base filename (without extension).
        save_dir: Directory path; skipped if ``None``.
    """
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(save_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
        logger.info("Saved figure: %s", path)


def _ema(values: list[float], alpha: float = 0.1) -> list[float]:
    """Exponential moving average smoothing.

    Args:
        values: Raw scalar series.
        alpha:  Smoothing factor (0 = no smoothing, 1 = no memory).

    Returns:
        Smoothed list of the same length.
    """
    smoothed = []
    s = values[0] if values else 0.0
    for v in values:
        s = alpha * v + (1.0 - alpha) * s
        smoothed.append(s)
    return smoothed


# ---------------------------------------------------------------------------
# Training dynamics panel
# ---------------------------------------------------------------------------

def plot_training_dynamics(
    all_histories: dict[str, dict],
    baseline_score: float,
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Plot training dynamics for all methods on shared axes.

    Creates a 2×2 panel:
    - (0, 0) Eval score over gradient steps  (with pretrained baseline hline)
    - (0, 1) Training loss over steps         (EMA smoothed)
    - (1, 0) Gradient alignment over steps
    - (1, 1) Representation drift (L2) over steps

    Args:
        all_histories: ``{method_name: history_dict}`` where each history dict
                       contains lists keyed by metric name.
        baseline_score: Pretrained model eval score (horizontal reference line).
        save_dir:      Optional directory for saving the figure.

    Returns:
        Matplotlib ``Figure`` object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    fig.suptitle("Training Dynamics — All Methods", fontsize=13, fontweight="bold")

    metric_axes = {
        "eval_score":    (axes[0, 0], "Eval Score", True),
        "loss":          (axes[0, 1], "Training Loss (EMA)", False),
        "grad_align":    (axes[1, 0], "Gradient Alignment", False),
        "repr_drift":    (axes[1, 1], "Repr. Drift (L2)", False),
    }

    for method, history in all_histories.items():
        color = _method_color(method)
        step_xs = history.get("step", [])          # per gradient step
        eval_xs = history.get("eval_step", [])     # per eval interval
        for key, (ax, ylabel, add_baseline) in metric_axes.items():
            if key not in history:
                continue
            values = [v for v in history[key] if v is not None]
            if not values:
                continue
            # loss is recorded every step; all other metrics at eval intervals.
            if key == "loss":
                xs = step_xs if len(step_xs) == len(history[key]) else list(range(len(values)))
                smoothed = _ema(values)
            else:
                xs = eval_xs if len(eval_xs) == len([v for v in history[key] if v is not None]) else list(range(len(values)))
                smoothed = values
            ax.plot(xs, smoothed, label=method, color=color, linewidth=1.5)

    for key, (ax, ylabel, add_baseline) in metric_axes.items():
        if add_baseline:
            ax.axhline(
                baseline_score, color=PALETTE["pretrained"],
                linestyle="--", linewidth=1.2, label="pretrained",
            )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Gradient step", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    _save_figure(fig, "training_dynamics", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Summary comparison figures
# ---------------------------------------------------------------------------

def plot_summary_bars(
    results_dict: dict[str, dict],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Bar charts comparing final eval score, drift, and gradient alignment.

    Creates a 1×3 panel with one bar chart per metric.  Methods are sorted by
    final eval score descending.

    Args:
        results_dict: ``{method_name: {'final_score': float, 'final_drift': float,
                       'final_grad_align': float, 'score_std': float, ...}}``.
        save_dir:     Optional save directory.

    Returns:
        Matplotlib ``Figure`` object.
    """
    methods = sorted(
        results_dict.keys(),
        key=lambda m: results_dict[m].get("final_score", 0.0),
        reverse=True,
    )
    colors = [_method_color(m) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Final-State Summary", fontsize=13, fontweight="bold")

    metrics = [
        ("final_score",     "score_std",     "Final Eval Score"),
        ("final_drift",     None,            "Repr. Drift (L2)"),
        ("final_grad_align", None,           "Gradient Alignment"),
    ]

    for ax, (metric, std_key, title) in zip(axes, metrics):
        vals = [results_dict[m].get(metric, 0.0) for m in methods]
        errs = (
            [results_dict[m].get(std_key, 0.0) for m in methods]
            if std_key else None
        )
        x = np.arange(len(methods))
        ax.bar(x, vals, color=colors, alpha=0.85, yerr=errs, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, "summary_bars", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Scatter diagnostic plots
# ---------------------------------------------------------------------------

def plot_scatter_diagnostics(
    results_dict: dict[str, dict],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Scatter plots: gradient alignment vs score and drift vs score.

    Each method is plotted as a labelled dot, colour-coded by ablation family.

    Args:
        results_dict: ``{method_name: {'final_score': float,
                         'final_drift': float, 'final_grad_align': float}}``.
        save_dir:     Optional save directory.

    Returns:
        Matplotlib ``Figure`` object.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Diagnostic vs. Eval Score", fontsize=13, fontweight="bold")

    pairs = [
        ("final_grad_align", "final_score",
         "Gradient Alignment", "Final Eval Score"),
        ("final_drift", "final_score",
         "Repr. Drift (L2)", "Final Eval Score"),
    ]

    for ax, (xkey, ykey, xlabel, ylabel) in zip(axes, pairs):
        for method, result in results_dict.items():
            x = result.get(xkey, 0.0)
            y = result.get(ykey, 0.0)
            color = _method_color(method)
            ax.scatter(x, y, color=color, s=80, zorder=3)
            ax.annotate(
                method, (x, y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7,
            )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, "scatter_diagnostics", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Per-method deep-dive
# ---------------------------------------------------------------------------

def plot_per_method_deep_dive(
    name: str,
    history: dict[str, Any],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Per-method deep-dive panel with entropy, return distribution, and gradient norms.

    Creates a 2×2 panel:
    - (0, 0) Token entropy over denoising steps (line plot over training)
    - (0, 1) Return distribution at iter 0 vs. final (overlaid histograms)
    - (1, 0) Per-layer gradient norm over training (stacked line plot)
    - (1, 1) Output KL drift over training

    Args:
        name:     Method name (used in title and file name).
        history:  History dict from ``run_ablation_v2``.
        save_dir: Optional save directory.

    Returns:
        Matplotlib ``Figure`` object.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(f"Deep Dive — {name}", fontsize=12, fontweight="bold")
    color = _method_color(name)

    steps = history.get("step", [])

    # Token entropy over training.
    ax = axes[0, 0]
    if "token_entropy" in history:
        ax.plot(steps, history["token_entropy"], color=color, linewidth=1.5)
        ax.set_title("Token Entropy", fontsize=10)
        ax.set_ylabel("Entropy (nats)", fontsize=9)
        ax.set_xlabel("Gradient step", fontsize=9)
        ax.grid(True, alpha=0.3)

    # Return distribution at start vs. end.
    ax = axes[0, 1]
    if "return_dist_start" in history and "return_dist_final" in history:
        ax.hist(
            history["return_dist_start"], bins=20, alpha=0.5,
            label="iter 0", color=PALETTE["pretrained"],
        )
        ax.hist(
            history["return_dist_final"], bins=20, alpha=0.5,
            label="final", color=color,
        )
        ax.set_title("Return Distribution", fontsize=10)
        ax.set_xlabel("Episode return", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Per-layer gradient norm heatmap.
    ax = axes[1, 0]
    if "per_layer_grad_norm" in history and history["per_layer_grad_norm"]:
        layer_data = history["per_layer_grad_norm"]  # list of dicts
        layer_names = [k for k in layer_data[0] if k != "__total__"]
        matrix = np.array(
            [[d.get(ln, 0.0) for ln in layer_names] for d in layer_data]
        ).T  # [num_layers, T]
        im = ax.imshow(
            matrix, aspect="auto", cmap="viridis", origin="lower",
            interpolation="nearest",
        )
        ax.set_yticks(np.arange(len(layer_names)))
        ax.set_yticklabels(layer_names, fontsize=6)
        ax.set_title("Per-Layer Grad Norm", fontsize=10)
        ax.set_xlabel("Gradient step", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Output KL drift.
    ax = axes[1, 1]
    if "output_kl" in history:
        ax.plot(steps, history["output_kl"], color=color, linewidth=1.5)
        ax.set_title("Output KL from Pretrained", fontsize=10)
        ax.set_ylabel("KL (nats)", fontsize=9)
        ax.set_xlabel("Gradient step", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save_figure(fig, f"deep_dive_{name}", save_dir)
    return fig


# ---------------------------------------------------------------------------
# T-bin ELBO heatmap
# ---------------------------------------------------------------------------

def plot_t_bin_heatmap(
    all_histories: dict[str, dict],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of per-t-bin ELBO loss over training, one sub-plot per method.

    Args:
        all_histories: ``{method_name: history_dict}`` where history dict
                       contains ``'per_t_loss'`` key with shape ``[T, n_bins]``.
        save_dir:      Optional save directory.

    Returns:
        Matplotlib ``Figure`` object.
    """
    n_methods = len(all_histories)
    ncols = min(n_methods, 3)
    nrows = (n_methods + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    fig.suptitle("Per-t-Bin ELBO Heatmap", fontsize=12, fontweight="bold")

    for idx, (method, history) in enumerate(all_histories.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        if "per_t_loss" not in history:
            ax.set_visible(False)
            continue
        mat = np.array(history["per_t_loss"])  # [T, n_bins]
        im = ax.imshow(
            mat.T, aspect="auto", cmap="RdYlBu_r",
            origin="lower", interpolation="nearest",
        )
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("Gradient step", fontsize=8)
        ax.set_ylabel("t-bin", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, label="loss")

    # Hide unused axes.
    for idx in range(n_methods, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    _save_figure(fig, "t_bin_heatmap", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Failure mode taxonomy
# ---------------------------------------------------------------------------

def plot_failure_mode_map(
    results_dict: dict[str, dict],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """2×2 failure-mode taxonomy: drift (x) vs. gradient alignment (y).

    Each method is plotted as a labelled dot.  Quadrant labels:
    - High drift, low alignment  → Catastrophic forgetting
    - High drift, high alignment → Mode collapse
    - Low drift, low alignment   → Gradient conflict
    - Low drift, high alignment  → No learning (or success)

    The threshold for "high/low" is the median of each axis across all methods.

    Args:
        results_dict: ``{method_name: {'final_score': float,
                         'final_drift': float, 'final_grad_align': float}}``.
        save_dir:     Optional save directory.

    Returns:
        Matplotlib ``Figure`` object.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    ax.set_title("Failure Mode Taxonomy", fontsize=12, fontweight="bold")

    drifts = [r.get("final_drift", 0.0) for r in results_dict.values()]
    aligns = [r.get("final_grad_align", 0.0) for r in results_dict.values()]
    scores = [r.get("final_score", 0.0) for r in results_dict.values()]

    drift_med = float(np.median(drifts))
    align_med = float(np.median(aligns))

    # Quadrant annotation boxes.
    for x_side, y_side, label, alpha in [
        (True,  False, "Catastrophic\nForgetting",  0.08),
        (True,  True,  "Mode\nCollapse",             0.08),
        (False, False, "Gradient\nConflict",         0.08),
        (False, True,  "No Learning\n/ Success",     0.08),
    ]:
        xc = drift_med * (1.5 if x_side else 0.5)
        yc = align_med * (1.5 if y_side else 0.5)
        ax.text(
            xc, yc, label, ha="center", va="center",
            fontsize=9, color="gray", alpha=0.6,
        )

    # Threshold lines.
    ax.axvline(drift_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(align_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    norm_scores = (
        (np.array(scores) - np.min(scores))
        / (np.ptp(scores) + 1e-8)
    )
    cmap = cm.get_cmap("RdYlGn")

    for method, drift, align, norm_s in zip(
        results_dict.keys(), drifts, aligns, norm_scores
    ):
        color = cmap(float(norm_s))
        ax.scatter(drift, align, s=120, color=color, zorder=3, edgecolors="k", linewidths=0.5)
        ax.annotate(
            method, (drift, align),
            textcoords="offset points", xytext=(6, 4), fontsize=7,
        )

    ax.set_xlabel("Representation Drift (L2)", fontsize=10)
    ax.set_ylabel("Gradient Alignment", fontsize=10)
    ax.grid(True, alpha=0.2)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Normalised Eval Score", fraction=0.03)

    fig.tight_layout()
    _save_figure(fig, "failure_mode_map", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Achievement bar chart
# ---------------------------------------------------------------------------

def plot_achievement_bars(
    all_histories: dict[str, dict],
    save_dir: Optional[str] = None,
) -> plt.Figure:
    """Per-achievement unlock rate bar chart for all methods.

    Args:
        all_histories: ``{method_name: history_dict}`` where history contains
                       ``'achievements'`` as a dict mapping achievement name
                       to final unlock rate.
        save_dir:     Optional save directory.

    Returns:
        Matplotlib ``Figure`` object.
    """
    methods = list(all_histories.keys())
    n_ach = len(CRAFTAX_ACHIEVEMENTS)
    x = np.arange(n_ach)
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 5))
    ax.set_title("Per-Achievement Unlock Rate", fontsize=12, fontweight="bold")

    for i, method in enumerate(methods):
        history = all_histories[method]
        ach_dict = history.get("achievements", {})
        rates = [float(ach_dict.get(name, 0.0)) for name in CRAFTAX_ACHIEVEMENTS]
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(
            x + offset, rates, width * 0.9,
            label=method, color=_method_color(method), alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CRAFTAX_ACHIEVEMENTS, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Unlock Rate", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save_figure(fig, "achievement_bars", save_dir)
    return fig


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def make_summary_table(
    results_dict: dict[str, dict],
    save_dir: Optional[str] = None,
    latex: bool = True,
) -> str:
    """Build a summary results table (LaTeX-ready).

    Columns: method, final_score, delta_from_pretrained, grad_align, drift, verdict.

    Args:
        results_dict: ``{method_name: {'final_score', 'pretrained_score',
                         'score_std', 'final_grad_align', 'final_drift'}}``.
        save_dir:     Optional directory; saves ``.csv`` and ``.tex`` if provided.
        latex:        If ``True``, also format as LaTeX tabular string.

    Returns:
        LaTeX ``tabular`` string (or plain CSV string if ``latex=False``).
    """
    def _verdict(row: dict) -> str:
        score = row.get("final_score", 0.0)
        pre = row.get("pretrained_score", 0.0)
        drift = row.get("final_drift", 0.0)
        align = row.get("final_grad_align", 0.0)
        if score > pre * 1.05:
            return "improved"
        if drift > 5.0 and align < 0.0:
            return "catastrophic_forgetting"
        if align < -0.2:
            return "gradient_conflict"
        return "no_change"

    rows = []
    for method, res in results_dict.items():
        pre = res.get("pretrained_score", 0.0)
        score = res.get("final_score", 0.0)
        std = res.get("score_std", float("nan"))
        delta = score - pre
        rows.append(
            {
                "method": method,
                "final_score": round(score, 4),
                "score_std": round(std, 4),
                "delta": round(delta, 4),
                "grad_align": round(res.get("final_grad_align", float("nan")), 4),
                "drift": round(res.get("final_drift", float("nan")), 4),
                "verdict": _verdict(res),
            }
        )

    df = pl.DataFrame(rows).sort("final_score", descending=True)

    if save_dir:
        os.makedirs(os.path.join(save_dir, "tables"), exist_ok=True)
        csv_path = os.path.join(save_dir, "tables", "summary.csv")
        df.write_csv(csv_path)
        logger.info("Saved summary CSV: %s", csv_path)

    if not latex:
        return df.write_csv()

    # Build LaTeX tabular.
    header = (
        r"\begin{tabular}{lrrrrrl}" + "\n"
        r"\toprule" + "\n"
        r"Method & Score & $\pm$Std & $\Delta$Pretrained & "
        r"Grad.\ Align & Drift & Verdict \\" + "\n"
        r"\midrule" + "\n"
    )
    body_lines = []
    for row in df.iter_rows(named=True):
        std_str = f"{row['score_std']:.4f}" if not np.isnan(row["score_std"]) else "---"
        body_lines.append(
            f"{row['method']} & {row['final_score']:.4f} & {std_str} & "
            f"{row['delta']:.4f} & {row['grad_align']:.4f} & "
            f"{row['drift']:.4f} & {row['verdict']} \\\\"
        )
    footer = r"\bottomrule" + "\n" + r"\end{tabular}"
    latex_str = header + "\n".join(body_lines) + "\n" + footer

    if save_dir:
        tex_path = os.path.join(save_dir, "tables", "summary.tex")
        with open(tex_path, "w") as fh:
            fh.write(latex_str)
        logger.info("Saved summary LaTeX: %s", tex_path)

    return latex_str


# ---------------------------------------------------------------------------
# Correlation table
# ---------------------------------------------------------------------------

def make_correlation_table(
    results_dict: dict[str, dict],
    target_key: str = "final_score",
) -> pl.DataFrame:
    """Pearson and Spearman correlation of each diagnostic vs. the target metric.

    Computes correlations across all methods and returns a Polars DataFrame
    identifying which diagnostics best predict final eval performance.

    Args:
        results_dict: ``{method_name: result_dict}`` where each result dict
                      contains numeric values for all diagnostics.
        target_key:   Column to correlate against (default ``'final_score'``).

    Returns:
        Polars ``DataFrame`` with columns:
        ``['diagnostic', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p']``,
        sorted by absolute Pearson correlation descending.
    """
    if not results_dict:
        return pl.DataFrame()

    # Collect all numeric keys across all result dicts (excluding target).
    all_keys: set[str] = set()
    for res in results_dict.values():
        for k, v in res.items():
            if isinstance(v, (int, float)) and k != target_key and not np.isnan(float(v)):
                all_keys.add(k)

    target_vals = [
        float(res.get(target_key, float("nan")))
        for res in results_dict.values()
    ]

    rows = []
    for key in sorted(all_keys):
        diag_vals = [
            float(res.get(key, float("nan")))
            for res in results_dict.values()
        ]
        # Filter pairs with valid (non-nan) values.
        valid_pairs = [
            (x, y) for x, y in zip(diag_vals, target_vals)
            if not (np.isnan(x) or np.isnan(y))
        ]
        if len(valid_pairs) < 3:
            continue

        xs = [p[0] for p in valid_pairs]
        ys = [p[1] for p in valid_pairs]

        p_r, p_p = scipy.stats.pearsonr(xs, ys)
        s_r, s_p = scipy.stats.spearmanr(xs, ys)

        rows.append(
            {
                "diagnostic": key,
                "pearson_r": round(float(p_r), 4),
                "pearson_p": round(float(p_p), 4),
                "spearman_r": round(float(s_r), 4),
                "spearman_p": round(float(s_p), 4),
                "abs_pearson_r": abs(float(p_r)),
            }
        )

    if not rows:
        return pl.DataFrame()

    df = (
        pl.DataFrame(rows)
        .sort("abs_pearson_r", descending=True)
        .drop("abs_pearson_r")
    )
    return df
