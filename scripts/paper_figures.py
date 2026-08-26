"""Rebuild the manuscript figures as vector PDF at NeurIPS column width.

The eleven figures in *Return-Weighted ELBO Fine-Tuning Degrades Masked
Diffusion Planners* put Craftax Classic and MiniHack side by side, so they read
both repositories' ablation ``results.json``. The script that drew the
published PNGs was never committed; this is its replacement.

Usage::

    uv run python scripts/paper_figures.py --outdir results/paper_figures

Copy the emitted PDFs into ``papers/current/src/figures/`` and change the
``\\includegraphics`` extensions.

Conventions that must not drift, because published numbers depend on them:

* The per-condition value on every score axis is ``score``, the mean of
  ``all_scores`` from the post-loop final evaluation. It is *not* the last
  in-loop ``eval_score``, which is a separate draw.
* :math:`\\mathrm{CV}_A = \\sqrt{B/\\mathrm{ESS} - 1}` from
  ``history.effective_batch_size``, with :math:`B` the collected batch size the
  run recorded in its own config: 1024 on Craftax Classic, 4608 on MiniHack.
* In ``fig9`` the filled markers are the five conditions with a weighting rule
  of their own; the rest share the baseline's and are hollow. Hollow
  ``advantage_clip`` and ``normalized_adv`` carry their own logged
  pre-transform value, not the baseline's.

Defects in the published PNGs corrected here:

* ``fig9``: the "pretrained" annotation sits on the checkpoint line it labels
  rather than beside the lowest-scoring point, and ``normalized_adv`` is
  inside the scatter y-limits rather than silently clipped away.
* ``fig8``: baseline RL is drawn last and larger. Its data was always present;
  at equal zorder four conditions within a marker width of it -- two pixels,
  for ``mixed_replay`` -- painted over it, so the legend's black swatch
  pointed at nothing the reader could find. The legend now uses marker
  swatches rather than line swatches, and the log exponents are legible.
* ``fig2``: LoRA's Craftax drift is exactly 0 at every probe, which a log axis
  drops silently, taking the curve the caption's low end refers to with it. It
  is now pinned to ``KL_FLOOR`` and drawn dotted with markers, so it reads as
  pinned rather than measured there.
* ``fig6``: ``collect_diamond`` is the only tier-4 achievement and runs at
  0.001 before and 0 after, so its bars were sub-pixel and the tier-4 legend
  colour appeared nowhere. Near-zero bars are drawn at a floor height with
  their true value annotated.
* ``fig2``, ``fig3``, ``fig4``: the conditions the manuscript names in prose
  are labelled, since these panels otherwise encode 25 conditions in five
  group colours.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from experiments.rl_finetuning.ablations.registry import REGISTRY  # noqa: E402

CRAFTAX_RESULTS = (
    REPO
    / "results/experiments/rl_finetuning/outputs/craftax_classic_ablations/results.json"
)
MINIHACK_RESULTS = (
    REPO.parent
    / "minihack-ReMDM-planner"
    / "results/experiments/rl_finetuning/outputs/minihack_ablations/results.json"
)

# The B in CV_A = sqrt(B/ESS - 1). Read from each run's own recorded config;
# these are what the published runs used, kept only as a fallback for a
# results.json that predates the config being stored.
BATCH_SIZE_FALLBACK = {"craftax": 1024, "minihack": 4608}

ENV_TITLE = {"craftax": "Craftax Classic", "minihack": "MiniHack"}
SCORE_LABEL = {"craftax": "Final score", "minihack": "Final ID win rate"}

# Sampled from the published PNGs so the rebuild is visually continuous with
# the figures the manuscript's prose already describes.
ENV_COLOR = {"craftax": "#1b6ca8", "minihack": "#c44e52"}
GROUP_COLOR = {
    "Baseline": "#000000",
    "A": "#1b6ca8",
    "B": "#e07b39",
    "C": "#2e8b57",
    "D": "#a4508b",
}
GROUP_LABEL = {
    "A": "A: regularisation",
    "B": "B: training signal",
    "C": "C: freezing",
    "D": "D: data quality",
}
TIER_COLOR = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B3"]

# The five conditions whose loss applies a weighting rule of its own. Every
# other condition reuses the baseline's, which is what the hollow markers in
# fig9 mean.
DISTINCT_WEIGHTING = frozenset(
    {"baseline_rl", "bc_wins", "reward_filtering", "running_stats", "reward_model"}
)

# Tech-tree tier per Craftax Classic achievement, as Appendix "Per-Achievement
# Analysis" defines it: 0 requires nothing, 1 a crafting table or placed
# sapling, 2 wooden tools, 3 stone tools and a furnace, 4 an iron pickaxe.
ACHIEVEMENT_TIER = {
    "collect_wood": 0,
    "collect_sapling": 0,
    "collect_drink": 0,
    "eat_cow": 0,
    "defeat_zombie": 0,
    "wake_up": 0,
    "place_table": 1,
    "make_wood_pickaxe": 1,
    "make_wood_sword": 1,
    "place_plant": 1,
    "eat_plant": 1,
    "collect_stone": 2,
    "place_stone": 2,
    "place_furnace": 2,
    "make_stone_pickaxe": 2,
    "make_stone_sword": 2,
    "collect_coal": 2,
    "defeat_skeleton": 2,
    "collect_iron": 3,
    "make_iron_pickaxe": 3,
    "make_iron_sword": 3,
    "collect_diamond": 4,
}

# A zero KL cannot be placed on a log axis. LoRA's drift probe reads its frozen
# base weights and records exactly 0 on Craftax Classic, so it is pinned to the
# decade below the smallest non-zero value in the panel; the appendix already
# warns that its drift figures are not comparable with the others.
KL_FLOOR = 1e-4

# Conditions the manuscript singles out by name ("excluding LoRA",
# "normalised advantages aside"). These get a leader-line label in the
# group-coloured trace panels so the reader can find the line being discussed.
NAMED_IN_PROSE = ("lora", "normalized_adv")

# Minimum drawn bar height in fig6, as a fraction of the 0-1 rate axis. A
# rate below this is drawn as a floor stub with its true value annotated:
# without it a tier whose only achievement runs at ~0 is invisible, and the
# tier legend points at a colour the reader cannot find.
BAR_FLOOR = 0.012

# Marker area in pt^2 for the score-vs-KL scatter, shared with its legend so
# the swatch is the same size as the mark it stands for.
SCATTER_AREA = 18

COLUMN_WIDTH_IN = 5.5  # NeurIPS \linewidth

STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.5,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#333333",
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "legend.fontsize": 6.5,
    "legend.frameon": False,
    "lines.linewidth": 0.9,
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def group_of(name: str) -> str:
    spec = REGISTRY.get(name)
    return spec.group if spec else "Baseline"


def condition_color(name: str) -> str:
    return GROUP_COLOR[group_of(name)]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that keeps the array length and ignores NaN."""
    if window <= 1:
        return values
    out = np.empty_like(values, dtype=float)
    half = window // 2
    for i in range(values.size):
        lo, hi = max(0, i - half), min(values.size, i + half + 1)
        chunk = values[lo:hi]
        chunk = chunk[~np.isnan(chunk)]
        out[i] = chunk.mean() if chunk.size else math.nan
    return out


def batch_size(res: dict, env: str) -> int:
    """Collected batch size B, from the run's own recorded config."""
    cfg = res.get("config") or {}
    for key in ("BATCH_SIZE", "batch_size"):
        if cfg.get(key):
            return int(cfg[key])
    return BATCH_SIZE_FALLBACK[env]


def cv_a(ess: list[float], batch: int) -> np.ndarray:
    """Weight dispersion recovered from logged effective sample size."""
    arr = np.asarray(ess, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.maximum(batch / arr - 1.0, 0.0))


def series(entry: dict[str, Any], key: str) -> np.ndarray:
    return np.asarray(entry["history"].get(key) or [], dtype=float)


def percentile_band(
    curves: list[np.ndarray], lo: float = 10.0, hi: float = 90.0
) -> tuple[np.ndarray, np.ndarray] | None:
    """Element-wise percentile envelope over equal-length curves."""
    usable = [c for c in curves if c.size]
    if len(usable) < 3:
        return None
    n = min(c.size for c in usable)
    stack = np.vstack([c[:n] for c in usable])
    return (
        np.nanpercentile(stack, lo, axis=0),
        np.nanpercentile(stack, hi, axis=0),
    )


def new_figure(
    ncols: int, height: float, *, sharey: bool = False
) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(1, ncols, figsize=(COLUMN_WIDTH_IN, height), sharey=sharey)
    return fig, np.atleast_1d(axes)


def figure_legend(
    fig: plt.Figure, handles: list, labels: list[str], *, ncol: int, y: float = 0.0
) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=6.5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.2,
    )


def _marker_handle(color: str) -> Line2D:
    """A swatch that matches a plotted scatter mark, not a line."""
    return Line2D(
        [],
        [],
        linestyle="none",
        marker="o",
        markerfacecolor=color,
        markeredgecolor="white",
        markeredgewidth=0.3,
        markersize=math.sqrt(SCATTER_AREA),
    )


def group_legend_handles(
    *, baseline: bool = True, pretrained: bool = False, marker: bool = False
) -> tuple:
    """Group swatches. ``marker=True`` for a scatter plot, where a line
    swatch matches nothing the reader can see."""
    handles, labels = [], []
    if pretrained:
        handles.append(Line2D([], [], color="#404040", linestyle="--", linewidth=1.0))
        labels.append("Pretrained")
    if baseline:
        handles.append(
            _marker_handle("black")
            if marker
            else Line2D([], [], color="black", linewidth=1.6)
        )
        labels.append("Baseline RL")
    for key, label in GROUP_LABEL.items():
        handles.append(
            _marker_handle(GROUP_COLOR[key])
            if marker
            else Line2D([], [], color=GROUP_COLOR[key], linewidth=1.0)
        )
        labels.append(label)
    return handles, labels


def save(fig: plt.Figure, outdir: Path, stem: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{stem}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def fig1_finetuning_trajectories(data: dict, outdir: Path) -> Path:
    """In-loop eval score for all 25 conditions, one panel per environment."""
    fig, axes = new_figure(2, 2.3)
    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        res = data[env]
        for name, entry in res["ablations"].items():
            if name == "baseline_rl":
                continue
            ax.plot(
                series(entry, "eval_iters"),
                series(entry, "eval_score"),
                color=condition_color(name),
                linewidth=0.7,
                alpha=0.65,
            )
        base = res["ablations"]["baseline_rl"]
        ax.plot(
            series(base, "eval_iters"),
            series(base, "eval_score"),
            color="black",
            linewidth=1.6,
            zorder=5,
        )
        ax.axhline(
            res["pretrained_score"],
            linestyle="--",
            color="#404040",
            linewidth=1.0,
            zorder=4,
        )
        ax.annotate(
            "pretrained checkpoint",
            xy=(0.03, res["pretrained_score"]),
            xycoords=("axes fraction", "data"),
            xytext=(0, -1.5),
            textcoords="offset points",
            va="top",
            fontsize=6,
            color="#555555",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.75},
        )
        _annotate_condition(
            ax, res["ablations"].get("normalized_adv"), "normalized_adv"
        )
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Fine-tuning iteration")
    axes[0].set_ylabel("Craftax Classic score")
    axes[1].set_ylabel("MiniHack ID win rate")
    fig.tight_layout()
    figure_legend(fig, *group_legend_handles(pretrained=True), ncol=6, y=0.02)
    return save(fig, outdir, "fig1_finetuning_trajectories")


def _annotate_condition(
    ax: plt.Axes,
    entry: dict | None,
    label: str,
    *,
    key_x: str = "eval_iters",
    key_y: str = "eval_score",
    offset: tuple[float, float] = (-46, 26),
    y_override: float | None = None,
) -> None:
    """Label a named trace at its final point with a short leader line.

    The manuscript singles conditions out by name, but these panels encode by
    group colour only -- five colours for 25 conditions -- so a named
    condition is otherwise unfindable.
    """
    if entry is None:
        return
    x, y = series(entry, key_x), series(entry, key_y)
    if not x.size or not y.size:
        return
    yv = y[-1] if y_override is None else y_override
    if not np.isfinite(yv):
        return
    dx, dy = offset
    # Flip the label below the point when the point sits high in the panel,
    # so the text stays inside the axes instead of running into the title.
    frac = ax.transAxes.inverted().transform(ax.transData.transform((x[-1], yv)))[1]
    if (dy > 0 and frac > 0.62) or (dy < 0 and frac < 0.2):
        dy = -dy
    ax.annotate(
        label,
        xy=(x[-1], yv),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=6,
        color="#333333",
        ha="center",
        va="center",
        # These panels are dense: without a backing box the label is
        # unreadable wherever it lands among the traces.
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
        arrowprops={"arrowstyle": "-", "linewidth": 0.4, "color": "#999999"},
        zorder=10,
    )


def _annotate_named(
    ax: plt.Axes,
    res: dict,
    key_x: str,
    key_y: str,
    *,
    pinned: frozenset[str] = frozenset(),
) -> None:
    """Label every condition the prose names, where the panel has data.

    A pinned series carries no real endpoint to point at, so its label is
    anchored to the floor the curve was drawn at.
    """
    for i, name in enumerate(NAMED_IN_PROSE):
        _annotate_condition(
            ax,
            res["ablations"].get(name),
            name,
            key_x=key_x,
            key_y=key_y,
            offset=(-46, 26) if i == 0 else (-46, -26),
            y_override=KL_FLOOR if name in pinned else None,
        )


def _trace_panel(
    ax: plt.Axes,
    res: dict,
    key_x: str,
    key_y: str,
    *,
    logy: bool = False,
    zero_line: bool = False,
    pin_zero: bool = False,
) -> frozenset[str]:
    """All conditions faint by group, baseline RL bold in black.

    With ``pin_zero`` a series holding non-positive values -- which matplotlib
    drops silently from a log axis, taking the whole curve with it -- is
    redrawn at ``KL_FLOOR``, dotted and marked so it reads as pinned rather
    than measured there. Returns the names that were pinned.
    """
    pinned: list[str] = []
    for name, entry in res["ablations"].items():
        if name == "baseline_rl":
            continue
        y = series(entry, key_y)
        is_pinned = pin_zero and y.size and bool(np.any(y <= 0))
        if is_pinned:
            pinned.append(name)
            ax.plot(
                series(entry, key_x),
                np.where(y <= 0, KL_FLOOR, y),
                color=condition_color(name),
                linewidth=0.9,
                linestyle=":",
                marker="v",
                markersize=2.2,
                markevery=2,
                alpha=0.95,
                zorder=4,
            )
            continue
        ax.plot(
            series(entry, key_x),
            y,
            color=condition_color(name),
            linewidth=0.7,
            alpha=0.7,
        )
    base = res["ablations"]["baseline_rl"]
    ax.plot(
        series(base, key_x),
        series(base, key_y),
        color="black",
        linewidth=1.6,
        zorder=5,
    )
    if zero_line:
        ax.axhline(0.0, linestyle=":", color="#777777", linewidth=0.6)
    if logy:
        ax.set_yscale("log")
    return frozenset(pinned)


def fig2_repr_drift(data: dict, outdir: Path) -> Path:
    """KL from the pretrained checkpoint over fine-tuning, log scale."""
    fig, axes = new_figure(2, 2.3)
    any_pinned = False
    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        pinned = _trace_panel(
            ax, data[env], "repr_drift_iters", "repr_drift_kl", logy=True, pin_zero=True
        )
        if pinned:
            any_pinned = True
            print(
                f"  fig2 [{env}]: pinned to KL_FLOOR={KL_FLOOR:g} (exact zero, "
                f"absent from a log axis): {', '.join(sorted(pinned))}",
                file=sys.stderr,
            )
        _annotate_named(
            ax, data[env], "repr_drift_iters", "repr_drift_kl", pinned=pinned
        )
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Fine-tuning iteration")
        ax.set_ylabel("KL from pretrained")
    fig.tight_layout()
    handles, labels = group_legend_handles()
    if any_pinned:
        handles.append(
            Line2D(
                [],
                [],
                color="#777777",
                linestyle=":",
                linewidth=0.9,
                marker="v",
                markersize=2.2,
            )
        )
        labels.append(f"pinned at {KL_FLOOR:g} (exact 0)")
    figure_legend(fig, handles, labels, ncol=6, y=0.02)
    return save(fig, outdir, "fig2_repr_drift")


def fig3_cka(data: dict, outdir: Path) -> Path:
    """CKA similarity to the pretrained checkpoint over fine-tuning."""
    fig, axes = new_figure(2, 2.3)
    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        _trace_panel(ax, data[env], "cka_iters", "cka_similarity")
        ax.set_ylim(0.0, 1.02)
        _annotate_named(ax, data[env], "cka_iters", "cka_similarity")
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Fine-tuning iteration")
        ax.set_ylabel("CKA vs pretrained")
    fig.tight_layout()
    figure_legend(fig, *group_legend_handles(), ncol=5, y=0.02)
    return save(fig, outdir, "fig3_cka")


def fig4_grad_alignment(data: dict, outdir: Path) -> Path:
    """Cosine between the weighted gradient and the unweighted reference."""
    fig, axes = new_figure(2, 2.3)
    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        _trace_panel(ax, data[env], "grad_align_iters", "grad_align", zero_line=True)
        ax.set_ylim(-1.0, 1.0)
        _annotate_named(ax, data[env], "grad_align_iters", "grad_align")
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Fine-tuning iteration")
        ax.set_ylabel("cos(RL grad, BC grad)")
    fig.tight_layout()
    figure_legend(fig, *group_legend_handles(), ncol=5, y=0.02)
    return save(fig, outdir, "fig4_grad_alignment")


def fig5_minihack_per_env(data: dict, outdir: Path) -> Path:
    """MiniHack win rate disaggregated by layout."""
    res = data["minihack"]
    base_hist = res["ablations"]["baseline_rl"]["history"]
    layouts = list(base_hist["per_env_win_rates"][-1].keys())
    fig, axes = new_figure(len(layouts), 1.9, sharey=True)
    for ax, layout in zip(axes, layouts, strict=True):
        for name, entry in res["ablations"].items():
            rates = entry["history"].get("per_env_win_rates") or []
            xs = entry["history"].get("eval_iters") or []
            ys = [r.get(layout, math.nan) for r in rates]
            n = min(len(xs), len(ys))
            if not n:
                continue
            bold = name == "baseline_rl"
            ax.plot(
                xs[:n],
                ys[:n],
                color="black" if bold else condition_color(name),
                linewidth=1.6 if bold else 0.7,
                alpha=1.0 if bold else 0.6,
                zorder=5 if bold else 1,
            )
        ax.set_title(_layout_title(layout))
        ax.set_xlabel("Iteration")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Win rate")
    fig.tight_layout()
    figure_legend(fig, *group_legend_handles(), ncol=5, y=0.02)
    return save(fig, outdir, "fig5_minihack_per_env")


def _layout_title(layout: str) -> str:
    return (
        layout.replace("MiniHack-", "")
        .replace("-v0", "")
        .replace("Room-Random-", "Room ")
        .replace("Corridor-", "Corridor ")
        .replace("MazeWalk-", "MazeWalk ")
    )


def fig6_achievements(data: dict, outdir: Path) -> Path:
    """Craftax Classic achievement rates by tech-tree tier, before and after."""
    res = data["craftax"]
    pre = _achievement_rates(res["pretrained_ach_rates"])
    post = _achievement_rates(_final_achievements(res["ablations"]["baseline_rl"]))

    ordered: list[str] = []
    for tier in range(5):
        names = [a for a, t in ACHIEVEMENT_TIER.items() if t == tier]
        ordered.extend(sorted(names, key=lambda a: -pre.get(a, 0.0)))

    fig, (ax_bar, ax_tier) = plt.subplots(
        2, 1, figsize=(COLUMN_WIDTH_IN, 4.4), height_ratios=[2.1, 1.0]
    )
    width = 0.4
    for i, name in enumerate(ordered):
        color = TIER_COLOR[ACHIEVEMENT_TIER[name]]
        # collect_diamond is the only tier-4 achievement and runs at 0.001
        # before and 0.000 after, so at this axis height its bars are
        # sub-pixel: the legend advertised a tier colour that appeared
        # nowhere. Draw a floor stub instead, and mark the true value, so the
        # colour is findable without misreporting the rate.
        for offset, value, faint in (
            (-width / 2, pre.get(name, 0.0), False),
            (+width / 2, post.get(name, 0.0), True),
        ):
            stub = value < BAR_FLOOR
            ax_bar.bar(
                i + offset,
                BAR_FLOOR if stub else value,
                width,
                color=color,
                alpha=0.35 if faint else 0.95,
                hatch="///" if faint else None,
                edgecolor=color,
                linewidth=0.4 if faint else 0,
            )
            if stub:
                ax_bar.annotate(
                    f"{value:.3f}".rstrip("0").rstrip(".") if value else "0",
                    xy=(i + offset, BAR_FLOOR),
                    xytext=(0, 1.5),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=4.5,
                    color=color,
                )
    ax_bar.set_xticks(range(len(ordered)))
    ax_bar.set_xticklabels(
        [n.replace("_", " ") for n in ordered], rotation=45, ha="right", fontsize=6
    )
    ax_bar.set_ylabel("Completion rate")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.grid(axis="x", visible=False)
    for side in ("top", "right", "left"):
        ax_bar.spines[side].set_visible(False)

    tiers = sorted(set(ACHIEVEMENT_TIER.values()))
    pre_mean = [_tier_mean(pre, t) for t in tiers]
    post_mean = [_tier_mean(post, t) for t in tiers]
    ax_tier.plot(
        tiers,
        pre_mean,
        "-o",
        color="black",
        markersize=3.5,
        label="Pretrained (DAgger)",
    )
    ax_tier.plot(
        tiers,
        post_mean,
        "--s",
        color="#C44E52",
        markersize=3.5,
        label="After RL fine-tuning",
    )
    ax_tier.set_xticks(tiers)
    ax_tier.set_xlabel("Tech-tree tier")
    ax_tier.set_ylabel("Mean rate")
    ax_tier.legend(loc="upper right", fontsize=6.5)

    handles = [Patch(facecolor=TIER_COLOR[t], label=f"tier {t}") for t in tiers] + [
        Patch(facecolor="#808080", label="pretrained (solid)"),
        Patch(
            facecolor="#808080",
            alpha=0.35,
            hatch="///",
            edgecolor="#808080",
            label="post-RL (hatched)",
        ),
    ]
    figure_legend(fig, handles, [h.get_label() for h in handles], ncol=7, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return save(fig, outdir, "fig6_achievements")


def _achievement_rates(raw: dict[str, float]) -> dict[str, float]:
    return {k.split("/")[-1].lower(): float(v) for k, v in raw.items()}


def _final_achievements(entry: dict) -> dict[str, float]:
    """Post-loop achievement rates, falling back to the last in-loop entry.

    ``final_ach`` is the detail of the same evaluation that produced
    ``score``. Older suite runs only have the in-loop history, whose last
    element the harness overwrote with the final evaluation.
    """
    if entry.get("final_ach_rates"):
        return entry["final_ach_rates"]
    for record in reversed(entry["history"].get("per_achievement_rates") or []):
        if record:
            return record
    return {}


def _tier_mean(rates: dict[str, float], tier: int) -> float:
    vals = [rates.get(a, 0.0) for a, t in ACHIEVEMENT_TIER.items() if t == tier]
    return float(np.mean(vals)) if vals else math.nan


def fig7_tbin_gradients(data: dict, outdir: Path) -> Path:
    """Gradient L2 norm by diffusion timestep bin, baseline RL."""
    fig, axes = new_figure(2, 2.3)
    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        base = data[env]["ablations"]["baseline_rl"]
        iters = base["history"]["t_analysis_iters"]
        bins = base["history"]["t_bin_norms"]
        keys = list(bins[-1].keys())
        grid = np.array([[b.get(k, math.nan) for b in bins] for k in keys])
        im = ax.imshow(
            grid,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=(min(iters), max(iters), 0.0, 1.0),
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("grad L2 norm", fontsize=6.5)
        cbar.ax.tick_params(labelsize=6)
        ax.grid(visible=False)
        ax.set_title(f"{ENV_TITLE[env]} — baseline RL")
        ax.set_xlabel("Fine-tuning iteration")
        ax.set_ylabel("diffusion time $t$")
    fig.tight_layout()
    return save(fig, outdir, "fig7_tbin_gradients")


def _kl_point(
    ax: plt.Axes,
    name: str,
    entry: dict,
    *,
    size: float = SCATTER_AREA,
    edge: float = 0.3,
    zorder: int = 3,
) -> None:
    """One condition's (final KL, final score) mark, zero pinned to the floor."""
    ax.scatter(
        max(float(series(entry, "repr_drift_kl")[-1]), KL_FLOOR),
        entry["score"],
        s=size,
        color=condition_color(name),
        edgecolor="white",
        linewidth=edge,
        zorder=zorder,
    )


def fig8_score_vs_kl(data: dict, outdir: Path) -> Path:
    """Final score against final KL from the checkpoint, log x.

    Baseline RL is drawn last at a higher zorder. Several conditions land
    within a marker width of it -- ``mixed_replay`` is under two pixels away
    at print size -- and at equal zorder the later scatter call painted over
    it, which is why the published figure has no locatable black point.
    """
    fig, axes = new_figure(2, 2.3)
    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        res = data[env]

        skipped = []
        for name, entry in res["ablations"].items():
            if not series(entry, "repr_drift_kl").size:
                skipped.append(name)
                continue
            if name == "baseline_rl":
                continue
            _kl_point(ax, name, entry)
        if skipped:
            print(
                f"  fig8 [{env}]: no repr_drift_kl, point not plotted: "
                f"{', '.join(sorted(skipped))}",
                file=sys.stderr,
            )
        base = res["ablations"]["baseline_rl"]
        if series(base, "repr_drift_kl").size:
            _kl_point(
                ax, "baseline_rl", base, zorder=6, size=SCATTER_AREA + 6, edge=0.5
            )
        ax.axhline(
            res["pretrained_score"], linestyle="--", color="#404040", linewidth=1.0
        )
        ax.set_xscale("log")
        # The mathtext exponent renders at ~0.7x the tick size; at the 6.5 pt
        # default that is 4.55 pt, the smallest text in the figure set.
        ax.tick_params(axis="x", labelsize=8.0)
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("Final KL from pretrained")
        ax.set_ylabel(SCORE_LABEL[env])
    fig.tight_layout()
    # Marker swatches: this is a scatter, so a line swatch matches nothing.
    handles, labels = group_legend_handles(pretrained=True, marker=True)
    figure_legend(fig, handles, labels, ncol=6, y=0.02)
    return save(fig, outdir, "fig8_score_vs_kl")


def fig9_weight_dispersion(data: dict, outdir: Path) -> Path:
    """CV_A over training, and mean CV_A against final score per condition."""
    fig, axes = plt.subplots(1, 3, figsize=(COLUMN_WIDTH_IN, 2.1))

    ax = axes[0]
    for env in ("craftax", "minihack"):
        res = data[env]
        batch = batch_size(res, env)
        base = res["ablations"]["baseline_rl"]
        raw = cv_a(base["history"]["effective_batch_size"], batch)
        xs = np.asarray(base["history"]["iters"], dtype=float)[: raw.size]
        window = 11 if raw.size > 100 else 1
        if window > 1:
            ax.plot(xs, raw, color=ENV_COLOR[env], linewidth=0.4, alpha=0.3)
        ax.plot(
            xs,
            rolling_mean(raw, window),
            color=ENV_COLOR[env],
            linewidth=1.4,
            label=ENV_TITLE[env],
        )
        band = percentile_band(
            [
                cv_a(e["history"]["effective_batch_size"], batch)
                for e in res["ablations"].values()
            ]
        )
        if band is not None:
            lo, hi = band
            ax.fill_between(
                xs[: lo.size], lo, hi, color=ENV_COLOR[env], alpha=0.15, linewidth=0
            )
    ax.set_ylim(bottom=0.0)
    ax.set_title("Weight dispersion over training")
    ax.set_xlabel("Fine-tuning iteration")
    ax.set_ylabel("weight dispersion $\\mathrm{CV}_A$")
    ax.legend(loc="upper right", fontsize=6.5)

    for ax, env in zip(axes[1:], ("craftax", "minihack"), strict=True):
        res = data[env]
        batch = batch_size(res, env)
        scores = []
        for name, entry in res["ablations"].items():
            mean_cv = float(
                np.nanmean(cv_a(entry["history"]["effective_batch_size"], batch))
            )
            filled = name in DISTINCT_WEIGHTING
            scores.append(entry["score"])
            ax.scatter(
                mean_cv,
                entry["score"],
                s=30 if filled else 22,
                facecolor=condition_color(name) if filled else "none",
                edgecolor="#333333" if filled else "#999999",
                linewidth=0.6 if filled else 0.5,
                zorder=4 if filled else 2,
            )
        pretrained = res["pretrained_score"]
        ax.axhline(pretrained, linestyle="--", color="#404040", linewidth=1.0)
        # Every condition, normalized_adv included, must be inside the limits:
        # the published PNG silently clipped the lowest-scoring point away.
        lo, hi = min(scores), max(max(scores), pretrained)
        pad = 0.08 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        # The annotation labels the dashed checkpoint line, so it sits on it.
        ax.annotate(
            "pretrained",
            xy=(0.98, pretrained),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6,
            color="#555555",
        )
        ax.set_title(ENV_TITLE[env])
        ax.set_xlabel("mean $\\mathrm{CV}_A$")
        ax.set_ylabel(SCORE_LABEL[env])

    fig.tight_layout()
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="#555555",
            markeredgecolor="#333333",
            markersize=4.5,
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor="#999999",
            markersize=4.0,
        ),
    ]
    figure_legend(
        fig,
        handles,
        [
            f"distinct weighting rule (n={len(DISTINCT_WEIGHTING)})",
            "shares the baseline rule",
        ],
        ncol=2,
        y=0.03,
    )
    return save(fig, outdir, "fig9_weight_dispersion")


def fig10_timestep_conditioning(data: dict, outdir: Path) -> Path:
    """Gradient norm across diffusion time, and low-t/high-t agreement."""
    fig, axes = new_figure(2, 2.3)

    ax = axes[0]
    for env in ("craftax", "minihack"):
        bins = data[env]["ablations"]["baseline_rl"]["history"]["t_bin_norms"]
        keys = list(bins[-1].keys())
        grid = np.array([[b.get(k, math.nan) for b in bins] for k in keys])
        mean, std = np.nanmean(grid, axis=1), np.nanstd(grid, axis=1)
        scale = np.nanmax(mean)
        centres = np.array([_bin_centre(k) for k in keys])
        ax.plot(
            centres,
            mean / scale,
            "-o",
            color=ENV_COLOR[env],
            markersize=2.5,
            label=ENV_TITLE[env],
        )
        ax.fill_between(
            centres,
            (mean - std) / scale,
            (mean + std) / scale,
            color=ENV_COLOR[env],
            alpha=0.15,
            linewidth=0,
        )
    ax.set_ylim(bottom=0.0)
    ax.set_title("Gradient norm by diffusion time")
    ax.set_xlabel("diffusion time $t$  (0 = data, 1 = fully masked)")
    ax.set_ylabel("relative gradient norm")
    ax.legend(loc="upper left", fontsize=6.5)

    ax = axes[1]
    for env in ("craftax", "minihack"):
        res = data[env]
        for name, entry in res["ablations"].items():
            if name == "baseline_rl":
                continue
            ax.plot(
                series(entry, "t_analysis_iters"),
                series(entry, "lowhigh_cos"),
                color=ENV_COLOR[env],
                linewidth=0.5,
                alpha=0.18,
            )
        base = res["ablations"]["baseline_rl"]
        ax.plot(
            series(base, "t_analysis_iters"),
            series(base, "lowhigh_cos"),
            color=ENV_COLOR[env],
            linewidth=1.5,
            label=ENV_TITLE[env],
            zorder=5,
        )
    ax.axhline(0.0, linestyle=":", color="#777777", linewidth=0.6)
    ax.set_title("Low-$t$ vs high-$t$ gradient agreement")
    ax.set_xlabel("Fine-tuning iteration")
    ax.set_ylabel(r"cos($\nabla$ low-$t$, $\nabla$ high-$t$)")
    ax.legend(loc="upper right", fontsize=6.5)

    fig.tight_layout()
    return save(fig, outdir, "fig10_timestep_conditioning")


def _bin_centre(key: str) -> float:
    lo, hi = key.removeprefix("t_").split("-")
    return (float(lo) + float(hi)) / 2.0


def fig11_train_vs_eval(data: dict, outdir: Path) -> Path:
    """Collection-time return against harness eval score, baseline RL."""
    fig, axes = new_figure(2, 2.3)
    train_label = {
        "craftax": "collected episodic return",
        "minihack": "collected window return",
    }
    eval_label = {"craftax": "eval score", "minihack": "eval win rate"}
    red, blue = "#c44e52", "#1b6ca8"

    for ax, env in zip(axes, ("craftax", "minihack"), strict=True):
        res = data[env]
        base = res["ablations"]["baseline_rl"]
        xs = series(base, "env_score_iters")
        ys = series(base, "env_score")
        window = 11 if ys.size > 100 else 1
        if window > 1:
            ax.plot(xs, ys, color=red, linewidth=0.4, alpha=0.35)
        ax.plot(
            xs,
            rolling_mean(ys, window),
            color=red,
            linewidth=1.2,
            label="collected return (train)",
        )
        band = percentile_band(
            [series(e, "env_score") for e in res["ablations"].values()]
        )
        if band is not None:
            lo, hi = band
            ax.fill_between(xs[: lo.size], lo, hi, color=red, alpha=0.12, linewidth=0)
        ax.set_xlabel("Fine-tuning iteration")
        ax.set_ylabel(train_label[env], color=red)
        ax.tick_params(axis="y", colors=red)

        twin = ax.twinx()
        twin.plot(
            series(base, "eval_iters"),
            series(base, "eval_score"),
            "-o",
            color=blue,
            markersize=2.5,
            linewidth=1.2,
            label="eval score",
        )
        twin.axhline(res["pretrained_score"], linestyle="--", color=blue, linewidth=1.0)
        twin.annotate(
            "pretrained",
            xy=(0.98, res["pretrained_score"]),
            xycoords=("axes fraction", "data"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6,
            color=blue,
        )
        twin.set_ylabel(eval_label[env], color=blue)
        twin.tick_params(axis="y", colors=blue)
        twin.grid(visible=False)
        ax.set_title(ENV_TITLE[env])

        handles = [
            Line2D([], [], color=red, linewidth=1.2),
            Line2D([], [], color=blue, marker="o", markersize=2.5, linewidth=1.2),
        ]
        ax.legend(
            handles,
            ["collected return (train)", "eval score"],
            loc="lower left",
            fontsize=6,
        )
    fig.tight_layout()
    return save(fig, outdir, "fig11_train_vs_eval")


FIGURES = (
    fig1_finetuning_trajectories,
    fig2_repr_drift,
    fig3_cka,
    fig4_grad_alignment,
    fig5_minihack_per_env,
    fig6_achievements,
    fig7_tbin_gradients,
    fig8_score_vs_kl,
    fig9_weight_dispersion,
    fig10_timestep_conditioning,
    fig11_train_vs_eval,
)


def load(path: Path, label: str) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{label} results not found at {path}. Pass an explicit path; the "
            f"MiniHack default is the sibling checkout "
            f"../minihack-ReMDM-planner."
        )
    with path.open() as fh:
        return json.load(fh)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--craftax-results", type=Path, default=CRAFTAX_RESULTS)
    parser.add_argument("--minihack-results", type=Path, default=MINIHACK_RESULTS)
    parser.add_argument("--outdir", type=Path, default=REPO / "results/paper_figures")
    args = parser.parse_args()

    data = {
        "craftax": load(args.craftax_results, "Craftax Classic"),
        "minihack": load(args.minihack_results, "MiniHack"),
    }

    plt.rcParams.update(STYLE)
    for builder in FIGURES:
        print(f"wrote {builder(data, args.outdir)}")


if __name__ == "__main__":
    main()
