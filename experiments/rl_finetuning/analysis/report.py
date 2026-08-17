"""Hypothesis attribution dashboard and diagnosis report generation.

Produces ``diagnosis.md`` — a human-readable verdict that:
1. States the primary failure mode
2. Provides evidence from the ablation results
3. Ranks hypotheses by evidence strength
4. Recommends the next experiments

Also generates a ``diagnosis_decision_tree.png`` matplotlib figure
showing a decision tree: given the pattern of successes/failures, which
hypothesis each pattern implies.
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
from experiments.rl_finetuning.analysis.tables import (
    baseline_rl_score_of,
    verdict,
)

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

_DPI = 150


_HYPOTHESIS_GROUPS = {
    "Catastrophic Forgetting": {
        "supporting_ablations": [
            "kl_penalty",
            "ewc",
            "llrd",
            "frozen_backbone",
            "head_only",
        ],
        "description": "The pretrained representations are corrupted by RL gradients.",
        "recommendation": "Implement a strong parameter regularisation regime (EWC + LLRD) "
        "or use LoRA to restrict the parameter update space.",
    },
    "Gradient Conflict": {
        "supporting_ablations": ["gradient_surgery", "kl_penalty", "low_t"],
        "description": "RL and BC gradients point in conflicting directions, "
        "cancelling useful updates.",
        "recommendation": "Apply PCGrad in the full training pipeline, and investigate "
        "whether the t-distribution of RL batches is biased.",
    },
    "Signal Sparsity": {
        "supporting_ablations": [
            "bc_wins",
            "reward_filtering",
            "running_stats",
            "reward_model",
        ],
        "description": "Returns are too sparse or noisy to provide a useful training signal.",
        "recommendation": "Increase num_envs, use a reward shaping strategy, or apply "
        "curriculum-based episode selection.",
    },
    "Distributional Shift": {
        "supporting_ablations": ["mixed_replay", "action_diversity"],
        "description": "Online data distribution is too different from the offline pretraining "
        "distribution.",
        "recommendation": "Maintain a large offline replay buffer mixed into every batch, "
        "or apply importance sampling corrections.",
    },
    "Mode Collapse": {
        "supporting_ablations": ["entropy_bonus", "advantage_clip", "normalized_adv"],
        "description": "The model collapses to a degenerate distribution, losing action diversity.",
        "recommendation": "Add a strong entropy bonus and clip advantages to prevent "
        "gradient spikes from high-return samples.",
    },
    "t-Bias": {
        "supporting_ablations": ["low_t", "t_curriculum"],
        "description": "High-t (coarse structure) gradients dominate and carry a misleading "
        "signal.",
        "recommendation": "Restrict training to low-t regime or use a t-curriculum to "
        "introduce noise levels in order.",
    },
}


def _score_hypothesis(
    hyp_name: str,
    hyp_info: dict,
    results: dict[str, dict],
    pretrained_score: float,
) -> dict:
    """Score a hypothesis by how many of its supporting ablations succeeded.

    An ablation "supports" a hypothesis if its score exceeds
    ``max(pretrained_score, baseline_score) + 0.01``.

    Args:
        hyp_name:         Hypothesis name.
        hyp_info:         Dict with ``supporting_ablations``, ``description``, ``recommendation``.
        results:          Ablation results dict.
        pretrained_score: Pretrained baseline score.

    Returns:
        Dict with ``hypothesis``, ``evidence_score``, ``n_supporting``, ``n_tested``,
        ``description``, ``recommendation``.
    """
    baseline_score = results.get("baseline_rl", {}).get("score", pretrained_score)
    threshold = max(pretrained_score, baseline_score) + 0.01

    n_tested = 0
    n_supporting = 0
    supporting_names = []

    for abl_name in hyp_info["supporting_ablations"]:
        if abl_name not in results:
            continue
        n_tested += 1
        score = results[abl_name]["score"]
        if score > threshold:
            n_supporting += 1
            supporting_names.append(abl_name)

    evidence_score = n_supporting / max(n_tested, 1)
    return {
        "hypothesis": hyp_name,
        "evidence_score": evidence_score,
        "n_supporting": n_supporting,
        "n_tested": n_tested,
        "supporting_names": supporting_names,
        "description": hyp_info["description"],
        "recommendation": hyp_info["recommendation"],
    }


def _plot_decision_tree(
    scored_hypotheses: list[dict],
    output_dir: Path,
) -> None:
    """Generate a simple decision-tree matplotlib figure.

    The tree shows: IF ablation X succeeds THEN hypothesis H is supported.
    Drawn as a horizontal tree with matplotlib patches.

    Args:
        scored_hypotheses: List of scored hypothesis dicts (sorted by evidence_score).
        output_dir:        Output directory for the PNG.
    """
    fig, ax = plt.subplots(figsize=(16, max(8.0, len(scored_hypotheses) * 1.5)))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(scored_hypotheses) + 1)
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.set_title("Hypothesis Attribution Decision Tree", fontsize=14, fontweight="bold")

    cmap = matplotlib.colormaps["RdYlGn"]
    for i, hyp in enumerate(scored_hypotheses):
        y = len(scored_hypotheses) - i
        ev = hyp["evidence_score"]
        color = cmap(ev)

        # Hypothesis box
        rect = plt.Rectangle((0.1, y - 0.35), 3.5, 0.7, color=color, alpha=0.7)
        ax.add_patch(rect)
        ax.text(
            1.85,
            y,
            hyp["hypothesis"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

        # Evidence score
        ax.text(3.8, y, f"evidence={ev:.0%}", ha="left", va="center", fontsize=8)

        # Supporting ablations
        supp = ", ".join(hyp["supporting_names"]) if hyp["supporting_names"] else "none"
        ax.text(
            5.5,
            y,
            f"supported by: {supp}",
            ha="left",
            va="center",
            fontsize=7,
            color="dimgrey",
            wrap=True,
        )

    # Colour bar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.02, pad=0.01)
    cbar.set_label("Evidence strength (0=no support, 1=full support)", fontsize=8)

    path = output_dir / "figures" / "diagnosis_decision_tree.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved decision tree to %s", path)


def generate_diagnosis_report(
    results: dict[str, dict],
    pretrained_score: float,
    tables: dict[str, Any],
    output_dir: Path,
) -> str:
    """Generate the full diagnosis.md and decision tree figure.

    Args:
        results:          Dict mapping ablation_name -> {"history": AblationHistory, "score": float}.
        pretrained_score: Pretrained model eval score.
        tables:           Dict of polars DataFrames from ``generate_summary_tables``.
        output_dir:       Root output directory.

    Returns:
        Path string of the generated diagnosis.md file.
    """
    # Score all hypotheses
    scored = [
        _score_hypothesis(name, info, results, pretrained_score)
        for name, info in _HYPOTHESIS_GROUPS.items()
    ]
    scored.sort(key=lambda x: x["evidence_score"], reverse=True)

    # Identify primary failure mode
    primary = scored[0]
    baseline_rl_score = baseline_rl_score_of(results, pretrained_score)
    all_failed = all(
        verdict(res["score"], baseline_rl_score, pretrained_score) == "COLLAPSE"
        for name, res in results.items()
        if name != "baseline_rl"
    )

    # Build diagnosis text
    lines = [
        "# RL Fine-Tuning Ablation Suite — Diagnosis Report",
        "",
        f"**Pretrained baseline score:** {pretrained_score:.4f}",
        f"**Baseline RL score:** {results.get('baseline_rl', {}).get('score', float('nan')):.4f}",
        "",
        "---",
        "",
        "## Primary Failure Mode",
        "",
    ]

    if all_failed:
        lines += [
            "**ALL ablations collapsed.** This is the strongest evidence for a fundamental",
            "incompatibility between the RL fine-tuning signal and the model.",
            "",
        ]
    else:
        n_held = sum(
            1
            for res in results.values()
            if verdict(res["score"], baseline_rl_score, pretrained_score) != "COLLAPSE"
        )
        lines += [
            f"**{n_held}/{len(results)} ablations** held at or above `baseline_rl`.",
            "",
        ]

    lines += [
        f"**Most likely failure mode:** {primary['hypothesis']}",
        f"> {primary['description']}",
        "",
        f"**Evidence strength:** {primary['evidence_score']:.0%} "
        f"({primary['n_supporting']}/{primary['n_tested']} supporting ablations succeeded)",
        "",
    ]

    if primary["supporting_names"]:
        lines += [
            f"**Ablations that support this hypothesis:** {', '.join(primary['supporting_names'])}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Hypothesis Rankings (by Evidence Strength)",
        "",
        "| Hypothesis | Evidence | Supporting Ablations |",
        "|---|---|---|",
    ]
    for hyp in scored:
        supp = ", ".join(hyp["supporting_names"]) if hyp["supporting_names"] else "—"
        lines.append(
            f"| {hyp['hypothesis']} | {hyp['evidence_score']:.0%} "
            f"({hyp['n_supporting']}/{hyp['n_tested']}) | {supp} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Evidence Details per Ablation",
        "",
    ]

    # Per-ablation verdict
    for name, res in sorted(results.items(), key=lambda x: x[1]["score"], reverse=True):
        score = res["score"]
        delta = score - pretrained_score
        label = verdict(score, baseline_rl_score, pretrained_score)
        spec = REGISTRY.get(name)
        hypothesis_text = spec.hypothesis if spec else "N/A"
        lines += [
            f"### {name}  [{label}]",
            f"- **Score:** {score:.4f}  (delta vs pretrained: {delta:+.4f}, "
            f"delta vs baseline_rl: {score - baseline_rl_score:+.4f})",
            f"- **Hypothesis tested:** {hypothesis_text}",
        ]
        history: AblationHistory = res["history"]
        if history.grad_align:
            mean_align = float(np.mean(history.grad_align))
            lines.append(f"- **Mean grad alignment:** {mean_align:+.4f}")
        if history.repr_drift_kl:
            final_drift = history.repr_drift_kl[-1]
            lines.append(f"- **Final KL drift:** {final_drift:.6f}")
        lines.append("")

    lines += [
        "---",
        "",
        "## Recommendations",
        "",
    ]

    # Top-2 hypotheses' recommendations
    for hyp in scored[:2]:
        lines += [
            f"### {hyp['hypothesis']}",
            f"{hyp['recommendation']}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Next Experiments",
        "",
        "Based on the pattern of successes and failures above, the highest-priority",
        "follow-up experiments are:",
        "",
    ]

    if scored[0]["evidence_score"] > 0.5:
        lines.append(
            f"1. **Deep dive on {scored[0]['hypothesis']}**: run multi-seed experiments "
            f"with the best-performing ablations ({', '.join(scored[0]['supporting_names'][:3])}) "
            f"and tune their hyperparameters."
        )
    if len(scored) > 1 and scored[1]["evidence_score"] > 0.3:
        lines.append(
            f"2. **Combine top-2 interventions**: {scored[0]['hypothesis']} + "
            f"{scored[1]['hypothesis']} — run a combined ablation."
        )
    lines += [
        "3. **Increase num_envs and num_seeds**: current results may be high-variance. "
        "Confirm findings with num_seeds=3.",
        "4. **Profile the reward signal**: plot the return histogram over training to "
        "determine if rewards are collapsing to zero.",
        "",
    ]

    md_text = "\n".join(lines)
    md_path = output_dir / "diagnosis.md"
    md_path.write_text(md_text)
    logger.info("Saved diagnosis report to %s", md_path)

    # Generate decision tree figure
    _plot_decision_tree(scored, output_dir)

    return str(md_path)
