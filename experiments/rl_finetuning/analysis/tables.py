"""Summary tables for the ablation analysis suite.

Produces polars DataFrames for all summary tables and exports both
CSV and LaTeX formats.  LaTeX generation is done manually since
polars has no built-in ``to_latex()``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from experiments.rl_finetuning.ablations.registry import REGISTRY
from experiments.rl_finetuning.ablations.training import AblationHistory

logger = logging.getLogger(__name__)


def _latex_escape(text: str) -> str:
    """Escape LaTeX special characters in a string.

    Args:
        text: Raw string that may contain LaTeX-special characters.

    Returns:
        String safe for inclusion in a LaTeX document.
    """
    for ch in ("&", "%", "$", "#", "_", "{", "}"):
        text = text.replace(ch, f"\\{ch}")
    text = text.replace("~", "\\textasciitilde{}")
    return text.replace("^", "\\textasciicircum{}")


def _df_to_latex(df: pl.DataFrame, caption: str = "", label: str = "") -> str:
    """Convert a polars DataFrame to a LaTeX tabular string.

    Args:
        df:      Polars DataFrame.
        caption: Optional LaTeX table caption.
        label:   Optional LaTeX table label.

    Returns:
        LaTeX string with ``table`` and ``tabular`` environments.
    """
    cols = df.columns
    col_spec = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        " & ".join(f"\\textbf{{{_latex_escape(c)}}}" for c in cols) + " \\\\",
        "\\midrule",
    ]
    for row in df.iter_rows():
        cells = []
        for val in row:
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(_latex_escape(str(val)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def _save_table(
    df: pl.DataFrame, path_stem: Path, caption: str = "", label: str = ""
) -> None:
    """Save a polars DataFrame as CSV and LaTeX.

    Args:
        df:         Polars DataFrame.
        path_stem:  Output path without extension (e.g., tables_dir / "main_results").
        caption:    LaTeX caption string.
        label:      LaTeX label string.
    """
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path_stem) + ".csv")
    tex = _df_to_latex(df, caption=caption, label=label)
    Path(str(path_stem) + ".tex").write_text(tex)
    logger.info("Saved %s.csv and %s.tex", path_stem, path_stem)


def write_significance_test(results: dict[str, dict], out_dir: Path) -> None:
    """Baseline vs best condition, exact permutation test + bootstrap CI.

    Writes ``significance_test.txt``. With three seeds per condition the
    permutation test is exact (C(6,3) = 20 relabellings).
    """
    base = results.get("baseline_rl")
    if not base or not base.get("all_scores"):
        return
    others = {
        n: r
        for n, r in results.items()
        if n != "baseline_rl" and len(r.get("all_scores", [])) >= 2
    }
    if not others or len(base["all_scores"]) < 2:
        return
    import itertools

    best = max(others, key=lambda n: float(np.mean(others[n]["all_scores"])))
    a = [float(x) for x in base["all_scores"]]
    b = [float(x) for x in others[best]["all_scores"]]
    obs = float(np.mean(b) - np.mean(a))
    pooled = a + b
    n_b = len(b)
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), n_b):
        grp_b = [pooled[i] for i in idx]
        grp_a = [pooled[i] for i in range(len(pooled)) if i not in idx]
        if abs(float(np.mean(grp_b) - np.mean(grp_a))) >= abs(obs) - 1e-12:
            count += 1
        total += 1
    p_perm = count / total
    rng = np.random.default_rng(0)
    boots = [
        float(np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))))
        for _ in range(10000)
    ]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "significance_test.txt").write_text(
        f"baseline_rl scores: {a}\nbest condition: {best} scores: {b}\n"
        f"mean difference (best - baseline): {obs:.4f}\n"
        f"exact permutation test (two-sided, {total} relabellings): p = {p_perm:.3f}\n"
        f"bootstrap 95% CI of the difference (10000 resamples, seed 0): "
        f"[{lo:.4f}, {hi:.4f}]\n"
    )


def make_main_results_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Main results table: Method | Final Score | Delta vs Pretrained | Delta vs Baseline-RL | Verdict.

    Args:
        results:          Dict mapping ablation_name -> {"score": float, ...}.
        pretrained_score: Pretrained model eval score.

    Returns:
        Polars DataFrame with one row per ablation.
    """
    baseline_rl_score = results.get("baseline_rl", {}).get("score", pretrained_score)

    rows = []
    for name, res in results.items():
        score = res["score"]
        delta_pretrained = score - pretrained_score
        delta_baseline = score - baseline_rl_score
        if delta_pretrained > 0.05:
            verdict = "IMPROVEMENT"
        elif score < pretrained_score - 0.1:
            verdict = "COLLAPSE"
        else:
            verdict = "NEUTRAL"

        spec = REGISTRY.get(name)
        group = spec.group if spec else "?"
        rows.append(
            {
                "Method": name,
                "Group": group,
                "Final_Score": round(score, 4),
                "Score_Std": round(
                    float(res.get("score_std", 0.0)), 4
                ),  # popstd over seeds
                "Delta_vs_Pretrained": round(delta_pretrained, 4),
                "Delta_vs_Baseline_RL": round(delta_baseline, 4),
                "Verdict": verdict,
            }
        )

    # Sort by final score descending
    rows.sort(key=lambda r: r["Final_Score"], reverse=True)
    return pl.DataFrame(rows)


def make_gradient_analysis_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Gradient analysis table: Method | Mean Grad Align | Final Grad Align | Trend | Mean Drift | Final Drift.

    Args:
        results: Dict mapping name -> {"history": AblationHistory}.

    Returns:
        Polars DataFrame.
    """
    rows = []
    for name, res in results.items():
        history: AblationHistory = res["history"]
        aligns = history.grad_align
        drifts = history.repr_drift_kl

        mean_align = round(float(np.mean(aligns)), 4) if aligns else float("nan")
        final_align = round(aligns[-1], 4) if aligns else float("nan")
        trend = (
            "down"
            if (len(aligns) > 1 and aligns[-1] < aligns[0])
            else "up"
            if (len(aligns) > 1 and aligns[-1] > aligns[0])
            else "flat"
        )
        mean_drift = round(float(np.mean(drifts)), 6) if drifts else float("nan")
        final_drift = round(drifts[-1], 6) if drifts else float("nan")

        rows.append(
            {
                "Method": name,
                "Mean_Grad_Align": mean_align,
                "Final_Grad_Align": final_align,
                "Trend": trend,
                "Mean_KL_Drift": mean_drift,
                "Final_KL_Drift": final_drift,
            }
        )

    rows.sort(
        key=lambda r: (
            float("-inf") if np.isnan(r["Final_Grad_Align"]) else r["Final_Grad_Align"]
        ),
        reverse=True,
    )
    return pl.DataFrame(rows)


def make_t_distribution_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """t-distribution table: Method | High-t/Low-t Ratio | Low-High Alignment | Dominant regime.

    Args:
        results: Dict mapping name -> {"history": AblationHistory}.

    Returns:
        Polars DataFrame.
    """
    rows = []
    for name, res in results.items():
        history: AblationHistory = res["history"]
        if not history.norm_high_t:
            rows.append(
                {
                    "Method": name,
                    "HighLow_Ratio": float("nan"),
                    "LowHigh_Cos_Sim": float("nan"),
                    "Dominant_Regime": "N/A",
                }
            )
            continue

        ratio = float(np.mean(history.norm_high_t)) / (
            float(np.mean(history.norm_low_t)) + 1e-10
        )
        cos = float(np.mean(history.lowhigh_cos))
        dominant = "high-t" if ratio > 1.5 else "low-t" if ratio < 0.67 else "balanced"
        rows.append(
            {
                "Method": name,
                "HighLow_Ratio": round(ratio, 3),
                "LowHigh_Cos_Sim": round(cos, 4),
                "Dominant_Regime": dominant,
            }
        )

    return pl.DataFrame(rows)


def make_forgetting_analysis_table(
    results: dict[str, dict],
    pretrained_score: float,
    collapse_threshold: float = 0.1,
) -> pl.DataFrame:
    """Forgetting analysis: Method | First collapse iter | Min score | Recovery score | Recovered?.

    Args:
        results:            Dict mapping name -> {"history": AblationHistory, "score": float}.
        pretrained_score:   Pretrained model score.
        collapse_threshold: Fraction below pretrained to count as collapse.

    Returns:
        Polars DataFrame.
    """
    collapse_level = pretrained_score * (1 - collapse_threshold)
    rows = []
    for name, res in results.items():
        history: AblationHistory = res["history"]
        final_score = res["score"]
        evals = history.eval_score
        eval_iters = history.eval_iters

        first_collapse_iter = "never"
        min_score = round(min(evals), 4) if evals else float("nan")
        recovery_score = round(final_score, 4)
        recovered = "N/A"

        if evals:
            for i, (it, sc) in enumerate(zip(eval_iters, evals, strict=False)):
                if sc < collapse_level:
                    first_collapse_iter = str(it)
                    # Check if recovered later
                    later_scores = evals[i + 1 :]
                    recovered = (
                        "Y" if any(s >= collapse_level for s in later_scores) else "N"
                    )
                    break

        rows.append(
            {
                "Method": name,
                "First_Collapse_Iter": first_collapse_iter,
                "Min_Score": min_score,
                "Recovery_Score": recovery_score,
                "Recovered": recovered,
            }
        )

    return pl.DataFrame(rows)


def make_group_summary_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Group summary table: Group | N | Mean | Best | Worst | StdDev.

    Args:
        results: Dict mapping ablation_name -> {"score": float}.

    Returns:
        Polars DataFrame with one row per group.
    """
    group_scores: dict[str, list[float]] = {}
    for name, res in results.items():
        spec = REGISTRY.get(name)
        group = spec.group if spec else "?"
        group_scores.setdefault(group, []).append(res["score"])

    rows = []
    for group in ("Baseline", "A", "B", "C", "D"):
        scores = group_scores.get(group, [])
        if not scores:
            continue
        arr = np.array(scores)
        rows.append(
            {
                "Group": group,
                "N": len(scores),
                "Mean": round(float(arr.mean()), 4),
                "Best": round(float(arr.max()), 4),
                "Worst": round(float(arr.min()), 4),
                "StdDev": round(float(arr.std()), 4),
            }
        )
    return pl.DataFrame(rows)


def make_repr_drift_table(
    results: dict[str, dict],
) -> pl.DataFrame:
    """Repr drift table: Method | KL_mean | KL_low_t | KL_mid_t | KL_high_t (final).

    Args:
        results: Dict mapping name -> {"history": AblationHistory}.

    Returns:
        Polars DataFrame.
    """
    rows = []
    for name, res in results.items():
        history: AblationHistory = res["history"]
        kl_mean = (
            round(history.repr_drift_kl[-1], 6)
            if history.repr_drift_kl
            else float("nan")
        )
        kl_low = (
            round(history.repr_drift_kl_low_t[-1], 6)
            if history.repr_drift_kl_low_t
            else float("nan")
        )
        kl_mid = (
            round(history.repr_drift_kl_mid_t[-1], 6)
            if history.repr_drift_kl_mid_t
            else float("nan")
        )
        kl_high = (
            round(history.repr_drift_kl_high_t[-1], 6)
            if history.repr_drift_kl_high_t
            else float("nan")
        )
        rows.append(
            {
                "Method": name,
                "KL_mean": kl_mean,
                "KL_low_t": kl_low,
                "KL_mid_t": kl_mid,
                "KL_high_t": kl_high,
            }
        )
    rows.sort(key=lambda r: float("inf") if np.isnan(r["KL_mean"]) else r["KL_mean"])
    return pl.DataFrame(rows)


def make_per_env_table(
    results: dict[str, dict],
    pretrained_ach_rates: dict[str, float],
) -> pl.DataFrame:
    """Per-environment (per-achievement) table: Method + final rates.

    Rows = ablation methods, columns = achievement names.

    Args:
        results:              Dict mapping name -> {"history": AblationHistory}.
        pretrained_ach_rates: Per-achievement unlock rates for the pretrained
                              baseline (keys = achievement name, values in [0, 1]).

    Returns:
        Polars DataFrame.
    """
    ablation_finals: dict[str, dict[str, float]] = {}
    for name, res in results.items():
        rates = res["history"].per_achievement_rates
        ablation_finals[name] = rates[-1] if rates else {}

    all_keys: list[str] = sorted(
        set(pretrained_ach_rates) | {k for d in ablation_finals.values() for k in d}
    )
    if not all_keys:
        return pl.DataFrame()

    rows = []
    # Pretrained baseline row
    pt_row: dict[str, object] = {"Method": "pretrained"}
    for key in all_keys:
        pt_row[key] = round(pretrained_ach_rates.get(key, 0.0), 4)
    rows.append(pt_row)

    for name in sorted(ablation_finals):
        row: dict[str, object] = {"Method": name}
        for key in all_keys:
            row[key] = round(ablation_finals[name].get(key, 0.0), 4)
        rows.append(row)

    return pl.DataFrame(rows)


def make_hypothesis_verdict_table(
    results: dict[str, dict],
    pretrained_score: float,
) -> pl.DataFrame:
    """Hypothesis verdict table: Ablation | Hypothesis | Result | Conclusion.

    Args:
        results:          Dict mapping name -> {"score": float}.
        pretrained_score: Pretrained model score.

    Returns:
        Polars DataFrame.
    """
    rows = []
    for name, res in results.items():
        score = res["score"]
        spec = REGISTRY.get(name)
        if spec is None:
            continue

        delta = score - pretrained_score
        if delta > 0.05:
            result = "IMPROVEMENT"
            conclusion = "Hypothesis SUPPORTED — this intervention helps"
        elif score < pretrained_score - 0.1:
            result = "COLLAPSE"
            conclusion = "Hypothesis REFUTED — intervention did not prevent collapse"
        else:
            result = "NEUTRAL"
            conclusion = "Inconclusive — no significant change"

        rows.append(
            {
                "Ablation": name,
                "Group": spec.group,
                "Hypothesis": spec.hypothesis[:80]
                + ("..." if len(spec.hypothesis) > 80 else ""),
                "Result": result,
                "Conclusion": conclusion,
            }
        )

    return pl.DataFrame(rows)


def make_achievement_table(
    results: dict[str, dict],
    pretrained_ach_rates: dict[str, float],
) -> pl.DataFrame:
    """Per-achievement summary table.

    Rows = achievements, columns = ablation methods (final unlock rate) plus a
    ``delta_vs_pretrained`` column showing the change from the pretrained baseline.

    Args:
        results:              Dict mapping ablation_name -> {"history": AblationHistory}.
        pretrained_ach_rates: Per-achievement unlock rates for the pretrained baseline
                              (keys = achievement name, values in [0, 1]).

    Returns:
        Polars DataFrame with one row per achievement.
    """
    # Collect final achievement rates for every ablation.
    ablation_finals: dict[str, dict[str, float]] = {}
    for name, res in results.items():
        rates = res["history"].per_achievement_rates
        ablation_finals[name] = rates[-1] if rates else {}

    # Union of all achievement keys.
    all_keys: list[str] = sorted(
        set(pretrained_ach_rates) | {k for d in ablation_finals.values() for k in d}
    )
    if not all_keys:
        return pl.DataFrame()

    ablation_names = sorted(ablation_finals)
    rows = []
    for key in all_keys:
        pt_rate = pretrained_ach_rates.get(key, 0.0)
        row: dict[str, object] = {"Achievement": key, "Pretrained": round(pt_rate, 4)}
        for abl_name in ablation_names:
            final_rate = ablation_finals[abl_name].get(key, 0.0)
            row[abl_name] = round(final_rate, 4)
            row[f"delta_{abl_name}"] = round(final_rate - pt_rate, 4)
        rows.append(row)

    return pl.DataFrame(rows)


def generate_summary_tables(
    results: dict[str, dict],
    pretrained_score: float,
    output_dir: Path,
    pretrained_ach_rates: dict[str, float] | None = None,
) -> dict[str, pl.DataFrame]:
    """Generate all summary tables and save to output_dir/tables/.

    Args:
        results:              Dict mapping ablation_name -> {
                                  "history": AblationHistory,
                                  "score": float
                              }.
        pretrained_score:     Pretrained model eval score.
        output_dir:           Root output directory; tables go in output_dir/tables/.
        pretrained_ach_rates: Optional per-achievement unlock rates for the pretrained
                              baseline (keys = achievement name, values in [0, 1]).
                              When provided, a per-achievement summary table is generated.

    Returns:
        Dict mapping table name -> polars DataFrame.
    """
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    tables: dict[str, pl.DataFrame] = {}

    tables["main_results"] = make_main_results_table(results, pretrained_score)
    _save_table(
        tables["main_results"],
        tables_dir / "main_results",
        caption="Main ablation results.",
        label="tab:main_results",
    )
    write_significance_test(results, tables_dir)

    tables["gradient_analysis"] = make_gradient_analysis_table(results)
    _save_table(
        tables["gradient_analysis"],
        tables_dir / "gradient_analysis",
        caption="Gradient alignment and representation drift analysis.",
        label="tab:gradient",
    )

    tables["t_distribution"] = make_t_distribution_table(results)
    _save_table(
        tables["t_distribution"],
        tables_dir / "t_distribution",
        caption="Timestep distribution analysis.",
        label="tab:t_dist",
    )

    tables["forgetting_analysis"] = make_forgetting_analysis_table(
        results, pretrained_score
    )
    _save_table(
        tables["forgetting_analysis"],
        tables_dir / "forgetting_analysis",
        caption="Catastrophic forgetting timeline.",
        label="tab:forgetting",
    )

    tables["group_summary"] = make_group_summary_table(results)
    _save_table(
        tables["group_summary"],
        tables_dir / "group_summary",
        caption="Group summary statistics.",
        label="tab:group_summary",
    )

    tables["repr_drift"] = make_repr_drift_table(results)
    _save_table(
        tables["repr_drift"],
        tables_dir / "repr_drift",
        caption="Representation drift (KL divergence) at final iteration.",
        label="tab:repr_drift",
    )

    tables["hypothesis_verdict"] = make_hypothesis_verdict_table(
        results, pretrained_score
    )
    _save_table(
        tables["hypothesis_verdict"],
        tables_dir / "hypothesis_verdict",
        caption="Hypothesis verdict per ablation.",
        label="tab:hypothesis",
    )

    if pretrained_ach_rates is not None:
        ach_df = make_achievement_table(results, pretrained_ach_rates)
        if ach_df.height > 0:
            tables["achievement_summary"] = ach_df
            _save_table(
                ach_df,
                tables_dir / "achievement_summary",
                caption="Per-achievement final unlock rates and delta vs pretrained.",
                label="tab:achievements",
            )

        per_env_df = make_per_env_table(results, pretrained_ach_rates)
        if per_env_df.height > 0:
            tables["per_env"] = per_env_df
            _save_table(
                per_env_df,
                tables_dir / "per_env",
                caption="Per-environment (per-achievement) win rates at final eval.",
                label="tab:per_env",
            )

    logger.info("All tables saved to %s", tables_dir)
    return tables
