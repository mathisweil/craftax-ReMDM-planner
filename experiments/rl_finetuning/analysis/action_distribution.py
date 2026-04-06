"""Action distribution analysis: pre- vs post-finetuning comparison.

Compares action distributions of a pretrained model against a finetuned
model by rolling out episodes, computing divergence metrics, and
generating diagnostic plots.

Metrics computed:
- Per-action frequency histogram
- Jensen-Shannon divergence
- KL divergence (both directions)
- Total variation distance
- Effective number of actions (entropy-based)
- Gini coefficient of action distribution
- Bigram (action transition) matrices

Plots generated:
- Side-by-side action frequency bar chart
- JS divergence per ablation bar chart
- Transition matrix heatmaps (pre, post, difference)
- Metrics dashboard (2x2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
logger = logging.getLogger(__name__)

_DPI = 150
_EPS = 1e-10


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def _build_action_collector(
    env: Any,
    env_params: Any,
    apply_eval: Callable,
    config: dict,
) -> Callable:
    """Build a JIT-compiled function that rolls out and returns raw actions.

    Args:
        env:        Wrapped Craftax environment.
        env_params: Environment parameters.
        apply_eval: Eval apply fn (params, obs, z_t, t) -> logits.
        config:     UPPERCASE config dict.

    Returns:
        JIT-compiled fn(params, rng) -> (actions, rewards, dones) where
        actions is ``[n_steps, num_envs]`` int32.
    """
    from src.diffusion.sampling import sample_plan
    from src.diffusion.schedules import SCHEDULE_MAP

    schedule_fn, _ = SCHEDULE_MAP[config["DIFFUSION_SCHEDULE"]]
    num_actions = config["NUM_ACTIONS"]
    n_cycles = config["EVAL_STEPS"] // config["EVAL_REPLAN"]

    @jax.jit
    def collect(params: Any, rng: jax.Array) -> tuple:
        """Roll out episodes and return raw per-step data.

        Args:
            params: Model parameters.
            rng:    PRNG key.

        Returns:
            Tuple of (actions, rewards, dones, returned_episode) each
            with shape ``[n_cycles * replan_steps, num_envs]``.
        """
        rng, env_rng = jax.random.split(rng)
        obs, es = env.reset(env_rng, env_params)

        def _cycle(carry: tuple, _: None) -> tuple:
            es_c, obs_c, r = carry
            r, p_rng = jax.random.split(r)
            plan = sample_plan(
                apply_eval, params, p_rng, obs_c,
                num_actions, config["PLAN_HORIZON"],
                num_steps=config["VAL_DIFFUSION_STEPS"],
                schedule_fn=schedule_fn,
                remask_strategy=config["REMASK_STRATEGY"],
                eta=config["ETA"],
                use_loop=config["USE_LOOP"],
                t_on=config["T_ON"],
                t_off=config["T_OFF"],
                temperature=config["TEMPERATURE"],
                top_p=config["TOP_P"],
            )

            def _step(c: tuple, step_i: jax.Array) -> tuple:
                es_i, obs_i, r_i = c
                action = plan[:, step_i]
                r_i, s_rng = jax.random.split(r_i)
                obs_next, es_next, reward, done, info = env.step(
                    s_rng, es_i, action, env_params,
                )
                return (es_next, obs_next, r_i), (action, reward, done, info["returned_episode"])

            (es_c, obs_c, r), step_data = jax.lax.scan(
                _step, (es_c, obs_c, r), jnp.arange(config["EVAL_REPLAN"]),
            )
            return (es_c, obs_c, r), step_data

        _, cycle_data = jax.lax.scan(_cycle, (es, obs, rng), None, n_cycles)
        # cycle_data: each element is [n_cycles, replan_steps, num_envs]
        actions, rewards, dones, returned = cycle_data
        # Flatten cycles: [n_cycles * replan_steps, num_envs]
        actions = actions.reshape(-1, actions.shape[-1])
        rewards = rewards.reshape(-1, rewards.shape[-1])
        dones = dones.reshape(-1, dones.shape[-1])
        returned = returned.reshape(-1, returned.shape[-1])
        return actions, rewards, dones, returned

    return collect


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


@dataclass
class ActionDistMetrics:
    """Aggregated action distribution metrics for one model.

    Args:
        action_counts:     ``[num_actions]`` raw counts.
        action_probs:      ``[num_actions]`` normalised probabilities.
        entropy:           Shannon entropy of action distribution.
        effective_actions: Number of actions with > 1% probability.
        gini:              Gini coefficient (0=uniform, 1=degenerate).
        transition_matrix: ``[num_actions, num_actions]`` row-normalised
                           bigram P(next | current).
        mean_return:       Mean episode return.
        win_rate:          Fraction of episodes with positive return.
    """

    action_counts: np.ndarray
    action_probs: np.ndarray
    entropy: float
    effective_actions: int
    gini: float
    transition_matrix: np.ndarray
    mean_return: float
    win_rate: float


@dataclass
class ActionDistComparison:
    """Pairwise comparison between pretrained and finetuned action dists.

    Args:
        js_divergence:  Jensen-Shannon divergence.
        kl_pre_post:    KL(pretrained || finetuned).
        kl_post_pre:    KL(finetuned || pretrained).
        tv_distance:    Total variation distance.
        pre_metrics:    Pretrained action distribution metrics.
        post_metrics:   Finetuned action distribution metrics.
    """

    js_divergence: float
    kl_pre_post: float
    kl_post_pre: float
    tv_distance: float
    pre_metrics: ActionDistMetrics
    post_metrics: ActionDistMetrics


def _compute_metrics(
    actions: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    returned_episode: np.ndarray,
    num_actions: int,
    win_threshold: float,
) -> ActionDistMetrics:
    """Compute action distribution metrics from raw rollout data.

    Args:
        actions:          ``[T, E]`` int32 actions taken.
        rewards:          ``[T, E]`` per-step rewards.
        dones:            ``[T, E]`` done flags.
        returned_episode: ``[T, E]`` returned-episode flags.
        num_actions:      Size of action space.
        win_threshold:    Minimum return to count as a win.

    Returns:
        ``ActionDistMetrics`` instance.
    """
    flat_actions = actions.ravel()
    counts = np.bincount(flat_actions, minlength=num_actions).astype(np.float64)
    probs = counts / max(counts.sum(), 1)

    # Entropy
    log_probs = np.log(probs + _EPS)
    entropy = float(-np.sum(probs * log_probs))

    # Effective actions (>1% probability)
    effective = int(np.sum(probs > 0.01))

    # Gini coefficient
    sorted_probs = np.sort(probs)
    n = len(sorted_probs)
    index = np.arange(1, n + 1)
    gini = float((2.0 * np.sum(index * sorted_probs) / (n * np.sum(sorted_probs) + _EPS)) - (n + 1) / n)

    # Transition matrix (bigrams) — vectorised over all envs
    trans = np.zeros((num_actions, num_actions), dtype=np.float64)
    curr = actions[:-1].ravel()
    nxt = actions[1:].ravel()
    np.add.at(trans, (curr, nxt), 1.0)
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_normed = trans / np.maximum(row_sums, 1.0)

    # Episode returns: use returned_episode flag to extract completed episodes
    ep_returns_flat = rewards.sum(axis=0)  # per-env total reward
    mean_return = float(np.mean(ep_returns_flat))
    win_rate = float(np.mean(ep_returns_flat > win_threshold))

    return ActionDistMetrics(
        action_counts=counts,
        action_probs=probs,
        entropy=entropy,
        effective_actions=effective,
        gini=gini,
        transition_matrix=trans_normed,
        mean_return=mean_return,
        win_rate=win_rate,
    )


def _compute_divergences(
    pre: ActionDistMetrics,
    post: ActionDistMetrics,
) -> tuple[float, float, float, float]:
    """Compute pairwise divergence metrics between two distributions.

    Args:
        pre:  Pretrained action distribution metrics.
        post: Finetuned action distribution metrics.

    Returns:
        Tuple of (js_divergence, kl_pre_post, kl_post_pre, tv_distance).
    """
    p = pre.action_probs + _EPS
    q = post.action_probs + _EPS
    p = p / p.sum()
    q = q / q.sum()

    # KL divergences
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))

    # Jensen-Shannon
    m = 0.5 * (p + q)
    js = float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

    # Total variation
    tv = float(0.5 * np.sum(np.abs(p - q)))

    return js, kl_pq, kl_qp, tv


# ---------------------------------------------------------------------------
# Main collection + analysis entry point
# ---------------------------------------------------------------------------


def collect_action_statistics(
    pretrained_params: Any,
    finetuned_params: Any,
    apply_eval: Callable,
    env: Any,
    env_params: Any,
    config: dict,
    rng: jax.Array,
) -> ActionDistComparison:
    """Roll out both models and compute action distribution comparison.

    Args:
        pretrained_params: Pretrained model parameters.
        finetuned_params:  Finetuned model parameters.
        apply_eval:        Eval apply fn (no dropout).
        env:               Wrapped Craftax environment.
        env_params:        Environment parameters.
        config:            UPPERCASE config dict (must include NUM_ACTIONS).
        rng:               PRNG key.

    Returns:
        ``ActionDistComparison`` with all divergence metrics.
    """
    num_actions = config["NUM_ACTIONS"]
    win_threshold = config.get("WIN_THRESHOLD", 0.5)

    collector = _build_action_collector(env, env_params, apply_eval, config)

    rng, pre_rng, post_rng = jax.random.split(rng, 3)
    pre_actions, pre_rewards, pre_dones, pre_ret = collector(pretrained_params, pre_rng)
    post_actions, post_rewards, post_dones, post_ret = collector(finetuned_params, post_rng)

    # Move to host
    pre_actions = np.asarray(jax.device_get(pre_actions))
    pre_rewards = np.asarray(jax.device_get(pre_rewards))
    pre_dones = np.asarray(jax.device_get(pre_dones))
    pre_ret = np.asarray(jax.device_get(pre_ret))
    post_actions = np.asarray(jax.device_get(post_actions))
    post_rewards = np.asarray(jax.device_get(post_rewards))
    post_dones = np.asarray(jax.device_get(post_dones))
    post_ret = np.asarray(jax.device_get(post_ret))

    pre_metrics = _compute_metrics(pre_actions, pre_rewards, pre_dones, pre_ret, num_actions, win_threshold)
    post_metrics = _compute_metrics(post_actions, post_rewards, post_dones, post_ret, num_actions, win_threshold)
    js, kl_pq, kl_qp, tv = _compute_divergences(pre_metrics, post_metrics)

    return ActionDistComparison(
        js_divergence=js,
        kl_pre_post=kl_pq,
        kl_post_pre=kl_qp,
        tv_distance=tv,
        pre_metrics=pre_metrics,
        post_metrics=post_metrics,
    )


def interpret_results(comparison: ActionDistComparison) -> str:
    """Interpret JS divergence as a human-readable verdict.

    Args:
        comparison: Action distribution comparison.

    Returns:
        Interpretation string.
    """
    js = comparison.js_divergence
    if js < 0.05:
        return "representation_drift_only"
    if js < 0.15:
        return "mixed_behavioural_change"
    return "mode_collapse"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure and close.

    Args:
        fig:  Matplotlib figure.
        path: Output file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_action_frequencies(
    comparison: ActionDistComparison,
    ablation_name: str,
    output_dir: Path,
) -> None:
    """Side-by-side action frequency bar chart.

    Args:
        comparison:    Action distribution comparison.
        ablation_name: Name of the ablation (for title/filename).
        output_dir:    Output directory.
    """
    n = len(comparison.pre_metrics.action_probs)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), 5))
    ax.bar(x - width / 2, comparison.pre_metrics.action_probs, width, label="Pretrained", alpha=0.8, color="#1976D2")
    ax.bar(x + width / 2, comparison.post_metrics.action_probs, width, label="Finetuned", alpha=0.8, color="#F57C00")
    ax.set_xlabel("Action")
    ax.set_ylabel("Probability")
    ax.set_title(f"Action Frequency: {ablation_name}")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir / f"action_freq_{ablation_name}.png")


def plot_transition_matrices(
    comparison: ActionDistComparison,
    ablation_name: str,
    output_dir: Path,
) -> None:
    """Three-panel transition matrix heatmaps: pre, post, difference.

    Args:
        comparison:    Action distribution comparison.
        ablation_name: Name of the ablation.
        output_dir:    Output directory.
    """
    pre_t = comparison.pre_metrics.transition_matrix
    post_t = comparison.post_metrics.transition_matrix
    diff = post_t - pre_t

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Pretrained", "Finetuned", "Difference (Post - Pre)"]
    matrices = [pre_t, post_t, diff]
    cmaps = ["Blues", "Oranges", "RdBu_r"]
    vmins = [0, 0, -np.abs(diff).max() or -0.1]
    vmaxs = [1, 1, np.abs(diff).max() or 0.1]

    for ax, title, mat, cmap, vmin, vmax in zip(axes, titles, matrices, cmaps, vmins, vmaxs):
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Next Action")
        ax.set_ylabel("Current Action")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"Action Transition Matrices: {ablation_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir / f"transition_matrix_{ablation_name}.png")


def plot_metrics_dashboard(
    comparison: ActionDistComparison,
    ablation_name: str,
    output_dir: Path,
) -> None:
    """2x2 metrics dashboard: entropy, effective actions, Gini, divergences.

    Args:
        comparison:    Action distribution comparison.
        ablation_name: Name of the ablation.
        output_dir:    Output directory.
    """
    pre = comparison.pre_metrics
    post = comparison.post_metrics

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"Action Distribution Metrics: {ablation_name}", fontsize=13, fontweight="bold")

    # Entropy comparison
    ax = axes[0, 0]
    ax.bar(["Pretrained", "Finetuned"], [pre.entropy, post.entropy], color=["#1976D2", "#F57C00"], alpha=0.8)
    ax.set_ylabel("Shannon Entropy")
    ax.set_title("Action Entropy")

    # Effective actions
    ax = axes[0, 1]
    ax.bar(["Pretrained", "Finetuned"], [pre.effective_actions, post.effective_actions],
           color=["#1976D2", "#F57C00"], alpha=0.8)
    ax.set_ylabel("Count (prob > 1%)")
    ax.set_title("Effective Actions")

    # Gini coefficient
    ax = axes[1, 0]
    ax.bar(["Pretrained", "Finetuned"], [pre.gini, post.gini], color=["#1976D2", "#F57C00"], alpha=0.8)
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Action Concentration (Gini)")

    # Divergence metrics
    ax = axes[1, 1]
    metrics = {
        "JS": comparison.js_divergence,
        "KL(pre||post)": comparison.kl_pre_post,
        "KL(post||pre)": comparison.kl_post_pre,
        "TV": comparison.tv_distance,
    }
    ax.bar(metrics.keys(), metrics.values(), color="#00897B", alpha=0.8)
    ax.set_ylabel("Divergence")
    ax.set_title("Distribution Divergences")

    fig.tight_layout()
    _save(fig, output_dir / f"action_metrics_{ablation_name}.png")


def plot_js_comparison(
    comparisons: dict[str, ActionDistComparison],
    output_dir: Path,
) -> None:
    """Bar chart of JS divergence across all ablations.

    Horizontal lines at 0.05 (drift-only) and 0.15 (mode collapse).

    Args:
        comparisons: Dict mapping ablation_name -> ActionDistComparison.
        output_dir:  Output directory.
    """
    if not comparisons:
        return

    sorted_items = sorted(comparisons.items(), key=lambda x: x[1].js_divergence, reverse=True)
    names = [n for n, _ in sorted_items]
    js_vals = [c.js_divergence for _, c in sorted_items]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), 5))
    colors = []
    for js in js_vals:
        if js < 0.05:
            colors.append("#4CAF50")  # green: drift only
        elif js < 0.15:
            colors.append("#FF9800")  # orange: mixed
        else:
            colors.append("#F44336")  # red: mode collapse
    ax.bar(range(len(names)), js_vals, color=colors, alpha=0.8, edgecolor="white")
    ax.axhline(0.05, linestyle="--", color="green", alpha=0.6, label="drift threshold")
    ax.axhline(0.15, linestyle="--", color="red", alpha=0.6, label="collapse threshold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Jensen-Shannon Divergence")
    ax.set_title("Action Distribution Shift per Ablation")
    ax.legend()
    fig.tight_layout()
    _save(fig, output_dir / "js_divergence_comparison.png")


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def run_action_distribution_analysis(
    results: dict[str, dict],
    pretrained_params: Any,
    apply_eval: Callable,
    env: Any,
    env_params: Any,
    config: dict,
    rng: jax.Array,
    output_dir: Path,
) -> dict[str, ActionDistComparison]:
    """Run action distribution analysis for all completed ablations.

    Called after training. For each ablation, rolls out both the pretrained
    and finetuned models, computes divergence metrics, and generates plots.

    Args:
        results:           Dict mapping name -> {"final_params": params, ...}.
        pretrained_params: Pretrained model parameters.
        apply_eval:        Eval apply fn.
        env:               Wrapped Craftax environment.
        env_params:        Environment parameters.
        config:            UPPERCASE config dict.
        rng:               PRNG key.
        output_dir:        Root output directory.

    Returns:
        Dict mapping ablation_name -> ActionDistComparison.
    """
    fig_dir = output_dir / "figures" / "action_dist"
    fig_dir.mkdir(parents=True, exist_ok=True)

    comparisons: dict[str, ActionDistComparison] = {}

    for name, res in results.items():
        finetuned_params = res.get("final_params")
        if finetuned_params is None:
            logger.warning("Skipping action dist for %s: no final_params", name)
            continue

        rng, abl_rng = jax.random.split(rng)
        logger.info("Action distribution analysis: %s", name)

        comparison = collect_action_statistics(
            pretrained_params, finetuned_params, apply_eval,
            env, env_params, config, abl_rng,
        )
        comparisons[name] = comparison

        verdict = interpret_results(comparison)
        logger.info(
            "  %s: JS=%.4f KL(pre||post)=%.4f TV=%.4f -> %s",
            name, comparison.js_divergence, comparison.kl_pre_post,
            comparison.tv_distance, verdict,
        )

        # Per-ablation plots
        plot_action_frequencies(comparison, name, fig_dir)
        plot_transition_matrices(comparison, name, fig_dir)
        plot_metrics_dashboard(comparison, name, fig_dir)

    # Cross-ablation JS comparison
    plot_js_comparison(comparisons, fig_dir)

    logger.info("Action distribution analysis complete. Figures in %s", fig_dir)
    return comparisons
