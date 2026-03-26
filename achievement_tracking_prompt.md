# Claude Code Prompt: Per-Achievement Metric Tracking

Extend the RL fine-tuning ablation suite to track per-achievement metrics throughout training.

In `experiments/rl_finetuning/`, make the following changes:

1. **`ablations/training.py`** — in `eval_policy` and the `env_score` extraction inside `run_ablation`, extract per-achievement unlock rates from the `info` dict (Craftax's `LogWrapper` populates `info["achievements"]` or similar keys with per-achievement counts). Use the same `returned_episode` mask already used for `returned_episode_returns` to compute per-achievement rates correctly.

2. **`ablations/registry.py` or `training.py`** — add a `per_achievement_rates` field to `AblationHistory` (a `list[dict[str, float]]` keyed by achievement name, one entry per eval checkpoint).

3. **`analysis/plots.py`** — add two new figures:
   - A **stacked bar chart** showing achievement breakdown at start vs. end of training for each ablation — reveals whether "neutral" score actually hides a shift in which achievements are being unlocked
   - A **achievement collapse heatmap** — rows = achievements, columns = eval iterations, colour = unlock rate; one per ablation, showing which achievements are lost first during collapse

4. **`analysis/tables.py`** — add a per-achievement summary table: rows = achievements, columns = ablation methods, values = final unlock rate, with a delta column vs. pretrained baseline.

Keep everything consistent with the existing patterns in the codebase — the `eval_policy` JAX function, the `AblationHistory` dataclass serialisation to JSON, and the `generate_all_plots` interface.
