# ReMDM Experiments

Research and diagnostic scripts for investigating RL fine-tuning of the ReMDM diffusion planner.
These scripts are **standalone research code** — they import from `src/` but do not modify it.

---

## `rl_finetuning/` — RL Fine-Tuning Ablation Suite

Diagnoses why RL fine-tuning of the diffusion model collapses and which interventions fix it.
Implements **25 ablations** across four groups, plus a comprehensive diagnostic and analysis pipeline.

### Directory structure

```
rl_finetuning/
├── run_ablations.py          # CLI entry point
├── ablations/
│   ├── losses.py             # All loss/objective variants as factory functions
│   ├── optimizers.py         # LLRD, LoRA, gradient surgery, param masking
│   ├── registry.py           # AblationSpec dataclass + REGISTRY (25 ablations)
│   └── training.py           # make_run_ablation() factory + AblationHistory dataclass
├── diagnostics/
│   ├── gradient.py           # Grad alignment, per-layer norms, surgery metrics
│   ├── representation.py     # KL drift, CKA similarity, activation norms
│   └── timestep.py           # t-bin gradient norms, per-t loss decomposition
├── analysis/
│   ├── action_distribution.py # Pre- vs post-finetuning action dist divergences + plots
│   ├── plots.py              # All matplotlib figure generators
│   ├── tables.py             # Summary tables as polars DataFrames + LaTeX export
│   └── report.py             # diagnosis.md + decision tree figure
└── configs/
    ├── ablations_default.yaml   # Full-run hyperparameters (self-contained)
    └── ablations_fast.yaml      # Smoke-test overrides (50 iterations, 16 envs)
```

### Usage

The pretrained diffusion checkpoint can come from either offline training (`--mode offline`) or DAgger online training (`--mode online`) — the checkpoint format is identical. Use `--checkpoint_path` (or its alias `--offline_checkpoint_path`) to point to it.

Checkpoint paths accept `wandb:` prefixed artifact references (e.g., `wandb:team/project/artifact:latest`), which are downloaded automatically before training begins.

**Smoke test (2 ablations, fast config):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl kl_penalty \
    --fast \
    --checkpoint_path $PRETRAINED_CKPT \
    --ppo_checkpoint_path $PPO_CKPT
```

**Full suite (all 25 ablations):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --config configs/defaults.yaml \
    --ablations_config experiments/rl_finetuning/configs/ablations_default.yaml \
    --all \
    --num_seeds 3 \
    --checkpoint_path $PRETRAINED_CKPT \
    --ppo_checkpoint_path $PPO_CKPT \
    --use_wandb
```

**Specific ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations ewc lora gradient_surgery trust_region_kl \
    --checkpoint_path $PRETRAINED_CKPT \
    --ppo_checkpoint_path $PPO_CKPT
```

**From a W&B artifact:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl \
    --checkpoint_path wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy:latest \
    --ppo_checkpoint_path wandb:my-team/ppo-craftax/ppo-rnn-policy:best
```

**Re-plot from saved results (no training):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --analyze_only \
    --results_path experiments/rl_finetuning/outputs/run_20250101_120000/results.json
```

**Merge multi-GPU results:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --merge outputs/gpu0/results.json outputs/gpu1/results.json \
    --output_dir experiments/rl_finetuning/outputs/merged/
```

**List all ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py --list
```

### Ablations

| Group | Name | Tests |
|---|---|---|
| Baseline | `baseline_rl` | Standard return-weighted ELBO |
| **A: Regularisation** | `kl_penalty` | Soft KL constraint vs. pretrained |
| | `ewc` | Elastic Weight Consolidation (Fisher diagonal) |
| | `llrd` | Layer-wise Learning Rate Decay |
| | `lora` | Low-Rank Adaptation of attention projections |
| | `mixed_replay` | Offline PPO data mixed into online batches |
| | `trust_region_kl` | Hard KL trust region via quadratic barrier |
| **B: Training Signal** | `t_curriculum` | Anneal t range high→low over training |
| | `entropy_bonus` | Entropy regularisation for action diversity |
| | `gradient_surgery` | PCGrad: project conflicting RL/BC gradients |
| | `advantage_clip` | PPO-style advantage clipping [1-ε, 1+ε] |
| | `normalized_adv` | Std-normalised advantages |
| | `bc_wins` | Uniform ELBO on win windows (no advantage weighting) |
| | `low_t` | ELBO restricted to low-t (fine-detail) regime |
| **C: Architecture** | `frozen_backbone` | Only train the output head |
| | `head_only` | Only train the final linear projection |
| | `attention_only` | Only train attention weights (Q/K/V/O) |
| | `ffn_only` | Only train FFN layers |
| | `layer_ablation_top1` | Only train top-1 transformer block |
| | `layer_ablation_top2` | Only train top-2 transformer blocks |
| | `layer_ablation_top3` | Only train top-3 transformer blocks |
| **D: Data Quality** | `reward_filtering` | Top-75th-percentile return windows only |
| | `running_stats` | EMA running mean/std for advantage normalisation |
| | `action_diversity` | Discard degenerate (all-same-action) plans |
| | `reward_model` | MLP reward model soft-weighting of advantages |

### Output structure

```
experiments/rl_finetuning/outputs/{run_id}/
├── results.json               # All histories + final scores (machine-readable; see schema below)
├── diagnosis.md               # Human-readable verdict + evidence + recommendations
├── figures/
│   ├── curves_{name}.png                  # Per-ablation training curves (2×3 grid)
│   ├── final_score_comparison.png
│   ├── eval_scores_over_training.png
│   ├── score_delta_over_baseline_rl.png
│   ├── gradient_alignment.png
│   ├── gradient_conflict_map.png
│   ├── per_layer_grad_heatmap_{name}.png
│   ├── representation_drift.png
│   ├── cka_similarity.png
│   ├── t_distribution_analysis.png
│   ├── t_bin_grad_norms_{name}.png
│   ├── win_rate_and_effective_batch_size.png
│   ├── achievement_breakdown.png          # Start vs end achievement rates (stacked bars)
│   ├── achievement_collapse_{name}.png    # Per-ablation achievement heatmap over time
│   ├── diagnosis_decision_tree.png
│   └── action_dist/
│       ├── action_freq_{name}.png         # Side-by-side pre/post action frequency bars
│       ├── transition_matrix_{name}.png   # 3-panel heatmap (pre, post, difference)
│       ├── action_metrics_{name}.png      # 2x2 dashboard (entropy, effective, Gini, divergences)
│       └── js_divergence_comparison.png   # Cross-ablation JS divergence bar chart
└── tables/
    ├── main_results.{csv,tex}
    ├── gradient_analysis.{csv,tex}
    ├── t_distribution.{csv,tex}
    ├── forgetting_analysis.{csv,tex}
    ├── hypothesis_verdict.{csv,tex}
    └── achievement_summary.{csv,tex}      # Per-achievement final unlock rates
```

**`results.json` schema:**
```json
{
  "pretrained_score": 0.1234,
  "pretrained_ach_rates": {"achievement_collect_wood": 0.42, ...},
  "config": {"MAX_ITER": 1000, ...},
  "ablations": {
    "kl_penalty": {
      "score": 0.1456,        // mean across seeds
      "score_std": 0.008,     // std across seeds (0.0 if num_seeds=1)
      "all_scores": [0.1456], // per-seed scores
      "history": { ... }      // AblationHistory serialised
    }
  }
}
```

`results.json` is written incrementally after each ablation completes — a partial file with
N of 25 ablations is fully valid and loadable by `--analyze_only --results_path`.

### Diagnostic metrics collected

| Metric | Frequency | What it answers |
|---|---|---|
| Eval score | every `eval_every` iters | Primary performance |
| Training loss | every 10 iters | Optimisation signal |
| Env score | every 10 iters | Online rollout quality |
| Gradient alignment (cos sim) | every `grad_align_every` | Is the RL gradient useful? |
| Per-layer gradient norms | every `per_layer_every` | Which layers collapse? |
| KL drift from pretrained | every `repr_drift_every` | How much has the model changed? |
| CKA similarity | every `cka_every` | Representational drift (activation level) |
| t-bin gradient norms | every `t_analysis_every` | Is high-t gradient biased? |
| Win rate | every 10 iters | Signal sparsity |
| Effective batch size | every 10 iters | Gradient concentration |
| Gradient surgery fraction | every `grad_align_every` | PCGrad projected mass |
| Action dist JS divergence | post-training | Mode collapse vs drift? |
| Action dist KL / TV | post-training | Magnitude of behavioural shift |
| Action transition matrix | post-training | Bigram structure change |

### W&B namespace

All metrics are logged under `ablations/{method_name}/{metric}`, e.g.:
- `ablations/kl_penalty/eval_score`
- `ablations/gradient_surgery/grad_align`
- `ablations/ewc/repr_drift_kl`
