# ReMDM Experiments

Research and diagnostic scripts for investigating RL fine-tuning of the ReMDM diffusion planner.
These scripts are **standalone research code** — they import from `src/` but do not modify it.

---

## `rl_finetuning/` — RL Fine-Tuning Ablation Suite

Diagnoses why RL fine-tuning of the diffusion model collapses and which interventions fix it.
Implements **22 ablations** across four groups, plus a comprehensive diagnostic and analysis pipeline.

### Directory structure

```
rl_finetuning/
├── run_ablations.py          # CLI entry point
├── ablations/
│   ├── losses.py             # All loss/objective variants as factory functions
│   ├── optimizers.py         # LLRD, LoRA, EWC, gradient surgery, param masking
│   ├── registry.py           # AblationSpec dataclass + REGISTRY (22 ablations)
│   └── training.py           # run_ablation() loop + AblationHistory dataclass
├── diagnostics/
│   ├── gradient.py           # Grad alignment, per-layer norms, surgery metrics
│   ├── representation.py     # KL drift, CKA similarity, activation norms
│   └── timestep.py           # t-bin gradient norms, per-t loss decomposition
├── analysis/
│   ├── plots.py              # All matplotlib figure generators
│   ├── tables.py             # Summary tables as polars DataFrames + LaTeX export
│   └── report.py             # diagnosis.md + decision tree figure
└── configs/
    ├── ablations_default.yaml   # Full-run hyperparameters
    └── ablations_fast.yaml      # Smoke-test overrides (50 iterations, 16 envs)
```

### Usage

**Smoke test (2 ablations, fast config):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl kl_penalty \
    --fast \
    --offline_checkpoint_path $OFFLINE_CKPT \
    --ppo_checkpoint_path $PPO_CKPT
```

**Full suite (all 22 ablations):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --config configs/defaults.yaml \
    --ablations_config experiments/rl_finetuning/configs/ablations_default.yaml \
    --all \
    --num_seeds 3 \
    --offline_checkpoint_path $OFFLINE_CKPT \
    --ppo_checkpoint_path $PPO_CKPT \
    --use_wandb
```

**Specific ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations ewc lora gradient_surgery trust_region_kl \
    --offline_checkpoint_path $OFFLINE_CKPT \
    --ppo_checkpoint_path $PPO_CKPT
```

**Re-plot from saved results (no training):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --analyze_only \
    --results_path experiments/rl_finetuning/outputs/run_20250101_120000/results.json
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
| | `normalized_adv` | Std-normalised advantages (GRPO-style) |
| | `bc_wins` | Uniform ELBO on win windows (no advantage weighting) |
| | `low_t` | ELBO restricted to low-t (fine-detail) regime |
| **C: Architecture** | `frozen_backbone` | Only train the output head |
| | `head_only` | Only train the final linear projection |
| | `attention_only` | Only train attention weights (Q/K/V/O) |
| | `ffn_only` | Only train FFN layers |
| | `layer_ablation_top1/2/3` | Only train top-N transformer blocks |
| **D: Data Quality** | `reward_filtering` | Top-75th-percentile return windows only |
| | `running_stats` | EMA running mean/std for advantage normalisation |
| | `action_diversity` | Discard degenerate (all-same-action) plans |
| | `reward_model` | MLP reward model soft-weighting of advantages |

### Output structure

```
experiments/rl_finetuning/outputs/{run_id}/
├── results.json               # All histories + final scores (machine-readable)
├── diagnosis.md               # Human-readable verdict + evidence + recommendations
├── figures/
│   ├── curves_{name}.png      # Per-ablation training curves (2×3 grid)
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
│   └── diagnosis_decision_tree.png
└── tables/
    ├── main_results.{csv,tex}
    ├── gradient_analysis.{csv,tex}
    ├── t_distribution.{csv,tex}
    ├── forgetting_analysis.{csv,tex}
    └── hypothesis_verdict.{csv,tex}
```

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

### W&B namespace

All metrics are logged under `ablations/{method_name}/{metric}`, e.g.:
- `ablations/kl_penalty/eval_score`
- `ablations/gradient_surgery/grad_align`
- `ablations/ewc/repr_drift_kl`
