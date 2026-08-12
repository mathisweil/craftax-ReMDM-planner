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
    ├── ablations_default.yaml              # Base hyperparameters for every ablation run
    ├── ablations_fast.yaml                 # Smoke-test overlay (50 iterations, 16 envs)
    ├── ablations_final_classic_ucl.yaml    # Matches configs/final_classic_ucl.yaml   (UCL 3090 Ti, seed 42)
    ├── ablations_final_classic_qmul.yaml   # Matches configs/final_classic_qmul.yaml  (QMUL H200, seed 43)
    ├── ablations_final_craftax_ucl.yaml    # Matches configs/final_craftax_ucl.yaml   (UCL 4090,   seed 42)
    └── ablations_final_craftax_qmul.yaml   # Matches configs/final_craftax_qmul.yaml  (QMUL H200, seed 43)
```

The `ablations_final_*` presets pin the transformer architecture and sampling hyperparameters to exactly match the corresponding pretrained DAgger checkpoints under `configs/final_*`, so they can be consumed without re-specifying every field.

#### Config precedence

Lowest to highest:

```
configs/defaults.yaml -> ablations_default.yaml -> machine config -> ablations_fast.yaml (--fast only) -> CLI flags
```

Any file given to `--ablations-config` layers on top of `ablations_default.yaml` automatically, so the `ablations_final_*` presets carry only their own deltas. A config may set `extends: <file>` to name a different base (resolved relative to the declaring file), or `extends:` with an empty value to opt out of inheritance entirely. Cycles and missing bases raise an error rather than falling back silently, and `extends` is stripped before the config namespace is built.

`ablations_fast.yaml` is deliberately **not** part of that chain: `--fast` reads it raw and overlays it last. Routing it through `extends` would drag `ablations_default.yaml`'s values back over whichever machine config is in use.

**Presets hold only deltas, never restate a default.** A key belongs in a machine config only if its value differs from `ablations_default.yaml`. Restating a default silently pins the preset when the base later moves.

### Usage

The pretrained diffusion checkpoint can come from either offline training (`--mode offline`) or DAgger online training (`--mode online`) — the checkpoint format is identical. Use `--checkpoint` to point to it. For DAgger runs, either the final (`{env}-policy`) or best-validation (`{env}-policy-best`) artifact produced by `--mode online` can be consumed directly.

Checkpoint paths accept `wandb:` prefixed artifact references (e.g., `wandb:team/project/artifact:latest`), which are downloaded automatically before training begins.

**Smoke test (2 ablations, fast config):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl kl_penalty \
    --fast \
    --checkpoint $PRETRAINED_CKPT \
    --ppo-checkpoint $PPO_CKPT
```

**Full suite (all 25 ablations):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --config configs/defaults.yaml \
    --ablations-config experiments/rl_finetuning/configs/ablations_default.yaml \
    --all \
    --num-seeds 3 \
    --checkpoint $PRETRAINED_CKPT \
    --ppo-checkpoint $PPO_CKPT \
    --use-wandb
```

**Full suite against a pinned `final_*` checkpoint:**
```bash
# Craftax Classic, UCL hardware (seed 42 checkpoint)
python experiments/rl_finetuning/run_ablations.py \
    --ablations-config experiments/rl_finetuning/configs/ablations_final_classic_ucl.yaml \
    --all --num-seeds 3 \
    --checkpoint wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy-best:latest \
    --ppo-checkpoint wandb:my-team/ppo-craftax/ppo-rnn-policy:best \
    --use-wandb
```

**Specific ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations ewc lora gradient_surgery trust_region_kl \
    --checkpoint $PRETRAINED_CKPT \
    --ppo-checkpoint $PPO_CKPT
```

**From a W&B artifact:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl \
    --checkpoint wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy:latest \
    --ppo-checkpoint wandb:my-team/ppo-craftax/ppo-rnn-policy:best
```

**Re-plot from saved results (no training):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --analyze-only \
    --results-path experiments/rl_finetuning/outputs/run_20250101_120000/results.json
```

**Merge multi-GPU results:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --merge outputs/gpu0/results.json outputs/gpu1/results.json \
    --output-dir experiments/rl_finetuning/outputs/merged/
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
N of 25 ablations is fully valid and loadable by `--analyze-only --results-path`.

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
