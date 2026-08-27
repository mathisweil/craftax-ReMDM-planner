# ReMDM Experiments

Research and diagnostic scripts for investigating RL fine-tuning of the ReMDM diffusion planner.
These scripts are **standalone research code** — they import from `src/` but do not modify it.

---

## `rl_finetuning/` — RL Fine-Tuning Ablation Suite

Diagnoses why RL fine-tuning of the diffusion model collapses and which interventions fix it.
Implements **26 ablations**: a baseline plus four groups (A: Regularisation, B: Training Signal, C: Architecture, D: Data Quality), with a comprehensive diagnostic and analysis pipeline.

**Training data is on-policy.** Each iteration rolls the *current* model out under its EMA weights (`diffusion_steps_collect` denoising steps per plan, `num_steps // plan_horizon` plan cycles) and trains on those windows, weighted by each window's own H-step reward sum. The suite therefore needs no expert: `--ppo-checkpoint` is gone, and only the pretrained diffusion `--checkpoint` is required.

### Directory structure

```
rl_finetuning/
├── run_ablations.py          # CLI entry point
├── ablations/
│   ├── losses.py             # All loss/objective variants as factory functions
│   ├── optimizers.py         # LLRD, LoRA, gradient surgery, param masking
│   ├── registry.py           # AblationSpec dataclass + REGISTRY (26 ablations)
│   └── training.py           # make_run_ablation() factory + AblationHistory dataclass
├── diagnostics/
│   ├── gradient.py           # Grad alignment, per-layer norms, surgery metrics
│   ├── representation.py     # KL drift, CKA similarity, activation norms
│   └── timestep.py           # t-bin gradient norms, per-t loss decomposition
├── analysis/
│   ├── action_distribution.py # Pre- vs post-finetuning action dist divergences + plots
│   ├── gdelta.py             # Return term g_delta of the decomposition (no training)
│   ├── plots.py              # All matplotlib figure generators
│   ├── tables.py             # Summary tables as polars DataFrames + LaTeX export
│   └── report.py             # diagnosis.md + decision tree figure
└── configs/
    ├── ablations_default.yaml              # Base hyperparameters for every ablation run
    ├── ablations_fast.yaml                 # Smoke-test overlay (50 iterations, 16 envs)
    ├── ablations_final_craftax_classic_gpu_24gb.yaml    # Matches configs/final_craftax_classic_gpu_24gb.yaml   (RTX 3090 Ti, seed 42)
    ├── ablations_final_craftax_classic_gpu_h200.yaml   # Matches configs/final_craftax_classic_gpu_h200.yaml  (H200, seed 43)
    ├── ablations_final_craftax_gpu_24gb.yaml    # Matches configs/final_craftax_gpu_24gb.yaml   (GPU-24GB reference machine (GPU model unrecorded),   seed 42)
    └── ablations_final_craftax_gpu_h200.yaml   # Matches configs/final_craftax_gpu_h200.yaml  (H200, seed 43)
```

`ablations_default.yaml` carries the transformer architecture of the released DAgger checkpoints — 384-dim, 8 heads, 6 layers, `d_ff` 768, `plan_horizon` 32, the same for Classic and Full — so the `ablations_final_*` presets need not restate it. A run against a differently-shaped checkpoint must override those keys, or the model build fails on a shape mismatch.

### Config layering

Lowest to highest:

```
configs/defaults.yaml -> ablations_default.yaml -> machine config -> ablations_fast.yaml (--fast only) -> CLI flags
```

Any file given to `--ablations-config` layers on top of `ablations_default.yaml` automatically, so the `ablations_final_*` presets carry only their own deltas. An ablations config never inherits from another ablations config.

`ablations_fast.yaml` is deliberately **not** layered that way: `--fast` reads it raw and overlays it last.

**Presets hold only deltas, never restate a default** — `tests/test_config.py` enforces it.

The two craftax presets each restate the three keys where Full Craftax departs from the Classic base (`env_name`, `val_diffusion_steps`, `temperature`); with no inheritance between configs there is nowhere shared to put them, so a change to those must be made in both files.

### Compilation cache

The graph is identical across the seeds of one ablation — only the PRNG key differs, and that is a runtime argument — so at `num_seeds: 3` two runs in three are a cache hit, as are reruns and the per-GPU processes of `--merge`. Off unless `jax_compilation_cache_dir` is set, and **`run_ablations.py` has no `--override`**, so it must be set in `configs/defaults.yaml`. Point it at local disk, not an NFS home:

```yaml
# configs/defaults.yaml
jax_compilation_cache_dir: /var/tmp/your-user/jax-cache
```

### Usage

The pretrained diffusion checkpoint can come from either offline training (`--mode offline`) or DAgger online training (`--mode online`). Use `--checkpoint` to point to it. For DAgger runs, either the final (`{env}-policy`) or best-validation (`{env}-policy-best`) artifact produced by `--mode online` can be consumed directly.

Checkpoint paths accept `wandb:` prefixed artifact references (e.g., `wandb:team/project/artifact:latest`), which are downloaded automatically before training begins.

**Smoke test (2 ablations, fast config):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl kl_penalty \
    --fast \
    --checkpoint $PRETRAINED_CKPT
```

**Full suite (all 26 ablations):**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --config configs/defaults.yaml \
    --ablations-config experiments/rl_finetuning/configs/ablations_default.yaml \
    --all \
    --num-seeds 3 \
    --checkpoint $PRETRAINED_CKPT \
    --use-wandb
```

**Full suite against a pinned `final_*` checkpoint:**
```bash
# Craftax Classic, GPU-24GB hardware (seed 42 checkpoint)
python experiments/rl_finetuning/run_ablations.py \
    --ablations-config experiments/rl_finetuning/configs/ablations_final_craftax_classic_gpu_24gb.yaml \
    --all --num-seeds 3 \
    --checkpoint wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy-best:latest \
    --use-wandb
```

**Specific ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations ewc lora gradient_surgery trust_region_kl \
    --checkpoint $PRETRAINED_CKPT
```

**From a W&B artifact:**
```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl \
    --checkpoint wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy:latest
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

**`--merge` only pools runs from configs that agree on result-affecting keys.**
It compares the configs the results files recorded and refuses, naming every
diverging key with both values; a file that records no config is refused too.
Each family's GPU-24GB config is its reference, and the GPU-H200 sibling is **not
poolable** with it:

| Key | Classic GPU-24GB | Classic GPU-H200 | Craftax GPU-24GB | Craftax GPU-H200 | Effect |
|---|---|---|---|---|---|
| `num_envs` | 192 | 64 | 128 | 64 | rollout diversity per iteration |
| `batch_size` | 1024 | 256 | 1024 | 512 | per-update SNR |
| `eval_steps` | 1024 | 512 | 1024 | 512 | noisier score |
| `mixed_replay_buffer_size` | 20000 | 10000 | 10000 | 10000 | replay horizon |

Values above are post-layering: a config's own value, or what it inherits from
`ablations_default.yaml`. Differences in diagnostic cadence (`eval_every`, `cka_every`,
`cka_batch_size`, `per_layer_every`, `repr_drift_every`, `grad_align_every`,
`t_analysis_every`) are wall-clock only and do not affect poolability.

`tests/test_config.py` enforces this: every `ablations_final_*.yaml` must be
declared poolable or not, configs declared poolable must match their family
reference on the result-affecting keys, and the recorded GPU-H200 divergences must
stay accurate. Aligning a GPU-H200 config later fails the test until it is moved to
the poolable set. The key set itself is declared once, in
`run_ablations._RESULT_AFFECTING`, so the classification the tests check and
the refusal `--merge` performs are the same policy.

**List all ablations:**
```bash
python experiments/rl_finetuning/run_ablations.py --list
```

### Measuring the return term (`--measure-gdelta`)

Loads the pretrained checkpoint, collects one on-policy batch from it, and evaluates
`grad L_BC`, `g_delta` and `grad L_RW` on that batch at those parameters under a shared
`(z_t, t)` draw, so the only difference between the three is the weight vector. It repeats
for the four weighting ablations (`baseline_rl`, `advantage_clip`, `normalized_adv`,
`bc_wins`) and reports `CV_A`, `Abar`, `ESS/B`, the norm ratio and the cosine, plus a
shuffled-`delta` null that keeps the weight multiset and destroys its association with each
window's return. No training and no optimiser step occur; it runs on a laptop CPU.

Results land in `gdelta/` under the run's own output directory, beside `results.json`, and
the aggregate additionally produces `tables/gdelta.{csv,tex}`. With `--emit-tex-macros`,
the analysis pass picks the aggregate up and emits the measured quantities as `\rwGdelta*`
macros. Those are kept separate from the `\rwCvA*` macros, which recover `CV_A` from the
ESS logged during training: the two are measured on different batches and do not agree.

Config comes from `--results-path`, so the weight transforms measured are the ones that run
trained under; without it the standard layering applies.

**Reproduction (three rollout seeds, aggregated in one pass):**
```bash
python experiments/rl_finetuning/run_ablations.py --measure-gdelta --gdelta-seeds 0 1 2 \
    --checkpoint checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M \
    --results-path experiments/rl_finetuning/outputs/craftax_classic_ablations/results.json \
    --output-dir experiments/rl_finetuning/outputs/craftax_classic_ablations
```

Seeds run on separate machines are aggregated afterwards with `--gdelta-inputs`, the
counterpart to `--merge`:
```bash
python experiments/rl_finetuning/run_ablations.py --run-id gdelta \
    --gdelta-inputs experiments/rl_finetuning/outputs/gdelta/gdelta_seed{0,1,2}.json
```

A single seed's `ratio_std_draws` / `cos_std_draws` are dispersions over that seed's eight
`(z_t, t)` draws. The aggregate averages the per-seed means and reports the standard
deviation **across seeds**, which is what the paper's table prints.

### Ablations

| Group | Name | Tests |
|---|---|---|
| Baseline | `baseline_rl` | Standard return-weighted ELBO |
| **A: Regularisation** | `kl_penalty` | Soft KL constraint vs. pretrained |
| | `ewc` | Elastic Weight Consolidation (Fisher diagonal) |
| | `llrd` | Layer-wise Learning Rate Decay |
| | `lora` | Low-Rank Adaptation of attention projections |
| | `mixed_replay` | Self-replay: the run's own past online windows resampled into each batch |
| | `trust_region_kl` | Hard KL trust region via quadratic barrier |
| **B: Training Signal** | `t_curriculum` | Anneal t range high→low over training |
| | `entropy_bonus` | Entropy regularisation for action diversity |
| | `gradient_surgery` | PCGrad: project conflicting RL/BC gradients |
| | `advantage_clip` | PPO-style advantage clipping [1-ε, 1+ε] |
| | `normalized_adv` | Std-normalised advantages |
| | `bc_wins` | Uniform ELBO on win windows (no advantage weighting) |
| | `bc_all` | Uniform ELBO on all rollout windows (no advantage weighting) |
| | `low_t` | ELBO restricted to low-t (fine-detail) regime |
| **C: Architecture** | `frozen_backbone` | Train the action head + token embeddings (backbone frozen) |
| | `head_only` | Train only the final action projection |
| | `attention_only` | Train only the attention projections (Q/K/V/O) |
| | `ffn_only` | Train only the per-block FFN layers |
| | `layer_ablation_top1` | Train only the top-1 transformer block + head |
| | `layer_ablation_top2` | Train only the top-2 transformer blocks + head |
| | `layer_ablation_top3` | Train only the top-3 transformer blocks + head |
| **D: Data Quality** | `reward_filtering` | Top-75th-percentile return windows only |
| | `running_stats` | EMA running mean/std for advantage normalisation |
| | `action_diversity` | Discard degenerate (all-same-action) plans |
| | `reward_model` | MLP reward model soft-weighting of advantages |

### Output structure

```
experiments/rl_finetuning/outputs/{run_id}/
├── results.json               # All histories + final scores (machine-readable; see schema below)
├── diagnosis.md               # Human-readable verdict + evidence + recommendations
├── checkpoint_{name}/         # Per-ablation fine-tuned params, last seed (Orbax)
├── gdelta/                    # --measure-gdelta only
│   ├── gdelta_seed{n}.json    # Per rollout seed; +/- within is across that seed's draws
│   └── gdelta_aggregate.json  # Across seeds; the dispersion the paper's table prints
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
│   ├── t_bin_norms_heatmap.png            # Per-t-bin gradient norms, final iteration
│   ├── group_comparison.png              # Boxplot of scores by ablation group
│   ├── win_rate_and_effective_batch_size.png
│   ├── achievement_breakdown.png          # Start vs end achievement rates (stacked bars)
│   ├── achievement_collapse_{name}.png    # Per-ablation achievement heatmap over time
│   ├── diagnosis_decision_tree.png
│   └── action_dist/                        # On by default; --no-action-dist skips it
│       ├── action_freq_{name}.png         # Side-by-side pre/post action frequency bars
│       ├── transition_matrix_{name}.png   # 3-panel heatmap (pre, post, difference)
│       ├── action_metrics_{name}.png      # 2x2 dashboard (entropy in nats, effective, Gini, divergences)
│       └── js_divergence_comparison.png   # Cross-ablation JS divergence bar chart
└── tables/
    ├── main_results.{csv,tex}
    ├── significance_test.txt              # Max-statistic permutation test + p floor + bootstrap CI
    ├── group_summary.{csv,tex}            # Group-level summary table
    ├── gradient_analysis.{csv,tex}
    ├── t_distribution.{csv,tex}
    ├── repr_drift.{csv,tex}               # KL drift values at the final iteration
    ├── per_env.{csv,tex}                  # Per-achievement rates; needs pretrained_ach_rates
    ├── forgetting_analysis.{csv,tex}
    ├── hypothesis_verdict.{csv,tex}
    ├── achievement_summary.{csv,tex}      # Per-achievement final unlock rates
    ├── gdelta.{csv,tex}                   # --measure-gdelta only: the decomposition per weight transform
    └── results.tex                        # --emit-tex-macros only: \newcommand per headline number
```

**Action distribution analysis** runs by default and is disabled with
`--no-action-dist`. The rollout is one vectorised scan over `num_envs`, sized
from the config rather than by an episode count, so it costs a fraction of a
training run — which is why the default differs from the minihack twin, where
the same flag defaults off because MiniHack rollouts are not vectorised. It
reads each ablation's `final_params`, so it only covers ablations that
completed.

**`results.json` schema:**
```json
{
  "pretrained_score": 0.1234,
  "pretrained_ach_rates": {"achievement_collect_wood": 0.42, ...},
  "config": {"MAX_ITER": 1000, ...},   // the merged config, keys uppercase
  "merge_provenance": { ... },         // --merge only: inputs + which supplied config
  "ablations": {
    "kl_penalty": {
      "score": 0.1456,        // mean across seeds
      "score_std": 0.008,     // std across seeds (0.0 if num_seeds=1)
      "all_scores": [0.1456], // per-seed scores
      "base_seed": 42, "seeds": [42], // seeding actually used
      "wall_clock_s": 812.4,
      "per_seed_finals": [{...}],     // per-seed end-of-run metrics
      "final_ach_rates": {"achievement_collect_wood": 0.40, ...},
                              // achievement detail of the same post-loop
                              // evaluations that produced all_scores, seed-averaged
      "all_final_ach_rates": [{...}],   // per-seed, before averaging
      "history": { ... }      // AblationHistory serialised
    }
  }
}
```

`results.json` is written incrementally after each ablation completes — a partial file with
N of 26 ablations is fully valid and loadable by `--analyze-only --results-path`.

### W&B logging

Three metrics per ablation, logged under `ablations/{name}/` against `iteration`:
`train_loss`, `env_score` (both every logged iteration) and `eval_score` (every
`eval_every`). They are written after the arm finishes, not during it, and there is
no `wandb.summary` write.

Every other quantity in the table below — gradient alignment, per-layer norms, KL
drift, CKA, the t-bin norms — is collected into `AblationHistory` and reaches
`results.json` only. Read those from the run directory, not from W&B.

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
