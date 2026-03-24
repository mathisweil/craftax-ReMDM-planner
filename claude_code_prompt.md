# Claude Code Prompt: RL Fine-Tuning Ablation Analysis — Full Implementation Plan

## Context

You are working on **craftax-ReMDM-planner**, a JAX implementation of a Remasking Discrete Diffusion Model (ReMDM) planner for the Craftax environment. The project is fully described in the README. The training pipeline has four stages: PPO pre-training → offline diffusion training → online GRPO fine-tuning → evaluation.

We are actively investigating **why RL fine-tuning of the diffusion model collapses** and want to systematically test many interventions. There is already a notebook `craftax_ablations_full.ipynb` that tests four ablations (KL Penalty, Frozen Backbone, BC on Wins, Low-t Only) alongside a synthetic sanity check.

**Your task:** Read and deeply understand all existing code, then design and implement a comprehensive expansion of the ablation study infrastructure — adding new RL fine-tuning techniques, richer diagnostics, and a full suite of visualisations, tables, and figures. Everything must be integrated cleanly into the existing project structure.

---

## Step 0 — Read First, Plan Second

Before writing a single line of code:

1. Read `README.md` fully.
2. Read `craftax_ablations_full.ipynb` fully.
3. Read every file under `src/` (diffusion, models, planners, envs).
4. Read `main.py` and `configs/defaults.yaml`.
5. Read `Craftax_Baselines/ppo_rnn.py` and `Craftax_Baselines/ppo.py`.

Only after understanding the full codebase should you begin designing changes.

---

## Step 1 — Brainstorm & Implementation Planning

Think carefully and produce a written plan covering all of the following. Do not implement anything yet.

### 1a. New RL Fine-Tuning Techniques to Test

For each technique, reason about *why* it might help given the collapse hypothesis, how it maps onto the existing MDLM/GRPO infrastructure, and what the expected signal would be if it works or fails:

**Regularisation family**
- Elastic Weight Consolidation (EWC): penalise deviation from pretrained weights weighted by Fisher information
- Layer-wise Learning Rate Decay (LLRD): exponentially smaller LR for earlier transformer layers
- LoRA / low-rank adapter fine-tuning: train only small rank-decomposed delta matrices, freeze base weights
- Gradient projection: project RL gradient onto the orthogonal complement of the BC gradient subspace to prevent forgetting

**Reward / advantage shaping family**
- Clipped advantages (PPO-style clipping on the GRPO policy ratio)
- Advantage normalisation per diffusion time-step t (not just per batch)
- Curiosity-augmented reward: add RND bonus to environment reward before computing group advantages
- Return rescaling: map raw returns through a running percentile normaliser rather than mean-subtract

**Data / sampling family**
- Mixed replay: blend online self-generated windows with frozen offline PPO windows (e.g. 50/50, 80/20)
- Prioritised experience replay for the offline buffer keyed on TD-error proxy
- Curriculum over t: start training only on low-t (refinement) steps, gradually add higher-t steps
- Denoising step dropout: randomly skip denoising steps during training to encourage robustness

**Architecture / loss family**
- Time-conditioned layer norm: separate γ,β per discretised t-bin
- Confidence-weighted loss: down-weight tokens the model is already certain about
- Token-level advantage: assign each action token its own advantage based on that token's marginal contribution
- Entropy regularisation: add H[p(a|obs)] bonus to encourage exploration at the action distribution level

**Training dynamics family**
- Warm restarts of the LR schedule after each performance plateau
- Gradient surgery: if RL gradient conflicts with BC gradient (negative cosine similarity), zero the conflicting component
- Trust-region projection: scale the RL update so the KL from pretrained never exceeds a threshold ε
- Two-timescale updates: update the observation encoder at 1/10th the LR of the action head

### 1b. Diagnostics and Metrics to Track During Training

For every training run, track (per gradient step and/or per eval interval):

**Loss decomposition**
- Total loss
- ELBO loss (return-weighted)
- KL penalty term (where applicable)
- Per-t-bin ELBO: divide [0,1] into 10 equal bins and log mean loss per bin
- Loss on winning windows vs. losing windows separately

**Gradient health**
- Global gradient norm (before and after clipping)
- Per-layer gradient norm (transformer blocks 0–N, obs encoder, output head)
- Cosine similarity between RL gradient and BC oracle gradient ("gradient alignment")
- Gradient variance across minibatches within the same update step

**Representation drift**
- L2 distance of all parameters from pretrained checkpoint (total and per layer)
- CKA (Centred Kernel Alignment) between pretrained and current representations on a fixed probe batch
- Output logit KL divergence on a fixed probe batch (same obs, measure how much the distribution has shifted)

**Policy quality**
- Token entropy of the action distribution per denoising step
- Fraction of plans that are "collapsed" (same action repeated ≥ 50% of tokens)
- Per-achievement unlock rate during eval (Craftax gives a structured achievement vector)

**GRPO-specific**
- Group advantage mean, std, min, max
- Effective group size (after filtering invalid plans)
- Policy ratio min/max (detect KL blowup)
- PPO expert injection probability (should decay from `ppo_init_prob`)

**Data quality**
- Return distribution of the current rollout batch (histogram)
- Fraction of windows that are "wins" (reward > threshold)
- Episode length distribution

### 1c. Visualisations, Figures, and Tables

Design a complete set of outputs that will be produced at the end of the notebook:

**Training dynamics panel** (one panel per ablation, all on same axes for comparison)
- Eval score over gradient steps (primary metric, horizontal line at pretrained baseline)
- Training loss over gradient steps (smoothed with EMA)
- Gradient alignment over gradient steps
- Representation drift (L2) over gradient steps
- Per-t-bin ELBO heatmap: x=gradient step, y=t-bin, colour=loss value

**Summary comparison figures**
- Bar chart: final eval score ± std for all methods (including pretrained baseline)
- Bar chart: final representation drift for all methods
- Bar chart: final gradient alignment for all methods
- Scatter plot: gradient alignment vs. eval score (each method is a point; colour by ablation family)
- Scatter plot: representation drift vs. eval score (same style)

**Per-method deep-dive panels** (generated for each ablation that differs significantly from baseline)
- Action entropy over denoising steps (violin plot across eval episodes)
- Return distribution histogram at iteration 0, 250, 500
- Per-layer gradient norm heatmap over training (x=step, y=layer, colour=norm)
- Achievement unlock rate bar chart (17 achievements in Craftax Classic)

**Regression / correlation table**
- Pearson and Spearman correlation between each tracked diagnostic and final eval score, across all methods
- This identifies which diagnostics are predictive of success

**Summary results table** (LaTeX-ready)
- Rows: methods; columns: final score, delta from pretrained, gradient alignment, drift, verdict
- Formatted with ± where multiple seeds are run

**Failure mode taxonomy figure**
- For each collapsed run, classify the failure mode based on the diagnostic pattern:
  1. Catastrophic forgetting (high drift, low alignment)
  2. Mode collapse (low entropy, high drift)
  3. Gradient conflict (negative alignment, moderate drift)
  4. No learning (low alignment, low drift, no score improvement)
- Produce a 2×2 heatmap: drift (x) vs. alignment (y), each method as a labelled dot

### 1d. Project Integration Plan

Design where each new piece of code should live, following existing conventions:

```
src/
├── ablations/                     # NEW — ablation-specific utilities
│   ├── __init__.py
│   ├── losses.py                  # All ablation loss functions (KL, EWC, LoRA, etc.)
│   ├── techniques.py              # New fine-tuning method wrappers
│   ├── diagnostics.py             # Gradient alignment, CKA, drift computation
│   └── visualisations.py          # All plotting functions (called from notebook)
├── diffusion/
│   └── (existing — extend loss.py if needed for new loss variants)
└── planners/
    └── (existing — do not break existing train/online/inference modes)

notebooks/
├── craftax_ablations_full.ipynb   # EXISTING — refactor to import from src/ablations/
└── ablation_results/              # NEW — auto-saved figures and tables
    ├── figures/
    └── tables/
```

Key integration rules:
- All reusable logic goes into `src/ablations/` as importable Python modules
- The notebook becomes a thin orchestration layer that imports from `src/ablations/`
- No existing files in `src/` should be modified in a breaking way
- New configs for new ablation hyperparameters go into `configs/ablations.yaml`
- All figure-saving uses a single `save_figure(fig, name)` helper that respects a configurable output directory

---

## Step 2 — Implementation

After planning and getting approval (or immediately if no review step is needed), implement everything:

### Priority 1 — Core Infrastructure (do this first)

1. **Create `src/ablations/` package** with `__init__.py`

2. **`src/ablations/losses.py`**
   Implement all new loss functions as pure JAX functions with the signature:
   ```python
   def make_loss_<name>(apply_fn, pretrained_params=None, **kwargs):
       def loss_fn(params, rng, acts, obs, valid, advantages):
           ...
       return loss_fn
   ```
   Required losses:
   - `make_loss_baseline` (already exists in notebook — migrate here)
   - `make_loss_kl` (already exists — migrate)
   - `make_loss_bc_wins` (already exists — migrate)
   - `make_loss_low_t` (already exists — migrate)
   - `make_loss_ewc` (NEW: EWC penalty, requires Fisher info computation)
   - `make_loss_mixed_replay` (NEW: blend offline buffer + online rollout)
   - `make_loss_t_curriculum` (NEW: time-step curriculum over training)
   - `make_loss_entropy_reg` (NEW: entropy bonus on action distribution)
   - `make_loss_token_advantage` (NEW: token-level advantage weighting)
   - `make_loss_trust_region` (NEW: hard KL budget projection)

3. **`src/ablations/techniques.py`**
   Implement training technique wrappers:
   - `make_ablation_grad_step` with `frozen_backbone`, `llrd_decay`, `gradient_surgery` flags
   - `apply_lora_params`: LoRA parameter injection and extraction utilities
   - `compute_ewc_fisher`: compute diagonal Fisher information matrix from a batch of rollouts
   - `gradient_projection_step`: project RL gradient orthogonally to BC gradient

4. **`src/ablations/diagnostics.py`**
   ```python
   def compute_gradient_alignment(rl_grad, bc_grad) -> float
   def compute_representation_drift(params, ref_params) -> dict[str, float]
   def compute_output_kl(apply_fn, params, ref_params, probe_obs, rng) -> float
   def compute_per_t_loss(apply_fn, params, rng, acts, obs, valid, n_bins=10) -> jnp.ndarray
   def compute_token_entropy(logits) -> float
   def compute_collapse_fraction(plan_tokens) -> float
   def compute_per_layer_grad_norm(grads) -> dict[str, float]
   ```

5. **`src/ablations/visualisations.py`**
   ```python
   def plot_training_dynamics(all_histories, baseline_score, save_dir=None)
   def plot_summary_bars(results_dict, save_dir=None)
   def plot_scatter_diagnostics(results_dict, save_dir=None)
   def plot_per_method_deep_dive(name, history, save_dir=None)
   def plot_t_bin_heatmap(all_histories, save_dir=None)
   def plot_failure_mode_map(results_dict, save_dir=None)
   def plot_achievement_bars(all_histories, save_dir=None)
   def make_summary_table(results_dict, latex=True) -> str
   def make_correlation_table(results_dict) -> pd.DataFrame
   ```
   All functions:
   - Use consistent colour palette (define a `PALETTE` dict at top of file)
   - Accept optional `save_dir` and call `plt.savefig` if provided
   - Return the figure object so the notebook can also display it inline
   - Use `matplotlib` only (no seaborn dependency)

6. **`configs/ablations.yaml`**
   ```yaml
   # New ablation-specific hyperparameters
   ewc_lambda: 100.0
   llrd_decay: 0.9
   lora_rank: 8
   lora_alpha: 16.0
   t_curriculum_start: 0.2
   t_curriculum_end: 1.0
   t_curriculum_steps: 200
   mixed_replay_ratio: 0.5
   entropy_coef: 0.01
   trust_region_kl: 0.05
   gradient_surgery: false
   ablation_output_dir: "notebooks/ablation_results"
   ```

### Priority 2 — Notebook Refactor

Refactor `craftax_ablations_full.ipynb` so that:
- All imports come from `src/ablations/`
- The notebook cells are purely orchestration: configure, run, visualise
- A new section "5b. Extended Ablations" runs the new techniques after the existing four
- A new section "12. Correlation Analysis" computes the diagnostic-vs-score correlation table
- A new section "13. Failure Mode Taxonomy" produces the 2×2 drift/alignment scatter
- All figures are saved to `notebooks/ablation_results/figures/`
- The final summary table is saved as both `.csv` and `.tex` to `notebooks/ablation_results/tables/`

The `run_ablation()` function should be extended (or a new `run_ablation_v2()` created) so its history dict now includes all the new diagnostic keys: `per_t_loss`, `per_layer_grad_norm`, `output_kl`, `token_entropy`, `collapse_fraction`.

### Priority 3 — Multi-Seed Support

Extend the training loop to support `NUM_SEEDS` independent runs (default: 3) using `jax.vmap` where possible, or sequential loops where vmap is not practical. Mean ± std should be computed across seeds for all metrics and displayed in all figures.

### Priority 4 — Automated Report

Create `notebooks/generate_report.py` — a standalone script (not notebook) that:
1. Loads saved results from `notebooks/ablation_results/`
2. Regenerates all figures and tables deterministically
3. Prints a human-readable summary to stdout with verdicts per method
4. Can be called with `python notebooks/generate_report.py --results_dir notebooks/ablation_results`

---

## Step 3 — Quality Checks

After implementing everything, verify:

1. All new `src/ablations/` functions have docstrings and type annotations
2. `src/ablations/diagnostics.py` functions are pure JAX (JIT-compatible where the computation is on device, Python-level where it is not)
3. No circular imports between `src/ablations/` and `src/planners/` (ablations can import from planners, not vice versa)
4. Notebook runs cell-by-cell without errors on a small `MAX_ITER=10` smoke test
5. `configs/ablations.yaml` values are loaded via the existing config mechanism in `main.py` (extend the argparse/yaml loader to recognise these keys without breaking existing modes)
6. All figures save correctly to the output directory and display inline in the notebook
7. The `generate_report.py` script runs cleanly and produces consistent output

---

## Guiding Principles

- **Minimise disruption**: do not touch `src/planners/train.py`, `src/planners/online.py`, `src/diffusion/loss.py`, or `main.py` unless absolutely necessary. All new logic belongs in `src/ablations/`.
- **Composability**: loss functions, diagnostic functions, and plotting functions are all independently usable — they do not assume a specific ablation loop structure.
- **JAX idioms**: use `jax.tree.map` for parameter operations, `jax.grad` for gradient computation, `jax.lax.scan` for loops inside JIT where possible.
- **Reproducibility**: every random operation uses an explicit `rng` key threaded through; no global state.
- **Legibility**: the notebook should tell a clear scientific story — hypothesis, experiment, result, interpretation. Comments in cells should explain what each section is testing and why.
