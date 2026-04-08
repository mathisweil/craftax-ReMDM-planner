# Claude Code: Create Craftax Demo Notebook

## Objective

Create a **fully self-contained** Jupyter notebook (`demo_craftax.ipynb`) for our COMP0258 coursework submission. The marker uploads **only the `.ipynb` file** to Google Colab and runs it top-to-bottom — the notebook must download everything it needs.

### Submission Requirements

> "prepare a self-contained notebook demonstrating your approach so that it can be uploaded and executed on Colab"
> "If your project has a stronger machine learning component then please avoid training a model in the notebook. Instead, give us an **easy way** to test your system on unseen inputs."
> "In case you need to train models, do this training offline and load the model into Colab."
> "demonstrate that your findings are reproducible"

**Our project has both a strong ML component AND explorative research.** The notebook must:
1. Load the pre-trained model (no training in notebook)
2. Give an easy way to test on unseen inputs (change env variant, seeds, eval steps)
3. Demonstrate findings are reproducible (live inference ≈ reported results; ablation analysis clearly presented)
4. Demonstrate the approach (ReMDM denoising, observation encoding, plan generation)

---

## 0. Distribution Strategy: Single HuggingFace Repo

Everything the notebook needs lives in **one public HuggingFace repo**:

```python
from huggingface_hub import snapshot_download
snapshot_download(HF_REPO_ID, local_dir="remdm-craftax")
```

Use `TODO_HF_REPO_ID` as a placeholder. Put a constant at the top so it only needs changing once.

### What to upload to the HF repo

1. **Full source code** — `src/`, `configs/`, `Craftax_Baselines/` (submodule), `main.py`, etc.
2. **Checkpoints** from `checkpoints/`. There should be:
   - Offline diffusion checkpoint(s)
   - Online (DAgger) diffusion checkpoint(s)
   - Two PPO agents: one for Craftax-Classic (`Craftax-Classic-Symbolic-v1`), one for Full Craftax (`Craftax-Symbolic-v1`)
   
   Craftax uses orbax/JAX serialisation (not PyTorch). Check checkpoint format via `src/planners/model.py`. Strip to inference-only weights if possible.
3. **Pre-computed ablation assets** — key `.png` and `.csv` files from `experiments/rl_finetuning/outputs/craftax_final_results/`

---

## ⚠️ JAX/Craftax Installation on Colab

Colab has JAX pre-installed but may have version mismatches.

The notebook MUST:
1. Install correct JAX version (read `pyproject.toml` — needs `jax>=0.9.2`)
2. Install Craftax: `!pip install craftax>=1.5.0`
3. Install other deps: `flax>=0.12.6`, `optax>=0.2.8`, `orbax>=0.1.9`, `distrax>=0.1.7`, `chex>=0.1.91`
4. Verify:
   ```python
   import jax, craftax
   print(f"✓ JAX {jax.__version__} on {jax.devices()[0].platform}")
   print(f"✓ Craftax {craftax.__version__}")
   ```
5. Warn if CPU-only (slow but functional).

---

## 1. Project Summary

We apply **ReMDM** (Remasking Discrete Diffusion Model) to action-sequence planning in **Craftax**, a JAX-based procedurally-generated open-world survival game (fast Crafter reimplementation with NetHack-like mechanics). A bidirectional transformer generates `plan_horizon`-length action plans by iteratively denoising masked token sequences, conditioned on the current environment observation.

**Framework:** JAX + Flax

**Training pipeline:**
```
[Stage 1] Train PPO agent (Craftax_Baselines/ppo_rnn.py)
[Stage 2] Train diffusion offline from PPO rollouts (main.py --mode offline)
[Stage 3] Online DAgger fine-tuning (main.py --mode online)
[Stage 4] Evaluate (main.py --mode inference)
```

**Core finding:** DAgger-trained ReMDM imitates the PPO expert, but NO RL ablation (25 tested) meaningfully improves beyond the DAgger checkpoint — same finding as MiniHack, confirming framework independence.

---

## 2. Architecture

**`DenoisingTransformer`** (JAX/Flax):

```
Observation encoder: symbolic obs → MLP (obs_encoder_layers=2, width=512) → [B, d_model]
Action stream:       Embedding(num_actions+2, d_model) + timestep_emb + position_emb
                     num_actions = 17 (Classic) or 43 (Full) + MASK + PAD tokens
Transformer:         [1 obs token + plan_horizon action tokens] → 4-layer encoder
                     (d_model=256, n_heads=4, d_ff=512, pre-norm, dropout=0.1)
Output head:         action tokens → Linear(d_model, num_actions) → logits
```

Key config parameters: `plan_horizon=32`, `diffusion_steps=15` (inference), `diffusion_schedule=cosine`, `remask_strategy=rescale`, `eta=0.5`, `temperature=0.5`, `top_p=0.95`.

---

## 3. Environments

| Environment | Description |
|---|---|
| `Craftax-Classic-Symbolic-v1` | Crafter in JAX (22 achievements, 17 actions, 1 floor) |
| `Craftax-Symbolic-v1` | Extended with NetHack mechanics (65 achievements, 43 actions, 9 floors) |

Both procedurally generated. Symbolic observations. Evaluation: mean return (% of max: 22 Classic, 226 Full) + per-achievement success rates.

---

## 4. Pre-Computed Results Location

- Final results: `experiments/rl_finetuning/outputs/craftax_final_results/`
- Checkpoints: `checkpoints/` — offline, online (DAgger), PPO Classic, PPO Full
- Key files: `results.json`, `main_results.csv`, `hypothesis_verdicts.csv`, `.png` plots

---

## 5. Notebook Structure

### Cell 0: Configuration

```python
HF_REPO_ID = "TODO_HF_REPO_ID"
SEED = 42
ENV_NAME = "Craftax-Classic-Symbolic-v1"  # or "Craftax-Symbolic-v1"
EVAL_STEPS = 10000
EVAL_NUM_ENVS = 32  # reduce if OOM
DIFFUSION_STEPS = 10
```

### Cell 1: Setup & Installation
- pip install JAX + GPU, craftax, flax, optax, orbax, distrax, chex, wandb, polars, orjson
- Read `pyproject.toml` for versions
- Verify JAX + Craftax + GPU
- snapshot_download, sys.path, ensure `Craftax_Baselines/` on path

### Cell 2: Project Overview (Markdown)
- Problem, approach (PPO expert → diffusion planner via DAgger), research question

### Cell 3: Load Pre-Trained Model
- Read `src/models/denoiser.py` for `DenoisingTransformer` Flax module
- Read `configs/defaults.yaml` for hyperparameters
- Load checkpoint via orbax / project utilities (`src/planners/model.py`)
- JAX models are stateless — params are a pytree
- Print param count + summary

### Cell 4: Test on Environment (Live Inference) ⭐ **PRIORITY**
- Use existing inference pipeline (`src/planners/inference.py`)
- Evaluate on `ENV_NAME` for `EVAL_STEPS` steps with `EVAL_NUM_ENVS` parallel envs
- Print mean return (% of max), completed episodes, per-achievement unlocks
- Compare with reported results
- Marker changes `ENV_NAME`, `SEED`, `EVAL_STEPS` in Cell 0

### Cell 5: Visualise Agent Behaviour (Live) ⭐
- Run episodes, capture obs + actions
- Plot achievement unlocks over time
- Show symbolic observation and agent decisions
- Optionally use Craftax pixel renderer if available

### Cell 6: Visualise ReMDM Denoising (Live) ⭐
- Show fully-masked → iterative unmasking over denoising steps
- Read `src/diffusion/sampling.py` to capture intermediates
- Grid visualisation: rows=steps, columns=positions, colour=masked/committed
- Show ReMDM remasking in action

### Cell 7: PPO Expert Baseline (Static/Live)
- If PPO checkpoint available: load via `src/planners/ppo.py`, evaluate, compare
- Otherwise: hardcode reported PPO-RNN results
- Show diffusion planner ≈ expert performance

### Cell 8: RL Ablation Findings (Pre-Computed Figures)
- From `craftax_final_results/`: `score_comparison.png`, `group_comparison.png`, `grad_alignment.png`, `score_delta.png`
- Brief markdown per figure

### Cell 9: Ablation Results Tables
- `main_results.csv` + `hypothesis_verdicts.csv` as DataFrames

### Cell 10: Conclusions (Markdown)
- DAgger works; RL doesn't improve; double intractability; consistent across both codebases

---

## 6. What You Must Discover By Reading Source

1. `DenoisingTransformer` Flax module → `src/models/denoiser.py`
2. Checkpoint loading → `src/planners/model.py`
3. Inference loop → `src/planners/inference.py` (MPC with historical inpainting, `hist_len`)
4. Diffusion sampling → `src/diffusion/sampling.py`
5. Environment construction → `src/planners/env.py` (wrapper stack)
6. PPO agent loading → `src/planners/ppo.py`
7. Files in `experiments/rl_finetuning/outputs/craftax_final_results/` → `ls`
8. Files in `checkpoints/` → `ls`, identify Classic vs Full, offline vs online, PPO vs diffusion
9. Config defaults → `configs/defaults.yaml`
10. Dependency versions → `pyproject.toml`

---

## 7. Key Differences from MiniHack Notebook

| Aspect | MiniHack | Craftax |
|---|---|---|
| Framework | PyTorch | JAX + Flax |
| Environment | MiniHack (NLE, CPU) | Craftax (JAX, GPU-accelerated) |
| Observation | Dual-stream: 9×9 local + 21×79 global CNNs | Single symbolic vector → MLP |
| Actions | 12 navigation | 17 (Classic) or 43 (Full) |
| Expert | BFS oracle | PPO-RNN agent |
| Metric | Win rate (reach staircase) | Mean return (% of max) + achievements |
| Checkpoint | PyTorch `.pth` | Orbax / JAX pytree |
| Installation risk | NLE C compilation | JAX version mismatch |
| Env variants | 4 ID + 3 OOD maps | Classic vs Full Craftax |
| PPO checkpoints | Not needed (BFS oracle) | Needed (PPO is expert teacher) |

---

## 8. Critical Constraints

- **Fully standalone.** Only `.ipynb` uploaded to Colab.
- **One HuggingFace repo.** Public, no auth.
- **No training.** Load pre-trained checkpoint only.
- **Marker changes env/seeds/eval_steps in Cell 0.**
- **JAX GPU:** Should auto-detect Colab GPU; warn if CPU-only.
- **Live inference is priority** — Cells 4-6 > pre-computed figures.
- **Target runtime**: <10 min on Colab GPU (Craftax is fast).
- **JAX/Flax codebase.** Do NOT reference MiniHack, NLE, or PyTorch.
- **Procedurally generated** — every seed = unseen world.
- **Two PPO checkpoints** (Classic + Full) — use the one matching `ENV_NAME`.
- **Historical inpainting:** `hist_len` plan positions locked to observed history at inference.
- **Submodule:** `Craftax_Baselines/` must be in HF repo (PPO code + env wrappers).
