# ReMDM Planner — Discrete Diffusion Planning on Craftax

A JAX implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in the [Craftax](https://github.com/MichaelTMatthews/Craftax) environment. A bidirectional transformer learns to generate action plans by iteratively denoising masked token sequences, conditioned on the current environment observation.

---

## Description

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `T` denoising steps, producing a `plan_horizon`-length plan. ReMDM extends MDLM with remasking strategies that let committed tokens be re-predicted, improving plan coherence.

**Offline BC** and **Online DAgger** are independent pipelines, both supervised by a pre-trained PPO expert. Neither depends on the other; the paper compares them head-to-head.

```
[Shared]   Train PPO agent              Craftax_Baselines/ppo_rnn.py | ppo_rnd.py
               |
               v  checkpoint
       ┌───────┴────────┐
       │                │
  [Offline BC]     [Online DAgger]
  --mode offline    --mode online
  (live PPO         (from scratch; mixed policy
   rollouts)         + expert labels -> buffer)
       │                │
       v                v
   checkpoint        checkpoint
       │                │
       └───────┬────────┘
               v
[Evaluate] --mode inference --checkpoint_path ...
```

| Mode | Purpose |
|---|---|
| `--mode collect` | Save PPO rollouts to disk |
| `--mode smoke` | Quick end-to-end check |
| `--offline_checkpoint_path` | Warm-start DAgger from an offline BC checkpoint (not used in the paper) |

---

## Installation

### Prerequisites (system-level)

`uv` manages Python packages only. The following must be installed at the OS level before
running on a GPU node — they are **not** in `pyproject.toml`:

- **CUDA 13** driver and toolkit (`libcuda.so`, `libcudnn`)

On HPC clusters these are typically loaded via `module load cuda/13.x`.

### 1. Create the virtual environment

```bash
# CPU-only (local development / macOS)
uv sync

# NVIDIA CUDA 13 (GPU node — Linux only)
uv sync --extra cuda

# Activate
source .venv/bin/activate
```

`uv sync` reads `pyproject.toml`, resolves a fully-reproducible lockfile (`uv.lock`),
and installs into `.venv/`. Commit `uv.lock` to pin the exact dependency graph.

### 2. Initialise the submodule

```bash
git submodule update --init --recursive
```

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `jax` | >=0.9.2 | JIT compilation and functional arrays |
| `flax` | >=0.12.6 | Neural network definitions |
| `optax` | >=0.2.8 | Adam optimiser and gradient clipping |
| `craftax` | >=1.5.0 | Procedurally-generated Minecraft-like environment |
| `chex` | >=0.1.91 | JAX testing and assertion utilities |
| `distrax` | >=0.1.7 | Probability distributions |
| `orbax` | >=0.1.9 | Model checkpointing |
| `wandb` | >=0.25.1 | Experiment logging |
| `numpy` | >=2.4.4 | Array operations |
| `matplotlib` | >=3.10.8 | Plotting |
| `polars` | >=1.39.3 | DataFrame analysis |
| `orjson` | >=3.11.8 | Fast JSON serialisation |
| `pyyaml` | >=6.0.3 | Config file parsing |
| `huggingface-hub` | >=1.9.1 | Checkpoint download and upload |

Full specification in `pyproject.toml`; exact transitive pins in `uv.lock`. The `dev` group adds `pytest` and is installed by `uv sync`.

---

## Usage

All modes share the same entry point. Defaults are loaded from `configs/defaults.yaml`; any value can be overridden on the command line.

```bash
python main.py --mode <MODE> [--config PATH] [OVERRIDES...]
```

Pass `--no-jit` to disable JIT compilation (useful for debugging):

```bash
python main.py --mode offline --no-jit --num_envs 4
```

### Stage 1 — Train a PPO agent

PPO training is handled by the `Craftax_Baselines` submodule and produces the checkpoint consumed by all downstream stages.

```bash
cd Craftax_Baselines

# PPO with GRU hidden state (recommended)
python ppo_rnn.py \
    --env_name Craftax-Classic-Symbolic-v1 \
    --total_timesteps 500000000 \
    --save_policy --use_wandb

# PPO with Random Network Distillation
python ppo_rnd.py \
    --env_name Craftax-Classic-Symbolic-v1 \
    --total_timesteps 500000000 \
    --save_policy --use_wandb

cd ..
```

### Stage 2a — Collect trajectories to disk

Roll out the PPO checkpoint and save `(obs, actions, rewards, dones)` as a `.npz` file for reuse across multiple diffusion training runs.

```bash
python main.py --mode collect \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_data_path data/trajectories.npz \
    --collect_num_steps 1000000 \
    --collect_num_envs 128
```

Arrays are shaped `[num_envs, num_iters, ...]`, preserving per-environment contiguity so episode boundaries survive window sampling.

### Stage 2b — Train offline (BC)

Rolls out the PPO agent live at each update. Windows crossing an episode boundary are masked out; the rest are weighted by cumulative reward, clipped to `[0.1, return_weight_cap]`.

```bash
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_total_timesteps 100000000 \
    --save_policy
```

### Stage 3 — Train online (DAgger)

Trained **from scratch**. Per iteration: a mixed policy (expert vs learner, ratio `beta`, exponentially decayed) rolls out; the expert labels every visited state; `(obs, expert_plan)` pairs enter a circular replay buffer; the model trains on the whole buffer with the MDLM ELBO — pure BC, no reward weighting.

```bash
# From scratch (requires PPO expert checkpoint)
python main.py --mode online \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --online_num_updates 1000 \
    --save_policy

# Optional: warm-start from a pre-trained offline checkpoint
# (not used in the paper — both methods are compared independently)
python main.py --mode online \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_checkpoint_path /path/to/offline_checkpoint \
    --online_num_updates 1000 \
    --save_policy
```

With `save_policy=true` this uploads two W&B artifacts, either consumable via `--checkpoint_path wandb:…`: `{env_name}-policy` (final) and `{env_name}-policy-best` (highest validation return).

### Stage 4 — Evaluate

```bash
python main.py --mode inference \
    --checkpoint_path /path/to/checkpoint \
    --eval_steps 10000 \
    --eval_num_envs 32
```

Prints mean episode return, completed episodes, steps per second, and per-achievement unlock counts. Uses historical inpainting: the first `hist_len` plan positions are locked to observed history.

### Smoke test

```bash
python main.py --mode smoke
```

Full DAgger pipeline (rollout, expert labelling, gradient updates, validation) under `configs/smoke.yaml`, ~25 s on CPU. `src/planners/smoke.py` delegates to `run_online`.

- **Expert is random** unless `--ppo_checkpoint_path` is given, so the mode runs on a clean clone with no downloads. It proves the pipeline executes, nothing about learning.
- **Returns and achievements read `0.000`** — they are episode-weighted and no episode terminates in so short a run. Watch `mean step reward`, `loss`, `all metrics finite`.
- `smoke.yaml` holds overrides only, layered on `defaults.yaml`. Sizing obeys invariants `make_train_dagger` asserts on *derived* values, so shrinking the frame budget alone does not shorten the run — see the comments in the file.

### Loading checkpoints from W&B artifacts

Any checkpoint path argument (`--checkpoint_path`, `--offline_checkpoint_path`, `--ppo_checkpoint_path`) accepts a W&B artifact reference prefixed with `wandb:`. The artifact is downloaded automatically before training or evaluation begins.

```bash
# Fully qualified: entity/project/artifact_name:version_or_alias
python main.py --mode inference \
    --checkpoint_path wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy:latest

# Online fine-tuning from a W&B offline checkpoint
python main.py --mode online \
    --offline_checkpoint_path wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy:v3

# PPO checkpoint from W&B
python main.py --mode offline \
    --ppo_checkpoint_path wandb:my-team/ppo-craftax/ppo-rnn-policy:best
```

Control the download location with `--wandb_download_dir` (defaults to `./artifacts/`).

### Loading pre-trained checkpoints from the Hugging Face Hub

`checkpoints/` is gitignored; the weights live on the Hub at [`MathisW78/remdm-craftax-checkpoints`](https://huggingface.co/MathisW78/remdm-craftax-checkpoints), mirroring the layout below. Download into the repository root and every file lands where the CLI expects it.

| Checkpoint directory | Environment | Role | Trained for |
|---|---|---|---|
| `checkpoints/offline/Craftax-Classic-Symbolic-v1-OfflineDiffusion-BC-100M` | Craftax Classic | Offline BC planner | 1e8 env frames |
| `checkpoints/offline/Craftax-Symbolic-v1-OfflineDiffusion-BC-100M` | Full Craftax | Offline BC planner | 1e8 env frames |
| `checkpoints/online/Craftax-Classic-Symbolic-v1-OnlineDiffusion-DAgger-100M` | Craftax Classic | Online DAgger planner | 1e8 env frames |
| `checkpoints/online/Craftax-Symbolic-v1-OnlineDiffusion-DAgger-100M` | Full Craftax | Online DAgger planner | 1e8 env frames |
| `checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M` | Craftax Classic | PPO-RNN expert | 1e9 env frames |
| `checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M` | Full Craftax | PPO-RNN expert | 1e9 env frames |

```bash
# All six (~470 MB); narrow the --include glob for a single checkpoint.
uv run hf download MathisW78/remdm-craftax-checkpoints --include "checkpoints/**" --local-dir .
```

**Pass the checkpoint directory, not the step subdirectory** — `CheckpointManager` resolves the latest step itself.

**Match the config to the checkpoint.** The model is built from the config, not the checkpoint, so a released checkpoint under `defaults.yaml` (`d_model` 256, `n_layers` 4) fails with a shape mismatch. All released diffusion checkpoints are `d_model` 384, `n_heads` 8, `n_layers` 6, `d_ff` 768 — use the matching `final_*` config, which also sets the right `env_name`:

```bash
python main.py --mode inference \
    --config configs/final_classic_ucl.yaml \
    --checkpoint_path checkpoints/online/Craftax-Classic-Symbolic-v1-OnlineDiffusion-DAgger-100M

# Train a new planner against the released Full Craftax PPO expert
python main.py --mode online \
    --config configs/final_craftax_ucl.yaml \
    --ppo_checkpoint_path checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M \
    --save_policy
```

Diffusion checkpoints carry a `resume_metadata.json` sidecar — the authoritative record of the producing run's config, and what `--resume_checkpoint_path` reads to auto-detect `resume_step` and `resume_wandb_run_id`. PPO checkpoints carry `config.yaml` and `wandb-summary.json`.

Re-upload after retraining with `scripts/hf_upload.py` (rediscovers checkpoints, strips wandb environment metadata, regenerates the model card):

```bash
HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py --repo-id MathisW78/remdm-craftax-checkpoints
```

### Resuming a Training Run

Continue a completed run — for extending the training budget or restarting a preempted job.

```bash
# Offline. --resume_checkpoint_path also accepts a wandb: artifact reference.
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --resume_checkpoint_path /path/to/completed_offline_checkpoint \
    --offline_total_timesteps 200000000 \
    --save_policy

# Online. Add --resume_step / --resume_wandb_run_id to override the sidecar.
python main.py --mode online \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --resume_checkpoint_path /path/to/completed_online_checkpoint \
    --online_num_updates 2000 \
    --save_policy
```

**Notes:**
- The DAgger replay buffer is **not** persisted; it refills within a few iterations.
- JIT is preserved — resume only affects initialisation outside `jax.jit`.
- The cosine LR schedule spans the full `num_updates`; the step counter is offset so the LR resumes exactly where it stopped.
- With a metadata sidecar, `resume_step` and `resume_wandb_run_id` are auto-detected; CLI flags override. Without one, pass `--resume_step`.


---

## Testing

```bash
uv run pytest
```

141 smoke tests, ~55 s, CPU-only. `pytest` ships in the `dev` group, so `uv sync` installs it. Tiny synthetic data and a shrunken model throughout — no real checkpoints, datasets or network calls, and nothing written outside `tmp_path`. The suite asserts that things **run**, not that results are correct.

| File | Covers |
|------|--------|
| `tests/test_smoke_src.py` | `src/`: module imports, model built from the real config, forward pass, one gradient step, checkpoint round-trip, samplers, config resolvers, and every CLI entry point (including `--mode smoke` end to end) |
| `tests/test_smoke_experiments.py` | `experiments/rl_finetuning/`: all 25 ablations' losses and optimizers, the LoRA path, diagnostics, and the analysis/report stage |

---

## Configuration

All hyperparameters are in `configs/defaults.yaml`. Override any value on the command line:

```bash
python main.py --mode offline --lr 1e-4 --plan_horizon 64 --num_minibatches 16
```

Point to a custom config file:

```bash
python main.py --mode online --config configs/my_experiment.yaml
```

Preset configs for larger runs are provided in `configs/`:

| File | Purpose |
|------|---------|
| `defaults.yaml` | Base defaults for all modes |
| `smoke.yaml` | `--mode smoke` overrides, layered on `defaults.yaml` |
| `{classic,craftax}_exp_a_beta_fix.yaml` | DAgger — beta decay fix only (isolates data quality) |
| `{classic,craftax}_exp_b_beta_big_model.yaml` | DAgger — beta fix + larger transformer (3.5× on Classic) |
| `{classic,craftax}_exp_c_full_recipe.yaml` | DAgger — beta + big model + training dynamics |
| `classic_exp_d_{100K,250K,850K,3M}_model.yaml` | Craftax Classic model-size scaling sweep |
| `craftax_exp_d_{500K,1M,3M,7M}_model.yaml` | Full Craftax model-size scaling sweep |
| `final_classic_ucl.yaml` | Final Classic DAgger — UCL 3090 Ti, seed 42 (feeds the ablation suite) |
| `final_craftax_ucl.yaml` | Final Full Craftax DAgger — UCL 4090, seed 42 (feeds the ablation suite) |
| `final_{classic,craftax}_qmul.yaml` | Env-frame-matched second seeds — QMUL H200, seed 43 |

`final_*_qmul.yaml` differs from its UCL counterpart only in `num_envs` and `seed`; fairness-critical values are env-frame denominated and rescaled by `resolve_scaled_hyperparams()` at load, so no manual derivation is needed across hardware tiers.

Ablation hyperparameters live in `experiments/rl_finetuning/configs/`, loaded by `run_ablations.py`, not `main.py`. See `experiments/README.md`.

### Key hyperparameters

**Environment**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `env_name` | `Craftax-Classic-Symbolic-v1` | Craftax environment ID. Use `Craftax-Symbolic-v1` for Full Craftax. |
| `use_optimistic_resets` | `false` | Use `OptimisticResetVecEnvWrapper` instead of `AutoResetEnvWrapper` |
| `optimistic_reset_ratio` | 16 | Fraction of envs reset per step when optimistic resets are enabled |

**Diffusion model**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plan_horizon` | 32 | Action plan length H |
| `diffusion_steps` | 15 | Denoising steps T at inference |
| `diffusion_schedule` | `cosine` | Noise schedule: `cosine` or `linear` |
| `remask_strategy` | `rescale` | Remasking strategy: `rescale`, `cap`, or `conf` |
| `train_sigma` | 0.0 | Per-token remasking correction during training (0 = standard MDLM) |
| `label_smoothing` | 0.0 | Cross-entropy label smoothing epsilon (0 = exact ELBO) |
| `eta` | 0.5 | Remasking strength |
| `use_loop` | `true` | Three-phase loop remasking (Algorithm 3) |
| `t_on` / `t_off` | 0.7 / 0.3 | Time window boundaries for loop remasking |
| `temperature` | 0.5 | Softmax temperature for token sampling |
| `top_p` | 0.95 | Nucleus sampling threshold |

**Transformer architecture**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Hidden dimension |
| `n_heads` | 4 | Attention heads |
| `n_layers` | 4 | Transformer blocks |
| `d_ff` | 512 | FFN inner dimension |
| `obs_encoder_layers` | 2 | MLP layers in the observation encoder |
| `obs_encoder_width` | 512 | Observation encoder hidden width |
| `dropout_rate` | 0.1 | Dropout rate (disabled at inference) |

**Offline training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `offline_total_timesteps` | 1e8 | **PRIMARY** env-frame budget for live-PPO data collection. Derives `num_updates` as `offline_total_timesteps // (num_envs * num_steps)`, making the run hardware-portable across `num_envs` changes. |
| `offline_num_updates` | `null` | **LEGACY** outer update count; used only when `offline_total_timesteps` is unset. |
| `num_envs` | 1024 | Parallel environments |
| `num_steps` | 64 | Environment steps collected per update |
| `num_minibatches` | 8 | Gradient minibatches per epoch |
| `update_epochs` | 4 | SGD epochs per update step |
| `num_repeats` | 1 | Independent training seeds (vmapped) |
| `lr` | 3e-4 | Adam learning rate (cosine-decayed to 10% over all gradient steps) |
| `lr_warmup_frames` | `null` | **PRIMARY** env-frame warm-up budget. Derives `lr_warmup_steps` as `lr_warmup_frames // (num_envs * num_steps)`. |
| `lr_warmup_steps` | 0 | **LEGACY** linear warm-up steps before cosine decay (used when `lr_warmup_frames` is unset; 0 = disabled). |
| `max_grad_norm` | 1.0 | Global gradient clipping norm |
| `return_weight_cap` | 5.0 | Clip ceiling for per-window return weights (lower clip is fixed at 0.1) |
| `collect_temperature` | 1.0 | Softmax temperature on PPO logits during live data collection |
| `val_interval_frames` | `null` | **PRIMARY** env-frames between validation rollouts. Overrides `val_interval` via `val_interval = val_interval_frames // (num_envs * num_steps)`. |
| `val_interval` | 50 | **LEGACY** validation frequency in update steps (used when `val_interval_frames` is unset). |
| `val_diffusion_steps` | 50 | Denoising steps used during validation rollouts |
| `val_replan_every` | 4 | Environment steps executed per diffusion plan during validation |
| `val_steps` | 128 | Total environment steps per validation rollout |

**Online DAgger training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `online_total_timesteps` | `null` | **PRIMARY** env-frame budget for online DAgger (hardware-portable). Derives `num_updates` as `online_total_timesteps // (num_envs * num_steps)`. |
| `online_num_updates` | 1000 | **LEGACY** outer DAgger iterations (used when `online_total_timesteps` is unset). |
| `dagger_beta_init` | 1.0 | Initial expert mixing probability `beta_1` (1.0 = pure expert on the first iteration). |
| `dagger_beta_final` | `null` | **PRIMARY** target mixing ratio at the end of training. Overrides `dagger_beta_decay` via `decay = (beta_final / beta_init) ** (1 / num_updates)`. |
| `dagger_beta_decay` | 0.95 | **LEGACY** per-update decay: `beta_i = beta_init * decay^i` (used when `dagger_beta_final` is unset). |
| `dagger_buffer_cycles` | `null` | **PRIMARY** buffer capacity denominated in update cycles of history (1 cycle = `num_envs * num_steps` frames). Overrides `dagger_buffer_max` via `buffer_max = cycles * (num_envs * num_steps)`. |
| `dagger_buffer_max` | 100000 | **LEGACY** max samples in the DAgger replay buffer (circular eviction when full). |
| `dagger_train_passes` | `null` | Passes per update over the aggregated buffer. `null` = 1 pass (matches offline BC per-update gradient work exactly for fair compute comparison). Raise to >1 to trade BC fairness for wider per-update buffer coverage. |
| `dagger_expert_deterministic` | `true` | If `true`, the PPO expert takes the argmax action (fixed `s → a*` map); if `false`, it samples categorically. Deterministic removes label noise from the aggregated dataset. |

**Data collection**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `collect_num_steps` | 10000000 | Total environment steps to collect |
| `collect_num_envs` | 128 | Parallel environments during collection |
| `ppo_model_type` | `ppo_rnn` | PPO architecture: `ppo`, `ppo_rnn`, or `ppo_rnd` |
| `layer_size` | 512 | PPO actor-critic hidden layer width |

**Inference**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `eval_steps` | 10000 | Environment steps for evaluation |
| `eval_num_envs` | 32 | Parallel agents during evaluation (independent of `num_envs`) |
| `diffusion_steps_eval` | 10 | Denoising steps T used at evaluation time |

**Checkpointing**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_policy` | `true` | Save final checkpoint at end of training and upload it as a W&B artifact |

**Resume**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `resume_checkpoint_path` | `null` | Path to a completed checkpoint to resume from (accepts `wandb:` refs) |
| `resume_wandb_run_id` | `null` | W&B run ID to resume logging into (auto-read from checkpoint metadata) |
| `resume_step` | `null` | Update step the checkpoint was saved at (auto-read from checkpoint metadata) |

**Logging**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_wandb` | `true` | Enable Weights & Biases logging |
| `wandb_project` | `remdm-craftax` | W&B project name |
| `wandb_entity` | `"mathis-weil-university-college-london-ucl-"` | W&B entity (team or username) |
| `wandb_download_dir` | `null` | Download directory for W&B artifacts; null = `./artifacts/` |
| `seed` | `null` | RNG seed (random if null) |

---

## Remasking Strategies

Controlled by `--remask_strategy`. All strategies operate on top of the three-phase loop controlled by `--use_loop`, `--t_on`, and `--t_off`.

| Strategy | Formula | Description |
|----------|---------|-------------|
| `rescale` | `sigma = eta * sigma_max` | Scales maximum remasking probability proportionally |
| `cap` | `sigma = min(eta, sigma_max)` | Caps remasking at a fixed rate |
| `conf` | `sigma = eta * sigma_max * (1 - confidence)` | High-confidence tokens are remasked less |

---

## Environment Wrappers

**From `Craftax_Baselines/wrappers.py`** (submodule):

| Wrapper | Purpose |
|---------|---------|
| `LogWrapper` | Tracks episode returns and lengths; adds stats to the info dict |
| `AutoResetEnvWrapper` | Automatically resets episodes on `done` |
| `BatchEnvWrapper` | Vmaps `reset` and `step` over `num_envs` environments |
| `OptimisticResetVecEnvWrapper` | Batched resets with reduced overhead; enable via `--use_optimistic_resets` |

Stack (identical for training and inference):

```
env -> LogWrapper -> AutoResetEnvWrapper -> BatchEnvWrapper
```

With `--use_optimistic_resets`, `OptimisticResetVecEnvWrapper` replaces the last two.

---

## Project Structure

```
craftax-ReMDM-planner/
├── Craftax_Baselines/             # Git submodule — PPO agents and standard wrappers
│   ├── wrappers.py                # LogWrapper, BatchEnvWrapper, AutoResetEnvWrapper, etc.
│   ├── ppo_rnn.py                 # PPO-RNN training script
│   ├── ppo_rnd.py                 # PPO-RND training script
│   ├── ppo.py                     # PPO model definitions
│   └── models/
│       ├── actor_critic.py        # ActorCritic variants
│       ├── rnd.py                 # RND network
│       └── icm.py                 # ICM encoder, forward, and inverse networks
├── configs/                       # defaults.yaml, smoke.yaml, exp_{a,b,c,d}, final_* (see Configuration)
├── src/
│   ├── diffusion/
│   │   ├── forward.py             # Forward masking process q(z_t | x_0)
│   │   ├── loss.py                # Continuous-time MDLM ELBO loss
│   │   ├── sampling.py            # Reverse diffusion with ReMDM remasking
│   │   └── schedules.py           # Linear and cosine noise schedules
│   ├── models/
│   │   └── denoiser.py            # DenoisingTransformer (obs encoder + transformer)
│   └── planners/
│       ├── collect.py             # --mode collect: PPO rollouts -> .npz
│       ├── common.py              # Shared utilities
│       ├── env.py                 # Environment construction
│       ├── inference.py           # --mode inference: MPC evaluation with inpainting
│       ├── logging.py             # Centralised W&B logging utilities
│       ├── model.py               # Diffusion model lifecycle
│       ├── offline.py             # --mode offline: make_train (live PPO rollouts)
│       ├── online.py              # --mode online: DAgger fine-tuning
│       ├── ppo.py                 # PPO agent adapter and checkpoint loading utilities
│       └── smoke.py               # --mode smoke: shrunken end-to-end run via run_online
├── experiments/
│   └── rl_finetuning/             # RL fine-tuning ablation suite (see experiments/README.md)
│       ├── run_ablations.py       # CLI entry point
│       ├── ablations/             # Loss, optimizer, registry, and training modules
│       ├── diagnostics/           # Gradient, representation, and timestep diagnostics
│       ├── analysis/              # Plots, tables, and report generation
│       └── configs/               # ablations_default.yaml, ablations_fast.yaml,
│                                  # ablations_final_{classic,craftax}_{ucl,qmul}.yaml
├── tests/                         # Smoke suite — uv run pytest
│   ├── conftest.py                # Shared fixtures, CPU/seed guards, entry-point runner
│   ├── test_smoke_src.py          # src/ pipeline
│   └── test_smoke_experiments.py  # experiments/ ablation pipeline
├── scripts/
│   ├── count_params.py            # Exact parameter counts per config (+--verify_checkpoint)
│   ├── eval_ppo_expert.py         # First-episode evaluation of a PPO expert
│   ├── hf_upload.py               # Publish checkpoints to the HF Hub
│   └── hf_upload_demo.py          # Publish the demo notebook
├── checkpoints/                   # Gitignored — offline/, online/, ppo_agents/ (see HF Hub)
├── demo_craftax.ipynb             # Walkthrough notebook
├── main.py                        # CLI entry point
├── pyproject.toml                 # uv project — direct deps, dev group, pytest config
└── uv.lock                        # Reproducible lockfile (commit this)
```

---

## Implementation Notes

| Topic | Note |
|---|---|
| JAX purity | `make_train` / `make_train_dagger` are fully JIT-compatible; env construction and checkpoint I/O sit outside `jax.jit`. |
| Offline data | `--mode offline` rolls out PPO live. `--mode collect` saves an `.npz` for inspection only — re-feeding it to `--mode offline` is unsupported; pass `--ppo_checkpoint_path`. |
| Episode-boundary masking | A window at `(e, t)` is valid only if `dones[e, t+1:t+H-1]` are all `False`. |
| Return weighting | Valid windows are weighted by cumulative reward, normalised by the batch mean, clipped to `[0.1, return_weight_cap]`, and applied as per-sample multipliers before loss reduction. |
| LR schedule | Cosine decay `lr → lr * 0.1` over all gradient steps. `lr_warmup_frames` (PRIMARY) or `lr_warmup_steps` (LEGACY) prepends linear warm-up. |
| Env-frame invariance | PRIMARY keys (`*_total_timesteps`, `lr_warmup_frames`, `val_interval_frames`, `dagger_beta_final`, `dagger_buffer_cycles`) are converted to update-step form by `resolve_scaled_hyperparams()` using `num_envs * num_steps`, so one config runs on any hardware tier. |
| DAgger sizing | `dagger_sizing()` in `src/planners/common.py` is the single source of truth for `samples_per_update`, buffer capacity and `n_train_passes`; the runner and the config banner both read it. |
| Loss weight clipping | The MDLM SUBS weight `-alpha'(t) / (1 - alpha_t)` is clipped to 1000 for stability as `alpha_t → 1`. |
| Validation rollouts | Every `val_interval` updates, using inference sampling parameters with `val_diffusion_steps`, `val_replan_every` and `val_steps`. |
| W&B namespaces | Centralised in `src/planners/logging.py`: `diffusion/`, `train/`, `env/`, `val/`, `dagger/`. `train/sps` only in modes with live env interaction. |
| DAgger aggregation | Ross et al. (2011). A circular buffer accumulates `(obs, expert_plan)` across iterations; each update samples the full buffer. Windows use a sliding stride so every visited state contributes a label. The expert receives correct `done` flags so its RNN state resets at episode boundaries. |
| Best-checkpoint tracking | Highest-validation-return parameters are kept alongside the live ones and uploaded as `{env_name}-policy-best`. |
| Denoising indexing | Reverse scan runs `step_idx = 0 → T-1`, mapping to `t = (T - step_idx) / T` (high to low noise). |
| PPO agents | Training lives entirely in `Craftax_Baselines/`; planner modes only consume checkpoints. |
