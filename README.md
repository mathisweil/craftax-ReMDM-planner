# ReMDM Planner — Discrete Diffusion Planning on Craftax

A JAX implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in the [Craftax](https://github.com/MichaelTMatthews/Craftax) environment. A bidirectional transformer learns to generate action plans by iteratively denoising masked token sequences, conditioned on the current environment observation.

---

## Description

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `T` denoising steps, producing a `plan_horizon`-length plan. The ReMDM framework extends standard Masked Discrete Language Modelling (MDLM) with remasking strategies that allow committed tokens to be re-predicted, improving plan coherence.

Two independent training pipelines are available — **Offline BC** and **Online DAgger** — both supervised by a pre-trained PPO expert but otherwise separate. Neither depends on the other; the paper compares them head-to-head.

```
[Shared]   Train PPO agent              Craftax_Baselines/ppo_rnn.py | ppo_rnd.py
               |
               v  checkpoint
       ┌───────┴────────┐
       │                │
  [Offline BC]     [Online DAgger]
  main.py              main.py
  --mode offline        --mode online
  (train on live        (train from scratch;
   PPO rollouts)         mixed policy + expert
       │                 labels into replay buffer)
       v                 v
   checkpoint        checkpoint
       │                │
       └───────┬────────┘
               v
[Evaluate] main.py --mode inference --checkpoint_path ...

Optional: an offline BC checkpoint can warm-start DAgger
via --offline_checkpoint_path (not used in the paper).

  [Offline BC] ──checkpoint──> [Online DAgger]
```

**Optional utility modes:**
```
[Collect]     Save PPO rollouts to disk   main.py --mode collect
[Smoke test]  Quick end-to-end check      main.py --mode smoke
```

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

Full specification in `pyproject.toml`. Exact transitive pins are in `uv.lock`.

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

The file stores arrays shaped `[num_envs, num_iters, ...]`, preserving per-environment contiguity so episode boundaries are respected during window sampling.

### Stage 2b — Train offline from live PPO rollouts

Roll out the PPO agent live at each update step and train the diffusion model on the collected windows. Windows that cross episode boundaries are masked out; windows with higher cumulative reward receive proportionally larger gradient contributions (clipped to `[0.1, return_weight_cap]`).

```bash
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_total_timesteps 100000000 \
    --save_policy
```

### Online DAgger Training

The diffusion model is trained **from scratch** via DAgger (Dataset Aggregation). At each iteration a mixed policy blends the PPO expert and the diffusion learner (controlled by an exponentially decaying `beta`). The mixed policy rolls out trajectories; the expert labels every visited state with the action it would take. These `(obs, expert_plan)` pairs are appended to a growing circular replay buffer, and the diffusion model is trained on the full buffer with the standard MDLM ELBO loss (pure behavioural cloning — no reward weighting).

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

When `save_policy=true`, online training uploads **two** W&B artifacts: `{env_name}-policy` (final weights) and `{env_name}-policy-best` (weights from the validation iteration with the highest return). Either artifact can be consumed downstream by `--checkpoint_path wandb:…`.

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

Runs the full DAgger pipeline — rollout, expert labelling, gradient updates, validation — under the shrunken overrides in `configs/smoke.yaml`, then prints the final loss, returns and achievement metrics. Roughly 25 seconds on CPU. `src/planners/smoke.py` delegates the training loop to `run_online`; it does not reimplement it.

Two things to know before reading the output:

- **The expert is random unless you supply one.** With no `--ppo_checkpoint_path`, the mode generates a randomly initialised PPO expert so it runs on a clean clone with no downloads. It proves the pipeline executes; it says nothing about learning. Pass `--ppo_checkpoint_path` to use a trained expert.
- **Returns and achievements read `0.000`.** Those metrics are episode-weighted, and a smoke run is far shorter than a Craftax episode, so no episode terminates. Watch `mean step reward`, `loss` and `all metrics finite` instead.

`configs/smoke.yaml` holds overrides only and is layered on top of `configs/defaults.yaml`. Sizing is constrained by invariants `make_train_dagger` asserts on *derived* values (`num_steps % plan_horizon`, `samples_per_update % num_minibatches`, `samples_per_update <= dagger_buffer_max`), so shrinking the frame budget alone does not shrink the run — see the comments in the file.

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

The trained weights are too large for git — `checkpoints/` is gitignored — and are hosted on the Hugging Face Hub at [`MathisW78/remdm-craftax-checkpoints`](https://huggingface.co/MathisW78/remdm-craftax-checkpoints). The Hub repo mirrors the layout below, so downloading into the repository root puts every file exactly where the CLI expects it.

| Checkpoint directory | Environment | Role | Trained for |
|---|---|---|---|
| `checkpoints/offline/Craftax-Classic-Symbolic-v1-OfflineDiffusion-BC-100M` | Craftax Classic | Offline BC planner | 1e8 env frames |
| `checkpoints/offline/Craftax-Symbolic-v1-OfflineDiffusion-BC-100M` | Full Craftax | Offline BC planner | 1e8 env frames |
| `checkpoints/online/Craftax-Classic-Symbolic-v1-OnlineDiffusion-DAgger-100M` | Craftax Classic | Online DAgger planner | 1e8 env frames |
| `checkpoints/online/Craftax-Symbolic-v1-OnlineDiffusion-DAgger-100M` | Full Craftax | Online DAgger planner | 1e8 env frames |
| `checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M` | Craftax Classic | PPO-RNN expert | 1e9 env frames |
| `checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M` | Full Craftax | PPO-RNN expert | 1e9 env frames |

Download all six (~470 MB):

```bash
uv run hf download MathisW78/remdm-craftax-checkpoints --include "checkpoints/**" --local-dir .
```

Or fetch a single checkpoint:

```bash
uv run hf download MathisW78/remdm-craftax-checkpoints --include "checkpoints/online/Craftax-*/**" --local-dir .
```

**Pass the checkpoint directory, not the step subdirectory.** `orbax.checkpoint.CheckpointManager` resolves the latest step itself.

**Match the config to the checkpoint.** The model is built from the config, not from the checkpoint, so loading a released checkpoint under `configs/defaults.yaml` (`d_model` 256, `n_layers` 4) fails with a shape mismatch during restore. All released diffusion checkpoints are `d_model` 384, `n_heads` 8, `n_layers` 6, `d_ff` 768 — use the matching final config, which also sets the correct `env_name`:

```bash
# Craftax Classic — online DAgger planner
python main.py --mode inference \
    --config configs/final_classic_ucl.yaml \
    --checkpoint_path checkpoints/online/Craftax-Classic-Symbolic-v1-OnlineDiffusion-DAgger-100M

# Full Craftax — offline BC planner
python main.py --mode inference \
    --config configs/final_craftax_ucl.yaml \
    --checkpoint_path checkpoints/offline/Craftax-Symbolic-v1-OfflineDiffusion-BC-100M

# Train a new planner against the released Full Craftax PPO expert
python main.py --mode online \
    --config configs/final_craftax_ucl.yaml \
    --ppo_checkpoint_path checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M \
    --save_policy
```

Each diffusion checkpoint carries a `resume_metadata.json` sidecar holding the full config snapshot of the run that produced it; the PPO checkpoints carry `config.yaml` and `wandb-summary.json`. The sidecar is the authoritative record of a run's hyperparameters, and is what `--resume_checkpoint_path` reads to auto-detect `resume_step` and `resume_wandb_run_id`.

Re-uploading after retraining is handled by `scripts/hf_upload.py`, which rediscovers every checkpoint, strips wandb environment metadata, and regenerates the model card:

```bash
HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py --repo-id MathisW78/remdm-craftax-checkpoints
```

### Resuming a Training Run

A completed training checkpoint can be used as the starting point for a new run that continues where the previous one left off. This is useful when extending the training budget or when a preempted job needs to be restarted.

**Offline resume:**

```bash
# Auto-detect step and wandb run ID from checkpoint metadata
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --resume_checkpoint_path /path/to/completed_offline_checkpoint \
    --offline_total_timesteps 200000000 \
    --save_policy

# Explicit step and wandb run ID override
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --resume_checkpoint_path /path/to/completed_offline_checkpoint \
    --resume_step 1525 \
    --resume_wandb_run_id abc123xyz \
    --offline_total_timesteps 200000000 \
    --save_policy

# Resume from a W&B artifact
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --resume_checkpoint_path wandb:my-team/remdm-craftax/policy:latest \
    --offline_total_timesteps 200000000 \
    --save_policy
```

**Online resume:**

```bash
python main.py --mode online \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --resume_checkpoint_path /path/to/completed_online_checkpoint \
    --online_num_updates 2000 \
    --save_policy
```

**Notes:**
- The DAgger replay buffer is **not** persisted across resumes. It starts empty and refills within the first few iterations.
- JIT compilation is fully preserved. Resume only affects initialisation outside `jax.jit` (loading checkpoint, setting the optimizer step counter, adjusting scan length).
- The cosine LR schedule is constructed for the full `num_updates` range. The optimizer step counter is set to the resume offset so the learning rate picks up exactly where the previous run stopped.
- When `resume_checkpoint_path` points to a checkpoint with a metadata sidecar, `resume_step` and `resume_wandb_run_id` are auto-detected. Explicit CLI flags override the metadata values.
- Checkpoints without a metadata sidecar (created before this feature) still load; provide `--resume_step` explicitly.


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
| `configs/defaults.yaml` | Base defaults for all modes |
| `configs/classic_exp_a_beta_fix.yaml` | Craftax Classic DAgger — beta decay fix only (isolates data quality) |
| `configs/classic_exp_b_beta_big_model.yaml` | Craftax Classic DAgger — beta fix + 3.5× larger transformer |
| `configs/classic_exp_c_full_recipe.yaml` | Craftax Classic DAgger — beta + big model + training dynamics |
| `configs/craftax_exp_a_beta_fix.yaml` | Full Craftax DAgger — beta decay fix only |
| `configs/craftax_exp_b_beta_big_model.yaml` | Full Craftax DAgger — beta fix + larger transformer |
| `configs/craftax_exp_c_full_recipe.yaml` | Full Craftax DAgger — full recipe |
| `configs/final_classic_ucl.yaml` | Final Craftax Classic DAgger — UCL 3090 Ti, seed 42 (produces the Classic checkpoint consumed by the ablation suite) |
| `configs/final_classic_qmul.yaml` | Env-frame-matched second seed of `final_classic_ucl.yaml` (QMUL H200, seed 43) |
| `configs/final_craftax_ucl.yaml` | Final Full Craftax DAgger — UCL 4090, seed 42 (produces the Full Craftax checkpoint consumed by the ablation suite) |
| `configs/final_craftax_qmul.yaml` | Env-frame-matched second seed of `final_craftax_ucl.yaml` (QMUL H200, seed 43) |

RL fine-tuning ablation hyperparameters live under `experiments/rl_finetuning/configs/` and are loaded by `run_ablations.py`, not by `main.py`. See `experiments/README.md`.

The `final_*_qmul.yaml` presets differ from their UCL counterparts only in `num_envs` (smaller partition) and `seed`. All fairness-critical hyperparameters are denominated in env frames or update cycles and automatically rescaled by `resolve_scaled_hyperparams()` at load time, so no manual derivation is needed when moving between hardware tiers.

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

**Wrapper stacks:**

```
Training:   env -> LogWrapper -> AutoResetEnvWrapper -> BatchEnvWrapper
Inference:  env -> LogWrapper -> AutoResetEnvWrapper -> BatchEnvWrapper
```

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
├── configs/
│   ├── defaults.yaml                        # Base hyperparameters (CLI-overridable)
│   ├── classic_exp_a_beta_fix.yaml          # Classic DAgger — beta decay fix only
│   ├── classic_exp_b_beta_big_model.yaml    # Classic DAgger — beta fix + big model
│   ├── classic_exp_c_full_recipe.yaml       # Classic DAgger — full recipe
│   ├── craftax_exp_a_beta_fix.yaml          # Full Craftax DAgger — beta fix
│   ├── craftax_exp_b_beta_big_model.yaml    # Full Craftax DAgger — beta + big model
│   ├── craftax_exp_c_full_recipe.yaml       # Full Craftax DAgger — full recipe
│   ├── final_classic_ucl.yaml               # Classic DAgger — UCL 3090 Ti, seed 42
│   ├── final_classic_qmul.yaml              # Classic DAgger — QMUL H200, seed 43
│   ├── final_craftax_ucl.yaml               # Full Craftax DAgger — UCL 4090, seed 42
│   └── final_craftax_qmul.yaml              # Full Craftax DAgger — QMUL H200, seed 43
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
├── main.py                        # CLI entry point
├── pyproject.toml                 # uv project — direct deps + tool config
└── uv.lock                        # Reproducible lockfile (commit this)
```

---

## Implementation Notes

**JAX functional purity**: training closures (`make_train`, `make_train_dagger`) are fully JIT-compatible. Environment construction and checkpoint I/O happen outside `jax.jit`.

**Offline training**: `--mode offline` rolls out the PPO agent live at each update step via `make_train`. Use `--mode collect` to save a trajectory `.npz` for inspection or analysis; re-feeding it to `--mode offline` is not supported — pass `--ppo_checkpoint_path` instead.

**Episode-boundary masking**: the offline sampler pre-computes a validity mask over all `(env, time)` positions. A window at `(e, t)` is valid only if `dones[e, t+1:t+H-1]` are all `False`.

**Return weighting**: valid windows are weighted by their cumulative reward, normalised by the batch mean and clipped to `[0.1, RETURN_WEIGHT_CAP]`. Weights are passed as per-sample multipliers into the MDLM loss before reduction, so they correctly scale each sample's gradient contribution.

**LR schedule**: cosine decay from `lr` to `lr * 0.1` over all gradient steps. Set `lr_warmup_frames > 0` (env-frame-invariant, PRIMARY) or `lr_warmup_steps > 0` (LEGACY) to prepend a linear warm-up phase.

**Env-frame-invariant hyperparameters**: the PRIMARY keys `offline_total_timesteps`, `online_total_timesteps`, `lr_warmup_frames`, `val_interval_frames`, `dagger_beta_final`, and `dagger_buffer_cycles` are denominated in env frames (or update cycles). At config load time, `resolve_scaled_hyperparams()` in `src/planners/common.py` converts them to the equivalent update-step-denominated quantities (`num_updates`, `lr_warmup_steps`, `val_interval`, `dagger_beta_decay`, `dagger_buffer_max`) using the current `num_envs * num_steps` frames-per-update. This lets the same config run on different hardware tiers without re-tuning.

**Loss weight clipping**: the MDLM SUBS weight `-alpha'(t) / (1 - alpha_t)` is clipped to 1000 to prevent numerical instability when `alpha_t ≈ 1`.

**Validation rollouts**: during offline training, a held-out rollout runs every `val_interval` steps. It uses the same sampling parameters as inference (`remask_strategy`, `eta`, `use_loop`, `t_on`, `t_off`, `temperature`, `top_p`) with `val_diffusion_steps` denoising steps and `val_replan_every` env steps per plan, for a total of `val_steps` environment steps.

**W&B logging**: all metric aggregation is centralised in `src/planners/logging.py`. Metric namespaces: `diffusion/` (loss, accuracy), `train/` (data quality, throughput), `env/` (episode returns, achievements), `val/` (validation rollouts, emitted every `val_interval` steps), `dagger/` (online DAgger training: beta, buffer fill, reward mean, valid fraction). `train/sps` (environment frames/sec) is only logged in modes that perform live environment interaction.

**DAgger dataset aggregation**: online training (`--mode online`) implements DAgger (Ross et al., 2011). A circular replay buffer accumulates `(obs, expert_plan)` pairs across all iterations. Each update samples uniformly from the full buffer, not just the latest batch. Training samples that cross episode boundaries (any `done` within the plan-horizon window) are marked invalid. The expert (PPO agent) receives correct `done` flags so its RNN hidden state resets on episode boundaries. Windows are extracted with a sliding stride (one per env-time position) rather than stepping the buffer in plan-horizon chunks, so every visited state contributes a label.

**Best-checkpoint tracking**: during online training, the parameters from the validation iteration with the highest validation return are preserved alongside the current live parameters. The final checkpoint and the best-validation checkpoint are both uploaded as separate W&B artifacts (`{env_name}-policy` and `{env_name}-policy-best`).

**Denoising step indexing**: the reverse scan runs from `step_idx = 0` to `T-1`, mapping to diffusion time `t = (T - step_idx) / T` (high noise to low noise).

**Submodule PPO agents**: PPO training lives entirely in `Craftax_Baselines/`. Planner scripts only consume pre-trained checkpoints via `--ppo_checkpoint_path`.
