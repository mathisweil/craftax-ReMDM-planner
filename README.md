# ReMDM Planner — Discrete Diffusion Planning on Craftax

A JAX implementation of **ReMDM** (Remasking Discrete Diffusion Model) applied to action-sequence planning in the [Craftax](https://github.com/MichaelTMatthews/Craftax) environment. The model learns to generate action plans by iteratively denoising masked token sequences, conditioned on the current environment observation.

---

## Overview

The planner trains a **bidirectional transformer** to denoise masked action sequences. At inference time, it starts from a fully-masked plan and iteratively unmasks tokens over `T` denoising steps, producing a `plan_horizon`-length action sequence. The ReMDM framework extends standard MDLM with four **remasking strategies** that allow previously-committed tokens to be re-masked and re-predicted, improving plan coherence.

```
Observation  ──►  DenoisingTransformer  ──►  Action plan [a₁, a₂, …, aH]
                  (conditioned on obs)
```

### Key components

| File | Role |
|------|------|
| `src/planners/planners.py` | Training and inference entry point (collect / offline / online / inference modes) |
| `src/models/remdm.py` | Diffusion logic: schedules, forward process, MDLM SUBS loss, remasking strategies, reverse sampling |
| `src/models/denoiser.py` | `DenoisingTransformer`: observation MLP encoder + sinusoidal time embedding + bidirectional transformer |
| `src/envs/wrappers.py` | Custom environment wrappers: `PlannerWrapper`, `SequenceHistoryWrapper`, `OfflineTrajectoryWrapper`, `DiscreteTokenizationWrapper` |
| `Craftax_Baselines/` | Git submodule — PPO agents (`ppo_rnn.py`, `ppo_rnd.py`) and standard Gymnax wrappers (`LogWrapper`, `BatchEnvWrapper`, etc.) |
| `configs/defaults.yaml` | All hyperparameters with documentation |

---

## Installation

```bash
conda env create -f environment.yml
conda activate craftax
```

After creating the environment, initialise the `Craftax_Baselines` submodule:

```bash
git submodule update --init --recursive
```

Core dependencies: `jax[cuda12]`, `flax`, `optax`, `orbax-checkpoint`, `craftax`, `gymnax`, `distrax`, `chex`, `wandb`.

> **GPU note**: the `environment.yml` pins `jax[cuda12]`. For CPU-only or different CUDA versions, edit that line before creating the environment.

---

## Workflow

The pipeline has four sequential stages. Each stage can be run independently.

```
[Stage 1]  Train PPO agent           Craftax_Baselines/ppo_rnn.py / ppo_rnd.py
               │
               ▼ checkpoint
[Stage 2]  Collect / train offline   planners.py --mode collect
               │                     planners.py --mode offline
               ▼ diffusion checkpoint
[Stage 3]  Online fine-tuning        planners.py --mode online
               │
               ▼ fine-tuned checkpoint
[Stage 4]  Evaluate                  planners.py --mode inference
```

All commands below assume the repo root as working directory and the `craftax` conda environment is active.

### Stage 1 — Train a PPO agent

PPO training is handled by the `Craftax_Baselines` submodule and is fully separate from the diffusion planner.

```bash
# PPO with GRU hidden state (recommended)
cd Craftax_Baselines
python ppo_rnn.py \
    --env_name Craftax-Symbolic-v1 \
    --total_timesteps 500000000 \
    --save_policy \
    --use_wandb
cd ..

# PPO with Random Network Distillation (exploration bonus)
cd Craftax_Baselines
python ppo_rnd.py \
    --env_name Craftax-Symbolic-v1 \
    --total_timesteps 500000000 \
    --save_policy \
    --use_wandb
cd ..
```

Both scripts save an `ActorCritic` checkpoint to the W&B run directory when `--save_policy` is set.

---

### Stage 2a — Collect trajectories to disk

Roll out the trained PPO checkpoint and save `(obs, actions, dones)` as a `.npz` file. This lets you re-use the same data for multiple diffusion training runs.

```bash
python -m src.planners.planners --mode collect \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_data_path data/trajectories.npz \
    --collect_num_steps 1000000 \
    --collect_num_envs 64
```

The saved file has shape `[num_envs, num_iters, ...]`, preserving per-environment trajectory contiguity so that the offline training sampler can respect episode boundaries.

---

### Stage 2b — Train offline directly from a PPO agent (no disk I/O)

Pass `--ppo_checkpoint_path` to `--mode offline` to skip saving trajectories. The PPO agent is rolled out live at each training step and the diffusion model is trained immediately on the collected windows.

```bash
python -m src.planners.planners --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --num_train_steps 100000 \
    --save_policy
```

This is the recommended approach when you want to iterate quickly on diffusion model hyperparameters without re-running data collection.

---

### Stage 3 — Train offline from saved trajectories

```bash
python -m src.planners.planners --mode offline \
    --offline_data_path data/trajectories.npz \
    --num_train_steps 100000 \
    --save_policy
```

The sampler automatically masks out windows that cross episode boundaries (`dones`), so the model is never trained on transitions from two different episodes.

---

### Stage 4 — Online fine-tuning

The diffusion model acts as its own policy: it generates plans, executes them in the environment, and trains on the resulting `(obs, plan)` pairs via self-imitation.

```bash
# Fine-tune from scratch
python -m src.planners.planners --mode online \
    --num_updates 1000 \
    --save_policy

# Warm-start from an offline checkpoint
python -m src.planners.planners --mode online \
    --offline_checkpoint_path /path/to/offline_checkpoint \
    --num_updates 1000 \
    --save_policy
```

---

### Stage 5 — Evaluate

```bash
python -m src.planners.planners --mode inference \
    --checkpoint_path /path/to/checkpoint \
    --eval_steps 1000 \
    --num_envs 32
```

Prints mean episode return, number of completed episodes, and steps per second.

---

## Configuration

All hyperparameters live in `configs/defaults.yaml`. Any value can be overridden from the command line.

```bash
# Example: override learning rate and plan horizon
python -m src.planners.planners --mode offline \
    --ppo_checkpoint_path /path/to/ppo \
    --lr 1e-4 \
    --plan_horizon 64
```

You can also point to a custom config file:

```bash
python -m src.planners.planners --mode online --config configs/my_experiment.yaml
```

### Key hyperparameters

**Diffusion model**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plan_horizon` | 32 | Length of the action plan H |
| `diffusion_steps` | 50 | Number of denoising steps T at inference |
| `diffusion_schedule` | `cosine` | Noise schedule: `cosine` or `linear` |
| `remask_strategy` | `rescale` | Remasking strategy: `rescale`, `cap`, `conf`, `loop` |
| `eta` | 0.5 | Remasking strength |
| `t_on` / `t_off` | 0.7 / 0.3 | Time window for the `loop` remasking strategy |

**Transformer architecture**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Transformer hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 4 | Number of transformer blocks |
| `d_ff` | 512 | FFN inner dimension |
| `obs_encoder_layers` | 2 | MLP layers in the observation encoder |
| `obs_encoder_width` | 512 | Observation encoder hidden width |

**Online training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 32 | Parallel environments |
| `num_steps` | 128 | Environment steps per update |
| `replan_every` | 8 | Steps executed per plan before replanning |
| `num_updates` | 1000 | Number of outer update iterations |
| `use_optimistic_resets` | false | Use `OptimisticResetVecEnvWrapper` for faster resets |

**Logging**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_wandb` | true | Enable Weights & Biases logging |
| `wandb_project` | `remdm-craftax` | W&B project name |
| `save_policy` | false | Save model checkpoint at end of training |
| `seed` | null | RNG seed (random if null) |
| `debug` | true | Enable per-step W&B logging |

---

## Remasking Strategies

The reverse diffusion pass uses one of four remasking strategies, controlled by `--remask_strategy`:

| Strategy | Description |
|----------|-------------|
| `rescale` | `σ = η · σ_max`. Proportionally scales maximum remasking probability. |
| `cap` | `σ = min(η, σ_max)`. Caps remasking at a fixed rate. |
| `conf` | Per-token: high-confidence tokens are remasked less. `σ = η · σ_max · (1 − confidence)`. |
| `loop` | Remasking only active in time window `[t_off, t_on]`; zero outside. Controlled by `--t_on` and `--t_off`. |

---

## Environment Wrappers

**From `Craftax_Baselines/wrappers.py`** (submodule):

| Wrapper | Purpose |
|---------|---------|
| `LogWrapper` | Tracks episode returns and lengths; adds `returned_episode_returns`, `returned_episode_lengths`, `returned_episode` to the info dict. |
| `AutoResetEnvWrapper` | Automatically resets episodes on `done`. |
| `BatchEnvWrapper` | Vmaps `reset` and `step` over `num_envs` parallel environments. |
| `OptimisticResetVecEnvWrapper` | Efficient batched resets: only resets a subset of environments at each step, reducing reset overhead. Enable with `--use_optimistic_resets`. |

**From `src/envs/wrappers.py`** (project-specific):

| Wrapper | Purpose |
|---------|---------|
| `PlannerWrapper` | Manages the plan/replan cycle for inference. Calls the diffusion model every `replan_every` steps. |
| `SequenceHistoryWrapper` | Maintains a sliding window of past observations and actions in the env state. |
| `OfflineTrajectoryWrapper` | Accumulates transitions into a circular replay buffer inside the JAX state. |
| `DiscreteTokenizationWrapper` | Quantizes continuous observations into discrete token indices. |

**Standard wrapper stack used in training:**

```
env  →  LogWrapper  →  AutoResetEnvWrapper  →  BatchEnvWrapper
```

**Inference wrapper stack:**

```
env  →  LogWrapper  →  AutoResetEnvWrapper  →  BatchEnvWrapper  →  PlannerWrapper
```

---

## Project Structure

```
craftax-ReMDM-planner2/
├── Craftax_Baselines/             # Git submodule (MichaelTMatthews/Craftax_Baselines)
│   ├── wrappers.py                # LogWrapper, BatchEnvWrapper, AutoResetEnvWrapper, etc.
│   ├── ppo_rnn.py                 # PPO-RNN training script (data generation)
│   ├── ppo_rnd.py                 # PPO-RND training script (data generation)
│   ├── ppo.py                     # PPO model definitions
│   ├── models/
│   │   ├── actor_critic.py        # ActorCritic MLP variants
│   │   ├── rnd.py                 # RND network and ActorCriticRND
│   │   └── icm.py                 # ICM encoder, forward, and inverse networks
│   ├── logz/
│   │   └── batch_logging.py       # W&B batch logging utilities
│   └── analysis/
│       └── view_ppo_agent.py      # PPO agent visualisation
├── configs/
│   └── defaults.yaml              # All hyperparameters
├── src/
│   ├── planners/
│   │   └── planners.py            # Main entry point (collect/offline/online/inference)
│   ├── models/
│   │   ├── denoiser.py            # DenoisingTransformer
│   │   └── remdm.py               # Diffusion core: schedules, loss, remasking, sampling
│   └── envs/
│       └── wrappers.py            # Project-specific wrappers (PlannerWrapper, etc.)
├── environment.yml                # Conda environment spec
└── README.md
```

---

## Implementation Notes

**JAX functional purity**: all training functions (`make_train_offline`, `make_train_online`) return a `train(rng)` callable that is fully JIT-compatible. Environment construction and checkpoint I/O happen outside `jax.jit`.

**Episode-boundary masking**: the offline sampler pre-computes a validity mask over all `(env, time)` positions. A window starting at `(e, t)` is valid only if `dones[e, t:t+H-1]` are all `False`, ensuring the model is never trained across episode boundaries.

**Loss weight clipping**: the MDLM SUBS loss weight `−α'(t) / (1 − α_t)` is clipped to 1000 to prevent numerical instability when `t` is very small and `α_t ≈ 1`.

**Online training repeats**: `run_online` uses a sequential loop over `num_repeats` rather than `jax.vmap`, because `BatchEnvWrapper` already vmaps `step/reset` internally — adding an outer vmap would create `vmap(vmap(step))`.

**Denoising step indexing**: the scan over denoising steps runs from `step_idx = 0` to `T−1`, mapping to time `t = (T − step_idx) / T` (high noise → low noise), matching the standard reverse diffusion convention.

**Submodule PPO agents**: PPO training lives entirely in `Craftax_Baselines/`. The planner scripts only consume pre-trained checkpoints via `--ppo_checkpoint_path`. This separation ensures the diffusion code has no dependency on the PPO training hyperparameters.
