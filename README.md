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
| `src/planners/collect.py` | `--mode collect`: roll out a PPO agent and save trajectories to disk |
| `src/planners/offline.py` | `--mode offline`: train the diffusion model from `.npz` trajectories or a live PPO agent |
| `src/planners/online.py` | `--mode online`: GRPO fine-tuning with self-rollout and reward model co-training |
| `src/planners/inference.py` | `--mode inference`: evaluation with historical inpainting |
| `src/planners/train_reward.py` | `--mode train_reward`: standalone neural reward model training |
| `src/planners/common.py` | Shared gradient step, advantage weighting, action statistics |
| `src/planners/utils.py` | Model/environment construction, checkpoint I/O, PPO agent wrapper |
| `src/models/remdm.py` | Diffusion core: noise schedules, MDLM SUBS loss, remasking strategies, reverse sampling |
| `src/models/denoiser.py` | `DenoisingTransformer`: observation MLP encoder + sinusoidal time embedding + bidirectional transformer |
| `src/models/reward_models.py` | Neural reward models: `DeterministicNeuralReward` (MLP), `RNDReward`, `VisionRNDReward` |
| `src/envs/wrappers.py` | Custom environment wrappers: `PlannerWrapper`, `SequenceHistoryWrapper`, `OfflineTrajectoryWrapper`, `DiscreteTokenizationWrapper` |
| `Craftax_Baselines/` | Git submodule — PPO agents (`ppo_rnn.py`, `ppo_rnd.py`) and standard Gymnax wrappers (`LogWrapper`, `BatchEnvWrapper`, etc.) |
| `configs/defaults.yaml` | All hyperparameters with documentation |

---

## Installation

```bash
conda env create -f environment.yaml
conda activate craftax
```

After creating the environment, initialise the `Craftax_Baselines` submodule:

```bash
git submodule update --init --recursive
```

Core dependencies: `jax`, `flax`, `optax`, `orbax-checkpoint`, `craftax`, `gymnax`, `distrax`, `chex`, `wandb`.

## GPU-Enabled JAX
By default, both of the above methods will install JAX on the CPU.  If you want to run JAX on a GPU/TPU, you'll need to install the correct wheel for your system from <a href="https://github.com/google/jax?tab=readme-ov-file#installation">JAX</a>.
For NVIDIA GPU the command is:
```
pip install -U "jax[cuda12]"
```

---

## Testing

The test suite uses **pytest** and covers core components: diffusion logic, denoiser architecture, environment wrappers, planner utilities, and integration tests.

### Run all tests

```bash
pytest src/tests/ -v
```

### Run specific test files

```bash
# Test diffusion schedules, forward process, loss, remasking, and sampling
pytest src/tests/test_remdm.py -v

# Test denoiser transformer architecture and forward pass
pytest src/tests/test_denoiser.py -v

# Test environment wrappers (PlannerWrapper, SequenceHistoryWrapper, etc.)
pytest src/tests/test_wrappers.py -v

# Test planner utility functions (sampling, masking, scheduling)
pytest src/tests/test_planner_utils.py -v

# Test collect/offline/online/inference pipeline integration
pytest src/tests/test_planners.py -v

# Test main entry point argument parsing and mode routing
pytest src/tests/test_main.py -v
```

### Run tests with coverage

```bash
pytest src/tests/ --cov=src --cov-report=html
```

### Run a single test function

```bash
pytest src/tests/test_remdm.py::TestCosineSchedule::test_boundary_t0 -v
```

### Test fixtures

All tests share fixtures defined in `src/tests/conftest.py`:
- `rng`: Deterministic JAX PRNG key (seed 42)
- `small_config`: Minimal config dict with fast hyperparameters for testing
- `dummy_env`: Mock Gymnax environment for wrapper testing
- `dummy_model_apply`: Trivial model function for integration tests
- `dummy_params`: Empty parameter dict

---

## Workflow

The pipeline has five sequential stages. Each stage can be run independently.

```
[Stage 1]  Train PPO agent           Craftax_Baselines/ppo_rnn.py / ppo_rnd.py
               │
               ▼ checkpoint
[Stage 2]  Collect / train offline   main.py --mode collect
               │                     main.py --mode offline
               ▼ diffusion checkpoint
[Stage 3]  Online fine-tuning        main.py --mode online
               │
               ▼ fine-tuned checkpoint
[Stage 4]  Evaluate                  main.py --mode inference

[Optional] Train reward model        main.py --mode train_reward
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
python main.py --mode collect \
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
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --num_train_steps 100000 \
    --save_policy
```

This is the recommended approach when you want to iterate quickly on diffusion model hyperparameters without re-running data collection.

---

### Stage 3 — Train offline from saved trajectories

```bash
python main.py --mode offline \
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
python main.py --mode online \
    --num_updates 1000 \
    --save_policy

# Warm-start from an offline checkpoint
python main.py --mode online \
    --offline_checkpoint_path /path/to/offline_checkpoint \
    --num_updates 1000 \
    --save_policy
```

---

### Stage 5 — Evaluate

```bash
python main.py --mode inference \
    --checkpoint_path /path/to/checkpoint \
    --eval_steps 1000 \
    --num_envs 32
```

Runs 10,000 environment steps (configurable via `--eval_steps`) and prints mean episode return, number of completed episodes, steps per second, and per-achievement unlock counts. Uses historical inpainting: the first `hist_len` positions in each plan are locked to observed history.

---

### Stage 6 (Optional) — Train a neural reward model

Train a standalone reward model on offline trajectories. Useful for initialising the reward model used during online GRPO fine-tuning.

```bash
# MLP discriminator (fast, default)
python main.py --mode train_reward \
    --offline_data_path data/trajectories.npz \
    --reward_model_type mlp \
    --reward_epochs 10

# Random Network Distillation (intrinsic curiosity)
python main.py --mode train_reward \
    --offline_data_path data/trajectories.npz \
    --reward_model_type rnd \
    --reward_save_path checkpoints/rnd_reward.msgpack

# Vision RND with CNN encoder (works with any obs shape)
python main.py --mode train_reward \
    --offline_data_path data/trajectories.npz \
    --reward_model_type vision_rnd
```

The reward model is then passed to online training via `--reward_load_path`.

---

## Configuration

All hyperparameters live in `configs/defaults.yaml`. Any value can be overridden from the command line.

```bash
# Example: override learning rate and plan horizon
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo \
    --lr 1e-4 \
    --plan_horizon 64
```

You can also point to a custom config file:

```bash
python main.py --mode online --config configs/my_experiment.yaml
```

### Key hyperparameters

**Diffusion model**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plan_horizon` | 32 | Length of the action plan H |
| `diffusion_steps` | 15 | Number of denoising steps T at inference |
| `diffusion_schedule` | `cosine` | Noise schedule: `cosine` or `linear` |
| `remask_strategy` | `rescale` | Remasking strategy: `rescale`, `cap`, `conf`, `loop` |
| `train_sigma` | 0.0 | Per-token remasking probability during training (0 = vanilla MDLM) |
| `eta` | 0.5 | Remasking strength |
| `t_on` / `t_off` | 0.7 / 0.3 | Time window for the `loop` remasking strategy |
| `top_p` | 4 | Nucleus sampling: keep top-k logits at each denoising step |
| `use_loop` | true | Enable three-phase loop remasking during reverse diffusion |

**Transformer architecture**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Transformer hidden dimension |
| `n_heads` | 4 | Number of attention heads |
| `n_layers` | 4 | Number of transformer blocks |
| `d_ff` | 512 | FFN inner dimension |
| `obs_encoder_layers` | 2 | MLP layers in the observation encoder |
| `obs_encoder_width` | 512 | Observation encoder hidden width |
| `dropout_rate` | 0.1 | Dropout rate (disabled at inference) |

**Offline training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_train_steps` | 1e9 | Gradient steps for offline training |
| `batch_size` | 256 | Minibatch size |
| `lr` | 3e-4 | Adam learning rate |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `collect_temperature` | 2.0 | Softmax temperature applied to PPO logits during collection |

**Online training (GRPO)**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_envs` | 32 | Parallel environments |
| `num_steps` | 1200 | Environment steps per update (must be divisible by `replan_every`) |
| `replan_every` | 4 | Steps executed per plan before replanning |
| `num_updates` | 1000 | Number of outer update iterations |
| `grpo_group_size` | 4 | Plans sampled per state for group relative advantage |
| `ppo_init_prob` | 0.1 | Initial probability of injecting PPO expert actions |
| `ppo_decay_rate` | 0.99 | Exponential decay of PPO injection probability per update |
| `use_optimistic_resets` | false | Use `OptimisticResetVecEnvWrapper` for faster episode resets |

**Reward model training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_model_type` | `mlp` | Architecture: `mlp`, `rnd`, or `vision_rnd` |
| `reward_epochs` | 10 | Training epochs over offline data |
| `reward_lr` | 1e-4 | Adam learning rate for the reward model |
| `reward_save_path` | `checkpoints/reward_model.msgpack` | Where to save the final reward weights |
| `reward_load_path` | null | Pre-trained reward weights to warm-start from |

**Logging**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_wandb` | true | Enable Weights & Biases logging |
| `wandb_project` | `remdm-craftax` | W&B project name |
| `save_policy` | true | Save model checkpoint at end of training |
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

## Reward Models

Three neural reward model architectures are available, selected via `--reward_model_type`:

| Model | Type | Description |
|-------|------|-------------|
| `mlp` | `DeterministicNeuralReward` | MLP discriminator (hidden dims 512 → 256 → 128). Trained with a discriminator loss: early trajectory frames are labelled −1 (negative), late frames +1 (positive). |
| `rnd` | `RNDReward` | Random Network Distillation. Intrinsic reward = MSE between a frozen target MLP and a trained predictor MLP (hidden dims 256 → 128 → 64). |
| `vision_rnd` | `VisionRNDReward` | CNN-based RND. Projects the flat observation vector into a 9×9×16 grid, applies two conv layers (Conv 64 → Conv 128), then computes target/predictor MSE. Works with any observation shape. |

During online GRPO training, the reward model is co-trained every 100 gradient steps alongside the diffusion model. Observations from the first 80% of each episode are used as negatives; the last 20% as positives.

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
craftax-ReMDM-planner/
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
│   └── defaults.yaml              # All hyperparameters (CLI-overridable)
├── src/
│   ├── planners/
│   │   ├── __init__.py            # run_collect / run_offline / run_online / run_inference entry points
│   │   ├── collect.py             # --mode collect: PPO rollouts → .npz
│   │   ├── offline.py             # --mode offline: train from .npz or live PPO agent
│   │   ├── online.py              # --mode online: GRPO fine-tuning with reward co-training
│   │   ├── inference.py           # --mode inference: evaluation with inpainting
│   │   ├── train_reward.py        # --mode train_reward: standalone reward model training
│   │   ├── common.py              # Shared gradient step, advantage weighting, action stats
│   │   └── utils.py               # Model/env construction, checkpoint I/O, PPO wrapper
│   ├── models/
│   │   ├── remdm.py               # Diffusion core: schedules, MDLM loss, remasking, sampling
│   │   ├── denoiser.py            # DenoisingTransformer architecture
│   │   └── reward_models.py       # DeterministicNeuralReward, RNDReward, VisionRNDReward
│   ├── envs/
│   │   └── wrappers.py            # PlannerWrapper, SequenceHistoryWrapper, etc.
│   └── tests/                     # pytest suite (test_remdm, test_denoiser, test_wrappers, ...)
├── main.py                        # CLI entry point (--mode collect|offline|online|inference|train_reward)
├── environment.yaml               # Conda environment spec
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
