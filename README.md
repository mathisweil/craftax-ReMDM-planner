# ReMDM Planner — Discrete Diffusion Planning on Craftax

A JAX implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in the [Craftax](https://github.com/MichaelTMatthews/Craftax) environment. A bidirectional transformer learns to generate action plans by iteratively denoising masked token sequences, conditioned on the current environment observation.

---

## Description

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `T` denoising steps, producing a `plan_horizon`-length plan. The ReMDM framework extends standard Masked Discrete Language Modelling (MDLM) with remasking strategies that allow committed tokens to be re-predicted, improving plan coherence.

Training follows a four-stage pipeline:

```
[Stage 1]  Train PPO agent          Craftax_Baselines/ppo_rnn.py | ppo_rnd.py
               |
               v  checkpoint
[Stage 2]  Collect / train offline  main.py --mode collect
               |                    main.py --mode offline
               v  diffusion checkpoint
[Stage 3]  Online fine-tuning       main.py --mode online
               |
               v  fine-tuned checkpoint
[Stage 4]  Evaluate                 main.py --mode inference

[Optional] Train reward model       main.py --mode train_reward
```

---

## Installation

### 1. Create the conda environment

```bash
conda env create -f environment.yaml
conda activate craftax
```

### 2. Initialise the submodule

```bash
git submodule update --init --recursive
```

### 3. GPU support (optional)

By default JAX runs on CPU. For NVIDIA CUDA 12:

```bash
pip install -U "jax[cuda12]"
```

---

## Dependencies

| Package | Version | Role |
|---------|---------|------|
| `jax` | 0.9.1 | JIT compilation and functional arrays |
| `flax` | 0.12.2 | Neural network definitions |
| `optax` | 0.2.7 | Adam optimiser and gradient clipping |
| `craftax` | 1.5.0 | Procedurally-generated Minecraft-like environment |
| `gymnax` | 0.0.9 | Batched environment interface |
| `distrax` | 0.1.7 | Probability distributions |
| `orbax-checkpoint` | 0.5+ | Model checkpointing |
| `wandb` | 0.25.0 | Experiment logging |
| `pyyaml` | 6.0.3 | Config file parsing |

Full specification in `environment.yaml` and `requirements.txt`.

---

## Usage

All modes share the same entry point. Defaults are loaded from `configs/defaults.yaml`; any value can be overridden on the command line.

```bash
python main.py --mode <MODE> [--config PATH] [OVERRIDES...]
```

### Stage 1 — Train a PPO agent

PPO training is handled by the `Craftax_Baselines` submodule and produces the checkpoint consumed by all downstream stages.

```bash
cd Craftax_Baselines

# PPO with GRU hidden state (recommended)
python ppo_rnn.py \
    --env_name Craftax-Symbolic-v1 \
    --total_timesteps 500000000 \
    --save_policy --use_wandb

# PPO with Random Network Distillation
python ppo_rnd.py \
    --env_name Craftax-Symbolic-v1 \
    --total_timesteps 500000000 \
    --save_policy --use_wandb

cd ..
```

### Stage 2a — Collect trajectories to disk

Roll out the PPO checkpoint and save `(obs, actions, dones)` as a `.npz` file for reuse across multiple diffusion training runs.

```bash
python main.py --mode collect \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_data_path data/trajectories.npz \
    --collect_num_steps 1000000 \
    --collect_num_envs 64
```

The file has shape `[num_envs, num_iters, ...]`, preserving per-environment contiguity so the offline sampler can enforce episode boundaries.

### Stage 2b — Train offline (live PPO, no disk I/O)

Pass `--ppo_checkpoint_path` directly to `--mode offline` to skip saving trajectories. The PPO agent is rolled out live at each training step.

```bash
python main.py --mode offline \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --num_train_steps 100000 \
    --save_policy
```

### Stage 3 — Train offline from saved trajectories

```bash
python main.py --mode offline \
    --offline_data_path data/trajectories.npz \
    --num_train_steps 100000 \
    --save_policy
```

The sampler masks out windows that cross episode boundaries, so the model is never trained across episode resets.

### Stage 4 — Online GRPO fine-tuning

The diffusion model acts as its own policy: it generates plans, executes them, and trains on the resulting `(obs, plan)` pairs via group relative advantage weighting.

```bash
# From scratch
python main.py --mode online \
    --num_updates 1000 \
    --save_policy

# Warm-start from an offline checkpoint
python main.py --mode online \
    --offline_checkpoint_path /path/to/offline_checkpoint \
    --num_updates 1000 \
    --save_policy
```

### Stage 5 — Evaluate

```bash
python main.py --mode inference \
    --checkpoint_path /path/to/checkpoint \
    --eval_steps 10000 \
    --num_envs 32
```

Prints mean episode return, completed episodes, steps per second, and per-achievement unlock counts. Uses historical inpainting: the first `hist_len` plan positions are locked to observed history.

### Optional — Train a reward model

```bash
# MLP discriminator (default)
python main.py --mode train_reward \
    --offline_data_path data/trajectories.npz \
    --reward_model_type mlp

# Random Network Distillation
python main.py --mode train_reward \
    --offline_data_path data/trajectories.npz \
    --reward_model_type rnd \
    --reward_save_path checkpoints/rnd_reward.msgpack

# CNN-based Vision RND
python main.py --mode train_reward \
    --offline_data_path data/trajectories.npz \
    --reward_model_type vision_rnd
```

Pass the saved model to online training via `--reward_load_path`.

---

## Configuration

All hyperparameters are in `configs/defaults.yaml`. Override any value on the command line:

```bash
python main.py --mode offline --lr 1e-4 --plan_horizon 64 --batch_size 512
```

Point to a custom config file:

```bash
python main.py --mode online --config configs/my_experiment.yaml
```

Preset configs for larger runs are provided in `configs/`:

| File | Purpose |
|------|---------|
| `configs/defaults.yaml` | Base defaults for all modes |
| `configs/big_diffusion_offline.yaml` | Larger model for offline training |
| `configs/big_diffusion_online.yaml` | Larger model for online training |
| `configs/A100_diffusion_offline.yaml` | A100-tuned offline config |
| `configs/A100_diffusion_online.yaml` | A100-tuned online config |

### Key hyperparameters

**Diffusion model**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `plan_horizon` | 32 | Action plan length H |
| `diffusion_steps` | 15 | Denoising steps T at inference |
| `diffusion_schedule` | `cosine` | Noise schedule: `cosine` or `linear` |
| `remask_strategy` | `rescale` | Remasking strategy: `rescale`, `cap`, or `conf` |
| `train_sigma` | 0.0 | Per-token remasking during training (0 = vanilla MDLM) |
| `eta` | 0.5 | Remasking strength |
| `use_loop` | `true` | Three-phase loop remasking (Algorithm 3) |
| `t_on` / `t_off` | 0.7 / 0.3 | Time window for loop remasking |
| `temperature` | 0.5 | Softmax temperature for sampling |
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
| `dropout_rate` | 0.1 | Dropout (disabled at inference) |

**Offline training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_timesteps` | 1e8 | Environment steps for data collection |
| `num_envs` | 1024 | Parallel environments |
| `batch_size` | 256 | Minibatch size |
| `lr` | 3e-4 | Adam learning rate |
| `max_grad_norm` | 1.0 | Gradient clipping norm |
| `collect_temperature` | 1.0 | Softmax temperature on PPO logits during collection |

**Online GRPO training**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_updates` | 1000 | Outer update iterations |
| `replan_every` | 4 | Steps executed per plan before replanning |
| `grpo_group_size` | 4 | Plans sampled per state for group advantage |
| `ppo_init_prob` | 0.1 | Initial probability of injecting PPO expert actions |
| `ppo_decay_rate` | 0.99 | Exponential decay of PPO injection per update |

**Reward model**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `reward_model_type` | `mlp` | Architecture: `mlp`, `rnd`, or `vision_rnd` |
| `reward_epochs` | 10 | Training epochs |
| `reward_lr` | 1e-4 | Adam learning rate |
| `reward_save_path` | `checkpoints/reward_model.msgpack` | Output path |

**Logging**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_wandb` | `true` | Enable Weights & Biases logging |
| `wandb_project` | `remdm-craftax` | W&B project name |
| `save_policy` | `true` | Save checkpoint at end of training |
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

## Reward Models

Selected via `--reward_model_type`. During online GRPO training, the reward model is co-trained every 100 gradient steps. Observations from the first 80% of each episode are labelled negative; the last 20% are labelled positive.

| Model | Type | Description |
|-------|------|-------------|
| `mlp` | `DeterministicNeuralReward` | MLP discriminator with hidden dims 512 → 256 → 128 |
| `rnd` | `RNDReward` | Random Network Distillation: frozen target MLP vs. trained predictor MLP (256 → 128 → 64) |
| `vision_rnd` | `VisionRNDReward` | CNN-based RND: projects obs to 9x9x16 grid, applies two conv layers (64 → 128), then computes target/predictor MSE |

---

## Environment Wrappers

**From `Craftax_Baselines/wrappers.py`** (submodule):

| Wrapper | Purpose |
|---------|---------|
| `LogWrapper` | Tracks episode returns and lengths; adds stats to the info dict |
| `AutoResetEnvWrapper` | Automatically resets episodes on `done` |
| `BatchEnvWrapper` | Vmaps `reset` and `step` over `num_envs` environments |
| `OptimisticResetVecEnvWrapper` | Batched resets with reduced overhead; enable via `--use_optimistic_resets` |

**From `src/envs/wrappers.py`**:

| Wrapper | Purpose |
|---------|---------|
| `SequenceHistoryWrapper` | Maintains a sliding window of past observations and actions in the env state |
| `DiscreteTokenizationWrapper` | Quantizes continuous observations into discrete token indices |

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
│   ├── defaults.yaml              # Base hyperparameters (CLI-overridable)
│   ├── big_diffusion_offline.yaml
│   ├── big_diffusion_online.yaml
│   ├── A100_diffusion_offline.yaml
│   └── A100_diffusion_online.yaml
├── src/
│   ├── diffusion/
│   │   ├── schedules.py           # Linear and cosine noise schedules
│   │   ├── forward.py             # Forward masking process q(z_t | x_0)
│   │   ├── loss.py                # Continuous-time MDLM ELBO loss
│   │   └── sampling.py           # Reverse diffusion with ReMDM remasking
│   ├── models/
│   │   ├── denoiser.py            # DenoisingTransformer (obs encoder + transformer)
│   │   └── reward_models.py       # DeterministicNeuralReward, RNDReward, VisionRNDReward
│   ├── envs/
│   │   └── wrappers.py            # SequenceHistoryWrapper, DiscreteTokenizationWrapper
│   └── planners/
│       ├── collect.py             # --mode collect: PPO rollouts -> .npz
│       ├── train.py               # --mode offline: train from .npz or live PPO agent
│       ├── online.py              # --mode online: GRPO fine-tuning
│       ├── inference.py           # --mode inference: MPC evaluation with inpainting
│       ├── train_reward.py        # --mode train_reward: standalone reward model training
│       ├── data.py                # PPOAgent wrapper, environment construction, data loading
│       └── state.py               # Model init, checkpoint I/O, TrainState creation
├── main.py                        # CLI entry point
├── environment.yaml               # Conda environment specification
└── requirements.txt               # pip requirements
```

---

## Implementation Notes

**JAX functional purity**: training closures (`make_train`, `make_train_online`) are fully JIT-compatible. Environment construction and checkpoint I/O happen outside `jax.jit`.

**Episode-boundary masking**: the offline sampler pre-computes a validity mask over all `(env, time)` positions. A window at `(e, t)` is valid only if `dones[e, t:t+H-1]` are all `False`.

**Loss weight clipping**: the MDLM SUBS weight `-alpha'(t) / (1 - alpha_t)` is clipped to 1000 to prevent numerical instability when `alpha_t ~= 1`.

**Online training repeats**: `run_online` uses a sequential loop over `num_repeats` rather than `jax.vmap`, because `BatchEnvWrapper` already vmaps `step/reset` internally.

**Denoising step indexing**: the scan runs from `step_idx = 0` to `T-1`, mapping to time `t = (T - step_idx) / T` (high noise to low noise).

**Submodule PPO agents**: PPO training lives entirely in `Craftax_Baselines/`. Planner scripts only consume pre-trained checkpoints via `--ppo_checkpoint_path`.
