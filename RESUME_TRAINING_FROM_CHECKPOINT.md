# Task: Add Resume-from-Checkpoint for Offline and Online Training

## Overview

Add the ability to **resume training from a previously completed checkpoint** for both `--mode offline` and `--mode online`. This is NOT mid-run checkpointing — the idea is to take the final checkpoint produced by a fully completed training run and use it as the starting point for a new training run that continues where the previous one left off. The training loops must remain **fully JIT-compiled** exactly as they are now. All checkpoint I/O must stay outside `jax.jit`. W&B logging must be resumable so metrics appear as a single continuous run.

---

## Phase 1 — Analyse the Codebase

Before writing any code, read and understand these files in order. Build a mental model of the full data flow, then output a summary of your plan before making changes.

### Files to read (in this order)

1. **`main.py`** — CLI entry point, mode dispatch, config loading
2. **`src/planners/offline.py`** — `make_train()` closure, the `jax.lax.scan` training loop, runner state shape, how `train_state` (params + opt_state) is initialised
3. **`src/planners/online.py`** — `make_train_dagger()` closure, the DAgger outer loop, how the replay buffer and beta schedule work, how `train_state` is initialised
4. **`src/planners/model.py`** — `create_train_state()`, `save_checkpoint()`, `load_checkpoint()`, what exactly is saved/restored (params, opt_state, step)
5. **`src/planners/logging.py`** — W&B initialisation, metric namespaces, how `wandb.init()` is called, whether a run ID is stored
6. **`src/planners/common.py`** — shared utilities
7. **`configs/defaults.yaml`** — all current config keys, especially checkpointing and logging sections

### What to figure out

- What is the exact shape of the `runner_state` tuple that `jax.lax.scan` carries? Which elements need restoring vs reinitialising?
- How is the optimizer state (Adam moments) stored in the checkpoint? Can it be restored into a fresh `TrainState`?
- How is the learning rate schedule constructed? It uses cosine decay over a fixed number of total gradient steps — resuming must account for steps already completed so the schedule is correct for the remaining steps.
- For online mode: what is the state of the DAgger replay buffer, beta schedule, and iteration counter? Which of these need to be saved/restored?
- How is `wandb.init()` called? Does it currently accept a `run_id`? What needs to change to support `wandb.init(resume="must", id=<run_id>)`?
- Are there any `jax.random.split` chains seeded from the initial key that would need to be advanced to the correct position on resume?

### Output your plan

After reading, output a structured plan with:
- Every file you will modify and why
- Every new config key you will add
- The exact CLI interface for resuming (flag names, example commands)
- Any edge cases or gotchas (LR schedule, RNG state, buffer state)
- Confirmation that the `jax.lax.scan` / `jax.jit` boundaries are NOT moved

---

## Phase 2 — Implement the Changes

### 2.1 New CLI / Config Parameters

Add these configuration options (in `configs/defaults.yaml` and the argument parser):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `resume_checkpoint_path` | `str \| null` | `null` | Path to a completed training checkpoint to resume from. Accepts `wandb:` artifact refs. When set, training continues from this checkpoint's params + optimizer state. |
| `resume_wandb_run_id` | `str \| null` | `null` | W&B run ID to resume logging into. When set, `wandb.init` uses `resume="must"` with this ID so metrics append to the existing run. |
| `resume_step` | `int \| null` | `null` | The update step the checkpoint was saved at. Required for resume so the LR schedule and logging step offset are correct. If null and `resume_checkpoint_path` is set, attempt to read it from the checkpoint metadata. |

### 2.2 Checkpoint Metadata

When saving checkpoints (in `src/planners/model.py`), also persist a small JSON sidecar file alongside the orbax checkpoint containing:

```json
{
  "mode": "offline | online",
  "update_step": 1525,
  "total_gradient_steps_completed": 48800,
  "wandb_run_id": "abc123xyz",
  "config_snapshot": { ... }
}
```

This metadata file enables resume without needing the user to manually specify `--resume_step` or `--resume_wandb_run_id` — they can be auto-read from the checkpoint directory. The user-provided CLI flags should override the metadata if both are present.

### 2.3 Offline Training Resume (`src/planners/offline.py`)

The offline training loop uses `jax.lax.scan` over `num_updates` steps. On resume:

1. **Outside `jax.jit`**: load the checkpoint (params + optimizer state) via the existing `load_checkpoint` path. Construct the `TrainState` with the restored params and opt_state.
2. **LR schedule**: the cosine schedule is defined over `total_gradient_steps = num_updates * update_epochs * num_minibatches`. On resume, you need to either:
   - Recompute the schedule for the *remaining* steps (from `resume_step` to `num_updates`), starting from wherever the cosine was at that point, OR
   - Keep the full original schedule but offset the step counter so it indexes into the correct position.
   The second approach is cleaner — construct the full schedule as before, but initialise the `TrainState.step` to `resume_step * update_epochs * num_minibatches` so `optax` picks up the LR at the right point. **Verify this is how the current optax schedule is consumed.**
3. **`jax.lax.scan` range**: the scan currently runs for `num_updates` iterations. On resume, it should run for `num_updates - resume_step` iterations. This is controlled by the length of the scan, which is set *outside* jit. Adjust this without touching anything inside the jitted function.
4. **RNG state**: the PRNG key is split at each step inside the scan. On resume, initialise a fresh key (from the same or a new seed) — the stochasticity of future steps does not need to reproduce the original run, it just needs to be valid.
5. **Logging step offset**: metrics logged to W&B must use step numbers that continue from where the previous run stopped, not restart from 0. Pass the `resume_step` offset into the logging callback or metric computation so `wandb.log(step=resume_step + current_step)` is correct.
6. **Environment state**: envs are freshly created on each run — no need to restore env state. This is fine.

**Critical constraint**: do NOT refactor `make_train()` or move any code across the `jax.jit` boundary. The only changes should be:
- How `TrainState` is initialised (restored vs fresh)
- The scan length (num_updates vs num_updates - resume_step)
- The step offset for logging
- Any args passed into the closure from outside

### 2.4 Online Training Resume (`src/planners/online.py`)

The online DAgger loop is a Python-level `for` loop over `num_updates` iterations. On resume:

1. **Params + optimizer**: same as offline — load checkpoint, construct `TrainState` with restored state.
2. **LR schedule**: same approach as offline — set `TrainState.step` to the correct offset.
3. **DAgger beta schedule**: `beta_i = beta_init * beta_decay^i`. On resume from iteration `i`, the beta must start at `beta_init * beta_decay^resume_step`. This is trivially computed — just adjust the loop's starting index or the beta computation.
4. **Replay buffer**: the DAgger replay buffer accumulates data across iterations. On resume, the buffer starts empty and must be rebuilt. This is acceptable — the first few iterations after resume will have a smaller buffer, but it refills quickly. Document this trade-off. Do NOT try to serialize/restore the buffer (it would break jit purity and massively complicate things).
5. **Loop range**: change `range(num_updates)` to `range(resume_step, num_updates)`.
6. **Logging**: same step offset approach as offline.

### 2.5 W&B Resume (`src/planners/logging.py`)

Modify the W&B initialisation function to:

1. Accept optional `resume_run_id` parameter.
2. When `resume_run_id` is provided:
   - Call `wandb.init(id=resume_run_id, resume="must", ...)` so it attaches to the existing run.
   - All other config (project, entity, tags) should remain the same.
3. When saving checkpoints, persist the current `wandb.run.id` into the metadata sidecar (section 2.2) so that future resumes can auto-detect it.
4. When a new run starts (no resume), `wandb.init` works as it does now, but the run ID is captured and saved in checkpoint metadata.

### 2.6 Main Entry Point (`main.py`)

In `main.py`, add the resume logic:

1. If `resume_checkpoint_path` is set:
   - Resolve the path (including `wandb:` artifact download, same as existing checkpoint resolution).
   - Read the metadata sidecar if it exists; use it to populate `resume_step` and `resume_wandb_run_id` if they weren't explicitly provided on the CLI.
   - Validate: if `resume_step` is still null after metadata lookup, error out with a clear message telling the user to provide `--resume_step`.
   - Pass the resume info down to the training function.
2. Validate that `resume_step < num_updates` (offline) or `resume_step < num_updates` (online), otherwise error out.

### 2.7 Update `configs/defaults.yaml`

Add the new keys under a `# Resume` section in the config with null defaults and clear comments.

### 2.8 Update `README.md`

Add a new section **"Resuming a Training Run"** after the existing training sections. Include:

- Explanation of when/why you'd resume (preempted job, extending training budget).
- Example commands for offline resume:
  ```bash
  # Auto-detect step and wandb run ID from checkpoint metadata
  python main.py --mode offline \
      --ppo_checkpoint_path /path/to/ppo_checkpoint \
      --resume_checkpoint_path /path/to/completed_offline_checkpoint \
      --total_timesteps 200000000 \
      --save_policy

  # Explicit step and wandb run ID override
  python main.py --mode offline \
      --ppo_checkpoint_path /path/to/ppo_checkpoint \
      --resume_checkpoint_path /path/to/completed_offline_checkpoint \
      --resume_step 1525 \
      --resume_wandb_run_id abc123xyz \
      --total_timesteps 200000000 \
      --save_policy
  
  # Resume from a W&B artifact
  python main.py --mode offline \
      --ppo_checkpoint_path /path/to/ppo_checkpoint \
      --resume_checkpoint_path wandb:my-team/remdm-craftax/policy:latest \
      --total_timesteps 200000000 \
      --save_policy
  ```
- Example commands for online resume:
  ```bash
  python main.py --mode online \
      --ppo_checkpoint_path /path/to/ppo_checkpoint \
      --resume_checkpoint_path /path/to/completed_online_checkpoint \
      --num_updates 2000 \
      --save_policy
  ```
- A note that the DAgger replay buffer is not persisted and will be rebuilt from scratch on resume.
- A note that JIT compilation is fully preserved — resume only affects initialisation outside jit.
- Add the new parameters to the configuration table in the existing Checkpointing section.

---

## Constraints — Read Carefully

1. **DO NOT break JIT compilation.** The `make_train()` and `make_train_dagger()` closures and everything inside `jax.jit` must remain exactly as functional and jittable as they are now. All resume logic (loading checkpoints, reading metadata, adjusting schedule parameters) happens OUTSIDE jit. The only thing that changes inside jit is the initial state it receives and the scan length.
2. **DO NOT add mid-run checkpointing.** This task is specifically about restarting from a checkpoint produced at the end of a complete run. The existing `checkpoint_interval` / `max_checkpoints` periodic saving is separate (and for online mode only). Do not conflate the two.
3. **DO NOT serialize the DAgger replay buffer.** Accept the trade-off of rebuilding it.
4. **DO NOT change the checkpoint format** for the core orbax checkpoint (params + opt_state). Only ADD the metadata sidecar as a separate file. Existing checkpoints without the sidecar must still load correctly (with the user manually providing `--resume_step`).
5. **Backward compatibility**: all new config keys default to null/disabled. A run without any `--resume_*` flags must behave identically to the current code.
6. **Test your changes mentally** by tracing through a resume scenario: load checkpoint → construct TrainState with restored params/opt_state and correct step count → pass to jitted train function → scan runs for remaining steps → LR schedule is correct → metrics log at correct step numbers → final checkpoint saves normally.

---

## Checklist

Before marking this done, confirm:

- [ ] `configs/defaults.yaml` has the three new keys with null defaults
- [ ] `main.py` handles `resume_checkpoint_path` resolution (including `wandb:` artifacts) and metadata reading
- [ ] `src/planners/model.py` saves metadata sidecar alongside checkpoints (mode, update_step, total_gradient_steps, wandb_run_id)
- [ ] `src/planners/model.py` has a function to read the metadata sidecar
- [ ] `src/planners/offline.py` accepts resume state and adjusts scan length + TrainState initialisation
- [ ] `src/planners/online.py` accepts resume state and adjusts loop range + beta + TrainState initialisation
- [ ] `src/planners/logging.py` supports `wandb.init(resume="must", id=...)` when resuming
- [ ] LR cosine schedule is correct on resume (verified by checking the step counter in TrainState)
- [ ] W&B step numbers are continuous across original and resumed runs
- [ ] Existing checkpoints without metadata sidecar still load (backward compat)
- [ ] `README.md` has a new "Resuming a Training Run" section with examples
- [ ] All new config keys are documented in the README config table
- [ ] No code was moved across jit boundaries
- [ ] Running without any `--resume_*` flags behaves identically to before
