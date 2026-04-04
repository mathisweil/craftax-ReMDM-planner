# Claude Code — Diagnostic: DAgger Checkpoint Compatibility with rl_finetuning Experiments

## Goal

We need to run the `experiments/rl_finetuning/` ablation suite using a **DAgger-trained diffusion model** (produced by `main.py --mode online`) as the starting checkpoint, instead of only an offline-trained one. Run a full diagnostic to determine whether this works today, and if not, exactly what breaks and how to fix it.

> **Scope**: `experiments/rl_finetuning/` and the `src/` modules it imports from. Do not modify `Craftax_Baselines/`. Report findings first — do not apply changes until approved.

---

## Phase 1 — Checkpoint Format Analysis

The core question: **is a DAgger checkpoint structurally identical to an offline checkpoint?**

### 1.1 What gets saved

Read the checkpoint saving code in both training modes and document what each writes to disk:

- `src/planners/online.py` — what does DAgger save? (params only? params + opt state? extra state like buffer contents, beta, step count?)
- `src/planners/offline.py` — what does offline save?
- `src/planners/model.py` — is there a shared `save_checkpoint` / `load_checkpoint` utility both modes use?

Produce a side-by-side comparison:

```
Offline checkpoint contains:    DAgger checkpoint contains:
─────────────────────────────   ─────────────────────────────
params: {...}                   params: {...}
opt_state: {...}                opt_state: {...}
step: int                      step: int
???                             ???
```

### 1.2 Param tree structure

The model architecture (`DenoisingTransformer`) is the same regardless of training mode. But verify:

- Are the param pytree keys identical? (Same layer names, same shapes.)
- Did DAgger training add or remove any model components (e.g., an extra output head, a value function, a buffer encoder)?
- Could the optimizer state differ in structure? (e.g., different optimizer, different number of Adam slots if DAgger uses a different LR schedule setup.)

### 1.3 What gets loaded

Read the checkpoint **loading** code in the experiment suite:

- `experiments/rl_finetuning/run_ablations.py` — how does it parse `--offline_checkpoint_path`?
- `experiments/rl_finetuning/ablations/training.py` — how does `make_run_ablation()` load the pretrained model?
- Does it call `src/planners/model.py` loading utilities, or does it do its own thing?

**Critical questions:**
1. Does the loader use `--offline_checkpoint_path` exclusively, or is there a generic `--checkpoint_path`? If it's `--offline_checkpoint_path`, passing a DAgger checkpoint still works if the format is identical — but the name is misleading.
2. Does the loader extract only `params` from the checkpoint (and ignore opt state)? Or does it restore the full train state? Ablation training should create a **fresh** optimizer, so it should only need params.
3. Does the loader validate anything about the checkpoint (e.g., check for a `mode` key, assert specific keys exist)? If a DAgger checkpoint has extra keys (like buffer state), does the loader crash or silently ignore them?

### 1.4 Inference compatibility

Also check: can `main.py --mode inference --checkpoint_path <dagger_ckpt>` load a DAgger checkpoint? This uses the same loading path as experiments would, so if inference works, experiments likely work too.

---

## Phase 2 — Experiment Training Loop Assumptions

Even if the checkpoint loads fine, the experiment's training loop might make assumptions about the pretrained model's behaviour that don't hold for a DAgger-trained model.

### 2.1 Pretrained score evaluation

The experiment evaluates the pretrained model before any fine-tuning (`pretrained_score` in `results.json`). Check:
- Does this use `src/diffusion/sampling.py` for plan generation? (It should — same model, same sampling.)
- Are there any sampling parameters (temperature, top_p, diffusion_steps) that are set differently for "offline-pretrained" vs "online-pretrained" models? There shouldn't be, but check.

### 2.2 KL divergence baseline

Several ablations compute KL drift from the pretrained model (e.g., `kl_penalty`, `ewc`, `trust_region_kl`, and the `repr_drift_kl` diagnostic metric). Check:
- How is the pretrained model's distribution captured? Is it a frozen copy of the initial params?
- Does this work regardless of how those initial params were produced (offline vs DAgger)?

### 2.3 EWC Fisher diagonal

The `ewc` ablation computes a Fisher Information matrix from the pretrained model. Check:
- Does the Fisher computation use rollout data from the pretrained model, or from the PPO agent, or from the environment?
- If it generates rollout data using the pretrained diffusion model, does it work correctly with a DAgger-trained model? (It should — sampling is architecture-dependent, not training-procedure-dependent.)

### 2.4 Mixed replay

The `mixed_replay` ablation mixes offline PPO data into online batches. Check:
- Does it load PPO rollout data from a separate `.npz` file, or does it use the `--ppo_checkpoint_path` to generate data on the fly?
- Is this compatible with a DAgger starting point? (DAgger already trained on PPO expert data, so mixing in more PPO data is fine semantically — just check the mechanics.)

### 2.5 Loss function `weights` parameter

The `baseline_rl` ablation uses return-weighted ELBO. Several ablations modify the weighting (advantage clipping, normalisation, filtering). Check:
- Does `src/diffusion/loss.py` accept a `weights` or `advantages` parameter?
- Has this parameter been renamed or removed during the GRPO → DAgger migration?
- If the loss signature changed, do the experiment's loss factories in `ablations/losses.py` still match?

### 2.6 Online rollout loop

The experiment fine-tunes the model via online RL rollouts (the model generates plans, executes them, collects returns). Check:
- Does the rollout loop in `ablations/training.py` use the same environment setup as `src/planners/online.py`?
- Does it use the same diffusion sampling for plan generation?
- Are there any references to GRPO-specific rollout mechanics (e.g., group sampling, multiple plans per state for advantage computation)? These would be dead code now.

---

## Phase 3 — CLI & Config Path

### 3.1 Accepting a DAgger checkpoint

Currently, the experiment CLI likely has `--offline_checkpoint_path`. Check:
- Can a user pass a DAgger checkpoint to this flag and have it work? (If the format is identical, yes — but the name is confusing.)
- Should we add `--checkpoint_path` as a generic alias, or rename the existing flag?
- Does the experiment also accept `--online_checkpoint_path`? If not, should it?

### 3.2 Config interaction

When the experiment loads `configs/defaults.yaml` as a base config:
- Does it read any online-mode parameters (e.g., `num_updates`, `dagger_beta_init`) that might cause confusion?
- Are there any config keys that behave differently depending on whether the pretrained model was trained offline vs online?

### 3.3 W&B artifact paths

The main project supports `wandb:` prefixed checkpoint paths. Check:
- Does the experiment suite support this too?
- Can a user pass `--offline_checkpoint_path wandb:team/project/dagger-policy:latest` and have it resolve?

---

## Phase 4 — End-to-End Trace

Do a full mental dry-run of this command:

```bash
python experiments/rl_finetuning/run_ablations.py \
    --ablations baseline_rl kl_penalty ewc \
    --offline_checkpoint_path /path/to/dagger_checkpoint \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --fast
```

Trace every step from CLI parsing through to the first training iteration of `baseline_rl`:

1. CLI args parsed → config dict built
2. DAgger checkpoint loaded → params extracted
3. PPO checkpoint loaded → expert agent created
4. Environment created
5. Pretrained model evaluated → `pretrained_score` computed
6. `baseline_rl` ablation started → fresh optimizer created with pretrained params
7. First rollout: model generates plans via diffusion sampling, executes in env, collects returns
8. First gradient step: return-weighted ELBO computed, grads applied
9. Diagnostics collected: grad norms, KL drift from pretrained, etc.

At each step, note whether it would **work**, **crash**, or **produce wrong results** with a DAgger checkpoint.

---

## Output

Produce a diagnostic report with:

### Table 1 — Checkpoint Compatibility

| Aspect | Status | Detail |
|--------|--------|--------|
| Param tree structure | MATCH / MISMATCH | ... |
| Optimizer state | MATCH / MISMATCH / NOT LOADED | ... |
| Extra keys in DAgger ckpt | IGNORED / CAUSES CRASH | ... |
| Loading path | WORKS / NEEDS FIX | ... |

### Table 2 — Training Loop Compatibility

| Component | Status | Detail |
|-----------|--------|--------|
| Pretrained eval | WORKS / BROKEN | ... |
| KL baseline | WORKS / BROKEN | ... |
| EWC Fisher | WORKS / BROKEN | ... |
| Mixed replay | WORKS / BROKEN | ... |
| Loss function API | WORKS / BROKEN | ... |
| Rollout loop | WORKS / BROKEN | ... |

### Table 3 — CLI & Config

| Item | Status | Fix needed |
|------|--------|-----------|
| Checkpoint flag naming | OK / CONFUSING | ... |
| Config interaction | OK / PROBLEMATIC | ... |
| W&B artifact support | YES / NO | ... |

### Summary

One paragraph: can we run the experiments on a DAgger checkpoint **today**, and if not, what's the minimum set of changes needed?

### Recommended Changes

If changes are needed, list them in priority order:
1. **Must fix** — things that crash or produce wrong results
2. **Should fix** — things that are confusing but technically work
3. **Nice to have** — UX improvements (e.g., better flag names)

**Present the report and pause. Do not apply changes until I confirm.**
