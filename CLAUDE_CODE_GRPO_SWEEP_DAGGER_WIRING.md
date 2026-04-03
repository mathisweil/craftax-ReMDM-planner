# Claude Code — GRPO Dead Code Sweep & DAgger Wiring Check

## Goal

We recently **deleted the GRPO online fine-tuning** and replaced it with **DAgger (Dataset Aggregation)**. This prompt asks you to do two things:

1. **Sweep the entire project** for any dead code, dead config, stale comments, or orphaned references left over from the deleted GRPO implementation.
2. **Verify that the new DAgger implementation is properly wired** through the full call chain — from CLI entry point to training loop to logging to checkpointing.

> **Scope**: scan everything **except** the `experiments/` directory (ignore it entirely for now). The `Craftax_Baselines/` submodule is read-only and out of scope for changes, but still grep it for any GRPO strings that might confuse someone.

---

## Part 1 — GRPO Dead Code & Dead Config Sweep

### 1.1 String search

Run a **case-insensitive** search across the entire project (excluding `experiments/`) for every one of these strings. For each hit, report the file, line number, and surrounding context, then classify it as: **dead** (must remove/replace), **stale comment** (must update), or **false positive** (legitimate use unrelated to GRPO).

**Primary search terms:**
- `grpo` / `GRPO`
- `group_size` (the GRPO-specific parameter `grpo_group_size`)
- `group_advantage` / `group advantage`
- `group-relative`
- `advantage_weight`
- `make_train_online` (if this was the GRPO entry point and has been renamed, check both old and new name exist; if the old name lingers as a dead alias, flag it)

**Secondary search terms (may have non-GRPO uses — verify each hit):**
- `advantage` (could be GRPO advantage computation remnants)
- `reward_weight` / `return_weight` (legitimate in offline, but check if any GRPO-specific weighting logic survives in online.py or common.py)

### 1.2 Config files

Open and fully read each of these files. For every parameter, check whether it is still consumed by the current code. Flag any that are orphaned:

- `configs/defaults.yaml`
- `configs/big_diffusion_online.yaml`
- `configs/A100_diffusion_online.yaml`

**Known GRPO config parameters to hunt** (from the old README):

| Parameter | Status expected |
|-----------|----------------|
| `grpo_group_size` | Should be **deleted** from all YAML files and the config dataclass/dict |
| `ppo_init_prob` | Should be **repurposed** for DAgger β₁ — check description is updated |
| `ppo_decay_rate` | Should be **repurposed** for DAgger βᵢ decay — check description is updated |
| `num_updates` | Likely still used by DAgger — confirm it's wired |
| `replan_every` | Likely still used — confirm |

For each surviving online-mode parameter, trace it from YAML → config parsing → `online.py` usage. If the parameter is defined in YAML but never read in code (or vice versa), flag it.

### 1.3 Config dataclass / argument parsing

Find wherever the config is defined as a Python dataclass, dict, or argparse namespace (likely in `main.py` or a config utility). Check:

- Is `grpo_group_size` still declared as a field? If so, **dead code**.
- Are there any DAgger-specific parameters (e.g., `buffer_capacity`, `dagger_epochs`, `beta_schedule`) that are used in `online.py` but **missing** from the config definition or YAML defaults?

### 1.4 Logging

Open `src/planners/logging.py` and search for:

- Any metric keys prefixed with `grpo/` — these are dead and should be replaced with `dagger/` equivalents.
- Any helper functions that compute GRPO-specific metrics (e.g., group advantage stats, group return variance).
- Check that `online.py` logs under a `dagger/` namespace (or whatever namespace was chosen), not `grpo/`.

Also check the README's **Implementation Notes** section which explicitly mentions `grpo/` as a W&B namespace.

### 1.5 Imports

In every `.py` file under `src/` and in `main.py`, check for:

- Imports from modules that no longer exist (e.g., if a `grpo.py` utility file was deleted but something still imports from it).
- Imports of specific functions/classes that were part of GRPO (e.g., `compute_group_advantage`, `sample_group_plans`).
- Unused imports in `online.py` that were needed for GRPO but not for DAgger.

### 1.6 README.md

The README has several GRPO references that must be updated. Confirm each of the following is either already fixed or still needs changing:

| Location in README | What to check |
|---|---|
| Pipeline diagram (Stage 3) | Should say "DAgger" not "GRPO" |
| "Stage 3 — Online GRPO fine-tuning" section header + body | Full rewrite needed |
| Hyperparameters table "Online GRPO training" | Header + parameter list |
| `grpo_group_size` row in that table | Should be removed |
| `ppo_init_prob` / `ppo_decay_rate` descriptions | Should reference DAgger β, not GRPO expert injection |
| Project structure tree comment `# --mode online: GRPO fine-tuning` | Should say DAgger |
| Implementation Notes: `grpo/` namespace mention | Should say `dagger/` |
| Any CLI examples in the online section | Verify they still work with current code |

---

## Part 2 — DAgger Wiring Verification

Trace the full DAgger execution path end-to-end and confirm every link in the chain is connected. **Do not review mathematical correctness here** — just confirm the plumbing works.

### 2.1 CLI → online mode dispatch

1. Open `main.py`.
2. Find where `--mode online` is handled.
3. Confirm it calls the correct entry function in `src/planners/online.py` (not a leftover GRPO function).
4. Check that all config values relevant to online mode are passed through.

### 2.2 Environment construction

1. In `online.py`, find where environments are created.
2. Confirm it uses `src/planners/env.py` (shared with other modes), not a bespoke environment setup.
3. Check wrapper stack is correct for online mode (LogWrapper → AutoResetEnvWrapper → BatchEnvWrapper).

### 2.3 Model initialisation

1. Confirm the diffusion model is initialised via `src/planners/model.py` (shared).
2. If warm-starting from an offline checkpoint, confirm the loading path works:
   - `--offline_checkpoint_path` is parsed.
   - Weights are loaded into the train state.
   - The optimiser state is handled (reset or continued — just check it's intentional, not accidental).

### 2.4 PPO expert loading

1. Confirm the PPO agent is loaded via `src/planners/ppo.py`.
2. Confirm `--ppo_checkpoint_path` is respected in online mode.
3. Check: is the PPO agent **required** for DAgger online mode? If so, is there a clear error message when it's missing?
4. If DAgger can run without a PPO expert (pure self-play), confirm that code path also works.

### 2.5 Training loop structure

Read through the main training loop in `online.py` and confirm this skeleton exists:

```
for iteration i = 1 to num_updates:
    1. Compute βᵢ (mixing probability)
    2. Rollout with mixed policy (β chance of expert, 1-β chance of learner)
    3. Collect expert labels for all visited states
    4. Add (obs, expert_action) pairs to aggregated buffer
    5. Sample minibatches from aggregated buffer
    6. Compute MDLM loss on minibatches → gradient update
    7. Log metrics
    8. (Optionally) run validation
    9. (Optionally) save checkpoint
```

For each step, confirm the function call chain:
- Step 1 → uses `ppo_init_prob` and `ppo_decay_rate` from config
- Step 2 → calls environment step + diffusion sampling + PPO inference
- Step 3 → calls PPO forward pass to get expert actions
- Step 4 → writes to a buffer data structure
- Step 5 → samples from that buffer
- Step 6 → calls the loss function from `src/diffusion/loss.py`
- Step 7 → calls logging from `src/planners/logging.py`
- Step 8 → uses inference/sampling for validation rollouts
- Step 9 → uses checkpoint utilities from `model.py` or orbax

### 2.6 Loss function wiring

Confirm that the training step in `online.py`:
- Calls the **same** MDLM loss function used by `offline.py` (from `src/diffusion/loss.py`).
- Passes the correct inputs: model params, observations, action targets (expert labels), diffusion time samples.
- Does **not** have a copy-pasted or modified loss function inlined in `online.py`.

### 2.7 Checkpointing wiring

1. Are periodic checkpoints saved during online training? (Check `checkpoint_interval` is respected.)
2. Is the final model saved when training completes? (Check `save_policy` is respected.)
3. Does the checkpoint contain the full train state (params + opt state) or just params?
4. Can `--mode inference --checkpoint_path <online_checkpoint>` load an online-trained model? (Same format as offline checkpoints.)

### 2.8 W&B logging wiring

1. Is `wandb.init()` called for online mode?
2. Are metrics logged every iteration?
3. Are the metric keys meaningful and consistent with other modes?
4. Is the run tagged or configured differently from offline runs so they're distinguishable in the W&B dashboard?

### 2.9 Validation wiring (if applicable)

1. Does online mode run periodic validation rollouts?
2. If so, does it use the same evaluation logic as `inference.py` (or a shared utility)?
3. Are `val_interval`, `val_diffusion_steps`, `val_replan_every`, `val_steps` respected?

---

## Execution — Audit, Plan, Apply

Complete all three stages **in order**. Do not skip ahead.

---

### Stage A — Audit (read-only)

Work through Part 1 and Part 2 above. Collect all findings and produce two tables:

**Table 1 — GRPO Remnants**

| File | Line(s) | Content | Classification | Action needed |
|------|---------|---------|---------------|---------------|

Where **Classification** is one of:
- `DEAD_CODE` — function, variable, or import that is never called
- `DEAD_CONFIG` — YAML key or dataclass field that is never read
- `STALE_COMMENT` — comment or docstring referencing GRPO
- `STALE_README` — README text referencing GRPO
- `STALE_LOGGING` — metric key or logging helper referencing GRPO
- `FALSE_POSITIVE` — legitimate code that happens to match a search term

**Table 2 — DAgger Wiring Checklist**

```
[ ] 2.1  CLI dispatch: main.py --mode online → online.py:<function_name>
[ ] 2.2  Env construction: uses shared env.py, correct wrapper stack
[ ] 2.3  Model init: uses shared model.py, warm-start loads correctly
[ ] 2.4  PPO expert: loaded via ppo.py, clear error if missing
[ ] 2.5  Training loop: all 9 steps present and connected
[ ] 2.6  Loss function: shared with offline, no inlined copy
[ ] 2.7  Checkpointing: periodic + final save, loadable by inference
[ ] 2.8  W&B logging: init, per-iteration logging, correct namespace
[ ] 2.9  Validation: periodic rollouts using shared eval logic
```

For any **FAIL**, explain what's broken and where.
For any **PARTIAL**, explain what works and what's missing.

**Present these two tables and pause for my confirmation before proceeding.**

---

### Stage B — Change Plan

Using the audit results, produce a concrete change plan split into two groups:

**Group 1 — GRPO Cleanup** (every row from Table 1 that isn't `FALSE_POSITIVE`):

For each item, state:
- **File** and line(s)
- **Action**: the exact edit — delete line, rename `X` → `Y`, replace comment with `Z`, remove YAML key, etc.

Organise by file so edits to the same file are batched together.

**Group 2 — Wiring Fixes** (every `FAIL` or `PARTIAL` from Table 2):

For each broken link, state:
- **File** and function/line
- **What's wrong** (one sentence)
- **Fix** (brief technical description — what to add, remove, or reconnect)

If a wiring issue overlaps with a GRPO remnant (e.g., logging namespace is both stale GRPO *and* a wiring failure), list it in Group 2 only to avoid duplicate edits.

**Present the plan and pause for my confirmation before proceeding.** I may ask you to skip certain changes or modify the approach.

---

### Stage C — Apply Changes

Once I confirm, apply the approved changes. Follow these rules:

1. **Group 1 first (GRPO cleanup), then Group 2 (wiring fixes).** Keeping them separate makes it easy to reason about what changed.

2. **Work file by file.** Before editing each file, state its path and briefly list what you're changing in it.

3. **Surgical edits only.** Do not rewrite entire files from scratch. Make targeted deletions, renames, and replacements.

4. **Specific guidance per file type:**

   - **YAML configs** (`defaults.yaml`, `big_diffusion_online.yaml`, `A100_diffusion_online.yaml`, and any other preset configs like `qmul_h200.yaml`, `ucl_4070.yaml`, `ali_gpu.yaml`):
     Delete dead GRPO keys (`grpo_group_size`), rename section headers, update comments. If DAgger parameters are missing from YAML but used in code, add them with sensible defaults.

   - **`main.py`** (CLI argument parsing):
     Remove `--grpo_group_size`. Add any missing DAgger CLI args. Update comments/help strings.

   - **`src/planners/logging.py`**:
     Rename `_GRPO_KEYS` → `_DAGGER_KEYS`. Change all `grpo/` metric prefixes to `dagger/`. Add any missing DAgger metric keys (e.g., `beta`). Update docstrings.

   - **`src/planners/online.py`**:
     Fix any wiring issues found in Group 2. Update stale comments. If the code reads config keys that don't exist in YAML (e.g., `DAGGER_BETA_INIT`), update it to read the correct YAML key names.

   - **`README.md`**:
     Update every GRPO reference. This includes: pipeline diagram, Stage 3 section header + body, hyperparameter table header + rows, project structure tree comment, implementation notes W&B namespace, CLI examples. Do **not** rewrite sections that are already correct.

   - **Other `.py` files** (`ppo.py`, `loss.py`, `common.py`, etc.):
     Update stale comments and docstrings only. Do not change logic unless a wiring fix requires it.

5. **Do not modify:**
   - `Craftax_Baselines/` — read-only submodule.
   - `experiments/` — out of scope.
   - `src/diffusion/loss.py` logic — only update comments if flagged as stale. Do not change the loss computation.
   - `src/diffusion/sampling.py` — do not touch.

6. **After all changes**, produce a final summary:

```
Files modified:
- <file>: <one-line description of what changed>
- ...

Files NOT modified (confirmed clean):
- <file>: no GRPO remnants, wiring intact
- ...

Remaining issues (if any):
- <description of anything that couldn't be fixed here>
```
