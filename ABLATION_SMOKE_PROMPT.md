# Task: smoke-test and repair every ablation in the Craftax Classic suite

## Where you are

`/workspace/craftax-ReMDM-planner` on the GPU box, a clone of `mathisweil/craftax-ReMDM-planner`. A uv project: `uv sync`, run with `uv run`.

**This box is an RTX 4070 Ti, 16 GB VRAM. The suite it is testing is configured for a 24 GB RTX 3090 Ti.** You are here to find code defects, which are hardware independent. You cannot certify memory behaviour for the real run from this card, and you must not re-size anything to make it fit. Phase 3 says what is and is not in scope.

The stage 3 Classic ablation suite (25 ablations x 3 seeds, launched 02:50 on 2026-08-11) died at ablation 5 of 25. `run_ablations.py` has no exception handling around `run_ablation`, so one crash kills every ablation after it. Four ablations completed, 21 did not, and relaunching blind costs about 13 hours of GPU per attempt.

Your job is to find every remaining crash in minutes rather than hours, so the relaunch runs to completion.

## First: pull the fixes, then preflight

```
cd /workspace/craftax-ReMDM-planner && git pull origin main && git log --oneline -1
```

Expect `2d41229 fix lora ablation crash and matplotlib 3.11 boxplot`. If the working tree is dirty, stop and report it rather than discarding anything.

Preflight, all of it, before you start. Report and stop on any failure rather than working around it:

```
uv sync && uv sync --extra cuda
uv run python -c "import jax; print(jax.devices())"          # must show a CUDA device
nvidia-smi --query-gpu=name,memory.total --format=csv
ls checkpoints/online/ checkpoints/ppo_agents/                # gitignored, must already be on this box
uv run python -c "import craftax.craftax_classic.constants, craftax.craftax.constants"
```

The last line is the cold-node texture-cache warm-up, not a formality. Craftax builds two compressed caches on first use with a non-atomic write, and parallel first touch races it (`EOFError: Compressed file ended before the end-of-stream marker was reached`). Plain `import craftax` does not build the classic cache. On a fresh box the first run takes a while, once.

If `checkpoints/` is absent, this box has never run the suite and you cannot proceed: the smoke test loads a real pretrained checkpoint. Say so and stop.

## What is already fixed

| # | File | Defect | Fix |
|---|---|---|---|
| 1 | `ablations/training.py:1179` | per-layer grad-norm `lax.cond` branches disagreed under LoRA, 161 leaves against 113, aborting the suite | diagnostic restricted to the base tree, matching `params_diag` at L1110 |
| 2 | `ablations/training.py:1526` | the LoRA merge duplicated `apply_fn_with_lora` but dropped its `.reshape`, which fails on the 3-D attention kernels | extracted `merge_lora_into_base` in `optimizers.py`, used at both call sites |
| 3 | `analysis/plots.py:949` | `ax.boxplot(labels=)` was removed in matplotlib 3.11 and the lock pins 3.11.1, so `--analyze-only` crashed after the whole suite | `tick_labels=` |

Do not redo these. Do look for more of the same shape:

- **LoRA is the only ablation whose parameter pytree differs from the pretrained tree.** It trains `{"base": ..., "lora": ...}`, 161 leaves against 113 for every other ablation. Anything that sizes an array from `pretrained_params` while reading from `state.params` or from gradients breaks under it, and only under it. Group C ablations mask through optax and leave the tree unchanged.
- **Duplicated logic drifts.** Defect 2 existed because one copy was corrected and the other was not. If you find a third copy of anything, consolidate it rather than patching the copy.
- **Post-scan code is unexercised.** Everything after `jax.lax.scan` returns runs once per seed and has never executed for LoRA, so it has never been tested at all.

---

# Phase 1: structural sweep, all 25, isolated

Each ablation runs in its own process, so a failure costs one ablation and not the sweep.

`--fast` (`run_ablations.py:248`) overrides to `max_iter=50, num_envs=16, num_steps=64, batch_size=128, eval_steps=128`, and drops every diagnostic cadence to 10 (`grad_align`, `repr_drift`, `t_analysis`, `per_layer`; `cka` to 25), with `ewc_fisher_batches=5` and `mixed_replay_buffer_size=500`. Two consequences, both of which matter here:

- Every conditional branch in the scan body fires within the 50 iterations, so this reaches the code paths that crashed. That is what makes it a valid structural test rather than a shortcut.
- At 16 envs and batch 128 it is far inside 16 GB, so this phase is unaffected by the card.

CLI overrides are applied after `--fast` (`run_ablations.py:556-558`), so `--num-seeds 1` survives.

Verify the names against `--list` before you start, and verify every flag against `--help`. Do not reproduce a flag from this file without checking it exists.

```
cd /workspace/craftax-ReMDM-planner
mkdir -p logs/smoke
uv run python -c "import craftax.craftax_classic.constants, craftax.craftax.constants"

CKPT=checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M/
PPO=checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M
CFG=experiments/rl_finetuning/configs/ablations_final_classic_ucl.yaml

ABL="baseline_rl kl_penalty ewc llrd lora mixed_replay trust_region_kl t_curriculum \
entropy_bonus gradient_surgery advantage_clip normalized_adv bc_wins low_t \
frozen_backbone head_only attention_only ffn_only layer_ablation_top1 \
layer_ablation_top2 layer_ablation_top3 reward_filtering running_stats \
action_diversity reward_model"

for a in $ABL; do
  SECONDS=0
  uv run python experiments/rl_finetuning/run_ablations.py \
    --checkpoint "$CKPT" --ppo-checkpoint "$PPO" --ablations-config "$CFG" \
    --ablations "$a" --fast --num-seeds 1 --no-use-wandb \
    --output-dir /workspace/smoke/"$a" 2>&1 | tee logs/smoke/"$a".log
  echo "$a exit=${PIPESTATUS[0]} secs=$SECONDS" | tee -a logs/smoke/status.txt
done
```

Run these loops under `bash`. `${PIPESTATUS[0]}` is not optional and is bash syntax: `$?` after a pipe reports `tee`, so every ablation would record success, and zsh spells the array differently.

Run the whole sweep before fixing anything. The full failure set up front is worth more than the first failure early.

---

# Phase 2: repair, one ablation at a time

For each failure in `logs/smoke/status.txt`, in registry order:

1. Read the traceback in `logs/smoke/<name>.log`. Identify the defect at file and line.
2. State the root cause before editing. If it is a pytree structure or shape mismatch, give both structures and both counts.
3. Apply the smallest fix that is correct. Prefer consolidating a duplicated implementation over patching one copy.
4. Re-run that one ablation with the Phase 1 command. Do not proceed while it is still red.
5. `uv run pytest tests/` (154 tests, about 60s) and `uv run ruff check experiments/` must both pass.
6. Commit that fix alone, with a message stating the failure and the fix. Do not push. The author reviews before anything reaches `main`.

---

# Phase 3: memory behaviour, at the largest size this card can honestly test

`--fast` runs at batch 128, so it cannot exercise the parameter-sized buffers that `ewc` (Fisher diagonal), `lora` (adapters plus their optimiser state) and `mixed_replay` (transition buffer) carry at scale.

**Do not run the UCL config at production sizes on this card.** It is `num_envs: 192, batch_size: 1024`, sized for 24 GB, and its own comment records `192/2048` OOM'ing on EWC at 19.4 GiB. On 16 GB an OOM would tell you about this 4070 Ti and nothing about the machine that runs the suite. It would be a measurement of the wrong thing.

Use instead the 8 GB Classic config the repo already ships, `ablations_final_classic_qmul.yaml`: architecturally identical to the UCL config (`d_model 384, n_heads 8, n_layers 6, d_ff 768, plan_horizon 32`, so the same checkpoint loads) and differing only in `num_envs 64`, `batch_size 256`, `eval_steps 512`, `cka_batch_size 96`, `mixed_replay_buffer_size 10000`. It fits 16 GB with headroom and still exercises the buffers at a real scale.

```
QCFG=experiments/rl_finetuning/configs/ablations_final_classic_qmul.yaml
for a in $ABL; do
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > /tmp/vram_$a.txt &
  SAMPLER=$!
  XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python experiments/rl_finetuning/run_ablations.py \
    --checkpoint "$CKPT" --ppo-checkpoint "$PPO" --ablations-config "$QCFG" \
    --ablations "$a" --max-iter 3 --eval-every 1 --num-seeds 1 --no-use-wandb \
    --output-dir /workspace/mem/"$a" 2>&1 | tee logs/smoke/mem_"$a".log
  ST=${PIPESTATUS[0]}; kill $SAMPLER
  echo "$a exit=$ST peak_mib=$(sort -n /tmp/vram_$a.txt | tail -1)" | tee -a logs/smoke/mem_status.txt
done
```

`--max-iter 3 --eval-every 1` forces an eval inside three iterations, so the eval path's allocation is included rather than skipped.

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is there for measurement only. JAX otherwise preallocates about 75% of VRAM on startup, which on a 16 GB card is roughly 12 GB and makes both the peak figure and any OOM meaningless. It changes fragmentation behaviour, so it is a probe setting and must not be carried into the real run.

What this phase delivers, stated in exactly these terms in the report:

- **Does deliver:** every ablation's buffers allocate and run at 64 envs / batch 256, plus a measured peak VRAM ranking across the 25.
- **Does not deliver:** any statement about the UCL config at 192 / 1024 on 24 GB. That check stays open and belongs on the 3090 Ti before the relaunch. Say so explicitly rather than letting a green Phase 3 read as a clearance.

An OOM at the QMUL sizes on a 16 GB card would be genuinely surprising and is worth investigating as a possible leak. An OOM at UCL sizes is expected here and is not evidence of anything.

While reading that config, note that its header comment says "64 envs / 512 batch uses ~5 GB" while `batch_size` is 256. Flag the inconsistency, change nothing.

---

# Phase 4: confirm and report

Re-run Phase 1 end to end. Every ablation must exit 0. Then write `/workspace/ABLATION_SMOKE_REPORT.md`:

| Ablation | Phase 1 | Phase 3 (QMUL sizes) | Peak MiB | Seconds | Failure | Fix (commit) |

followed by the full diff, the total wall clock for the sweep, the hardware this ran on, and any item you did not resolve. Head the report with one line stating that memory at UCL production sizes is unverified and why.

---

# Constraints

1. **Never change what an ablation measures.** Fixes make the code run correctly. A change to a loss, a hyperparameter, a seed, a diagnostic's definition or a config value changes the experiment. If the only way to make an ablation run is such a change, stop, report it and leave it red.
2. **Never re-size a config to fit this card.** `num_envs` and `batch_size` change what every ablation measures, four ablations have already completed at the UCL sizes, and the suite is only interpretable if all 25 share one setting. Editing `ablations_final_classic_ucl.yaml` to fit 16 GB would silently invalidate the comparison. Phase 3 selects a different existing config; it does not modify either.
3. **Evidence for every claim.** File and line for code, command and output for anything executed. Never state a number you have not measured. A peak VRAM figure from this card is a figure from this card, and must be labelled as such.
4. **Do not touch `experiments/rl_finetuning/outputs/retrain_fix1_classic/`.** It holds the four completed ablations, and `results.json` is rewritten from empty by any run that targets that directory. Every smoke run in this task writes to `/workspace/smoke/` or `/workspace/mem/`.
5. **Do not add exception handling around `run_ablation`.** Loud failure is deliberate in this repo. The bash loop provides the isolation.
6. **Do not push, do not start the real suite.** The relaunch is the author's call.
7. UK English, no em dashes, short structured entries, tables over prose.

# Do not decide these, report them

- `RETRAIN_LOG.md` section 3 says the Classic ablations consume the seed-0 retrained DAgger checkpoint, but the literal command in that section points at `checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M/`, which the superseded table in the same file lists as superseded by FIX-1. Use whichever checkpoint loads for the smoke test, since its identity does not affect whether the code runs, and flag the discrepancy in the report.
- Under LoRA the per-layer heatmap now reports base-tree gradient norms, so its axis matches the other 24 ablations. Reporting the 48 adapter leaves instead is defensible. Flag it, do not change it.
- This prompt assumes the 4070 Ti is the test box only and the real suite returns to the 24 GB card. If the author intends to run the real suite here, that is a different task: the UCL config does not fit, choosing a config that does changes what every ablation measures, and the four already-completed cells would have to be re-run to match. Do not begin that. State it as a question.

# When the sweep is green

The author relaunches the remaining 21 into a fresh directory and merges, because reusing the original output directory would overwrite the four completed cells:

```
--ablations lora mixed_replay trust_region_kl t_curriculum entropy_bonus gradient_surgery \
  advantage_clip normalized_adv bc_wins low_t frozen_backbone head_only attention_only \
  ffn_only layer_ablation_top1 layer_ablation_top2 layer_ablation_top3 reward_filtering \
  running_stats action_diversity reward_model
--output-dir experiments/rl_finetuning/outputs/retrain_fix1_classic_part2
```

Confirm from `results.json` which ablations actually completed before trusting that list.
