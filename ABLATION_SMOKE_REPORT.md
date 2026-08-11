# Craftax Classic ablation suite: smoke test and repair report

**Memory behaviour at UCL production sizes (`num_envs: 192`, `batch_size: 1024`) is UNVERIFIED.**
Every memory figure below was measured on a 16 GB RTX 4070 Ti SUPER at the QMUL config's
64 envs / batch 256. The UCL config is sized for a 24 GB 3090 Ti and its own comment records
`192/2048` OOM'ing on EWC at 19.4 GiB. Running it here would measure this card, not the machine
that runs the suite. That check remains open and belongs on the 3090 Ti before the relaunch.
A green Phase 3 below is not a memory clearance.

---

## 1. Outcome

All 25 ablations pass the structural sweep and the QMUL-size memory run. **No ablation code
defects were found, and no ablation code was changed.**

The single change in this report is a dependency addition needed to make the test box run JAX
on its GPU at all. It touches no ablation logic, no loss, no hyperparameter, no config value.

### Why nothing was red

The stage 3 run died at ablation 5 of 25. In registry order that is `lora`. Because
`run_ablations.py` has no exception handling around `run_ablation`, the 21 ablations after it
were **never executed** rather than being broken. Fixes 1 and 2 (commit `2d41229`, pre-existing)
removed the actual defect. This sweep confirms the remaining 21 were only unreached.

Both pre-existing fixes were confirmed under execution, not assumed:

| Fix | Evidence |
|---|---|
| 1, per-layer grad-norm restricted to base tree | All 25 ablations report exactly **113** per-layer leaves, `lora` included. `per_layer_grad_heatmap_lora.png` written. |
| 2, `merge_lora_into_base` consolidated | `lora` completed the post-scan path, FINAL score 2.1000, merged params fed the action-distribution analysis. |
| 3, matplotlib `tick_labels=` | Full analysis and plotting stage ran to completion in all 25 runs, twice. |

---

## 2. Results

Phase 1: `--fast` (50 iter, 16 envs, batch 128), UCL config, 1 seed, one process per ablation.
Phase 3: QMUL config (64 envs, batch 256), `--max-iter 3 --eval-every 1`, 1 seed,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`.

| Ablation | Phase 1 | Phase 3 (QMUL sizes) | Peak MiB | Seconds | Failure | Fix (commit) |
|---|---|---|---|---|---|---|
| baseline_rl | pass | pass | 5572 | 150 | none | n/a |
| kl_penalty | pass | pass | 5602 | 153 | none | n/a |
| ewc | pass | pass | **8806** | 194 | none | n/a |
| llrd | pass | pass | 5572 | 149 | none | n/a |
| lora | pass | pass | 5678 | 155 | none | n/a |
| mixed_replay | pass | pass | 5572 | 147 | none | n/a |
| trust_region_kl | pass | pass | 5602 | 155 | none | n/a |
| t_curriculum | pass | pass | 5570 | 146 | none | n/a |
| entropy_bonus | pass | pass | 5576 | 156 | none | n/a |
| gradient_surgery | pass | pass | 5582 | 157 | none | n/a |
| advantage_clip | pass | pass | 5572 | 146 | none | n/a |
| normalized_adv | pass | pass | 5572 | 146 | none | n/a |
| bc_wins | pass | pass | 5572 | 146 | none | n/a |
| low_t | pass | pass | 5572 | 146 | none | n/a |
| frozen_backbone | pass | pass | 5672 | 145 | none | n/a |
| head_only | pass | pass | 5674 | 145 | none | n/a |
| attention_only | pass | pass | 5640 | 148 | none | n/a |
| ffn_only | pass | pass | 5672 | 151 | none | n/a |
| layer_ablation_top1 | pass | pass | 5664 | 148 | none | n/a |
| layer_ablation_top2 | pass | pass | 5656 | 145 | none | n/a |
| layer_ablation_top3 | pass | pass | 5650 | 148 | none | n/a |
| reward_filtering | pass | pass | 5572 | 144 | none | n/a |
| running_stats | pass | pass | 5572 | 149 | none | n/a |
| action_diversity | pass | pass | 5572 | 149 | none | n/a |
| reward_model | pass | pass | 5572 | 149 | none | n/a |

### Peak VRAM ranking, this card only

| Rank | Ablation | Peak MiB |
|---|---|---|
| 1 | ewc | 8806 |
| 2 | lora | 5678 |
| 3 | head_only | 5674 |
| 4 | frozen_backbone | 5672 |
| 4 | ffn_only | 5672 |
| ... | 20 others | 5570 to 5664 |
| 25 | t_curriculum | 5570 |

`ewc` is the only meaningful outlier, carrying the Fisher diagonal at roughly +3.2 GB over
baseline. `lora` adds only ~106 MiB for rank-8 adapters plus optimiser state. `mixed_replay`
at a 10000-transition buffer shows no measurable increase over baseline (5572 MiB).
No OOM occurred at QMUL sizes, so there is no evidence of a leak.

Figures are total GPU memory used as sampled by `nvidia-smi` every 2s, which includes ~11 MiB
of Xorg. They are figures from a 16 GB RTX 4070 Ti SUPER with preallocation disabled.

---

## 3. Verification that "all pass" is real

Exit codes alone can hide silent failure, so each was checked independently.

| Check | Result |
|---|---|
| Phase 1 status lines | 25/25 `exit=0` |
| Phase 3 status lines | 25/25 `exit=0` |
| `Traceback`/`ERROR`/`Exception` in any Phase 1 log | none |
| `Traceback`/`OutOfMemory`/`RESOURCE_EXHAUSTED` in any Phase 3 log | none |
| `results.json` present and populated | 25/25 both phases |
| Diagnostics populated | 5 logged points per Phase 1 run (cadence 10 over 50 iters), all channels non-empty |
| Eval path actually fired in Phase 3 | 25/25 recorded at least one eval |
| **Negative control** | run with a bad checkpoint path recorded `exit=1`, proving `${PIPESTATUS[0]}` reports python and not `tee` |
| Phase 4 confirmation re-run | PHASE4_RESULT |

`uv run pytest tests/` → **154 passed**, 74.81s.
`uv run ruff check experiments/` → **All checks passed**.

---

## 4. The one change made

Preflight failed before any ablation could run:

```
E0811 12:01:18 platform_util.cc:279] Failed to create stream executor for device CUDA:0:
CUDA Runtime error: cudaErrorInsufficientDriver: CUDA driver version is insufficient
for CUDA runtime version
RuntimeError: Unable to initialize backend 'cuda'
```

Root cause: `pyproject.toml:25` pins `cuda = ["jax[cuda13]>=0.9.2"]`. A CUDA 13 runtime needs a
580-series driver. This box has driver `560.35.03` and only `/usr/local/cuda-12.6`. This is a
property of the test box, not a suite defect; the 02:50 run would not have hit it.

Fixed additively so the 3090 Ti is untouched. Both extras resolve to the same `jaxlib==0.11.0`
and differ only in the CUDA plugin build, so numerics are unchanged.

### Full diff, commit `0e7bdb9`

```diff
diff --git a/pyproject.toml b/pyproject.toml
index 38fde08..9508947 100644
--- a/pyproject.toml
+++ b/pyproject.toml
@@ -23,6 +23,10 @@ dependencies = [

 [project.optional-dependencies]
 cuda = ["jax[cuda13]>=0.9.2"]
+# CUDA 13 needs a 580-series driver. Boxes on a 12.x driver (e.g. the 4070 Ti
+# SUPER smoke-test box, driver 560.35.03 / CUDA 12.6) use this instead.
+# Resolves to the same jaxlib version as `cuda`, so numerics are unchanged.
+cuda12 = ["jax[cuda12]>=0.9.2"]

 [tool.uv]
 package = false
```

`uv.lock` also changed: +189 lines, the `nvidia-*-cu12` plugin entries. Mechanically generated
by `uv sync --extra cuda12`, not hand-edited.

Not pushed. This is the only commit on top of `79bcf7c`.

**This extra is test-box enablement.** The real suite on the 3090 Ti should keep using
`--extra cuda` unless that box is also on a 12.x driver.

---

## 5. Wall clock and hardware

| Item | Value |
|---|---|
| Phase 1 sweep | 3767 s (62.8 min) |
| Phase 3 sweep | 4519 s (75.3 min) |
| Phase 4 re-run | PHASE4_TIME |
| GPU | NVIDIA GeForce RTX 4070 Ti **SUPER**, 16376 MiB |
| Driver / CUDA | 560.35.03 / 12.6 |
| JAX / jaxlib | 0.11.0 / 0.11.0, `cuda12` plugin |
| Repo path | `/cs/student/project_msc/2025/dsml/mathweil/craftax-ReMDM-planner` |
| HEAD at start | `79bcf7c` (parent `2d41229`, fixes 1 to 3 present) |

---

## 6. Flagged, not decided

1. **UCL memory unverified.** As stated at the top. Open, belongs on the 3090 Ti.
2. **`RETRAIN_LOG.md` does not exist** in the working tree or anywhere in git history
   (`git log --all -- "*RETRAIN_LOG.md"` is empty). The checkpoint discrepancy described in the
   task cannot be examined from this box. I used the checkpoint the literal command names,
   `checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M/`, which loaded
   cleanly at step 100000000. Checkpoint identity does not affect whether the code runs.
3. **LoRA per-layer heatmap axis.** Confirmed empirically: under LoRA the heatmap reports the
   113 base-tree leaves, matching the other 24 ablations, rather than the 48 adapter leaves.
   Reporting adapter leaves instead is defensible. Not changed.
4. **QMUL config comment inconsistency.** `ablations_final_classic_qmul.yaml:39` says
   "64 envs / 512 batch uses ~5 GB" while `batch_size: 256` is set on line 43. Its header also
   names the target as a "QMUL H200 8 GB partition". Flagged, unchanged.
5. **`experiments/rl_finetuning/outputs/retrain_fix1_classic/` does not exist on this box**, so
   I could not confirm from `results.json` which four ablations actually completed. That must be
   checked on the 3090 Ti before trusting the 21-ablation relaunch list. Constraint 4 was
   satisfied trivially: nothing here wrote near that path.
6. **A third LoRA wrapper site exists** at `training.py:1503` (`_lora_apply_eval`). It is a thin
   partial application delegating to the shared `apply_fn_with_lora`, not a duplicated
   implementation, so it carries no drift risk of the kind that caused defect 2. Left alone.
7. **Is this box the real run box?** This report assumes the 4070 Ti SUPER is a test box and the
   suite returns to the 24 GB card. If you intend to run the real suite here, that is a different
   task: the UCL config does not fit, choosing one that does changes what every ablation measures,
   and the four completed cells would need re-running to match. Not begun. Please confirm.

---

## 7. Artefacts

| Path | Contents |
|---|---|
| `logs/smoke/*.log` | Phase 1 per-ablation logs, `status.log` |
| `logs/smoke/mem_*.log` | Phase 3 per-ablation logs, `mem_status.log` |
| `logs/smoke/rerun_*.log` | Phase 4 logs, `status2.log` |
| `logs/smoke/phase1.sh`, `phase3.sh` | Sweep drivers |
| `/cs/.../mathweil/smoke2/` | Phase 4 outputs (Phase 1 outputs deleted after use) |
| `/cs/.../mathweil/mem/` | Phase 3 outputs |

All logs are covered by the existing `*.log` ignore rule, so the working tree stays clean.
