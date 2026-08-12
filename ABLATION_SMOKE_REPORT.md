# Craftax Classic ablation suite: structural smoke test

**Memory behaviour at the UCL production sizes (`num_envs: 192`, `batch_size: 1024`) is
unverified and stays open.** This ran on a 16 GB RTX 4070 Ti; the suite is configured for a
24 GB card, and its own comment records `192/2048` OOM'ing on EWC at 19.4 GiB. An OOM here
would measure this card, not the machine that runs the suite. Phase 3 below is a 64-env /
256-batch check on an existing 8 GB config, and a green Phase 3 is not a clearance for the
production run.

## Corrections applied to `ABLATION_SMOKE_PROMPT.md`

The prompt predates the three performance passes in this repo and two of its preflight
lines are now wrong. Both corrections come from `PERF_ORCHESTRATOR_PROMPT_CRAFTAX.md`:

| Prompt line | Says | Actually |
|---|---|---|
| 24 | `uv sync --extra cuda` | Not a valid extra. `pyproject.toml:28-29` defines `cuda12` and `cuda13` only, and they conflict by design. The driver here is 560.35.03 and CUDA 13 needs 580, so **`uv sync --extra cuda12`**, the same extra the three perf passes used. |
| 16, 19 | `git pull origin main`, expect HEAD `2d41229` | Not done. `2d41229` is an ancestor; this branch is many commits ahead with unpushed local work, and pulling risks a merge with no mandate to resolve it. **Actual HEAD recorded below.** |

Paths: the prompt assumes `/workspace/craftax-ReMDM-planner` with scratch in `/workspace/smoke/`
and `/workspace/mem/`. Real paths on this box:

| | Path |
|---|---|
| Repo | `/cs/student/project_msc/2025/dsml/mathweil/craftax-ReMDM-planner` (NFS) |
| Phase 1 scratch | `/var/tmp/mathweil-smoke/smoke/` (local disk) |
| Phase 3 scratch | `/var/tmp/mathweil-smoke/mem/` (local disk) |
| Logs | `logs/smoke/` in the repo, as the prompt specifies |

The rest of the preflight applied as written, including the Craftax texture-cache warm-up
and the `checkpoints/` check.

## Hardware and preflight

```
$ nvidia-smi --query-gpu=name,memory.total --format=csv
name, memory.total [MiB]
NVIDIA GeForce RTX 4070 Ti SUPER, 16376

$ uv run --no-sync python -c "import jax; print(jax.devices())"
[CudaDevice(id=0)]

$ uv run --no-sync python -c "import craftax.craftax_classic.constants, craftax.craftax.constants"
(both caches already built, 1.9 s)

$ ls checkpoints/online/ checkpoints/ppo_agents/
Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M
Craftax-Classic-Symbolic-v1-PPO_RNN-1000M
Craftax-Symbolic-v1-PPO_RNN-1000M
```

Host `outback.cs.ucl.ac.uk`, driver 560.35.03, JAX 0.11.0, extra `cuda12`.
Checkpoint: `checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M`
(`resume_metadata.json`: `mode: online`, `update_step: 1525`).

## Result: the sweep is green, no repairs were needed

**All 25 ablations pass Phase 1 and all 25 pass Phase 3.** Phase 2 is empty: there was
nothing to fix. The three defects the prompt lists as already fixed are present in this
tree and hold, including under LoRA:

| # | Fix | Verified present |
|---|---|---|
| 1 | per-layer grad-norm restricted to the base tree | `training.py:1182`, `g = grads_for_diag["base"] if is_lora else grads_for_diag` |
| 2 | `merge_lora_into_base` extracted | `optimizers.py:269`, used at `optimizers.py:334` and `training.py:1548` |
| 3 | `ax.boxplot(tick_labels=)` | `plots.py:949` |

`lora` passes both phases, so the post-scan code the prompt flags as never having executed
for LoRA now does.

## Per-ablation results

Phase 1: `--fast` (50 iterations, 16 envs, batch 128), UCL config, 1 seed, isolated process.
Phase 3: QMUL config (64 envs, batch 256), `--max-iter 3 --eval-every 1`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.

| Ablation | Phase 1 | Seconds | Phase 3 (QMUL sizes) | Peak MiB | Failure | Fix |
|---|---|---|---|---|---|---|
| `baseline_rl` | pass | 150 | pass | 5572 | none | none needed |
| `kl_penalty` | pass | 160 | pass | 5602 | none | none needed |
| `ewc` | pass | 200 | pass | 8806 | none | none needed |
| `llrd` | pass | 150 | pass | 5572 | none | none needed |
| `lora` | pass | 158 | pass | 5678 | none | none needed |
| `mixed_replay` | pass | 152 | pass | 5572 | none | none needed |
| `trust_region_kl` | pass | 157 | pass | 5600 | none | none needed |
| `t_curriculum` | pass | 151 | pass | 5570 | none | none needed |
| `entropy_bonus` | pass | 156 | pass | 5576 | none | none needed |
| `gradient_surgery` | pass | 160 | pass | 5582 | none | none needed |
| `advantage_clip` | pass | 154 | pass | 5572 | none | none needed |
| `normalized_adv` | pass | 147 | pass | 5572 | none | none needed |
| `bc_wins` | pass | 148 | pass | 5572 | none | none needed |
| `low_t` | pass | 152 | pass | 5572 | none | none needed |
| `frozen_backbone` | pass | 151 | pass | 5672 | none | none needed |
| `head_only` | pass | 151 | pass | 5674 | none | none needed |
| `attention_only` | pass | 151 | pass | 5640 | none | none needed |
| `ffn_only` | pass | 152 | pass | 5672 | none | none needed |
| `layer_ablation_top1` | pass | 153 | pass | 5664 | none | none needed |
| `layer_ablation_top2` | pass | 150 | pass | 5656 | none | none needed |
| `layer_ablation_top3` | pass | 151 | pass | 5650 | none | none needed |
| `reward_filtering` | pass | 147 | pass | 5572 | none | none needed |
| `running_stats` | pass | 149 | pass | 5572 | none | none needed |
| `action_diversity` | pass | 147 | pass | 5572 | none | none needed |
| `reward_model` | pass | 151 | pass | 5572 | none | none needed |

Total Phase 1 wall clock: **3848 s (1 h 3 min)** for 25 ablations, 150 s median.
Phase 3 added about 40 minutes.

## Phase 3: what it does and does not establish

**Does establish.** Every ablation's parameter-sized buffers allocate and run at 64 envs /
batch 256, with an eval forced inside three iterations so the eval path's allocation is
included. Peak VRAM ranking across the 25, sampled at 2 s with preallocation disabled:

| Rank | Ablation | Peak MiB |
|---|---|---|
| 1 | `ewc` | **8,806** |
| 2 | `lora` | 5,678 |
| 3 | `head_only` | 5,674 |
| 4= | `frozen_backbone`, `ffn_only` | 5,672 |
| ... | the remaining 20 | 5,570 to 5,664 |

`ewc` is the clear outlier at **1.58x** the next ablation, which is the Fisher diagonal:
one parameter-sized buffer plus its accumulation. Everything else sits inside a 108 MiB
band, so no other ablation carries a buffer that shows at this scale. `lora` and
`mixed_replay` (5,678 and 5,572 MiB) do not stand out, which is worth knowing: the LoRA
adapters and the transition buffer are small next to activations at batch 256.

No OOM anywhere, and none was expected: the largest figure is 8.8 GiB of a 16.4 GiB card.

**Does not establish.** Nothing about the UCL config at 192 envs / batch 1024 on a 24 GB
card. That check stays open and belongs on the 3090 Ti before the relaunch. The scaling is
not linear in a way that can be extrapolated from here: `ewc` at 64/256 uses 8.8 GiB, and
the config's own comment records `192/2048` OOM'ing at 19.4 GiB, so EWC at 192/1024 is the
cell to watch and this card cannot tell you about it.

These peaks are from an RTX 4070 Ti with `XLA_PYTHON_CLIENT_PREALLOCATE=false`, a probe
setting, and must not be read as production figures.

## Interruption: results invalidated and re-run

Three Phase 3 runs were destroyed mid-flight at 18:58 to 19:00 by concurrent activity in
this working tree, not by any defect:

- `checkpoints/` was deleted and re-uploaded by the author while the sweep was running.
- A concurrent `git` operation (commit `3c654a4`, 18:58:42) rewrote the working tree,
  including `logs/smoke/mem_status.txt` mid-append.

`running_stats` lost its status line entirely, and `action_diversity` died with
`KeyError: 'NUM_ENVS'` against a traceback whose line numbers did not match the file on
disk, which is the signature of the source changing under a running process.

**All three (`running_stats`, `action_diversity`, `reward_model`) were re-run against
the restored checkpoints and all three pass.** The 22 results from before 18:58 stand. The
table above carries the re-run values.

## Items for the author, not decided here

1. **A Hugging Face token is exposed in the repo root** as an empty file named
   `HF_TOKEN=hf_ZkVS…`, created 19:00, almost certainly a shell-redirect mishap during the
   re-upload. It is untracked and `git log --all -- 'HF_TOKEN=*'` is empty, so it has not
   been committed. **Rotate the token, then delete the file.** Not deleted here: it is a
   credential and destroying the evidence is the author's call.
2. **The re-upload put MiniHack checkpoints into the Craftax repo.**
   `mathisweil/remdm-craftax-checkpoints` now carries
   `checkpoints/offline/Minihack-Offline-Diffusion-BC-100M` (PyTorch: `model.safetensors`,
   `offline_step50000.pth`) and `checkpoints/online/Minihack-Online-Diffusion-DAgger-100M`
   (only a `config.yaml`, no weights). Neither belongs in a Craftax repo.
3. **There is still no full-Craftax planner checkpoint**, and no Craftax offline BC
   checkpoint. `README.md:245-248` lists four; one exists. This blocks the full-Craftax
   ablation arm entirely, as recorded in `PERF_EXPERIMENTS_RESULTS_CRAFTAX.md` section 2.1.
4. **`RETRAIN_LOG.md` checkpoint discrepancy.** That file does not exist in this tree, so
   the discrepancy the prompt describes could not be checked. The smoke test used
   `checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M`, which
   loads and whose identity does not affect whether the code runs.
5. **LoRA per-layer heatmap axis.** Under LoRA it now reports base-tree gradient norms so
   its axis matches the other 24. Reporting the 48 adapter leaves instead is defensible.
   Flagged, not changed.
6. **QMUL config header inconsistency.** `ablations_final_classic_qmul.yaml:39` says
   "64 envs / 512 batch uses ~5 GB" while `batch_size` is 256. Flagged, not changed. The
   measured figure at 64/256 is 5.57 GiB, so the "~5 GB" is about right for the sizes the
   file actually sets and the "512" is wrong.
7. **Is the real suite meant to run here?** This report assumes not. If it is, that is a
   different task: the UCL config does not fit at 192/1024 on 16 GB, choosing one that does
   changes what every ablation measures, and the four completed cells would have to be
   re-run to match. Not begun.

## Diff

No source changes were needed. The sweep was green on the first pass, so this task produced
no commits to `experiments/` beyond the report itself.

## When the sweep is relaunched

Confirm from `results.json` which ablations actually completed before trusting any list of
survivors. The relaunch is the author's call and was not started.
