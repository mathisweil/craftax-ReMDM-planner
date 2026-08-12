# Task: make the Craftax RL fine-tuning ablation suite affordable on the 4070 Ti

**This is prompt 3 of 3.** `PERF_DAGGER_PROMPT_CRAFTAX.md` and `PERF_OFFLINE_PROMPT_CRAFTAX.md`
ran first and own the shared files. Start from prompt 2's HEAD and read both reports. Prompt 1's
compile-versus-execute measurement, its persistent compilation cache decision and its XLA memory
settings are inputs here, not things to re-derive.

## What is different about this repo, and about this suite

**This is JAX.** The MiniHack sibling's ablation suite is a Python loop calling PyTorch, and its
speed pass is about environment pooling and host-device syncs. This one is
"entirely inside `jax.lax.scan` with no Python-level loops" (`training.py:7`): the whole
`max_iter: 500` run is one scan (`training.py:1250-1254`) under one `jax.jit`
(`training.py:1490`). Structurally it is already good. Do not port the MiniHack findings; profile
this on its own terms.

That changes where the cost is. In a program of this shape the dominant avoidable costs are
**XLA compilation**, which is paid once per distinct program, and **VRAM**, which decides whether
the configured sizes run at all. Everything else is a rounding error until proved otherwise.

## The scale

25 ablations (`registry.py`), `num_seeds: 3`
(`ablations_final_craftax_ucl.yaml:110`, and the same in the other three final ablation configs),
across two game variants. That is **150 runs of 500 iterations each**.

Each ablation has a different loss function, so each is a distinct XLA program and pays its own
compilation. Seeds within an ablation share shapes and should reuse the compiled executable
inside one process. Across processes and reruns they do not, unless the persistent compilation
cache prompt 1 considered is in place.

Diagnostics compile even when they do not run: they are `jax.lax.cond` branches inside the scan
(`training.py:1128-1196`), so both sides of every conditional are compiled into the program
whether the cadence fires or not. That is a JAX-specific compile-time cost with no MiniHack
equivalent, and it is worth attributing rather than assuming.

## Where you are

`craftax-ReMDM-planner` on the 4070 Ti box (`outback.cs.ucl.ac.uk`, RTX 4070 Ti SUPER, 16 GB,
i7-14700K). Repo conventions are in `CLAUDE.md` one level up: UK English, no em dashes, evidence
for every claim. Everything happens on this box.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --no-sync python -c "import jax;print(jax.__version__, jax.devices())"
git log --oneline -8 && git status --short
uv run --no-sync python -m pytest tests -q && uv run --no-sync ruff check src tests experiments
uv run --no-sync python experiments/rl_finetuning/run_ablations.py --list
```

The same three environment facts apply and should be restated in your report: `--extra cuda12`
because the driver is 560.35.03 and CUDA 13 needs 580 (`README.md:32`); `LD_LIBRARY_PATH` unset
if a CUDA module is in the shell profile; and an explicit choice about JAX's default 75%
preallocation, which nothing in the repo sets.

**This pass is speed only.** If an ablation is structurally broken, record it, skip it, and say
so. Do not repair the science, and do not change what any ablation computes: the suite's output
is a comparison between ablations, so a change that speeds one arm and not another is worse than
no change.

### Two config defects to report, not fix

`ablations_final_craftax_ucl.yaml:3` says the config matches the checkpoint produced by
`configs/final_craftax_full_ucl.yaml`. **That file does not exist**; the configs directory has
`final_craftax_ucl.yaml`. Establish which checkpoint the suite is actually meant to start from
and record it, because a mismatch between the config's architecture keys and the checkpoint will
fail at load, and a silent substitution would invalidate every ablation result.

`ablations_final_craftax_ucl.yaml:5` says "UCL RTX 3090 Ti (24 GB VRAM)" while
`configs/final_craftax_ucl.yaml:2` says "UCL RTX 4090 24 GB". Both are 24 GB, so nothing derives
from it, but one of them is wrong about which machine produced the checkpoint.

---

## Phase 1: measure one ablation, before changing anything

Run `baseline_rl` at the real `ablations_final_classic_ucl.yaml` settings with `--max-iter`
reduced enough to be quick but past an `eval_every: 25` and a `cka_every: 50` boundary, so the
conditional branches are exercised. Then the same for the Craftax variant, where `obs_dim` is
documented as 8268 and is about six times Classic's per-environment memory
(`ablations_final_craftax_ucl.yaml:39`).

| Quantity | How |
|---|---|
| compile time per ablation | time to first result, separated from execution |
| execute time per iteration | total minus compile, divided by iterations |
| split: rollout vs gradient step vs diagnostics | `jax.profiler.trace`, read it |
| peak VRAM | `jax.local_devices()[0].memory_stats()` |
| does the executable get reused across seeds | instrument, do not assume |

Then project the full suite: 25 ablations x 3 seeds x 2 variants, with compile and execute
separated, and label it a projection.

**Three gates before you write any code.**

1. **Does it fit?** `num_envs: 128` and `batch_size: 1024`
   (`ablations_final_craftax_ucl.yaml:41,43`) were sized for 24 GB, on a Craftax observation six
   times Classic's. If the Craftax variant OOMs at 16 GB, **stop and report before changing
   anything.** Unlike the DAgger configs, the ablation config has no frame-denominated resizing
   mechanism: `num_envs` and `batch_size` here feed the ablation comparison directly, and every
   ablation must use identical values or the comparison is meaningless. Reducing them is a
   decision about the experiment, and the honest options (run the suite at a smaller size
   uniformly, or run the Craftax variant elsewhere) are the author's to choose. Quantify what
   would fit so the choice is informed.
2. **What fraction of the suite is compilation?** This is the number that decides whether Phase 2
   is worth doing at all. If a single ablation compiles in seconds, the compilation work is
   pointless and you should say so. If it compiles in minutes, then across 150 runs it is the
   largest single item in the budget.
3. **Where does execution time go?** Rollout, gradient step, or diagnostics. If it is the
   diagnostics, note that their cadences are pinned and the answer is not to reduce them.

---

## Phase 2: the changes

One commit each, suite and lint green at every commit.

**E1. Bulk-transfer the metrics history.** `metrics_to_history` already does this correctly for
nine arrays at `training.py:1286-1294`, and then the loop at `training.py:1296` undoes the lesson:
`float(jax.device_get(all_metrics.win_rate[i]))` at `training.py:1305`, and the same shape at
1307, 1313, 1321, 1323 and onwards, each slicing a device array and transferring one scalar. At
`max_iter: 500` across 150 runs that is a lot of individually tiny device-to-host transfers.
Hoist them into the same bulk `jax.device_get` block ten lines above. This is a mechanical change
to code that has already been written correctly next to it, the values are identical by
construction, and it is the one item on this list that is certain to be safe. Measure what it
was worth anyway.

**E2. Persistent compilation cache, if prompt 1 did not already land it.** If it did, verify it
is being hit here: the suite is where it pays, because 150 runs re-enter the same 25 programs.
Report the hit rate and the wall clock with and without. Point it at local disk, not an NFS home.

**E3. Consider vmapping the seed loop.** `run_ablations.py:679` runs seeds in a Python `for`
loop, one jitted call each, while `training.py:7` states the closure is safe for `jax.vmap`.
Vmapping three seeds would compile once instead of three times and would fill the GPU better at
`num_envs: 128`. It also triples the working set, which on this card may be exactly what you
cannot afford, and it is the reason to measure gate 1 first.

If you do it, the numerical requirement is strict: each replica must receive the same PRNG key it
would have received sequentially, so that per-seed results are unchanged. Verify that directly by
comparing a vmapped 3-seed run against three sequential runs at the same seeds, not by
inspection. If they differ at all, revert it: the multi-seed mean and standard deviation are what
the ablation tables report.

**E4. Whatever Phase 1 found.** Follow the profile.

**Not in scope.** The environment wrapper, the sampler, the model and `common.py` belong to
prompt 1. `use_optimistic_resets: false` at `ablations_final_craftax_ucl.yaml:11` is the same
question prompt 1 measured and left alone; quote its number rather than re-opening it.

---

## Phase 3: what must not change

Every ablation must stay comparable with every other ablation and with the published tables.
Pinned:

`max_iter`, `num_envs`, `num_steps`, `batch_size`, `lr`, `weight_decay`, `max_grad_norm`,
`collect_temperature`, `ema_decay`, `eval_every`, `eval_steps`, `eval_replan`, every diagnostic
cadence (`grad_align_every`, `repr_drift_every`, `t_analysis_every`, `cka_every`,
`per_layer_every`, `t_analysis_n_bins`, `cka_batch_size`), the advantage knobs (`win_threshold`,
`return_weight_floor`, `return_weight_cap`), every per-group hyperparameter in Groups A to D,
`num_seeds`, and every architecture and diffusion key, which must match the pretrained checkpoint.

The diagnostics are the experiment. Reducing their cadence is not a speed-up available to you,
even though it is the largest one on the table.

`num_seeds: 3` is set consistently across all four final ablation configs. Leave it there.

---

## Phase 4: verify

1. **Same-seed comparison on at least three ablations from different groups**, before and after.
   Suggested: `baseline_rl`, one Group B signal modification, and one that exercises the mixed
   replay buffer. Compare the loss series, the effective batch size and the eval scores.
   Establish the control band first by running identical code twice: JAX is deterministic given a
   seed and fixed shapes, so if two identical runs differ, find out why before interpreting
   anything.
2. **For E3 specifically**, the vmapped-versus-sequential seed comparison described above, on at
   least one ablation, reported as exact agreement or as a reason to revert.
3. **One full ablation end to end**, both variants if both fit, with its final score against the
   pre-change tree.
4. **Suite, lint, and the fast smoke path** (`--fast`).

---

## Phase 5: the run plan

| Quantity | Classic | Craftax |
|---|---|---|
| compile time per ablation, before / after | | |
| execute time per ablation-seed, before / after | | |
| minutes per ablation-seed | | |
| hours for 25 ablations x 3 seeds | | |
| peak VRAM, headroom against 16 GB | | |
| does the configured size fit at all | | |

Derive the totals from measurement, label the projections, and state whether the two variants can
run concurrently on one card or must be sequential.

Then a go / no-go for launching the full suite here, with conditions. If the Craftax variant does
not fit at the configured sizes, say so plainly and give the author the numbers they need to
decide between resizing uniformly and running that variant elsewhere. Do not resize it yourself.

---

## Deliverable

`PERF_EXPERIMENTS_RESULTS_CRAFTAX.md`, in the repo's evidence style:

- Box, driver, JAX version, CUDA extra, XLA memory settings, and which checkpoint and config
  pair you ran against.
- The compile-versus-execute split per ablation, and the projected share of the whole suite. This
  is the headline for a JAX suite and it is the number nobody currently has.
- The VRAM answer for both variants at the configured sizes, with what would fit if they do not.
- Per change: what it was worth, measured, or why the gate closed it.
- Phase 4 results, including the control band and the vmap seed-equivalence check if you did E3.
- The Phase 5 run plan and the go / no-go.
- The two config defects above, and anything you handed off as a structural failure rather than a
  speed problem.
- What you did not do and why.

Commit only when the suite and lint are green. Ask before committing; do not push.
