# Task: make the Craftax online DAgger run fit and run fast on the 4070 Ti

**This is prompt 1 of 3.** `PERF_OFFLINE_PROMPT_CRAFTAX.md` and
`PERF_EXPERIMENTS_PROMPT_CRAFTAX.md` follow. This one owns every shared file
(`src/planners/common.py`, `src/planners/env.py`, `src/diffusion/`, `src/models/`), so run it
first and let the other two branch off its HEAD.

## Read this before anything else

The sibling repo `minihack-ReMDM-planner` has been through this exercise twice and its reports
(`PERF_GPU_RESULTS_MINIHACK.md`, `PERF_MEASURE_3090_RESULTS_MINIHACK.md`) are worth reading for
method, not for findings. **Almost nothing in them transfers.** That repo is PyTorch with CPU
MiniHack environments, and its wins were environment pooling, dropping a pixel render, and
removing host-device syncs from a Python training loop. This repo is JAX end to end: the
environment is Craftax running on the GPU, and `online.py:707` wraps the entire training run in
a single `jax.jit(jax.vmap(...))`, with the outer update loop as one `jax.lax.scan`
(`online.py:657`). There is no Python-level per-step loop to fix and no host environment to pool.

Do not go looking for the MiniHack findings here. Profile this repo on its own terms.

The questions that actually matter in a JAX program of this shape are: how much of the wall
clock is XLA compilation rather than execution, whether the working set fits in VRAM under
JAX's preallocation, and whether the recipe's derived quantities are what the configs claim.

## Where you are

`craftax-ReMDM-planner` on the 4070 Ti box (`outback.cs.ucl.ac.uk`, RTX 4070 Ti SUPER, 16 GB,
i7-14700K, 20 physical cores / 28 threads). A uv project. Repo conventions are in `CLAUDE.md`
one level up: UK English, no em dashes, evidence for every claim (command and output, file and
line). Everything happens on this box.

Confirm the box first and record the output.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --no-sync python -c "import jax;print(jax.__version__, jax.devices())"
lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
free -g && df -h /tmp
```

### Three environment facts

1. **This box needs `--extra cuda12`, not `cuda13`.** `README.md:32` requires driver >= 580 for
   CUDA 13; the driver here is 560.35.03. The two extras conflict by design
   (`pyproject.toml:33`), and in the sibling repo flipping between them corrupted the venv
   because the `nvidia-*-cu12` and `nvidia-*-cu13` wheels share namespace directories. Sync once
   with `uv sync --extra cuda12` and use `uv run --no-sync` thereafter.
2. **`LD_LIBRARY_PATH` can shadow the wheel libraries.** `README.md:32` warns that a
   `module load cuda/13.x` in the shell profile breaks the pip-installed CUDA. Check and unset it
   if present, and say whether it was.
3. **JAX preallocates 75% of the GPU by default**, and nothing in this repo sets
   `XLA_PYTHON_CLIENT_PREALLOCATE` or `XLA_PYTHON_CLIENT_MEM_FRACTION` (grep confirms: no
   occurrences in `main.py`, `src/`, `Dockerfile` or `scripts/`). On 16 GB that is a ~12 GB
   ceiling. Record what you set and why; raising the fraction is legitimate on a box you are not
   sharing, disabling preallocation trades peak headroom for fragmentation.

---

## The problem this pass exists to solve

Both final configs were written for 24 GB cards. `final_craftax_ucl.yaml:2` says "UCL RTX 4090
24 GB" and `final_classic_ucl.yaml:2` says "UCL RTX 3090 Ti 24 GB". You have 16 GB.

The good news is that the recipe was designed to be re-sized. `resolve_num_updates`
(`common.py:22`) and `resolve_scaled_hyperparams` (`common.py:87`) denominate the schedule in
environment frames rather than update steps, precisely so that changing `NUM_ENVS` yields "the
same effective experiment on any GPU". `NUM_ENVS` is therefore the sanctioned knob if the
configured size does not fit.

It is not a free knob, and the docstring's claim is a design intent, not a proof. Under a change
to `NUM_ENVS`:

| Invariant | Not invariant |
|---|---|
| total env frames | number of updates |
| LR warmup in frames | minibatch size, so gradient noise per update |
| final beta | buffer size in samples (`DAGGER_BUFFER_CYCLES` x fpu) |

So a re-sized run is comparable, not identical. Record what changed and say so.

### The configs' own comments are already stale, and this is where you start

Deriving from `common.py:22-86` and `common.py:87-168` with the values in the config files, by
arithmetic rather than measurement:

| | `final_craftax_ucl.yaml` | `final_classic_ucl.yaml` |
|---|---|---|
| `num_envs` x `num_steps` = fpu | 448 x 128 = 57,344 | 512 x 128 = 65,536 |
| `NUM_UPDATES` | 3,487 | 1,525 |
| `LR_WARMUP_STEPS` | 1,371 | 1,600 |
| comment at line 41 claims | "300 update steps (300 * 2048 * 128)" | "200 update steps (200 * 4096 * 128)" |
| `DAGGER_BUFFER_MAX` | 43,750 | 125,000 |
| comment at line 59 claims | "~200K samples on UCL" | "~1M samples on UCL" |
| `samples_per_update` | 43,456 | 49,664 |

Every one of those comments describes a `num_envs` the file no longer sets: 2048 for the Craftax
config and 4096 for the Classic one. The frame-denominated mechanism did its job, so the warmup
is still 78.6M / 104.9M frames and the final beta is still 0.385 / 0.344 as intended, but the
buffer in *samples* is 4.6x and 8x smaller than the comments say, and `dagger_beta_final: 0.385
# 0.9995^1907` at `final_craftax_ucl.yaml:58` describes 1,907 updates against an actual 3,487.

Confirm all of this with `print_config_snapshot` (`common.py:227`), which prints the resolved
warmup, decay, effective buffer size and cycles, rather than trusting my arithmetic or the
comments. Then **report it**. Whether the buffer was meant to hold a fixed number of samples or a
fixed number of update cycles is an author decision that changes the recipe, and it is not yours
to make. Correct the stale comments in the same commit as your report, since a wrong comment is
worse than none.

---

## Preflight

```
git status --short && git log --oneline -8
uv sync --extra cuda12
uv run --no-sync python -m pytest tests -q
uv run --no-sync ruff check src tests experiments
uv run --no-sync python main.py --mode smoke
```

Record the pre-existing lint and test baseline before you change anything, and do not fix
unrelated findings. `Craftax_Baselines/` is excluded from lint by `pyproject.toml:48` and is a
submodule: leave it alone.

You need a PPO expert checkpoint. DAgger asserts on it (`online.py:105-107`). Locate it, record
its path and provenance, and if there is not one on this box, stop and report rather than
training a substitute.

---

## Phase 1: separate compilation from execution, then size the run

This is the first question in any JAX program and the repo currently cannot answer it: the only
timing is `t0 = time.time()` around the whole `train_fn(rngs)` call (`online.py:709-713`), which
reports compile plus execute as one number and calls the result SPS.

Measure, at `configs/final_classic_ucl.yaml` first because it is the cheaper of the two:

| Quantity | How |
|---|---|
| compile time | time to first result with a 1-update run, or `jax.block_until_ready` on a trivial call after `.lower().compile()` |
| execute time per update | total minus compile, over a run with `online_total_timesteps` cut to a few dozen updates |
| peak VRAM | `jax.local_devices()[0].memory_stats()` |
| frames per second, honestly split | derived |

Cut the run length with `--override online_total_timesteps=...`; because `resolve_num_updates`
re-snaps the total to an integer multiple of fpu, a short run is a faithful scale model of a long
one for everything except the beta schedule.

Then the same for `configs/final_craftax_ucl.yaml`, where `obs_dim` is documented as 8268
against Classic's much smaller observation
(`experiments/rl_finetuning/configs/ablations_final_craftax_ucl.yaml:39`), and the diffusion
sampler runs 25 steps rather than 15.

**Three gates.**

1. **Does it fit?** If `num_envs: 448` / `512` OOMs at 16 GB, re-size via `NUM_ENVS` as above,
   record the full before-and-after snapshot from `print_config_snapshot`, and state plainly in
   the report that the run is comparable rather than identical. Do not reach for
   `optimistic_reset_ratio`, the model dimensions, or `plan_horizon` to save memory: those change
   the experiment in ways the frame-denominated mechanism does not compensate for.
2. **Is compilation a material share of the run?** For a full 3,487-update run it almost
   certainly is not, but for the 25-ablation suite in prompt 3 it may dominate, so measure it here
   where the program is simplest and hand the number forward. If it is minutes, the persistent
   compilation cache below is worth having.
3. **Where does execution time go?** Split it between the rollout scan (`online.py:422`), which
   includes `sample_plan` at `DIFFUSION_STEPS` per plan cycle, and the training scan
   (`online.py:546-561`), which is `update_epochs x num_minibatches` gradient steps. Those are the
   only two candidates and the ratio determines everything you do next.

---

## Phase 2: the changes

One commit each, suite and lint green at every commit.

**D1. Turn on the persistent compilation cache.** Nothing in the repo sets
`jax_compilation_cache_dir`. Setting it makes repeated runs, multi-seed runs launched as separate
processes, and the whole ablation suite skip compilation they currently repeat. It changes no
numerics. Point it at local disk, not an NFS home. Worth it in proportion to the Phase 1 compile
measurement, so quote that measurement when you justify it.

**D2. Report compile and execute separately.** `online.py:709-713` prints a single elapsed time
and divides total timesteps by it, which understates SPS by the compile time and misleads anyone
sizing a run from the log. Time the two separately and print both. Same code shape in
`offline.py:370` onwards, but leave that one to prompt 2.

**D3. Settle the `jax.debug.callback` for W&B.** `online.py:619` fires a host callback inside the
jitted update scan on every logged update. Host callbacks are ordered against the computation and
can stall the pipeline. Measure the cost with logging on and off at a fixed update count. If it
is material, batch the metrics on device and emit them once at the end of the scan, which is what
the returned `metrics` from `online.py:657` already carries. If it is not material, say so and
leave it: this is the sort of change that is easy to justify by theory and hard to justify by
measurement.

**D4. Anything Phase 1 actually found.** The list above is what reading the code suggests. If the
profile puts the time elsewhere, follow the profile and say so in the report. Use
`jax.profiler.trace` and read the result rather than guessing at kernel-level attribution.

**Deliberately not on the list.** `use_optimistic_resets: false` at `final_craftax_ucl.yaml:9`
and `final_classic_ucl.yaml:9` selects `AutoResetEnvWrapper` + `BatchEnvWrapper` over
`OptimisticResetVecEnvWrapper` (`env.py:45-53`). Optimistic resets are Craftax's documented
throughput trick and would compute `num_envs / reset_ratio` resets instead of `num_envs`, which
in full Craftax means far less world generation. It is very likely a large win. It is also a
change to the environment's reset distribution, the wrapper's own docstring warns of duplicate
resets (`Craftax_Baselines/wrappers.py:87-88`), and `env.py:37-39` records the current setting as
a deliberate benchmark-forced choice. **Measure what it would be worth and report it. Do not
switch it on.** That is the author's call, and it belongs in the paper's method section if taken.

---

## Phase 3: what must not change

The DAgger recipe produces the checkpoint the ablation suite starts from, so changing it
invalidates prompt 3 as well as the paper's headline runs. Pinned:

`env_name`, `plan_horizon`, `diffusion_schedule`, `diffusion_steps`, `diffusion_steps_eval`,
`remask_strategy`, `eta`, `t_on`, `t_off`, `temperature`, `top_p`, `d_model`, `n_heads`,
`n_layers`, `d_ff`, `obs_encoder_layers`, `obs_encoder_width`, `dropout_rate`, `lr`,
`max_grad_norm`, `num_steps`, `num_minibatches`, `update_epochs`, `online_total_timesteps`,
`offline_total_timesteps`, `dagger_beta_init`, `dagger_beta_final`, `dagger_buffer_cycles`,
`lr_warmup_frames`, `val_*`, `use_optimistic_resets`.

`NUM_ENVS` is the one sizing knob, and only under the frame-denominated mechanism, with the
resulting snapshot recorded.

The compute match against offline BC is `update_epochs x num_minibatches` gradient steps per
update with `DAGGER_TRAIN_PASSES` at 1, which `dagger_sizing` (`common.py:169`) documents as the
thing that keeps the DAgger-versus-BC comparison fair. Do not raise `DAGGER_TRAIN_PASSES`.

---

## Phase 4: verify

1. **Loss and validation-return trajectories**, short run, fixed seed, before and after. Establish
   the noise floor first by running identical code twice: JAX is deterministic given a seed and
   fixed shapes, so if two identical runs differ at all, find out why before reading anything
   into a comparison.
2. **`print_config_snapshot` output**, before and after, for both configs. Every derived quantity
   should be identical unless you re-sized `NUM_ENVS` deliberately.
3. **A real run to a meaningful fraction of the schedule** on the Classic config, with peak VRAM
   and validation returns recorded, and the same on the Craftax config if it fits.
4. **Suite, lint, and `--mode smoke`.**

---

## Phase 5: the run plan

| Quantity | Classic, before | Classic, after | Craftax, before | Craftax, after |
|---|---|---|---|---|
| `num_envs` used, and why | | | | |
| compile time | | | | |
| s per update, rollout / train split | | | | |
| `NUM_UPDATES` and total frames | | | | |
| hours per seed, `--mode online` | | | | |
| peak VRAM, headroom against 16 GB | | | | |

Then the seed question, which is a real decision on this card. `num_repeats: 1` in both configs,
and `online.py:707` vmaps over `NUM_REPEATS`, so three seeds run concurrently in one program and
multiply the working set by three. Measure whether three vmapped seeds fit in 16 GB. If they do
not, three sequential runs cost the same compute and three times the wall clock, and with D1's
compilation cache they pay compilation once. State which the run plan assumes.

Finish with a go / no-go for the paper runs, with conditions.

---

## Deliverable

`PERF_DAGGER_RESULTS_CRAFTAX.md`, in the repo's evidence style:

- Box identification, driver, JAX version, the CUDA extra, `LD_LIBRARY_PATH` state, and the XLA
  memory settings you used. Every number below is conditional on these.
- The compile-versus-execute split, which is the number this repo currently cannot report.
- The `print_config_snapshot` table with the stale comments corrected, and the buffer-sizing
  question stated as a question for the author.
- Per change: what it was worth, measured, or why you dropped it.
- What `use_optimistic_resets: true` would be worth, measured, and left off.
- The Phase 5 run plan and the go / no-go.
- What you did not do and why.

Commit only when the suite and lint are green. Ask before committing; do not push.

**Hand-off.** State which shared files you touched, so prompts 2 and 3 start from your HEAD.
