# Task: make the Craftax offline BC baseline fast, and keep it matched to DAgger

**This is prompt 2 of 3.** `PERF_DAGGER_PROMPT_CRAFTAX.md` ran first and owns the shared files.
Start from its HEAD and read `PERF_DAGGER_RESULTS_CRAFTAX.md` before doing anything: its
compile-versus-execute measurement, its VRAM figures, and above all its `NUM_ENVS` decision are
inputs to this pass, not things to re-derive.

`PERF_EXPERIMENTS_PROMPT_CRAFTAX.md` follows.

## Two things that will otherwise waste your time

**This is JAX, not PyTorch.** `offline.py:370` wraps the whole training run in one
`jax.jit(jax.vmap(...))` with the update loop as a single `jax.lax.scan` (`offline.py:333`).
The MiniHack repo's findings do not transfer; read its reports for method only.

**Craftax offline BC has no dataset and no replay buffer.** This is the biggest difference from
the sibling repo, where the equivalent pass is mostly about a multi-gigabyte `.pt` file. Here,
`--mode offline` rolls the PPO expert out live: `offline.py:197-221` is an expert-action scan
with no diffusion sampling in it, `offline.py:263` builds the training set from that rollout
alone, and there is no buffer state anywhere in the module. `--mode collect` writes an `.npz`
that offline does **not** consume (`README.md:121`). Do not go looking for a dataset problem.

What that leaves is a loop that is strictly cheaper per update than DAgger, because DAgger
additionally runs `sample_plan` inside its rollout (`online.py:317`) at `DIFFUSION_STEPS` model
forwards per plan cycle, and carries a replay buffer. If your measurements do not show offline as
the cheaper of the two per update, something is wrong and that is the finding.

## Where you are

`craftax-ReMDM-planner` on the 4070 Ti box (`outback.cs.ucl.ac.uk`, RTX 4070 Ti SUPER, 16 GB,
i7-14700K). Repo conventions are in `CLAUDE.md` one level up: UK English, no em dashes, evidence
for every claim. Everything happens on this box.

Confirm the box, then apply the same three environment facts prompt 1 established, and say that
you did: `--extra cuda12` because the driver is 560.35.03 and CUDA 13 needs 580
(`README.md:32`), `LD_LIBRARY_PATH` unset if a CUDA module is in the profile, and an explicit
choice about JAX's default 75% preallocation, which nothing in the repo sets.

```
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
uv run --no-sync python -c "import jax;print(jax.__version__, jax.devices())"
git log --oneline -8 && git status --short
uv run --no-sync python -m pytest tests -q && uv run --no-sync ruff check src tests experiments
```

You need the PPO expert checkpoint again; offline requires it for the rollout. Use the same one
prompt 1 used and record the path.

---

## The fairness constraint, which outranks every speed-up here

Offline BC exists to be the compute-matched baseline for DAgger. The match is structural:

- Both modes resolve `NUM_UPDATES` from env frames through the same function
  (`common.py:22`, called for both modes), so at the same `NUM_ENVS` and `NUM_STEPS` they run the
  same number of updates over the same number of frames.
- Both do `UPDATE_EPOCHS x NUM_MINIBATCHES` gradient steps per update. `offline.py:106` computes
  total gradient steps that way, and `dagger_sizing` (`common.py:169`) documents that
  `DAGGER_TRAIN_PASSES: 1` exists precisely so DAgger "does exactly the same gradient work per
  update as offline BC".
- Both final configs set the two budgets equal: `final_craftax_ucl.yaml:51` and `:56` are both
  `2.0e+8`; `final_classic_ucl.yaml:51` and `:56` are both `1.0e+8`.

**Therefore: whatever `NUM_ENVS` prompt 1 settled on for DAgger, offline must use the same
value.** Offline has no replay buffer and no sampler in its rollout, so it will fit at a larger
`NUM_ENVS` than DAgger does. Using that headroom would silently change the minibatch size and the
update count on one side of the comparison only, which breaks the baseline. If you find yourself
about to run offline at a different `NUM_ENVS` than DAgger, stop.

Verify the match rather than assuming it: run `print_config_snapshot` (`common.py:227`) in both
modes at the same config and confirm `num_updates`, `total_grad_steps` and the minibatch size
agree. If they do not, that is a blocker-level finding and it matters more than anything else in
this document.

---

## Phase 1: measure, both game variants

Start with `configs/final_classic_ucl.yaml`, then `configs/final_craftax_ucl.yaml` where
`obs_dim` is documented as 8268 against Classic's much smaller observation
(`experiments/rl_finetuning/configs/ablations_final_craftax_ucl.yaml:39`).

Cut the run with `--override offline_total_timesteps=...` for the timing work.
`resolve_num_updates` re-snaps the total to an integer multiple of frames-per-update, so a short
run is a faithful scale model.

| Quantity | How |
|---|---|
| compile time, separately from execution | as prompt 1 established |
| execute time per update | total minus compile, over a few dozen updates |
| split: expert rollout scan vs training scan | `offline.py:215` against `offline.py:286` |
| peak VRAM | `jax.local_devices()[0].memory_stats()` |
| updates per second and frames per second | derived, both stated honestly |

Then the comparison that is the point of this phase: **offline against DAgger, same config, same
`NUM_ENVS`, same update count.** Report per-update time for both and the ratio. That number tells
you how much of DAgger's cost is the diffusion sampling in its rollout, which is the single most
useful thing this pass can produce for the paper's compute accounting.

**Gate.** If the training scan dominates in both modes, then offline has no meaningful
mode-specific work left: the gradient step is shared code and prompt 1 already owned it. Say so,
skip Phase 2 except for the items marked shared, and spend the time on Phase 3 instead. Closing
this pass early with a measurement is a good outcome, not a failure.

---

## Phase 2: the changes

One commit each, suite and lint green at every commit.

**O1. Report compile and execute separately.** `offline.py` has the same defect prompt 1 fixed in
`online.py`: a single elapsed time around the jitted call, divided by total timesteps and printed
as SPS, with compilation folded in. Mirror whatever prompt 1 did, so the two modes report
comparable numbers. This matters more than it sounds: the offline-versus-DAgger throughput
comparison is a paper claim, and it is currently computed from two numbers that both include an
unknown amount of compilation.

**O2. The W&B host callback.** `offline.py:318` fires `jax.debug.callback` inside the jitted
update scan, exactly as `online.py:619` does. If prompt 1 measured this as material and changed
it, make the same change here so the two modes stay symmetric. If prompt 1 measured it as
immaterial, leave it and say so. Do not let the two modes diverge on this: an asymmetric logging
cost would land directly in the throughput comparison.

**O3. Whatever Phase 1 found.** Follow the profile, not the list. Use `jax.profiler.trace` and
read it rather than reasoning about kernels.

**Not in scope.** The gradient step, the model, the sampler and the environment wrapper are
prompt 1's. If you find something there, report it and coordinate rather than editing shared code
from this pass.

---

## Phase 3: what must not change

Everything pinned in prompt 1 is pinned here, plus the offline-specific keys:

`offline_total_timesteps`, `collect_temperature`, `return_weight_cap`, `num_steps`,
`num_minibatches`, `update_epochs`, `lr`, `lr_warmup_frames`, `max_grad_norm`, `plan_horizon`,
and every architecture and diffusion key.

`NUM_ENVS` must equal DAgger's. It is the one knob prompt 1 was allowed to turn, and turning it
independently here breaks the comparison.

The return weighting at `offline.py:255-262` (clip to non-negative, normalise by batch mean, clip
to `[0.1, return_weight_cap]`) is the BC objective. It is not a performance knob.

---

## Phase 4: verify

1. **Loss trajectory**, short run, fixed seed, before and after, both variants. Establish the
   noise floor by running identical code twice first: JAX is deterministic given a seed and fixed
   shapes, so any difference between two identical runs needs explaining before you interpret a
   comparison.
2. **The fairness check, restated as output.** `print_config_snapshot` for both modes at the same
   config, side by side in the report, with `num_updates`, `total_grad_steps` and minibatch size
   visible. This is the artefact the paper's compute-matching claim rests on and it should exist
   as a quotable block, not as an assertion.
3. **A real run to a meaningful fraction of the schedule**, Classic at minimum, with peak VRAM
   and the validation return curve.
4. **Suite, lint, and `--mode smoke`.**

---

## Phase 5: the run plan

| Quantity | Classic offline | Craftax offline | Classic DAgger | Craftax DAgger |
|---|---|---|---|---|
| `num_envs` (must match across the pair) | | | | |
| compile time | | | | |
| s per update, rollout / train split | | | | |
| `NUM_UPDATES`, total frames, total grad steps | | | | |
| hours per seed | | | | |
| peak VRAM, headroom against 16 GB | | | | |

Fill the DAgger columns from prompt 1's report rather than re-measuring, and label them as such.

Then the seed question: `num_repeats: 1` in both configs and `offline.py:370` vmaps over
`NUM_REPEATS`, so three seeds run concurrently and multiply the working set by three. Offline's
smaller footprint may make three vmapped seeds fit where DAgger's do not. Measure it, and note
that if the two modes end up using different seed strategies the wall-clock numbers are not
comparable even though the compute is.

Finish with hours for the full paper matrix: two game variants, two modes, however many seeds,
and a go / no-go.

---

## Deliverable

`PERF_OFFLINE_RESULTS_CRAFTAX.md`, in the repo's evidence style:

- Box, driver, JAX version, CUDA extra, XLA memory settings.
- The compile-versus-execute split for both variants.
- The offline-versus-DAgger per-update comparison at matched `NUM_ENVS`, which is the headline.
- The fairness verification block, quoted from `print_config_snapshot`.
- Per change: what it was worth, measured, or why the gate closed it. An honest "the gate closed
  and here is the number" is the expected outcome for at least one of these.
- The Phase 5 run plan and the go / no-go.
- What you did not do and why.

Commit only when the suite and lint are green. Ask before committing; do not push.

**Hand-off.** State which files you touched so prompt 3 starts from your HEAD.
