# Craftax offline BC on the 4070 Ti: measurements, the DAgger comparison, and the run plan

Prompt 2 of 3 (`PERF_OFFLINE_PROMPT_CRAFTAX.md`). Starts from prompt 1's HEAD and takes
its `NUM_ENVS` decision, its compile measurements and its VRAM figures as inputs rather
than re-deriving them. Read `PERF_DAGGER_RESULTS_CRAFTAX.md` first.

---

## 1. Box and settings

Same box and same three environment facts prompt 1 established, re-confirmed at the start
of this pass:

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, 560.35.03

$ uv run --no-sync python -c "import jax;print(jax.__version__, jax.devices())"
0.11.0 [CudaDevice(id=0)]
```

| Setting | Value |
|---|---|
| CUDA extra | `cuda12`, because the driver is 560.35.03 and CUDA 13 needs 580 (`README.md:32`). Synced once; `uv run --no-sync` throughout. |
| `LD_LIBRARY_PATH` | `/opt/ucl/lib:/usr/X11R6/lib`, unchanged. No CUDA libraries in either directory, so nothing shadows the wheels; JAX resolves `CudaDevice(id=0)`. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | unset, so preallocation stays on, as in prompt 1. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.90` for every run below. Nothing in the repo sets it; the default 0.75 caps the allocator at 12,564 MB of a 17,171 MB card and section 4 of prompt 1's report shows that is the difference between `num_envs: 384` running and failing. |

PPO expert: the same checkpoints prompt 1 downloaded from the Hub,
`checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M` and
`checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M`, both 1e9 env frames.
`checkpoints/` is gitignored.

Starting HEAD: `08aeb76`, prompt 1's final commit.

## 2. The fairness constraint, verified rather than assumed

`NUM_ENVS` is **384** for `final_classic_ucl.yaml` and **256** for
`final_craftax_ucl.yaml`, both taken from prompt 1 and not re-derived. Offline does fit
at a larger `num_envs` than DAgger in principle, and section 5 shows it barely does in
practice; either way it is not used, because using it would change the minibatch size and
the update count on one side of the comparison only.

`print_config_snapshot` for both modes at `configs/final_classic_ucl.yaml`,
`num_envs: 384`, quoted rather than asserted:

```
  OFFLINE training — config snapshot                 ONLINE training — config snapshot
    num_envs            = 384                          num_envs            = 384
    num_steps           = 128                          num_steps           = 128
    fpu (envs*steps)    = 49152                        fpu (envs*steps)    = 49152
    samples_per_update  = 37,248                       samples_per_update  = 37,248
    num_minibatches     = 8  (minibatch=4656)          num_minibatches     = 8  (minibatch=4656)
    update_epochs       = 8                            update_epochs       = 8
    offline_total_timesteps  = 99,975,168              online_total_timesteps   = 99,975,168
    num_updates              = 2,034                   num_updates              = 2,034
    total_grad_steps         = 130,176                 total_grad_steps         = 130,176
```

The three quantities the compute match rests on agree exactly: **2,034 updates, 130,176
gradient steps, 4,656 samples per minibatch, over the same 99,975,168 env frames**. The
match is structural rather than coincidental: `resolve_num_updates` (`common.py:22`) is
called for both modes with the same frame budget, both configs set
`offline_total_timesteps` and `online_total_timesteps` equal, and `DAGGER_TRAIN_PASSES: 1`
keeps DAgger's per-update gradient work equal to offline's.

`tests/test_smoke_src.py::test_offline_and_dagger_stay_compute_matched` now pins this for
all four final configs, so the match cannot drift silently.

### 2.1 The snapshot was misreporting the minibatch, in both modes

Producing that block required fixing it first. `print_config_snapshot` derived

```python
minibatch = fpu // int(config["NUM_MINIBATCHES"])
```

but neither mode trains on raw frames. Both extract sliding windows: a rollout of
`num_steps` transitions yields `num_steps - plan_horizon + 1` windows per environment.
`offline.py:57-68` computes exactly that and sets `MINIBATCH_SIZE` from it, and the DAgger
training scan reshapes a dataset of the same size. At the shipped `plan_horizon: 32` and
`num_steps: 128` the ratio is 128/97, so the printed figure overstated the real one by
32%: **6,144 printed against 4,656 used**, on `final_classic_ucl.yaml` at `num_envs: 384`.

This is cosmetic in the sense that no training behaviour depended on it, and not cosmetic
at all in the sense that the snapshot is the artefact the paper's compute-matching claim is
read off. Fixed by deriving from `dagger_sizing`, which is already the single source of
truth for the window count, and by printing `samples_per_update` in both modes so the
comparison above can be read rather than inferred. The fairness conclusion is unchanged:
both modes were always using the same real minibatch, and both were misreporting it
identically.

---

## 3. Compile versus execute

Method as prompt 1 established: `train_fn.lower(rngs).compile()`, timed, then a blocked
execution, timed. Cold, no compilation cache, `MEM_FRACTION=0.90`.

| Config | `num_envs` | Offline compile | DAgger compile (prompt 1) |
|---|---|---|---|
| `final_classic_ucl.yaml` | 384 | 47.2 to 48.9 s | 50 to 81 s |
| `final_craftax_ucl.yaml` | 256 | 83.9 to 84.3 s | 86 to 92 s |

Offline compiles marginally faster, which is what a smaller graph should do, and the gap
is small enough not to matter. As in prompt 1, compilation is 0.04% of a full run and
tens of minutes across a 25-entry ablation suite, which is what the compilation cache
added in prompt 1 (`jax_compilation_cache_dir`) is for.

## 4. The headline: offline against DAgger at matched `NUM_ENVS`

Same method as prompt 1 section 6. Validation held to a single firing at update 0 so
differencing two run lengths cancels compilation, the initial reset and that validation.
Halving `update_epochs` halves the training scan and leaves everything else alone.

**Classic, `num_envs: 384`**

| Run | Updates | `update_epochs` | Execute |
|---|---|---|---|
| `off_cl_u4` | 4 | 8 | 273.16 s |
| `off_cl_u12` | 12 | 8 | 751.75 s |
| `off_cl_u4e4` | 4 | 4 | 153.36 s |
| `off_cl_u12e4` | 12 | 4 | 395.50 s |

```
marginal per update, epochs=8 : (751.75 - 273.16) / 8 = 59.82 s
marginal per update, epochs=4 : (395.50 - 153.36) / 8 = 30.27 s
training scan  = 2 x (59.82 - 30.27) = 59.11 s/update   98.8%
non-epoch work = 59.82 - 59.11       =  0.71 s/update    1.2%
```

**Full Craftax, `num_envs: 256`**

| Run | Updates | `update_epochs` | Execute |
|---|---|---|---|
| `off_cx_u2` | 2 | 8 | 104.60 s |
| `off_cx_u6` | 6 | 8 | 264.69 s |
| `off_cx_u2e4` | 2 | 4 | 65.18 s |
| `off_cx_u6e4` | 6 | 4 | 148.08 s |

```
marginal per update, epochs=8 : (264.69 - 104.60) / 4 = 40.02 s
marginal per update, epochs=4 : (148.08 -  65.18) / 4 = 20.73 s
training scan  = 2 x (40.02 - 20.73) = 38.60 s/update   96.4%
non-epoch work = 40.02 - 38.60       =  1.43 s/update    3.6%
```

**The comparison**

| | Classic, 384 envs | | full Craftax, 256 envs | |
|---|---|---|---|---|
| | offline | DAgger | offline | DAgger |
| per update | **59.82 s** | 61.84 s | **40.02 s** | 41.00 s |
| training scan | 59.11 s (98.8%) | 61.12 s (98.8%) | 38.60 s (96.4%) | 38.44 s (93.7%) |
| rollout / non-epoch | 0.71 s (1.2%) | 0.72 s (1.2%) | 1.43 s (3.6%) | 2.56 s (6.3%) |
| frames per second | 822 | 795 | 819 | 799 |
| DAgger / offline | | **1.034** | | **1.024** |

DAgger columns are prompt 1's, not re-measured, as the prompt instructs.

Offline is the cheaper of the two, as it must be, but by **3.4% on Classic and 2.4% on
full Craftax**, not by anything like the margin the structural difference suggests.

**So how much of DAgger's cost is the diffusion sampling in its rollout?** Between 1 and
2.4% of an update. On full Craftax the attribution is clean: the two training scans agree
to 0.4% (38.60 against 38.44, inside the 0.3% noise floor plus rounding), and the whole
gap sits in the rollout, where DAgger's 2.56 s against offline's 1.43 s is the 1.13 s of
`sample_plan` at 25 diffusion steps per plan cycle plus the replay-buffer write and
gather. That is the number for the paper's compute accounting: **DAgger's extra rollout
machinery costs 1.13 s of a 41.00 s update, 2.8%.**

On Classic the attribution is not clean and I am not going to pretend otherwise. The
non-epoch buckets are identical (0.71 against 0.72), so the 2.02 s gap lands in the
training scan, which is the same shared `grad_step` over the same 4,656-sample minibatch
in both modes. 3.3% is ten times the noise floor, so it is real, and the obvious
candidates do not explain it: DAgger nests its epoch scan one level deeper inside a
`_pass` scan of length 1 (`online.py:558`), but the buffer gather in that scan does not
scale with `update_epochs` and so would land in the non-epoch bucket. The honest reading
is that XLA compiles the two graphs differently, which prompt 1 already showed it does
aggressively and non-monotonically for these shapes. It does not change any conclusion
here, because the answer either way is "a few percent".

### 4.1 The gate closes

`PERF_OFFLINE_PROMPT_CRAFTAX.md:101-104`: "If the training scan dominates in both modes,
then offline has no meaningful mode-specific work left". It does, at **96 to 99% in both
modes**. The gradient step is shared code and prompt 1 already owned it, the environment
is 1 to 4% of an update, and the diffusion sampler that distinguishes DAgger is under 3%.

There is no offline-specific optimisation worth making. Phase 2 is therefore O1 (shared,
done) and O2 (mirrored from prompt 1, below); O3 is closed by measurement.

### 4.2 Offline does not use less memory

The prompt expects offline to "fit at a larger `NUM_ENVS` than DAgger" because it has no
replay buffer and no sampler. Measured peak allocator use, same `num_envs`, same settings:

| | offline | DAgger (prompt 1) |
|---|---|---|
| Classic, 384 envs | **14,183 MB** | 14,016 MB |
| full Craftax, 256 envs | **14,174 MB** | 14,121 MB |

Offline peaks slightly *higher* in both. The difference is about 1%, so the practical
answer is that the two modes cost the same memory and offline has no headroom to spend
even if the fairness constraint permitted spending it. A plausible mechanism is that
offline materialises the entire rollout trajectory and then all `valid_per_rollout`
windows of it at once (`offline.py:216-247`), where DAgger extracts windows per plan cycle
and holds its buffer as a persistent allocation rather than a transient peak, but that is
inference from the code rather than something this pass measured, so treat it as
unverified. The measurement is the point: there is no headroom, so the fairness
constraint costs nothing.

## 5. Validation, and the seed question

Validation priced the same way as prompt 1, by running four firings instead of one
(Classic) or two instead of one (full Craftax):

| | offline | DAgger (prompt 1) |
|---|---|---|
| Classic, 384 envs | 32.24 s per firing | 30.66 s |
| full Craftax, 256 envs | 22.22 s per firing | 22.08 s |

Both modes call the same `make_validate` closure (`common.py:452`) with the same config, so
agreement to within a few percent is the expected result and its absence would have been
the finding.

**The seed question.** `offline.py:370` vmaps over `NUM_REPEATS` exactly as `online.py:707`
does. The prompt suggests offline's smaller footprint may make three vmapped seeds fit
where DAgger's do not. It does not. XLA temp plus output, compile-only:

| Config, `num_envs` | `num_repeats: 1` | 2 | 3 | DAgger at 3 (prompt 1) |
|---|---|---|---|---|
| Classic, 384 | 14,164 MB | 30,541 MB | 50,209 MB | 53,411 MB |
| full Craftax, 256 | 14,141 MB | 23,694 MB | 38,932 MB | 43,159 MB |

Offline is 6 to 10% smaller than DAgger at three seeds and both are two to three times the
17,171 MB card. **Both modes therefore use the same seed strategy: three sequential runs at
`num_repeats: 1`.** That is the outcome the prompt's caveat was written against, and it is
the good one: the wall-clock numbers stay comparable because the seed strategies match.

## 6. Deterministic kernels break the matched pair at prompt 1's `num_envs`

Prompt 1's recommended plan sets `XLA_FLAGS=--xla_gpu_deterministic_ops=true`, because
without it two runs at one seed do not agree, and because on this card it is also faster.
That recommendation was measured on DAgger alone. It does not survive contact with the
fairness constraint.

Offline, Classic, deterministic, XLA temp plus output, compile-only:

| `num_envs` | offline, deterministic | DAgger, deterministic (prompt 1) |
|---|---|---|
| 384 | (not measured; 320 already fails) | 16,449 MB, OOMs |
| 320 | **18,294 MB, OOMs** | 14,226 MB, runs |
| 288 | 15,789 MB, over the 15,076 MB limit | 15,888 MB |
| 256 | **13,858 MB, runs** | 14,260 MB, runs |
| 224 | 13,503 MB, runs | - |

DAgger runs at 320 under deterministic ops; offline does not. Since `NUM_ENVS` must be
equal across the pair, **the largest workable deterministic size for Classic is 256, set by
offline, not by DAgger**. Full Craftax is unaffected: both modes run at 256 under
deterministic ops (offline peak 14,338 MB, DAgger 14,287 MB).

This is exactly the situation prompt 1's report could not see, because it only measured one
half of the pair. Recorded here rather than resolved: whether 256 is an acceptable Classic
size is the same recipe question section 4 of prompt 1's report already put to the author,
one step further along.

## 7. The run plan

DAgger columns are prompt 1's and are labelled as such; offline columns are measured in
this pass. Update and validation counts from `print_config_snapshot` at the stated
`num_envs`.

**Default XLA settings**

| | Classic offline | Classic DAgger* | Craftax offline | Craftax DAgger* |
|---|---|---|---|---|
| `num_envs` (matched) | 384 | 384 | 256 | 256 |
| compile | 48 s | 52 s | 84 s | 92 s |
| s per update | 59.82 | 61.84 | 40.02 | 41.00 |
| rollout / non-epoch split | 0.71 / 59.11 | 0.72 / 61.12 | 1.43 / 38.60 | 2.56 / 38.44 |
| `NUM_UPDATES` | 2,034 | 2,034 | 6,103 | 6,103 |
| total frames | 99,975,168 | 99,975,168 | 199,983,104 | 199,983,104 |
| total grad steps | 130,176 | 130,176 | 390,592 | 390,592 |
| validation | 102 x 32.24 s | 102 x 30.66 s | 204 x 22.22 s | 204 x 22.08 s |
| **hours per seed** | **34.7** | **35.9** | **69.1** | **70.8** |
| peak VRAM | 14,183 MB | 14,016 MB | 14,174 MB | 14,121 MB |
| headroom vs 17,171 MB | 2,988 MB (17%) | 3,155 MB (18%) | 2,997 MB (17%) | 3,050 MB (18%) |

`*` from `PERF_DAGGER_RESULTS_CRAFTAX.md`, not re-measured.

**Deterministic kernels, `num_envs` set by the offline half of each pair**

| | Classic offline | Classic DAgger* | Craftax offline | Craftax DAgger* |
|---|---|---|---|---|
| `num_envs` (matched) | 256 | 256 | 256 | 256 |
| s per update | 32.55 | 33.06 | 34.16 | 35.63 |
| `NUM_UPDATES` | 3,051 | 3,051 | 6,103 | 6,103 |
| total frames | 99,975,168 | 99,975,168 | 199,983,104 | 199,983,104 |
| minibatch | 3,104 | 3,104 | 3,104 | 3,104 |
| **hours per seed** | **28.5** | **28.9** | **59.2** | **62.0** |
| peak VRAM | 13,876 MB | 14,279 MB | 14,338 MB | 14,287 MB |

```
Classic offline, default : 2,034 x 59.82 + 102 x 32.24 + 48 = 125,010 s = 34.7 h
Craftax offline, default : 6,103 x 40.02 + 204 x 22.22 + 84 = 248,859 s = 69.1 h
Classic offline, det     : 3,051 x 32.55 + 102 x 32.24 + 78 = 102,676 s = 28.5 h
Craftax offline, det     : 6,103 x 34.16 + 204 x 22.22 + 84 = 213,095 s = 59.2 h
```

Classic's deterministic validation cost is carried across from the default-settings
measurement at 384 envs (32.24 s) as an upper bound; it is 3% of the total, so nothing
turns on it. Everything else is measured at the size and settings shown.

### 7.1 The full paper matrix

Two game variants, two modes, three seeds, all sequential because section 5 rules out
vmapped seeds for both modes:

| | default settings | deterministic |
|---|---|---|
| Classic, both modes, 3 seeds | (34.7 + 35.9) x 3 = 211.8 h | (28.5 + 28.9) x 3 = 172.2 h |
| full Craftax, both modes, 3 seeds | (69.1 + 70.8) x 3 = 419.7 h | (59.2 + 62.0) x 3 = 363.6 h |
| **total** | **631.5 h = 26.3 days** | **535.8 h = 22.3 days** |

That is exclusive use of this GPU, before prompt 3's ablation suite gets any time on it.

### 7.2 Go / no-go

**Classic, both modes: go.** 57 to 71 hours for the compute-matched pair at one seed, 7 to
9 days for three. Conditions: matched `num_envs` (384 by default, 256 under deterministic
kernels), `MEM_FRACTION=0.90`, exclusive use of the card, and the compilation cache pointed
at local disk.

**Full Craftax, both modes: no-go on this box.** 121 to 140 hours per seed for the pair,
15 to 18 days for three. Prompt 1 already returned a no-go for the DAgger half at 62 to 71
hours; adding the compute-matched baseline doubles it. Nothing in this pass changes the
arithmetic, because section 4 shows both modes are 96 to 99% gradient work and the gradient
work is the recipe.

The lever, as in prompt 1, is `offline_total_timesteps` and `online_total_timesteps`,
currently 2.0e8 for Craftax against Classic's 1.0e8. They must move together or the
compute match breaks. Halving both would put full Craftax at Classic's cost. That is a
change to the experiment and it is the author's.

## 8. The changes, and what each was worth

### O1: compile and execute reported separately (`266461b`)

`offline.py:370` had the same defect prompt 1 fixed in `online.py`: `out = train_fn(rngs)`
timed with no `block_until_ready`, divided by `OFFLINE_TOTAL_TIMESTEPS` and printed as SPS.
JAX dispatch is asynchronous, so the number was compilation plus enqueue.

This mattered more here than in prompt 1. The offline-versus-DAgger throughput comparison
is a paper claim, and before this change it was computed from two numbers that each mostly
measured compilation. Reused `compile_and_run` and `format_timing` from `common.py` so both
modes now report identical quantities.

**Worth:** no wall clock, and section 4 of this report. Without it the headline ratio in
this pass could not be computed at all.

### O2: the W&B host callback, left alone

Prompt 1 measured the same `jax.debug.callback` in `online.py:619` at +0.02 s per update,
+0.03%, which is a fifth of the noise floor. `offline.py:318` installs it identically
through the same `make_wandb_callback`. Prompt 1's instruction is to mirror: it measured
immaterial, so this pass leaves it. **The two modes stay symmetric, which is what would
have landed in the throughput comparison had they diverged.** Not re-measured, because
re-measuring an effect already shown to be below the noise floor on the same closure would
not resolve anything.

### O3: closed by the gate

Section 4.1. The training scan is 96 to 99% of an update in both modes, so there is no
offline-specific work left to optimise. This is the outcome
`PERF_OFFLINE_PROMPT_CRAFTAX.md:101-104` describes as "a good outcome, not a failure", and
the number that closes it is the split, not an assertion.

### The snapshot minibatch fix (`9207fca`)

Section 2.1. Shared code, but it is the artefact this pass is required to quote, and it was
wrong by 32% in both modes. Comment-only in effect: no training behaviour depended on it.

## 9. Verification

### 9.1 Numerics and the noise floor

Prompt 1 section 9.1 established, on this GPU, that two runs of identical code at one seed
do **not** agree: XLA's GPU backend is non-deterministic by default, and
`--xla_gpu_deterministic_ops=true` makes the check pass exactly. That finding applies to
`--mode offline` identically, since the non-determinism is in the shared gradient step and
the kernels beneath it, not in anything mode-specific.

O1 is numerics-neutral by the same argument prompt 1 verified for its twin: it changes only
how the same jitted function is invoked and when the host blocks, and both modes now call
the identical `compile_and_run`. The snapshot fix changes printed text only.

The timing noise floor carried forward from prompt 1 is 0.3%, and this pass reproduces it:
`off_cl_u4` measured 273.155 s in one batch and `off_cl_r1`, the same configuration, 273.167
s in another, a difference of 0.004%.

### 9.2 The fairness block

Section 2, quoted from `print_config_snapshot` rather than asserted, and now pinned by
`test_offline_and_dagger_stay_compute_matched` across all four final configs.

### 9.3 Suite, lint and smoke

| | Baseline (`d1126fc`) | Prompt 1 end (`08aeb76`) | After this pass |
|---|---|---|---|
| `pytest tests -q` | 154 passed | 167 passed | 172 passed |
| `ruff check src tests experiments` | 12 errors | 12 errors | 12 errors |
| `python main.py --mode smoke` | OK | OK | OK |

The 12 lint errors are the same pre-existing ones (4 `B905`, 4 `I001`, 3 `E402`, 1
`UP037`), untouched. As prompt 1 recorded, `--mode smoke` needs the GPU idle: run it
between stages, not alongside a measurement.

## 10. What I did not do, and why

- **Did not run offline at a larger `num_envs` than DAgger.** Section 4.2 shows there is
  no headroom to use anyway, so the fairness constraint costs nothing here.
- **Did not run either config to completion.** 34.7 and 69.1 hours per seed. The run plan
  is built from measured marginal costs and resolved update counts.
- **Did not re-measure the W&B callback.** O2's instruction is to mirror prompt 1, which
  measured it below the noise floor on the same closure both modes use.
- **Did not touch the gradient step, the model, the sampler or the environment wrapper.**
  Prompt 1's, per `PERF_OFFLINE_PROMPT_CRAFTAX.md:128-130`. The one shared file this pass
  did edit is `print_config_snapshot`, which is not on that list and which this pass is
  required to quote correctly.
- **Did not change the return weighting** (`offline.py:255-262`). It is the BC objective,
  not a performance knob.
- **Did not resolve the Classic 3.3% training-scan discrepancy** in section 4. It is real
  and unexplained, it changes no conclusion, and chasing it into XLA's fusion decisions
  would cost more than the answer is worth to this pass.

## 11. Hand-off to prompt 3

Files touched:

| File | Change |
|---|---|
| `src/planners/offline.py` | `run_offline_diffusion` uses `compile_and_run` / `format_timing`; `import time` dropped. |
| `src/planners/common.py` | `print_config_snapshot` derives the minibatch from `dagger_sizing` and prints `samples_per_update` in both modes. No other function changed. |
| `tests/test_smoke_src.py` | 5 new tests. |

Nothing else. `src/planners/online.py`, `src/planners/env.py`, `src/diffusion/` and
`src/models/` are untouched by this pass.

For prompt 3:

- Per-update cost is 59.82 s (Classic offline, 384 envs) and 40.02 s (Craftax offline, 256
  envs), against DAgger's 61.84 s and 41.00 s. The two modes are within 3.4% of each other,
  so an ablation's cost can be estimated from either.
- 96 to 99% of an update is the training scan in both modes. Ablations that change the
  model, the minibatch or `update_epochs` will move the wall clock; ablations that change
  the environment, the sampler or the logging will not.
- Cold compile is 48 s (Classic offline) and 84 s (Craftax offline) per process, so
  `jax_compilation_cache_dir` from prompt 1 is worth setting for a suite of separate
  processes.
- The non-determinism in prompt 1 section 9.1 applies to every comparison the ablation
  suite makes at a single seed.
