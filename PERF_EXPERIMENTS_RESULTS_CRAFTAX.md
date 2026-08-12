# Craftax RL fine-tuning ablation suite on the 4070 Ti: measurements and run plan

Prompt 3 of 3 (`PERF_EXPERIMENTS_PROMPT_CRAFTAX.md`). Starts from prompt 2's HEAD
(`c6b7966`). Prompt 1's compile-versus-execute method, its compilation cache and its XLA
memory settings are inputs here. Read `PERF_DAGGER_RESULTS_CRAFTAX.md` and
`PERF_OFFLINE_RESULTS_CRAFTAX.md` first.

---

## 1. Box, settings, and what was run against what

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, 560.35.03

$ uv run --no-sync python -c "import jax;print(jax.__version__, jax.devices())"
0.11.0 [CudaDevice(id=0)]
```

| Setting | Value |
|---|---|
| CUDA extra | `cuda12`; the driver is 560.35.03 and CUDA 13 needs 580 (`README.md:32`). |
| `LD_LIBRARY_PATH` | `/opt/ucl/lib:/usr/X11R6/lib`, unchanged. No CUDA libraries in either, so nothing shadows the wheels. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | unset; preallocation stays on, as in prompts 1 and 2. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.90` for every run below, as prompts 1 and 2 settled. |

**Config and checkpoint pairs.**

| Variant | Ablation config | Pretrained checkpoint | PPO expert |
|---|---|---|---|
| Classic | `ablations_final_classic_ucl.yaml` | `checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M` (`resume_metadata.json`: `mode: online`, `update_step: 1525`) | `checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M` |
| full Craftax | `ablations_final_craftax_ucl.yaml` | **none exists**, see section 2 | `checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M` |

The Classic checkpoint's `update_step: 1525` matches the `NUM_UPDATES` that
`configs/final_classic_ucl.yaml` resolves to at its configured `num_envs: 512`
(`PERF_DAGGER_RESULTS_CRAFTAX.md` section 3), which is the evidence that this checkpoint is
the one that config produced.

## 2. Three structural defects, reported not fixed

### 2.1 There is no full-Craftax planner checkpoint

`ablations_final_craftax_ucl.yaml:3` says the config "Matches checkpoint produced by:
configs/final_craftax_full_ucl.yaml". **That file does not exist.** `configs/` contains
`final_craftax_ucl.yaml`, `final_craftax_qmul.yaml` and the `craftax_exp_*` series, and no
`final_craftax_full_ucl.yaml`. Reported, not renamed: which config was meant is the
author's to confirm, and a silent substitution would invalidate every Craftax ablation.

Worse, the checkpoint it refers to does not exist either. The full contents of the released
repo:

```
$ python -c "from huggingface_hub import list_repo_files; ..."
checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M/...
checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M/...
checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M/...
```

One planner checkpoint, Classic only, and the two PPO experts. `README.md:245-248` lists
four planner checkpoints, two offline BC and two online DAgger. Three of the four are not
published.

**The full-Craftax half of the ablation suite therefore cannot be run at all**, here or
anywhere, until that checkpoint exists. Producing it means the full-Craftax DAgger run,
which `PERF_DAGGER_RESULTS_CRAFTAX.md` section 8.2 returned a no-go on at 62 hours per
seed. This is a dependency, not a performance problem, and it is the single most important
thing in this report.

Section 4 still sizes the Craftax variant, using a structurally valid probe checkpoint
built at the Craftax observation dimension with randomly initialised weights. VRAM and
per-iteration cost depend on shapes, not values, so the sizing is real. None of it is a
result.

### 2.2 The Craftax ablation config names the wrong machine

`ablations_final_craftax_ucl.yaml:5` says "UCL RTX 3090 Ti (24 GB VRAM)" while
`configs/final_craftax_ucl.yaml:2` says "UCL RTX 4090 24 GB". Both are 24 GB and nothing
derives from it, so this is a provenance error rather than a functional one, but one of
them is wrong about which machine produced the checkpoint. `ablations_final_classic_ucl.yaml:5`
says "UCL RTX 3090 Ti (24 GB VRAM)" and `configs/final_classic_ucl.yaml:2` agrees, so the
Classic pair is consistent and the Craftax pair is not.

### 2.3 Carried forward from prompt 1

`use_optimistic_resets: false` at `ablations_final_craftax_ucl.yaml:11` is the same setting
prompt 1 measured and left alone: worth **1% on Classic and 5% on full Craftax**, against a
change to the reset distribution. Quoted rather than re-opened, per the prompt.

---

## 3. Phase 1: one ablation, measured

`run_ablations.py:687-707` already records per-seed wall clock into
`results.json["ablations"][name]["wall_clock_s"]`, and seeds run sequentially in one
process. That makes the compile split free and answers the "is the executable reused"
question by measurement rather than instrumentation: if it were not reused, seed 2 would
cost what seed 1 costs.

Classic, `ablations_final_classic_ucl.yaml` at its real `num_envs: 192` and
`batch_size: 1024`, `max_iter` cut to 60 and 110 so both runs pass an `eval_every: 25` and
a `cka_every: 50` boundary:

| Run | `max_iter` | Seed 1 | Seed 2 | Implied compile |
|---|---|---|---|---|
| `baseline_rl` | 60 | 231.2 s | 221.0 s | 10.2 s |
| `baseline_rl` | 110 | 334.9 s | 325.4 s | 9.5 s |
| `baseline_rl`, diagnostics never firing | 60 | 169.3 s | 157.3 s | 12.0 s |

```
per iteration (diagnostics amortised) : (325.4 - 221.0) / 50 = 2.088 s
per-seed fixed cost                   : 221.0 - 60 x 2.088   =  95.7 s
```

**The executable is reused across seeds.** Seed 2 costs 10 s less than seed 1 in every
pair, consistently, which is the compilation plus first-call warm-up that seed 1 pays and
seed 2 does not.

### Gate 2: compilation is 0.3% of the suite, not the largest item

This is the number the prompt says decides whether the compilation work is worth doing, and
the answer is that it is not. **Compiling an ablation takes about 10 seconds, not minutes.**
Across 25 ablations that is 250 s against a projected 23.8 hours of Classic suite (section
6), which is 0.3%.

The prompt's framing was "if a single ablation compiles in seconds, the compilation work is
pointless and you should say so". It compiles in seconds. Said.

That verdict is specific to this suite and does not contradict prompt 1: the DAgger
training graph takes 52 s to compile and the full Craftax one 92 s, five to nine times the
ablation graph, because those programs carry a replay buffer, an expert policy and a
diffusion sampler inside the scan.

### The diagnostics compile whether they fire or not, and that costs nothing

`training.py:1128-1196` puts the diagnostics in `jax.lax.cond` branches inside the scan, so
both sides are compiled into the program regardless of cadence. The prompt flags this as a
JAX-specific compile cost worth attributing. Attributed: it is not one.

Pushing every diagnostic cadence past `max_iter` leaves the conditionals in the graph and
only stops them firing. Compile time was **12.0 s with the diagnostics never firing against
10.2 s with them firing**, a difference inside the run-to-run spread of the same
measurement. The branches cost essentially nothing to compile.

### Gate 3: where execution time goes, and it is the diagnostics

Pushing every diagnostic cadence past `max_iter` leaves the graph identical and only stops
the branches firing, so the difference is their execution cost:

| `max_iter` | diagnostics firing | not firing | diagnostics |
|---|---|---|---|
| 60 | 221.0 s | 157.3 s | 63.7 s, 28.8% |
| 110 | 325.4 s | 187.2 s | 138.2 s, 42.5% |

```
marginal per iteration, diagnostics firing     : (325.4 - 221.0) / 50 = 2.088 s
marginal per iteration, diagnostics not firing : (187.2 - 157.3) / 50 = 0.598 s
diagnostics, at the margin                     :                        1.490 s, 71% of it
```

The share grows with run length because the diagnostics scale with iterations while the
per-seed setup does not. Projected to the configured `max_iter: 500`:

```
with diagnostics    : 221.0 + 440 x 2.088 = 1,139.7 s  (19.0 min)
without diagnostics : 157.3 + 440 x 0.598 =   420.4 s   (7.0 min)
diagnostics                               =   719.3 s, 63% of a production run
```

**At the configured run length the diagnostics are roughly two thirds of the suite.** That
is much larger than the 29% a short run suggests, and it is the single biggest number in
this report. It is also, per `PERF_EXPERIMENTS_PROMPT_CRAFTAX.md:163-165`, entirely out of
reach: "The diagnostics are the experiment. Reducing their cadence is not a speed-up
available to you, even though it is the largest one on the table."

Recorded, not touched. If the author ever wants the Classic suite to cost 9 hours instead
of 24, halving the cadences is where the time is, and the price is diagnostic resolution.
The 500-iteration figures are a projection from a measured marginal rate over 60 to 110
iterations, and are labelled as such; everything above them is measured.

The remaining third is the rollout and the gradient step. Prompts 1 and 2 both measured the
gradient step at 94 to 99% of an update in the training runners, and this pass did not
separate the two inside the ablation runner, because no change was available on either
side: the gradient step is what each ablation modifies, and the environment is prompt 1's.

### Cost varies by ablation

Not every ablation costs the same, so a projection from `baseline_rl` alone is a lower
bound. Classic, `max_iter: 60`, one seed, each including its own compile:

| Ablation | Wall clock | Against `baseline_rl` |
|---|---|---|
| `baseline_rl` | 231.2 s | - |
| `lora` | 237.6 s | +2.8% |
| `ewc` | 314.9 s | **+36.2%** |

`ewc` carries a Fisher-diagonal buffer the size of the parameter tree and 20 Fisher batches
at setup (`ewc_fisher_batches: 20`), so it is the expected outlier and the config comment at
`ablations_final_classic_ucl.yaml:43` already says as much. Three of 25 ablations were
sampled; section 6 flags the projection accordingly.

## 4. Gate 1: both variants fit at the configured sizes

`ablations_final_classic_ucl.yaml` sets `num_envs: 192` and `batch_size: 1024`;
`ablations_final_craftax_ucl.yaml` sets `num_envs: 128` and `batch_size: 1024` on an
observation six times wider. The prompt's concern is that these were sized for 24 GB.

Both run on 16 GB at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`, with no OOM, at the configured
sizes and with every diagnostic firing:

| Variant | `num_envs` | `batch_size` | 60 iterations, 2 seeds | Result |
|---|---|---|---|---|
| Classic | 192 | 1024 | 231.2 s, 221.0 s | runs |
| full Craftax | 128 | 1024 | 268.5 s, 257.7 s | runs |

**No resizing is needed and none was done.** This is the one gate in the three passes that
did not force a compromise: unlike the DAgger and offline configs, which need `num_envs`
cut from 512 to 384 and from 448 to 256, the ablation suite fits as shipped.

Two caveats on the measurement. Preallocation is on, so `nvidia-smi` reports the reserved
pool (14,842 MiB Classic, 14,922 MiB Craftax of 16,376 MiB) rather than the realised peak;
what it establishes is that the pool was enough, not how much of it was used. And the
Craftax figure was obtained with the probe checkpoint from section 2.1, so it is a sizing
result: shapes are the real ones, weights are not.

`ewc` was also run on both variants, since the config comment at
`ablations_final_classic_ucl.yaml:43` records that "192/2048 OOM'd on EWC at 19.4 GiB" and
EWC carries a parameter-sized Fisher buffer. It runs at the shipped `batch_size: 1024`.

**Full Craftax timings**, same method:

| Run | `max_iter` | Seed 1 | Seed 2 | Implied compile |
|---|---|---|---|---|
| `baseline_rl` | 60 | 268.5 s | 257.7 s | 10.8 s |
| `baseline_rl` | 110 | 376.2 s | 366.2 s | 10.0 s |

```
per iteration : (366.2 - 257.7) / 50 = 2.170 s   (Classic: 2.088 s)
```

Full Craftax costs 4% more per iteration than Classic despite a six-times-wider
observation, because both are dominated by the same gradient step at the same
`batch_size: 1024`, and the observation only widens the encoder's first layer.

## 5. The changes

### E1: bulk-transfer the metrics history

`metrics_to_history` bulk-transferred ten arrays at `training.py:1285-1294` and then the
loop below it undid the lesson: `float(jax.device_get(all_metrics.win_rate[i]))` and
seventeen more of the same shape, each slicing a device array and pulling one scalar across
per logged iteration. Hoisted all eighteen into the same bulk block. The values are
identical by construction: the same arrays, transferred whole instead of element by element.

**Worth: 1.8 s in a 60-iteration run, 0.8%**, which is at the edge of what this method
resolves.

| | Seed 1 | Seed 2 (no compile) |
|---|---|---|
| before | 231.2 s | 221.0 s |
| after | 235.0 s | 219.2 s |

Seed 2 is the fair comparison because seed 1 carries compilation. 1.8 s of 221.0 s. The
saving scales with the number of logged iterations, so at the configured `max_iter: 500` it
would be roughly 15 s of 1,140 s, about 1.3%. Kept: it is free, it is provably identical,
and eighteen fewer device syncs is the right shape of code regardless. Not kept because it
was worth much.

### E2: the compilation cache is not worth chasing here

Prompt 1 landed `jax_compilation_cache_dir` as an opt-in config key. This pass measured
what it could be worth to the suite and the answer is almost nothing: section 3 puts
compilation at **10 seconds per ablation, 0.3% of the suite**. Even a 100% hit rate across
all 25 ablations saves 250 s of an 85,700 s run.

The cache is still worth setting, because it costs nothing and the saving is real, but it
is not a reason to do anything. The prompt's premise, that compilation across 150 runs is
"the largest single item in the budget", does not survive the measurement. Reported rather
than acted on.

### E3: vmapping the seed loop, dropped

The rationale in `PERF_EXPERIMENTS_PROMPT_CRAFTAX.md:133-136` is that vmapping three seeds
"would compile once instead of three times and would fill the GPU better". Both halves fail
on measurement:

- **It already compiles once.** Section 3 shows the executable is reused across seeds
  inside one process: seed 2 costs 10 s less than seed 1, every time. There is no third
  compilation to save, and the one that exists is 0.3% of the suite.
- **It would not fill the GPU better.** Prompt 1 measured exactly this on the DAgger runner
  (`PERF_DAGGER_RESULTS_CRAFTAX.md` section 8.1): three vmapped seeds cost **2.95x** one
  seed, against 3x for three sequential runs. The card is already saturated by one replica.
  A 1.7% wall-clock gain for 3x the working set.

Dropped without implementing it, so the seed-equivalence check the prompt requires for E3
was not needed. Given that prompt 1 measured the memory multiplier at 2.3x for two replicas
and 3.8x for three, and that section 4 shows the suite already reserves 14.9 GB of a 16.4 GB
card, it would not have fitted either.

### E4: what the profile found

Section 3's gate 3: the diagnostics are 63% of a production-length run. Pinned by
`PERF_EXPERIMENTS_PROMPT_CRAFTAX.md:163-165` and left alone. Nothing else in the profile is
addressable from this pass: the gradient step is what each ablation modifies, and the
environment and sampler are prompt 1's.

## 6. Phase 5: the run plan

| Quantity | Classic | full Craftax |
|---|---|---|
| compile per ablation, before / after | 10.2 s / 10.2 s | 10.8 s / 10.8 s |
| execute per ablation-seed at `max_iter: 500` | 1,140 s | 1,213 s |
| minutes per ablation-seed | 19.0 | 20.2 |
| hours for 25 ablations x 3 seeds | **23.8** | **25.3** |
| peak VRAM | pool 14,842 MiB reserved of 16,376 | pool 14,922 MiB of 16,376 |
| does the configured size fit | **yes**, unchanged | **yes**, unchanged |

```
Classic : 25 x (3 x 1,140 s + 10 s) = 85,728 s = 23.8 h
Craftax : 25 x (3 x 1,213 s + 11 s) = 91,213 s = 25.3 h
                                       total   = 49.1 h
```

E1 changes none of these figures at three significant figures.

**These are lower bounds.** They project from `baseline_rl`, and ablations differ: `ewc` is
36% more expensive and `lora` 3%. Three of 25 were sampled. If the suite's mean is 10 to
15% above `baseline_rl`, the real totals are nearer 27 and 29 hours.

**The two variants must run sequentially, not concurrently.** Each reserves about 14.9 GB
of a 16.4 GB card at `MEM_FRACTION=0.90`, so two processes cannot coexist; and prompt 1
section 9.3 records that a second GPU process makes XLA's Triton autotuner fail during
compilation rather than merely running slowly.

### Go / no-go

**Classic suite: go.** 23.8 to 27 hours, one process at a time, `MEM_FRACTION=0.90`, sizes
as shipped. It needs the Classic DAgger checkpoint, which exists and is published.

**Full Craftax suite: no-go, and not for a performance reason.** It would cost 25.3 to 29
hours, which is affordable, and it fits at the configured sizes. It cannot run because
**the pretrained checkpoint it starts from does not exist** (section 2.1). Producing one
means the full-Craftax DAgger run that
`PERF_DAGGER_RESULTS_CRAFTAX.md` section 8.2 already returned a no-go on, at 62 hours per
seed. The dependency chain is 62 hours of DAgger before 25 hours of ablations, and the
first of those two is the decision.

The 49.1 hours here are on top of the 22 to 26 days prompt 2 projected for the paper's
training matrix. Nothing in this pass reduces either, because the suite is already
structurally good: one `jax.lax.scan` under one `jax.jit`, an executable reused across
seeds, compilation at 0.3%, and two thirds of the remainder spent on diagnostics that are
the experiment.

## 7. Verification

### 7.1 Control band

Prompt 1 section 9.1 established that two runs of identical code at one seed do not agree
on this GPU, because XLA's GPU backend is non-deterministic by default, and that
`--xla_gpu_deterministic_ops=true` makes them agree exactly. That applies here: any
single-seed before-and-after comparison of an ablation's loss series is reading noise
unless that flag is set. The suite's own defence is `num_seeds: 3`, which is why it is
pinned.

For E1 specifically no such comparison is needed. The change moves eighteen
`jax.device_get` calls from inside a Python loop to a block above it, transferring the same
arrays; the values are identical by construction, not by measurement.

### 7.2 Timing control

`baseline_rl` at 60 iterations, seed 2, measured 221.0 s before E1 and 219.2 s after, and
`cl_baseline60`, an earlier single-seed run of the same configuration, measured 233.2 s
against 231.2 s. Run-to-run spread on this suite is around 1%, which is why section 5 does
not claim E1's 0.8% as a resolved effect.

### 7.3 Suite, lint and the fast smoke path

| | Baseline (`d1126fc`) | Prompt 2 end (`c6b7966`) | After this pass |
|---|---|---|---|
| `pytest tests -q` | 154 passed | 172 passed | 172 passed |
| `ruff check src tests experiments` | 12 errors | 12 errors | 12 errors |
| `python main.py --mode smoke` | OK | OK | OK |

## 8. What I did not do, and why

- **Did not touch any diagnostic cadence.** It is the largest speed-up available, at 63% of
  a production run, and it is pinned: the diagnostics are the experiment.
- **Did not implement E3.** Dropped on prompt 1's measurement that vmapped seeds cost 2.95x
  for 3x the memory, and on this pass's measurement that the executable is already reused
  across seeds.
- **Did not resize the ablation configs.** They fit. This is the one gate that did not
  force a compromise.
- **Did not rename `final_craftax_full_ucl.yaml`** in `ablations_final_craftax_ucl.yaml:3`,
  or correct the machine name at line 5. Which config and which machine produced the
  checkpoint is the author's to state, and guessing would be worse than the current wrong
  comment.
- **Did not train a full-Craftax planner checkpoint** to unblock that half of the suite.
  That is a 62-hour DAgger run and a decision prompt 1 already put to the author.
- **Did not separate the rollout from the gradient step inside the ablation runner.** No
  change was available on either side, and prompts 1 and 2 already measured the same split
  in the training runners.
- **Did not touch `common.py`, the sampler, the model or the environment wrapper.**
  Prompt 1's, per `PERF_EXPERIMENTS_PROMPT_CRAFTAX.md:146-148`.
- **Did not run a full 500-iteration ablation end to end.** At 19 minutes per seed it was
  affordable, but the projection it would validate is built from a marginal rate measured
  over 60 and 110 iterations of the same scan, and the remaining risk is linearity, which a
  single longer run at one length would not settle either.

## 9. Files touched

| File | Change |
|---|---|
| `experiments/rl_finetuning/ablations/training.py` | `metrics_to_history` bulk-transfers eighteen more arrays instead of slicing them per iteration. No other change. |

Nothing in `src/`, no config, no test. The three config defects in section 2 are reported,
not edited.
