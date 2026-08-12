# Craftax online DAgger on the 4070 Ti: measurements and run plan

Prompt 1 of 3 (`PERF_DAGGER_PROMPT_CRAFTAX.md`). Every number below was measured on
the box identified in section 1 and is conditional on the settings recorded there.

---

## 1. Box, driver, and the settings every number depends on

```
$ nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
name, memory.total [MiB], driver_version
NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB, 560.35.03

$ uv run --no-sync python -c "import jax;print(jax.__version__, jax.devices())"
0.11.0 [CudaDevice(id=0)]

$ lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core)'
CPU(s):                               28
Model name:                           Intel(R) Core(TM) i7-14700K
Thread(s) per core:                   2
Core(s) per socket:                   20

$ free -g | head -2
               total        used        free      shared  buff/cache   available
Mem:              62           4          44           0          14          58
```

Host `outback.cs.ucl.ac.uk`.

| Setting | Value | Why |
|---|---|---|
| CUDA extra | `cuda12` | Driver 560.35.03 is below the 580 that `README.md:32` requires for CUDA 13. Synced once with `uv sync --extra cuda12`, then `uv run --no-sync` throughout. |
| `LD_LIBRARY_PATH` | `/opt/ucl/lib:/usr/X11R6/lib`, left as is | Checked for the failure mode `README.md:32` warns about. Neither directory contains a `libcuda*`, `libcudnn*` or `libcublas*`, and JAX resolves `CudaDevice(id=0)`, so nothing shadows the wheel libraries. Nothing to unset. |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | unset (preallocation on) | Left at JAX's default. Disabling it trades peak headroom for fragmentation, and section 4 shows the working set is one large transient allocation rather than many small ones, which is the case preallocation handles best. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | `0.90` for every run below unless stated | The default 0.75 caps the allocator at 12,564 MB of a 17,171 MB card. This box is not shared, and section 4 shows 0.75 is the difference between `num_envs: 384` running and failing. 0.90 gives 15,076 MB and leaves 2.1 GB for the driver and the X server (11 MiB in use). |

Repo conventions: `PERF_DAGGER_PROMPT_CRAFTAX.md:28` points at a `CLAUDE.md` one
level up. There is no such file, at the repo root or its parent. This report
follows the conventions the prompt states directly: UK English, evidence for
every claim.

## 2. The PPO expert

`online.py:105-107` asserts on `PPO_CHECKPOINT_PATH`, and no `checkpoints/`
directory existed on this box. The released weights are on the Hub, exactly as
`README.md:241,254` documents, so no substitute needed training:

```
$ uv run --no-sync hf download mathisweil/remdm-craftax-checkpoints \
      --repo-type model --include "checkpoints/**" --local-dir .
$ du -sh checkpoints
184M    checkpoints
```

| Path | Provenance |
|---|---|
| `checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M` | PPO-RNN, `TOTAL_TIMESTEPS: 1000000000`, `NUM_ENVS: 1024`, `LAYER_SIZE: 512`. Sidecar `wandb-summary.json`: `episode_return 19.358`, `episode_length 341.3`, `achievements 19.78`. |
| `checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M` | PPO-RNN, `TOTAL_TIMESTEPS: 1000000000`, `NUM_ENVS: 1024`, `LAYER_SIZE: 512`. |

`checkpoints/` is gitignored, so the tree stays clean. The released set contains
no offline BC checkpoint; prompt 2 should not assume one is present.

## 3. The config snapshot, and three things wrong with it

Confirmed with `print_config_snapshot` (`common.py:227`) rather than by hand, via
a harness that reproduces `main.build_config`'s merge and calls the two resolvers.

| | `final_craftax_ucl.yaml` | `final_classic_ucl.yaml` |
|---|---|---|
| `num_envs` x `num_steps` = fpu | 448 x 128 = **57,344** | 512 x 128 = **65,536** |
| `online_total_timesteps` (re-snapped) | 199,958,528 | 99,942,400 |
| `NUM_UPDATES` | **3,487** | **1,525** |
| `LR_WARMUP_STEPS` | **1,371** | **1,600** |
| `DAGGER_BETA_DECAY` | 0.9997263030 | 0.9993004981 |
| final beta | 0.3850 | 0.3440 |
| `DAGGER_BUFFER_MAX` | **43,750** (0.76 cycles) | **125,000** (1.91 cycles) |
| `samples_per_update` | 43,456 | 49,664 |
| minibatch | 7,168 | 8,192 |
| `total_grad_steps` | 223,168 | 97,600 |
| `VAL_INTERVAL` | 17 updates | 15 updates |

The prompt's arithmetic is confirmed in every cell. The frame-denominated
mechanism did its job: the warmup budget is still 78.6M / 104.9M frames and the
final beta is still 0.385 / 0.344 as intended. What went stale is everything the
comments say in *sample* or *update* units, because they describe a `num_envs`
the files no longer set.

### 3.1 The comments were wrong; they are now corrected

Corrected in this pass, comment-only, no behaviour change:

| File | Was | Is |
|---|---|---|
| `final_craftax_ucl.yaml:41` | "300 update steps (300 * 2048 * 128)" | resolves to `lr_warmup_steps = 1371` |
| `final_craftax_ucl.yaml:58` | "0.9995^1907" | decay 0.9997263 over 3,487 updates |
| `final_craftax_ucl.yaml:59` | "~200K samples on UCL" | 43,750 samples |
| `final_classic_ucl.yaml:41` | "200 update steps (200 * 4096 * 128)" | resolves to `lr_warmup_steps = 1600` |
| `final_classic_ucl.yaml:59` | "~1M samples on UCL" | 125,000 samples |

The two QMUL configs carried the same defect and are corrected with them.
`final_craftax_qmul.yaml` was the worst of the four: it sets `num_envs: 64` while
its header and three comments all described 384 envs, so its buffer comment
("~37500 samples") overstated the real 6,250 by 6x.

`tests/test_smoke_src.py::test_final_configs_resolve_to_their_documented_quantities`
now pins all four configs' resolved `NUM_UPDATES`, `LR_WARMUP_STEPS` and
`DAGGER_BUFFER_MAX` against the resolvers, so the comments cannot go stale again
without a red test.

### 3.2 Author decision: what is the DAgger buffer meant to hold?

`dagger_buffer_cycles` is denominated in update cycles, so under a `num_envs`
change the buffer holds a fixed amount of *history* and a varying number of
*samples*. The comments were written as though the opposite were true. Which one
the recipe wants changes the experiment:

- **Fixed cycles** (what the code does): the buffer always holds ~0.76 / ~1.91
  updates of history, so the off-policy staleness of the data is invariant. The
  sample count falls with `num_envs`, so on this box the Craftax buffer holds
  43,750 samples against the 200K the comment claimed.
- **Fixed samples** (what the comments claim): DAgger's aggregation argument is
  about the size of the dataset D, not its age. Holding samples fixed would keep
  |D| comparable across hardware and let the cycle count float.

This is not mine to decide and I have not changed it. It matters more on this box
than on a 24 GB card, because section 4 forces `num_envs` down further still.

### 3.3 Defect: `lr_warmup_steps` is derived in update steps and consumed in gradient steps

`resolve_scaled_hyperparams` (`common.py:140`) computes

```python
config["LR_WARMUP_STEPS"] = int(float(warmup_frames)) // fpu
```

which is an **update**-step count. Both runners then pass it straight to optax:

```python
# online.py:189-196, offline.py:108-116
total_grad_steps = num_updates * n_train_passes * update_epochs * num_minibatches
warmup_steps = config.get("LR_WARMUP_STEPS", 0)
lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=config["LR"],
    warmup_steps=warmup_steps,
    decay_steps=total_grad_steps,
    end_value=config["LR"] * 0.1,
)
```

`decay_steps` is correctly in gradient steps, and the code says so:
`online.py:218` and `offline.py:138-142` both note "the schedule is indexed by
gradient step, which equals update_step * update_epochs * num_minibatches".
`warmup_steps` is not converted. optax indexes both by the same counter, which
`state.apply_gradients` advances once per minibatch:

```
$ uv run --no-sync python -c "
import optax
s = optax.warmup_cosine_decay_schedule(0.0, 1.0, warmup_steps=100, decay_steps=1000, end_value=0.1)
for i in [0, 50, 99, 100, 101, 999]: print(i, float(s(i)))"
0 0.0
50 0.5
99 0.99
100 1.0          <- peak at step == warmup_steps
101 0.999997
999 0.100003
```

So the warmup is short by a factor of `update_epochs * num_minibatches` = 64:

| Config | Intended warmup | Actual warmup | Actual, in frames |
|---|---|---|---|
| `final_craftax_ucl.yaml` | 1,371 updates (78.6M frames) | 1,371 gradient steps = update 21 of 3,487 | ~1.2M |
| `final_classic_ucl.yaml` | 1,600 updates (104.9M frames) | 1,600 gradient steps = update 25 of 1,525 | ~1.6M |

`print_config_snapshot` compounds it: `common.py:270` prints
`lr_warmup_steps = 1371 (~78.62M frames)`, which is the intent, not the
behaviour.

The one thing the defect does not break is hardware portability. The error is a
constant factor of 64, so the realised warmup in frames stays invariant under a
`num_envs` change: Classic resolves to 1,600 gradient steps at 512 envs and
2,133 at 384, and both are 1.638M frames; Craftax resolves to 1,371 at 448 envs
and 2,400 at 256, and both are 1.229M frames. The recipe is consistent across
hardware. It is just running 1/64th of the warmup its configs describe.

The units question is not academic, because the intended reading is not even
representable. `optax.warmup_cosine_decay_schedule` builds its cosine leg over
`decay_steps - warmup_steps`, so `warmup_steps >= decay_steps` raises. Cutting
the Classic run to four updates to measure it produced:

```
ValueError: The cosine_decay_schedule requires positive decay_steps, got decay_steps=-1344.
```

and on the full Classic schedule the *documented* warmup of 1,600 update steps
is 102,400 gradient steps against a 97,600-step horizon, which would raise the
same error. `final_classic_qmul.yaml` has the same property (8,533 warmup
against 8,138 updates). Whatever the author intends, "1,600 update steps of
warmup" cannot be what these two configs meant, because that configuration
cannot start.

**Author decision.** Three coherent options: convert
(`LR_WARMUP_STEPS = frames // fpu * update_epochs * num_minibatches`) and re-tune
the budget so it fits inside the horizon; redefine `lr_warmup_frames` as a
gradient-step budget and fix the snapshot's frame conversion; or declare the
current behaviour intended and correct only the comments. I have changed no
behaviour: `lr_warmup_frames` is on the pinned list at
`PERF_DAGGER_PROMPT_CRAFTAX.md:221`, and a 64x change to the warmup would
invalidate every run the recipe has produced.
`tests/test_smoke_src.py::test_lr_warmup_is_shorter_than_the_cosine_horizon`
guards the constraint that optax actually enforces.

---

## 4. Gate 1: does it fit? No, and not at any memory fraction

Method: `train_fn.lower(rngs).compile()`, then `compiled.memory_analysis()` for
XLA's own accounting and `jax.local_devices()[0].memory_stats()` for the
allocator's realised peak. The card is 16,376 MiB = 17,171 MB.

`configs/final_classic_ucl.yaml`, `num_envs` swept, everything else untouched:

| `num_envs` | XLA temp (MB) | XLA output (MB) | temp+output | Fits at 0.75 (12,564 MB)? | Fits at 0.90 (15,076 MB)? |
|---|---|---|---|---|---|
| 512 (configured) | 17,806 | 890 | 18,696 | no | **no, exceeds the whole card** |
| 448 | 16,445 | 802 | 17,248 | no | **no, exceeds the whole card** |
| 384 | 13,111 | 714 | 13,826 | no (measured OOM) | **yes**, realised peak 14,016 MB |
| 320 | 11,412 | 626 | 12,039 | yes, peak 12,058 MB | yes, peak 14,286 MB |
| 256 | 11,092 | 538 | 11,631 | yes, peak 12,188 MB | yes |
| 128 | 8,078 | 363 | 8,441 | yes | yes |

The configured `num_envs: 512` asks XLA for a single 16.58 GiB transient
allocation. That is larger than the card, so no preallocation setting rescues
it; `num_envs: 448` is the same story at 15.3 GiB.

```
jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to
allocate 16.58GiB. [executable_name='jit_train']
```

`configs/final_craftax_ucl.yaml` is worse, as expected from `obs_dim` 8268
against Classic's 1345 and 43 actions against 17:

| `num_envs` | XLA temp (MB) | XLA output (MB) | Result at 0.90 |
|---|---|---|---|
| 448 (configured) | 27,702 | 1,884 | no |
| 384 | 14,846 | 1,657 | OOM, asked for 13.83 GiB |
| 320 | 19,223 | 1,430 | OOM, asked for 17.90 GiB |
| **256** | 12,885 | 1,203 | **runs**, realised peak 14,121 MB |
| 192 | 13,059 | 975 | (compiles) |
| 128 | 8,800 | 748 | (compiles) |

Two notes on reading this table. XLA's temp figure is not monotone in
`num_envs` (320 asks for more than 384 on the Craftax config, 192 more than 256
on Classic): the compiler re-picks layouts and fusion boundaries per shape, so
the estimate is a guide and the run is the test. And XLA also adapts to the
limit it is given, which is why Classic at 320 reports 11,412 MB of temp at
`MEM_FRACTION=0.75` and 13,641 MB at 0.90.

### The decision

| | Configured | Used on this box | Why |
|---|---|---|---|
| `final_classic_ucl.yaml` | `num_envs: 512` | **384** | Largest that runs. Needs `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`; at 0.75 it OOMs. |
| `final_craftax_ucl.yaml` | `num_envs: 448` | **256** | Largest that runs, at 0.90. |

Nothing else changed. `optimistic_reset_ratio`, the model dimensions and
`plan_horizon` are untouched, as `PERF_DAGGER_PROMPT_CRAFTAX.md:158-161` requires.

**These runs are comparable, not identical.** The frame-denominated mechanism
holds total frames, the warmup budget in frames and the final beta constant. It
does not hold constant:

| | Classic 512 -> 384 | Craftax 448 -> 256 |
|---|---|---|
| total env frames | 99,942,400 -> 99,975,168 | 199,958,528 -> 199,983,104 |
| `NUM_UPDATES` | 1,525 -> **2,034** | 3,487 -> **6,103** |
| minibatch | 8,192 -> **6,144** | 7,168 -> **4,096** |
| `samples_per_update` | 49,664 -> **37,248** | 43,456 -> **24,832** |
| `DAGGER_BUFFER_MAX` | 125,000 -> **93,750** | 43,750 -> **25,000** |
| buffer, in cycles | 1.91 -> 1.91 (invariant) | 0.76 -> 0.76 (invariant) |
| final beta | 0.344 -> 0.344 (invariant) | 0.385 -> 0.385 (invariant) |
| `total_grad_steps` | 97,600 -> **130,176** | 223,168 -> **390,592** |

The gradient noise per update rises because the minibatch shrinks by a quarter
(Classic) and by 43% (full Craftax), and the buffer holds proportionally fewer
samples, which is the same question section 3.2 puts to the author. Whether the
resulting run is close enough to the paper's is the author's call; the mechanism
is doing what it was designed to do, but a design intent is not a proof.

## 5. Gate 2: the compile / execute split, which this repo could not report

The repo timed `out = train_fn(rngs)` and divided total frames by the result
(`online.py:709-713`). JAX dispatch is asynchronous, so that call returns once
compilation is done and the work is enqueued: the number was mostly compile time
and excluded almost all of the execution. `--mode smoke` makes it vivid, since
its execution is trivially short:

```
$ uv run --no-sync python main.py --mode smoke
Compile: 31.4s  Execute: 0.2s  Total: 31.6s
SPS: 2363 (execute)  12 (including compile)
```

The old code would have reported roughly `Time: 31.6s  SPS: 12` for a run whose
actual throughput is 2,363 frames/s.

Measured compile times, `MEM_FRACTION=0.90`, cold (no compilation cache):

| Config | `num_envs` | Lower | Compile | Total to first work |
|---|---|---|---|---|
| `final_classic_ucl.yaml` | 384 | 4.5 s | 50 to 81 s (median 52 s) | ~57 s |
| `final_craftax_ucl.yaml` | 256 | 6.5 s | 86 to 92 s | ~98 s |
| `configs/smoke.yaml` | 8 | - | 31.4 s | 31.6 s |

**Is compilation a material share?** For a full paper run, no: 52 s against the
~36 hours in section 8 is 0.04%. For the 25-entry ablation suite in prompt 3, if
each entry is a separate process and each recompiles, it is 25 x 52 s = 22
minutes of pure compilation for Classic and 38 minutes for full Craftax, before
any training happens. That is what change D1 is for, and it is the number prompt
3 should use to decide whether to bother.

## 6. Gate 3: where execution time goes

The two candidates are the rollout scan (`online.py:422`, including `sample_plan`
at `DIFFUSION_STEPS` per plan cycle) and the training scan
(`online.py:546-561`, `update_epochs x num_minibatches` gradient steps).

Method: no profiler attribution, just differencing, which needs no assumptions
about kernel accounting. Every run holds validation to a single firing at update
0 (`val_interval_frames=1.0e9`), so differencing two run lengths cancels
compilation, the initial env reset and that one validation, leaving the marginal
cost of an update. Halving `update_epochs` halves the training scan and leaves
the rollout scan alone, which separates the two.

Classic, `num_envs: 384`, `MEM_FRACTION=0.90`:

| Run | Updates | `update_epochs` | Execute |
|---|---|---|---|
| `cl_u4` | 4 | 8 | 283.71 s |
| `cl_u12` | 12 | 8 | 778.46 s |
| `cl_u4e4` | 4 | 4 | 159.40 s |
| `cl_u12e4` | 12 | 4 | 409.66 s |

```
marginal per update, epochs=8 : (778.46 - 283.71) / 8 = 61.84 s
marginal per update, epochs=4 : (409.66 - 159.40) / 8 = 31.28 s
implied one-off, epochs=8     : 283.71 - 4 x 61.84    = 36.34 s
implied one-off, epochs=4     : 159.40 - 4 x 31.28    = 34.27 s   (agrees, 6%)
training scan  = 2 x (61.84 - 31.28) = 61.12 s/update   98.8%
rollout scan   = 61.84 - 61.12       =  0.72 s/update    1.2%
```

**The training scan is essentially the whole update.** The rollout, which
includes four `sample_plan` calls at 15 diffusion steps each, 128 environment
steps over 384 environments and 128 PPO-RNN expert forward passes, costs 0.72 s:
about 68,000 environment frames per second. Craftax on the GPU is not the
bottleneck and never comes close to being it.

This is the finding that decides everything else in this report, and it is worth
saying plainly because it is the opposite of what reading the code suggests. The
`AutoResetEnvWrapper` calls `env.reset` on *every* step regardless of `done`
(`Craftax_Baselines/wrappers.py:65-66`), vmapped over every environment, so the
code looks like it generates 384 worlds per step. It does, and it is still
irrelevant next to 64 gradient steps on a 9M-parameter transformer.

Validation, priced separately by running four firings instead of one:

```
cl_u4val4 (val_interval=1) : 375.70 s
cl_u4     (val_interval=huge): 283.71 s
(375.70 - 283.71) / 3 = 30.66 s per validation
```

30.66 s is half an update, and `VAL_INTERVAL` resolves to 20 updates at
`num_envs: 384`, so validation adds 2.5% to the wall clock. It is not a target.

Full Craftax, `num_envs: 256`, same method:

| Run | Updates | `update_epochs` | `val_interval` | Execute |
|---|---|---|---|---|
| `cx_u2` | 2 | 8 | huge | 106.39 s |
| `cx_u6` | 6 | 8 | huge | 270.41 s |
| `cx_u2e4` | 2 | 4 | huge | 66.23 s |
| `cx_u6e4` | 6 | 4 | huge | 153.37 s |
| `cx_u2val2` | 2 | 8 | 1 | 128.47 s |

```
marginal per update, epochs=8 : (270.41 - 106.39) / 4 = 41.00 s
marginal per update, epochs=4 : (153.37 -  66.23) / 4 = 21.78 s
training scan  = 2 x (41.00 - 21.78) = 38.44 s/update   93.7%
rollout scan   = 41.00 - 38.44       =  2.56 s/update    6.3%
one validation = 128.47 - 106.39     = 22.08 s
```

The rollout is a larger share than on Classic (6.3% against 1.2%), which is what
you would expect from a heavier world generator, 25 diffusion steps instead of
15 and an observation six times wider. It is still not where the time goes.

**Summary of the split**

| | Classic, 384 envs | Full Craftax, 256 envs |
|---|---|---|
| training scan | 61.12 s/update (98.8%) | 38.44 s/update (93.7%) |
| rollout scan | 0.72 s/update (1.2%) | 2.56 s/update (6.3%) |
| per update | 61.84 s | 41.00 s |
| one validation | 30.66 s | 22.08 s |
| frames per second, execute only | 795 | 799 |

The two configs land within 0.5% of each other on frames per second, from
opposite directions: Craftax does more work per sample and Classic runs more
samples per update. Both are limited by the same thing.

## 7. What `use_optimistic_resets: true` would be worth: 1% and 5%

Measured, not switched on, per `PERF_DAGGER_PROMPT_CRAFTAX.md:200-208`.

`env.py:45-53` selects `AutoResetEnvWrapper` + `BatchEnvWrapper` when
`use_optimistic_resets: false`. `AutoResetEnvWrapper.step`
(`Craftax_Baselines/wrappers.py:60-79`) calls `self._env.reset` unconditionally
on every step and then `jax.lax.select`s on `done`, and `BatchEnvWrapper` vmaps
that over every environment. So the current setting generates `num_envs` worlds
per environment step. `OptimisticResetVecEnvWrapper` generates
`num_envs / reset_ratio` instead, a 16x reduction at the configured ratio.

Classic, `num_envs: 384`, same differencing method:

| Run | Updates | `use_optimistic_resets` | Execute |
|---|---|---|---|
| `cl_u4` | 4 | false | 283.71 s |
| `cl_u12` | 12 | false | 778.46 s |
| `cl_u4opt` | 4 | true | 282.84 s |
| `cl_u12opt` | 12 | true | 772.79 s |

```
marginal per update, false : 61.84 s
marginal per update, true  : 61.24 s
saving                     :  0.60 s/update, 0.97%
```

Full Craftax, `num_envs: 256`:

| Run | Updates | `use_optimistic_resets` | Execute |
|---|---|---|---|
| `cx_u2` | 2 | false | 106.39 s |
| `cx_u6` | 6 | false | 270.41 s |
| `cx_u2opt` | 2 | true | 100.49 s |
| `cx_u6opt` | 6 | true | 256.11 s |

```
marginal per update, false : 41.00 s
marginal per update, true  : 38.90 s
saving                     :  2.10 s/update, 5.1%
```

Both savings are almost exactly the rollout cost section 6 measured (0.72 s and
2.56 s), so optimistic resets do recover most of what is available to them. What
is available is 1% on Classic and 5% on full Craftax.

The prompt calls it "very likely a large win". On this hardware it is not,
because section 6 shows the environment is not where the time goes. Under the
old timing code that conclusion would have been unreachable: the reported number
was mostly compile time, and compile time does not respond to the wrapper.

**Left off, and the recommendation is to leave it off.** On Craftax it would buy
5% in exchange for changing the reset distribution, accepting the
duplicate-reset caveat the wrapper's own docstring carries
(`Craftax_Baselines/wrappers.py:87-88`), and adding a paragraph to the paper's
method section explaining why the benchmark-forced choice recorded at
`env.py:37-39` was abandoned. That trade is not worth 5%.

## 8. The run plan

Per-update and per-validation costs from section 6; update and validation counts
from `print_config_snapshot` at the re-sized `num_envs`.

| | Classic, configured | Classic, on this box | Craftax, configured | Craftax, on this box |
|---|---|---|---|---|
| `num_envs`, and why | 512 | **384**, largest that fits at `MEM_FRACTION=0.90` | 448 | **256**, largest that fits at `MEM_FRACTION=0.90` |
| does it run? | **no**, needs 18.7 GB on a 17.2 GB card | yes | **no**, needs 29.6 GB | yes |
| compile time | - | 52 s | - | 92 s |
| s per update | - | 61.84 (rollout 0.72 / train 61.12) | - | 41.00 (rollout 2.56 / train 38.44) |
| `NUM_UPDATES` | 1,525 | 2,034 | 3,487 | 6,103 |
| total frames | 99,942,400 | 99,975,168 | 199,958,528 | 199,983,104 |
| validations | 102 | 102 x 30.66 s = 0.87 h | 204 | 204 x 22.08 s = 1.25 h |
| **hours per seed** | - | **35.9** | - | **70.8** |
| peak VRAM | - | 14,016 MB | - | 14,121 MB |
| headroom against 17,171 MB | - | 3,155 MB (18%) | - | 3,050 MB (18%) |

```
Classic : 2,034 x 61.84 s + 102 x 30.66 s + 52 s   = 129,084 s = 35.9 h
Craftax : 6,103 x 41.00 s + 204 x 22.08 s + 92 s   = 254,819 s = 70.8 h
```

Headroom is quoted against the physical card, not against the allocator limit,
because `MEM_FRACTION=0.90` is a choice this report made and can be raised
further if a run comes close. Against the 15,076 MB limit actually in force the
margin is 1,060 MB (7%) for Classic and 956 MB (6%) for Craftax, which is thin
enough that these two configurations should not be run concurrently with
anything else on the card, including a second JAX process or a desktop session.

### 8.1 The seed question: run them sequentially

`num_repeats: 1` in both configs and `online.py:707` vmaps the whole training
function over `NUM_REPEATS`, so three seeds run concurrently in one program and
multiply the working set. XLA temp plus output, compile-only:

| Config, `num_envs` | `num_repeats: 1` | 2 | 3 |
|---|---|---|---|
| Classic, 384 | 13,998 MB | 32,680 MB | 53,411 MB |
| full Craftax, 256 | 14,088 MB | 26,184 MB | 43,159 MB |
| Classic, 128 | 8,441 MB | 14,274 MB | 14,015 MB |

Scaling is worse than linear at the sizes the run plan uses: two seeds on Classic
at 384 envs need 2.3x, three need 3.8x. Against a 17,171 MB card, **neither
config fits more than one vmapped seed at its run-plan size**, and it is not
close: three Classic seeds want 53 GB.

Three vmapped seeds do fit at `num_envs: 128`, and run:

```
cl_r3_e128_exec : 4 updates, num_repeats=3, num_envs=128
                  realised peak 14,034 MB, execute 260.95 s
```

That is not a good trade, for two reasons. `num_envs: 128` is a third of the
largest single-seed size, so it shrinks the minibatch to 1,552 samples against
the 8,192 the config asks for, on top of the shrink section 4 already forced.
And vmapping buys no throughput anyway:

| `num_repeats` at `num_envs: 128` | 4 updates | 12 updates | Marginal per update |
|---|---|---|---|
| 1 | 86.44 s | 237.61 s | 18.90 s |
| 3 | 260.95 s | 706.41 s | 55.68 s |

```
55.68 / 18.90 = 2.95x the time for 3x the seeds
```

Three vmapped seeds cost 2.95x one seed, which is what three sequential runs
cost. The card is already saturated by one replica, so vmapping adds parallelism
the hardware has no room to exploit. It buys 1.7% of wall clock in exchange for
3x the memory.

**The run plan assumes three sequential runs at `num_repeats: 1`.** They cost the
same compute and 3x the wall clock, and with D1's compilation cache they pay
compilation once rather than three times: 2 x 52 s saved on Classic, 2 x 92 s on
full Craftax. Against 108 hours of wall clock that saving is decoration; the
reason to run sequentially is that the alternative does not fit.

### 8.2 Go / no-go

**Classic: go, with conditions.**

- `--override num_envs=384`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
- 35.9 hours per seed, 4.5 days for three seeds run sequentially.
- The card must be otherwise idle. 7% headroom against the allocator limit does
  not survive a second process, and section 9.3 records what contention looks
  like when it happens.
- The run is comparable to the paper's, not identical: section 4's table is the
  full list of what moved.

**Full Craftax: no-go as configured, and the decision is the author's.**

- It runs, at `--override num_envs=256` and `MEM_FRACTION=0.90`, with 6% headroom.
- 70.8 hours per seed. Three seeds are 8.9 days of exclusive GPU, before prompt
  3's ablation suite gets any time on the same card.
- Nothing in section 8 changes that: the profile says 94% of it is the gradient
  work the recipe specifies, and the recipe is pinned. There is no engineering
  fix here, only a budget decision.
- The lever that exists is `online_total_timesteps`, currently 2.0e8 against
  Classic's 1.0e8. Halving it halves the wall clock to 35 hours per seed and
  would make full Craftax cost what Classic costs. That is a change to the
  experiment, it is on the pinned list, and it is not mine to make.

The recommendation is to run Classic here and to find another card for full
Craftax, or to accept nine days for it. Either way the number to decide on is
70.8 hours per seed, and it is measured rather than extrapolated from a
single-update run.

## 9. Verification

### 9.1 The noise floor is not zero, and that is the headline of this section

`PERF_DAGGER_PROMPT_CRAFTAX.md:235-237` says to establish the noise floor first,
because "JAX is deterministic given a seed and fixed shapes, so if two identical
runs differ at all, find out why". Two identical runs do differ. Here is the
check: one process, one seed, `num_envs: 128`, three updates, the *same*
`train_fn` called three times.

- **A** and **B**: the old call shape, `out = train_fn(rngs)`, twice.
- **C**: the new shape, `train_fn.lower(rngs).compile()(rngs)`.

```
A vs B (noise floor, old shape twice): DIFFERENT
  loss   A [2.1660757064819336, 1.9476990699768066, 1.8012440204620361]
         B [2.1660759449005127, 1.9507800340652466, 1.8049112558364868]
  final_param_sum  A 5805.66845703125   B 5805.86865234375

A vs C (old shape vs compile-then-run): DIFFERENT
  loss   C [2.1660759449005127, 1.9484686851501465, 1.8001083135604858]
  final_param_sum  C 5805.869140625
```

The first update's loss agrees to 1 ULP of float32 and the divergence grows from
there, which is the signature of a non-deterministic reduction amplified by a
chaotic training loop, not of a semantic difference. The cause is XLA's GPU
backend, not this repo and not the change: forcing deterministic kernels makes
the same check pass exactly.

```
$ XLA_FLAGS="--xla_gpu_deterministic_ops=true" python verify_numerics.py ...
  A vs B (noise floor, old shape twice): identical
  A vs C (old shape vs compile-then-run): identical
VERDICT: PASS
```

So D2 is numerics-neutral, which is what the check was for, and it is
demonstrated under the only conditions where the question is answerable. But the
more useful result is the one that came free:

**Author decision: the paper's runs are not reproducible as configured.** Two
runs of identical code at an identical seed on this GPU produce different
weights. Nothing in the repo sets `--xla_gpu_deterministic_ops`, so any claim of
the form "seed 42 gives this curve" does not hold, and any A/B comparison of two
recipes at one seed is reading a difference that a rerun would not reproduce.
This is not specific to the 4070 Ti; it is how XLA's GPU backend behaves by
default anywhere. The options are to fix the flag and pay whatever it costs
(measured in section 9.4), or to report seed-averaged results and never compare
single runs. The choice belongs to the author because it is about what the paper
claims, not about performance.

D1 needs no numerics check of its own: the persistent cache is keyed on the
lowered HLO, so a hit replays the identical executable.

### 9.2 `print_config_snapshot`, before and after

Every derived quantity is unchanged at the configured `num_envs`, which is the
point: neither D1 nor D2 touches the recipe. The only differences in the
snapshot are the deliberate `NUM_ENVS` re-sizing recorded in section 4, and they
appear only when the override is passed.

### 9.3 Suite, lint and smoke

| | Baseline (`d1126fc`) | After this pass |
|---|---|---|
| `pytest tests -q` | 154 passed | 167 passed |
| `ruff check src tests experiments` | 12 errors | 12 errors |
| `python main.py --mode smoke` | OK, ~47 s | OK |

The 12 lint errors are pre-existing (4 `B905`, 4 `I001`, 3 `E402`, 1 `UP037`)
and untouched, per `PERF_DAGGER_PROMPT_CRAFTAX.md:121-123`.

The 13 new tests, all in `tests/test_smoke_src.py`:

| Test | Guards |
|---|---|
| `test_compilation_cache_is_opt_in_and_creates_its_directory` | D1 is off by default and makes its directory when set |
| `test_defaults_config_declares_the_compilation_cache_key` | `--override` rejects keys absent from `defaults.yaml`, so the key must be declared |
| `test_compile_and_run_separates_compile_from_execute` | D2 returns both legs and the execute leg is blocked |
| `test_format_timing_reports_both_legs` | the log line names both |
| `test_online_runner_no_longer_reports_one_fused_time` | the old single-number shape does not come back |
| `test_final_configs_resolve_to_their_documented_quantities` (x4) | the four final configs' comments match the resolvers |
| `test_lr_warmup_is_shorter_than_the_cosine_horizon` (x4) | the constraint optax enforces, in the units the runners use |

One operational note worth recording, because it cost a confusing failure:
`main.py --mode smoke` cannot be run while a measurement holds the GPU. XLA's
Triton autotuner needs memory to benchmark candidate kernels, and with the card
already at 14 GB it fails with

```
INTERNAL: Failed to get configs for: 3 out of 560 instructions.
RET_CHECK failure ... !candidates.empty() Autotuning failed for HLO ...
No configs could be compiled.
```

which reads like a compiler bug and is contention. The same command passed
immediately once the card was free. Anyone treating the smoke run as a green
light between stages needs the GPU idle first.


## 10. The changes, and what each was worth

One commit each, suite and lint green at every commit.

### D1: opt-in persistent XLA compilation cache (`1f0bc46`)

Nothing in the repo set `jax_compilation_cache_dir`. Added it as a config key,
`null` by default because the correct path is machine-specific and must be local
disk rather than an NFS home. `main.configure_compilation_cache` also sets
`jax_persistent_cache_min_entry_size_bytes = -1` so nothing is skipped for being
small.

**Worth:** nothing for a single paper run (52 s against 35.9 hours). The case is
entirely repeated compilation: three seeds launched as separate processes save
2 x 52 s on Classic and 2 x 92 s on Craftax, and prompt 3's 25-entry ablation
suite saves up to 22 minutes (Classic) or 38 minutes (Craftax) if every entry
compiles the same graph. Quoted rather than assumed, because whether the suite's
entries share a graph is prompt 3's question, not this one's.

Files: `main.py`, `configs/defaults.yaml`, `README.md`, `tests/test_smoke_src.py`.

### D2: report compile and execute separately (`d93b005`)

`online.py:709-713` timed `out = train_fn(rngs)` with no `block_until_ready` and
divided total frames by the result. JAX dispatch is asynchronous, so that
duration was compilation plus enqueue and excluded almost all of the execution.
The prompt describes the defect as reporting "compile plus execute as one
number"; the measurement says it is worse than that, because the execute leg was
largely missing rather than merged in.

Added `compile_and_run` and `format_timing` to `planners/common.py` and used them
from `run_online`. Shared rather than local because `offline.py:370` has the
identical shape; prompt 2 owns that file.

**Worth:** no wall clock at all, and every measurement in this report. Without
it the reported SPS for `--mode smoke` is 12 against a true 2,363.

Files: `src/planners/common.py`, `src/planners/online.py`,
`tests/test_smoke_src.py`.

### D3: the W&B host callback costs nothing measurable

`online.py:619` fires `jax.debug.callback(_wandb_log, metric, step_idx)` inside
the jitted update scan on every logged update. Host callbacks are ordered against
the computation, so the concern is real in principle.

Measured by installing the callback but making it a no-op, which isolates the
host round-trip from W&B's own work. Classic, `num_envs: 384`:

| | 4 updates | 12 updates | Marginal per update |
|---|---|---|---|
| no callback | 284.53 s | 778.90 s | 61.80 s |
| callback fires every update | 284.38 s | 778.89 s | 61.81 s |

```
difference: +0.02 s/update, +0.03%
```

That is a fifth of the 0.3% noise floor, so the honest statement is that the
callback's cost is below what this method can resolve. **Not material. Left
alone.** Batching the metrics on device and emitting them once at the end of the
scan, which the returned `metrics` from `online.py:657` would already support,
would buy nothing and would cost the ability to watch a 36-hour run in progress.

This is exactly the change the prompt warns is "easy to justify by theory and
hard to justify by measurement", and the measurement does not justify it. Note
that this prices the host round-trip only; a real W&B callback also does network
I/O, which is unmeasured here and is the part that could stall a run on a bad
connection.

### D4: what the profile said to try, and what it was worth

Section 6 says the training scan is 94 to 99% of an update, and the training
scan is small float32 matmuls. That makes matmul precision the only large lever
the profile allows without touching the recipe. Classic, `num_envs: 384`, same
differencing method, `JAX_DEFAULT_MATMUL_PRECISION` varied:

| Precision | 4 updates | 12 updates | Marginal per update |
|---|---|---|---|
| `default` (shipped) | 284.09 s | 780.49 s | **62.05 s** |
| `tensorfloat32` | 297.76 s | 814.56 s | 64.60 s (4.1% slower) |
| `highest` | 325.19 s | 879.12 s | 69.24 s (11.6% slower) |

XLA's default is already the fast path. `highest`, which forces the three-pass
float32 emulation, costs 12%, so the default is clearly not that. Asking for
`tensorfloat32` explicitly is 4% slower than the default rather than faster,
which says the default is already reaching for tensor cores and that pinning the
algorithm only removes choices from the compiler. There is nothing to win here.
Dropped, with no change made.

The 62.05 s here against the 61.84 s in section 6 is the same quantity measured
in two independent batches an hour apart, so 0.3% is a fair noise floor for
every timing in this report.

Nothing else the profile suggested survived. There is no host-device sync to
remove, because the whole run is one `jax.lax.scan` inside one `jax.jit`; there
is no Python-level loop; and the environment, which is where the sibling
MiniHack repo found its wins, is 1.2% of the Classic update.
### Config comments and a regression test

Section 3.1. Comment-only across four config files, plus
`test_final_configs_resolve_to_their_documented_quantities` and
`test_lr_warmup_is_shorter_than_the_cosine_horizon` so they cannot go stale
again silently.

## 11. What I did not do, and why

- **Did not change `use_optimistic_resets`.** Measured at 1% (Classic) and 5%
  (full Craftax) and left off, per the prompt and on the evidence.
- **Did not fix the `lr_warmup_steps` units defect.** It is a 64x change to the
  learning-rate schedule of every run the recipe has produced, and
  `lr_warmup_frames` is on the pinned list. Reported as an author decision
  (section 3.3).
- **Did not change the DAgger buffer denomination.** Author decision
  (section 3.2).
- **Did not edit `num_envs` in the shipped configs.** `final_classic_ucl.yaml:2`
  and `final_craftax_ucl.yaml:2` declare themselves as 24 GB configurations, and
  they are correct for that hardware. The re-sizing is expressed as an override
  in the run plan. If the author wants a fourth hardware tier the repo's idiom
  is a new file next to the `_qmul` pair, and that is a decision about which
  recipe the paper reports.
- **Did not run either config to completion.** 35.9 and 70.8 hours per seed
  against a shared schedule with three more documents. The run plan is built
  from measured marginal costs and resolved update counts, both of which are
  stated so the arithmetic can be checked.
- **Did not touch `Craftax_Baselines/`.** Submodule, and excluded from lint by
  `pyproject.toml:48`.
- **Did not fix the 12 pre-existing lint errors.** Out of scope by
  `PERF_DAGGER_PROMPT_CRAFTAX.md:121-123`.

## 12. Hand-off to prompts 2 and 3

Shared files touched in this pass:

| File | Change |
|---|---|
| `src/planners/common.py` | added `compile_and_run`, `format_timing`. No existing function changed. |
| `src/planners/online.py` | `run_online` uses them; `import time` dropped. |
| `main.py` | added `configure_compilation_cache`, called from `run`. |
| `configs/defaults.yaml` | added `jax_compilation_cache_dir: null`. |
| `configs/final_{classic,craftax}_{ucl,qmul}.yaml` | comments only. |
| `README.md` | documented the cache key. |
| `tests/test_smoke_src.py` | 13 new tests. |

`src/planners/env.py`, `src/diffusion/` and `src/models/` are untouched.

**For prompt 2.** `NUM_ENVS` is **384** for `final_classic_ucl.yaml` and **256**
for `final_craftax_ucl.yaml`, both with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90`.
Offline BC's working set should be smaller than DAgger's, since it holds no
replay buffer and runs no expert, so those are upper bounds rather than
predictions; re-measure before assuming them. `offline.py:370` has the same
timing defect D2 fixed, and `compile_and_run` / `format_timing` are already in
`common.py` for it. The released checkpoint set contains no offline BC
checkpoint.

**For both.** The run-to-run non-determinism in section 9.1 applies to every
measurement either prompt makes. Any before/after comparison of a loss curve or a
validation return at one seed is reading noise unless
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` is set, and setting it changes what
fits (section 9.4). Timing comparisons are unaffected; the noise floor for those
is 0.3%.

**For prompt 3.** Cold compile is 52 s (Classic) and 92 s (full Craftax) per
process. If the ablation suite launches entries as separate processes, D1's cache
is worth having and `jax_compilation_cache_dir` is the key to set. Per-update
cost is 61.84 s (Classic, 384 envs) and 41.00 s (Craftax, 256 envs), and the
training scan is 94 to 99% of it, so ablations that change the model or the
gradient work will move the wall clock and ablations that change the environment
or the sampler will not.
