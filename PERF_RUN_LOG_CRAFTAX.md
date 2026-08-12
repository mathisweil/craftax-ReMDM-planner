# Craftax performance passes: covering note

Four documents run back to back on one box, per `PERF_ORCHESTRATOR_PROMPT_CRAFTAX.md`.
This is the index; the evidence is in the four reports.

## The box, once

| | |
|---|---|
| Host | `outback.cs.ucl.ac.uk` |
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB, driver 560.35.03 |
| CPU / RAM | Intel i7-14700K, 20 cores / 28 threads, 62 GB |
| JAX | 0.11.0, `CudaDevice(id=0)` |
| CUDA extra | **`cuda12`**. Driver 560.35.03 is below the 580 CUDA 13 needs (`README.md:32`). Synced once, then `uv run --no-sync` throughout |
| `LD_LIBRARY_PATH` | `/opt/ucl/lib:/usr/X11R6/lib`, left alone. Neither directory holds a CUDA library, so nothing shadows the wheels |
| XLA memory | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` for every measurement; preallocation left on. Nothing in the repo sets either |
| Checkpoints | Downloaded from the Hub into `checkpoints/` (gitignored) |

## The four passes

| Pass | Result | Commits | Wall clock | Open items |
|---|---|---|---|---|
| 1. `PERF_DAGGER_PROMPT_CRAFTAX.md` | Done. `PERF_DAGGER_RESULTS_CRAFTAX.md` | 4 | ~5 h | Classic go, full Craftax no-go |
| 2. `PERF_OFFLINE_PROMPT_CRAFTAX.md` | Done. `PERF_OFFLINE_RESULTS_CRAFTAX.md` | 3 | ~2 h | Same |
| 3. `PERF_EXPERIMENTS_PROMPT_CRAFTAX.md` | Done. `PERF_EXPERIMENTS_RESULTS_CRAFTAX.md` | 2 | ~2 h | Craftax arm blocked on a missing checkpoint |
| 4. `ABLATION_SMOKE_PROMPT.md` | Done. `ABLATION_SMOKE_REPORT.md` | 1 | ~2 h | All 25 green, no repairs needed |

Nothing was pushed. Every commit is local.

## Baseline versus final

| | Baseline (`d1126fc`) | Final |
|---|---|---|
| `pytest tests -q` | 154 passed | **172 passed** |
| `ruff check src tests experiments` | 12 errors | **11 errors** |
| `python main.py --mode smoke` | OK | OK |

The 18 new tests are listed in the pass-1 and pass-2 reports. The lint count fell by one
`E402` through a change made by a concurrent session in this tree, not by this work; the
12 pre-existing errors were left untouched throughout, as the prompts require.

## The headline numbers

| | Classic | full Craftax |
|---|---|---|
| configured `num_envs`, DAgger | 512, **does not fit** | 448, **does not fit** |
| largest that runs | 384 (320 deterministic; 256 to keep the offline pair matched) | 256 |
| DAgger, s per update | 61.84 | 41.00 |
| offline BC, s per update | 59.82 | 40.02 |
| training scan share, both modes | 98.8% | 94 to 96% |
| hours per seed, DAgger + offline | 30.6 + 28.5 (deterministic) | 62.0 + 59.2 |
| ablation suite, 25 x 3 seeds | 23.8 h | 25.3 h (blocked) |

Full paper matrix: **22 to 26 days of exclusive GPU**, plus 49 h of ablations.

## Everything left for the author to decide

The four documents deliberately push recipe and sizing decisions back rather than guessing.
Collected here.

### Recipe and science

1. **`lr_warmup_steps` units.** Derived in update steps (`common.py:140`), consumed by optax
   as gradient steps, so the realised warmup is 1/64th of what the configs describe: 1.23M
   frames on Craftax, 1.64M on Classic, against a documented 78.6M and 104.9M.
   **Assessed: do not retrain.** The documented reading is not representable at all, optax
   rejects it (`decay_steps=-4800`), and what ships, a 1.6% linear ramp into a cosine decay,
   is a standard schedule. The frame-denominated invariance survives intact. This is a
   documentation defect. Fix the paper text; do not "correct" the units.
   `PERF_DAGGER_RESULTS_CRAFTAX.md` section 3.3.
2. **DAgger buffer denomination.** `dagger_buffer_cycles` holds a fixed number of update
   cycles, so |D| varies with hardware: 125,000 samples at UCL against 23,438 at QMUL for
   the same config. Internally valid at one tier. **Do not pool tiers in one table.**
   Section 3.2.
3. **Reproducibility.** Two runs of identical code at one seed do not agree: XLA's GPU
   backend is non-deterministic by default. `--xla_gpu_deterministic_ops=true` fixes it and
   is 12 to 18% **faster** here. No retraining needed; enable it going forward and avoid
   single-seed claims. Section 9.1 and 9.4.
4. **`num_envs` re-sizing.** Required to run here at all. Re-sized runs are comparable, not
   identical: update count, minibatch and buffer sample count all move. Do not mix them
   with the four completed 24 GB ablation cells. Section 4.
5. **`use_optimistic_resets`.** Measured at 1% (Classic) and 5% (full Craftax). Left off;
   the recommendation is to leave it off. Section 7.

### Budget

6. **Full Craftax is a no-go as configured**, at 62 h per seed for DAgger plus 59 h for the
   matched offline baseline. The only lever is `online_total_timesteps` /
   `offline_total_timesteps`, currently 2.0e8 against Classic's 1.0e8; they must move
   together. Pass 1 section 8.2, pass 2 section 7.2.

### Missing artefacts

7. **Three of four planner checkpoints are unpublished.** `README.md:245-248` lists two
   offline BC and two online DAgger; the Hub repo has one, Classic DAgger. There is **no
   full-Craftax planner checkpoint**, which blocks that half of the ablation suite outright.
8. **The re-upload put MiniHack checkpoints into the Craftax repo**:
   `checkpoints/offline/Minihack-Offline-Diffusion-BC-100M` (PyTorch) and
   `checkpoints/online/Minihack-Online-Diffusion-DAgger-100M` (config only, no weights).
9. **A Hugging Face token is exposed** in the repo root as an empty file named
   `HF_TOKEN=hf_ZkVS…`. Untracked and never committed. **Rotate it, then delete the file.**

### Config text

10. `ablations_final_craftax_ucl.yaml:3` points at `configs/final_craftax_full_ucl.yaml`,
    which does not exist. Line 5 names a different machine than
    `configs/final_craftax_ucl.yaml:2`. Both reported, neither changed.
11. `ablations_final_classic_qmul.yaml:39` says "64 envs / 512 batch" while `batch_size` is
    256. The measured 5.57 GiB matches the sizes the file sets, so the "512" is the error.
12. `common.py:270` still prints `lr_warmup_steps = 1600 (~104.86M frames)`, converting with
    the wrong units. Print-only, same class as the minibatch bug fixed in `9207fca`. Offered
    and not applied, awaiting a decision on item 1.

## A note on the working tree

Another session committed into this repo while these passes ran: it rebased this work twice,
committed in-progress files under "commit for push" and "mid session, push to pull", and at
18:58 rewrote the tree while a measurement was running, destroying three Phase 3 results
(re-run, all green). The checkpoint directory was also deleted and re-uploaded mid-sweep.
Measurements taken during those windows were identified and repeated; everything reported
here is from a clean run.
