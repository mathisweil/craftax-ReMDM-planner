# Task: the four runs the paper review says are missing

> **STATUS: COMPLETE — 2026-08-27.** All four runs (plus a required fifth anchor) are done and
> the results are in [Report back](#report-back--done-2026-08-27) at the bottom of this file.
> The task text below is kept as the original spec; two of its instructions were not followed
> and the reasons are recorded under "Deviations". Do not re-run these without reading that
> section first.

Everything here needs a GPU. **Use the `/ucl-gpu` skill to find and reserve a free GPU on the
UCL CS cluster before starting, and release it when done.** Only the runs below are in scope —
they are sized in single-digit GPU-hours against the ~1,330 already spent on this project.

Read `CLAUDE.md` and `experiments/README.md` first.

**Sizing, from the paper's own accounting:** the 25-condition × 3-seed Craftax Classic suite
cost 24.6 GPU-hours for 75 runs, so **one condition × 3 seeds ≈ 1 GPU-hour**.

### Hard constraints — read before touching any config

- **Never edit `configs/final_*` or `experiments/rl_finetuning/configs/ablations_final_*` to
  fit the cluster hardware.** `CLAUDE.md` forbids it and `tests/test_config.py` enforces the
  sibling rules. Pick the closest existing machine config (`*_gpu_24gb` or `*_gpu_h200`) for
  whatever GPU you reserve and use it as-is.
- Presets are delta-only. Restating a defaults value silently pins it, and the config test will
  fail.
- Set `jax_compilation_cache_dir` in `configs/defaults.yaml` to **local disk, not an NFS home**.
  `run_ablations.py` has no `--override`, so it cannot be set on the command line. At 3 seeds
  the graph is identical across seeds, so two runs in three are a cache hit.
- Every new run must use the **same pretrained checkpoint** all 25 published conditions start
  from. A different checkpoint makes the comparison meaningless.
- Record the wandb run id for each run. Every compute figure in the paper is sourced to one.

---

## Run 1 — the unweighted-ELBO-on-all-rollouts arm  (~1 GPU-hour, 3 seeds)

**Highest value of the four. This is the control the paper says it did not run.**

The manuscript's headline inference is that the degradation comes from fine-tuning on
self-generated rollouts rather than from the return weighting — and it concedes there is no arm
that separates the two. Currently the paper's own limitations section keeps four rival readings
open, one of which (the weighting is *protective*) is the exact inverse of its conclusion.

The arm: fine-tune from the same checkpoint on the same on-policy rollouts with **uniform
weights** — i.e. plain behavioural cloning on self-generated data, no return weighting at all.
Everything else identical to `baseline_rl`: same rollout collection, same iteration count, same
learning rate, same 3 seeds.

Add it as a condition in `experiments/rl_finetuning/ablations/registry.py` following the
existing `AblationSpec` pattern, rather than as a one-off script.

**What the outcome means — state it plainly, do not spin it:**
- If it degrades about as much as `baseline_rl`, the self-generated-data account is supported
  and the paper's central inference becomes a measurement.
- If it degrades **less**, the return weighting does carry blame and the paper's conclusion is
  wrong.
- If it degrades **more**, the weighting was protective — the reading the paper currently keeps
  open as the inverse of its own.

Report the score and per-seed values. Do not adjust anything to reach a preferred outcome.

---

## Run 2 — advantage clipping at matched effective step  (~1 GPU-hour, 3 seeds)

The paper's central negative control compares `baseline_rl` against `advantage_clip` and
concludes the return term is not causal. The comparison is confounded: Eq. 4 makes $\bar{A}$
the *effective learning rate*, the baseline's weights live in `[0.1, 5.0]` with most windows at
the floor, and clipped weights live in `[0.8, 1.2]`. The two arms plausibly differ in step
size — which is the mechanism the paper's own drift-vs-score ordering says drives the result.

**Do `GDELTA_VERIFICATION.md` §2 first** — it adds `Abar` reporting to `measure_gdelta.py`,
which runs on CPU and needs no reservation. You need
$\bar{A}_{\text{base}} / \bar{A}_{\text{clip}}$ before this run is worth starting.

Then rerun `advantage_clip`, 3 seeds, with the learning rate scaled by
$\bar{A}_{\text{base}} / \bar{A}_{\text{clip}}$ so the two arms are matched on effective step.
Report it alongside the published `advantage_clip` score of 5.06 and the baseline's 8.22.

If the gap survives matching, the paper's control stands and this run rescues it. If it
collapses, the control was measuring step size.

---

## Run 3 — score the PPO-RNN expert  (minutes)

The paper reports no external anchor. The PPO-RNN expert that supervised DAgger is never
scored, so a reader cannot tell whether the pretrained planner at 11.81 is close to its teacher
or far below it. If the student is far below, the negative result may be about a weak
checkpoint rather than about the objective.

`scripts/eval_ppo_expert.py` already exists. Note `CLAUDE.md`: **released PPO expert
checkpoints fail to restore on CPU-only machines**, which is why this needs the reservation.

Report the expert's score under the *same* harness settings the planner is evaluated with
(50 denoising steps, replanning every 8), so the two numbers are comparable.

While you are there, settle a definition question the review raised: the paper calls its metric
"Score, the mean episodic reward", but Craftax's published *Score* is the geometric mean of
achievement rates. Confirm from the code which quantity 11.81 actually is, and report it. This
decides whether the paper's numbers are comparable to any published Craftax result.

---

## Run 4 — learning-rate sweep  (~4 GPU-hours; ask before starting)

**Larger than the other three. Check with the author before running it.**

Fine-tuning uses `3e-4`, *identical to the pretraining learning rate*, on a converged
checkpoint. No condition in the suite varies it, and the five best-scoring conditions are the
five that most reduce the effective step. So the most parsimonious reading of the entire
results table is "a converged model fine-tuned at full pretraining LR degrades, and anything
that shrinks the step degrades less" — which has nothing to do with the return-weighted ELBO
and would not support the paper's title.

Full version: `baseline_rl` at 3e-4, 1e-4, 3e-5, 1e-5, three seeds each — 12 runs, ~4
GPU-hours. Reduced version if that is too much: 1e-4 and 1e-5 only, 6 runs, ~2 GPU-hours,
which still answers the question "does *any* learning rate recover 11.81?".

---

## Report back — DONE, 2026-08-27

All four runs executed, plus a fifth (`baseline_rl` re-run) that turned out to be required; see
"Why there is an anchor run" below. Reservations released. Report artifact:
<https://claude.ai/code/artifact/c609a865-34d2-42a7-a057-3b018a6b5073>

**Common setup for every training run below.** Deviating from any of these invalidates the
comparison:

| | |
|---|---|
| Checkpoint | `checkpoints/ablation_src/policy-best-v2` (Orbax step 40370176) — see "Recovering the checkpoint" |
| Ablations config | `experiments/rl_finetuning/configs/ablations_final_craftax_classic_gpu_24gb.yaml` |
| Seeds | 0, 1, 2 (`--num-seeds 3`, base seed 0) |
| Iterations | `MAX_ITER` 500, `NUM_ENVS` 192, `BATCH_SIZE` 1024, `EVAL_STEPS` 1024 |
| Hardware | RTX 3090 Ti 24 GB — `bufflehead-l` and `mallard-l` (duck cluster) |
| W&B project | `myopic-planner/craftax-ReMDM-planner-ablations` |
| `pretrained_score` | **11.975** in all five runs (published suite records 11.808 — see caveat) |

### The table

| Run | Condition | LR | Per-seed scores | Mean | Seed sd | wandb | Wall-clock | GPU-h |
|---|---|---|---|---|---|---|---|---|
| — | pretrained (no fine-tuning) | — | — | 11.975 | — | — | — | — |
| anchor | `baseline_rl` | 3.0e-4 | 8.259, 8.545, 8.214 | **8.340** | 0.147 | `d94k5ai4` | 3426 s | 0.95 |
| 1 | `bc_all` | 3.0e-4 | 4.609, 4.999, 4.596 | **4.735** | 0.187 | `beixln41` | 3479 s | 0.97 |
| 2 | `advantage_clip` | 3.223e-4 | 5.278, 4.824, 4.672 | **4.925** | 0.257 | `cqqepmc9` | 3393 s | 0.94 |
| 4a | `baseline_rl` | 1.0e-4 | 10.232, 9.897, 10.316 | **10.149** | 0.181 | `3y8g4ajy` | 3431 s | 0.95 |
| 4b | `baseline_rl` | 1.0e-5 | 10.936, 10.685, 10.810 | **10.810** | 0.102 | `bfqh9ewf` | 3350 s | 0.93 |
| 3 | PPO-RNN expert | — | 18.298, 18.724, 18.807 | **18.610** | 0.223 | n/a (script, no W&B) | 34 s | <0.01 |
| | | | | | | | **17 079 s** | **4.74** |

Published values quoted for comparison (NOT re-run): `bc_wins` 7.285 ± 0.261,
`advantage_clip` 5.063 ± 0.066, `lora` 11.631 ± 0.028, `trust_region_kl` 11.526 ± 0.055.

### The four answers

1. **More.** `bc_all` 4.735 ± 0.187 against `baseline_rl` 8.340 ± 0.147 on the same stack — a
   3.6-point gap with no per-seed overlap. By this file's own decision rule that is the third
   branch: **the return weighting was protective**, the inverse of the paper's conclusion.
2. **It survives.** `advantage_clip` at the matched LR scores 4.925 ± 0.257 against 5.063 ± 0.066
   published at 3e-4 — a move of −0.14, inside noise — so the gap to `baseline_rl`'s 8.340 stands
   and the control was not measuring step size.
3. **Expert 18.610 ± 0.223** (18.819 over 683 completed episodes at 256 envs), so the planner sits
   at 64% of its teacher; and "Score" is **mean episodic reward**
   (`LogWrapper.returned_episode_returns`), **not** the Craftax geometric mean — recomputed
   properly the planner's Craftax Score is 37.9 and the expert's is 74.5.
4. **No.** 8.340 (3e-4) → 10.149 (1e-4) → 10.810 (1e-5): monotone but plateaus 1.17 short of
   11.975, and `lora` (11.631) and `trust_region_kl` (11.526) beat every LR tested including the
   much smaller 1e-5 step, so "anything that shrinks the step" does not explain the results table.

### Where the numbers live

Everything under `experiments/rl_finetuning/outputs/` and `results/` is **gitignored** — these
files exist on this machine only and are not in the repository. Do not expect to find them after
a fresh clone.

| What | Path |
|---|---|
| Per-run scores, config, history, achievement rates | `experiments/rl_finetuning/outputs/review_{anchor_baseline_rl,run1_bc_all,run2_advclip_lr_matched,run4_baseline_lr1e-4,run4_baseline_lr1e-5}/results.json` |
| Per-run tables (`main_results`, `achievement_summary`, …) | `…/<run_id>/tables/*.{csv,tex}` |
| Per-run figures and `diagnosis.md` | `…/<run_id>/figures/`, `…/<run_id>/diagnosis.md` |
| Fine-tuned params, last seed | `…/<run_id>/checkpoint_<condition>/` (~37 MB per run) |
| Expert eval, 3 seeds × 2 temperatures + 256-env anchor | `results/inference/expert_classic_n{32,256}_s{0,1,2}_t{1.0,0.5}.json` |
| Expert eval re-run on healthy hardware | `results/inference/verify/` |
| Published 25-condition suite (untouched) | `experiments/rl_finetuning/outputs/craftax_classic_ablations/results.json` |
| Ābar / g_delta measurement used for Run 2's LR | `experiments/rl_finetuning/outputs/craftax_classic_ablations/gdelta/gdelta_aggregate.json` |

Score key inside `results.json` is `ablations.<name>.score` (seed mean), `.all_scores` (per seed),
`.score_std` (population sd over 3), `.wall_clock_s` (list, per seed).

### Reproducing a run

```bash
JAX_COMPILATION_CACHE_DIR=/var/tmp/$USER/jax-cache \
JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=-1 \
JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1.0 \
uv run python experiments/rl_finetuning/run_ablations.py \
    --ablations-config experiments/rl_finetuning/configs/ablations_final_craftax_classic_gpu_24gb.yaml \
    --ablations bc_all --num-seeds 3 \
    --checkpoint checkpoints/ablation_src/policy-best-v2 \
    --run-id review_run1_bc_all --use-wandb
```

Runs 2, 4a and 4b add `--lr 0.000322321`, `--lr 0.0001`, `--lr 0.00001` respectively. Run 3 is
`scripts/eval_ppo_expert.py --path checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M
--env-name Craftax-Classic-Symbolic-v1 --num-envs 32 --steps 1024 --seed 0 --temperature 1.0`.

### Deviations from this file's instructions, and why

- **`jax_compilation_cache_dir` was NOT set in `configs/defaults.yaml`.** The instruction above
  conflicts with `tests/test_smoke_src.py::test_defaults_config_declares_the_compilation_cache_key`,
  which asserts the shipped default stays `null` ("the right path is machine-specific"). The
  `JAX_COMPILATION_CACHE_DIR` environment variable has an identical effect — JAX reads it natively
  and `configure_compilation_cache` does not overwrite it when the config key is `null` — with no
  machine path committed. Use the env var; do not edit `defaults.yaml`.
- **`GDELTA_VERIFICATION.md` §2 was already done.** That file was deleted in `6aa11dd`; its work is
  merged into `run_ablations.py --measure-gdelta` and `analysis/gdelta.py`, which already emit
  `abar` and `abar_ratio_to_baseline`. No CPU run was needed.
- **Run 4 was the reduced sweep** (1e-4 and 1e-5, 6 runs), per the author's answer when asked.
  3e-5 was not tested.
- **Run 3's harness settings.** "50 denoising steps, replanning every 8" has no counterpart for a
  reactive PPO-RNN. What is matched is the environment, the eval size (32 envs × 1024 steps) and
  the return statistic (episode-weighted completed-episode return).

### Recovering the checkpoint

The path the published suite recorded — `.../train_space/remdm-planner-workspace/.../checkpoints/
ablation_src/policy-best-v2` — no longer exists, and `checkpoints/` is gitignored, so the local
copy used here will not survive a fresh clone. Two ways back to the identical weights:

```bash
# either: the W&B artifact the published suite ran from
python -c "import wandb; wandb.Api().artifact(
    'myopic-planner/craftax-ReMDM-planner/Craftax-Classic-Symbolic-v1-policy-best:v2',
    type='model').download(root='checkpoints/ablation_src/policy-best-v2')"

# or: the released checkpoint, which is byte-identical (verified with diff -r)
--checkpoint checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M
```

Both are Orbax step 40370176 and both give `pretrained_score` 11.975 on this stack. The released
one is simpler and needs no W&B access.

### Why there is an anchor run

The suite does **not** reproduce exactly across hosts. Identical checkpoint, config and seed give
`pretrained_score` 11.975 here against 11.808 published, and `baseline_rl` 8.340 against 8.223 — a
consistent +1.4% offset on the same GPU model. Within a host it is exactly deterministic (all five
runs report 11.975). So a newly measured condition cannot be compared against a number from the
published table below ~0.2 points, and `baseline_rl` was re-run to give Run 1 a same-stack
reference. Any future arm needs the same treatment.

### Ā values used for Run 2

From `gdelta_aggregate.json`, across 3 rollout seeds:
Ā_base = 1.0235 ± 0.0101, Ā_clip = 0.9527 ± 0.0090, so **Ā_base/Ā_clip = 1.0744 ± 0.0041** and the
matched LR is 3e-4 × 1.0744 = **3.223e-4**.

Note this falsifies the premise stated in the Run 2 section above: the baseline's Ā is **not** ≪ 1.
The `[0.1, 5.0]` floor and cap roughly cancel in the mean, so both arms sit near Ā ≈ 1 and differ in
effective step by 7.4%, not by a large factor. Two further reasons the confound is weak: logged
`rl_grad_norm` is median 0.41 (`baseline_rl`) and 0.40 (`advantage_clip`), both well under
`MAX_GRAD_NORM = 1.0`, so the global-norm clip never fires; and AdamW's update is invariant to a
uniform gradient rescale up to `eps = 1e-5`.

### Code changes (committed, branch `fix-stale-results-paths-and-publish-paper-figures`)

- `cdace5e` — adds `bc_all`: `make_loss_bc_all` in `ablations/losses.py`, the `AblationSpec` in
  `ablations/registry.py`, its pinned description in `tests/test_config.py`, and the README row
  plus the 25→26 count updates. The registry now holds **26** ablations, of which the published
  paper table covers 25.
- `0f05b7e` — `scripts/eval_ppo_expert.py` passed `config={"SEED": …}` into `ActorCriticRNN`, which
  reads `config["LAYER_SIZE"]`, so every documented invocation died with `KeyError: 'LAYER_SIZE'`.
  Fixed; Run 3 was impossible before this.

`uv run pytest` passes 416 tests. `tests/test_gpu_agreement.py` fails for an unrelated
pre-existing reason (it needs a CUDA jaxlib; the repo `.venv` is CPU-only) and was deselected.

Nothing under `src/` was touched and no `final_*` or `ablations_final_*` config was edited.

### Still open

- The sibling `minihack-ReMDM-planner` needs the matching `bc_all` entry in its registry,
  description table and README to keep the deliberate parity.
- `brent-l` (duck cluster) is producing bad computation — `dmesg` shows `invalid opcode` in
  `libnvrtc.so.13` and random segfaults in `ptxas` and `xla_cuda_plugin.so`, on binaries that run
  clean on `bufflehead-l` and `mallard-l`. It killed two jobs silently. Do not use it until CS
  support has looked at it; Run 3 was re-run on `bufflehead-l` and reproduces to the third decimal
  (`results/inference/verify/`).
