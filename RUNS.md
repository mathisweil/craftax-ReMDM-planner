# Task: the four runs the paper review says are missing

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

## Report back

One table: run, condition, learning rate, per-seed scores, mean, seed sd, wandb run id,
wall-clock, GPU-hours.

Then answer, in one line each:
1. Does the unweighted-rollout arm degrade more, less, or the same as `baseline_rl`?
2. Does the clipping gap survive matching on effective step?
3. What does the expert score, and is "Score" mean episodic reward or the Craftax geometric mean?
4. (If run) Does any learning rate recover 11.81?

Release the GPU reservation when finished. **Do not edit anything under the manuscript's
`src/`** — these are findings for the authors. If a run contradicts the paper, say so plainly.
