# Task: verify and wire in `measure_gdelta.py`

`experiments/rl_finetuning/measure_gdelta.py` produces every number in the paper's
gradient-decomposition appendix (`tab:gdelta`). An external review found the script is
**untracked in git**, covered by **no test**, mentioned in **no README**, and **missing two
outputs the paper's argument depends on**. Fix all of that.

Read `CLAUDE.md` and `experiments/README.md` first. Run `uv run pytest` when done.

---

## 0. Context: what the script computes

Writing per-window weights as $A_i$ with batch mean $\bar{A}$, and $\delta_i = A_i/\bar{A} - 1$:

```
grad L_RW  =  Abar * ( grad L_BC  +  g_delta )
g_delta    =  (1/B) sum_i delta_i grad l_i
```

The script loads the pretrained checkpoint, collects one on-policy batch, and evaluates the
three gradients under a shared `(z_t, t)` draw so the only difference is the weight vector.

**Its path is already correct** — the paper names
`experiments/rl_finetuning/measure_gdelta.py` verbatim. Do not move it.

---

## 1. Commit it (the defect that actually breaks reproducibility)

`git status` reports the file as untracked, so it is absent from the released repository. The
paper's code appendix tells readers to run it and its checklist answers **Yes** to open code
access. `git add` it as part of this task.

---

## 2. Add `Abar` to the output — highest priority

`main()` computes `wbar = jnp.mean(weights)` per variant and uses it for `delta` and for the
Eq. 4 residual, but **never writes it to `out`**. It is therefore not recoverable from the
JSON, and the manuscript does not report it.

This is the single most load-bearing missing number in the paper. Eq. 4 makes $\bar{A}$ the
effective learning rate of the update: the baseline's weights live in `[0.1, 5.0]` with most
windows pinned at the floor (only ~8% carry reward), so $\bar{A} \ll 1$, while
`advantage_clip`'s weights live in `[0.8, 1.2]` so $\bar{A} \approx 1$. The paper's central
control compares those two arms and attributes the difference to $g_\delta$ — but if their
$\bar{A}$ differ, they also differ in effective step size, which is a complete alternative
explanation. A reviewer will ask for this number and it is one line away.

Emit per variant, alongside the existing fields:

- `abar` — `float(wbar)`
- `abar_ratio_to_baseline` — this variant's `abar` divided by `baseline_clipped_ratio`'s

Print `Abar` as a column in the summary table too. Report the value; do not editorialise about
what it implies.

---

## 3. Add a shuffled-$\delta$ null

`g_delta = (1/B) sum_i delta_i grad l_i` with `sum_i delta_i = 0`. For **any** zero-mean weight
vector of the same dispersion, this is a batch-heterogeneity direction that is near-orthogonal
to the batch-mean gradient by construction, with norm scaling as `CV_A`. So the measured
`ratio` and `cos` may say nothing about *returns* — they may just be reporting per-window
gradient variance.

Add a control that settles it. For each variant, alongside the real `delta`, compute
`delta_shuffled = jax.random.permutation(key, delta)` — same multiset of weights, same `CV_A`,
all association with each window's actual return destroyed — and record its `ratio` and `cos`
under `ratio_shuffled_mean/std` and `cos_shuffled_mean/std`.

Use a **separate** PRNG key for the permutation, and keep the `(z_t, t)` key shared with the
real measurement so the comparison is like-for-like.

If the shuffled null lands near the real value, that is the finding — report it plainly. Do not
tune the control until it separates.

---

## 4. Reconcile the seed/draw axis with what the paper claims

The appendix says: *"We repeat over 8 independent $(z_t, t)$ draws and 3 rollout seeds on
1,024-window batches, and report mean and standard deviation across seeds."*

The script computes `ratio.std()` and `cos.std()` **over the 8 draws within a single
`--seed`**. That is a different quantity from a standard deviation across 3 seeds, and there is
no aggregator that combines multiple output JSONs.

Do both of these:

1. Rename the existing within-seed fields to `ratio_std_draws` / `cos_std_draws` so they cannot
   be mistaken for the across-seed figure. Keep the means as they are.
2. Add a small aggregation path — either a `--inputs a.json b.json c.json --aggregate` mode on
   this script or a sibling helper — that takes the three per-seed JSONs, averages each
   variant's per-seed mean, and reports the **standard deviation across the three seed means**.
   That is the number `tab:gdelta` prints.

Report which of the two the current published table actually corresponds to. If they disagree,
say so; **do not silently change a published number**.

---

## 5. Verify against the published table

Run all three seeds against the released Craftax Classic checkpoint:

```bash
for s in 0 1 2; do
  python experiments/rl_finetuning/measure_gdelta.py \
      --ckpt <released-craftax-classic-checkpoint> \
      --config <results.json from the ablation suite> \
      --out gdelta_seed${s}.json --seed ${s}
done
```

Then check every value below. Report a table of **published vs reproduced** and flag any
mismatch. Do not edit the manuscript, and do not adjust the script to make a number match.

| Quantity | Published |
|---|---|
| `D` (parameters) | 9.33M |
| random-cosine null sd, $D^{-1/2}$ | 3.3e-4 |
| `cos(grad L_BC, grad L_BC)`, independent draws | 0.893 ± 0.010 |
| Eq. 4 max relative residual | < 4e-4 |
| ESS on the measurement batch | 0.51 of batch |
| baseline — CV_A / ratio / cos | 0.98 / 0.49 ± 0.01 / +0.02 |
| `bc_wins` — CV_A / ratio / cos | 0.77 / 0.417 ± 0.025 / +0.14 |
| `advantage_clip` — CV_A / ratio / cos | 0.20 / 0.097 ± 0.003 / −0.00 |
| `normalized_adv` | undefined, $\bar{A} \approx 0$ violates (A1) |

Two things to check while you are in there:

- **`bc_wins` weight construction.** The script reconstructs it as
  `win * batch / n_win`, but the real ablation gets its weights from
  `_compute_advantages(..., wins_only=True, ...)`. Confirm these produce the same vector on the
  same batch. If they do not, the `bc_wins` row is measuring something the trainer never uses.
- **The `variants` dict is hardcoded.** It does not derive from
  `experiments/rl_finetuning/ablations/registry.py`, so a registry change silently desynchronises
  the measurement from the suite. Derive it from the registry, or add an assertion that the four
  names still exist there with the weighting rules assumed here.

---

## 6. Wire it in

- **Test.** Add coverage in the style of `tests/test_smoke_experiments.py` / `test_spec_*.py`.
  `tests/conftest.py` already forces `JAX_PLATFORMS=cpu` and disables W&B. A checkpoint-free
  unit test of the algebra is worth more than a slow end-to-end run: build a small random
  parameter tree and a synthetic batch, then assert
  `grad L_RW == Abar * (grad L_BC + g_delta)` to float32 tolerance, assert `mean(delta) == 0`,
  and assert the `normalized_adv` branch is detected as `(A1)`-violating rather than returning
  a silently wrong number.
- **`experiments/README.md`.** The `rl_finetuning/` directory tree block does not list
  `measure_gdelta.py`. Add it, with a one-line description and the exact reproduction command,
  matching the surrounding style.
- **Output location.** The script writes `--out` wherever it is pointed. Every other artifact
  in the suite lands under `results/experiments/rl_finetuning/outputs/{run_id}/`. Give it a
  sensible default under that tree rather than the current working directory.

---

## 7. Report back

State clearly:

1. Whether every published number reproduces, with the mismatches listed.
2. **The `Abar` value for `baseline_clipped_ratio` and for `advantage_clip`, and their ratio.**
3. What the shuffled-$\delta$ null gives for each variant.
4. Whether the published ± is across seeds or across draws.

Do not edit anything under the manuscript's `src/`. If a published number does not reproduce,
say so and stop — that is a finding for the authors, not something to correct here.
