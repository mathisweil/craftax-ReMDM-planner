# Complete architectural makeover of src/ablations/ and notebooks/

Read everything first, then redesign.

---

## Step 0 — Read every file in full before touching anything

Read these files completely, in this order, before writing a single line:

1. `src/ablations/__init__.py`
2. `src/ablations/losses.py`
3. `src/ablations/techniques.py`
4. `src/ablations/diagnostics.py`
5. `src/ablations/visualisations.py`
6. `src/ablations/runner.py`
7. `notebooks/craftax_ablations_full.ipynb` — every cell
8. `notebooks/generate_report.py`
9. `src/diffusion/loss.py`
10. `src/planners/common.py`
11. `.claude/rules/ablations.md`
12. `.claude/rules/training.md`
13. `configs/ablations.yaml`
14. `configs/defaults.yaml`

Do not begin any edits until you have read all 14 files.

---

## Step 1 — Write a full diagnosis before writing any code

Before editing, produce a written diagnosis covering:

**A. Interface mismatches**
- List every place where a `src/ablations/` function has a signature, return type, or contract that the notebook cannot directly call without an adapter. Be specific: name the function, the expected signature, the actual signature, and what breaks.

**B. Dead or duplicate code**
- List every function or block that is defined in both `src/ablations/` and inline in the notebook.
- List every function in `src/ablations/` that is never called by the notebook or by `generate_report.py`.
- List every inline helper in the notebook that belongs in `src/ablations/`.

**C. History schema inconsistency**
- The notebook's `make_empty_history()` defines one set of keys. `runner.py`'s history dict defines a different set. `generate_report.py` and the visualisation cells read from yet another assumed schema. Document the full mismatch as a table: key → where it is written → where it is read → whether the shapes/types match.

**D. The run_ablation vs run_ablation_v2 decision**
- `runner.py`'s `run_ablation_v2` expects `collect_rollout(rng, params) -> (obs, acts, valid, adv)`. The notebook's `collect_rollout` is a 5-arg stateful function. Document exactly what restructuring is required for the notebook to use `run_ablation_v2`, including what must change in `runner.py` and what wrapper the notebook would need.

**E. Loss function contract**
- `src/ablations/losses.py` functions return `(scalar, dict)`. The notebook's `make_ablation_grad_step` unwraps tuples with `result[0] if isinstance(result, tuple) else result`. Document every place this mismatch propagates.

**F. Visualisation contract**
- `generate_report.py` calls visualisation functions and assumes `all_histories` has specific keys. The notebook builds `ALL_HISTORIES_COMBINED`. Document whether they are compatible or not.

---

## Step 2 — Design the target architecture (write it out before implementing)

Propose a clean target architecture. The design constraints are:

1. The notebook must be thin. It handles: config loading, environment setup, checkpoint loading, calling `run_ablation` (or equivalent), and displaying results. No loss maths, no gradient computations, no diagnostic implementations inline.
2. `src/ablations/` owns all reusable logic. Every loss factory, every diagnostic function, the training loop, and every plot function lives in `src/`. The notebook imports and calls them.
3. One history schema. There is exactly one canonical history dict schema, defined in one place, used consistently by the training loop, the notebook visualisation cells, and `generate_report.py`. Define it.
4. One training loop. Either `run_ablation` (notebook-inline) or `run_ablation_v2` (`src/`) — pick one and eliminate the other. Justify your choice. If you choose `run_ablation_v2`, specify how `collect_rollout` must change to make it compatible. If you keep a notebook-level loop, specify what it must delegate to `src/`.
5. Loss functions have a single contract. Decide: do they return scalar or `(scalar, dict)`? Apply it uniformly across all 10 factories and all call sites.
6. `diagnostics.py` is the single source of truth for metrics. No diagnostic logic lives in the notebook or in `runner.py` inline.

Write this design as a table:

| File | Purpose after makeover | Depends on | Called by |
| :--- | :--- | :--- | :--- |
| | | | |

And a section: "Functions removed / merged / moved".

---

## Step 3 — Implement the makeover

Only after Steps 1 and 2 are complete, implement the changes. Follow this order strictly:

1. `src/ablations/losses.py` — fix the loss function contract. All factories return `(scalar, dict)` (the dict carries per-step diagnostics like unweighted_loss, frac_masked). Update `_base_elbo` to return the same tuple. This is the foundation everything else depends on.
2. `src/ablations/diagnostics.py` — ensure the canonical set of diagnostic functions covers everything the notebook currently computes inline. If `compute_repr_drift` (output KL) is needed separately from `compute_representation_drift` (parameter L2), add it. If `compute_t_gradient_analysis` (gradient norms by t-range) is needed, add it. Each function must be JIT-compatible where applicable.
3. `src/ablations/techniques.py` — `make_ablation_grad_step` must accept loss functions that return `(scalar, dict)` and surface the dict as part of its returned metrics. Remove any assumptions about loss return type.
4. `src/ablations/runner.py` — `run_ablation_v2` must use the canonical history schema. Its `collect_rollout` interface must be documented and match what the notebook can provide (either make it stateful-compatible or provide an adapter function in `runner.py` itself). The function must return `(history, final_params)` — not just dict — so callers can extract the trained parameters.
5. `src/ablations/__init__.py` — export everything that the notebook needs to import. Nothing more.
6. `configs/ablations.yaml` — add any hyperparameters that are currently hardcoded in the notebook.
7. `notebooks/craftax_ablations_full.ipynb` — rewrite every code cell to use the updated `src/ablations/` API. The notebook must not contain: loss maths, gradient computation, diagnostic implementations, or history schema definitions. It must contain: imports, config, env setup, calls to `src/` functions, and result display.
8. `notebooks/generate_report.py` — verify it is fully compatible with the new history schema and visualisation API. Fix any breakages.

---

## Step 4 — Integration check

After implementing, verify the following without running any code — by reading the final state of each file:

- Every notebook cell that calls a `src/ablations/` function uses the correct signature.
- The history dict written by the training loop contains exactly the keys read by the visualisation cells and `generate_report.py`. No key is written but never read; no key is read but never written.
- All 10 loss factories have the same return contract.
- `make_ablation_grad_step` correctly handles the loss factory return type.
- No inline diagnostic function definitions remain in any notebook cell.
- `run_ablation_v2` (or whatever the canonical training loop is) returns `(history, final_params)`.
- `src/ablations/__init__.py` exports every symbol the notebook imports from `src.ablations`.
- `generate_report.py` has no hardcoded assumptions that break with the new schema.

---

## Constraints throughout

- Follow all rules in `.claude/rules/ablations.md` and `.claude/rules/jax-idioms.md`.
- All new `src/` functions must have docstrings and type annotations.
- No new global mutable state. All functions that use randomness take an explicit `rng: jax.Array`.
- No `plt.show()` inside `src/`. Figures are returned and displayed by the caller.
- `Craftax_Baselines/` is read-only. Do not touch it.
- `src/diffusion/loss.py`'s `compute_loss` is the canonical ELBO implementation. Do not re-implement ELBO math in `src/ablations/`.
- Use `jax.tree.map` (not `jax.tree_map`).