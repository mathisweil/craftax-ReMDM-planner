# Claude Code Prompt: Notebook Review, Integration & Cleanup

## Context

You are working on `craftax-ReMDM-planner`. The full project is described in `README.md` and `CLAUDE.md`. Read both before touching anything.

The notebook at `notebooks/craftax_ablations_full.ipynb` was written iteratively — a clean first half (Cells 0–31) followed by a second half (Cells 33–49) that was bolted on later to import from `src/ablations/`. The two halves were never reconciled. The notebook has accumulated real bugs, logical errors, integration failures, and redundancies.

Your job is to fix all of it. This is a review-and-repair task, not a feature task. Do not add new ablations or new diagnostics. Make what exists correct, clean, and properly integrated.

---

## Step 0 — Read everything before changing anything

1. Read `README.md` in full.
2. Read `CLAUDE.md` and all files under `.claude/rules/`.
3. Read `configs/defaults.yaml`.
4. Read `configs/ablations.yaml` (if it exists; note if it doesn't).
5. Read all files under `src/planners/`: `model.py`, `ppo.py`, `env.py`, `common.py`, `train.py`, `logging.py`.
6. Read all files under `src/diffusion/`: `forward.py`, `loss.py`, `sampling.py`, `schedules.py`.
7. Read all files under `src/ablations/` (if they exist; note what's there).
8. Read the full notebook `notebooks/craftax_ablations_full.ipynb`.

Only after reading all of this should you begin diagnosis.

---

## Step 1 — Diagnosis: Catalogue every problem before fixing anything

Go through the notebook cell by cell and produce a written diagnosis. For each problem found, state:
- The cell number and what the problem is
- The category: `BUG` / `LOGIC` / `INTEGRATION` / `REDUNDANCY` / `STYLE`
- Severity: `CRITICAL` (would cause wrong results or a crash) / `MODERATE` (causes confusion or silent error) / `MINOR` (cosmetic or inefficiency)

Below is the list of known problems you MUST find and address. There may be additional problems — find those too.

### Known bugs and problems to fix

**CRITICAL — Signature mismatch: `collect_rollout` calls in the extended ablation cells**

`collect_rollout` is defined in Cell 10 with the signature:
```python
collect_rollout(env_state, obs, done, hstate, rng) -> (env_state, obs, done, hstate, rng, flat_obs, flat_acts, flat_valid, flat_returns, env_score)
```
But Cells 37 and 38 call it as:
```python
collect_rollout(roll_rng, pretrained_params)   # Cell 37
collect_rollout(buf_rng, pretrained_params)    # Cell 38
```
This is completely wrong — wrong argument order, wrong number of arguments, wrong types. These cells would crash immediately.

**CRITICAL — Signature mismatch: `run_ablation` calls in the extended ablation cells**

`run_ablation` in Cell 13 has the signature:
```python
run_ablation(name, params_init, loss_fn, frozen_backbone=False, wins_only=False)
```
But Cells 37–42 call it with a totally different signature:
```python
run_ablation(loss_fn, pretrained_params, collect_rollout, apply_eval, apply_train,
             schedule_fn, schedule_deriv_fn, num_actions, PLAN_HORIZON, env, env_params,
             rng, lr=..., batch_size=..., ...)
```
This would crash on every extended ablation. The calls look like they were written for a `run_ablation_v2` that was imported but never matched up with the actual function.

**CRITICAL — `make_empty_history` defined twice**

Defined in Cell 10 (minimal version, 6 keys) and redefined in Cell 13 (full version, 17 keys). The second definition silently overwrites the first. Any history dict created between Cells 10 and 13 — specifically the one used for the baseline eval — uses the wrong schema. Fix by removing the Cell 10 definition entirely.

**CRITICAL — `load_direct_checkpoint` in Cell 8 is redundant and overwrites correct result**

Cell 7 correctly calls `src.planners.model.load_checkpoint` and stores the result in `pretrained_params`. Cell 8 then defines and immediately calls a hand-rolled `load_direct_checkpoint` that overwrites `pretrained_params` with a potentially different loading path. This is an integration failure — there is no reason for Cell 8 to exist. The project already has `load_checkpoint` for exactly this purpose. Delete Cell 8 entirely. If there was a reason `load_checkpoint` was insufficient (e.g. a checkpoint format difference), that belongs as a fix to `src/planners/model.py`, not as an inline workaround in the notebook.

**CRITICAL — `orjson` dependency in Cell 49**

`orjson` is not listed in `environment.yaml` or `requirements.txt`. Cell 49 imports it and will crash on any environment that doesn't happen to have it installed. Replace with the standard `json` module, which is already used correctly in Cell 31.

**MODERATE — KL loss uses `rng` for two independent forward passes without splitting**

In `make_loss_kl` (Cell 11), the structure is:
```python
def loss(params, acts, obs, valid, rng, advantages):
    rl = _base_loss(apply_fn, params, rng, ...)        # consumes rng
    rng2, t_rng, mask_rng, drop_rng = jax.random.split(rng, 4)  # reuses same rng!
```
`rng` is passed to `_base_loss` and then split again from the same key. The `rng2` variable is then never used. Both passes will sample correlated noise. The fix is to split `rng` into two keys before either call: one for the RL loss and one for the KL computation.

**MODERATE — `compute_grad_alignment` computes BC gradient w.r.t. `ref_params`, not `params`**

```python
bc_grads = jax.grad(bc_loss)(ref_params)  # uses ref_params as oracle proxy
```
This computes the gradient of the BC loss at the pretrained parameters, not at the current parameters. The stated intent is "cosine similarity between RL gradient and BC gradient" — both should be evaluated at the same point (current `params`) to be geometrically meaningful. Using `ref_params` for the BC side means the comparison is between the current model's RL gradient and the pretrained model's BC gradient, which conflates two separate things. Fix: compute both gradients w.r.t. `params`.

**MODERATE — Frozen backbone gradient masking uses a fragile hard-coded layer name**

```python
if 'Dense_5' in path_str:
    return grad
```
`Dense_5` is a Flax auto-generated name that depends on the order modules are defined in `DenoisingTransformer.setup()`. If the model architecture changes, this silently stops masking the right layer. Fix: look up the actual name of the output head in `src/models/denoiser.py` and either use the correct name or, better, parameterise it as a configurable prefix. Add a `assert` that verifies at least one parameter was kept (non-zero gradient) to catch silent failures.

**MODERATE — Config cell re-declares all hyperparameters instead of loading from `defaults.yaml`**

The Config cell (Cell 4) hardcodes dozens of hyperparameters that already exist in `configs/defaults.yaml` — `plan_horizon`, `d_model`, `n_heads`, `n_layers`, `d_ff`, `dropout_rate`, `diffusion_schedule`, `remask_strategy`, `eta`, `use_loop`, `t_on`, `t_off`, `temperature`, `top_p`, `lr`, `max_grad_norm`, `collect_temperature`, `return_weight_cap`, `ppo_model_type`, `layer_size`. This means the notebook can silently diverge from the project defaults. Fix: load `defaults.yaml` with PyYAML at the top of the Config cell and use its values as the base, with only the notebook-specific overrides (checkpoint paths, `max_iter`, `eval_every`) defined locally.

**MODERATE — Output files saved to the wrong location**

Cell 30 saves `craftax_full_analysis.png` to the current working directory. Cell 31 saves `craftax_full_results.json` to the current working directory. The project has defined `notebooks/ablation_results/` as the canonical output location (in `CLAUDE.md` and `configs/ablations.yaml`). Fix: route all saved files through `OUTPUT_DIR` / `FIG_DIR` / `TABLE_DIR`.

**MODERATE — Duplicate `sys.path` manipulation**

Cell 2 already sets up the path. Cell 34 adds another `sys.path.insert` with `pathlib.Path.cwd().parent`. After moving the notebook to `notebooks/`, `cwd()` may point to `notebooks/` rather than the repo root, making the path insertion inconsistent. Fix: remove the second path manipulation and make Cell 2's path setup robust to the notebook running from `notebooks/` by using the `REPO` variable already computed there.

**MODERATE — `TRAIN_SIGMA`, `KL_COEF`, `WIN_THRESHOLD`, `RETURN_WEIGHT_CAP/FLOOR` used as closure captures in loss functions**

These module-level constants are silently captured in the loss function closures. If someone changes them after defining the loss function, the closure still holds the old value. Fix: pass them as explicit arguments to the loss factory functions.

**MINOR — Synthetic sanity check eval x-axis is computed incorrectly**

```python
ax6.plot(range(EVAL_EVERY, EVAL_EVERY * (len(synth_history['recovery_rate'])+1), EVAL_EVERY), ...)
```
The synthetic history doesn't record `eval_iter` (unlike the main histories), so the x-axis is reconstructed with this formula, which is off-by-one prone. Fix: add `eval_iter` to `synth_history` the same way the main `run_ablation` does.

**MINOR — `valid_per_rollout` computation is wrong**

```python
valid_per_rollout = NUM_STEPS - PLAN_HORIZON + 1
```
This computes the number of valid non-overlapping windows per environment, but the actual number of windows extracted per rollout is `valid_per_rollout * NUM_ENVS`. The print statement says "windows/rollout" but then correctly says "samples/rollout" for the product. The variable name is misleading — rename to `windows_per_env` to prevent confusion.

**MINOR — `all_histories` dict defined twice in Cell 29 and Cell 30**

The same `all_histories` dict is constructed identically in both cells. Extract it once before Cell 29.

**MINOR — Inconsistent figure output format**

Cell 30 saves as `.png` (raster). `CLAUDE.md` and `notebooks.md` specify PDF (vector). Fix: save as `.pdf`.

---

## Step 2 — Fix everything

After diagnosis, fix all identified problems in the following order:

### 2a. Config integration first

Modify Cell 4 (Config) to:
1. Load `configs/defaults.yaml` using PyYAML into a `cfg` dict.
2. Load `configs/ablations.yaml` (if it exists) into `abl_cfg`.
3. Keep only these items as local overrides in the cell:
   - `PRETRAINED_CKPT_DIR` — path, not in defaults
   - `PPO_CKPT_PATH` — path, not in defaults
   - `USE_WANDB` — experiment-specific toggle
   - `MAX_ITER` — ablation-specific, not in defaults
   - `EVAL_EVERY`, `EVAL_STEPS`, `EVAL_REPLAN` — ablation-specific
4. For everything else, read from `cfg`: `PLAN_HORIZON = cfg['plan_horizon']`, `LR = cfg['lr']`, etc.
5. Construct the `config` dict for `make_env` / `build_model` directly from `cfg` rather than re-declaring every key manually.

### 2b. Remove Cell 8 entirely

Delete the `load_direct_checkpoint` cell. Cell 7's `load_checkpoint` call is the correct integration point. If `load_checkpoint` needs changes to handle the checkpoint format, fix it in `src/planners/model.py` and document why.

### 2c. Fix the two critical signature mismatches

The extended ablation cells (37–42) call `collect_rollout` and `run_ablation` with completely wrong signatures.

Fix `collect_rollout` calls: these cells need a single batch of PPO data for seeding (EWC Fisher, offline buffer). The correct pattern is:
```python
# Initialise env state properly, then call collect_rollout with the right args
rng, env_rng = jax.random.split(rng)
_obs, _es = env.reset(env_rng, env_params)
_done = jnp.zeros(NUM_ENVS, dtype=bool)
_hstate = ppo.init_hidden(NUM_ENVS)
rng, roll_rng = jax.random.split(rng)
_es, _obs, _done, _hstate, rng, flat_obs, flat_acts, flat_valid, flat_returns, _ = \
    collect_rollout(_es, _obs, _done, _hstate, roll_rng)
```

Fix `run_ablation` calls: the extended cells were written expecting `run_ablation_v2` which takes the function objects as arguments. Since `run_ablation` in Cell 13 already has all those objects in its closure (env, apply_eval, apply_train, etc.), the extended cells should use the same `run_ablation` signature as Cells 17–25:
```python
history_ewc, score_ewc, _ = run_ablation(
    name='EWC',
    params_init=jax.tree.map(jnp.array, pretrained_params),
    loss_fn=loss_ewc,
    frozen_backbone=False,
    wins_only=False,
)
```

### 2d. Fix all other bugs

In the order they appear in the diagnosis above. For each fix:
- Make the minimal change that resolves the problem
- Add a comment explaining what was wrong and what was fixed, if non-obvious

### 2e. Consolidate the two halves of the notebook

The original four ablations (Cells 16–25) and the extended ablations (Cells 35–48) currently use different patterns and different `run_ablation` signatures. After fixing the signature bugs, both halves should use identical call patterns. Remove all redundant infrastructure from the first half that was superseded by the second half, and remove the awkward "backward compatibility" comment in Cell 33.

The final structure should follow the ordering specified in `.claude/rules/notebooks.md`:
```
## 0. Setup
## 1. Config
## 2. Environment, Model, PPO
## 3. Shared Infrastructure
## 4. Baseline Evaluation
## 5. Ablation 0: Baseline RL
## 6. Ablation 1: KL Penalty
## 7. Ablation 2: Frozen Backbone
## 8. Ablation 3: BC on Wins
## 9. Ablation 4: Low-t Only
## 10. Extended Ablations
## 11. Synthetic Sanity Check
## 12. Summary and Analysis
## 13. Visualisations
## 14. Correlation Analysis
## 15. Failure Mode Taxonomy
## 16. Save Outputs
```

---

## Step 3 — Integration review

After fixing bugs, verify that the notebook makes proper use of project functions rather than re-implementing them:

- **`build_ppo_network` + `load_ppo_params`**: Cell 6 already uses these correctly. Verify the arguments match the current signatures in `src/planners/ppo.py` exactly.

- **`build_model`, `init_params`, `create_train_state`, `make_apply_fns`**: Cell 6 already uses these. Verify arguments match `src/planners/model.py` exactly, particularly that `create_train_state` is called with the right argument names and order.

- **`make_grad_step` from `src/planners/common.py`**: The notebook defines its own `make_ablation_grad_step`. Read `make_grad_step` in `common.py` and determine if `make_ablation_grad_step` could use it as a base or delegate to it. If `make_grad_step` needs a `frozen_backbone` flag added, add it there and use it from the notebook. If the notebook version is genuinely different, document why.

- **`compute_loss` from `src/diffusion/loss.py`**: The notebook's `_base_loss` function reimplements MDLM loss logic from scratch. Read `compute_loss` in `src/diffusion/loss.py`. If `_base_loss` is computing the same thing, replace it with a call to `compute_loss`. If there are small differences (e.g. the configurable `t_min`/`t_max` range), check if `compute_loss` can be extended to support these parameters. A `t_min` / `t_max` parameter would be a clean addition to `compute_loss` with defaults of `0` and `1` for full backward compatibility.

- **`sample_plan` from `src/diffusion/sampling.py`**: `eval_policy` already calls this correctly. Verify the keyword arguments match the current function signature.

- **`SCHEDULE_MAP` from `src/diffusion/schedules.py`**: Used correctly. Verify the schedule name string `"cosine"` is a valid key in `SCHEDULE_MAP`.

- **W&B logging**: If `USE_WANDB` is True, metric logging should go through `src/planners/logging.py`. Currently the notebook logs nothing to W&B despite having `USE_WANDB = True` in the config. Either wire up the existing logging utilities or set `USE_WANDB = False` and add a TODO comment.

---

## Step 4 — Final quality checks

After all fixes and integrations:

1. Verify the notebook runs cleanly from Cell 0 to Cell 2 (setup) without errors on a machine with the project installed. You cannot run the full training, but you can verify imports, path setup, and environment construction.

2. Verify Cell 4 (Config) produces a `cfg` dict that exactly matches what `main.py` would produce for `--mode offline` with default arguments. They should be reading the same YAML source.

3. Verify that every `plt.savefig` call routes to `FIG_DIR` and saves as `.pdf`.

4. Verify there is no `plt.show()` call inside any function in `src/`. (Only in notebook cells — that is fine.)

5. Verify there is no `print()` call inside any function in `src/` other than `__main__` blocks.

6. Verify the notebook cell outputs are cleared (no embedded figures or stdout in the `.ipynb` JSON).

7. Check the `results.json` save cell — it should use the standard `json` module with `default=float` for numpy/JAX scalar conversion, not `orjson`.

---

## Guiding principles for this task

- **Minimal change**: fix what is broken, do not refactor things that work. If a cell is ugly but correct, leave it.
- **No new features**: this is not the place to add new ablations, new diagnostics, or new plots. Those belong in a separate task.
- **Prefer project functions**: if `src/` already has a function that does what an inline notebook cell does, use the project function. If the project function needs a small, safe extension to support the notebook's use case, make that extension.
- **Explain fixes**: for every non-trivial change, add a brief comment in the notebook cell stating what was wrong and what the fix is. Future readers (including you) should understand the history.
