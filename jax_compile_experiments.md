# Claude Code Task: JAX-Compile `experiments/rl_finetuning/`

## Goal

Refactor the RL fine-tuning ablation suite in `experiments/rl_finetuning/` to maximise JAX compilation coverage, following the architecture patterns established in `train.py`. The training loop, diagnostic collection, and evaluation should be JIT-compiled and vmappable. Python-level loops that JAX can absorb should move into `jax.lax.scan` or `jax.vmap`. Side-effects (logging, metric accumulation) must be funnelled through `jax.debug.callback` or extracted after the compiled region.

---

## Reference: patterns from `train.py` to replicate

Study `@src/planners/train.py` before making any changes. The key patterns to propagate are:

| Pattern | Where in `train.py` | What to replicate |
|---|---|---|
| **Static pre-computation outside the closure** | `make_train()` body | Move all Python-level setup (model builds, schedule construction, dataset slicing shapes) outside `run_ablation()` into a `make_run_ablation()` factory |
| **`jax.lax.scan` over update steps** | `jax.lax.scan(_update_step, ...)` | Replace Python `for` loops over training iterations in `training.py` with `jax.lax.scan` |
| **`jax.lax.scan` over epochs / minibatches** | `jax.lax.scan(_epoch, ...)` + `jax.lax.scan(_mb, ...)` | Replicate for each ablation's inner SGD loop |
| **`jax.lax.scan` over env steps** | `jax.lax.scan(_env_step, ...)` | Validation rollouts in `run_ablation()` and `_validate()` |
| **`jax.lax.cond` for conditional logic** | `jax.lax.cond(step_idx % val_interval == 0, ...)` | Eval branching, curriculum schedule switching, gradient-surgery toggle |
| **`jax.vmap` over seeds** | `jax.vmap(make_train(config))` at call site | Wrap `make_run_ablation(spec, config)` the same way — one vmap over seeds |
| **`jax.debug.callback` for logging** | `jax.debug.callback(_wandb_log, metric, step_idx)` | All W&B / print logging inside compiled regions |
| **Carry-based state** | `runner = (state, env_state, ...)` | Ablation runner carry should hold `(train_state, env_state, obs, done, hstate, rng, step_idx)` plus any ablation-specific state (EWC Fisher, KL baseline params, EMA stats) |
| **`jax.jit` at the outermost call site** | `jax.jit(jax.vmap(make_train(config)))` | `jax.jit(jax.vmap(make_run_ablation(spec, config)))` |

---

## File-by-file instructions

### `ablations/training.py` — primary target

This is where the most work is needed.

1. **Introduce `make_run_ablation(spec, config) -> Callable[[rng], AblationHistory]`**  
   Mirror `make_train`. Everything that does not depend on `rng` or on JAX-traced values moves here:
   - Environment construction (`make_env`)
   - Model and PPO network construction
   - Optimizer and LR schedule construction
   - Loss/objective factory call (`spec.loss_fn(config)`)
   - Gradient-surgery, EWC Fisher, or LoRA mask pre-computation where these are static

2. **Replace the Python training loop with `jax.lax.scan`**  
   ```python
   # BEFORE (pseudocode)
   for step in range(num_updates):
       batch = sample_batch(...)
       state, metrics = grad_step(state, batch)
       history.append(metrics)

   # AFTER
   def _update_step(runner, _):
       state, env_state, obs, done, hstate, rng, step_idx, extra = runner
       # ... trajectory collection via jax.lax.scan(_env_step, ...)
       # ... minibatch SGD via jax.lax.scan(_epoch, ...)
       # ... conditional eval via jax.lax.cond(step_idx % eval_every == 0, ...)
       # ... jax.debug.callback for logging
       return new_runner, metrics

   runner_init = (state, env_state, obs, jnp.zeros(num_envs, bool),
                  hstate, rng, 0, extra_init)
   runner_final, all_metrics = jax.lax.scan(
       _update_step, runner_init, None, num_updates
   )
   ```

3. **Ablation-specific carry extensions**  
   Each ablation may need extra state in the carry:

   | Ablation | Extra carry fields |
   |---|---|
   | `ewc` | `fisher: PyTree`, `ref_params: PyTree` |
   | `kl_penalty` / `trust_region_kl` | `ref_params: PyTree` |
   | `running_stats` | `ema_mean: float`, `ema_std: float` |
   | `mixed_replay` | `replay_buffer: Array` (circular, pre-allocated) |
   | `t_curriculum` | `t_max: float` (annealed each step) |
   | `reward_model` | `reward_model_state: TrainState` |

   All extra carry fields must be valid JAX arrays (no Python dicts or lists inside the scan body).

4. **Return `AblationHistory` from arrays, not Python list appends**  
   `AblationHistory` should be reconstructed from the stacked `all_metrics` pytree returned by `jax.lax.scan`, not populated inside the loop.

---

### `diagnostics/gradient.py`

- `compute_grad_alignment` and `compute_per_layer_norms` must accept and return JAX arrays only.
- Wrap calls in `jax.debug.callback` if they must run during a compiled step; otherwise gate them with `jax.lax.cond(step_idx % grad_align_every == 0, ...)` and return a structured dummy when skipped.
- `gradient_surgery` (PCGrad) must be a pure JAX function — no Python loops over parameter leaves. Use `jax.tree.map` and `jax.vmap` over the two gradient trees.

---

### `diagnostics/representation.py`

- `kl_drift` and `cka_similarity` must be pure JAX — no NumPy calls inside the compiled region. Convert any `np.*` calls to `jnp.*`.
- These are expensive; gate with `jax.lax.cond` and return zeros on skipped steps.

---

### `diagnostics/timestep.py`

- `t_bin_grad_norms` must use `jnp.digitize` (or equivalent) and `jax.vmap` over bins, not a Python `for` loop over bins.
- The per-t loss decomposition scan can reuse the same `jax.lax.scan` as the minibatch loop — add a `t_bin` accumulator to the metrics pytree.

---

### `ablations/optimizers.py`

- **LLRD**: build a single `optax.multi_transform` or masked optimizer *outside* the scan (in `make_run_ablation`). Do not reconstruct the optimizer inside the loop.
- **LoRA**: LoRA masks / rank decompositions are static — compute once in the factory.
- **EWC**: Fisher diagonal estimation must be a JIT-compiled function that takes `params` and a batch and returns a pytree of the same structure. Call it once before the scan and store in `runner_init`.
- **Gradient surgery (PCGrad)**: implement as a pure `optax.GradientTransformation` so it composes naturally with `optax.chain`.

---

### `run_ablations.py`

Update the call site:

```python
# One compiled, vmapped function per ablation spec
for spec in selected_specs:
    run_fn = jax.jit(jax.vmap(make_run_ablation(spec, config)))
    rngs = jax.random.split(base_rng, num_seeds)
    histories = run_fn(rngs)   # histories: AblationHistory with leading seed dim
    results[spec.name] = histories
```

If ablations are independent and memory allows, consider vmapping over ablations too (requires uniform carry shapes — feasible for groups A/B/D but likely not group C given different parameter masks).

---

## Constraints and gotchas

1. **No Python control flow on traced values.** Any `if`/`for` that depends on a JAX array must become `jax.lax.cond` / `jax.lax.scan` / `jax.lax.switch`.

2. **Pre-allocate buffers.** `mixed_replay` needs a fixed-size ring buffer (shape `[buffer_size, obs_dim + plan_horizon]`) allocated in `make_run_ablation`, not grown dynamically.

3. **`jax.debug.callback` for all side-effects.** W&B logging, `print`, and file I/O must not appear inside a `jax.lax.scan` body directly — use `jax.debug.callback`.

4. **Dummy values for conditional diagnostics.** When a diagnostic is skipped (`jax.lax.cond`), return a pytree of `jnp.zeros_like` matching the true branch output so the scan output shape is static.

5. **`jax.lax.scan` output shape is `[num_steps, ...]`.** After the scan, reshape / index as needed before passing to `analysis/`.

6. **Avoid `jax.tree.map` inside the scan body on Python containers.** Carry and output pytrees must have fixed structure. Define them as named tuples or dataclasses with fixed fields.

7. **`reward_model` ablation**: the reward MLP has its own `TrainState` — include it as a sub-field of the carry and update it inside `_update_step` using its own `grad_step`.

8. **Checkpointing**: orbax `CheckpointManager` calls live *outside* the compiled region, same as `train.py`. Extract `runner_final[0]` (the `TrainState`) after `run_fn` returns.

---

## Acceptance criteria

- [ ] `run_ablations.py --fast --ablations baseline_rl kl_penalty` compiles and runs without Python loops inside the hot path (verify with `jax.make_jaxpr`).
- [ ] All 22 ablations pass a smoke test (`--fast`).
- [ ] `jax.vmap` over seeds works for all ablations (shapes are seed-independent).
- [ ] No `jnp` / JAX calls outside JIT that force device-to-host syncs in the training loop.
- [ ] W&B logging still works via `jax.debug.callback`.
- [ ] `AblationHistory` is populated from scan outputs, not Python-loop appends.
- [ ] Analysis (`analysis/plots.py`, `analysis/tables.py`) is unchanged — it consumes `AblationHistory` after the compiled region.
