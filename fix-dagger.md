# Fix DAgger Online Training Implementation

## Context

We have a DAgger (Dataset Aggregation) implementation for training a diffusion planner (MDLM) to imitate a PPO expert in Craftax. BC currently outperforms DAgger, and code-level issues likely contribute. Your job is to find and fix them.

The main file is `src/planners/online.py`. Related files you must also read:
- `src/planners/common.py` (shared `make_grad_step`, `make_validate`, `resolve_num_updates`)
- `src/planners/offline.py` (the BC baseline — compare data handling, windowing, weighting)
- `src/planners/ppo.py` / `src/planners/env.py` (expert interface, env interface)
- `src/diffusion/sampling.py` (`sample_plan`)
- `configs/defaults.yaml` (default hyperparameters)

## DAgger Reference (Ross et al. 2011, Algorithm 3.1)

DAgger works as follows:

```
Initialize D ← ∅
Initialize π̂₁ to any policy in Π
for i = 1 to N do
    Let πᵢ = βᵢπ* + (1 − βᵢ)π̂ᵢ
    Sample T-step trajectories using πᵢ
    Get dataset Dᵢ = {(s, π*(s))} of visited states and expert actions
    Aggregate datasets: D ← D ∪ Dᵢ
    Train classifier π̂ᵢ₊₁ on D       ← NOTE: train on ALL of D, not a subsample
end for
Return best π̂ᵢ on validation
```

Key theoretical properties:
1. **Full dataset retraining**: each iteration trains on the *entire* aggregated dataset D, not a fixed-size subsample. This is what gives the no-regret guarantee (Theorem 3.1).
2. **β schedule**: Ross et al. found βᵢ = I(i=1) (expert-only on first iteration, pure learner after) often works best in practice. Exponential decay is valid but the requirement is only that the average β̄_N → 0.
3. **Deterministic expert preferred**: the theory assumes π* is a fixed deterministic mapping s → a. A stochastic expert adds label noise that compounds in the buffer.
4. **Expert labels every visited state**: the expert is queried at every state the mixed policy visits, not just at cycle boundaries.

## Known / Suspected Bugs to Verify

Check each of the following against the code. Confirm whether it is a real bug (i.e. deviates from correct DAgger in a way that hurts performance), then fix it if so.

### Bug 1: Subsampling from buffer instead of training on full dataset
The code samples `total_samples` (= `n_cycles * num_envs`) uniformly from the replay buffer each update. As the buffer grows, the model sees a shrinking fraction of D per update. DAgger requires training on all of D. Fix: scale the number of training samples to `buf_fill` (capped at some practical max), or increase `update_epochs` proportionally, so effective coverage of D stays high.

### Bug 2: Stochastic expert labels
`expert_act` is sampled via `jax.random.categorical(rng, pi.logits)`. Two queries to the same state can yield different labels, injecting noise into D. Fix: add a config flag `DAGGER_EXPERT_DETERMINISTIC` (default True). When set, use `jnp.argmax(pi.logits, axis=-1)` instead.

### Bug 3: Expert only queried at plan-horizon granularity
The code collects one `(obs, expert_plan)` per cycle of `plan_horizon` steps. The expert actions within a cycle are collected step-by-step (good), but only the cycle-start observation is stored. Intermediate states visited during the cycle are discarded. This means the dataset is much sparser than it could be. Compare this with how `offline.py` handles windowing — BC likely gets overlapping windows. Consider whether storing intermediate `(obs_t, expert_plan_from_t)` pairs is feasible and correct.

### Bug 4: No prioritisation of recent or high-quality data
Uniform sampling from a circular buffer means early, low-quality data (collected under near-random learner policy) gets the same weight as later, higher-quality data. Consider adding either:
- Recency weighting (sample more recent data with higher probability)
- Valid-fraction weighting (upweight samples where the expert was more confident / the trajectory was more successful)

### Bug 5: Validity masking may be too aggressive or too lenient
Check exactly how `cycle_valid` works (line ~404). It marks entire cycles invalid if *any* done occurs. This is correct for plan-level validity, but check: are invalid samples still written to the buffer? Are they correctly handled during training (the `val_b` multiplier in `grad_step`)?

## Additional Analysis Required

Beyond the suspected bugs above, do a thorough read of the code looking for:

1. **Shape mismatches or silent broadcasting errors** — especially in the scan bodies `_plan_and_execute` and `_sim_step`. Trace tensor shapes through carefully.

2. **PPO hidden state handling** — the PPO is an RNN. Check that `ppo_hs` and `prev_done` are threaded correctly through cycles and across updates. A bug here means the expert gives incoherent actions after episode resets.

3. **Buffer write logic** — verify the circular buffer indexing is correct. Check for off-by-one errors in `buf_write_idx` and `buf_fill` updates.

4. **Beta schedule** — verify `beta = beta_init * beta_decay^step_idx` decays correctly and that `step_idx` increments properly (especially with resume).

5. **Learning rate schedule alignment** — the cosine schedule is computed over `total_grad_steps`. Verify this matches the actual number of gradient steps taken, especially when resuming from a checkpoint.

6. **Comparison with offline.py** — read the BC training loop carefully. Identify every structural difference in data handling (windowing, weighting, sampling, masking). Document which differences are intentional (BC vs DAgger) and which look like bugs or missed optimisations.

7. **Gradient step interface** — check that `grad_step` receives correct inputs. The `advantages` field is set to `jnp.ones(...)` — is this the right interface? Does `grad_step` use the validity mask (`val_b`) correctly as a per-sample weight?

## Execution Plan

1. **Read all files** listed above. Do not start making changes until you have read everything.
2. **Document every confirmed bug** with file, line number, what's wrong, and what the fix is.
3. **Document any new issues** you find beyond the list above.
4. **Plan all changes** — write out the full list of edits before applying any.
5. **Apply changes** to `src/planners/online.py` (and any other files if needed).
6. **Verify** the changes don't break the scan/JIT structure (shapes must be static, no Python-level conditionals on traced values, buffer sizes must remain fixed).

Important constraint: all training runs inside `jax.lax.scan`, so array shapes cannot change dynamically. Any fix that scales training samples with buffer fill must use static max sizes with masking, not dynamic reshaping.
