# Investigate Training Fairness: BC vs DAgger Sample & Compute Budget

## Problem

BC (offline) is significantly slower in steps-per-second than DAgger (online). This likely means DAgger does less gradient work per update, which undermines a fair comparison. We need to understand exactly where the asymmetry is and fix it.

## What to Do

1. **Read both files**: `src/planners/offline.py` and `src/planners/online.py` in full. Also read `src/planners/common.py` for shared `make_grad_step`.

2. **For each mode, compute and document these quantities** (trace them from the code, don't guess):
   - Env frames collected per update (`num_envs × num_steps`)
   - Training samples extracted per update (count the windows/pairs actually produced from one rollout)
   - Whether overlapping windows are used
   - `total_samples` passed to the minibatch loop
   - Minibatch size (`total_samples / num_minibatches`)
   - Gradient steps per update (`update_epochs × num_minibatches`)
   - Total gradient steps across training (`num_updates × update_epochs × num_minibatches`)

3. **Identify every asymmetry** between the two modes. For each one, state whether it is:
   - *Intentional* (inherent BC vs DAgger difference), or
   - *Incidental* (implementation choice that disadvantages one mode unfairly)

4. **Propose and apply fixes** so that the comparison is fair. Fairness means: given the same `num_updates`, `update_epochs`, `num_minibatches`, and `num_envs`, both modes should perform comparable gradient work per update. The DAgger buffer grows over time — if BC trains on N samples per update, DAgger should train on at least N samples per update too (sampled from its buffer). Remember all shapes must be static inside `jax.lax.scan`.

5. **Check the LR schedule**: both modes use cosine decay over `total_grad_steps`. If the number of actual gradient steps differs between modes (because of different `total_samples` or `num_minibatches`), the schedules won't match. Flag this if it's the case.
