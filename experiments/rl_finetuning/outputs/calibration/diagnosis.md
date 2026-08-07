# RL Fine-Tuning Ablation Suite — Diagnosis Report

**Pretrained baseline score:** 11.5600
**Baseline RL score:** nan

---

## Primary Failure Mode

**ALL ablations collapsed.** This is the strongest evidence for a fundamental
incompatibility between the RL fine-tuning signal and the model.

**Most likely failure mode:** Catastrophic Forgetting
> The pretrained representations are corrupted by RL gradients.

**Evidence strength:** 0% (0/1 supporting ablations succeeded)

---

## Hypothesis Rankings (by Evidence Strength)

| Hypothesis | Evidence | Supporting Ablations |
|---|---|---|
| Catastrophic Forgetting | 0% (0/1) | — |
| Gradient Conflict | 0% (0/1) | — |
| Signal Sparsity | 0% (0/0) | — |
| Distributional Shift | 0% (0/0) | — |
| Mode Collapse | 0% (0/0) | — |
| t-Bias | 0% (0/0) | — |

---

## Evidence Details per Ablation

### kl_penalty  [COLLAPSE]
- **Score:** 11.4445  (delta vs pretrained: -0.1155)
- **Hypothesis tested:** If this helps: catastrophic forgetting is the primary cause; soft regularisation suffices
- **Mean grad alignment:** -0.0035
- **Final KL drift:** 0.050974

---

## Recommendations

### Catastrophic Forgetting
Implement a strong parameter regularisation regime (EWC + LLRD) or use LoRA to restrict the parameter update space.

### Gradient Conflict
Apply PCGrad in the full training pipeline, and investigate whether the t-distribution of RL batches is biased.

---

## Next Experiments

Based on the pattern of successes and failures above, the highest-priority
follow-up experiments are:

3. **Increase num_envs and num_seeds**: current results may be high-variance. Confirm findings with num_seeds=3.
4. **Profile the reward signal**: plot the return histogram over training to determine if rewards are collapsing to zero.
