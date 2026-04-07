# RL Fine-Tuning Ablation Suite — Diagnosis Report

**Pretrained baseline score:** 10.0955
**Baseline RL score:** 10.3813

---

## Primary Failure Mode

**16/25 ablations** achieved scores near or above the pretrained baseline.

**Most likely failure mode:** Gradient Conflict
> RL and BC gradients point in conflicting directions, cancelling useful updates.

**Evidence strength:** 67% (2/3 supporting ablations succeeded)

**Ablations that support this hypothesis:** gradient_surgery, kl_penalty

---

## Hypothesis Rankings (by Evidence Strength)

| Hypothesis | Evidence | Supporting Ablations |
|---|---|---|
| Gradient Conflict | 67% (2/3) | gradient_surgery, kl_penalty |
| Catastrophic Forgetting | 60% (3/5) | kl_penalty, ewc, llrd |
| Signal Sparsity | 50% (2/4) | reward_filtering, running_stats |
| Distributional Shift | 50% (1/2) | action_diversity |
| t-Bias | 50% (1/2) | t_curriculum |
| Mode Collapse | 33% (1/3) | entropy_bonus |

---

## Evidence Details per Ablation

### action_diversity  [IMPROVEMENT]
- **Score:** 10.5622  (delta vs pretrained: +0.4667)
- **Hypothesis tested:** If diversity filtering helps: degenerate PPO plans corrupt training
- **Mean grad alignment:** +0.0352
- **Final KL drift:** 0.161787

### reward_filtering  [IMPROVEMENT]
- **Score:** 10.5125  (delta vs pretrained: +0.4170)
- **Hypothesis tested:** If filtering helps: noisy/low-return data poisons gradients
- **Mean grad alignment:** +0.0361
- **Final KL drift:** 0.141319

### gradient_surgery  [IMPROVEMENT]
- **Score:** 10.5012  (delta vs pretrained: +0.4058)
- **Hypothesis tested:** If PCGrad helps: gradients are conflicting and resolvable by projection
- **Mean grad alignment:** +0.0338
- **Final KL drift:** 0.161617

### ewc  [IMPROVEMENT]
- **Score:** 10.4972  (delta vs pretrained: +0.4017)
- **Hypothesis tested:** If EWC helps: forgetting pretrained representations is the proximate cause
- **Mean grad alignment:** +0.0166
- **Final KL drift:** 0.140858

### llrd  [IMPROVEMENT]
- **Score:** 10.4760  (delta vs pretrained: +0.3805)
- **Hypothesis tested:** If LLRD helps: deep gradient flow into early layers corrupts representations
- **Mean grad alignment:** +0.0371
- **Final KL drift:** 0.141797

### running_stats  [IMPROVEMENT]
- **Score:** 10.4160  (delta vs pretrained: +0.3205)
- **Hypothesis tested:** If running stats help: batch normalisation is too noisy for small batches
- **Mean grad alignment:** +0.0234
- **Final KL drift:** 0.160108

### entropy_bonus  [IMPROVEMENT]
- **Score:** 10.4019  (delta vs pretrained: +0.3065)
- **Hypothesis tested:** If entropy bonus helps: collapse is mode-collapse; not a gradient problem
- **Mean grad alignment:** +0.0482
- **Final KL drift:** 0.140658

### t_curriculum  [IMPROVEMENT]
- **Score:** 10.4017  (delta vs pretrained: +0.3062)
- **Hypothesis tested:** If curriculum helps: ordering of learning signals matters
- **Mean grad alignment:** +0.0404
- **Final KL drift:** 0.158725

### kl_penalty  [IMPROVEMENT]
- **Score:** 10.3999  (delta vs pretrained: +0.3044)
- **Hypothesis tested:** If this helps: catastrophic forgetting is the primary cause; soft regularisation suffices
- **Mean grad alignment:** +0.0527
- **Final KL drift:** 0.133675

### baseline_rl  [IMPROVEMENT]
- **Score:** 10.3813  (delta vs pretrained: +0.2858)
- **Hypothesis tested:** Diagnoses whether the RL signal alone causes collapse
- **Mean grad alignment:** +0.0342
- **Final KL drift:** 0.161030

### mixed_replay  [IMPROVEMENT]
- **Score:** 10.3474  (delta vs pretrained: +0.2519)
- **Hypothesis tested:** If mixed replay helps: online data distribution alone is too corrupted
- **Mean grad alignment:** +0.0132
- **Final KL drift:** 0.140192

### reward_model  [IMPROVEMENT]
- **Score:** 10.3153  (delta vs pretrained: +0.2198)
- **Hypothesis tested:** If reward model helps: raw returns are too sparse; learned model smooths signal
- **Mean grad alignment:** +0.0307
- **Final KL drift:** 0.163440

### advantage_clip  [IMPROVEMENT]
- **Score:** 10.3026  (delta vs pretrained: +0.2071)
- **Hypothesis tested:** If clipping helps: large advantage magnitudes destabilise training
- **Mean grad alignment:** +0.0265
- **Final KL drift:** 0.168065

### trust_region_kl  [IMPROVEMENT]
- **Score:** 10.2338  (delta vs pretrained: +0.1383)
- **Hypothesis tested:** If hard constraint helps: soft KL is insufficient — a hard boundary is needed
- **Mean grad alignment:** +0.0655
- **Final KL drift:** 0.062820

### bc_wins  [IMPROVEMENT]
- **Score:** 10.1957  (delta vs pretrained: +0.1002)
- **Hypothesis tested:** If BC on wins helps: the return weighting is the specific cause
- **Mean grad alignment:** +0.0059
- **Final KL drift:** 0.170706

### low_t  [NEUTRAL]
- **Score:** 10.0873  (delta vs pretrained: -0.0081)
- **Hypothesis tested:** If low-t helps: high-t (coarse-structure) gradients are biased
- **Mean grad alignment:** +0.0598
- **Final KL drift:** 0.142888

### layer_ablation_top1  [COLLAPSE]
- **Score:** 2.2213  (delta vs pretrained: -7.8742)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** -0.0044
- **Final KL drift:** 4.435149

### normalized_adv  [COLLAPSE]
- **Score:** 1.6395  (delta vs pretrained: -8.4560)
- **Hypothesis tested:** If std normalisation helps: simple mean normalisation is too loose
- **Mean grad alignment:** -0.0266
- **Final KL drift:** 80.792145

### ffn_only  [COLLAPSE]
- **Score:** 1.1056  (delta vs pretrained: -8.9899)
- **Hypothesis tested:** If FFN-only works: stored knowledge (FFN as memory) needs updating; not attention
- **Mean grad alignment:** -0.0020
- **Final KL drift:** 14.420309

### attention_only  [COLLAPSE]
- **Score:** 0.2168  (delta vs pretrained: -9.8787)
- **Hypothesis tested:** If attention-only works: model needs routing updates, not feature updates
- **Mean grad alignment:** +0.0001
- **Final KL drift:** 1.053753

### head_only  [COLLAPSE]
- **Score:** 0.0790  (delta vs pretrained: -10.0165)
- **Hypothesis tested:** If head-only works: backbone representations are fine; only decision boundary needs updating
- **Mean grad alignment:** -0.0016
- **Final KL drift:** 34.089348

### layer_ablation_top2  [COLLAPSE]
- **Score:** -0.0198  (delta vs pretrained: -10.1153)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** -0.0072
- **Final KL drift:** 2.569427

### layer_ablation_top3  [COLLAPSE]
- **Score:** -0.0612  (delta vs pretrained: -10.1567)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0031
- **Final KL drift:** 4.177799

### frozen_backbone  [COLLAPSE]
- **Score:** -0.1766  (delta vs pretrained: -10.2721)
- **Hypothesis tested:** If frozen backbone helps: deep gradient flow into backbone causes collapse
- **Mean grad alignment:** -0.0018
- **Final KL drift:** 33.133343

### lora  [COLLAPSE]
- **Score:** -0.9148  (delta vs pretrained: -11.0103)
- **Hypothesis tested:** If LoRA works: too many unconstrained degrees of freedom cause collapse
- **Mean grad alignment:** +0.0301
- **Final KL drift:** 1839625.500000

---

## Recommendations

### Gradient Conflict
Apply PCGrad in the full training pipeline, and investigate whether the t-distribution of RL batches is biased.

### Catastrophic Forgetting
Implement a strong parameter regularisation regime (EWC + LLRD) or use LoRA to restrict the parameter update space.

---

## Next Experiments

Based on the pattern of successes and failures above, the highest-priority
follow-up experiments are:

1. **Deep dive on Gradient Conflict**: run multi-seed experiments with the best-performing ablations (gradient_surgery, kl_penalty) and tune their hyperparameters.
2. **Combine top-2 interventions**: Gradient Conflict + Catastrophic Forgetting — run a combined ablation.
3. **Increase num_envs and num_seeds**: current results may be high-variance. Confirm findings with num_seeds=3.
4. **Profile the reward signal**: plot the return histogram over training to determine if rewards are collapsing to zero.
