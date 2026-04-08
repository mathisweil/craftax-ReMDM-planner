# RL Fine-Tuning Ablation Suite — Diagnosis Report

**Pretrained baseline score:** 10.5286
**Baseline RL score:** 10.8216

---

## Primary Failure Mode

**16/25 ablations** achieved scores near or above the pretrained baseline.

**Most likely failure mode:** Distributional Shift
> Online data distribution is too different from the offline pretraining distribution.

**Evidence strength:** 100% (2/2 supporting ablations succeeded)

**Ablations that support this hypothesis:** mixed_replay, action_diversity

---

## Hypothesis Rankings (by Evidence Strength)

| Hypothesis | Evidence | Supporting Ablations |
|---|---|---|
| Distributional Shift | 100% (2/2) | mixed_replay, action_diversity |
| Signal Sparsity | 75% (3/4) | bc_wins, reward_filtering, running_stats |
| t-Bias | 50% (1/2) | t_curriculum |
| Gradient Conflict | 33% (1/3) | kl_penalty |
| Mode Collapse | 33% (1/3) | entropy_bonus |
| Catastrophic Forgetting | 20% (1/5) | kl_penalty |

---

## Evidence Details per Ablation

### action_diversity  [IMPROVEMENT]
- **Score:** 11.0773  (delta vs pretrained: +0.5487)
- **Hypothesis tested:** If diversity filtering helps: degenerate PPO plans corrupt training
- **Mean grad alignment:** +0.0153
- **Final KL drift:** 0.209855

### reward_filtering  [IMPROVEMENT]
- **Score:** 11.0502  (delta vs pretrained: +0.5216)
- **Hypothesis tested:** If filtering helps: noisy/low-return data poisons gradients
- **Mean grad alignment:** +0.0208
- **Final KL drift:** 0.151835

### kl_penalty  [IMPROVEMENT]
- **Score:** 10.9880  (delta vs pretrained: +0.4594)
- **Hypothesis tested:** If this helps: catastrophic forgetting is the primary cause; soft regularisation suffices
- **Mean grad alignment:** +0.0206
- **Final KL drift:** 0.167528

### running_stats  [IMPROVEMENT]
- **Score:** 10.9616  (delta vs pretrained: +0.4330)
- **Hypothesis tested:** If running stats help: batch normalisation is too noisy for small batches
- **Mean grad alignment:** +0.0230
- **Final KL drift:** 0.218010

### entropy_bonus  [IMPROVEMENT]
- **Score:** 10.9190  (delta vs pretrained: +0.3904)
- **Hypothesis tested:** If entropy bonus helps: collapse is mode-collapse; not a gradient problem
- **Mean grad alignment:** +0.0139
- **Final KL drift:** 0.189308

### t_curriculum  [IMPROVEMENT]
- **Score:** 10.9168  (delta vs pretrained: +0.3882)
- **Hypothesis tested:** If curriculum helps: ordering of learning signals matters
- **Mean grad alignment:** +0.0152
- **Final KL drift:** 0.225474

### mixed_replay  [IMPROVEMENT]
- **Score:** 10.9151  (delta vs pretrained: +0.3865)
- **Hypothesis tested:** If mixed replay helps: online data distribution alone is too corrupted
- **Mean grad alignment:** -0.0176
- **Final KL drift:** 0.173193

### bc_wins  [IMPROVEMENT]
- **Score:** 10.8676  (delta vs pretrained: +0.3390)
- **Hypothesis tested:** If BC on wins helps: the return weighting is the specific cause
- **Mean grad alignment:** +0.0092
- **Final KL drift:** 0.219727

### gradient_surgery  [IMPROVEMENT]
- **Score:** 10.8241  (delta vs pretrained: +0.2955)
- **Hypothesis tested:** If PCGrad helps: gradients are conflicting and resolvable by projection
- **Mean grad alignment:** +0.0153
- **Final KL drift:** 0.209463

### baseline_rl  [IMPROVEMENT]
- **Score:** 10.8216  (delta vs pretrained: +0.2930)
- **Hypothesis tested:** Diagnoses whether the RL signal alone causes collapse
- **Mean grad alignment:** +0.0153
- **Final KL drift:** 0.209600

### reward_model  [IMPROVEMENT]
- **Score:** 10.8205  (delta vs pretrained: +0.2919)
- **Hypothesis tested:** If reward model helps: raw returns are too sparse; learned model smooths signal
- **Mean grad alignment:** +0.0169
- **Final KL drift:** 0.222742

### ewc  [IMPROVEMENT]
- **Score:** 10.7862  (delta vs pretrained: +0.2577)
- **Hypothesis tested:** If EWC helps: forgetting pretrained representations is the proximate cause
- **Mean grad alignment:** +0.0251
- **Final KL drift:** 0.194954

### advantage_clip  [IMPROVEMENT]
- **Score:** 10.7793  (delta vs pretrained: +0.2507)
- **Hypothesis tested:** If clipping helps: large advantage magnitudes destabilise training
- **Mean grad alignment:** -0.0014
- **Final KL drift:** 0.219086

### llrd  [IMPROVEMENT]
- **Score:** 10.7272  (delta vs pretrained: +0.1986)
- **Hypothesis tested:** If LLRD helps: deep gradient flow into early layers corrupts representations
- **Mean grad alignment:** +0.0125
- **Final KL drift:** 0.188783

### low_t  [IMPROVEMENT]
- **Score:** 10.6598  (delta vs pretrained: +0.1312)
- **Hypothesis tested:** If low-t helps: high-t (coarse-structure) gradients are biased
- **Mean grad alignment:** +0.0246
- **Final KL drift:** 0.187926

### trust_region_kl  [IMPROVEMENT]
- **Score:** 10.6206  (delta vs pretrained: +0.0920)
- **Hypothesis tested:** If hard constraint helps: soft KL is insufficient — a hard boundary is needed
- **Mean grad alignment:** +0.0555
- **Final KL drift:** 0.064425

### layer_ablation_top2  [COLLAPSE]
- **Score:** 6.3875  (delta vs pretrained: -4.1411)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0461
- **Final KL drift:** 9.672604

### layer_ablation_top3  [COLLAPSE]
- **Score:** 6.0164  (delta vs pretrained: -4.5122)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0474
- **Final KL drift:** 16.208744

### normalized_adv  [COLLAPSE]
- **Score:** 5.0739  (delta vs pretrained: -5.4547)
- **Hypothesis tested:** If std normalisation helps: simple mean normalisation is too loose
- **Mean grad alignment:** +0.0111
- **Final KL drift:** 63.240700

### layer_ablation_top1  [COLLAPSE]
- **Score:** 4.0468  (delta vs pretrained: -6.4818)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0475
- **Final KL drift:** 13.324247

### ffn_only  [COLLAPSE]
- **Score:** 1.8791  (delta vs pretrained: -8.6494)
- **Hypothesis tested:** If FFN-only works: stored knowledge (FFN as memory) needs updating; not attention
- **Mean grad alignment:** +0.0466
- **Final KL drift:** 23.891375

### attention_only  [COLLAPSE]
- **Score:** 0.5447  (delta vs pretrained: -9.9838)
- **Hypothesis tested:** If attention-only works: model needs routing updates, not feature updates
- **Mean grad alignment:** +0.0171
- **Final KL drift:** 90.191254

### frozen_backbone  [COLLAPSE]
- **Score:** 0.2878  (delta vs pretrained: -10.2408)
- **Hypothesis tested:** If frozen backbone helps: deep gradient flow into backbone causes collapse
- **Mean grad alignment:** +0.0152
- **Final KL drift:** 115.526100

### head_only  [COLLAPSE]
- **Score:** 0.2743  (delta vs pretrained: -10.2543)
- **Hypothesis tested:** If head-only works: backbone representations are fine; only decision boundary needs updating
- **Mean grad alignment:** +0.0151
- **Final KL drift:** 115.826111

### lora  [COLLAPSE]
- **Score:** -0.9103  (delta vs pretrained: -11.4389)
- **Hypothesis tested:** If LoRA works: too many unconstrained degrees of freedom cause collapse
- **Mean grad alignment:** +0.0827
- **Final KL drift:** 1679591.000000

---

## Recommendations

### Distributional Shift
Maintain a large offline replay buffer mixed into every batch, or apply importance sampling corrections.

### Signal Sparsity
Increase num_envs, use a reward shaping strategy, or apply curriculum-based episode selection.

---

## Next Experiments

Based on the pattern of successes and failures above, the highest-priority
follow-up experiments are:

1. **Deep dive on Distributional Shift**: run multi-seed experiments with the best-performing ablations (mixed_replay, action_diversity) and tune their hyperparameters.
2. **Combine top-2 interventions**: Distributional Shift + Signal Sparsity — run a combined ablation.
3. **Increase num_envs and num_seeds**: current results may be high-variance. Confirm findings with num_seeds=3.
4. **Profile the reward signal**: plot the return histogram over training to determine if rewards are collapsing to zero.
