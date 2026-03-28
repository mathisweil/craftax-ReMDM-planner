# RL Fine-Tuning Ablation Suite — Diagnosis Report

**Pretrained baseline score:** 12.0493
**Baseline RL score:** 12.8540

---

## Primary Failure Mode

**14/25 ablations** achieved scores near or above the pretrained baseline.

**Most likely failure mode:** Signal Sparsity
> Returns are too sparse or noisy to provide a useful training signal.

**Evidence strength:** 100% (4/4 supporting ablations succeeded)

**Ablations that support this hypothesis:** bc_wins, reward_filtering, running_stats, reward_model

---

## Hypothesis Rankings (by Evidence Strength)

| Hypothesis | Evidence | Supporting Ablations |
|---|---|---|
| Signal Sparsity | 100% (4/4) | bc_wins, reward_filtering, running_stats, reward_model |
| Distributional Shift | 100% (2/2) | mixed_replay, action_diversity |
| Gradient Conflict | 67% (2/3) | gradient_surgery, kl_penalty |
| Mode Collapse | 67% (2/3) | entropy_bonus, advantage_clip |
| Catastrophic Forgetting | 60% (3/5) | kl_penalty, ewc, llrd |
| t-Bias | 0% (0/2) | — |

---

## Evidence Details per Ablation

### llrd  [IMPROVEMENT]
- **Score:** 12.9572  (delta vs pretrained: +0.9079)
- **Hypothesis tested:** If LLRD helps: deep gradient flow into early layers corrupts representations
- **Mean grad alignment:** +0.0200
- **Final KL drift:** 0.123600

### bc_wins  [IMPROVEMENT]
- **Score:** 12.8671  (delta vs pretrained: +0.8178)
- **Hypothesis tested:** If BC on wins helps: the return weighting is the specific cause
- **Mean grad alignment:** +0.0247
- **Final KL drift:** 0.137008

### baseline_rl  [IMPROVEMENT]
- **Score:** 12.8540  (delta vs pretrained: +0.8047)
- **Hypothesis tested:** Diagnoses whether the RL signal alone causes collapse
- **Mean grad alignment:** +0.0169
- **Final KL drift:** 0.139703

### reward_filtering  [IMPROVEMENT]
- **Score:** 12.7864  (delta vs pretrained: +0.7371)
- **Hypothesis tested:** If filtering helps: noisy/low-return data poisons gradients
- **Mean grad alignment:** +0.0272
- **Final KL drift:** 0.129017

### ewc  [IMPROVEMENT]
- **Score:** 12.6798  (delta vs pretrained: +0.6306)
- **Hypothesis tested:** If EWC helps: forgetting pretrained representations is the proximate cause
- **Mean grad alignment:** +0.0180
- **Final KL drift:** 0.140318

### action_diversity  [IMPROVEMENT]
- **Score:** 12.6593  (delta vs pretrained: +0.6101)
- **Hypothesis tested:** If diversity filtering helps: degenerate PPO plans corrupt training
- **Mean grad alignment:** +0.0169
- **Final KL drift:** 0.139664

### running_stats  [IMPROVEMENT]
- **Score:** 12.6333  (delta vs pretrained: +0.5840)
- **Hypothesis tested:** If running stats help: batch normalisation is too noisy for small batches
- **Mean grad alignment:** +0.0189
- **Final KL drift:** 0.139692

### trust_region_kl  [IMPROVEMENT]
- **Score:** 12.4493  (delta vs pretrained: +0.4000)
- **Hypothesis tested:** If hard constraint helps: soft KL is insufficient — a hard boundary is needed
- **Mean grad alignment:** +0.0141
- **Final KL drift:** 0.056624

### kl_penalty  [IMPROVEMENT]
- **Score:** 12.4189  (delta vs pretrained: +0.3696)
- **Hypothesis tested:** If this helps: catastrophic forgetting is the primary cause; soft regularisation suffices
- **Mean grad alignment:** +0.0219
- **Final KL drift:** 0.127981

### advantage_clip  [IMPROVEMENT]
- **Score:** 12.4117  (delta vs pretrained: +0.3624)
- **Hypothesis tested:** If clipping helps: large advantage magnitudes destabilise training
- **Mean grad alignment:** +0.0141
- **Final KL drift:** 0.137774

### gradient_surgery  [IMPROVEMENT]
- **Score:** 12.3924  (delta vs pretrained: +0.3432)
- **Hypothesis tested:** If PCGrad helps: gradients are conflicting and resolvable by projection
- **Mean grad alignment:** +0.0169
- **Final KL drift:** 0.139680

### mixed_replay  [IMPROVEMENT]
- **Score:** 12.3638  (delta vs pretrained: +0.3146)
- **Hypothesis tested:** If mixed replay helps: online data distribution alone is too corrupted
- **Mean grad alignment:** -0.0026
- **Final KL drift:** 0.129276

### entropy_bonus  [IMPROVEMENT]
- **Score:** 12.3347  (delta vs pretrained: +0.2854)
- **Hypothesis tested:** If entropy bonus helps: collapse is mode-collapse; not a gradient problem
- **Mean grad alignment:** +0.0214
- **Final KL drift:** 0.131264

### reward_model  [IMPROVEMENT]
- **Score:** 12.2380  (delta vs pretrained: +0.1887)
- **Hypothesis tested:** If reward model helps: raw returns are too sparse; learned model smooths signal
- **Mean grad alignment:** +0.0202
- **Final KL drift:** 0.140539

### t_curriculum  [COLLAPSE]
- **Score:** 11.8874  (delta vs pretrained: -0.1619)
- **Hypothesis tested:** If curriculum helps: ordering of learning signals matters
- **Mean grad alignment:** +0.0159
- **Final KL drift:** 0.142742

### low_t  [COLLAPSE]
- **Score:** 11.7222  (delta vs pretrained: -0.3271)
- **Hypothesis tested:** If low-t helps: high-t (coarse-structure) gradients are biased
- **Mean grad alignment:** +0.0050
- **Final KL drift:** 0.124161

### normalized_adv  [COLLAPSE]
- **Score:** 7.7800  (delta vs pretrained: -4.2692)
- **Hypothesis tested:** If std normalisation helps: simple mean normalisation is too loose
- **Mean grad alignment:** -0.0223
- **Final KL drift:** 158.024277

### lora  [COLLAPSE]
- **Score:** -0.0911  (delta vs pretrained: -12.1404)
- **Hypothesis tested:** If LoRA works: too many unconstrained degrees of freedom cause collapse
- **Mean grad alignment:** +0.0074
- **Final KL drift:** 4692599.000000

### layer_ablation_top2  [COLLAPSE]
- **Score:** -0.3468  (delta vs pretrained: -12.3960)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0151
- **Final KL drift:** 6.395013

### ffn_only  [COLLAPSE]
- **Score:** -0.3507  (delta vs pretrained: -12.3999)
- **Hypothesis tested:** If FFN-only works: stored knowledge (FFN as memory) needs updating; not attention
- **Mean grad alignment:** +0.0126
- **Final KL drift:** 7.972916

### frozen_backbone  [COLLAPSE]
- **Score:** -0.3750  (delta vs pretrained: -12.4243)
- **Hypothesis tested:** If frozen backbone helps: deep gradient flow into backbone causes collapse
- **Mean grad alignment:** +0.0023
- **Final KL drift:** 323.774475

### head_only  [COLLAPSE]
- **Score:** -0.3750  (delta vs pretrained: -12.4243)
- **Hypothesis tested:** If head-only works: backbone representations are fine; only decision boundary needs updating
- **Mean grad alignment:** +0.0021
- **Final KL drift:** 325.598602

### attention_only  [COLLAPSE]
- **Score:** -0.3750  (delta vs pretrained: -12.4243)
- **Hypothesis tested:** If attention-only works: model needs routing updates, not feature updates
- **Mean grad alignment:** +0.0025
- **Final KL drift:** 312.413666

### layer_ablation_top3  [COLLAPSE]
- **Score:** -0.9034  (delta vs pretrained: -12.9527)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0134
- **Final KL drift:** 6.220894

### layer_ablation_top1  [COLLAPSE]
- **Score:** -0.9098  (delta vs pretrained: -12.9591)
- **Hypothesis tested:** Minimal unfrozen depth needed; collapse depth correlates with gradient flow depth
- **Mean grad alignment:** +0.0132
- **Final KL drift:** 8.088361

---

## Recommendations

### Signal Sparsity
Increase num_envs, use a reward shaping strategy, or apply curriculum-based episode selection.

### Distributional Shift
Maintain a large offline replay buffer mixed into every batch, or apply importance sampling corrections.

---

## Next Experiments

Based on the pattern of successes and failures above, the highest-priority
follow-up experiments are:

1. **Deep dive on Signal Sparsity**: run multi-seed experiments with the best-performing ablations (bc_wins, reward_filtering, running_stats) and tune their hyperparameters.
2. **Combine top-2 interventions**: Signal Sparsity + Distributional Shift — run a combined ablation.
3. **Increase num_envs and num_seeds**: current results may be high-variance. Confirm findings with num_seeds=3.
4. **Profile the reward signal**: plot the return histogram over training to determine if rewards are collapsing to zero.
