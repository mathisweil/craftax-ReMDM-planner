#!/bin/bash
# Phase 1: structural sweep, 25 ablations, one process each.
cd /cs/student/project_msc/2025/dsml/mathweil/craftax-ReMDM-planner || exit 1

CKPT=checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M/
PPO=checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M
CFG=experiments/rl_finetuning/configs/ablations_final_classic_ucl.yaml
OUT=/cs/student/project_msc/2025/dsml/mathweil/smoke2

ABL="baseline_rl kl_penalty ewc llrd lora mixed_replay trust_region_kl t_curriculum \
entropy_bonus gradient_surgery advantage_clip normalized_adv bc_wins low_t \
frozen_backbone head_only attention_only ffn_only layer_ablation_top1 \
layer_ablation_top2 layer_ablation_top3 reward_filtering running_stats \
action_diversity reward_model"

: > logs/smoke/status2.log

for a in $ABL; do
  SECONDS=0
  uv run --extra cuda12 python experiments/rl_finetuning/run_ablations.py \
    --checkpoint "$CKPT" --ppo-checkpoint "$PPO" --ablations-config "$CFG" \
    --ablations "$a" --fast --num-seeds 1 --no-use-wandb \
    --output-dir "$OUT/$a" 2>&1 | tee "logs/smoke/rerun_$a.log"
  echo "$a exit=${PIPESTATUS[0]} secs=$SECONDS" | tee -a logs/smoke/status2.log
done
echo "SWEEP COMPLETE" | tee -a logs/smoke/status2.log
