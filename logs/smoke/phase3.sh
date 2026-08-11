#!/bin/bash
# Phase 3: memory behaviour at QMUL sizes (64 envs / batch 256).
# XLA_PYTHON_CLIENT_PREALLOCATE=false is a measurement probe only.
cd /cs/student/project_msc/2025/dsml/mathweil/craftax-ReMDM-planner || exit 1

CKPT=checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M/
PPO=checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M
QCFG=experiments/rl_finetuning/configs/ablations_final_classic_qmul.yaml
OUT=/cs/student/project_msc/2025/dsml/mathweil/mem
VR=/cs/student/project_msc/2025/dsml/mathweil/tmp/vram

mkdir -p "$VR"

ABL="baseline_rl kl_penalty ewc llrd lora mixed_replay trust_region_kl t_curriculum \
entropy_bonus gradient_surgery advantage_clip normalized_adv bc_wins low_t \
frozen_backbone head_only attention_only ffn_only layer_ablation_top1 \
layer_ablation_top2 layer_ablation_top3 reward_filtering running_stats \
action_diversity reward_model"

: > logs/smoke/mem_status.log

for a in $ABL; do
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 2; done ) > "$VR/$a.txt" &
  SAMPLER=$!
  SECONDS=0
  XLA_PYTHON_CLIENT_PREALLOCATE=false uv run --extra cuda12 python experiments/rl_finetuning/run_ablations.py \
    --checkpoint "$CKPT" --ppo-checkpoint "$PPO" --ablations-config "$QCFG" \
    --ablations "$a" --max-iter 3 --eval-every 1 --num-seeds 1 --no-use-wandb \
    --output-dir "$OUT/$a" 2>&1 | tee "logs/smoke/mem_$a.log"
  ST=${PIPESTATUS[0]}
  kill $SAMPLER 2>/dev/null; wait $SAMPLER 2>/dev/null
  echo "$a exit=$ST peak_mib=$(sort -n "$VR/$a.txt" | tail -1) secs=$SECONDS" | tee -a logs/smoke/mem_status.log
done
echo "MEM SWEEP COMPLETE" | tee -a logs/smoke/mem_status.log
