#!/bin/csh

while ( `nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l` > 0 )
    sleep 60
end

env CUDA_VISIBLE_DEVICES=0 python main.py --mode online --config configs/exp_a_craftax_full.yaml --ppo_checkpoint_path checkpoints/ppo_agents/policies/Craftax-Symbolic-v1-PPO_RNN-1000M --checkpoint_dir checkpoints_exp_a_full
