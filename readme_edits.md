# README Edits

---

## 1. Replace lines 9–28 (Description + pipeline diagram)

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `T` denoising steps, producing a `plan_horizon`-length plan. The ReMDM framework extends standard Masked Discrete Language Modelling (MDLM) with remasking strategies that allow committed tokens to be re-predicted, improving plan coherence.

Two independent training pipelines are available — **Offline BC** and **Online DAgger** — both supervised by a pre-trained PPO expert but otherwise separate. Neither depends on the other; the paper compares them head-to-head.

```
[Shared]   Train PPO agent              Craftax_Baselines/ppo_rnn.py | ppo_rnd.py
               |
               v  checkpoint
       ┌───────┴────────┐
       │                │
  [Offline BC]     [Online DAgger]
  main.py              main.py
  --mode offline        --mode online
  (train on live        (train from scratch;
   PPO rollouts)         mixed policy + expert
       │                 labels into replay buffer)
       v                 v
   checkpoint        checkpoint
       │                │
       └───────┬────────┘
               v
[Evaluate] main.py --mode inference --checkpoint_path ...

Optional: an offline BC checkpoint can warm-start DAgger
via --offline_checkpoint_path (not used in the paper).

  [Offline BC] ──checkpoint──> [Online DAgger]
```

**Optional utility modes:**
```
[Collect]     Save PPO rollouts to disk   main.py --mode collect
[Smoke test]  Quick end-to-end check      main.py --mode smoke
```

---

## 2. Replace lines 150–167 (Stage 3 heading + description + examples)

### Online DAgger Training

The diffusion model is trained **from scratch** via DAgger (Dataset Aggregation). At each iteration a mixed policy blends the PPO expert and the diffusion learner (controlled by an exponentially decaying `beta`). The mixed policy rolls out trajectories; the expert labels every visited state with the action it would take. These `(obs, expert_plan)` pairs are appended to a growing circular replay buffer, and the diffusion model is trained on the full buffer with the standard MDLM ELBO loss (pure behavioural cloning — no reward weighting).

```bash
# From scratch (requires PPO expert checkpoint)
python main.py --mode online \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --online_num_updates 1000 \
    --save_policy

# Optional: warm-start from a pre-trained offline checkpoint
# (not used in the paper — both methods are compared independently)
python main.py --mode online \
    --ppo_checkpoint_path /path/to/ppo_checkpoint \
    --offline_checkpoint_path /path/to/offline_checkpoint \
    --online_num_updates 1000 \
    --save_policy
```
