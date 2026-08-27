# ReMDM Planner for Craftax

JAX implementation of **ReMDM** (Remasking Discrete Diffusion Model) for action-sequence planning in [Craftax](https://github.com/MichaelTMatthews/Craftax), a JAX-accelerated, procedurally generated open-world survival game. A bidirectional transformer generates `plan_horizon`-length action plans by iteratively denoising masked token sequences, conditioned on the current symbolic observation. Trained under a pre-trained PPO expert, either offline (behavioural cloning on live rollouts) or online (DAgger).

The sibling repository [`minihack-ReMDM-planner`](../minihack-ReMDM-planner) implements the same method in PyTorch on MiniHack. Both repos share the same CLI, config layout and README structure; commands transfer between them by swapping the repo name and benchmark-specific values.

## Method

The planner starts from a fully-masked action sequence and iteratively unmasks tokens over `T` denoising steps; ReMDM extends MDLM with remasking strategies that let committed tokens be re-predicted, improving plan coherence. Two independent training pipelines are compared head-to-head in the accompanying paper (under submission; citation to follow), both supervised by a pre-trained PPO expert:

One PPO expert checkpoint feeds both pipelines: `--mode offline` behaviour-clones from live expert rollouts, `--mode online` runs DAgger from scratch against expert labels. Either output is scored with `--mode inference`.

## Setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/). Linux GPU use needs NVIDIA driver >= 580 for CUDA 13, or >= 525 with `--extra cuda12`. CUDA and cuDNN come from the pip wheels, so no OS-level toolkit is required; if `module load cuda/13.x` is in your shell profile, unset `LD_LIBRARY_PATH`, which otherwise shadows the wheel libraries.

```bash
git clone --recurse-submodules https://github.com/mathisweil/craftax-ReMDM-planner.git
cd craftax-ReMDM-planner
# Or, if already cloned without submodules:
git submodule update --init --recursive

# Default: CPU-only JAX (macOS, or Linux without a GPU).
# Installs the dev group (pytest) too.
uv sync

# Linux GPU, CUDA 13 (driver >= 580)
uv sync --extra cuda13

# Linux GPU, CUDA 12 fallback (driver >= 525, or Maxwell/Pascal cards)
uv sync --extra cuda12
```

Extras: `cuda13` and `cuda12` are mutually exclusive and Linux-only. JAX ships GPU support only through these extras, so a GPU node needs one explicitly.

## Repo layout

```
craftax-ReMDM-planner/
├── Craftax_Baselines/       Git submodule — PPO expert training and env wrappers
├── configs/                 Experiment configs (defaults.yaml + presets, see Configuration)
├── src/                     Model, diffusion, planner pipelines
├── experiments/
│   └── rl_finetuning/       RL fine-tuning ablation suite (run_ablations.py)
├── scripts/                 Param counter, PPO evaluator, paper figures, HF upload utilities
├── tests/                   Smoke suite — uv run pytest
├── checkpoints/             Gitignored — offline/, online/, ppo_agents/ (see Checkpoints)
├── results/inference/       Eval JSONs from --mode inference (published, see Checkpoints)
├── demo_craftax.ipynb       Demo notebook
├── main.py                  CLI entry point
└── pyproject.toml           uv project — deps, cuda12/cuda13 extras, dev group
```

## Quickstart

Full DAgger pipeline (rollout, expert labelling, gradient updates, validation) under `configs/smoke.yaml`, ~25 s on CPU. The expert is randomly initialised unless `--ppo-checkpoint` is given, so this runs on a clean clone with no downloads. Watch `mean step reward`, `loss` and `all metrics finite`; returns stay at `0.000`, since no episode terminates in so short a run.

```bash
python main.py --mode smoke
```

## Training

Two independent training methods; neither depends on the other. An offline BC checkpoint can warm-start DAgger via `--checkpoint`, but this was not used for the paper results. All training modes need a PPO expert checkpoint.

### Stage 1 — Train the PPO expert (submodule)

```bash
cd Craftax_Baselines
python ppo_rnn.py --env_name Craftax-Classic-Symbolic-v1 \
    --total_timesteps 1000000000 --save_policy --use_wandb
cd ..
```

(`ppo_rnd.py` for Random Network Distillation.) Released experts are on the HF Hub, see [Checkpoints](#checkpoints).

### Offline BC

Rolls out the PPO agent live at each update.

```bash
python main.py --mode offline --ppo-checkpoint /path/to/ppo_checkpoint
python main.py --mode offline --ppo-checkpoint /path/to/ppo_checkpoint \
    --override offline_total_timesteps=100000000
```

### Online DAgger

Trained from scratch. Per iteration a mixed expert/learner policy rolls out, the expert labels every visited state, and the model trains on the aggregated buffer.

```bash
python main.py --mode online --ppo-checkpoint /path/to/ppo_checkpoint
python main.py --mode online --ppo-checkpoint /path/to/ppo_checkpoint \
    --override online_total_timesteps=100000000

# Optional: warm-start from a pre-trained offline checkpoint
python main.py --mode online --ppo-checkpoint /path/to/ppo_checkpoint \
    --checkpoint /path/to/offline_checkpoint
```

With `save_policy: true` (default) and W&B on, training uploads two artifacts, either consumable via `--checkpoint wandb:…`: `{env_name}-policy` (final) and `{env_name}-policy-best` (highest validation return).

### Collect trajectories to disk

Rolls out the PPO checkpoint and saves `(obs, actions, rewards, dones)` as `.npz`, for inspection; `--mode offline` does not consume it (it rolls out live).

```bash
python main.py --mode collect --ppo-checkpoint /path/to/ppo_checkpoint \
    --data data/trajectories.npz \
    --override collect_num_steps=1000000 --override collect_num_envs=128
```

### Resuming a training run

```bash
# Offline. --resume also accepts a wandb: artifact reference.
python main.py --mode offline --ppo-checkpoint /path/to/ppo_checkpoint \
    --resume /path/to/completed_offline_checkpoint \
    --override offline_total_timesteps=200000000

# Online. --resume-step / --resume-wandb-run-id override the metadata sidecar.
python main.py --mode online --ppo-checkpoint /path/to/ppo_checkpoint \
    --resume /path/to/completed_online_checkpoint \
    --override online_total_timesteps=200000000
```

The DAgger replay buffer is not persisted; it refills within a few iterations. The cosine LR schedule spans the full `num_updates`, with the step counter offset so the LR resumes exactly where it stopped. With a metadata sidecar, `resume_step` and `resume_wandb_run_id` are auto-detected; without one, pass `--resume-step`.

`--resume` restores the optimiser state, so it needs a checkpoint written by the current AdamW chain; an older one fails loudly on the optimiser-state structure, and there is no compatibility path. Use `--checkpoint` instead — parameters only, warm-starting a fresh run.

## Evaluation from a checkpoint

```bash
python main.py --mode inference --checkpoint /path/to/checkpoint --output results/inference/eval.json

# Released checkpoints need their matching config (see Checkpoints)
python main.py --mode inference \
    --config configs/final_craftax_classic_gpu_24gb.yaml \
    --checkpoint checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M
```

Prints steps per second, per-achievement unlock counts, and two returns that must not be quoted against one another; `--output` also writes both as JSON:

| Reported | JSON key | Meaning |
|---|---|---|
| Mean return, completed episodes | `mean_return_completed_episodes` (with `n_completed_episodes`) | Mean over every episode that terminated inside the rollout — the `returned_episode_returns` statistic the ablation tables and the paper report |
| Mean return, first life only | `mean_return_first_life`, and `mean_score` for backwards compatibility | Strict single-life return: the first episode of each env only. A harsher statistic |

By default this replans from scratch every `eval_replan` (8) steps, conditioned only on the current observation — the same sampler and cadence as `build_eval_fn` in the ablation harness, so it is the protocol behind the published numbers. Evaluation length is set by the `eval_steps` / `eval_num_envs` config keys.

`--override inference_sampler=inpainting` switches to the historical-inpainting sampler, which replans every step with each executed action locked as a fixed prefix (Diffuser Sec. 3.3). At step *k* of a `plan_horizon` window only `plan_horizon - k` positions are still free, so execution tends towards open-loop with a periodic reset. It is kept as an ablation on the planning-as-inpainting design choice and **scores far lower on the same weights**; no published number comes from it.

Write eval JSONs into `results/inference/` (created for you): `scripts/hf_upload.py` publishes every JSON it finds there.

Any checkpoint flag (`--checkpoint`, `--ppo-checkpoint`, `--resume`) accepts a W&B artifact reference prefixed `wandb:`; the artifact downloads automatically (location: `wandb_download_dir`, default `./artifacts/`).

```bash
python main.py --mode inference \
    --checkpoint wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy:latest
```

## Baselines and ablations

### RL baselines

PPO baselines (the expert family: `ppo`, `ppo_rnn`, `ppo_rnd`) train in the `Craftax_Baselines` submodule, see [Training](#stage-1--train-the-ppo-expert-submodule). Evaluate an expert with `scripts/eval_ppo_expert.py`:

```bash
uv run python scripts/eval_ppo_expert.py \
    --path checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M \
    --env-name Craftax-Classic-Symbolic-v1
```

### Method ablations (named configs)

Each paper experiment is a named config; pass it via `--config`:

```bash
python main.py --mode online --ppo-checkpoint <ppo> --config configs/classic_exp_a_beta_fix.yaml
python main.py --mode online --ppo-checkpoint <ppo> --config configs/classic_exp_b_beta_big_model.yaml
python main.py --mode online --ppo-checkpoint <ppo> --config configs/classic_exp_c_full_recipe.yaml
python main.py --mode online --ppo-checkpoint <ppo> --config configs/classic_exp_d_850K_model.yaml
```


### RL fine-tuning ablation suite

25 registered ablations (same names as in the minihack repo). See `experiments/README.md`.

```bash
python experiments/rl_finetuning/run_ablations.py --list
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint $PRETRAINED_CKPT --all
python experiments/rl_finetuning/run_ablations.py \
    --checkpoint wandb:my-team/remdm-craftax/Craftax-Classic-Symbolic-v1-policy-best:latest \
    --ablations baseline_rl kl_penalty --fast
```

The same entry point measures the return term of the gradient decomposition at the
pretrained checkpoint, with no training and no accelerator:

```bash
python experiments/rl_finetuning/run_ablations.py --measure-gdelta --gdelta-seeds 0 1 2 \
    --checkpoint $PRETRAINED_CKPT --results-path $RUN/results.json --output-dir $RUN
```

## Configuration

One YAML config holds the experiment; the CLI holds the run.

- **Config files** (`configs/*.yaml`): hyperparameters, model and method settings, ablation definitions. Any file passed via `--config` is merged onto `configs/defaults.yaml`. Exactly two layers: a preset never inherits from another preset.
- **CLI flags**: per-invocation values — `--seed`, `--checkpoint`, `--ppo-checkpoint`, `--data`, `--output`, `--resume*`, `--jit/--no-jit` (disable JIT for debugging).
- **`--override KEY=VALUE`** (repeatable): ad hoc config overrides. Keys are validated against `defaults.yaml` and values are cast to the key's type; a typo is an error, not a silent no-op.

Precedence, lowest to highest: `configs/defaults.yaml` < `--config` file < `--override` and run flags.

**`defaults.yaml` is the final Craftax Classic recipe, not a neutral baseline.** Run `main.py` with no `--config` and you get the paper's Classic DAgger run: a 384-dim, 6-layer model over 100M env frames.

**Presets hold only deltas, never restate a value they would inherit.** A key belongs in a preset only if its value differs from `defaults.yaml`. Restating one is not harmless duplication: it silently pins the preset when the recipe later moves. `tests/test_config.py` enforces this.

> **Schedule keys are denominated in env frames, not update steps.** Six settings — `lr_warmup_frames`, `offline_total_timesteps`, `online_total_timesteps`, `dagger_beta_final`, `dagger_buffer_cycles`, `val_interval_frames` — declare the *hardware-invariant* quantity; `resolve_num_updates()` and `resolve_scaled_hyperparams()` derive the update-step forms the runners consume (`num_updates`, `LR_WARMUP_STEPS`, `DAGGER_BETA_DECAY`, `DAGGER_BUFFER_MAX`, `VAL_INTERVAL`) from them at load. Set the frame-denominated key; the derived ones are outputs, not inputs.

```bash
python main.py --mode offline --ppo-checkpoint <ppo> \
    --override lr=1e-4 --override plan_horizon=64 --override num_minibatches=16
python main.py --mode offline --ppo-checkpoint <ppo> --no-jit --override num_envs=4
```

| Preset | Purpose |
|---|---|
| `configs/defaults.yaml` | The final Craftax Classic recipe, and what every other preset layers onto |
| `configs/smoke.yaml` | `--mode smoke` overrides (see the sizing invariants commented in the file) |
| `configs/{classic,craftax}_exp_a_beta_fix.yaml` | DAgger — beta decay fix only (isolates data quality) |
| `configs/{classic,craftax}_exp_b_beta_big_model.yaml` | DAgger — beta fix + larger transformer |
| `configs/{classic,craftax}_exp_c_full_recipe.yaml` | DAgger — beta + big model + training dynamics |
| `configs/classic_exp_d_{100K,250K,850K,3M}_model.yaml` | Craftax Classic model-size scaling sweep |
| `configs/craftax_exp_d_{500K,1M,3M,7M}_model.yaml` | Full Craftax model-size scaling sweep |
| `configs/final_craftax_classic_{gpu_h200,gpu_24gb}.yaml` | Final Classic DAgger — `num_envs` and `seed` only; the recipe is `defaults.yaml` |
| `configs/final_craftax_{gpu_h200,gpu_24gb}.yaml` | Final Full Craftax DAgger — the 11 keys where Full Craftax departs from the Classic recipe, plus `num_envs` and `seed` |

Within each family the two cluster configs differ only in `num_envs` and `seed`. Nothing in the loader enforces that: the guard is `test_cluster_siblings_differ_only_in_num_envs_and_seed`. **A Full Craftax hyperparameter change must be made in both `final_craftax_*` files**, since with no inheritance those 11 keys are duplicated verbatim in each; a Classic one belongs in `defaults.yaml`.

Fairness-critical values are env-frame denominated (the six keys above) and rescaled by `resolve_scaled_hyperparams()` at load, so one recipe runs on any hardware tier. Key hyperparameters are documented inline in `configs/defaults.yaml`; the [appendix](#key-hyperparameters) tabulates them. Ablation-suite hyperparameters live in `experiments/rl_finetuning/configs/`, loaded by `run_ablations.py`, not `main.py`.

## Checkpoints

With `save_policy: true` (the default), training saves Orbax checkpoints to `policies` (final) and `policies_best` (highest validation return). With W&B on these sit under `wandb.run.dir` and are uploaded as W&B artifacts named `{env_name}-policy` and `{env_name}-policy-best`; with W&B off they go to `{checkpoint_dir}/{mode}/{run_name}/` instead, so a run never discards its weights. Diffusion checkpoints carry a `resume_metadata.json` sidecar — the authoritative record of the producing run's config, and what `--resume` reads to auto-detect `resume_step` and `resume_wandb_run_id`. PPO checkpoints carry `config.yaml` and `wandb-summary.json`.

**Pass the checkpoint directory, not the step subdirectory** — `CheckpointManager` resolves the latest step itself.

Offline checkpoints save at the resolved env-frame budget, which is 99,942,400 for the Classic recipe at 512 envs (1525 updates × 512 × 128).

`checkpoints/` is gitignored; released weights live on the Hub at [`mathisweil/remdm-craftax-checkpoints`](https://huggingface.co/mathisweil/remdm-craftax-checkpoints), mirroring the layout below.

| Checkpoint directory | Environment | Role | Trained for |
|---|---|---|---|
| `checkpoints/offline/Craftax-Classic-Symbolic-v1-Offline-Diffusion-BC-100M` | Craftax Classic | Offline BC planner | 1e8 env frames |
| `checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M` | Craftax Classic | Online DAgger planner | 1e8 env frames |
| `checkpoints/ppo_agents/Craftax-Classic-Symbolic-v1-PPO_RNN-1000M` | Craftax Classic | PPO-RNN expert | 1e9 env frames |
| `checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M` | Full Craftax | PPO-RNN expert | 1e9 env frames |

Full-Craftax diffusion planner checkpoints are not released: no full-Craftax training run has completed.

```bash
# All four (~470 MB); narrow the --include glob for a single checkpoint.
uv run hf download mathisweil/remdm-craftax-checkpoints --include "checkpoints/**" --local-dir .
```

### Experiment outputs

Ablation figures, tables and `results.json` are **not in the repository** and never
should be: they are regenerated output, and 244 MB of them was rewritten out of the
history. `experiments/rl_finetuning/outputs/` and `results/inference/` are gitignored.

Obtain them either way:

```bash
# Fetch the published run (figures, tables, results.json, diagnosis.md)
uv run hf download mathisweil/remdm-craftax-checkpoints \
    --include "experiments/rl_finetuning/outputs/**" --local-dir .

# Or regenerate from a checkpoint; writes to outputs/{run_id}/
python experiments/rl_finetuning/run_ablations.py --checkpoint $PRETRAINED_CKPT --all
```

`scripts/hf_upload_demo.py` reads `outputs/craftax_classic_final_results/{figures,tables}`
from the working copy, so fetch or regenerate before running it. `demo_craftax.ipynb`
needs no local copy — it reads them from its own `snapshot_download`.

### Paper figures

The manuscript's figures put Craftax Classic and MiniHack in one figure, so they are
built by `scripts/paper_figures.py` rather than by the single-environment
`analysis/plots.py`. It reads both repositories' `results.json` and emits vector PDF
at NeurIPS column width:

```bash
uv run python scripts/paper_figures.py \
    --minihack-results ../minihack-ReMDM-planner/results/experiments/rl_finetuning/outputs/minihack_ablations/results.json \
    --outdir results/paper_figures
```

The MiniHack path defaults to that sibling checkout. Pass `--emit-tex-macros` to
`run_ablations.py` to also write `tables/results.tex`, one `\newcommand` per headline
quantity, so the manuscript cites generated numbers instead of retyping them.

**Match the config to the checkpoint.** The model is built from the config, not the checkpoint, and a mismatch raises at restore. All released diffusion checkpoints are `d_model` 384, `n_heads` 8, `n_layers` 6, `d_ff` 768 — the architecture `defaults.yaml` also carries — so use the matching `final_*` config, which additionally sets the right `env_name` and recipe values:

```bash
python main.py --mode inference \
    --config configs/final_craftax_classic_gpu_24gb.yaml \
    --checkpoint checkpoints/online/Craftax-Classic-Symbolic-v1-Online-Diffusion-DAgger-100M

# Train a new planner against the released Full Craftax PPO expert
python main.py --mode online \
    --config configs/final_craftax_gpu_24gb.yaml \
    --ppo-checkpoint checkpoints/ppo_agents/Craftax-Symbolic-v1-PPO_RNN-1000M
```

### Publishing to the Hub

`scripts/hf_upload.py` rediscovers and uploads three things, each keeping its repo-relative path: `checkpoints/`, every `experiments/rl_finetuning/outputs/<run>/` holding a `results.json` (with `diagnosis.md`, `tables/`, `figures/`), and the eval JSONs in `results/inference/`. It drops W&B and hub config keys, shortens absolute paths and regenerates the model card.

```bash
HF_TOKEN=hf_xxx uv run python scripts/hf_upload.py --repo-id mathisweil/remdm-craftax-checkpoints --dry-run
```

`--dry-run` prints the staged tree and card without uploading; drop it to upload. Also `--inference-results <FILE|DIR> ...` (eval JSONs kept elsewhere), `--private`, `--yes`.

**Checkpoint discovery expects the released layout**, `checkpoints/<role>/<name>/<step>/` — the layout the Hub repo mirrors. A training run writes elsewhere, so copy its `wandb.run.dir/policies` directory to `checkpoints/{offline,online}/<name>` first, or nothing is staged. `checkpoints/hf/` is skipped: that is where a Hub *download* lands, and publishing from it would push already-published artefacts back up into a nested `checkpoints/hf/checkpoints/...` tree.

## Results, citation, licence

Results tables and the full method description are in the accompanying paper (under submission); `demo_craftax.ipynb` reproduces the headline evaluation. Citation to be added on publication. Licence: MIT, see `LICENSE`.

---

# Appendix: benchmark-specific detail

## Environments

| Environment | Achievements | Actions | Notes |
|---|---|---|---|
| `Craftax-Classic-Symbolic-v1` | 22 | 17 | Crafter ported to JAX |
| `Craftax-Symbolic-v1` | 65 | 43 | + NetHack mechanics, 9 floors |

Set via the `env_name` config key.

## Remasking strategies

Controlled by the `remask_strategy` key. All strategies operate on top of the three-phase loop controlled by `use_loop`, `t_on`, and `t_off`.

| Strategy | Formula | Description |
|---|---|---|
| `rescale` | `sigma = eta * sigma_max` | Scales maximum remasking probability proportionally |
| `cap` | `sigma = min(eta, sigma_max)` | Caps remasking at a fixed rate |
| `conf` | `sigma = softmax(-psi) * eta * sigma_max` over committed tokens | Low-confidence tokens are remasked preferentially (psi = decode probability at last unmask) |

## Key hyperparameters

**Environment**

| Parameter | Default | Description |
|---|---|---|
| `env_name` | `Craftax-Classic-Symbolic-v1` | Craftax environment ID. Use `Craftax-Symbolic-v1` for Full Craftax. |
| `use_optimistic_resets` | `false` | Use `OptimisticResetVecEnvWrapper` instead of `AutoResetEnvWrapper` |
| `optimistic_reset_ratio` | 16 | Fraction of envs reset per step when optimistic resets are enabled |

**Diffusion model**

| Parameter | Default | Description |
|---|---|---|
| `plan_horizon` | 32 | Action plan length H |
| `diffusion_steps` | 15 | Denoising steps T during training |
| `diffusion_steps_eval` | 10 | Denoising steps T at inference |
| `diffusion_schedule` | `cosine` | Noise schedule: `cosine` or `linear` |
| `remask_strategy` | `rescale` | Remasking strategy: `rescale`, `cap`, or `conf` |
| `train_sigma` | 0.0 | Per-token remasking correction during training (0 = standard MDLM) |
| `label_smoothing` | 0.0 | Cross-entropy label smoothing epsilon (0 = exact ELBO) |
| `eta` | 0.5 | Remasking strength |
| `use_loop` | `true` | Three-phase loop remasking (Algorithm 3) |
| `t_on` / `t_off` | 0.7 / 0.3 | Time window boundaries for loop remasking |
| `temperature` | 0.5 | Softmax temperature for token sampling |
| `top_p` | 0.95 | Nucleus sampling threshold |

**Transformer architecture**

| Parameter | Default | Description |
|---|---|---|
| `d_model` | 384 | Hidden dimension |
| `n_heads` | 8 | Attention heads |
| `n_layers` | 6 | Transformer blocks |
| `d_ff` | 768 | FFN inner dimension |
| `obs_encoder_layers` | 2 | MLP layers in the observation encoder |
| `obs_encoder_width` | 768 | Observation encoder hidden width |
| `dropout_rate` | 0.1 | Dropout rate (disabled at inference) |

**Offline training**

| Parameter | Default | Description |
|---|---|---|
| `offline_total_timesteps` | 1e8 | Env-frame budget. Derives `num_updates` as `offline_total_timesteps // (num_envs * num_steps)`. |
| `num_envs` | 1024 | Parallel environments |
| `num_steps` | 128 | Environment steps collected per update |
| `num_minibatches` | 8 | Gradient minibatches per epoch |
| `update_epochs` | 8 | SGD epochs per update step |
| `num_repeats` | 1 | Independent training seeds (vmapped) |
| `lr` | 3e-4 | AdamW learning rate (cosine-decayed to 10% over all gradient steps) |
| `weight_decay` | 0.0 | Decoupled AdamW decay for core training; 0.0 is Adam exactly (the ablation suite keeps 1e-4) |
| `lr_warmup_frames` | 1.6384e6 | Env-frame linear warm-up budget (0 = disabled). Derives `LR_WARMUP_STEPS` in gradient steps. |
| `max_grad_norm` | 1.0 | Global gradient clipping norm |
| `return_weight_cap` | 5.0 | Clip ceiling for per-window return weights (lower clip fixed at 0.1) |
| `collect_temperature` | 1.0 | Softmax temperature on PPO logits during live data collection |
| `val_interval_frames` | 1e6 | Env-frames between validation rollouts. Derives `VAL_INTERVAL` in update steps. |
| `val_diffusion_steps` | 50 | Denoising steps during validation rollouts |
| `val_replan_every` | 4 | Env steps executed per diffusion plan during validation |
| `val_steps` | 256 | Total env steps per validation rollout |

**Online DAgger training**

| Parameter | Default | Description |
|---|---|---|
| `online_total_timesteps` | 1e8 | Env-frame budget. Derives `num_updates`. |
| `dagger_beta_init` | 1.0 | Initial expert mixing probability `beta_1` |
| `dagger_beta_final` | 0.344 | Target final mixing ratio. Derives the per-update decay `beta_i = beta_init * decay^i`. |
| `dagger_buffer_cycles` | 1.90735 | Replay-buffer capacity in update cycles of history. Derives `DAGGER_BUFFER_MAX` in samples. |
| `dagger_train_passes` | `null` | Passes per update over the buffer; `null` = 1 (matches offline BC per-update gradient work) |
| `dagger_expert_deterministic` | `true` | Argmax expert (fixed `s -> a*` map) vs categorical sampling |

**Data collection / inference**

| Parameter | Default | Description |
|---|---|---|
| `collect_num_steps` | 10000000 | Total environment steps to collect |
| `collect_num_envs` | 128 | Parallel environments during collection |
| `ppo_model_type` | `ppo_rnn` | PPO architecture: `ppo`, `ppo_rnn`, or `ppo_rnd` |
| `layer_size` | 512 | PPO actor-critic hidden layer width |
| `eval_steps` | 10000 | Environment steps for evaluation |
| `eval_num_envs` | 32 | Parallel agents during evaluation (independent of `num_envs`) |

**Checkpointing / resume / logging**

| Parameter | Default | Description |
|---|---|---|
| `save_policy` | `true` | Save final checkpoint and upload as W&B artifact |
| `resume_checkpoint_path` | `null` | Per-run: `--resume` (accepts `wandb:` refs) |
| `resume_wandb_run_id` | `null` | Per-run: `--resume-wandb-run-id` (auto-read from metadata) |
| `resume_step` | `null` | Per-run: `--resume-step` (auto-read from metadata) |
| `seed` | `null` | RNG seed (random if null; per-run: `--seed`) |
| `use_wandb` | `true` | Enable Weights & Biases logging |
| `wandb_project` | `craftax-ReMDM-planner` | W&B project name |
| `wandb_entity` | (author's) | W&B entity |
| `wandb_download_dir` | `null` | Download dir for W&B artifacts; null = `./artifacts/` |
| `jax_compilation_cache_dir` | `null` | Persistent XLA compilation cache; null = off. See below |

### Persistent compilation cache

The whole training run is one `jax.jit`, so every process pays one large
compilation before any work happens, and multi-seed runs, resumed runs and the
ablation suite each repeat it. `jax_compilation_cache_dir` makes the second and
later runs of the same graph skip it. The cache is keyed on the lowered HLO, so
a hit is bit-identical to a miss. **Point it at local disk, not an NFS home:**

```bash
python main.py --mode online --ppo-checkpoint <ppo> \
    --config configs/final_craftax_classic_gpu_24gb.yaml \
    --override jax_compilation_cache_dir=/var/tmp/$USER/jax-cache
```

## Environment wrappers

From `Craftax_Baselines/wrappers.py` (submodule):

| Wrapper | Purpose |
|---|---|
| `LogWrapper` | Tracks episode returns and lengths; adds stats to the info dict |
| `AutoResetEnvWrapper` | Automatically resets episodes on `done` |
| `BatchEnvWrapper` | Vmaps `reset` and `step` over `num_envs` environments |
| `OptimisticResetVecEnvWrapper` | Batched resets with reduced overhead; enable via `use_optimistic_resets` |

Stack (identical for training and inference): `env -> LogWrapper -> AutoResetEnvWrapper -> BatchEnvWrapper`.

## Testing

```bash
uv run pytest
```

A CPU-only suite, 13 modules. Tiny synthetic data and a shrunken model throughout — no real checkpoints, datasets or network calls, and nothing written outside `tmp_path`. `conftest.py` forces `JAX_PLATFORMS=cpu` and disables W&B; there are no custom markers.

| File | Covers |
|---|---|
| `test_smoke_src.py`, `test_smoke_experiments.py` | that things **run**: imports, model from the real config, a gradient step, checkpoint round-trip, samplers, resolvers, every CLI entry point, and all 25 ablations' losses and optimizers |
| `test_spec_*.py`, `test_method_spec*.py` | that things are **correct**: each canonical statement of `research/spec-*.md` pinned against the implementation |
| `test_config.py`, `test_recipe_values.py` | the preset, delta-only, cluster-sibling and poolability rules, and the shipped recipe values |
| `test_gpu_agreement.py` | CPU/GPU agreement, skipped without a device |

## Implementation notes

| Topic | Note                                                                                                                                                                                                                                                       |
|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| JAX purity | `make_train_offline_diffusion` / `make_train_online_dagger` are fully JIT-compatible; env construction and checkpoint I/O sit outside `jax.jit`.                                                                                                           |
| Offline data | `--mode offline` rolls out PPO live. `--mode collect` saves an `.npz` for inspection only — re-feeding it to `--mode offline` is unsupported; pass `--ppo-checkpoint`.                                                                                     |
| Episode-boundary masking | A window at `(e, t)` is valid only if `dones[e, t+1:t+H-1]` are all `False`.                                                                                                                                                                               |
| Return weighting | Valid windows are weighted by cumulative reward, normalised by the batch mean, clipped to `[0.1, return_weight_cap]`, and applied as per-sample multipliers before loss reduction.                                                                         |
| LR schedule | Cosine decay `lr -> lr * 0.1` over all gradient steps. `lr_warmup_frames` prepends linear warm-up, converted to gradient steps as `(frames // fpu) * update_epochs * num_minibatches (* dagger_train_passes online)`.                                       |
| Env-frame invariance | The six frame-denominated keys are converted to update-step form by `resolve_scaled_hyperparams()` using `fpu = num_envs * num_steps`, so one config runs on any hardware tier.                                                                            |
| DAgger sizing | `dagger_sizing()` in `src/planners/common.py` is the single source of truth for `samples_per_update`, buffer capacity and `n_train_passes`.                                                                                                                |
| Loss weight clipping | The MDLM SUBS weight `-alpha'(t) / (1 - alpha_t)` is clipped to 1000 for stability as `alpha_t -> 1`.                                                                                                                                                      |
| Validation rollouts | Every `val_interval` updates, using inference sampling parameters with `val_diffusion_steps`, `val_replan_every` and `val_steps`.                                                                                                                          |
| W&B namespaces | Centralised in `src/planners/logging.py`: `diffusion/`, `train/`, `env/`, `val/`, `dagger/`. `train/sps` only in modes with live env interaction.                                                                                                          |
| DAgger aggregation | Ross et al. (2011). A circular buffer accumulates `(obs, expert_plan)` across iterations; windows use a sliding stride so every visited state contributes a label. The expert receives correct `done` flags so its RNN state resets at episode boundaries. |
| Best-checkpoint tracking | Highest-validation-return parameters are kept alongside the live ones and uploaded as `{env_name}-policy-best`.                                                                                                                                            |
| Denoising indexing | Reverse scan runs `step_idx = 0 -> T-1`, mapping to `t = (T - step_idx) / T` (high to low noise).                                                                                                                                                          |
| PPO experts | Training lives entirely in `Craftax_Baselines/`; planner modes only consume checkpoints. Released PPO checkpoints were saved on GPU and fail to restore on a CPU-only machine.                                                                   |
