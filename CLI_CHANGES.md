# CLI migration (branch chore/align-cli)

Interface alignment with `minihack-ReMDM-planner`. Old names are removed outright; there are no aliases.

## main.py: renamed flags

| Old | New | Layer change |
|---|---|---|
| `--checkpoint_path` | `--checkpoint` (inference weights) | none (CLI) |
| `--offline_checkpoint_path` | `--checkpoint` (online/smoke warm start; same flag, resolved by mode) | none (CLI) |
| `--ppo_checkpoint_path` | `--ppo-checkpoint` | none (CLI) |
| `--offline_data_path` | `--data` | none (CLI) |
| `--inference_output` | `--output` | none (CLI) |
| `--resume_checkpoint_path` | `--resume` | none (CLI) |
| `--resume_step` | `--resume-step` | none (CLI) |
| `--resume_wandb_run_id` | `--resume-wandb-run-id` | none (CLI) |

Unchanged: `--config`, `--mode`, `--seed`, `--jit/--no-jit`.

## main.py: per-key hyperparameter flags removed

All of the following moved from dedicated CLI flags to `--override KEY=VALUE` (repeatable; key validated against `configs/defaults.yaml`, value cast to the key's type). The config file remains the single home of their defaults.

`--env_name --use_optimistic_resets --optimistic_reset_ratio --d_model --n_heads --n_layers --d_ff --obs_encoder_layers --obs_encoder_width --dropout_rate --plan_horizon --diffusion_schedule --diffusion_steps --diffusion_steps_eval --train_sigma --label_smoothing --remask_strategy --eta --use_loop --t_on --t_off --temperature --top_p --lr --max_grad_norm --offline_total_timesteps --offline_num_updates --num_envs --num_steps --num_minibatches --update_epochs --num_repeats --collect_temperature --val_interval --val_diffusion_steps --val_replan_every --val_steps --return_weight_cap --lr_warmup_steps --lr_warmup_frames --val_interval_frames --online_num_updates --online_total_timesteps --dagger_beta_init --dagger_beta_decay --dagger_beta_final --dagger_buffer_max --dagger_buffer_cycles --collect_num_steps --collect_num_envs --ppo_model_type --layer_size --eval_steps --eval_num_envs --save_policy --checkpoint_interval --max_checkpoints --use_wandb --wandb_project --wandb_entity --wandb_download_dir`

Example: `--num_envs 512 --lr 1e-4` becomes `--override num_envs=512 --override lr=1e-4`.

Removed with **no** replacement (dead flags, no reader anywhere in `src/`, `experiments/` or `tests/`): `--checkpoint_dir`, `--checkpoint_interval`, `--max_checkpoints`, `--batch_size` (the ablation suite's `--batch-size` is separate and remains).

## main.py: config semantics

- `--config FILE` is now deep-merged onto `configs/defaults.yaml` instead of replacing it. All shipped configs either define a key or define its PRIMARY frame-denominated source (`val_interval_frames` etc.), so resolved values are unchanged; presets may now be written as deltas.
- Unknown keys in a config file or `--override` are rejected instead of silently ignored. The path keys (`ppo_checkpoint_path`, `checkpoint_path`, `offline_checkpoint_path`, `offline_data_path`, `inference_output`) remain valid in config files (smoke.yaml sets `ppo_checkpoint_path: null`).
- `remask_strategy`, `diffusion_schedule` and `ppo_model_type` values are now validated in every mode (previously only when set via their CLI flags).

## experiments/rl_finetuning/run_ablations.py

| Old | New |
|---|---|
| `--checkpoint_path` | `--checkpoint` |
| `--ppo_checkpoint_path` | `--ppo-checkpoint` |
| `--ablations_config` | `--ablations-config` |
| `--analyze_only` | `--analyze-only` |
| `--results_path` | `--results-path` |
| `--output_dir` | `--output-dir` |
| `--run_id` | `--run-id` |
| `--num_seeds` | `--num-seeds` |
| `--use_wandb` | `--use-wandb` |
| `--wandb_project` | `--wandb-project` |
| `--wandb_entity` | `--wandb-entity` |
| `--max_iter` | `--max-iter` |
| `--num_envs` | `--num-envs` |
| `--batch_size` | `--batch-size` |
| `--eval_every` | `--eval-every` |

Config key (`experiments/rl_finetuning/configs/*.yaml`): `collect_temp` -> `collect_temperature` (matches `configs/defaults.yaml`).

## scripts/

| Old | New |
|---|---|
| `eval_ppo_expert.py --env_name` | `--env-name` |
| `eval_ppo_expert.py --num_envs` | `--num-envs` |
| `eval_ppo_expert.py --model_type` | `--model-type` |
| `eval_ppo_expert.py --layer_size` | `--layer-size` |
| `count_params.py --verify_checkpoint` | `--verify-checkpoint` |

`hf_upload.py`, `hf_upload_demo.py` were already kebab-case; unchanged. The `Craftax_Baselines` submodule keeps its own upstream CLI (`--env_name` etc.) untouched.

## pyproject.toml

Unchanged (the `cuda` extra already existed; the minihack repo gained a matching one).

## Defaults deliberately different from minihack-ReMDM-planner

Benchmark-tuned values, unchanged by this alignment:

| Key | craftax | minihack |
|---|---|---|
| `eta` | 0.5 | 0.15 |
| `remask_strategy` | `rescale` | `conf` |
| noise schedule | `diffusion_schedule: cosine` | `noise_schedule: linear` |
| dropout | `dropout_rate: 0.1` | `dropout: 0.0` |
| plan length | `plan_horizon: 32` | `seq_len: 64` |
| training budget | `offline/online_total_timesteps` (per mode, env frames) | `total_timesteps: 2e6` (unified, env steps) |
| LR | single `lr` | per-mode `offline_lr`/`dagger_lr` |
| top-p vs top-K | `top_p: 0.95` | `top_k: 4` |

Model-architecture key names (`d_model`/`n_heads`/`n_layers` vs `n_embd`/`n_head`/`n_layer`, etc.) are deliberately **not** renamed: every released HF checkpoint's `resume_metadata.json` snapshot and the minihack config snapshots use the existing names.

## Noticed but not touched

- `pyproject.toml` `description` is still the uv placeholder.
- Checkpoints save only when `use_wandb` AND `save_policy` are on; W&B-less runs cannot save (behavioural, out of scope).
- In-code `config.get(key, default)` fallbacks duplicating `defaults.yaml` values remain: the ablation suite builds partial configs that rely on them, so removing them would change behaviour there. The one internally disagreeing site (`VAL_INTERVAL` fallback 0 in `src/planners/common.py` vs 50 elsewhere) is in a display-only banner and is unreachable now that defaults always merge.
- `main.py` mutates `sys.path` inline with a semicolon statement.
- Released PPO checkpoints fail to restore on CPU-only machines (noted in `configs/smoke.yaml`).
