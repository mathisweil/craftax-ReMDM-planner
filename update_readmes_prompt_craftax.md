# Task: Audit Codebase and Update READMEs

You are updating the documentation for a research project that implements **ReMDM (Remasking Discrete Diffusion Model)** for action-sequence planning in Craftax environments, with an RL fine-tuning ablation suite. The project has two READMEs that have drifted from the actual codebase and need to be brought up to date.

## Files to update

1. **`README.md`** — Root project README (Craftax ReMDM planner)
2. **`experiments/README.md`** — Experiments README (RL fine-tuning ablation suite)

## How to proceed

### Step 1: Analyse the codebase

Before making any edits, read and understand the current state of the code. At minimum inspect:

**Root project:**
- `main.py` — entry point, available modes (`offline`, `online`, `collect`, `inference`), CLI arguments
- `src/config.py` — config loading, all registered hyperparameters, defaults
- `src/models/denoiser.py` — model architecture (`DenoisingTransformer`): obs encoder, transformer, parameter counts, input/output shapes
- `src/diffusion/` — `loss.py`, `sampling.py`, `forward.py`, `schedules.py`
- `src/planners/` — `offline.py`, `online.py`, `collect.py`, `inference.py`, `common.py`, `env.py`, `model.py`, `logging.py`, `ppo.py`
- `src/envs/wrappers.py` — `SequenceHistoryWrapper`, `DiscreteTokenizationWrapper`
- `configs/defaults.yaml`, `configs/big_diffusion_offline.yaml`, `configs/big_diffusion_online.yaml`, `configs/A100_diffusion_offline.yaml`, `configs/A100_diffusion_online.yaml`
- `pyproject.toml` — dependencies, project metadata
- `Craftax_Baselines/` — PPO submodule: `ppo_rnn.py`, `ppo_rnd.py`, `ppo.py`, `models/`

**Experiments:**
- `experiments/rl_finetuning/run_ablations.py` — CLI, registered ablations, entry point
- `experiments/rl_finetuning/ablations/registry.py` — all ablation specs
- `experiments/rl_finetuning/ablations/losses.py` — loss variants
- `experiments/rl_finetuning/ablations/optimizers.py` — optimizer configs (LLRD, LoRA, frozen, etc.)
- `experiments/rl_finetuning/ablations/training.py` — training loop, `AblationHistory` dataclass
- `experiments/rl_finetuning/diagnostics/` — all diagnostic modules (`gradient.py`, `representation.py`, `timestep.py`)
- `experiments/rl_finetuning/analysis/` — all analysis/plotting modules (`plots.py`, `tables.py`, `report.py`)
- `experiments/rl_finetuning/configs/` — `ablations_default.yaml`, `ablations_fast.yaml`

Also check for any **new files or directories** not mentioned above — `find . -name '*.py' -newer README.md` can help identify recently changed files.

### Step 2: Identify discrepancies

Compare the current README content against what you found in the code. Pay special attention to:

1. **Pipeline description** — The README describes a four-stage pipeline (Train PPO → Collect/Offline → Online DAgger → Evaluate). Verify this still reflects the actual workflow. In particular, check whether `--mode collect` is still a separate stage or has been folded into `--mode offline` (which rolls out the PPO agent live). Verify that the pipeline diagram accurately represents the data flow and which stages are optional vs. required.

2. **Architecture description** — The README documents an MLP observation encoder feeding into a bidirectional transformer with `d_model=256`, `n_heads=4`, `n_layers=4`, `d_ff=512`, and `plan_horizon=32`. Check `src/models/denoiser.py` and `configs/defaults.yaml` to verify these values. Confirm the observation encoder dimensions (`obs_encoder_layers=2`, `obs_encoder_width=512`), the total parameter count, and whether the architecture diagram (if any) matches the actual token sequence construction.

3. **Hyperparameter tables** — Cross-reference every parameter in the README tables against `src/config.py` and `configs/defaults.yaml`. Look for:
   - Parameters that exist in code but are missing from the README
   - Parameters in the README that no longer exist or have been renamed
   - Default values that have changed
   - New config presets (check `configs/` directory for any new YAML files)

4. **CLI interface** — Check `main.py` argument parsing for any new flags, changed flag names, or removed options. Verify the `--no-jit` flag, `--config` flag, checkpoint path arguments (`--checkpoint_path`, `--offline_checkpoint_path`, `--ppo_checkpoint_path`), and `wandb:` artifact prefixes all work as documented. Do the same for `experiments/rl_finetuning/run_ablations.py`.

5. **Ablation registry** — Check `experiments/rl_finetuning/ablations/registry.py` for the current list of registered ablations. The README lists 25 across groups A–D (Baseline, Regularisation, Training Signal, Architecture, Data Quality). Verify the count, names, group assignments, and descriptions all match.

6. **Checkpoint format** — Check the actual checkpoint save/load calls in `src/planners/online.py`, `src/planners/offline.py`, and `src/planners/model.py` to verify the checkpoint dict keys and metadata sidecar format match what's documented. Verify the resume logic (step counter, LR schedule, W&B run ID).

7. **W&B metric namespaces** — Check the actual logging calls in `src/planners/logging.py` and throughout the codebase to verify namespace prefixes (`diffusion/`, `train/`, `env/`, `val/`, `dagger/`, `ablations/`) and metric names.

8. **Environment configuration** — Verify the default environment ID (`Craftax-Classic-Symbolic-v1`), the observation space dimensions, action space size, and any wrapper configuration (`SequenceHistoryWrapper`, `DiscreteTokenizationWrapper`, `OptimisticResetVecEnvWrapper`) against `src/envs/wrappers.py` and `src/planners/env.py`.

9. **Reward and data pipeline** — Verify the return-weighting description: episode-boundary masking, cumulative reward normalisation, clipping to `[0.1, return_weight_cap]`, MDLM loss integration. Check these against `src/planners/offline.py`.

10. **Diffusion sampling parameters** — Verify remasking strategies (`rescale`, `cap`, `conf`), loop remasking (`use_loop`, `t_on`, `t_off`), nucleus sampling (`top_p`), temperature, and `eta` against `src/diffusion/sampling.py` and `configs/defaults.yaml`.

11. **Diagnostic metrics** — Check `experiments/rl_finetuning/diagnostics/` for the current set of metrics and their collection frequencies. Verify the diagnostic table in the experiments README matches.

12. **Output structure** — Verify the experiment output directory structure and file list (figures, tables, `results.json` schema, `diagnosis.md`) against what the analysis pipeline in `experiments/rl_finetuning/analysis/` actually produces.

13. **Installation instructions** — Check `pyproject.toml` for current dependencies and versions. Verify the `uv sync` workflow, the `--extra cuda` flag, and submodule initialisation steps. Check whether the C/C++ build toolchain note (for NLE/MiniHack) is still relevant or should be removed for a pure Craftax/JAX project.

14. **Project structure tree** — Run `find` or `tree` on the actual directory structure and compare against what's documented. Add any new files/dirs, remove any that no longer exist. Pay special attention to the `Craftax_Baselines/` submodule structure and any new config files.

15. **Validation and inference** — Verify `val_interval`, `val_diffusion_steps`, `val_replan_every`, `val_steps`, `eval_steps`, `eval_num_envs`, `hist_len` (historical inpainting) and MPC replan logic against the code.

### Step 3: Apply updates

Edit both READMEs to reflect the true state of the codebase. Follow these principles:

- **Accuracy over aesthetics** — every claim must match the code
- **Don't remove useful documentation** — if a mode or feature still exists in code, keep its docs even if it's no longer the primary workflow
- **Keep the same markdown style** — tables, code blocks, section hierarchy should stay consistent
- **Update all code examples** — if CLI flags changed, fix the example commands
- **Mark genuinely uncertain items** — if the code is ambiguous, add a `<!-- TODO: verify -->` comment rather than guessing

## Research context

This project supports a research paper investigating discrete diffusion models for planning. The key claims the README should align with:

- **ReMDM** generates action sequences via masked discrete diffusion with stochastic inference-time remasking, enabling iterative refinement of committed tokens
- A pre-trained **PPO agent** (PPO-RNN or PPO-RND, from the `Craftax_Baselines` submodule) provides the expert policy for data collection and DAgger supervision
- **Offline training** collects windows from live PPO rollouts with return-weighted MDLM ELBO loss
- **Online DAgger fine-tuning** uses the PPO expert to label visited states, accumulating experience in a circular replay buffer trained with pure BC (no reward weighting)
- **RL fine-tuning** via return-weighted ELBO causes catastrophic collapse — the ablation suite diagnoses why
- **Stratified Plasticity** (LLRD) and success-filtered trajectory anchoring are the most promising stabilisation interventions
- The architecture uses an **MLP observation encoder** (processing the flat symbolic observation vector) feeding into a **bidirectional transformer**
- Training uses **MDLM ELBO** loss with optional SUBS importance weighting and label smoothing
- The project targets **Craftax-Classic-Symbolic-v1** with the full Craftax environment as a stretch goal
- **Craftax** is a JAX-native procedurally-generated environment (Crafter rewrite) with 17 discrete actions, 22 achievements, and symbolic observations of size 1345

## Quality checklist

Before finishing, verify:

- [ ] Pipeline diagram reflects the actual stage dependencies and which stages are optional
- [ ] Architecture description matches actual `d_model`, `n_layers`, `n_heads`, `d_ff`, `plan_horizon`, obs encoder dims from code
- [ ] All hyperparameter defaults match `configs/defaults.yaml` and `src/config.py`
- [ ] All CLI examples actually work with current argument parsing
- [ ] Ablation count and names match `registry.py`
- [ ] Project structure tree matches actual filesystem (including `Craftax_Baselines/` submodule)
- [ ] Checkpoint format and resume logic match actual save/load code
- [ ] W&B metric namespaces match actual `wandb.log()` calls
- [ ] Dependency versions in the table match `pyproject.toml`
- [ ] No dead links or references to nonexistent files
- [ ] Installation prerequisites are correct for a JAX/Craftax project (not MiniHack/NLE)
- [ ] Both READMEs are updated, not just one
