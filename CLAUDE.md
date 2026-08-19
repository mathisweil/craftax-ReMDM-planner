# CLAUDE.md — craftax-ReMDM-planner

JAX/Flax ReMDM planner on Craftax; supervision by a pre-trained PPO expert (`Craftax_Baselines` submodule). Sibling repo `minihack-ReMDM-planner` (PyTorch) implements the same method with deliberately shared scaffolding: keep structure, configs, CLIs, docs and tests aligned unless the divergence is environment/framework-forced. When present, the parent workspace `CLAUDE.md` (one level up) governs.

## Configuration

- Exactly two config layers: a `--config` preset merges onto `configs/defaults.yaml`; presets never inherit from presets. `--override KEY=VALUE` keys are validated against `defaults.yaml`.
- `configs/defaults.yaml` IS the final Craftax Classic recipe, not a neutral baseline. Presets are delta-only; restating a defaults value silently pins it (enforced by `tests/test_config.py`).
- Schedule keys (budgets, warmup, beta target, buffer retention, val cadence) are declared in env frames / update cycles; `resolve_num_updates()` and `resolve_scaled_hyperparams()` derive the update-step forms at load. Never set a derived form. Read README.md §Configuration before touching any YAML.
- Cluster siblings `final_craftax_classic_{qmul,ucl}.yaml` / `final_craftax_{qmul,ucl}.yaml` may differ only in `num_envs` and `seed` (test-enforced); Full-Craftax recipe changes go in both files. Never edit `final_*` or ablation machine configs to fit local hardware.
- The ablation suite has its own config precedence chain and no `--override`; read experiments/README.md before touching `experiments/rl_finetuning/configs/`.

## Checkpoints

- The model is built from the config: match config to checkpoint. Released HF checkpoints load with `final_*` configs, not `defaults.yaml`.
- Pass a checkpoint directory to checkpoint flags, not a step subdirectory.
- Released PPO expert checkpoints fail to restore on CPU-only machines (README.md §Implementation notes).

## Tests

- Run `uv run pytest` after any change. `tests/conftest.py` forces CPU (`JAX_PLATFORMS=cpu`) and disables W&B; no custom markers.
- `tests/test_config.py` guards the preset/pin/cluster-sibling rules above — a config change that breaks it is wrong until proven otherwise.

## Environment and dependencies

- `Craftax_Baselines/` is an upstream-derived submodule (PPO expert training, env wrappers; upstream CLI preserved). Do not edit it as project code; changes land in the fork and are pinned via submodule bump.

