# Config inheritance refactor

Make presets carry only their deltas, so `defaults.yaml` is the single source of
truth. The sibling repo `../minihack-ReMDM-planner/` has already had this done;
read its `configs/`, `src/config.py` and `experiments/rl_finetuning/run_ablations.py`
for the reference implementation.

Nothing about training behaviour may change. This is the hard constraint: no
config value may move.

## Scope

**Main repo.** `configs/` presets are fat (`final_craftax_qmul.yaml` 56 keys,
`final_craftax_ucl.yaml` 53, against a 65-key `defaults.yaml`); the six
`classic_exp_*` / `craftax_exp_*` families likewise. Strip every key that is
byte-identical to its `defaults.yaml` value. Keep the prose comments.

Config loading lives in `main.py` (~lines 186-210), not `src/config.py`. It
already deep-merges `--config` onto `defaults.yaml`, so stripping alone needs no
code change. Add an `extends:` key so a preset can inherit from another preset,
then use it to make each `*_ucl` config inherit from its `*_qmul` sibling, which
puts the "identical hyperparameters across clusters" invariant in code rather
than prose.

**Experiments.** `experiments/rl_finetuning/configs/` has four machine configs
(~73 keys each) that currently *replace* `ablations_default.yaml` rather than
layer onto it. Give `run_ablations.py` an `extends` loader and make them deltas.

Merge order: `defaults.yaml` -> `ablations_default.yaml` -> machine config ->
`ablations_fast.yaml` (`--fast` only) -> CLI flags.

## Traps

- Load `ablations_fast.yaml` **raw**, outside the extends chain. Through the
  chain it drags default values back over the machine config.
- Because the machine configs currently replace the base, keys absent from them
  fall through to `getattr(cfg, ..., fallback)` defaults in `ablations/`. Once
  inherited they take the YAML value. For every newly inherited key, confirm the
  `getattr` fallback equals the `ablations_default.yaml` value. Where it differs,
  runs predating the change used the fallback and their numbers shift: report it,
  do not silently absorb it.
- `extends` must be valid in a config file but rejected as an `--override`.
- Strip `extends` before building the namespace.
- Cycles raise `ValueError` naming the loop; a missing base raises
  `FileNotFoundError` naming both files.

## Verify

Diff every preset's *fully merged* config against `git show HEAD:<path>` merged
the old way, for both repos' config sets. Report any key whose value moved, and
list keys that are newly inherited. Then run `pytest`, `main.py --mode smoke`, a
`--fast` ablation run on a machine config (confirm its values survive the fast
overlay), and `ruff check` / `ruff format --check` against a pre-change baseline.

Do not change anything under `src/planners/`.

## Docs

Update `README.md` and `experiments/README.md`: the precedence chain, the
"presets hold only deltas, never restate a default" rule, and the preset table.
Only claim `extends` where a file actually uses it.
