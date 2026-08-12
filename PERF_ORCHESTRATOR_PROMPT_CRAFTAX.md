# Task: run the three Craftax performance passes, then the ablation smoke test

You are running four documents back to back on the 4070 Ti box. Each one is a full task with its
own phases, gates and deliverable. Your job is to run them in order, keep the tree green between
them, and commit as you go.

Do not summarise the four documents or plan them all up front. Open one, do it, finish it, then
open the next.

## Order

| # | Document | Deliverable |
|---|---|---|
| 1 | `PERF_DAGGER_PROMPT_CRAFTAX.md` | `PERF_DAGGER_RESULTS_CRAFTAX.md` |
| 2 | `PERF_OFFLINE_PROMPT_CRAFTAX.md` | `PERF_OFFLINE_RESULTS_CRAFTAX.md` |
| 3 | `PERF_EXPERIMENTS_PROMPT_CRAFTAX.md` | `PERF_EXPERIMENTS_RESULTS_CRAFTAX.md` |
| 4 | `ABLATION_SMOKE_PROMPT.md` | `ABLATION_SMOKE_REPORT.md` |

1 to 3 are sequential by design: prompt 1 owns the shared files and its `NUM_ENVS` decision is an
input to prompt 2. Do not reorder them and do not run them in parallel.

## Baseline, once, before document 1

```
uv sync --extra cuda12
uv run --no-sync python -m pytest tests -q
uv run --no-sync ruff check src tests experiments
uv run --no-sync python main.py --mode smoke
git log --oneline -1 && git status --short
```

Record the test count, the lint error count and the current HEAD. That is your baseline. Any
pre-existing failure stays pre-existing: note it, do not fix it, do not let it drift.

## Between each document

Before moving to the next one, all of this must pass:

```
uv run --no-sync python -m pytest tests -q
uv run --no-sync ruff check src tests experiments
uv run --no-sync python main.py --mode smoke
git status --short          # must be clean
```

Test count must be at or above baseline and lint errors at or below it. **If anything is red,
stop and report. Do not start the next document on a red tree.**

## Commits

Commit granularly, as each document already instructs: one commit per change, with the suite and
lint green at that commit, so any single change can be reverted without unpicking the others.
Message states the defect and the fix.

Commit the results document as its own final commit for that pass.

**Never push.** The author reviews before anything reaches `main`. You have standing approval to
commit locally for this task and no approval to push, tag, rebase or force anything.

## Stop conditions

Each document has its own gates that say "stop and report". Honour them. Then:

- **An unresolved author decision** (a VRAM shortfall, a recipe question, a missing checkpoint):
  record it in that document's report and check whether the next document's inputs are still
  defined. Prompt 2 needs prompt 1's `NUM_ENVS`; prompt 3 does not. Continue if you can, stop if
  you cannot, and say which.
- **A red tree you cannot make green**: stop entirely. Do not continue to document 4.
- Document 4 is independent of 1 to 3. Run it even if an earlier pass stopped on an author
  decision, provided the tree is green.

## Before document 4: compact

Documents 1 to 3 will have filled your context with measurements you no longer need. Before
opening `ABLATION_SMOKE_PROMPT.md`, run `/compact` so the smoke pass starts with room to work.

Carry forward only these, and write them down before compacting so they survive:

- the repo path and the current HEAD
- the CUDA extra, the `LD_LIBRARY_PATH` state and the XLA memory settings you settled on
- the checkpoint paths you used
- the test and lint baseline
- any stop condition still open

Everything else is in the three results documents on disk. Re-read them if you need them.

## Two corrections to `ABLATION_SMOKE_PROMPT.md`

That document was written before this work and two of its preflight lines are now wrong. Apply
these and note them in the smoke report:

1. **`uv sync --extra cuda` (line 24) is not a valid extra.** `pyproject.toml:28-29` defines
   `cuda12` and `cuda13` only, and they conflict by design. On this box the driver is 560.35.03
   and CUDA 13 needs 580, so the correct command is `uv sync --extra cuda12`. Use the same extra
   the three perf passes used.
2. **Do not run `git pull origin main` (line 16), and do not expect HEAD at `2d41229`.** That
   commit is an ancestor of this branch, and by the time you reach document 4 you will be many
   commits ahead of it with unpushed local work. Pulling risks a merge you have no mandate to
   resolve. Record the actual HEAD instead and carry on. The rest of that preflight, including
   the Craftax texture-cache warm-up and the `checkpoints/` check, still applies exactly as
   written.

Also reconcile the paths: that document assumes `/workspace/craftax-ReMDM-planner` and writes
scratch output to `/workspace/smoke/` and `/workspace/mem/`. Use the real repo path on this box
and a writable scratch directory on local disk, and state both in the report.

## Reporting

Four documents, four reports, each in the repo's evidence style. Then one short covering note,
`PERF_RUN_LOG_CRAFTAX.md`:

| Pass | Result | Commits | Wall clock | Open items |

Plus the box identification once, the baseline-versus-final test and lint counts, and a single
list of everything left for the author to decide. That list is the most useful thing you produce:
these documents deliberately push recipe and sizing decisions back to the author rather than
guessing, so collect them in one place instead of leaving them scattered across four reports.
