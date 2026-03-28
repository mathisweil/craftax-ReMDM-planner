# Claude Code Prompt: Intermediate Saves in `run_ablations.py`

## Context

There are two known gaps in crash recovery:

1. `results.json` is only written **once** after *all* ablations finish (around line 496-498). A crash mid-run loses everything.
2. Within a single ablation, training is a single `jax.lax.scan` call — no Python-level hook exists to checkpoint mid-training without restructuring.

---

## Tasks

### 1. Investigate

Read `run_ablations.py` and confirm:

- Exact location of the `results.json` write
- The structure of the ablation loop (`for abl_name in selected_names` or similar)
- How `--analyze_only` + `--results_path` reload works (confirm the infrastructure is already there)
- Whether the `jax.lax.scan` training call could feasibly be chunked

### 2. Plan

Briefly state what you'll change and where **before touching anything**.

### 3. Apply

Make two changes:

- **Cross-ablation save (priority):** Move the `results_path.write_bytes(...)` call (or equivalent) *inside* the ablation loop, so results are written after each ablation completes. The partial file should be loadable by `--analyze_only` without modification.
- **Within-ablation chunking (if scan is straightforward to chunk):** Wrap the `jax.lax.scan` in a Python loop with a `checkpoint_every` parameter (default: sensible fraction of `max_iter`). Add a `--checkpoint_every` CLI flag. If this restructure looks risky or tangled, **skip it and leave a `# TODO` comment instead** — don't break working training code.

### 4. Verify

Do a quick dry-run or trace to confirm the partial results file is valid JSON and the loop still runs correctly.
