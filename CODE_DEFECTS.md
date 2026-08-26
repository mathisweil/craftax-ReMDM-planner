# Task: figure-generator and results-export defects

An external review rendered every paper figure at true print size and cross-checked each one
against the manuscript. Four figure/caption disagreements came back, all traceable to
`scripts/paper_figures.py`. This file also covers finishing the `--emit-tex-macros` work
already in progress.

Read `CLAUDE.md` first. Run `uv run pytest` when done. Regenerate figures with:

```bash
python scripts/paper_figures.py          # reads ../minihack-ReMDM-planner for the MiniHack half
```

**Verify each claim below before changing anything.** Two of the four are reported as figure
defects but look, from the code, like they may be correct code with a presentation problem — a
wrong "fix" there would make things worse. Where the finding turns out to be a manuscript
wording problem rather than a code problem, say so and change nothing.

---

## 0. Commit `scripts/paper_figures.py`

`git status` reports it untracked. It generates every figure in the paper and reads MiniHack's
`results.json` from the sibling checkout. It must be in the released repo.

---

## 1. `fig8_score_vs_kl` — the legend advertises a marker that is not plotted

**Reported:** the legend carries a bold black "Baseline RL" swatch, but pixel analysis of a
400 dpi render finds no black marker in either panel. A reader cannot locate the baseline's own
point.

**What the code says.** `baseline_rl` has `group="Baseline"` in
`experiments/rl_finetuning/ablations/registry.py`, and `GROUP_COLOR["Baseline"] = "#000000"`,
so `condition_color("baseline_rl")` *is* black and the scatter loop does not skip it. The point
should be drawn.

**Most likely cause:** the loop does

```python
kl = series(entry, "repr_drift_kl")
if not kl.size:
    continue
```

so if `repr_drift_kl` is empty or absent for `baseline_rl`, the point is silently dropped while
`group_legend_handles(pretrained=True)` still emits the black swatch (`baseline=True` by
default).

**Do:** check whether `repr_drift_kl` is populated for `baseline_rl` in the Craftax and MiniHack
`results.json`. If it is missing, that is the bug — either recover the series or drop the
baseline from the legend, and make the silent `continue` log which conditions it skipped. If it
is present and the point genuinely is drawn, the finding is wrong; say so.

Two smaller points in the same figure: the legend handles are `Line2D` line swatches for a
**scatter** plot, so no legend entry visually matches any plotted mark — use marker handles.
And the log-axis exponent glyphs render at ~4.55 pt, the smallest text in the figure set.

---

## 2. `fig2_repr_drift` — LoRA's curve is absent, and the caption reports its endpoint

**Reported:** the caption states final Craftax drift "spans 0.00 to 0.34", the 0.00 being LoRA.
At 400 dpi the lowest visible line ends near 0.06 and nothing reaches the axis floor.

**This one is real and the code already explains it.** `fig2_repr_drift` calls `_trace_panel`
with `logy=True`. The comment on `KL_FLOOR` says LoRA's drift probe reads its frozen base
weights and records **exactly 0** on Craftax Classic. Matplotlib drops non-positive values on a
log axis, so LoRA's curve is simply not drawn — while the caption quotes its value as the low
end of the range.

**Do:** apply the same treatment `fig8_score_vs_kl` already uses — pin the exact-zero series to
`KL_FLOOR` — and give it a visually distinct treatment (dotted, or an explicit marker) so it
reads as "pinned, not measured at this value". Leave the underlying data untouched.

The caption also needs rewording, but that is a manuscript change; do not edit the manuscript
here. Just report what the figure now shows so the wording can be fixed there.

---

## 3. `fig6_achievements` — the tier-4 legend colour never appears

**Reported:** the legend declares five tier colours but only four are drawn, with
`collect_diamond` rendered in tier-3 orange rather than tier-4 purple.

**The code looks correct.** `TIER_COLOR[4] = "#8172B3"` (purple) and
`ACHIEVEMENT_TIER["collect_diamond"] = 4`, so the bar is coloured purple.

**The likely truth:** `collect_diamond`'s completion rate is ~0 both before and after, so its
bar has essentially zero height and is invisible; a pixel probe at that x-position picks up the
neighbouring tier-3 bar. The legend then advertises a colour the reader can never find.

**Do:** verify by printing `pre` and `post` for `collect_diamond`. If the rate really is ~0,
**do not change the colour mapping** — instead make the near-zero bar legible (a minimum drawn
height, or a marked "0" annotation), so the tier-4 entry corresponds to something visible.
Report the actual rates either way.

---

## 4. `fig2`, `fig3`, `fig4` — named conditions cannot be identified

The manuscript singles conditions out by name ("excluding LoRA", "normalised advantages
aside"), but these three panels encode by **group colour only** — 5 colours for 25 conditions,
up to 7 per group. A reader has to take the text's word for which line is which.

`_annotate_condition(ax, entry, label)` already exists and is used elsewhere. Apply it in these
three figures to the conditions the text names — at minimum `lora` and `normalized_adv` — so
every condition discussed in prose is findable in the figure.

---

## 5. Finish `--emit-tex-macros`

There is uncommitted work adding `--emit-tex-macros` to `run_ablations.py` and ~196 lines to
`analysis/tables.py`, emitting `tables/results.tex` with one `\newcommand` per headline number.

This matters more than it looks: **the manuscript currently has no `results.tex` at all** and
every reported number is a hand-typed literal, with no link from any number back to the run
that produced it. Finishing this is what makes the paper's numbers auditable.

- Complete and test the path, including under `--merge`.
- Cover at least: pretrained score, per-condition score and seed sd, the four group means,
  pooled seed sd, `CV_A`, and ESS.
- Macro names must be valid TeX control sequences — letters only, no digits or underscores —
  so `advantage_clip` needs a documented mangling rule. Make it deterministic and collision-free,
  and assert on collisions rather than silently overwriting.
- Emit `\newcommand` (not `\def`), one per line, definitions only — the manuscript convention is
  that the file contains nothing else.
- Add a test asserting the emitted file parses as macro definitions only and that every macro
  name is unique.

---

## 6. Confirm, do not change: `fig9` filled-point count

`DISTINCT_WEIGHTING` has **five** members — `baseline_rl`, `bc_wins`, `reward_filtering`,
`running_stats`, `reward_model` — and `fig9_weight_dispersion` fills exactly those.

The manuscript's hyperparameter appendix names only **four** conditions that change the
weighting step, omitting `baseline_rl` itself. The code and the caption ("five conditions with a
distinct weighting rule") agree with each other; the appendix prose is the odd one out.

**Verify this reading and report it. Change no code.** The fix belongs in the manuscript.

---

## 7. Report back

For each of items 1–4, say whether the reported defect was confirmed in code, confirmed as a
presentation problem, or not reproduced. List every regenerated figure. Where the real fix is a
manuscript wording change, state the wording the figure now supports — but do not edit anything
under the paper's `src/`.
