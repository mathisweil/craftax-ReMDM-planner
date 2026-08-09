# Correctness changes (CORRECTNESS_PLAN.md, approved 2026-08-09)

Baseline tag: `parity-baseline`. One change per commit; predicted fingerprint
movement stated in each commit message and verified after. Fingerprint
references in `parity/reference/` are the pre-change baseline and are not
regenerated until the Stage 4 retrain.

| Change | Commit | Source | Invalidates |
|---|---|---|---|
| Group 3: annotations, pseudocode cross-refs, `train/env_steps` log alias | align/incidental | METHOD_PARITY 2.1/2.5/2.6 | nothing |
| FIX-2: inference sampler uses the configured Sec 4.1 remasking schedule (+ loop), carry-over restored, Diffuser prefix locking kept; replaces hardcoded `0.15*(1-ratio)` | fix/inference-sampler | Wang Alg 1, Sec 4.1-4.2; Janner Sec 3.3 | every evaluation table/figure produced with the old inference sampler; no checkpoints |
| FIX-1: per-sample loss normalised by `1/H` instead of the realised masked count | fix/loss-estimator | MDLM eq (8)/(10); Shi eq (4) | all trained diffusion checkpoints (superseded, see ../RETRAIN_LOG.md) and every result derived from them; PPO experts unaffected |

Documented decisions (no code change): CH-2 optimiser stack stays Adam +
cosine + warmup (no source governs; per-benchmark tuning documented); CH-3
no EMA added (source absent; would confound the FIX-1 retrain); CH-4 the
DAgger replay buffer stays circular — Ross Alg 3.1 aggregates all data, but
full aggregation at 1e8 frames is memory-infeasible (~5e11 floats); the
deviation is documented here and in ADJUDICATION B-2; CH-5 return weighting
stays clipped-linear, described in its own terms (not AWR; manuscript to
avoid citing AWR/RWR for it).

Old-weights, corrected-sampler evaluation fingerprints: `parity/stage2_eval/`.
