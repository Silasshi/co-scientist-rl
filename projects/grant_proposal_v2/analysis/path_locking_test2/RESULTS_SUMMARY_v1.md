# Path-Locking Test 2 — Results Summary v1

**Date**: 2026-04-23
**Notebook**: `path_locking_test2_v1.ipynb` (executed version: `_executed.ipynb`)
**Data**: 3 v8 FoundOpt runs (B4, MAIN_C3 SDPO, MAIN_C4 aggregate), 276 plans after filter
**Method**: TF-IDF (1-3 grams, stopword-filtered) cosine similarity to reference proposal

## Verdict: Path-locking NOT confirmed — RL pushes policy AWAY from reference

### Primary measurement

| Condition | n | sim_to_ref mean | sim_to_ref std | Bootstrap 95% CI on Δsim vs B4 |
|---|:-:|:-:|:-:|:-:|
| B4 (no RL) | 88 | **0.0965** | 0.059 | — |
| MAIN_C3 (SDPO) | 93 | 0.0837 | 0.053 | Δ=**−0.013** [−0.029, +0.004] (crosses 0) |
| MAIN_C4 (agg) | 95 | 0.0745 | 0.051 | Δ=**−0.022** [−0.038, −0.006] (significant) |

**Pre-registered threshold** (from DECISIONS.md 2026-04-23): Δsim > +0.05 required to confirm path-locking. Observed Δ is **negative** → falsified.

### Supplementary findings

- **Intra-run diversity** (1 − mean pairwise sim): B4=0.872, MAIN_C3=0.889, MAIN_C4=0.882.
  - MAIN runs are slightly **more** diverse, not collapsed.
  - Rules out "template collapse toward ref" hypothesis.

- **Word counts** (mean per condition): B4=874, MAIN_C3=897, MAIN_C4=852.
  - No dramatic length drift; length hacking from DECISIONS.md is more subtle (appears only relative to B4 MAIN iter-0 baseline, or at signal-level not full-text-length).

- **Opus total (/20)** from `depth_audit_mid.jsonl` (n=3 per condition — small):
  - B4=16.67, MAIN_C3=17.00, MAIN_C4=16.00
  - Does not support "MAIN has lower Opus" side of path-locking criterion either.

- **Audit join**: Failed (0/9 matched by (condition, iter, aggregate_reward)) — aggregate_reward is not unique per plan within iter. Does not affect primary verdict; sim_ref × Opus correlation deferred.

## Interpretation

RL in v8 Qwen-30B setup is **not** making policy imitate reference. Instead it moves policy LEXICALLY away from ref, with no template collapse and no obvious length drift at the full-text level.

This is consistent with "template collapse toward rubric-maxima" (DECISIONS.md 2026-04-20) rather than "imitation of reference". The Goodhart mode is **rubric-hacking**, not **reference-imitation**.

## Caveats

1. **TF-IDF captures lexical overlap only.** Semantic path-locking (same methodology/ideas in different words) would not be detected. However, if RL were path-locking semantically, we would at minimum expect neutral lexical similarity, not significantly negative.
2. **Small Opus audit sample (n=9 total, 3 per run).** The Opus column of the verdict is low-power.
3. **Only one goal (01_foundopt).** Cannot rule out that other goals show path-locking.
4. **Only 3 RL algorithms (SDPO, aggregate, B4).** Does not generalize to untested algorithms.

## Implications (feed forward)

1. **R_persist choice revised**: Option C (retain v8 signals as R_persist) — no path-locking defense needed. See DECISIONS.md 2026-04-23 (follow-up entry).
2. **Paper framing**: v9 story becomes "RL Goodharts toward rubric-maxima templates, not reference imitation" — cleaner than path-locking framing.
3. **GER-CR-v1 motivation strengthened**: If failure mode is rubric-hacking, evolving rubric (to catch newly-emerging hack modes) is the natural counter. Consensus-anchoring loses its main justification.
4. **Decision queue item #1 removed**: Path-Locking Test 2 completed (outcome: retract framing).
5. **Alt reference collection not required** for GER-CR-v1; R_active buffer alone is the new mechanism.

## Files

- Notebook (executable): `path_locking_test2_v1.ipynb`
- Notebook (executed with outputs): `path_locking_test2_v1_executed.ipynb`
- This summary: `RESULTS_SUMMARY_v1.md`
