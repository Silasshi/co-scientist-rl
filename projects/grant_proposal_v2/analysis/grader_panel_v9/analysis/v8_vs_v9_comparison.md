# v8 vs v9 Rubric — Minimal Grader Panel Comparison

*2026-04-22. 15 FoundOpt plans × 5 changed signals × GPT-OSS-120B + Opus-4.7 (v9-re-graded).*

## Summary

v9 **closes length bias** (all 3 target R² → < 0.15, down from 0.77-0.87) and
**revives dead channels** (G4/G10 variance restored from ≤0.38 to ≥0.89).

BUT v9 **breaks Opus alignment** on the 3 length-hacked signals — the stricter
prompts cause GPT-OSS and Opus to disagree on what counts as "verified fair
baseline" (G11), "used equation" (G12), or "complete risk 3-tuple" (G13).

**Trade-off, not clean win**. Not ready for training run yet.

## Per-signal table

| Signal | v8 R²(len) | v9 R²(len) | v8 var | v9 var | v8 ρ(Opus,GPTOSS) | v9 ρ(Opus,GPTOSS) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| G4_focus | 0.117 | **0.619** | 0.38 | **0.89** | +0.58 | **+0.79** | dead-channel ✅, new length bias ⚠️ |
| G10_approach_coverage | 0.000 | 0.548 | 0.00 | 2.24 | nan | **+0.75** | dead-channel ✅, new length bias ⚠️ |
| G11_evidence_rigor | 0.773 | **0.137** | 3.17 | 0.14 | +0.91 | **+0.23** | length ✅, Opus-ρ ❌ (−0.68) |
| G12_formalism | 0.872 | **0.137** | 4.03 | 1.98 | +0.93 | +0.63 | length ✅, Opus-ρ just below 0.65 |
| G13_risk_awareness | 0.788 | **0.000** | 2.81 | 0.00 | +0.93 | nan | length ✅, over-strict: Opus var=0 |

## Success criteria verdict (5 changed signals × 2 criteria = 10 targets)

- **Length-decorrelation + variance/ceiling**: 5/5 ✅
- **Opus-ρ preservation (≥ 0.65)**: 2/5 ✅ (G4, G10), 3/5 ❌ (G11, G12, G13)

## What went right

- v9 G4 adds composition-logic gate → variance 0.38 → 0.89, AND Opus agrees more
  (ρ +0.58 → +0.79)
- v9 G10 3-tuple procedure+dataset+metric → breaks v8's saturation at 5.0 (100%
  of plans at ceiling → 13% at ceiling), AND Opus alignment +0.75
- v9 G11/G12/G13 3-tuple / used-equation / trigger+detection+fallback
  requirements → length R² collapses from 0.77-0.87 to 0.00-0.14

## What went wrong

**New length bias introduced on G4 and G10**: composition-logic + 3-tuple
coverage both require MORE text to satisfy, so longer plans pass them more
often. R²(length → G4) climbs from 0.117 to 0.619 (worse than v11 at 0.46).
DECISIONS.md Risk 2 realized.

**Opus-ρ collapses on G11/G12/G13**:
- G11 ρ +0.91 → +0.23. GPT-OSS and Opus disagree on whether a baseline has a
  "quoted methodology snippet showing usage". The ambiguity lets each grader
  apply different strictness.
- G12 ρ +0.93 → +0.63. Borderline. GPT-OSS may be applying the "referenced
  downstream" check more leniently than Opus.
- G13 ρ = nan. Opus gave 11/15 plans score 1 and 4/15 score 2 (var=0.21) — the
  rubric is too strict at the current sample. GPT-OSS gave more 1s but some 2s
  too; the low variance on both sides breaks the Spearman calculation.

## Root cause: strictness-to-clarity trade-off

The v9 rewrite traded two properties:
- **Gained**: length-decorrelation by requiring verifiable sub-structures
  (citation+usage+scale, defined+used, trigger+detection+fallback)
- **Lost**: shared interpretation between graders. "What counts as a USAGE
  sentence?" and "What makes a trigger MEASURABLE?" are judgment calls where
  GPT-OSS and Opus now diverge.

## Decision

v9 as-written is **not ready for training**. Two paths forward:

**Path A — iterate v9 prompts**:
- Relax G13 from 3-tuple (trigger+detection+fallback) to 2-tuple (trigger+fallback),
  allowing at least implicit detection — should raise Opus score distribution off
  the floor.
- Add length caps to G4 composition-logic section (≤3 sentences, counted-based
  "N_composition_statements") to stop length creep.
- G11/G12 may need less strict "used" definition — accept implicit reference
  patterns (e.g., pronoun "it" after equation counts as use).

**Path B — different v9 direction per STATUS.md 2026-04-22 update**:
- Goel 2512 retrofit: violation-based fraction scoring + hard length cap + reference
  as privileged grader info
- Path-locking pilot first (reference-grounded methods contamination check)
- These reframe "rubric design" as a higher-order problem than prompt tweaks

Recommendation: investigate Path B pilot (Test 2) before iterating v9 prompts
further — if path-locking is confirmed, all reference-grounded rubrics (including
v9) have contamination ceiling, and Path A becomes a local optimization on a
flawed setup.

## Data files

- `grades/gpt_oss_120b.jsonl` (150 v9 grades)
- `grades/opus_v9.jsonl` (75 v9 re-grades for 5 changed signals)
- `grades/../grader_panel_v8/grades/opus.jsonl` (150 v8 Opus, baseline)
- `grades/../grader_panel_v8/grades/gpt_oss_120b.jsonl` (150 v8 GPT-OSS, baseline)
- `analysis/comparison.json` (structured comparison data)

## Methods

Metrics computed in `scripts/compute_agreement.py`:
- R²(log_wc → score): OLS on log word count
- Spearman ρ(Opus_v9, GPTOSS_v9): pairwise rank agreement, v9 rubric on both sides
- Variance: sample variance (ddof=1) of scores across 15 plans
- max%: fraction of plans scoring 5 (ceiling hit rate)

Opus v9 grades produced by 5 parallel Claude subagents × 3 plans each × 5 changed
signals using the exact `build_single_signal_prompt` output from v9 rubric.
