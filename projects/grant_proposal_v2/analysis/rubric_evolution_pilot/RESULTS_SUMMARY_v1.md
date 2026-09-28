# Rubric Evolution Pilot — Results v1

**Date**: 2026-04-23
**Purpose**: End-to-end validation of the GER-CR-v1 rubric-evolution wiring
(`rubric_buffer.py` + `rubric_gen_prompt.py` + `rubric_evolution.py`) using
Claude subagents as the generator client, before committing to a full
`train_ger_cr_v1.py` fork.

## Setup

- **R_persist (shown to generator, not scored here)**: v8 signal set, 10 items
- **R_active (input)**: empty (first elicitation)
- **Policy rollouts**: existing `projects/grant_proposal_v2/runs/2026_04_22_v8_foundopt_B4/buffer.jsonl`, iter 10 revision entries (4 plans)
- **Control plan**: reference proposal `dataset/goals/01_foundopt/reference_proposal.md`
- **Generator**: `Agent` tool (`subagent_type=general-purpose`), one subagent per pair

## Three contrast pairs

| Pair ID | Plan A | Plan B |
|---|---|---|
| `pair_01_best_vs_ref` | iter-10 best B4 plan (reward=0.835) | Reference proposal |
| `pair_02_mid_vs_ref` | iter-10 median B4 plan (reward=0.720) | Reference proposal |
| `pair_03_best_vs_worst` | iter-10 best (0.835) | iter-10 worst (0.710) |

## Output — 9 total items, 5 positive + 4 negative, no duplicates

### Pair 01 (best vs ref): 4 items
- **POS** *Methodological Conceptual Innovation* — "Plan B defines 'functional stationarity' as a shift from parameter-space to function-space convergence; Plan A lists standard machinery without a comparable unifying new concept."
- **POS** *Theoretical Limitation Acknowledgment* — "Plan B notes NTK describes only the 'lazy training' regime and commits to parallel mean-field analysis."
- **NEG** *Avoids Fabricated Quantitative Precision* — "Plan A attributes '85%', '60%', '40%', '30%', '90%' to citations that don't contain those numbers."
- **NEG** *Avoids Self-Scoring Rubric Padding* — "Plan A closes with 'Revisions Summary' listing twelve rubric dimensions with '(5/5)' self-scores."

### Pair 02 (median vs ref): 2 items
- **POS** *Methodological Thesis* — "Single named conceptual pivot vs stacking orthogonal techniques."
- **NEG** *Fabricated Numerical Findings* — "Plan A attributes '60% of ResNet-50 models' with curvature κ < 10⁻³ to Zhang et al. 2022 — not in that paper."

### Pair 03 (best vs worst): 3 items
- **POS** *Goal-Gap Aim Coverage* — "Plan maps each gap in the goal to a distinct aim; worst plan omits benchmark design."
- **POS** *Baseline-Relative Metrics* — "Quantitative targets paired with named baseline vs standalone thresholds."
- **NEG** *Rubric-Named Section Headers* — "Avoids '## Problem Specificity (5/5)' style headers with embedded scores; worst plan structures entire body this way."

## Success criteria verdict

| Criterion | Status |
|---|:-:|
| All 3 response JSONs parse without errors | ✓ |
| ≥3 total new rubric items across pairs (got 9) | ✓ |
| ≥1 negative rubric elicited (got 4) | ✓ |
| No duplicates across the 3 pairs | ✓ |
| Items target project-known Goodhart modes | ✓ (3 out of 4 negatives map directly: fabrication × 2, template/self-scoring × 2) |
| Items grounded in observable differences (not priors) | ✓ (every item cites specific content from a specific plan) |
| Buffer mechanism works (dedup + add + filter) | ✓ (dedup + add verified; std-filter unit-tested in `rubric_buffer._run_sanity_checks`) |

**Filter dry-run note**: the smoke-test seeded random grades produced no
dead-channel cases by chance; filter logic itself is unit-tested in the
buffer module.

## Qualitative observations

1. **Generator understood the Goodhart-targeting instruction**: the two
   independent negatives about fabricated numerical precision (pairs 01 & 02)
   converged on the same failure mode from different starting text, and the
   section-header hack (pair 03) is exactly the class of template-collapse
   artifact mentioned in the system prompt's priority-target list.

2. **Pair 03 (best-vs-worst, policy-internal) surfaces template-collapse
   cleanest**: the worst iter-10 B4 plan literally uses rubric item names as
   section headers with self-awarded '(5/5)' scores — a textbook Goodhart
   artifact invisible to rubric items that only check presence, not framing.

3. **Pair 02 (median-vs-ref) extracted a "methodological thesis" positive
   criterion that is not covered by any existing v8 signal** and is harder to
   game by adding more equations — suggests R_active can fill a genuine gap
   left by R_persist.

4. **Generator refrained from generic criteria**: no "should cite sources", no
   "should be feasible", no "should have a timeline". This validates the
   "based on observed differences" constraint from the OnlineRubrics-style
   prompt.

## Implication for full trainer fork

The pilot clears the gating criterion I set in the plan (section G, Phase 2).
Proceeding to `train_ger_cr_v1.py` fork is justified. Remaining design choices
for the fork (unchanged by pilot):

1. **Sync vs async subagent**: Pilot used sync Agent calls from Claude Code
   session. For an automated 35-iter run, the trainer needs a file-bus: write
   request at iter end → sleep → read response before next iter scoring.
   A separate daemon (or manual user trigger) runs the subagent.
2. **Binary grader for R_active**: R_persist keeps Likert 1-5 (v8 grader
   prompts unchanged); R_active items need a new binary grader prompt. Items
   are already phrased as discrete criteria so a "does Plan X satisfy Y?
   Yes/No" prompt works.
3. **Aggregate with R_persist**: cleanest is weighted-sum with R_persist
   normalized to [0,1] and R_active aggregate (from DR Tulu Eq. 1) combined
   with a free coefficient α. Start α=0.3 (R_active contributes 30% of total)
   and ablate.

## Files

- Request JSONs: `requests/pair_{01,02,03}_*.json`
- Response JSONs: `responses/pair_{01,02,03}_*.json`
- Build script: `build_pair_requests.py`
- Ingest script: `ingest_responses.py`
- Final buffer state: `pilot_buffer_after_merge.jsonl`
