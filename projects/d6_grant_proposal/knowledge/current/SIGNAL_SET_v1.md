# D6 Signal Set v1 — 12-signal Grant Proposal Rubric

**Source**: `src/co_scientist/shared/grant_signal_reward.py`
**Migrated from**: D4 (grant_proposal) v7 signal set

## Signal Table

| Signal ID | Name | Weight | Notes |
|---|---|---|---|
| G1_problem_specificity | Problem Specificity | 0.04 | Structural sanity |
| G2_specific_aims | Specific Aims Clarity | 0.04 | Structural sanity |
| G3_technical_evidence | Technical Evidence | 0.10 | Citations, named data |
| G4_focus | Research Focus | 0.12 | Depth signal from D3 S9 |
| G5_gap_identification | Gap Identification | 0.04 | Structural sanity |
| G6_reasoning_depth | Reasoning Depth | 0.12 | Depth signal from D3 S1 |
| G8_deliverable_clarity | Deliverable Clarity | 0.08 | Proposal-specific |
| G9_scope_feasibility | Scope Feasibility | 0.08 | Proposal-specific |
| G10_approach_coverage | Approach Coverage | 0.04 | Structural sanity |
| G11_evidence_rigor | Evidence Rigor | 0.12 | Depth signal from D3 S2 |
| G12_formalism | Mathematical Formalism | 0.12 | STEM only (D3 S2a) |
| G13_risk_awareness | Risk Awareness | 0.10 | Depth signal from D3 S6 |

**Total weight**: 1.00

## Domain override

For non-STEM domains (social science, public policy, criminal justice, education):
- Replace `G12_formalism` with `G12a_analytical_framework`
- G12a definition also in `shared/grant_signal_reward.py:SIGNAL_VARIANTS`

Per-goal weight overrides available in `dataset/goals/{goal_id}/weights.json`.

## Score scale

- All signals: 1–5 (integer)
- Aggregate: weighted mean of normalized scores, where `normalize(s) = (s-1)/(5-1)`
- Result range: 0.0 (all signals = 1) to 1.0 (all signals = 5)

## Hard gates (D4 legacy — NOT used in D6 Phase 1)

D4 used two hard gates (Goal-Contrast Margin G1, Claim Verification G2) from
`shared/seven_signal_reward.py`. D6 Phase 1 does NOT use hard gates — the
reviewer critique provides the same filtering function via SDPO.

If hard gates are added in a later phase, import from:
  `from co_scientist.shared.seven_signal_reward import SIGNALS as HARD_GATE_SIGNALS`

## Prompt builders

All signal prompts are in `shared/grant_signal_reward.py`:
- `build_single_signal_prompt(goal, plan, signal)` — single-signal grading
- `build_single_call_prompt(goal, plan, signals)` — all-at-once grading

The D6 audit uses `build_single_call_prompt` for efficiency.
