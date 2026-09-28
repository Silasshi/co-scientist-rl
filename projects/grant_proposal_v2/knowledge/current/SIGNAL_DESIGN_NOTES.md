# D4 Signal Design — Current State & History

*Updated: 2026-04-22*

## Current: D4-v7 (12 hybrid signals)

### Signal Set

| Signal | Name | Weight | Category | What it checks |
|--------|------|--------|----------|----------------|
| G1_problem_specificity | Problem Specificity | 0.04 | Structural | Named technical problem vs generic framing |
| G2_specific_aims | Specific Aims Clarity | 0.04 | Structural | Measurable aims with expected outcomes |
| G3_technical_evidence | Technical Evidence | 0.10 | Evidence | Named papers, findings, failure cases cited |
| G4_focus | Research Focus | 0.12 | Depth | Tight core idea vs scattered techniques |
| G5_gap_identification | Gap Identification | 0.04 | Structural | Why current approaches are insufficient |
| G6_reasoning_depth | Reasoning Depth | 0.12 | Depth | Multi-step causal chain reasoning |
| G8_deliverable_clarity | Deliverable Clarity | 0.08 | Proposal | Named tools, benchmarks, publications |
| G9_scope_feasibility | Scope Feasibility | 0.08 | Proposal | Realistic scope, scope red flags |
| G10_approach_coverage | Approach Coverage | 0.04 | Structural | Aims vs methods coverage ratio |
| G11_evidence_rigor | Evidence Rigor | 0.12 | Depth | Domain-specific evidence quality |
| G12_formalism | Mathematical Formalism | 0.12 | Depth | Non-trivial formulas and derivations |
| G13_risk_awareness | Risk Awareness | 0.10 | Depth | Method-specific failure modes identified |

**Weights sum to 1.0.** Structural signals (G1/G2/G5/G10) saturate early → low weight (0.04). Depth signals (G4/G6/G11/G12) drive most learning signal → high weight (0.12).

### Weight Categories
- **Structural (0.16 total)**: G1, G2, G5, G10 — binary/easy to satisfy, saturate at 5/5 early
- **Depth (0.48 total)**: G4, G6, G11, G12 — hard to satisfy, drive quality differentiation
- **Evidence (0.20 total)**: G3, G13 — depends on grounding/citations
- **Proposal (0.16 total)**: G8, G9 — practical deliverables and scope

### Per-Goal Weight Overrides

Via `dataset/goals/{goal}/weights.json`:
- **AI goals (01, 08)**: default D4-v7 weights, G12_formalism active
- **Healthcare (05)**: G3↑0.12, G12↓0.10
- **Biomedical (07)**: G12→G12a_analytical_framework, G3↑0.14, G4↓0.08
- **Criminal justice (09)**: G12→G12a_analytical_framework
- **Climate policy (12)**: G12→G12a, G5↑0.08, G4↓0.06, G13↑0.12

### Signal Source

Code: `src/co_scientist/shared/grant_signal_reward.py`

D4-v7 is a hybrid of:
- **D3 depth signals** adapted for proposals: G4 (from S9_focus), G6 (from S1_depth), G11 (from S2_rigor), G12 (from S2a_formalism), G13 (from S6_risk)
- **Proposal-specific signals**: G8 (deliverables), G9 (scope feasibility)
- **Structural sanity checks**: G1, G2, G3, G5, G10 (partially from D4-v4/v5)

### Known Issues (from SDPO experiment, 2026-04-22)

1. **Qwen-Opus inversion**: D4-v7 aggregate reward correlates NEGATIVELY with Opus quality within high-score plans (ρ ≈ -0.9 on C2/C3/C4). The weighted mean hides individual signal failures.
2. **Signal saturation**: Structural signals (G1/G2/G5/G10) all reach 5/5 from iter 0 → contribute zero gradient
3. **Depth signal Goodhart**: G6, G12 can be gamed by writing "syntactically complex justifications" or "equations that look non-trivial" without genuine depth (confirmed by Opus depth audit)
4. **Citation hallucination**: G3 rewards citing papers, but doesn't verify they exist (Opus flagged fabricated references in C3 best plan, grounding=2/10)

## Version History

### D4-v1 (discarded): D3-adapted, 12 signals
- Micro-adjusted from D3's S1-S9 + S2a + SA
- Problem: D3 bias — signals designed for ML research plans, not grants

### D4-v2 (discarded): Agency-first, 6 signals
- 6 signals from NSF/NIH/ERC criteria
- Problem: too coarse for grader

### D4-v3 (discarded): EPSRC-calibrated, 5 signals
- Problem: still too coarse, noise too high

### D4-v4 (superseded): 8 orthogonal narrow signals
- Best discrimination on EPSRC outlines (0.035-0.735 range)
- Problem: EPSRC-specific (G7 beneficiaries, G8 UK positioning)

### D4-v5 (superseded): 10 agency-neutral signals
- Redesigned from v4 using NIH reviewer feedback
- Perturbation-validated: 6/7 target drops ≥1 point
- Problem: counting-based rubrics trivially gamed by RL

### D4-v6 (superseded): 10 hybrid signals
- Added depth signals from D3 (focus, reasoning depth, evidence rigor)
- Problem: G6 Reasoning Depth Goodharted. Missing formalism and risk signals.

### D4-v7 (superseded by v8): 12 hybrid signals
- Added G12 Mathematical Formalism + G13 Risk Awareness
- Down-weighted saturated structural signals
- Per-goal weight overrides for non-STEM domains
- Status after SDPO experiment (2026-04-22): the AGGREGATE design is sound but
  several individual signals are Goodhart-able. See v8 for fixes.

### D4-v8 (current, 2026-04-22): 10 signals, stricter thresholds, confidence-gated RL

**File**: `src/co_scientist/shared/grant_rubric_v8.py`

**Motivation**: SDPO experiment showed RL increases Qwen aggregate (0.855→0.970)
while Opus quality drops (19→13). Systematic rubric review (see RUN_AUDIT §signal
analysis) found counting-based rubrics with low thresholds enable Goodhart even
under scaffolding, because 'produce N items of the right shape' is the easiest
kind of optimization target for RL.

**Changes from v7**:

| Signal | v7 | v8 | Reason |
|---|---|---|---|
| G3 Technical Evidence | 0.10 | **REMOVED** | Rewards hallucinated citations; Qwen can't verify. C3 SDPO got G3=5/5 but Opus grounding=2/10 (fabricated refs). |
| G5 Gap Identification | 0.04 | **REMOVED** | 99.8% saturated at 5/5 across 1655 plans — zero gradient. Partially covered by G4. |
| G12 Formalism | 0.12 | 0.04 | Qwen can't verify "non-trivial" equations semantically. Kept at low weight as a weak regularizer. |
| G4 Focus | 0.12 | 0.16 | Best discriminator (std 0.57, two-dim gate). Upweight. |
| G6 Reasoning Depth | 0.12 | 0.16 | Best discriminator (std 1.63, inverse counting of asserted choices). Upweight. |
| G9 Scope Feasibility | 0.08 | 0.14 | Inverse signal (red flag counting) — hardest to Goodhart. Upweight. |
| G13 Risk Awareness | 0.10 | 0.14 | Non-counting qualitative — hardest to game by pattern-match. Upweight. |
| G11 Evidence Rigor | 0.12 | 0.14 | Kept; slightly upweighted to cover what G3 used to. |
| G1, G2, G8, G10 | low | low | Stricter thresholds (need 5+/4+ items for top score, stricter 'specific' definitions). |

**Counting threshold changes** (top score 5/5 now requires more):
- G1 problem facets: 3 → 4 distinct independent facets
- G2 measurable aims: 3 → 4 independent aims with outcomes
- G8 deliverables: 4 → 5 specific (with substantive content, not templates)
- G10 coverage: "covered" now requires SPECIFIC methodology, not vague hand-waves

**Confidence-gated RL** (new in v8, in `train_cr_v7.py`):
- Grader runs ≥2 samples per signal at `grader_temperature>0` (e.g., 0.3)
- Confidence per signal = `1 - range / (score_max - 1)` over samples
- RL config `min_grader_confidence` (e.g., 0.6) skips per-signal datums when
  grader confidence is low — prevents RL from exploiting grader blind spots
- Literature: *Cycles of Thought* (2024), *Confidence Improves Self-Consistency* (2025)
- Rationale against verbalized confidence: *EMNLP 2023 Strategies...* showed
  direct single-call "what's your confidence?" is systematically overconfident,
  especially when combined with self-preference bias.

**Validation plan**: B4_v8 and MAIN_v8 runs on FoundOpt → Opus eval. If B4_v8 ≥
D4-v7 B4 (19/40), rubric fix helped frozen grading. If MAIN_v8 ≥ B4_v8, RL under
v8 rubric stops Goodharting.

## Also Exists: Better Reward Custom Rubric (R1-R8)

Separate signal set used in "Better Reward" experiments (2026-04-20):
- 8 signals (R1-R8), 1-10 scale
- Module: `co_scientist.shared.foundopt_rubric_v1`
- Goal: `dataset/goals/01_foundopt_v2/`
- **NOT comparable to D4-v7** (different signals, different scale, different rubric text)
- Results: 235B scored 26-29 Opus, 30B scored 12 Opus (but confounded with model scale)
