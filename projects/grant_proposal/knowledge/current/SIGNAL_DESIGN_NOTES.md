# D4 Signal Design History & Current State

*Updated: 2026-04-19*

## Current: D4-v4 (8 orthogonal signals for EPSRC outlines)

### Signal Set

| Signal | Name | Weight | What it checks |
|--------|------|--------|---------------|
| G1_problem_specificity | Problem Specificity | 0.12 | Named technical problem vs generic framing |
| G2_objectives_clarity | Objectives Clarity | 0.12 | Countable concrete objectives vs vague aspirations |
| G3_technical_evidence | Technical Evidence | 0.14 | Named papers, findings, failure cases cited |
| G4_named_methods | Named Methods | 0.12 | Specific techniques named vs "AI methods" |
| G5_gap_identification | Gap Identification | 0.12 | Why current approaches are insufficient |
| G6_novel_contribution | Novel Contribution | 0.14 | Substantive vs asserted novelty claims |
| G7_beneficiaries | Named Beneficiaries | 0.10 | Specific sectors/groups named |
| G8_impact_mechanisms | Impact Mechanisms | 0.14 | Concrete delivery mechanisms + UK positioning |

### Sanity Check Results (18 funded EPSRC AI outlines)

```
Per-signal averages:
  G1_problem_specificity   2.56/5
  G2_objectives_clarity    3.17/5  ← best performing
  G3_technical_evidence    1.33/5  ← most discriminating (good outlines cite papers)
  G4_named_methods         1.56/5  ← many outlines use generic "AI" language
  G5_gap_identification    3.11/5
  G6_novel_contribution    2.83/5
  G7_beneficiaries         2.59/5
  G8_impact_mechanisms     3.00/5

Aggregate avg: 0.375
Range: 0.035 — 0.735
≥ 0.7: 2/18 (FoundOpt 0.700, MOSAIC 0.735)
≥ 0.5: 4/18
```

### Key Finding: D4-v4 DISCRIMINATES between good and bad outlines

| Quality Tier | Examples | Aggregate |
|---|---|---|
| Best | MOSAIC (0.735), FoundOpt (0.700) | ≥ 0.7 |
| Good | Probabilistic AI (0.640), Knowledge Integration (0.570) | 0.5-0.7 |
| Average | Edge AI (0.350), CoDa (0.360) | 0.3-0.5 |
| Weak | Transport Hub (0.090), Robotics (0.100), Erlangen (0.035) | < 0.2 |

The spread (0.035 — 0.735) is excellent for training. FoundOpt/MOSAIC score high because they cite specific papers, name concrete methods, and articulate clear gaps. Transport/Robotics score low because they use generic "AI" language without technical specifics.

## Version History

### D4-v1 (discarded): D3-adapted
- 12 signals micro-adjusted from D3's S1-S9 + S2a + SA
- Problem: D3 bias — signals designed for ML research plans, not grants
- Funded NIH proposals scored 0.33 aggregate

### D4-v2 (discarded): Agency-first, 6 signals
- 6 signals from NSF/NIH/ERC criteria (significance, approach, preliminary, feasibility, positioning, broader impacts)
- Problem: signals too coarse for grader — each checking 3-4 sub-dimensions simultaneously
- Funded NIH proposals scored 0.30-0.41 aggregate

### D4-v3 (discarded): EPSRC-calibrated, 5 signals
- 5 signals aligned to EPSRC criteria
- Problem: still too coarse (noise too high per signal), rubric granularity mismatched outline abstraction level
- Funded EPSRC outlines scored 0.288 avg

### D4-v4 (superseded by v5): 8 orthogonal narrow signals
- 8 signals, each checking ONE dimension via simple counting
- Best discrimination yet: 0.035-0.735 range on same 18 outlines
- Top outlines (FoundOpt, MOSAIC) correctly identified as highest quality

### D4-v5 (superseded by v7): 10 signals for full grant proposals
- Redesigned from v4 using NIH reviewer feedback on Clayton R21 pilot
- Perturbation-validated on Clayton R21 pilot: 6/7 target drops ≥1 point
- **Problem**: Opus eval showed counting-based rubrics trivially gamed (all generated plans 7/20 vs reference 19/20)

### D4-v6 (superseded by v7): 10 hybrid signals
- Replaced G4 Named Methods → Research Focus (D3 S9), G6 Novel Contribution → Reasoning Depth (D3 S1), G11 Preliminary Evidence → Evidence Rigor (D3 S2)
- **Problem**: G6 Reasoning Depth also Goodharted — model writes syntactically complex "justifications" without semantic content. Missing formalism and risk signals flagged by Opus eval.

### D4-v7 (current): 12 hybrid signals
- Added G12 Mathematical Formalism (from D3 S2a) and G13 Risk Awareness (from D3 S6)
- Saturated structural signals (G1/G2/G5/G10) down-weighted to 0.04 each
- G3 raised to 0.10 for ML domain (Qwen3 can cite real papers in AI)
- Depth signals (G4/G6/G11/G12) at 0.12, evidence (G3/G13) at 0.10, proposal (G8/G9) at 0.08
- Per-goal weight overrides via `dataset/goals/{goal}/weights.json`
- 8-goal dataset: 6 AI/ML (FoundOpt, RL, Probabilistic, Trustworthy, Causal, NeSy) + 2 non-AI (Oncology, Ecology)

## Dataset Collection (2026-04-19)

Collected 204 valid proposals + 46 reviewer summaries from 7 sources. Full catalog at `data/proposals/catalog.csv`.

Key stats: 82% US, heavily biomedical. 87 full proposals, 64 short proposals, 13 outlines (valid PDFs only).

## NIH R21 Pilot for Signal Redesign (2026-04-19)

Selected **R21_Returning_Pediatric_Genomic_Research_Results_Clayton** as pilot reference:
- Files: `pilots/nih_r21_pediatric_genomics/`
- Funded NIH R21 with 3 reviewers' detailed per-criterion scores
- NIH uses 5 criteria: **Significance / Investigator / Innovation / Approach / Environment** (1-9 scale, 1=best)

Reviewer scores for this proposal:

| Criterion | R1 | R2 | R3 | D4-v4 analog |
|-----------|----|----|----|----|
| Significance | 2 | 1 | 4 | G1 (problem) + G5 (gap) |
| Investigator | 1 | 1 | 1 | — (not in D4 scope) |
| Innovation | 3 | 2 | 3 | G6 (novelty) |
| Approach | 3 | 2 | 3 | G4 (methods) + partial G3 (evidence) |
| Environment | 1 | 1 | 1 | — (not in D4 scope) |

**Key insight**: NIH's 5 criteria map to D4-v4's 8 signals, but with gaps:
- Investigator and Environment are person/institution-dependent — **cannot be evaluated from proposal text alone**
- D4-v4's G7 (beneficiaries) and G8 (impact mechanisms) are UK/EPSRC-specific — need generalization
- G2 (objectives clarity) and G3 (evidence) don't map cleanly to any single NIH criterion

## D4-v5 Perturbation Sanity Check (2026-04-19)

Pilot: Clayton R21 (Returning Pediatric Genomic Research Results)
Grader: Qwen3-30B-A3B, temperature=0.0, single-call prompt

### Reference Scores

```
G1=3  G2=3  G3=5  G4=5  G5=4  G6=5  G8=3  G9=4  G10=5  G11=5  Avg=4.2
```

All signals ≥ 3/5 on reference. G1/G2 at floor (3) — proposal has single-sentence aim and legal framing, not multi-faceted STEM problem. G3/G4/G10/G11 at ceiling (5) — extensive citations, named databases, full methodology, strong preliminary data.

### Perturbation Results

| Perturbation | Target Signal | Ref | Pert | Δ | Status |
|---|---|---|---|---|---|
| P1 Vague Problem | G1_problem_specificity | 3 | 1 | -2 | PASS |
| P1 Vague Problem | G5_gap_identification | 4 | 5 | +1 | FAIL (noise) |
| P2 No Evidence | G3_technical_evidence | 5 | 1 | -4 | PASS |
| P2 No Evidence | G11_preliminary_evidence | 5 | 1 | -4 | PASS |
| P3 Generic Methods | G4_named_methods | 5 | 1 | -4 | PASS |
| P4 Scope Inflation | G9_scope_feasibility | 4 | 1 | -3 | PASS |
| P5 Missing Coverage | G10_approach_coverage | 5 | 3 | -2 | PASS |

**6/7 PASS.** One failure (P1→G5) is grader variance: across 4 runs G5 reference ranges 4-5, P1 ranges 3-5. Not a signal design issue; would resolve with N=3 median grading.

### Key Observations
- G4 broadening (v4→v5) critical: v4 scored reference 1/5 because rubric only had STEM examples. v5 includes legal databases + analytical frameworks → 5/5.
- Collateral effects exist: P1 also dropped G6 (5→1) because Innovation section was vague-ified. Expected — signals ARE correlated when perturbations are broad.
- G9 (scope feasibility) is the most sensitive new signal: P4's "193 UN member states" immediately drops from 4→1.

## Next Steps

1. Run D4-v5 on 5-10 diverse proposals from catalog to check generalization
2. First CR-v7 pipeline run with D4-v5 signals
3. Investigate G1/G2 floor effect — may need recalibration for legal/policy proposals
