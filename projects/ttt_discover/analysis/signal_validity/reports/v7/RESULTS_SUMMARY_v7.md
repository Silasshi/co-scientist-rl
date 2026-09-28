# V7 Rubric Results: Structural Separation via Explicit Counting

## TL;DR

V7 rewrote 7/9 signal rubrics with explicit COUNT → SCORE tables, replacing
qualitative judgment with deterministic rules. Result: structural separation
doubled, no aggregation tricks used.

## Detection Accuracy (target-signal drop ≥1 after perturbation)

| Signal | v6 | v7 | Δ |
|--------|----|----|---|
| S1_depth (new) | — | **70%** | new |
| S2_rigor (new) | — | **100%** | new |
| S3_positioning | 44% | 56% | +11 |
| S4_significance | 44% | 17% | **-28** |
| S5_feasibility | 89% | 89% | 0 |
| S6_risk_awareness | 88% | 88% | 0 |
| S7_specificity | 47% | 64% | +17 |
| S8_scope | 44% | **100%** | **+56** |
| S9_focus | 24% | 41% | +18 |

Mean detection (excluding S4): **v6 54% → v7 75%** (+21 pp)

## Aggregate Separation (refs vs perturbations)

| Metric | v6 | v7 | Δ |
|--------|----|----|---|
| Ref mean | 0.785 | 0.764 | −0.021 |
| Pert mean | 0.757 | 0.699 | −0.058 |
| Gap (ref μ − pert μ) | 0.028 | **0.065** | **×2.3** |
| AUC (P(ref > pert)) | 0.619 | **0.686** | +0.067 |
| Perts below ref P50 | 65% | **81%** | +16 pp |
| Perts below ref P25 | 37% | 40% | +3 pp |

Key: the gap between ref and pert distributions more than doubled through
structural rubric changes — no aggregation tricks.

## Regressions

1. **S4_significance: 44% → 17%**. The WHO/WHAT/WHY counting rule is too
   forgiving — the perturbation only modifies Problem section, and the other
   sections (Motivation, Core Idea) still contain significance components.
2. **Opus correlation dropped: r=0.395 → 0.158**. V7 rubrics penalize lexical
   surface markers (vague words, unjustified techniques) that humans don't
   weight heavily. Explicit counting ≠ human quality judgment.
3. **Halo effects amplified**. P_S5_vague_method now causes +3.07 drop on
   S7_specificity (the V/C counter triggers on the same vague words).
   S9_focus absorbs halos from most perturbations (+0.95 from P_S1, +0.94
   from P_S5, +0.59 from P_S6).

## Where V7 Worked and Why

- **S8 (100%)**: Counting trigger words like "universal", "across all domains"
  is unambiguous. Works when perturbation adds/removes specific lexical items.
- **S2 (100%)**: FAIR vs STRAWMAN baseline classification with operationalized
  criterion check. Perturbation replaces named baselines with "random" or
  "hand-coded heuristic from 2015" — easy for the grader to flag.
- **S1 (68%)**, **S7 (64%)**, **S3 (56%)**: Moderate wins on counting rules
  that tolerate some subjectivity.

## Where V7 Didn't Work

- **S4 (17%)**: The 3-component (WHO/WHAT/WHY) counting is too forgiving when
  the perturbation is localized to one section.
- **S9 (41%)**: "Justified vs unjustified technique" classification is
  inherently subjective. Grader varies widely between repeats.

## V8 Plan

1. **Keep**: S8, S2, S1, S7, S3 rubrics (clear wins).
2. **Fix S4**: Require the COMPLETE absence of the WHO/WHAT/WHY components
   in the PROBLEM section specifically, not the whole plan. Localize the test.
3. **Fix P_S4 perturbation**: modify Problem + Motivation sections (not just
   Problem) to ensure thorough degradation.
4. **Reduce S9 subjectivity**: simplify to count TECHNIQUES MENTIONED IN CORE
   IDEA only (not Methodology), to reduce over-counting.
5. **Address S5→S7 halo**: make vague-marker list for S7 NOT overlap with
   feasibility concerns (S5 domain).
6. **Investigate Opus correlation drop**: look at which refs lost aggregate
   score under v7 and compare with Opus ratings.
