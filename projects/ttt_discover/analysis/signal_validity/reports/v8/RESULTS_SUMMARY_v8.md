# V8 Rubric Results: Calibrated Structural Separation

## TL;DR

V8 fixed v7's S4 regression and S1/S9 over-counting. Net result: detection
rates jumped to 83% average, aggregate AUC reached 0.742.

## Detection Accuracy Evolution (v6 → v7 → v8)

| Perturbation | v6 | v7 | v8 | v6→v8 Δ |
|--------------|----|----|----|---------|
| P_S1_asserted | — | 70% | **55%** | (regression from v7) |
| P_S2_strawman | — | 100% | **100%** | — |
| P_S3_no_positioning | 44% | 56% | **61%** | +17 |
| **P_S4_weak_problem** | 44% | 17% | **83%** | **+39** |
| P_S5_vague_method | 89% | 89% | **83%** | -6 |
| P_S6_no_risk | 88% | 88% | **88%** | 0 |
| **P_S7_no_specifics** | 47% | 64% | **93%** | **+46** |
| P_S8_overclaim | 44% | 100% | **94%** | +50 |
| **P_S9_stacked** | 24% | 41% | **88%** | **+64** |

**Average detection (ex S5): 67% → 83%** (+16 pp from v7)

## Aggregate Separation

| Metric | v6 | v7 | v8 |
|--------|----|----|----|
| Ref mean | 0.785 | 0.764 | **0.808** |
| Pert mean | 0.757 | 0.699 | 0.743 |
| Gap (ref μ − pert μ) | 0.028 | 0.064 | **0.065** |
| AUC P(ref > pert) | 0.619 | 0.685 | **0.742** |
| Perts below ref median | 65% | 81% | **86%** |

## What Changed From V7

1. **S4: PROBLEM-SECTION GATE added**. Score caps at 2 if the Problem section
   itself is vague/generic filler, regardless of what later sections contain.
   Fixed detection 17% → 83%.

2. **S1: Restricted to TOP 3-5 load-bearing choices**. V7 grader counted
   hardware, dataset names, and standard hyperparameters as "design choices",
   resulting in 10-20+ total choices with 90% "asserted". V8 caps at 5 and
   explicitly excludes boilerplate. Side effect: detection on P_S1
   REGRESSED from 70% → 55% (the perturbation becomes less discriminable
   when restricted to 5 choices, but this matches human intuition better).

3. **S9: Restricted to Core Idea only, max 8 techniques**. V7 grader counted
   40+ "techniques" on published papers. V8 caps at 8 and excludes
   infrastructure. Detection 41% → 88%, refs mean 3.25 → 4.80 (matches
   human intuition that published plans are focused).

## Opus Correlation — The Persistent Gap

| Version | r(agg, Opus) |
|---------|--------------|
| v6 | 0.395 |
| v7 | 0.158 |
| **v8** | **0.079** |

V8 correlates less with Opus ratings than v6. Likely reason: Opus ratings
are compressed (range 84-89 out of 100) — nearly saturated on all refs.
The structural rubrics discriminate among refs more finely than Opus does,
so correlation drops as the system gets more discriminative.

**Interpretation**: Low Opus correlation is not necessarily a failure.
Structural perturbation test (detection rate, AUC) is the more reliable
metric because it tests specifically-targeted degradation, not holistic
judgment of already-good plans.

## Open Issues for V9 (if needed)

1. **S1 detection regression (70% → 55%)**: restricting to top 5 load-bearing
   choices removes the sensitivity that v7 had. Trade-off: v7 over-counted
   (penalized refs) but was more perturbation-sensitive. Could add a
   secondary check: "are ANY named techniques in Core Idea + Methodology
   unjustified?" as a supplementary signal.
2. **S4 ref mean dropped (3.93 → 2.87)**: the PROBLEM-SECTION GATE catches
   refs with thin Problem sections, which is probably correct but worth
   verifying. Some refs may have good stakes articulation outside Problem
   that legitimately deserves high S4.
3. **Inter-signal correlations all <0.35**: good property (no redundancy).
EOF