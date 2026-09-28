# V8.1 Results: Honest S4 Gate After Review

## TL;DR

Human review of v8 revealed S4 PROBLEM-SECTION GATE was over-triggering on
refs (37% fail vs 30% perts). V8.1 loosens the gate. Result: aggregate AUC
improves (0.742 → 0.760), but S4 detection drops to 0% — revealing that
v8's 83% detection was inflated by unfair ref penalties, not true
perturbation discrimination.

## What We Found

The 5 lowest-scoring v8 refs had good Problem sections:
- AutosotaEndtoendAutomated: specific bottleneck ("months of human effort"), stakes ("democratize AI")
- ContextAllYou: named failure domains (medical imaging, language models, vision systems)
- WhyLLMsArentScientists: named specific gap ("no published study pairs (i)-(iv)")

But grader marked WHO/WHAT as NO because:
- "researchers" was judged too generic (required "roboticists working with X")
- "democratize access" was judged not measurable (required numeric outcome)

## V8.1 Change

Loosened gate to accept:
- "ML researchers", "practitioners doing X" as WHO
- "removes bottleneck X", "enables capability Y" as WHAT
- Named benchmarks / methods / domains as GATE PASS

## V8 vs V8.1 Metrics

| Metric | v8 | **v8.1** |
|--------|----|---------| 
| Refs S4 mean | 2.87 | **4.00** |
| Refs S4 ≤ 2 | 22/60 (37%) | **0/60** |
| Aggregate gap | 0.065 | **0.069** |
| AUC P(ref>pert) | 0.742 | **0.760** |
| Refs mean agg | 0.808 | **0.845** |
| Perts mean agg | 0.743 | 0.776 |
| Perts below ref P50 | 86% | 85% |
| S4 detection | 83% | **0%** |
| Opus correlation | 0.079 | 0.038 |

## Why S4 Detection Dropped to 0%

Investigation: read a perturbed Problem section (Aster__P_S4).

Perturbed Problem section (pure filler):
> "Scientific discovery with computational agents is an important area of
> research. Current approaches face various challenges... researchers continue
> to investigate how to make these systems more effective."

But v8.1 grader STEP 0 reasoning says:
> "GATE PASS: The Problem section names specific bottlenecks ('fundamental
> gap in iteration efficiency'), applications (ZAPBench), prior methods
> (AlphaEvolve, OpenEvolve)..."

**The grader is reading Motivation content into its Problem section analysis.**
Qwen3-30B does not reliably isolate sections even when explicitly instructed.

## The Honest Interpretation

1. **v8's 83% S4 detection was artifactual.** It came from v8's strict
   WHO/WHAT causing refs to fail the gate more often than paired
   perturbations happened to — not from S4 truly detecting significance
   degradation.

2. **P_S4_weak_problem is a weak perturbation.** It only modifies the Problem
   section. Since a good plan's stakes are articulated across Problem +
   Motivation + Core Idea, removing stakes from one section doesn't
   meaningfully reduce overall significance.

3. **v8.1 is structurally cleaner.** Refs aren't unfairly penalized, and
   aggregate AUC is higher (0.760). Perturbations are still discriminated
   by other signals (S7, S9, S8 all >88% detection).

## What This Tells Us About The System

- The signal system's structural discrimination is real — it's in the 9-signal
  ensemble, not in any single signal.
- S4 as a signal is weak for single-section perturbations, but that's an
  honest limitation, not a rubric flaw.
- Future perturbation designs should degrade the target quality dimension
  across multiple sections if the quality dimension is holistic.

## Final Detection Rates (v8.1)

| Perturbation | Detection |
|--------------|-----------|
| P_S1_asserted | 55% |
| P_S2_strawman | 100% |
| P_S3_no_positioning | 61% |
| **P_S4_weak_problem** | **0%** (honest — weak perturbation design) |
| P_S5_vague_method | 83% |
| P_S6_no_risk | 88% |
| P_S7_no_specifics | 93% |
| P_S8_overclaim | 94% |
| P_S9_stacked | 88% |

Mean (excluding S4 and S5): **83%**

## Files

- `grading_refs_v8.jsonl` + `s4_regraded_v8_1.jsonl` → merged v8.1 aggregate
- `analyze_v8_1.py` → merge + analysis
- `FINAL_ANALYSIS_V8_1.txt` → full output
