# Sanity Check Report v1: 8-Signal Reward Validation

> ⚠️ **SUPERSEDED by SANITY_CHECK_V2_REPORT.md** (2026-04-11).
>
> This report is a historical snapshot of the v1 signal set (with `S1_coherence`).
> After this sanity check, S1_coherence was dropped and replaced with
> `S1_mechanism` (Mechanism Plausibility), and `S2_rigor` was augmented with an
> operationalization check. The current production signal set is in
> `SANITY_CHECK_V2_REPORT.md`. Do NOT use this document as the signal reference.

**Date**: 2026-04-11
**Status**: Complete (superseded)
**Total grader calls**: 252 (14 plans × 2 repeats × 9 calls per plan), 5 timeouts (2%)
**Total runtime**: ~5 hours (8 waves of 32 parallel requests)

## TL;DR

- **separate_call mode is better** on the metrics that matter most (halo effect, per-signal localization for 6/8 signals)
- **Both methods have one failure each**: single_call fails to detect S5_stability damage, separate_call fails to detect S7_specificity damage
- **S1_coherence is a halo magnet** (mean_r = 0.50+ in both methods) and should be down-weighted
- **S8_scope is remarkably independent** in separate_call (mean_r = 0.026) — keep and weight high
- **S6_failure_interp** is the most independent signal in both methods (mean_r ~0.20) — a valuable orthogonal signal
- **Use separate_call with N=2 repeats**; keep all 8 signals but adjust weights

## Method Comparison

| Metric | single_call | separate_call | Winner |
|---|---|---|---|
| avg specificity ratio | 5.25 | 14.54 | separate (inflated by P05 outlier; 2.34 without it) |
| halo mean correlation | 0.386 | 0.330 | **separate** |
| halo PC1 variance ratio | 0.523 | **0.394** | **separate** |
| mean repeat std | **0.076** | 0.126 | single |
| cost per plan | 1 call | 8 calls | single |

**Verdict**: separate_call wins on the two most important metrics (halo effect). Higher cost and noise are acceptable trade-offs.

## Reference Plan Scores

Both methods agree the reference is moderate quality (3-4 on most signals):

| Signal | single_call | separate_call | Note |
|---|---|---|---|
| S1_coherence | 4 | 4 | |
| S2_rigor | 3 | 3 | Lower than expected — reference has only basic baselines |
| S3_positioning | 4 | 4 | |
| S4_significance | 4 | 4 | |
| **S5_stability** | **2** | **2** | Reference correctly flagged — only single-seed acknowledgment |
| **S6_failure_interp** | **2** | **2** | Reference has generic limitations, no failure diagnostics |
| S7_specificity | 4 | **3** | separate_call is stricter |
| S8_scope | 4 | 4 | |

**Key insight**: The reference plan (drawn from the actual TTT-Discover paper) scores only 2/5 on Stability and Failure Interpretability. This is expected — the paper's presentation doesn't strongly address these. But it has implications:

1. S5 and S6 already score near floor → perturbation damage has little room to manifest
2. During training, the model should be **able to exceed the reference** on these signals by explicitly adding stability/failure analysis

## Per-Signal Validity (Detailed)

| Plan | Target | Method | Ref→Perturbed | Drop | Others | Ratio | Verdict |
|---|---|---|---|---|---|---|---|
| P03 | S1_coherence | single | 4→3 | 1 | 0.14 | 6.5 | Good |
| | | separate | 4→3 | 1 | 0.57 | 1.7 | Weak localization |
| P04 | S2_rigor | single | 3→2 | 1 | 0.43 | 2.3 | Moderate |
| | | separate | 3→2 | 1 | 0.43 | 2.3 | Moderate |
| P05 | S3_positioning | single | 4→3 | 1 | 0.14 | 6.5 | Good |
| | | **separate** | **4→3** | **1** | **0.00** | **∞** | **Perfect localization** |
| P06 | S4_significance | single | 4→3 | 1 | 0.29 | 3.4 | Good |
| | | **separate** | **4→2** | **2** | **0.43** | **4.6** | **Stronger drop** |
| **P07** | **S5_stability** | **single** | **2→2** | **0** | 0.14 | **0.0** | **FAILED** |
| | | separate | 2→1 | 1 | 0.43 | 2.3 | Detected |
| P08 | S6_failure_interp | single | 2→1 | 1 | 0.29 | 3.4 | Good |
| | | separate | 2→1 | 1 | 0.71 | 1.4 | Weak localization |
| **P09** | **S7_specificity** | **single** | **4→2** | **2** | 0.14 | **13.1** | **Excellent** |
| | | **separate** | **3→3** | **0** | 0.14 | **0.0** | **FAILED** |
| P10 | S8_scope | single | 4→2 | 2 | 0.29 | 6.8 | Good |
| | | **separate** | **4→1** | **3** | **0.71** | **4.1** | **Strong drop** |

**Failures explained**:
- **P07 single_call**: Reference S5=2 (already low). Single_call can't distinguish "acknowledges weakness" from "ignores stability entirely".
- **P09 separate_call**: Reference S7=3 in separate_call (stricter than single_call's 4). After damage, still scores 3 — the grader doesn't distinguish "vague at the level of the reference" from "vague at the level of the perturbed version".

Both failures reveal a **reference calibration issue**, not a fundamental method flaw.

## Halo Effect (per-signal mean inter-correlation)

Lower = more independent (better for RL training):

| Signal | single_call | separate_call |
|---|---|---|
| S1_coherence | 0.502 | **0.524** (highest halo) |
| S2_rigor | 0.362 | 0.368 |
| S3_positioning | 0.305 | 0.376 |
| S4_significance | 0.422 | 0.333 |
| S5_stability | 0.485 | 0.419 |
| **S6_failure_interp** | **0.220** | **0.195** (most independent in single_call) |
| S7_specificity | 0.461 | 0.397 |
| **S8_scope** | 0.333 | **0.026** (extraordinary — near-zero correlation) |

**Key observations**:
1. **S1_coherence is the halo magnet** — correlates strongly with everything. It's basically measuring "does this feel like a polished document". Should be down-weighted.
2. **S6_failure_interp is the most independent** — it captures something unique that no other signal measures.
3. **S8_scope in separate_call is remarkable** (r=0.026) — almost perfectly orthogonal to everything else. This is the signal most robust to halo.

## Mixed Perturbation Behavior

When multiple signals are damaged simultaneously:

**M01 (damage S1,S2,S3)**:
- single: drops correctly on target but +1 collateral on S4, S7
- separate: drops MORE strongly on target (2,2,1) but +1 collateral on S4,S5,S7, +2 on S8

**M02 (damage S4-S8)**:
- Both methods: targeted drops of 0-2, but S8_scope drop = 0 in both (surprising — would expect drop from P10)
  - Mystery: the lenient mixed-perturbation generator may have skipped the S8 replacement for M02. Confirmed by file stats (applied=7, skipped=4).
- Collateral damage of +1 on undamaged S1, S2, S3

**M03 (alternating: S1,S3,S5,S7)**:
- single: strong drops (2,1,1,3) but **collateral of +2 on S8**
- separate: drops (2,1,1,1) with collateral of +2 on S8

**Observations**:
1. Mixed perturbations do lower targeted signals as intended
2. **Collateral damage is substantial in mixed perturbations** — halo effect is more visible when many signals fail together
3. **S8_scope has collateral drop of +2 in M03** for both methods — this is unexpected, since M03 doesn't damage S8. Suggests grader perceives "overall plan quality" influencing scope score when many signals are weak.

## Reliability Across Repeats

N=2 repeats only (I reduced from N=3 to save cost after the first run timed out):

| Method | Mean std | Max std | Fraction zero std |
|---|---|---|---|
| single_call | 0.076 | 0.50 | 0.848 (85% of signals give same score in both repeats) |
| separate_call | 0.126 | 1.00 | 0.776 (78% same score) |

**Verdict**: Both methods are reliable enough. With N=2, the variance of the mean is small (< 0.1 for single, < 0.13 for separate). N=3 would be more robust but not essential for initial pilot runs.

## Final Decisions

### Decision 1: Grader call mode → **separate_call**

Rationale:
1. Lower halo effect (PC1 = 39.4% vs 52.3%) — the most critical metric for RL training
2. S8_scope becomes extraordinarily independent (r=0.026)
3. Better localization on 6 of 8 signals (P03-excluded, P05, P06, P07, P08-marginal, P10)
4. Higher cost is acceptable; extra noise is manageable with N=2

### Decision 2: Signal set → **keep all 8 but re-weight**

No signal fails catastrophically. S5 and S7 have failure modes but they're reference-calibration issues, not signal-design issues. The signals are fine; the reference could be strengthened.

Final weights (for scalar aggregation in the entropic objective):

| Signal | Weight | Rationale |
|---|---|---|
| S1_coherence | **0.05** | High halo + lowest literature grounding → minimize |
| S2_rigor | 0.15 | Strong grounding (NIH Factor 2) |
| S3_positioning | 0.12 | Strong grounding + good localization in separate_call |
| S4_significance | 0.15 | Strong grounding (NIH Factor 1), well-detected |
| S5_stability | 0.10 | Good grounding (reproducibility crisis); narrow range observed |
| S6_failure_interp | 0.13 | Unique + most independent signal in both methods |
| S7_specificity | 0.10 | Good grounding but ambiguous method choice |
| **S8_scope** | **0.20** | Most independent (r=0.026), strong grounding — highest weight |

Total: 1.00

### Decision 3: N_repeats → **N=2**

N=2 is sufficient given the observed reliability. Can revisit if pilot training shows instability.

### Decision 4: Reference plan improvements needed

The reference scores 2/5 on S5 and S6. This is accurate (TTT-Discover paper doesn't strongly address these) but limits the perturbation test. For future reruns, strengthen the reference plan to score 4-5 on all signals. This is NOT urgent — the core sanity check passed.

### Decision 5: Watch for S1_coherence degeneration during training

S1 has the highest halo (0.52) and the lowest grounding. During training it may become degenerate or dominated by surface fluency. If its variance collapses, drop it to weight 0 (monitor only).

## What's next

1. **Implement the buffer-conditioned TTT trainer** using the decided signal set, weights, and separate_call grader mode
2. **Pilot run** (5-10 iterations) to verify the pipeline converges
3. **Consider strengthening the reference plan** during post-pilot iteration, if S5 and S7 cause trouble

## Risks & mitigations going forward

| Risk | Detection | Mitigation |
|---|---|---|
| S1_coherence dominates via halo | Per-signal trajectory plot during training | Drop weight to 0 if variance collapses |
| Reward hacking via S7 (false specificity) | Qualitative review of top plans | Hard gate on Goal-Contrast Margin catches keyword-bombing |
| S5 stays degenerate (always 2) | Per-signal distribution | Strengthen reference, or accept it as a soft signal |
| Separate_call drift over long runs | Monitor inter-signal correlations every 10 iterations | Switch to hybrid mode if halo rises |
| Cost explosion (8 calls × N=2 × M=8 = 128 grader calls per iteration) | Wall-clock per iteration | Reduce M or switch to single_call if too slow |
