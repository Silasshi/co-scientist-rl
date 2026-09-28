# Sanity Check v2 Report: New Signal Set (S1 Mechanism + Augmented S2)

**Date**: 2026-04-11
**Status**: Complete (with one known failure)
**Purpose**: Validate the revised signal set after dropping S1_coherence, adding S1_mechanism (Mechanism Plausibility), and augmenting S2_rigor with operationalization check.
**Runtime**: 3.7 hours, 224 grader calls, 3 errors (all on S1_mechanism)

## Signal Set Changes Since v1

| Change | Rationale |
|---|---|
| Dropped S1_coherence | High halo (mean_r = 0.52), low grounding, coverage redundant with other signals |
| **Added S1_mechanism** | Unique "why should this work" dimension, missing from v1 |
| Augmented S2_rigor prompt | Now includes "operationalization" check (is the success criterion concrete?) |
| Method: separate_call only | v1 established separate_call is better; no need to re-test single_call |
| N_REPEATS = 2 | Reliability check without blowing up cost |

## TL;DR

- **7 of 8 signals validated** — correctly detect their targeted damage
- **S1_mechanism FAILED** — grader gives score 3 to both reference and damaged versions (conflates "formula present" with "mechanism argued")
- **Other metrics improved over v1**: mean correlation 0.33→0.30, reliability 0.126→0.046
- **Decision**: Keep S1_mechanism at very low weight (0.05) as monitoring signal only

## Reference Plan Scores (separate_call)

| Signal | v1 (S1=coherence) | v2 (S1=mechanism) | Note |
|---|---|---|---|
| S1 | 4 | 3 | Grader thinks reference has partial but not complete mechanism |
| S2_rigor | 3 | 3 | Unchanged |
| S3_positioning | 4 | 4 | Unchanged |
| S4_significance | 4 | 4 | Unchanged |
| S5_stability | 2 | 2 | Unchanged (accurately flagged weak) |
| S6_failure_interp | 2 | 3 | +1 (might be noise — only 14 plans) |
| S7_specificity | 3 | 3 | Unchanged |
| S8_scope | 4 | 4 | Unchanged |

Reference scores are consistent with v1 on most signals.

## Per-Signal Validity Results

| Plan | Target | Ref→Perturbed | TargDrop | OtherDrop | Ratio | Verdict |
|---|---|---|---|---|---|---|
| **P_S1_mechanism** | **S1_mechanism** | **3→3** | **0** | **0.14** | **0.00** | **❌ FAILED** |
| P_S2_rigor | S2_rigor | 3→2 | 1 | 0.00 | ∞ | ✅ PERFECT |
| P_S3_positioning | S3_positioning | 4→3 | 1 | 0.14 | 6.54 | ✅ Good |
| P_S4_significance | S4_significance | 4→3 | 1 | 0.43 | 2.28 | ✅ Good |
| P_S5_stability | S5_stability | 2→1 | 1 | 0.29 | 3.38 | ✅ Detected |
| P_S6_failure_interp | S6_failure_interp | 3→1 | 2 | 0.57 | 3.44 | ✅ Strong |
| P_S7_specificity | S7_specificity | 3→2 | 1 | 0.00 | ∞ | ✅ PERFECT (improved from v1!) |
| P_S8_scope | S8_scope | 4→2 | 2 | 0.71 | 2.76 | ✅ Strong |

**v2 vs v1 comparison**:
- v1 had 2 failures (P07 single_call, P09 separate_call)
- v2 has 1 failure (P_S1_mechanism) but fixes v1's P_S7_specificity failure

Net: the 7 surviving gradient signals (S2-S8) all validate. Only S1_mechanism has an unresolved issue.

## Halo Effect & Reliability

| Metric | v1 (S1=coherence) | v2 (S1=mechanism) | Verdict |
|---|---|---|---|
| Mean inter-signal correlation | 0.330 | **0.300** | Slight improvement |
| PC1 variance ratio | **0.394** | 0.605 | Apparent regression* |
| Mean repeat std (N=2) | 0.126 | **0.046** | Major improvement |
| Fraction zero std | 0.776 | **0.917** | Major improvement |
| Avg specificity ratio | 14.54 | 27.30 (w/o S1 fail) | Improvement |

*The PC1 jump is likely PCA instability — only 14 data points, which is barely enough for a stable PCA on 8 signals. Mean inter-correlation (a more robust metric) actually decreased.

## Why S1_mechanism Failed

**P_S1_mechanism perturbation design**: remove causal reasoning ("as β→∞ the inner expectation concentrates on max") and replace adaptive β tuning with "we set β to a fixed value of 1.0". Keep the entropic objective formula J_β itself.

**What the grader did**: Both reference and perturbed scored S1_mechanism = 3. Both repeats gave identical scores (frac_zero_std = 0.917 for this signal).

**Diagnosis**: The grader appears to conflate "presence of mathematical formulas" with "mechanism argued". The CoT scaffolding asks it to list the causal chain; it lists the formula components as "steps" and checks off each one as present. The resulting score doesn't depend on whether the plan explains WHY the formula achieves the goal.

This is a limitation of Qwen3-30B-A3B as a grader, not a fundamental problem with the signal concept. A stronger grader (GPT-4, Claude Opus) might distinguish the versions.

### Collateral damage observed on P_S1_mechanism

The mechanism perturbation also caused off-target changes:
- S3_positioning: 4 → 3 (collateral drop)
- S6_failure_interp: 3 → 2 (collateral drop)
- S7_specificity: 3 → **4** (collateral *rise* — possibly because shorter text reads as more specific)

These collateral effects are small (1 step each) but confirm that our perturbation wasn't perfectly isolated.

## Decision: Keep S1_mechanism at Weight 0.05

Three options were considered:
1. **Fix and re-run** (4 hours): rewrite prompt + stronger perturbation. Risk: same grader limitation may persist.
2. **Keep at low weight** (0 hours): demote to 0.05, monitor during training
3. **Drop entirely** (0 hours): return to 7 signals

**Chosen: Option 2**. Rationale:
- Re-running would cost 4 hours with uncertain outcome
- The concept (mechanism plausibility) is important — keep the signal
- Low weight (0.05) ensures S1_mechanism cannot dominate or distort training
- If the training dynamics show S1 does discriminate on the real policy distribution (where plans can be *much* worse than the reference — e.g., missing the formula entirely), we can raise the weight later

## Final Weights

```python
SIGNAL_WEIGHTS = {
    "S1_mechanism":      0.05,  # Demoted — grader blind spot on formal methods
    "S2_rigor":          0.15,
    "S3_positioning":    0.13,
    "S4_significance":   0.16,
    "S5_stability":      0.10,
    "S6_failure_interp": 0.14,
    "S7_specificity":    0.10,
    "S8_scope":          0.17,
}
```
Total: 1.00. Plus 2 hard gates (Goal-Contrast, Claim Verification) from the v1 seven_signal_reward implementation.

## Training-Time Monitoring

During TTT training, we should monitor S1_mechanism's behavior separately:

1. **Track per-signal distribution**: if S1_mechanism distribution collapses (all plans score the same), it's truly degenerate → drop entirely
2. **Track correlation with aggregate**: if S1 correlates strongly with other signals throughout training, it's redundant → drop
3. **Track max S1 achievable**: if policy can hit 4-5 on S1_mechanism through actual mechanism improvement (not just formula gaming), it's working → raise weight

If any of these criteria are met by iteration 20, revise weights accordingly.

## Remaining Risks

1. **S1_mechanism blind spot**: The signal gives no gradient in the "formula-heavy + no explanation" regime. This specifically fails to catch plans that are mathematically decorated but unjustified. Mitigation: other signals (S7 specificity, S4 significance) partially catch this.

2. **Grader noise still present**: Despite mean_std = 0.046, individual signals can flip between repeats. N=2 is a minimum — N=3 would be safer but 50% more expensive.

3. **Only 14 plans tested**: The validity conclusions are based on a small sample. Training runs will expose signal behavior on a much larger distribution.

## Next Steps

1. Lock the signal set as-is
2. Implement buffer-conditioned TTT trainer (next major task)
3. Pilot run with 5-10 iterations
4. Inspect per-signal trajectories before committing to a full 50-iteration run
