# Path-Locking in Reference-Grounded RL Rewards — Framing Note

**Status**: Problem statement for internal review. NOT yet validated experimentally. Drafted 2026-04-22 following discussion of Goel 2512 / RLVRR / DR Tulu.

## The Problem

Reference-grounded rubric rewards (Goel 2512, RLVRR, D1 rubric_reward, D4 grant_proposal) all share a structural flaw: the rubric extracted from a specific reference paper `P` encodes not just *what a good plan looks like* but *what `P`'s specific approach looks like*. The RL objective becomes `imitate P`, not `generate a good plan for goal G`.

**Concretely**: if paper `P` solves goal `G` via Bayesian optimization, the extracted rubric contains items like "specifies acquisition function", "addresses exploration-exploitation". A plan solving `G` via evolutionary search — equally or more valid — scores 0 on these items.

Goel 2512's human validation indirectly reveals this: **expert approval 84%** means **16% of automatically-extracted rubric items encode path-specific artifacts, not generic must-haves**. Under GRPO with 8-sample groups across 6872 goals, this 16% contamination is not random noise — it's a systematic bias the policy learns.

**Hypothesis (to test)**: The length-hacking and dead-gradient phenomena observed in D4 v8 runs are partial manifestations of path-locking. G11/G12/G13 correlate with length because the *reference proposals* in our dataset happen to be long-and-formal, not because length-and-formality are intrinsic quality.

## Proposed Quantification: Path-Locking Score (PLS)

Given goal `G`, reference `r`, rubric `R_r` extracted from `r`, and N-1 alternative references `{r'_i}` solving the same goal via different approaches:

```
PLS(R_r) = 1 - mean_i [ agreement(R_r, R_r'_i) ]

where agreement(R_a, R_b) = |R_a ∩ R_b| / |R_a ∪ R_b|
       (set intersection of rubric items after deduplication via embedding match)
```

Interpretation:
- `PLS = 0` → rubric items are invariant across references = path-free
- `PLS = 1` → rubric items are unique to `r` = fully path-locked
- Expect Goel 2512 rubric `PLS ≈ 0.6-0.8` given reports of 16% expert-rejected items; more if alternative references deviate substantively.

## Severity per Method

| Method | Expected PLS | Why |
|---|:-:|---|
| RLVRR content reward (keyword LCS) | **0.85-0.95** | keyword match is maximally surface-locked; synonyms get 0 |
| Goel 2512 goal-specific rubric | **0.55-0.75** | 16% expert rejection + path-specific residuals in "approved" items |
| D4 v8 10-signal rubric (shared across goals) | **0.20-0.40** | process-oriented signals are more path-invariant, but specific wording (e.g. "formal rigor") still encodes writing-style path |
| DR Tulu evolving buffer | **0.15-0.35** (predicted) | rubric adapts to policy's chosen path, dropping path-specific items that stop discriminating |
| Human expert judgment | **0.05-0.15** | experts recognize valid alternative paths |

## 3 Experimental Tests

### Test 1: Cross-reference rubric stability
- For each of 5 D4 goals, collect 3 alternative reference proposals solving the same goal via different approach
- Extract Goel-style rubrics from each reference independently
- Compute pairwise PLS across the 3 rubrics per goal
- Expected: PLS > 0.5 → path-locking confirmed at data level

### Test 2: Policy imitation signature
- Take MAIN_v8_C3/C4 final checkpoint plans
- Compute embedding similarity to reference proposal vs. alternative valid solutions
- If policy plans are systematically closer to reference than to alternatives (even when alternatives score higher on holistic Opus eval) → policy path-locked to reference
- Metric: `mean_sim(policy, ref) - mean_sim(policy, alt_valid) > threshold`

### Test 3: Rubric intervention
- Take policy trained on single-reference rubric → compute Opus holistic score
- Retrain same policy on consensus-rubric (items appearing in ≥2 of 3 references) → compute Opus score
- If consensus-rubric policy scores higher on Opus holistic, path-locking was suppressing quality
- This is the decisive test for "path-locking hurts generalization"

## Implications if Confirmed

If Tests 1-3 show PLS >> 0 AND path-locked policies underperform consensus-rubric policies on holistic eval:

1. **D4 v9 rubric extraction should use N-reference consensus**, not single reference per goal
2. **Hard gates (RLVRR-style) must use goal-invariant signals only** — citation authenticity (goal-independent), required section names (format-level), not keyword LCS (path-locked)
3. **DR Tulu's discriminativeness filter becomes more important** — it's the only component that *dynamically* reduces path-locking as policy diverges from reference
4. **Paper contribution reframe**: the NeurIPS 2026 paper can center on path-locking as a characterization of why reference-grounded rubric RL fails on open-ended tasks, with consensus-rubric + dynamic evolution as the proposed fix

## Relation to Existing Work

- **Goel 2512** acknowledges the 16% expert rejection but treats it as a data quality issue, not as a structural feature of the approach
- **RLVRR** uses 3 references per prompt but doesn't compute path-locking metrics
- **DR Tulu** dynamics are the only mechanism explicitly countering this, but authors don't frame the benefit this way
- **Goodhart literature** (InfoRM, Causal RM) focuses on feature-level spurious correlations, not path-level; path-locking is a higher-level structural issue

## Open Questions

1. Can PLS be computed without alternative references? (Cold start problem for goals with single reference)
2. Is there a principled way to separate "path-specific but still informative" items from "pure path artifacts"? (Expert approval 84% suggests yes — experts distinguish)
3. Does path-locking interact with length bias? (Our hypothesis: yes, length is one dimension along which reference-style leaks into rubric)

## Next Steps (if approved for investigation)

- Test 1 requires collecting 5 × 3 = 15 alternative reference proposals — ~1 day Opus task
- Test 2 needs only existing buffer + embedding computation — ~4 hours
- Test 3 requires one v9 training run with consensus rubric vs baseline — ~3 days compute

Decision point: run Test 2 first (cheapest, falsification-capable). If Test 2 shows no signature of path-locking, retract this framing. If it shows a clear signature, proceed to Test 1 and 3.
