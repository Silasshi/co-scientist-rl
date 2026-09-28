# Opus 4.7 Depth Audit: D5 Phase 0.6 Baselines (delta vs epsilon)

**Date**: 2026-04-23
**Auditor**: Claude Opus 4.7 (1M context)
**Rubric**: D3 canonical (4 dims x 1-5, max 20) — Math Formalism, Algorithmic Novelty, Implementation Realism, Empirical Rigor
**Calibration anchor**: Reference plan (J_beta entropic objective + MAX-PUCT) = ~17-18/20

## Conditions

- **delta (n=8)**: frozen Qwen3-30B-A3B + reference plan as in-context few-shot
- **epsilon (n=8)**: frozen Qwen3-235B-A22B-Instruct-2507 + reference plan as in-context few-shot
- **Comparison points**: smoke v3-rerun A (v3 abstraction, oracle no formula hints) = 9.25/20; smoke v2-rerun A (v2 abstraction, oracle + formula hints) = 11.25/20

## Headline Numbers

| Condition | Math | Novelty | Realism | Rigor | **Total** | n |
|---|---|---|---|---|---|---|
| delta (30B + ref-in-ctx) | 1.63 | 1.50 | 2.50 | 1.13 | **6.75/20** | 8 |
| epsilon (235B + ref-in-ctx) | 3.00 | 2.88 | 3.75 | 2.50 | **12.13/20** | 8 |
| smoke v3-rerun A | — | — | — | — | 9.25/20 | (prior) |
| smoke v2-rerun A | — | — | — | — | 11.25/20 | (prior) |

## Critical Findings

### 1. delta UNDERPERFORMS smoke v3-rerun A

delta total_mean **6.75** vs smoke v3-rerun A **9.25** — a 2.5 point gap. Reference-in-context with 30B is **worse** than abstraction-based pipeline. The v3 abstraction pipeline contributes value beyond literal exposure to the reference plan.

**Implication**: D5 Phase 1+ rationale survives. There is real headroom (2.5 points) above the trivial reference-in-context floor for an abstraction-retrieve-refine pipeline to add depth on 30B.

### 2. epsilon EXCEEDS smoke v2-rerun A

epsilon total_mean **12.13** vs smoke v2-rerun A **11.25** — a 0.88 point lead. 235B with reference-in-context already matches/exceeds the formula-hint-augmented abstraction pipeline.

**Implication**: Upper bound threat. If the D5 claim is "30B + abstraction beats 235B baseline," that bar is now 12.13 — significantly above what 30B alone has demonstrated under any condition so far.

### 3. Copying behavior — paraphrasing, not lifting

Across all 16 plans:
- **0/16** plans copy J_beta entropic objective `log E[exp(beta R)]`
- **0/16** plans copy MAX-PUCT with the AlphaZero one-character framing
- **0/16** plans reproduce reference numbers (Erdos 0.380924, AHC039 566,997, MAGIC 0.64)
- **All 16** paraphrase high-level paradigm: memory buffer + reward-weighted update + state reuse

The strongest copier is **plan 14 (epsilon_6)** at 15/20 — it adopts MAX in the descendant potential function and uses batch-wise lineage blocking, mirroring the reference's MAX-PUCT spirit. But even this plan invents its own potential-function formulation rather than lifting J_beta verbatim.

### 4. 30B vs 235B gap = 5.4 points

The 30B->235B jump under reference-in-context produces a 5.4 point gap (6.75 -> 12.13). This is **smaller** than the user's prior +10-14 Opus point gap on free-generation (no in-context reference). Conclusion: 30B benefits modestly from in-context reference but cannot match 235B-tier structural sophistication.

## Per-Plan Highlights

**delta scores (5/20 to 8/20)** — all in the "weak" range:

- **delta_3 (5/20)**: weakest — no formulas at all, vague compute budget inconsistencies (200 iters of 64 rollouts in 10K total tokens)
- **delta_2 (6/20)**: meta-learner narrative, zero formulas in methodology
- **delta_5 (6/20)**: no formulas, generic memory + LoRA
- **delta_0,1,6 (7/20)**: one standard formula each (reward-weighted PG / KL+UCB / normalized reward)
- **delta_4,7 (8/20)**: best delta plans — discounted PG / sigmoid-weighted top-10%

**epsilon scores (10/20 to 15/20)** — uniformly stronger:

- **epsilon_6 (15/20)**: STRONGEST — backward-induction potential `Phi(s) = R(s) + 0.95 max_descendant Phi(s')` with lineage blocking, top-k sparsified PG with `w_r = exp(lambda r)`, specific perovskite numbers
- **epsilon_0 (13/20)**: top-k censored PG `w(r) = exp(eta(r-kappa)) for r >= kappa, 0 otherwise`, FoldX targets with concrete deltas
- **epsilon_5 (13/20)**: rank-weighted PG + self-critique reflections in context (novel mechanism), AlphaEvolve baseline cited
- **epsilon_2,4 (12/20)**: off-policy with Retrace / quantile-filtered with semantic retrieval
- **epsilon_3,7 (11/20)**: curiosity-based / max-reward-anchored PG
- **epsilon_1 (10/20)**: weakest epsilon — standard PG with baseline + 70th percentile filter

## D5 Phase 1+ Investment Verdict (mixed)

**Pro-investment**: delta (6.75) underperforms smoke v3-rerun A (9.25). The abstraction pipeline genuinely adds value at 30B scale that naive reference-in-context cannot replicate. There is 2.5 points of headroom for "30B + pipeline > 30B + naive reference."

**Anti-investment**: epsilon (12.13) already exceeds smoke v2 + formula hints (11.25). 235B + reference-in-context is a cheap baseline that already competes with the entire abstraction infrastructure on 30B.

**Cleanest NeurIPS framing**:
- Primary claim: "30B + abstraction-pipeline > 30B baselines including reference-in-context (delta = 6.75)"
- Position 235B + reference-in-context (epsilon = 12.13) as an **oracle upper bound** rather than a direct competitor.
- The gap to close is 6.75 -> ~10+ on 30B; surpassing 12.13 with 30B-only would be a much stronger contribution but is not on the current trajectory.

## Files

- Raw scores: `opus_depth_audit_d3canonical.json`
- Plans: `buffer.jsonl` (16 entries, condition delta/epsilon)
- Reference: `dataset/reference_solution.txt`
- Goal: `dataset/research_goal.txt`
- Rubric source: `src/co_scientist/ttt_discover/opus_eval_agent.py` lines 30-77
