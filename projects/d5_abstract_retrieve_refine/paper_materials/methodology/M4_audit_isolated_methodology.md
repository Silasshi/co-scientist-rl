# M4 — Isolated audit methodology (per-plan + balanced batches + anonymization)

*Implementation reference: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py`*

## Goal

Produce per-plan quality scores that are NOT contaminated by other plans in the same
batch, suitable for cross-baseline / cross-iter ranking.

## Three-step protocol

### Step 1 — Build balanced cross-axis batches

Given M groups (e.g. 7 baselines, or 4 early iters), each with 8 plans:

```python
batches = [[] for _ in range(N_BATCHES)]  # N_BATCHES = 8 typically
for group in groups:
    for plan, batch_idx in zip(group_plans, batch_assignment):
        batches[batch_idx].append(plan)
# Now each batch has exactly 1 plan per group
```

Persist `decode_map.json`: anonymized_id → (group, true_plan_id).

### Step 2 — Anonymize and shuffle

Within each batch:
- Replace plan IDs with `batch{N}_plan{NN}` format
- Shuffle plan order (deterministic seed for reproducibility)
- Remove any group / iter information from plan text headers

### Step 3 — Dispatch parallel subagents with per-plan isolation

Spawn N parallel Opus subagents (one per batch). CRITICAL: each subagent's prompt
should fan out 8 sub-Tasks (one per plan), not score all 8 plans in a single chain-of-thought.

Each sub-Task receives:
- ONLY one plan's text + the goal
- The 9-dim rubric
- An anti-templating instruction ("two plans should never receive identical 9-dim score
  vectors unless content is genuinely interchangeable")
- A required `key_differentiator` field (1 sentence per plan, forces independent reasoning)

### Step 4 — Aggregate

Decode anonymized IDs back. Per-group aggregates: mean, std, min, max, distinct-totals
count. The distinct-totals count is the **diagnostic** for whether scoring was actually
isolated:
- **Distinct-totals ≥ 6/8 per group** → genuine per-plan isolation
- **Distinct-totals = 1/8** (all tied at anchor) → scoring collapsed to anchor; rerun
  with stricter isolation

## Cost

For N=56 plans (7 baselines × 8 plans): ~$0.20-0.40 in Opus token-equivalent
(via subagent file-bus, no OpenRouter). ~10-15 min wall clock with 8 parallel subagents.

## Comparison to alternatives

| Method | Cost | Variance preservation | Use case |
|---|---|---|---|
| Single Opus call, all N plans, sequential scoring | $0.05 | bad (within-batch anchoring) | quick sanity check only |
| Single Opus call per batch, batch contains M plans | $0.10 × N/M | medium (anchoring within batch) | trajectory monitor (training loop) |
| **Isolated subagent per plan, balanced batches** | **$0.40 × N/8** | **good** | **paper-grade ranking** |

## Verification heuristics

1. **Distinct-totals**: ≥75% of plans have unique 9-dim totals within each group
2. **Distinct vectors**: 100% of plans have unique 9-dim score vectors within each group
3. **Cross-batch consistency**: same group's mean across batches should converge
   (compute std-of-batch-means; should be < within-batch std)
4. **Anchor-stickiness check**: no group should have ALL plans tied at the same total

## Why this matters

In our work, the 4-dim D3-canonical batched audit gave all μ-v4 iter 0-4 EVAL plans
the same score (7.00, anchor-stuck) — the trajectory was invisible. The same plans
under 9-dim isolated audit revealed the full 24.00 → 28.00 monotonic ascent and the
+2.75-over-σ peak. The methodology choice changed the result from "training did
nothing" to "training succeeded".

This is documented in F6 (4-dim anchor saturation) and F2 (μ-v4 iter-4 peak).

## Cross-validation discipline (added 2026-04-27)

Absolute-grading methods can have surface-form bias even after isolation +
anonymization + 9-dim decomposition. Phase 0.6 saw audit_v1 say "μ-v2 > δ by
+2.13" while pairwise said "δ > μ-v2 by 8-0" — a complete inversion driven by
template artifacts that absolute scoring rewards but pairwise judges (forced to
discriminate one plan against another) penalize.

**Discipline going forward**: Any headline result from audit_v3 ISOLATED on a
*close-cluster* comparison (gap < ~3 points, where surface bias dominates true
signal) MUST be cross-checked by Opus pairwise tournament before being claimed
as a paper result.

- Reusable runner: `src/co_scientist/d5_abstract_retrieve_refine/mu_v4_pairwise_v1.py`
- File-bus client: `src/co_scientist/shared/opus_pairwise_subagent.py`
- Cost per matchup: ~$4 (8 pairs × 1 Opus call/pair)
- Recommended n: 8 pairs minimum for FLAG-vs-PASS distinction; 16 for tight CI
- Decoded via position_swapped flag in `matchups_meta.json` (subagents see
  anonymized A/B; analyze_phase decodes to baseline labels)

The μ-v4 paper-claim headline (+2.75 over σ) was cross-checked this way and
PASSED at 20/24 = 83.3% pairwise win rate (vs σ 6-2, vs δ 7-1, vs α 7-1).
Documented in F2 § "Cross-validation: pairwise corroboration" and E6 experiment
file.

**Wide-gap comparisons** (e.g., ε vs σ at +8.37, β vs μ-v4 at -5.25) do NOT
require pairwise cross-check — absolute-audit gap is too large for surface bias
to plausibly explain.
