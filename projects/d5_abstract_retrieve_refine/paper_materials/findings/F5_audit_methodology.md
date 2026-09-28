# F5 — Audit ensemble methodology: per-plan isolated subagents + balanced batches + anonymization

## Headline

For cross-baseline / cross-iter quality comparisons in long-form generation, **scoring
each plan in its own Opus subagent (no batched scoring)**, **anonymizing IDs**, and
**balancing batches across the comparison axis** is essential. Batched scoring with
visible IDs anchors plans to comparison neighbors, reducing variance and biasing toward
"all 8 plans hit the same anchor" artifacts (see F6).

## Methodology specification

### Step 1 — Build balanced batches

Given baselines / iterations to compare (e.g. 7 baselines or 4 early iters):
1. Sample 8 plans per group (baseline or iter)
2. Build N batches such that EACH batch contains **exactly 1 plan from EACH group**
3. Shuffle within each batch
4. Anonymize plan IDs (`batch3_plan05` instead of `mu_v4_iter_4_eval_2`)
5. Persist `decode_map.json` separately for post-hoc analysis

### Step 2 — Dispatch parallel subagents

Spawn N parallel subagents (one per batch). Each subagent receives:
- The 8 anonymized plans for its batch (and ONLY those)
- The 9-dim audit rubric (5 universal + 4 subfield-specific, 1-5 scale)
- The research goal

The subagents are independent Opus 4.7 sessions with no shared state.

### Step 3 — Per-plan isolation INSIDE each batch

This is the key innovation versus naive batched scoring. Each subagent should NOT score
all 8 plans in a single chain-of-thought. Instead, each subagent fans out 8 sub-Tasks
(one per plan), each receiving ONLY one plan + the goal + the rubric.

Why: if one subagent reads all 8 plans and scores them sequentially, plan #5's score
gets anchored by the (visible) scores given to plans #1-4. This reduces variance
artificially and creates within-batch correlations that mask real per-plan quality
differences.

### Step 4 — Aggregation

Decode anonymized IDs back to (baseline, iter) pairs. Compute per-group means + std
+ distinct-totals count. Cross-batch within the same group should correlate
(real signal); cross-batch within different groups should NOT correlate strongly
(no batch contamination).

## Empirical validation

In Phase 2F audit_v3 isolated (`runs/2026_04_26_phase2F_audit_isolated/`):
- 8 batches × 7 baselines = 56 plans, anonymized + balanced
- 8 parallel Opus subagents
- Distinct-totals counts per baseline: 4-7/8 (high variance preserved)
- Within-baseline std: 1.83 (μ) to 4.31 (δ)
- Cross-baseline mean separation: σ 25.25, δ 25.12, α 25.00 (all within 0.25 of each
  other; not artificially separated)

Without isolation (as in earlier μ-v2 / μ-v3 / μ-v4 daemon 4-dim audit):
- Distinct-totals counts often **1/8** (all plans tied at the rubric anchor)
- Within-baseline variance collapses to near-zero
- Real per-plan / per-iter differences invisible

## Comparison: 4-dim batched daemon audit vs 9-dim isolated

| Methodology | Per-iter μ-v4 audit (iter 0-4) |
|---|---|
| 4-dim batched daemon | 7.00 / 7.00 / 7.00 / 7.00 / 7.00 (anchor-stuck, all plans tied) |
| 9-dim isolated | 24.00 / 24.75 / 24.88 / 27.25 / 28.00 (clear monotonic ascent visible) |

Both audits are calling the same Opus model. The difference is purely in the prompting
methodology: per-plan isolation + 9-dim rubric resolves the trajectory; batched 4-dim
does not.

## Implementation reference

`src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py` is the reference
implementation. Key code paths:
- Balanced batch construction with cross-group shuffling
- Per-plan Task subagent dispatch (one Task per plan, not one Task per batch)
- Decode map persistence for post-hoc analysis
- Aggregation with std + distinct-totals validation

## What this means for the paper

- **The isolated audit methodology should be a paper contribution** — it's reusable
  across long-form generation evaluation, not specific to D5
- **Caveat any single-Opus batched scoring** in the paper (and in our own 4-dim
  daemon audit results): it's a coarse signal usable for trajectory monitoring but
  not for cross-baseline ranking
- **Cost**: isolated audit costs ~8x more Opus calls than batched scoring (one Task
  per plan instead of one Task per batch), but at $0.005-0.01 per 8K-token call this
  is ~$0.20-0.40 per N=8 audit — affordable for paper-grade conclusions

## Data pointers

- Isolated audit reference summary:
  `runs/2026_04_26_phase2F_audit_isolated/audit_v3_isolated_summary.md`
- Implementation: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py`
- Methodology snippet (paper-ready): `paper_materials/methodology/M4_audit_isolated_methodology.md`
- Rubric: `paper_materials/methodology/M2_audit_rubric_v3.md`
