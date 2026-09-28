# PLAN: Self-review ablation (Qwen3-30B reviewer vs Opus reviewer)

**Filed**: 2026-04-29
**Status**: To-execute (next session)
**Author**: D5 owner

## Motivation

All v8-d5sdpo + v9 (L1, L2) ablations were audited via M8-strict 1-plan/1-Opus subagent file-bus, costing ~$0.05/plan × 8 plans/iter × ~16 iter × 5 cells = **~$30** per cell + heavy rate-limit fragility (we hit Anthropic 5-hour cap once during overnight orchestration).

Three reasons to revisit the audit pipeline with **Qwen3-30B-A3B as the reviewer** (i.e., the same model used as policy):

1. **Methodological finding**: Self-review vs external-review divergence is a paper-grade comparison. Does Qwen3-30B-as-reviewer produce systematically different scores than Opus 4.7? If yes, which dimensions diverge most? Which agrees? This is a self-evaluation question that's central to TTT-discover-style work.

2. **Cost / availability**: ~$0 marginal cost (Tinker compute already paid). No rate-limit fragility. Re-runnable on demand for new ablations.

3. **Bias diagnosis**: If Opus systematically scores plans higher on U1 (math soundness) than Qwen3 does, that's evidence Opus is rewarding plans that *look* mathematical to a frontier model but a same-class reviewer (Qwen3) doesn't find them sound. This is the canonical "evaluator-policy mismatch" diagnostic.

## Hypotheses

**H_self_1 (Goodhart amplification)**: Qwen3-30B as reviewer will score Qwen3-30B-generated plans **higher** than Opus does, on all dimensions, because the reviewer recognizes its own surface patterns. **Falsification**: if Qwen3-self < Opus on average, this hypothesis is wrong and self-review is harsher (also a publishable finding).

**H_self_2 (axis-specific divergence)**: U1 (math soundness) will diverge most: Opus scores Qwen3's J_RS notation as "math present" (≥3) generously, but Qwen3-as-reviewer detects malformed derivations (eg `(1/β) log(Σexp(βr)/Σexp(βr))` reduces to log 1) and scores stricter. T2 (disentanglement) and T3 (compute) should agree most because they're checklist-like.

**H_self_3 (cell ranking preserved)**: The relative ranking of cells (σ_v8 > C+ > L2 > G+ > L1) should hold under self-review. **If cell ranking flips, the audit signal is auditor-specific not plan-quality-specific** — major paper signal that all our v8/v9 conclusions are reviewer-dependent.

## Deliverables

1. **Per-cell trajectory** for all 5 cells (σ_v8, G+, C+, L1, L2) under Qwen3-30B reviewer
2. **Per-dim divergence table**: for each dim D and cell C, compute `Δ_D = score_Opus(D, C) - score_Qwen3(D, C)`; identify systematically biased dims
3. **Spearman ρ between Opus-ranking and Qwen3-ranking** of the 5 cells (test H_self_3)
4. **Plan-level scatter**: for each (cell, iter, plan), plot `score_Opus vs score_Qwen3` to see if disagreement is uniform or concentrated on specific plans
5. **Finding doc F19**: writeup of the divergence pattern + paper claim about self-review limits

## Implementation plan

### Phase 1: Build self-review subagent dispatcher (~2 hr)

**File**: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_self_review.py`

Pattern mirror `audit_v3_isolated.py` (existing) but call Qwen3-30B-A3B via Tinker `service_client.create_sampling_client(model_path="Qwen3-30B-A3B")` instead of dispatching Claude subagent. Reuse:
- Same `AUDIT_RUBRIC_v3.md` (the rubric prompt is the only thing that matters for review)
- Same M8-strict protocol (1 plan / 1 reviewer-call, no batch anchoring)
- Same JSON schema for output partials
- Same `aggregate_audit_partials.py` aggregator

Key differences vs Opus subagent path:
- Reviewer prompt = full rubric md + plan text + JSON schema stub. Sample max 4096 tokens. Parse JSON from output.
- One Tinker call per (cell, iter, plan) → ~5 cells × 16 iter × 8 plans = 640 calls × ~30s each = ~5 hr if serialized; ~30 min if 32-parallel.
- No Anthropic API rate-limit concerns.

### Phase 2: Re-audit all 5 cells (~5-8 hr Tinker)

Run `audit_v3_self_review.py` on:
1. **σ_v8**: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/` (1 iter only, baseline)
2. **G+**: `runs/2026_04_29_mu_v8_d5sdpo_Gplus/` (13 iter)
3. **C+**: `runs/2026_04_29_mu_v8_d5sdpo_Cplus/` (12 iter)
4. **L1 v9_grounded**: `runs/2026_04_29_mu_v9_grounded_pilot/` (16 iter)
5. **L2 v9_kl_anchor**: `runs/2026_04_29_mu_v9_kl_anchor_pilot/` (16 iter)

Output: parallel `audit_responses_qwen3/iter_NNN.json` files alongside existing `audit_responses/iter_NNN.json` (Opus). Do NOT overwrite.

### Phase 3: Comparison analysis (~2 hr)

Build `analysis/compare_opus_vs_qwen3_review.py`:
- Load both review trajectories per cell
- Per-cell trajectory plot: Opus mean vs Qwen3 mean across 16 iter
- Per-dim divergence heatmap: rows = cells, cols = U1..U5/T1..T4, values = mean(Opus − Qwen3)
- Spearman ρ between cell rankings
- Plan-level scatter (640 points)
- Save to `analysis/2026_04_30_self_review_vs_opus/`

### Phase 4: Re-run grounding analysis (~30 min, $0)

Re-run the multi-axis grounding analysis (eq/cite/emp on gold set) — but this is reviewer-independent, so no new compute. Use as a "ground-truth" axis to anchor reviewer divergence: if Opus and Qwen3 disagree on cell ranking, which one tracks grounding hit-rate better?

### Phase 5: F19 finding + commit (~1 hr)

Write `paper_materials/findings/F19_self_review_vs_opus_divergence.md`. Decision rule for paper claim:
- If Spearman ρ ≥ 0.8 (rankings agree): "self-review is feasible substitute for external review at ~$0 cost"
- If 0.5 ≤ ρ < 0.8: "self-review captures coarse ordering but disagrees on close cells"
- If ρ < 0.5: "self-review fundamentally diverges; external review is necessary"

Commit per `feedback_daily_commits` rule.

## Cost estimate

| Phase | Cost | Wall |
|---|---:|---:|
| 1. Build self-review dispatcher | $0 | 2 hr |
| 2. Re-audit 5 cells (640 Qwen3 calls) | ~$0 (Tinker compute) | 30 min - 5 hr (depending on parallelism) |
| 3. Comparison analysis | $0 | 2 hr |
| 4. Grounding re-run | $0 | 30 min |
| 5. F19 + commit | $0 | 1 hr |
| **Total** | **$0 marginal** | **6-10 hr** |

## Files to create

- `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_self_review.py` (~250 LOC, mirror of audit_v3_isolated.py)
- `src/co_scientist/d5_abstract_retrieve_refine/analysis/compare_opus_vs_qwen3_review.py` (~150 LOC)
- `projects/d5_abstract_retrieve_refine/runs/{cell}/audit_responses_qwen3/iter_???.json` (5 cells × ~12-16 iter each)
- `projects/d5_abstract_retrieve_refine/analysis/2026_04_30_self_review_vs_opus/` (figures + tables)
- `projects/d5_abstract_retrieve_refine/paper_materials/findings/F19_self_review_vs_opus_divergence.md`

## Verification gates

1. **Smoke**: run self-review on σ_v8 iter 0 (1 plan), confirm JSON parses + scores in [1,5] range + matches schema
2. **Sanity check**: run self-review on the same plan twice with `temperature=0`, expect identical output (deterministic)
3. **Calibration check**: run self-review on `iter_ref_plan_0.partial.json` (the reference plan, ~28/45 under Opus). Expect Qwen3 score within ±5pt of Opus reference. If Qwen3 scores reference plan ≤15/45, the prompt is broken.

## Risks

1. **Qwen3 instruction-following**: Qwen3-30B-A3B may not produce valid JSON consistently. Mitigation: use structured generation if Tinker supports it, OR add JSON repair pass via simple regex/Python parsing.
2. **Self-review collapse**: model may give all plans the same generic high score. Mitigation: test calibration on weak vs strong plans (σ_v8 baseline vs μ-v4 production) — should produce different scores.
3. **Dimension-collapse**: Qwen3 may not differentiate U1 vs U4 vs U5. Mitigation: compute per-dim variance — if all dims correlate >0.9 within a cell, model isn't actually distinguishing axes.
4. **Reviewer = policy instability**: same model evaluating itself might create degenerate equilibria during training (relevant for future iter, not this offline re-audit).

## Connection to existing work

- F16 (oracle-transfer ceiling): self-review may reveal Opus was over-rewarding surface-form mimicry; if Qwen3 is harsher, raises the bar for "true grounding".
- F17 (RAG cliff): same trajectory should appear under self-review if cliff is plan-quality-driven; if cliff is auditor-perception-driven, won't.
- F18 (KL anchor stability): is L2's stability a real plan-quality signal or an Opus-perception signal? Self-review tests this.

## Pre-registered decision rule

After Phase 5, if:
- ρ ≥ 0.8 AND L2 still ranks above L1 under self-review: **L2 stability finding holds**, F18 strengthened
- ρ ≥ 0.8 AND L2 ranking flips to below L1: **F18 reviewer-dependent**, retract paper claim about KL anchor
- ρ < 0.5: **all v8/v9 conclusions are reviewer-dependent** — major paper-narrative pivot needed

This is a high-leverage diagnostic. Cost is $0 marginal. Run unconditionally before committing to any paper draft.

## Critical path files

- v9 trainer (already exists): `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v9_kl_anchor.py`
- Existing audit dispatcher: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py`
- Audit rubric: `projects/d5_abstract_retrieve_refine/knowledge/current/AUDIT_RUBRIC_v3.md`
- Aggregator: `shared/tools/aggregate_audit_partials.py`
- Reference plan (calibration): `projects/d5_abstract_retrieve_refine/runs/2026_04_29_reference_audit_v3/audit_responses/iter_ref_plan_0.partial.json`
- Tinker API: `shared/tools/use_api_profile.sh` + sampling-client pattern from any v8/v9 trainer
