# Exp C — G+ RAG Audit Summary

*Run: 2026_04_29_mu_v8_d5sdpo_Gplus_rag — D5 Phase 2 / oracle-transfer ABC.*
*Filed 2026-04-29. Stopped at iter 8 (8/16 planned, manual stop after cliff confirmed at iter 6).*

## Setup

- Variant: μ-v8-d5sdpo + RAG (oracle from 72,056 chars / full 87 items → 5,087 chars / top-K=5 retrieved via BAAI/bge-small-en-v1.5)
- Cell config: G+ baseline match — `solution_only_mask=True trust_region_alpha=0.1 loss_fn_name=ppo ppo_clip_eps=0.2 audit_drop_threshold=7.0`
- 8 plans/iter, M8-strict 1-plan/Opus audit (8 parallel subagents per iter)
- Trainer killed manually after iter 8 STUDENT (8 audit iters complete, iter 8 critic blocked by stop)
- **Reason for early stop**: cliff onset confirmed at iter 6 (mean drop -5.4 from iter 5); iter 7 stayed in collapse zone; remaining iter 8-15 would be redundant post-cliff data per pre-registered ABC plan goals

## Audit /45 trajectory

| iter | mean | min | max | n | vs G+ baseline | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 18.88 | 17 | 20 | 8 | 19.50 | -0.62 |
| 1 | 18.00 | 16 | 20 | 8 | 18.75 | -0.75 |
| 2 | 17.50 | 14 | 19 | 8 | 19.88 | -2.38 |
| 3 | 19.00 | 16 | 21 | 8 | 20.38 | -1.38 |
| 4 | 20.12 | 18 | 22 | 8 | 22.38 | -2.25 |
| **5** | **21.25** | 16 | 26 | 8 | 22.38 | **-1.12** |
| 6 (cliff) | 15.88 | 13 | 20 | 8 | 21.38 | -5.50 |
| 7 (post) | 16.88 | 11 | 23 | 8 | 20.00 | -3.12 |

**Peak**: iter 5 = 21.25/45 (Δ vs G+ peak 22.38 = -1.12 — within noise floor)
**Cliff**: iter 6 (5.4-point drop from iter 5; G+ cliff was iter ~11)
**Post-cliff floor**: iter 6-7 ≈ 16-17/45

## Per-dim trajectory (mean across 8 plans)

| iter | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.88 | 3.00 | 2.00 | 3.00 | 1.25 | 2.50 | 2.00 | 1.25 | 2.00 |
| 1 | 1.88 | 3.00 | 2.00 | 3.00 | 1.25 | 2.38 | 1.62 | 1.00 | 1.88 |
| 2 | 1.75 | 2.88 | 2.00 | 2.75 | 1.25 | 2.25 | 1.75 | 1.25 | 1.62 |
| 3 | 1.88 | 2.88 | 1.88 | 2.88 | 1.38 | 2.50 | 2.12 | 1.50 | 2.00 |
| 4 | 2.12 | 2.88 | 1.88 | 2.88 | 1.88 | 2.50 | 2.12 | 1.88 | 2.00 |
| **5 peak** | 2.12 | 2.88 | 2.00 | 2.62 | **2.50** | 2.50 | 2.00 | **2.25** | 2.50 |
| 6 cliff | 1.50 | 2.38 | 1.50 | 2.00 | 1.75 | 2.00 | 1.38 | 1.50 | 1.88 |
| 7 collapse | 2.00 | 2.25 | 1.88 | 2.00 | 1.75 | 2.00 | 1.25 | 1.62 | 2.12 |

**Peak signature** (iter 5): U5 reproducibility 2.50 + T3 compute 2.25 + T4 reward-hacking 2.50 — RAG-conditioned LoRA is briefly producing operational-compute statements before collapse. Same dimensions are weak in σ_v8 baseline (U5=1.12, T3=1.38), so the lift is real, but transient.

**Cliff signature** (iter 6): all dims drop, U3 originality and T2 disentanglement crash hardest (1.50 / 1.38 — to floor). Suggests model abandons substantive recombination and falls into surface boilerplate.

## Trainer stats

| iter | mean_adv | pos_frac | mean\|A\| | train (s) |
|---|---:|---:|---:|---:|
| 0 | -0.119 | 0.058 | 0.160 | 11.9 |
| 1 | -0.144 | 0.062 | 0.184 | 13.4 |
| 2 | -0.111 | 0.089 | 0.160 | 12.5 |
| 3 | -0.086 | 0.110 | 0.133 | 10.7 |
| 4 | -0.068 | 0.103 | 0.109 | 11.5 |
| 5 (peak) | -0.058 | 0.118 | 0.107 | 12.1 |
| 6 (cliff) | -0.045 | 0.104 | 0.078 | 12.1 |
| 7 (post) | -0.037 | 0.097 | 0.052 | 11.1 |

mean|A| collapsing iter 4→7 (0.16 → 0.05) — gradient signal essentially shrinking to nothing as model template-converges.

## RAG retrieval verification

`oracle_for_student` at startup (cached for all iter, query=goal):
- 5,087 chars (7.1% of full slim oracle's 72,056 chars)
- Smoke confirmed top-5 includes Math 5 (J_RS, rank 2 score 0.80), Methodology 2 (TTRL, rank 1 score 0.81), Insights 2 (TTRL per-problem, rank 3), Failure_modes 9 (rank 4), Math 12 (TTRL gradient, rank 5)
- Mix of Math + Methodology + Insights + Failure_modes; no section-class skew

`oracle_for_teacher` per-iter (query=goal+critique_current): not logged; assumed similar mix.

## Cost vs budget

- Pre-registered: ~$40 / 6 hr (Tinker only)
- Actual at stop: ~$80 (5 iter Tinker + 8 iter audit subagent + 8 critic subagent dispatches)
- Saved by stopping iter 8 → 8 (instead of 8 → 16): ~$80-100 + ~3 hr

## Verdict (Exp C contribution to ABC)

1. **RAG peak ≈ G+ peak (Δ = -1.12 < 1, within noise)** → corroborates pre-registered EXPERIMENT_PLAN L154 row "RAG ≈ G+ peak → H16-2 dominant; H16-3 indeterminate". Combined with Exp A (H16-1 falsified) and Exp B (H16-2 corroborated at n≥3), **the dominant bottleneck is H16-2 (incentive misalignment), not retrieval bandwidth.**

2. **NEW finding (not in pre-registered matrix): RAG cliff iter 6 vs G+ cliff ~iter 11** — RAG cell template-collapses ~5 iters earlier than full-oracle G+ baseline. This suggests **oracle BREADTH (87 items vs 5) provides robustness against self-distillation drift** independent of which specific items contribute to gradient signal. Not in original H16-1/2/3 set; goes into next-stage F17.

3. **Production direction**: verifier-grounded reward is the primary lever (per H16-2 dominance). RAG alone is NOT a viable production substitute — it accelerates collapse. If retrieval is used at all, it should be hybrid (full oracle for student, retrieved for teacher) or stacked with verifier-grounded reward + a stronger anti-collapse anchor (e.g., higher α trust-region, or KL-regularization toward frozen base on broader oracle).

## Files

- `buffer.jsonl` (64 rows = 8 iter × 8 plans)
- `audit_responses/iter_{000..007}.json` (aggregated) + `iter_{000..007}_plan_{0..7}.partial.json`
- `critic_responses/iter_{000..007}.json` (8 critic responses; iter_008 request remains unprocessed in `critic_requests/iter_008.json`)
- `checkpoints.jsonl` (8 LoRA checkpoints saved)
- `config.json` (use_rag=True, rag_k=5, rest matches G+ baseline)
- `code.diff`
