# STATUS

*Last updated: 2026-04-27*

## Project Goal

Build a universal tool that generates high-quality research plans for any research field, given any level of input specificity.

## Current Stage

Six research directions. **D5 (Abstract-Retrieve-Refine) is the active focus**, targeting ICLR 2027. **D5 Phase 2 COMPLETE (2026-04-27)**: μ-v4 plan-level SDPO + Opus critic at lr=5e-5 lifts Qwen3-30B-A3B from σ baseline 25.25 → **28.00 / 45 at iter 4 (+2.75)**, the first 30B-trained variant to beat σ. **Pairwise corroborated 20/24 (vs σ 6-2, vs δ 7-1, vs α 7-1)** — audit and pairwise both confirm. Multi-round SDPO instability (Hübotter 2026 §4) prevents sustained gains past iter 5; iter-4 weights are the production checkpoint. (Phase 3 F8 REVISED 2026-04-27 PM: τ_v4_clean = 24.50/45, pairwise 4-4 TIE vs σ_v4; original 7-1 STRONG headline retracted as lucky-draw artifact; μ-v4 28.00 unchanged.) D4v2 tabled (pivot to D5).

## Live Direction State

For day-to-day active work, read the direction-local harness:
- **D5 (Abstract-Retrieve-Refine) — active**: `projects/d5_abstract_retrieve_refine/STATUS.md`
- D4v2 (Grant Proposal v2 — tabled): `projects/grant_proposal_v2/STATUS.md`
- D4 (Grant Proposal — legacy): `projects/grant_proposal/STATUS.md` (58 runs, findings documented)
- D3 (TTT-Discover): `projects/ttt_discover/STATUS.md`
- D1 (Rubric Reward): `projects/rubric_reward/README.md` (ceiling reached)
- D2 (IBT): `projects/ibt/README.md` (under investigation)

## Research Directions

| Direction | Status | Key Result |
|---|---|---|
| **D1: Rubric Reward** | Ceiling reached | Best: 0.693. 80+ runs, no method beats bestversion. |
| **D2: IBT** | Under investigation | Run 7 completed (106 batches). |
| **D3: TTT-Discover** | Active (methodology foundation) | Signal v8.1 + CR-v5 pipeline validated (AUC 0.767 on 60 refs + 162 perturbations). |
| **D4: Grant Proposal** | Tabled (pivot to D5) | B4 (critique-revise, no RL) was best on Opus quality. RL Goodharted. |
| **D4v2: Grant Proposal v2** | Tabled (pivot to D5) | Investigating Qwen-Opus inversion; superseded by D5 direction. |
| **D5: Abstract-Retrieve-Refine** | **Active. Phase 2 COMPLETE; Phase 3 REVISED 2026-04-27 PM.** | μ-v4 iter 4 = 28.00/45 (+2.75 over σ 25.25). 8-baseline audit_v3 ISOLATED: ε 33.62 ≫ μ-v4 28.00 ≫ σ ≈ δ ≈ α > μ-v2 23.88 > β > ξ. Pairwise corroborated 20/24 (vs σ 6-2, vs δ 7-1, vs α 7-1). Phase 3 F8 v2: τ_v4_clean = 24.50/45, pairwise 4-4 TIE vs σ_v4 (original 7-1 STRONG retracted as lucky-draw). Target ICLR 2027. |

## D1 Performance Hierarchy

| Method | Eval Rubric | Notes |
|---|---|---|
| GPT-5.4 (zero-shot, no RL) | 0.843 | External upper bound |
| Reference solutions | 0.860 | Theoretical ceiling |
| **bestversion** (GRPO) | **0.693** | **Best RL result (D1)** |
| Base model (no training) | 0.654 | Zero-shot baseline |
| All other D1 methods | < 0.693 | See `projects/rubric_reward/runs/RUNS_INDEX.md` |

## Key Findings

1. **Selection gap = 91%**: Model generates oracle-level plans (best-of-8 = 0.85) but picks the wrong one (mean 0.70). The bottleneck is selection, not capability.
2. **Rubric reward is goal-specific**: 80.6% of failures come from rubric items that check the reference's specific methodology — not universal plan quality. This motivates D3's universal signal approach.
3. **Process shortcutting**: All multi-step methods (D1 multi-turn, D2 IBT) produce models that eliminate unrewarded intermediate processes.
4. **GPT-5.4 achieves 0.843 zero-shot**: Suggests the capability exists in large models — the challenge is reward/search design.

## D3 Progress

| Phase | Status | Result |
|---|---|---|
| A.0 Sanity check (5 hack templates, v1 signal set) | PASSED | All < 0.30 aggregate |
| A.0.5 Stress test (4 templates, v1 signal set) | PASSED | Hacks 0.000, good plans 0.71-0.73 |
| A.1 GRPO validation | Stopped | Degenerate signals (v1). Pivoted to TTT-Discover search + redesigned signals |
| Sanity check v1 (14 plans, both grader modes) | PASSED | separate_call wins on halo. Revealed S1_coherence as halo magnet |
| Sanity check v2 (14 plans, new signal set) | PASSED (7/8) | S1_mechanism failed → demoted to weight 0.05 |
| Buffer-conditioned TTT trainer | IN PROGRESS | `src/co_scientist/ttt_discover/train_buffer_ttt.py` |
| Pilot training run | PENDING | 5-10 iterations to verify pipeline |

See `projects/ttt_discover/README.md` and `projects/ttt_discover/analysis/sanity_check/SANITY_CHECK_V2_REPORT.md` for full D3 details.

## Active Work

- **D3 CR-v3 for paper**: Switching to Qwen3.5-4B (policy=grader=same model) per advisor guidance. Full experiment suite: 1 main + 4 baselines + 8 ablations = 13 runs. New features: UCB buffer selection, skip_hard_gates, disabled_signals, train_on_fresh flags. See `paper_experiments/EXPERIMENT_PLAN.md`.
- **Key insights discovered**:
  1. Entropic objective needs continuous rewards; integer 1-5 scale causes plateau (Buffer-TTT)
  2. Signal-targeted revision breaks plateau (CR-v1/v2)
  3. Human evaluation revealed Goodhart's Law: high grader score ≠ high plan quality → added S9_focus
  4. Skip-signal enables signal cascade: S5(skip) → S7(100%) → S6(8%) → S1(8%) → S2(new)
  5. Best-of-2 makes low-probability revisions possible (S1 went from 0% to 8%)
  6. Qwen3-235B grader fixes S1 blind spot and improves S9 discrimination
- **Paper preparation**: Presentation scheduled April 27, 2026

## File Locations (post-restructure)

| What | Where |
|---|---|
| D1 code | `src/co_scientist/rubric_reward/` |
| D2 code | `src/co_scientist/ibt/` |
| D3 code | `src/co_scientist/ttt_discover/` |
| D4 code | `src/co_scientist/grant_proposal/` |
| Shared code | `src/co_scientist/shared/` |
| D1 runs & configs | `projects/rubric_reward/` |
| D2 runs | `projects/ibt/` |
| D3 analysis & design | `projects/ttt_discover/` |
| D4 analysis & design | `projects/grant_proposal/` |
| Papers & literature | `shared/papers/` |
| Experiment catalog | `shared/knowledge/EXPERIMENT_CATALOG.md` |
