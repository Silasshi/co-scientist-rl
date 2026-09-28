# Grant Proposal Generation v2 STATUS

*Last updated: 2026-04-23*

## Current Phase

**GER-CR-v1 Phase 2 validated; Opus depth audits identify grader gap on binary rubric. Starting Phase 3 with grader-gap fix.**

Opus 4.7 depth audit on the 5 buffer_max milestones (2026-04-23,
`runs/2026_04_23_ger_cr_v1_smoke/depth_audit/SUMMARY.json`):

| milestone | pipeline | words | depth | meth | feas | **grnd** | **Opus/40** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| iter0 fresh | 0.525 | 711 | 4 | 4 | 3 | **3** | 14 |
| iter1 rev | 0.815 | 806 | 4 | 4 | 4 | **2** | 14 |
| iter2 rev | 0.880 | 826 | 5 | 5 | 4 | **3** | 17 |
| iter4 rev | 0.890 | 1157 | 5 | 4 | 4 | **3** | 16 |
| iter5 rev | 0.940 | 1058 | 5 | 5 | 4 | **3** | 17 |

ρ(pipeline, opus_total) = 0.757. CR alone hits 17/40 and plateaus;
every milestone flagged by Opus for **fabricated citations and template
Goodhart**. Meanwhile GPT-OSS-120B's binary grader says 50%-88% of
rollouts "avoid fabrication" on the exact same item. This is the v8
judge-gap reproducing on binary grading. Phase 3 must therefore first
**swap binary grader to Qwen3-235B** (already-plumbed `grader_client_alt`)
before wiring R_active into RL — details in
`.claude/plans/structured-snuggling-mccarthy.md`.

**Prior (Phase 2 summary)**:

6-iter FoundOpt smoke run at `runs/2026_04_23_ger_cr_v1_smoke/` with
`skip_rl_update=true`. Every Phase 2 mechanism exercised on live data:
- 5 rubric-gen requests written; 5 subagent responses ingested.
- R_active binary grader ran each iter on 8 rollouts × buffer items.
- `filter_and_truncate` dropped 5 dead-channel items across the run.
- `rubric_anchor_log.jsonl` at iter 5: `ref_aggregate=0.786` (n=5 buffer).
- Final buffer: 4 NEG items targeting fabricated precision, unconstructed
  named artifacts, decorative equations; near-duplicate slipped dedup
  (subtle description variant) — flagged for Phase 3 semantic dedup.
- Revision mean_delta positive every iter (+0.021 to +0.085); buffer_max
  climbed 0.525 → 0.815 → 0.880 → 0.880 → 0.890 → **0.940**. No RL,
  so improvement is pure CR + signal drift from ingested rubric context.
- No unhandled exception; exit 0.

R_persist path untouched (verified by code inspection: all new code gated
behind `config.rubric_evolution_enabled`, default False). Background
subagent prompt in `scripts/rubric_gen_subagent.md`. Phase 3 (R_active
→ RL advantage coupling) deferred.

**Prior status:** v8 runs killed at iter 13-14. Three failure modes confirmed. Path-locking framing under consideration for v9.

The 3 v8 FoundOpt runs (B4 / MAIN_C3 SDPO / MAIN_C4 aggregate) all plateaued on
GPT-OSS aggregate while **Opus length-normalized scores regressed** (B4 −1, C3 −1,
C4 −3 between iter 5 and iter 10). Killed at iter 13-14 after 3+ iters buffer_max
flat. Full writeup in `DECISIONS.md` and `analysis/v7_vs_v8_comparison.md`.

## Three Attack Surfaces Confirmed

From `analysis/attack_surfaces_2026_04_22.ipynb` (13 experiments × 3 v8 + 3 v7 runs):

| Surface | Status | Mechanism | Evidence |
|---|:-:|---|---|
| Length bias | **✅ confirmed** | Encoded in rubric (G11/G12/G13) at iter 0, RL amplifies | ρ(len, G12) = 0.78-0.82 across v8 runs; baseline ρ(len, agg) = 0.47-0.83 |
| Judge gap | **✅ confirmed** | GPT-OSS diverges from Opus length-normalized within 5 iters | iter 5→10: GPT-OSS +0.07 to +0.10, Opus_norm −1 to −3; rank inversion at iter 10 |
| Signal saturation | **✅ confirmed (dead-channel mechanism)** | G4_focus + G10_approach_coverage dead in all v8 runs; NOT weight concentration | S2 var<0.25 on 6/6 runs; S3 ρ(clip, unclip) > 0.95 rules out single-signal max |

**Key insight**: Grader choice routes Goodhart into different axes (Qwen → hallucination, GPT-OSS → length padding) but doesn't close it. Cross-family grader is necessary but not sufficient.

## v9 Design Under Consideration

Synthesis from 4 reference papers in `D:\AI\Co-scientist\reference\`:

1. **Goel 2512 (Meta, Dec 2025)** — same Qwen3-30B-A3B base, 70% expert preference
   - Auto-extracted rubrics from reference papers (not hand-designed)
   - **Violation-based fraction satisfied** scoring (not weighted numeric mean)
   - Hard length cap + violation penalty (not soft max_word)
   - Grader with full paper context as privileged info
2. **RLVRR (ICLR 2026)** — deterministic verifiable hard gates
   - Keyword LCS + LLM-generated Python checks
   - Trade: minimal Goodhart surface, but path-locking concern
3. **DR Tulu (Ai2 2025)** — evolving rubric buffer
   - Rubric co-evolves with policy; dead-signal drop + new criteria generation
   - Directly addresses dead-channel finding
4. **SDPO (ETH 2026)** — self-teacher for credit assignment (already tried as C3)

**Open concern (PATH_LOCKING_FRAMING.md)**: all reference-grounded methods suffer
path-locking — rubric extracted from paper P encodes P's specific approach, not
generic quality. Goel's 84% expert approval = 16% path-specific contamination.
This may be a paper-worthy critique in its own right.

## Established Findings (D4 → D4v2)

1. **Critique is redundant**: scores_only = full_critique on Opus (19 = 19)
2. **RL Goodharts on proxy grader**: B4 ≥ MAIN on Opus across 3 grader × 3 algorithm × 2 rubric (n=9)
3. **235B >> 30B**: +10-17 Opus points from scale alone
4. **Cross-domain gap ~19 points**: grounding is universal weakness
5. **Qwen coarse ranking OK (ρ=0.928), fine ranking saturated**
6. **v8 rubric not circular**: ρ(v8-Opus, depth-Opus) = +0.829 pre-RL
7. **Length bias is iter-0 baseline**: not created by RL, encoded in rubric design

## Decision Queue (priority order)

1. **Pilot Test 2 of PATH_LOCKING_FRAMING** (4 hours, cheapest) — does policy systematically closer to reference than to alternative valid plans? If no → retract framing. If yes → proceed to Tests 1+3.
2. **Goel 2512 replica retrofit** on current code:
   - Check whether grader prompt actually includes reference as privileged info
   - Check whether length violation is hard-enforced
   - Change aggregate to violation-based fraction
3. **Collect alt-reference proposals** (5 goals × 3 each) for consensus-rubric v9
4. **Citation authenticity hard gate** (arXiv/S2 retrieval) — goal-invariant verifiable signal

## Code

- Training: `src/co_scientist/grant_proposal/train_cr_v7.py` + `train_buffer_ttt.py`
- Rubric v8: `src/co_scientist/shared/grant_rubric_v8.py`
- Reward: `src/co_scientist/shared/grant_signal_reward.py`
- Opus eval: `src/co_scientist/grant_proposal/opus_eval_agent.py`
- Analysis: `projects/grant_proposal_v2/analysis/attack_surfaces_2026_04_22.ipynb`

## Dataset

12 goals (symlinked from `projects/grant_proposal/dataset/`).
`01_foundopt_v8/` routes to v8 rubric module; other 11 goals currently on v7.

## Prior Work

All 58 D4 runs + 9 v7 C-series + 3 v8 runs.
Structured config data in `analysis/run_classification/all_runs_config_table.jsonl`.
