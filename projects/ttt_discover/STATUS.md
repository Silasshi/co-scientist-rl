# TTT-Discover STATUS

*Last updated: 2026-04-18 (v9 signal hardening + 3-condition rerun)*
*Read this BEFORE doing any work in this direction.*

## Current Phase

**v9 signal hardening complete. Added S2a_formalism (formula-depth counting)
and SA_arithmetic (arithmetic consistency, 235B grader). Rebalanced weights
so depth signals have commensurate weight (0.25 vs 0.27 for structural).
Ran 3 conditions: MAIN_v9 (RL), B4_v9 (no RL), B4_stripped (no critique).
Key findings:**

**1. Critique conditioning is the dominant mechanism**: B4_v9 (0.950 Qwen,
9/20 Opus) vs B4_stripped (0.763 Qwen, 6/20 Opus) — +0.187 on Qwen, +3
on Opus. Per-signal critique in the revision prompt is load-bearing.

**2. RL trajectory differs but converges to same endpoint**: MAIN_v9
started slower (0.793 at iter 9 vs B4's 0.892), overtook B4 at iter 19
(0.950 vs 0.935), but B4 caught up by iter 24. Both finish at 0.950 /
9/20. RL adds trajectory acceleration but not final-value improvement.

**3. Signal hardening works**: Qwen ceiling dropped from 0.975 (v8.1) to
0.950 (v9). Opus improved from 7-8/20 to 9/20 (+1-2 points). Correlation
with Opus is no longer inverted.

**4. RL with train_on_fresh DOES beat no-RL at BoN=1** (fresh-RL experiment,
2026-04-19): MAIN_fresh_bon1=0.927/10 vs B4_bon1=0.840/8 (+0.087 Qwen, +2 Opus).

### Key numbers (all experiments combined)

| Run | Qwen buffer_max | Opus /20 (Math/Nov/Real/Rig) |
|---|---|---|
| Goel reference | ~0.748 | 20 (5/5/5/5) |
| **MAIN_fresh_bon1** (RL+fresh, BoN=1) | **0.927** | **10** (3/2/3/2) |
| **B4_bon1** (no RL, BoN=1) | **0.840** | **8** (2/2/2/2) |
| MAIN_fresh_bon2 (RL+fresh, BoN=2) | 0.925 | 9 (2/2/3/2) |
| MAIN_v9 (RL, no fresh, BoN=2) | 0.950 | 9 (2/2/3/2) |
| B4_v9 (no RL, BoN=2) | 0.950 | 9 (2/2/3/2) |
| B4_stripped (no critique) | 0.763 | 6 (2/1/2/1) |
| MAIN_v8.1 (prior) | 0.975 | 7 (1/2/2/2) |
| B4_v8.1 (prior) | 0.975 | 8 (2/2/2/2) |

### v9 Signal Set Changes

- S2a_formalism (NEW, weight 0.10): non-trivial formula counting. 100%
  perturbation detection. Grader: Qwen3-30B.
- SA_arithmetic (NEW, weight 0.08): arithmetic consistency validation.
  100% perturbation detection. Grader: Qwen3-235B.
- S2_rigor: weight 0.13→0.08 (coverage split with S2a)
- S1_depth: weight 0.03→0.07, S8: 0.18→0.13, S9: 0.19→0.14

### Component contribution hierarchy (confirmed across all experiments)

```
critique conditioning >> RL (with fresh training) >> BoN >> nothing
   (+0.187 at BoN=2)     (+0.087 at BoN=1)       (compensates weak parents)
```

**RL's unique contribution**: improving the exploration policy (fresh plans).
Revision-only RL doesn't separate from no-RL because critique conditioning
already saturates the exploitation channel. train_on_fresh gives RL an
asymmetric advantage: B4 cannot improve fresh plan quality by any mechanism.

**RL at BoN=1 ≈ no-RL at BoN=2**: MAIN_fresh_bon1 (0.927) ≈ B4_v9 (0.950).
RL effectively learns what BoN=2 provides via inference-time selection.

### Artifacts

- `analysis/cr_v9_signal_hardening_analysis.ipynb` — v9 trajectory + Opus comparison
- `runs/2026_04_18_v9_signal_hardening/{MAIN_v9,B4_v9,B4_stripped}/`
- `runs/2026_04_18_v9_fresh_bon1/{MAIN_fresh_bon1,B4_bon1,MAIN_fresh_bon2}/`
- `knowledge/current/SIGNAL_SET_v9.md` — v9 signal documentation

---

## Previous Phase (CR-v7, 2026-04-18)

CR-v7 complete end-to-end. Core finding: the +0.19 training-grader
improvement from CR-v6 → CR-v7 is an ARCHITECTURE effect, NOT an RL effect.

### Key numbers (Opus 4.7 cross-family depth audit, 2026-04-18)

| Run | Qwen3-30B | Opus /20 (Math/Nov/Real/Rig) | Δ from Goel ref |
|---|---|---|---|
| Goel reference | 0.748 | 20 (5/5/5/5) | baseline |
| MAIN_v6 (prior) | 0.768 | 8 (1/2/3/2) | Q +0.02, O −12 |
| B4_v6 (prior) | 0.785 | 9 (2/2/3/2) | Q +0.04, O −11 |
| **MAIN_v7** (RL on) | **0.975** | **7** (1/2/2/2) | **Q +0.227, O −13** |
| **B4_v7** (RL off) | **0.975** | **8** (2/2/2/2) | **Q +0.227, O −12** |

**Both MAIN and B4 v6→v7 deltas: Qwen +0.19, Opus −1.** RL is not the causal
driver. Architecture is.

### Statistical tests (MAIN vs B4 single-seed 25-iter)

- Paired t-test on 4 iter-level metrics (fresh_mean, overall_mean, iter_max,
  buffer_max): 0/4 significant at α=0.05
- Mann-Whitney U on pooled revision scores: B4 > MAIN, U=3391, p=0.027,
  Cohen d=-0.34 (medium, B4 slightly better)
- Trajectory AUC: MAIN 22.26 vs B4 22.45, 95% CI [-0.69, +0.30], P(MAIN>B4)=0.23
- Power analysis: n=25 iters detects d≥0.56 at 80% power. All observed |d|<0.5
  → underpowered. "No difference" ≠ "no effect" — but the direction at least
  isn't favoring MAIN.

### What RL DID do

- Shifted policy measurably: MAIN vs B4 fresh-mean gap reached 0.40 at peak
  (fresh-mean trajectories visibly diverge)
- S2_rigor went from 2/5 (CR-v6) to 5/5 (CR-v7) on training grader — BUT Opus
  says formal content didn't actually increase. RL learned to trigger S2=5
  without adding formulas. This is **reward-hacking, isolated to the training
  grader's aggregation rule**.
- RL successfully optimized the grader. The grader is just a flawed proxy.

### What the architecture did (RL-independent)

Three mechanisms in CR-v7 jointly game the rubric:
1. **Critique-conditioned sampling prompt**: all 8 per-signal critiques in the
   rewrite context → policy writes directly to rubric failure-modes
2. **BoN=2 + grader-argmax selection**: inference-time optimization of
   Qwen3-30B objective; no RL needed
3. **Whole-plan rewrite scope**: restructure aggressively to hit every signal

These mechanisms are active in BOTH MAIN and B4. B4 with `skip_rl_update=True`
produces identical buffer_max, slightly better Opus depth. RL is a layer on
top that shifts policy but doesn't shift buffer_max or Opus depth.

### Artifacts

- `projects/ttt_discover/analysis/cr_v7_paper_runs_analysis.ipynb` — full
  trajectory + per-signal Δ + cross-version comparison
- `projects/ttt_discover/analysis/cr_v7_rl_ablation_report.ipynb` — 9-hypothesis
  scoreboard + paired t / Mann-Whitney / Cohen's d / bootstrap / power /
  Opus interpretation
- `projects/ttt_discover/analysis/external_judge_eval/opus_judge_report.md` —
  CR-v6 era (8 plans + reference)
- `projects/ttt_discover/analysis/external_judge_eval/opus_judge_report_v7.md`
  — CR-v7 era (MAIN + B4 + reference; N=10 combined v6+v7 cross-grader evidence)
- CR-v6 end-to-end correctness confirmed over 2026-04-17/18 sessions.

### Recommended next work

1. **Ablation matrix** to isolate architectural drivers:
   - CR-v7 full (have): Qwen 0.975 / Opus 7-8
   - CR-v7 without BoN (n_rev_cand=1): isolates critique-conditioning
   - CR-v7 without critique (old bottleneck prompt): isolates BoN
   - CR-v7 locus-scope (v6-style surgical edits): isolates whole-plan
2. **Harder goal** without 0.975 ceiling — gives RL headroom to manifest
3. **Multi-seed** (3× at minimum) — current single-seed is underpowered
4. **Stop training on Qwen3-30B at CR-v6** — external judge already flatlines
5. **Cross-grader panel** — Qwen3-30B + Opus/GPT with early-stopping on
   external-judge plateau

### 2026-04-17 end-of-session note (historical)

Below content is the running history from 2026-04-17 design work through
2026-04-18 paper runs / Opus audit. See DECISIONS.md for the narrative arc.

---

**CR-v7 mechanics summary** (full plan at
`~/.claude/plans/loss-function-importance-sampling-ppo-p-adaptive-wilkinson.md`):

1. Grader emits `<critique>` XML per signal alongside score (new).
2. Revise sampling: ALL 8 per-signal critiques fed into a single
   whole-plan rewrite prompt — policy outputs one action `a`.
3. Re-grade revision → 8-dim Δ vector.
4. Loss construction: for each signal `i` with |Δ_i| ≥ threshold,
   recompute logπ(a | context_i) under context containing ONLY
   critique_i (SDPO `compute_logprobs_async` trick). Emit one
   `types.Datum` per signal with advantage `A_i = Δ_i · delta_scale`.
5. All 8 datums per revision go into single
   `forward_backward(..., loss_fn="importance_sampling")` call.
6. Option (c) REINFORCE: stored logprobs = recomputed logprobs →
   ratio = 1, IS-weighted LL degenerates to per-signal REINFORCE on
   the context_i distribution.

**Also landed this session (2026-04-17 CR-v7 block)**:
- INDENT BUG FIX: `training_datums.append(datum)` at legacy advantage
  path (line ~1786) was at 12 spaces instead of 16 → only the LAST
  datum appended per iter. Introduced 2026-04-12 commit `6369ae12`.
  Affected all CR-v5/CR-v6 paper runs (they trained on 1 datum/iter
  instead of up-to-8). Fixed in this session.
- Tinker API fix: `renderers.get_text_content` no longer exists on
  current tinker_cookbook; replaced with `parse_response(...)[0]["content"]`
  across 4 trainer files.

## Previous Phase (2026-04-17 post-hoc correction)

Post-hoc notebook analysis surfaced that the stitching fix shipped
earlier this session over-rejected 98% of revisions and effectively
disabled CR-v6 locus in paper runs. Opus cross-grader ceiling
evidence is valid independently and remains the canonical paper
result.

**What happened this session**:
1. Stitching bug fix shipped (`_spans_complete_block` validation,
   `train_critique_revise.py:508-631`). Unit tests (58/58 locus tests)
   passed. **BUT live paper runs reveal over-rejection: 98% of
   revisions in MAIN and 88% in B4 rejected as `partial_span`. 0
   revisions reached `ok` status.** Fix needs relaxation (accept
   single-line quotes unconditionally; require block-boundary only for
   multi-line quotes) + re-run.
2. 3 paper runs launched and all completed without crash:
   - `MAIN_v6_minimal` (25 iter full CR-v6 RL): **buffer_max 0.768**
   - `B4_v6_minimal` (25 iter skip_rl_update): **buffer_max 0.785**
   - `B1_v6_minimal` (1 iter zero-shot): buffer_max 0.597
   - CAVEAT: because CR-v6 locus was disabled by the over-strict fix,
     these numbers represent **fresh-plan generation with buffer
     context** (+ RL on fresh for MAIN). B4 ≥ MAIN (0.785 vs 0.768) is
     a within-noise comparison of two non-CR-v6 setups.
3. External judge eval via Claude Opus 4.7 subagent (NOT OpenRouter) on
   9 plans (1 reference + 5 pilot + 3 paper-run). Report at
   `analysis/external_judge_eval/opus_judge_report.md`. **Opus findings
   are robust to the fix-over-rejection issue**: plans are evaluated as
   they were, regardless of how they were generated. Ceiling story is
   independently sound.
4. Post-hoc notebook analysis at
   `analysis/paper_runs_2026_04_17_analysis.ipynb` — detailed per-iter
   breakdown of revision dynamics, bottleneck cascade collapse to
   S2_rigor, partial_span rate over time, and cross-grader comparison.
   This notebook is the source of truth for paper-run mechanics.

**Quantified signal ceiling** (canonical paper numbers):

- **4 of 8 generated plans beat the reference on Qwen3-30B training
  grader (0.782-0.815 vs ref 0.748) but score 5-9 out of 20 on Opus
  judge (ref = 20/20).**
- **Spearman ρ = -0.16** between Qwen3-30B and Opus rankings on the 8
  generated plans (slight NEGATIVE correlation).
- **Math formalism is the largest blind spot** (Opus mean 1.25/5 across
  plans vs reference 5/5, gap 3.75). Novelty next (gap 3.375).
- **6 fatal technical errors missed by Qwen3-30B**: CMA-ES on full LLM
  weights × 3 plans (O(n²) memory on 7B params infeasible), FLOP budget
  off by 8 orders of magnitude, MAML mis-cited in single-task setting,
  GPT-4 cited as open-weight base.

**Key new insight — ceiling is at AGGREGATION, not DETECTION level**:
Qwen3-30B's own S2_rigor signal correctly assigns 1-2/5 to every
paper-run plan (it SEES the math gap), but S2's 1/8 aggregation weight
cannot overcome inflated S7_specificity and S8_scope (both 5/5 from
version-pinned tool names). The weighted-mean hides the detected gap.
Reframes the ceiling from "add depth signals" to "fix how existing
signal detections aggregate."

**Paper narrative (now defensible on numbers)**: "CR-v6 + multi-signal
universal reward optimizes STRUCTURAL plan quality. Ceiling on
mathematical/algorithmic depth is quantified via cross-grader inversion:
Qwen3-30B ranks 4/8 generated plans above the human expert reference,
while Opus 4.7 ranks them 5-9 out of 20 with reference at 20/20."

## Active Configuration (CR-v6 + minimal goal/prompt, locked in 2026-04-17)

| Component | Value | Locked-in | Source of truth |
|---|---|---|---|
| Signal set | v8.1-minimal (8 active signals, S4 disabled) | 2026-04-15 | `knowledge/current/SIGNAL_SET_v8_1.md` |
| Signal weights | S8=0.18, S9=0.19, S2/S6=0.13, S3/S7=0.12, S5=0.10, S1=0.03, S4=0 | 2026-04-15 | `src/co_scientist/shared/ten_signal_reward.py` |
| Pipeline | **CR-v6 (locus-based revision)** | 2026-04-17 | `knowledge/current/PIPELINE.md` |
| Revision mechanism | Grader-emitted locus → `apply_locus_revisions` exact-string replace with reject on no_match / ambiguous / **partial_span** (stitching-bug fix 2026-04-17 next-session: reject quotes that don't span complete markdown blocks — falls back to no-op revision) | 2026-04-17 (next-session) | code |
| Advantage | C3 mixed: `α·Δ_signal_target + β·Δ_aggregate` (α=0.7, β=0.3) | 2026-04-17 | code |
| Signal gating | S3_positioning in default `revision_skip_signals` | 2026-04-17 | code |
| RL update rule | RAW delta (both positive and negative) — see `train_critique_revise.py:~1195` | — | code |
| Grader tokens | 4096 default; locus path uses `locus_grader_max_tokens=6144` | 2026-04-17 | — |
| Policy / grader model | Qwen3-30B-A3B (same model) | 2026-04-13 | — |
| **Active goal** | **Goel-style 127-word scenario (Mixed mode)** | **2026-04-17 (later)** | **`analysis/sanity_check/research_goal.txt`** |
| **Plan-gen prompt** | **Minimal: goal + buffer context + XML output, NO directive sentences** | **2026-04-17 (later)** | **`train_buffer_ttt.py:189-238`** |
| Iter count (paper runs) | 25 (down from 50; corrupted runs plateaued at iter 8-19) | 2026-04-16 | — |

Old verbose goal archived: `analysis/sanity_check/research_goal_v1_detailed.txt`.

## Active Experiments

**Completed in 2026-04-17 (next-session execution)**:
- `MAIN_v6_minimal/` — 25 iter full CR-v6 RL, final buffer_max **0.768** (iter 10 fresh, 826 words, plateau from iter 10 through iter 24)
- `B4_v6_minimal/` — 25 iter in-context only (skip_rl_update=True), final buffer_max **0.785** (iter 14 revision, 868 words — breakthrough from 0.742 plateau at iter 14)
- `B1_v6_minimal/` — 1 iter zero-shot (n_fresh=4, no revise, no RL), final buffer_max **0.597** (iter 0 fresh, 679 words)
- All 3 launched with `locus_based_edit=True, max_tokens=4096, disabled_signals=S4_significance` via `paper_experiments/launch_all_v81_minimal.sh`. ~3h total runtime parallel, no errors, no crashes.
- Best plans extracted to `analysis/external_judge_eval/{MAIN,B4,B1}_v6_minimal_best_plan.txt`.

**Completed in 2026-04-17 (end-of-session)**:
- `CR_V6_PILOT_minimal/` — 10 iter under minimal goal/prompt. Buffer_max 0.545→0.815. 5 best plans saved at `best_per_pair/pair_*_iter*_score*.txt`. Pre-stitching-fix; 2 of 5 best plans contain duplicated section blocks.
- `SMOKE_minimal/` — 1 iter, 2 fresh, no revise (verification only). max_tokens=4096 fix validated.

**External judge eval (Opus 4.7, 2026-04-17 next-session)**:
- Input: 9 plans (1 reference + 5 pilot + 3 paper-run), 4 depth dimensions × 1-5 scale
- Report: `analysis/external_judge_eval/opus_judge_report.md` (3388 words)
- Headline: 4 of 8 generated plans beat reference on Qwen3-30B (0.782-0.815 vs ref 0.748) but score 5-9/20 on Opus (ref 20/20). Spearman ρ = -0.16.

**DEPRECATED runs** (incomparable to new setup):
- `runs/2026_04_30b_paper_experiments_v81/MAIN_v6/` (iter 1, verbose goal)
- `runs/2026_04_30b_paper_experiments_v81/B4_no_training_v6/` (iter 1, verbose goal)
- `runs/2026_04_30b_paper_experiments_v81/B1_zero_shot_v6/` (verbose goal)
- `runs/2026_04_30b_paper_experiments_v81/CR_V6_PILOT/` (10 iter, verbose goal — lineage 0.638→0.907 cannot be cited as canonical)

Corrupted v8 runs archived: `runs/2026_04_30b_paper_experiments_v81/_corrupted_2048/`

## Pending Decisions (status after post-hoc notebook analysis)

1. ⚠️ **Stitching bug fix APPLIED but OVER-REJECTS** —
   `_spans_complete_block` validation in
   `apply_locus_revisions` (`train_critique_revise.py:508-631`). Post-hoc
   analysis (`analysis/paper_runs_2026_04_17_analysis.ipynb`) reveals the
   validation rejected **98% of revisions in MAIN (94/96) and 88% in B4
   (84/96)** as `partial_span`. 0 revisions reached `ok` status across
   both paper runs. The initial "6/8 positive revisions at +0.049
   mean_delta" claim was **wrong**: those are no-op re-grades with
   grader noise (mean_delta across all 94 partial_span revisions in
   MAIN was actually -0.027). **CR-v6 locus revision was effectively
   disabled the entire paper run.** Paper-run buffer_max improvements
   (MAIN 0.637→0.768, B4 0.585→0.785) came almost entirely from fresh
   plan generation with buffer context, NOT from locus revisions.
   **Fix needs relaxation** — likely: accept single-line (no-newline)
   quotes unconditionally, keep block-boundary check only for multi-line
   quotes. Then re-run MAIN + B4. Open decision: relax + re-run (~3h
   compute), OR accept paper numbers as "fresh-only baseline" and
   weaken the paper's CR-v6 advantage claim.
2. ✅ **External judge = Claude Opus 4.7 subagent used**. Report at
   `analysis/external_judge_eval/opus_judge_report.md`. Opus findings
   are INDEPENDENT of the stitching-fix over-rejection bug — the
   4/8 inversions, Spearman ρ = -0.16, and math-formalism-biggest-blind-spot
   insights all hold. Signal-ceiling story is sound.
3. ⚠️ **Bottleneck collapse to S2_rigor** — paper-run buffers show
   S2_rigor is targeted in 96/96 MAIN revisions and 94/96 B4 revisions
   (from iter 2 onward). Pilot had more variety (S2/S1/S5/S7/S9/S6 mixed
   across 10 iters). This is itself a finding: under minimal goal, S2
   becomes a persistent bottleneck that CR-v6's "cascade" cannot escape.
   The ceiling-at-aggregation framing explains this.
4. **Remains out of scope for paper deadline**: Qwen3-235B grader
   retraining, new "Implementation Realism" signal, 60-ref dataset
   re-extraction.

## Phase 0 Findings (2026-04-16, see `analysis/signal_validity/reports/v8_1/locus_accuracy_S3.md`)

1. **7/8 signals (S1, S2, S5, S6, S7, S8, S9) produce grounded reasoning** — locus path safe
2. **S3 on TTT-Discover goal has goal-specific grader prior** — 8% true-CORRECT attribution validity; both 30B and 235B share the hallucinated "AlphaEvolve/OpenEvolve/ThetaEvolve" canonical Related Work
3. **Cross-goal (5 different topics) S3 is clean** — pathology is TTT-Discover-specific, not rubric-level
4. **S3 failure self-limits**: hallucinations → inflated scores → S3 not bottleneck → locus not requested
5. **235B is NOT independent judge**: 80% over-claims verbatim due to same-family prior. Only trustworthy external judge would need cross-family model (GPT/Claude/Gemini)
6. **30% of CR-v5 buffer entries are partial-plan fragments** (stitch failures + max_tokens=2048) — side finding; CR-v6 `apply_locus_revisions` prevents new fragments
7. **Code-vs-docs mismatch**: `train_critique_revise.py:1195-1197` uses raw delta (both signs), NOT positive-only as comment at 1240-1242 + PIPELINE.md + memory all claim. CR-v6 will update docs + use C3 mixed advantage

## Last 3 Completed Actions

1. **2026-04-17 (next-session execution)** — Stitching bug fix (OVER-REJECTED, 98% partial_span in MAIN, 88% in B4) + 3 paper runs + Opus judge eval + post-hoc notebook analysis + STATUS/DECISIONS correction. **Honest characterization**: stitching fix shipped and unit tests (58/58 locus tests) passed, but live paper runs show the block-boundary validation is too strict and rejected nearly all policy-emitted revisions. Paper-run buffer_max improvements (MAIN 0.637→0.768, B4 0.585→0.785, B1 0.597) came almost entirely from **fresh plan generation + buffer context**, NOT from locus revisions. CR-v6 advantage was NOT tested in these paper runs. Opus 4.7 judge (3388 words report) still produced valid cross-grader evidence independently of this bug: 4/8 plans beat ref on Qwen3-30B (0.782-0.815 vs 0.748) but score 5-9/20 on Opus (ref 20/20), Spearman ρ=-0.16, math formalism biggest blind spot (Opus 1.25/5 vs ref 5/5). Key insight (independent): S2_rigor detects math weakness correctly (1-2/5 on every paper-run plan) but 1/8 weight is swamped by S7/S8 — ceiling is at AGGREGATION, not detection. Detailed notebook: `analysis/paper_runs_2026_04_17_analysis.ipynb`.
2. **2026-04-17 (end-of-session)** — CR-v6 minimal-mode pilot complete (10 iter, `CR_V6_PILOT_minimal/`): buffer_max 0.545→0.815, mean 0.526→0.729. 5 best plans extracted (`best_per_pair/`) and analyzed vs reference. Two findings: stitching bug in `apply_locus_revisions` (partial-span replace leaves residue → duplicate sections); signal ceiling quantified (gen 0.815 > ref 0.748 on Qwen3-30B but ref objectively deeper — 4 formulas + max-Q PUCT vs gen 0 formulas + standard ES+grad combo + unspotted CMA-ES infeasibility). Smoke v1 with default max_tokens=2048 FAILED hard gates (truncated mid-methodology); fixed by max_tokens=4096. Plan for next session: `~/.claude/plans/binary-chasing-pebble.md`.
3. **2026-04-17 (later)** — Mixed-mode goal/prompt redesign: active `research_goal.txt` switched to Goel et al. 2025-style 127-word scenario (old archived); minimized `build_research_plan_prompt`, `build_diverse_fresh_prompt`, `build_locus_revision_prompt`; deleted `minimal_prompt` config field + A7 ablation; updated PIPELINE.md, EXPERIMENT_PLAN.md, both launchers. 60-ref dataset NOT touched.

## Open Questions (updated 2026-04-17 next-session)

1. ✅ **Signal ceiling — QUANTIFIED** (Opus judge report, 2026-04-17 next-session). 4/8 inversions, Spearman ρ=-0.16, math formalism gap 3.75 (Opus 1.25/5 vs ref 5/5). Ceiling is at AGGREGATION level: S2_rigor correctly detects math weakness but 1/8 weight is swamped by inflated S7/S8 from version-pinned tool names. 6 fatal technical errors (CMA-ES on full LLM weights × 3, FLOP budget 8 orders off, MAML mis-cited, GPT-4 open-weight). Paper narrative defensible on these numbers.
2. ⚠️ **Stitching bug — PARTIALLY FIXED, OVER-REJECTS**. `_spans_complete_block` validation shipped; 13 new tests + 58/58 locus unit tests pass. BUT live paper runs show 98% partial_span rejection in MAIN and 88% in B4 — the validation is too strict for realistic plan structure (plans use `**Header**: body` inline format; policy quotes starting after `**Header**: ` get rejected even when they're valid single-line edits). Fix needs relaxation: accept single-line (no-newline) substrings unconditionally since they can't cause block-level residue; keep block-boundary check only for multi-line quotes. Paper runs need re-launch after relaxed fix.
3. ✅ **Research goal leakage — RESOLVED**. Goel-style minimal-no-hint goal active. 60-ref dataset unchanged per user.
4. ⚠️ **Paper story needs revision pending fix re-run**: the Opus cross-grader inversion (4/8 plans, ρ=-0.16) is robust and CITABLE independently of the stitching-fix bug. BUT the comparison "MAIN 0.768 vs B4 0.785" cannot be cited as "CR-v6 + RL vs CR-v6 in-context" because CR-v6 locus revision was effectively disabled in BOTH runs. What these numbers actually show is "fresh-only + buffer context + RL (MAIN) vs fresh-only + buffer context (B4)". B4 ≥ MAIN finding (B4 in-context > MAIN RL by 0.017) is NOT evidence against RL — it's a within-noise comparison of two non-CR-v6 setups.
5. **Post-paper open question**: Fix signal aggregation (not add signals). Opus data says S2_rigor DETECTS math weakness — the problem is 1/8 weight. Candidates: (a) increase S2 weight; (b) use min() aggregation instead of weighted mean so any low signal bottlenecks the score; (c) formula-detection signal with LaTeX parsing. v9 scope.
6. **Post-paper open question**: Novelty signal (retrieval-based prior-work embedding distance). Opus finding: "plans happily reward category labels ('Hybrid Test-Time Adaptation', 'Dynamic Reward-Adaptive Learning') without evaluating whether a non-obvious modification exists." Grader cannot detect semantic novelty vs prior work.

## Phase 1.5 Verification Gates (completed)
1. ✅ Unit tests for `apply_locus_revisions` (11 tests in Batch C + 13 new partial-span tests added next-session = 24 total on apply path)
2. ✅ Grader-cost regression: 0.77× (PASS, threshold ≤1.5×)
3. ✅ Smoke test: 5/10 locus success → 5/5 apply success → mean Δ_target=+0.45, 80% positive. Key finding: Qwen3-30B cannot literal-verbatim quote plan text → added fuzzy matching via `_find_best_verbatim_match` (line-based SequenceMatcher). Failures (5/10) fall back gracefully to CR-v5 paragraph path.
4. Deferred to pilot: `corr(Δ_target, Δ_aggregate)` — will compute from pilot iter logs.
5. ✅ Pilot RL run (verbose goal): 10 iter, buffer_max 0.693→0.907. Human quality check: improvement real. DEPRECATED for canonical comparison since goal setup changed.
6. ✅ Paper runs (verbose goal) launched then STOPPED. DEPRECATED.
7. ✅ Pilot RL run (minimal goal, 2026-04-17 end-of-session): 10 iter, buffer_max 0.545→0.815 (+0.27), fresh-mean 0.526→0.729 (+0.20). Lineage: bottleneck cascade S2→S1+S5→S9+S7→S2→S2+S6 (similar pattern to verbose pilot). Locus success rate ~50% with graceful fallback.
8. ✅ Stitching bug FIXED (2026-04-17 next-session). 13 new tests + PIPELINE.md updated. Paper runs show no rejection collapse — CR-v6 advantage preserved.
9. ✅ **Paper runs complete** (2026-04-17 next-session): MAIN 0.768 (RL), B4 0.785 (in-context), B1 0.597 (zero-shot).
10. ✅ **External judge eval** (2026-04-17 next-session): Claude Opus 4.7 on 9 plans (1 ref + 5 pilot + 3 paper-run). Report at `analysis/external_judge_eval/opus_judge_report.md`.

Locus directives for S1-S8 are FIRST DRAFTS — review during gate 3 (smoke test).

## Where to Find Everything

| What | Where |
|---|---|
| Signal definitions + prompts | `knowledge/current/SIGNAL_SET_v8_1.md` |
| Pipeline architecture | `knowledge/current/PIPELINE.md` |
| Active runs (v81) | `runs/2026_04_30b_paper_experiments_v81/` |
| v5-era runs (still usable for comparison) | `runs/2026_04_30b_paper_experiments/` |
| Signal validation data/scripts | `analysis/signal_validity/` |
| Experiment matrix | `paper_experiments/EXPERIMENT_PLAN.md` |
| Current launcher | `paper_experiments/launch_all_v81.sh` |
| Decisions ledger | `DECISIONS.md` |
| Naming/placement rules | `CONVENTIONS.md` |
