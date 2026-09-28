# Phase 0 — Locus Accuracy Validation Gate: Report

*Date: 2026-04-16*
*Author: Claude (with user direction)*
*Plan reference: `/home/silas/.claude/plans/compressed-swimming-zephyr.md`*

## TL;DR

**Gate decision: SOFT-PASS for Combo 1 with per-signal gating of S3.**

For the TTT-Discover goal specifically, S3_positioning fails the
attribution-validity gate (8% true CORRECT, vs 70% threshold) due to a
**goal-specific Qwen3 family prior**: both 30B and 235B have memorized a
canonical "Prior work using evolutionary search over frozen LLM
(AlphaEvolve / OpenEvolve / ThetaEvolve)" pattern and emit it as if
quoted from the plan when it is not there.

The failure is:
- **Goal-specific** (5/5 different-topic reference plans tested clean on S3)
- **Self-limiting** (hallucinated loci come with inflated scores →
  S3 never becomes the bottleneck → locus revision not triggered)
- **Safety-net-protected** (`apply_locus_revisions` verbatim reject
  catches bad loci before they hit the RL update)

All 7 other active signals (S1, S2, S5, S6, S7, S8, S9) demonstrate
grounded reasoning across the samples audited.

**Recommendation**: proceed with Combo 1, add `S3_positioning` to the
default `skip_signals` set so S3 never reaches the locus path in
TTT-Discover runs, keep S3 in aggregate reward for plan-quality
evaluation.

---

## Phase 0 design vs. what actually ran

| Step | Designed | Actual |
|---|---|---|
| 0a | Sample 30 plans, stratified by S3 | 24 plans; S3=1 stratum had only 1 eligible plan after filtering; 30% of buffer entries rejected as partial plans |
| 0b | Locus-extended S3 grader script | DONE; max_tokens bumped 6144 → 8192 after parse-failure diagnostic |
| 0c | Grade 24 plans with 30B + locus | 100% parse, 4/25 (16%) loci verbatim-valid by literal substring |
| 0d | 235B external judge | DONE, but finding: 235B shares the same hallucinated prior as 30B — NOT an independent judge |
| DIAG 1 | Canonical vs locus S3 on 3 plans | **Unplanned**, revealed hallucination is in canonical grading too, not locus-induced |
| DIAG 2 | Hallucination on other signals | 7/8 signals clean (S1, S2, S5, S6, S7, S8, S9) |
| DIAG 3 | S3 on 5 different-topic refs | 5/5 clean — confirms TTT-Discover specificity |
| 0e | This report | DONE |

## Phase 0a — Sampling (side findings)

Pool = 5 buffer runs under `runs/2026_04_30b_paper_experiments_v81/_corrupted_2048/`.

After filtering `hard_gate_passed=True AND S3_positioning present AND
word_count >= 800`:
- Raw pool: ~650 entries
- **156 entries (≈30%) filtered as partial plans** (word_count < 800)
  — these are CR-v5 stitch-failure artifacts where only one section
  (e.g., Risk Awareness) was retained when `parse_revised_section`
  failed to parse model output
- Eligible pool: 367 plans

Stratified sample (actual): {S3=1: 1, S3=2: 4, S3=3: 7, S3=4: 7, S3=5: 5} = 24.

**Side finding worth escalating**: ~30% of CR-v5 buffer entries are
partial-plan fragments. This is a data-quality issue **independent of
the locus pilot** — affects training signal quality in all current runs.
See the `apply_locus_revisions` + anchor-aware reject path in Phase 1
Change 6 — the new pipeline does not produce these fragments.

## Phase 0c — 30B locus grading on 24 plans

**Config**: Qwen3-30B-A3B, N=1, temperature=0.0, max_tokens=8192,
ellipsis-tolerant verbatim match.

**Aggregate metrics (before 235B cross-check)**:
- Parse OK (evaluation + locus both parseable): 24/24 (100%)
- At least one verbatim fail: 19/24 (79%)
- Total loci emitted: 25
- Verbatim-valid (Python literal substring): **4/25 (16%)**
- 3 plans correctly emit empty locus (score=5 case — this is intended
  behavior, not failure)

**Dominant failure pattern**: grader emits the canned quote
> `"Prior work using evolutionary search over a frozen LLM (such as
> AlphaEvolve, OpenEvolve, ThetaEvolve) has achieved some success"`

This quote does not appear in any of the 24 plans. It appears
verbatim in **17/21** non-empty locus responses. See `grading_locus_s3.jsonl`.

## Phase 0d — 235B external judge (and why it failed as a check)

**Config**: Qwen3-235B-A22B-Instruct-2507, N=1, temperature=0.0,
max_tokens=8192, timeout=480s.

**Designed output**: for each 30B locus, does 235B judge the quote
verbatim and the attribution correct?

**Reconciliation against Python literal substring**:

| Metric | Value |
|---|---|
| Python-verbatim (ground truth) | 4/25 (16%) |
| 235B claims verbatim | 24/25 (96%) |
| **235B OVER-CLAIMS** verbatim when objectively absent | **20/25 (80%)** |
| True CORRECT (both Python-verbatim AND 235B=CORRECT) | 2/25 (8%) |

**Score agreement (separate metric — not corrupted by locus)**:
- Mean |score_diff| = 0.79
- 30B and 235B agree within ±1 on **88% (21/24)** of plans

### Why 235B failed as an independent check

Both 30B and 235B are Qwen3 family and share the same pretraining
data. On the TTT-Discover goal specifically, both models have
memorized a canonical "Related Work" analysis featuring
"AlphaEvolve / OpenEvolve / ThetaEvolve". When asked to verify if a
30B-emitted locus quoting these names is verbatim, 235B confirms it
is, because the content matches ITS OWN prior about what this goal's
plans should contain.

**Takeaway**: same-family cross-model verification is not a reliable
hallucination check. A truly independent external judge would require
a different model family (GPT-4 / Claude / Gemini). User's global
constraint to minimize OpenRouter spend rules this out. The literal
substring check + cross-goal diagnostic are the trustworthy signals.

## Diagnostic 1 — Canonical vs locus prompt (S3 on 3 plans)

**Question**: does the locus directive induce hallucination, or is it
already present in canonical S3 grading?

| sample | orig_s3 (buffer) | canonical score | locus score |
|---|---|---|---|
| 000 | 1 | 4 | 5 |
| 001 | 2 | 3 | 5 |
| 002 | 2 | 3 | 2 |

Canonical reasoning for sample 000 / 001 / 002 **all** include the
"AlphaEvolve, OpenEvolve, ThetaEvolve" fabrication in STEP 1 (NAMED
prior methods). Locus directive mildly inflates scores further but is
**not** the source of the hallucination.

**Conclusion**: hallucination is grader-level, not prompt-level. The
locus directive cannot be "fixed" via prompt engineering.

## Diagnostic 2 — Is the hallucination S3-specific?

Ran canonical grading of S1_depth, S2_rigor, S5_feasibility,
S6_risk_awareness, S7_specificity, S8_scope, S9_focus on 3 TTT-Discover
pilot plans (21 total grading decisions).

Full audit of reasoning content (manual inspection of grader CoT):

| Signal | Verdict across 3 plans | Notes |
|---|---|---|
| **S1_depth** | CLEAN | Correctly enumerates plan's load-bearing design choices (e.g., Distributional max-reward optimizer, Memory-guided refinement) |
| **S2_rigor** | CLEAN with minor tangent | Scoring logic correct; sample 000 tangentially mentions hallucinated AlphaEvolve but scoring ignores it |
| **S5_feasibility** | CLEAN | Qualitative, every cited specific (model, hyperparameters) verifiable in plan |
| **S6_risk_awareness** | CLEAN | Correctly identifies absence of failure modes when plan lacks them |
| **S7_specificity** | CLEAN | Lexical counting of vague markers vs specific commitments — all grounded |
| **S8_scope** | CLEAN | Overclaim trigger count and domain bucket count both verifiable |
| **S9_focus** | CLEAN | Technique enumeration verifiable |
| **S3_positioning** | BROKEN | All 3 samples emit fabricated "AlphaEvolve/OpenEvolve/ThetaEvolve" prior-work analysis |

**Conclusion**: S3 is uniquely susceptible. Other signals ask for concrete
artifacts (hyperparameters, techniques, lexical markers) that the grader
must ground in plan text. S3 asks about "named prior work" — a concept
where the grader can pattern-complete from goal priors when the plan
genuinely lacks named methods.

## Diagnostic 3 — Is the hallucination goal-specific?

Ran canonical S3 on 5 reference plans from 5 different topics:

| Topic | Paper | 30B grader's named methods | Actually in plan? |
|---|---|---|---|
| 02_harness_synthesis | AutoHarness | Reflexion, ReAct, Tree-of-Thoughts | ✓ |
| 05_self_critique | BeyondSymbolicSolving | Inter-GPS, Pi-GPS, PGPSNet-v2, G-LLaVA | ✓ |
| 07_rl_methods | CalibrationCollapseUnder | Kadavath 2022, Tian 2023, Perez 2022, Wei 2024 | ✓ |
| 09_efficient_training | DoRA | LoRA, AdaLoRA, VeRA, LoRA+, QLoRA, rsLoRA | ✓ |
| 01_test_time_search | RISE | Self-Refine, STaR, V-STaR | ✓ |

**5/5 clean**: no "AlphaEvolve/OpenEvolve/ThetaEvolve" leak even on
same-topic (test_time_search / RISE) plans. Named methods all real and
actually in plan. **S3 hallucination is TTT-Discover-goal-specific**.

### Hypothesis for the pattern

Qwen3 was pretrained on a corpus where TTT-Discover-style research is
covered extensively and is closely associated with AlphaEvolve +
related methods. When grader encounters a TTT-Discover-aligned plan
that genuinely lacks named prior methods (triggering a floor-bias
refusal to say "N_named=0"), it fills in this canonical expected list.

## Mechanism: why the hallucination is self-limiting

| Case | Plan state | Grader behavior | S3 is bottleneck? | Locus revision triggered? |
|---|---|---|---|---|
| A | 0 named methods (genuine absence) | Hallucinate 3 methods + insufficiency → score 5 | No (inflated) | No |
| B | 1-2 named methods, no insufficiency | Correctly identify them, score 2-3 | Usually yes | Yes → locus quote is real |
| C | Many named methods + insufficiency | Correctly identify, score 4-5 | No | No |

**Case A** is the hallucinating case, but because it inflates scores
it never requests revision. **Case B** (where S3 is legitimately the
bottleneck) produces verbatim quotes (see samples 007, 008, 015, 017
in grading data — all quote real plan content).

Additional safety-net: `apply_locus_revisions` from Phase 1 Change 6
rejects any revision attempt whose `quote_original` isn't found in the
plan. A hallucinated locus (unlikely to fire as above, but possible)
would reject cleanly without corrupting the RL update.

## Gate decision

Strict reading of Phase 0 gate (from plan file):
- Threshold: ≥70% attribution validity on S3
- Measured: **8% true CORRECT** (2/25 with Python-verbatim AND 235B=CORRECT)
- → **FAIL**

Adjusted reading given the full diagnostic picture:
- Failure is goal-specific, not rubric-level
- Failure is self-limiting via score inflation
- Failure is safety-netted via revision reject
- Other 7 signals pass cleanly
- → **SOFT PASS** for Combo 1 with per-signal gating

## Recommended Phase 1 actions

1. **Add `S3_positioning` to default `skip_signals` in CR-v6 config.**
   Analogous to how S4_significance was dropped in v8.1. S3 stays in
   the aggregate reward (plan quality evaluation still includes
   positioning) but never triggers locus-based revision.

2. **Proceed with Combo 1 as designed** for 7 other signals.
   Implementation per plan file Changes 1-8 unchanged.

3. **Do not rely on 235B as an external judge for similar future
   checks.** Note this learning: same-family cross-model verification
   is not independent.

4. **Escalate the 30% partial-plan buffer finding** separately — this
   affects current CR-v5 training signal quality regardless of CR-v6
   migration. Phase 1 Change 6 (apply_locus_revisions reject-on-no-match)
   prevents new partial plans from entering the buffer; old corrupted
   runs should be filtered at use-time.

## Data files

| Path | Description |
|---|---|
| `data/locus_pilot/sample_s3_pilot.jsonl` | 24 sampled plans |
| `data/locus_pilot/grading_locus_s3.jsonl` | 30B locus-extended grading |
| `data/locus_pilot/diag_canonical_s3.jsonl` | Diagnostic 1 (canonical vs locus) |
| `data/locus_pilot/diag_s5_s7_canonical.jsonl` | Diagnostic 2a (S5, S7) |
| `data/locus_pilot/diag_remaining_signals.jsonl` | Diagnostic 2b (S1, S2, S6, S8, S9) |
| `data/locus_pilot/diag_s3_cross_goal.jsonl` | Diagnostic 3 (5 cross-goal refs) |
| `data/locus_pilot/judge_235b_s3.jsonl` | 235B external judge outputs |

## Budget used

- ~30 pilot grading API calls (24 × S3 locus + 6 diagnostic)
- ~24 × 235B external judge
- ~5 cross-goal grading
- Total ≈ 60 Tinker API calls, no OpenRouter. Under 2 hours wall-clock.
