# D4v2 Decisions

Append-only log. Newest first.

---

### 2026-04-23: Grader gap pre-flight REVISED — GPT-OSS is actually fine for binary R_active; Phase 3 proceeds with default grader

**What the pre-flight showed** (`depth_audit/grader_gap.json`):

| milestone | Opus fab? | GPT-OSS | Qwen-235B |
|---|:-:|:-:|:-:|
| iter0_fresh | yes | yes (avoids) | yes (avoids) |
| iter1_rev | yes | no (violates) | no (violates) |
| iter2_rev | yes | no (violates) | yes (avoids) |
| iter4_rev | yes | no (violates) | yes (avoids) |
| iter5_rev | yes | no (violates) | yes (avoids) |

GPT-OSS agreement with Opus: 4/5 (80%). Qwen-235B agreement: 1/5 (20%).

**Correction to prior narrative**: the earlier "GPT-OSS is missing fabrication
62-88% of the time" conclusion came from comparing `rubric_buffer.jsonl`
mean-across-8-rollouts to Opus's audit of the 1 buffer_max plan. These are
different populations — the mean includes 7 rollouts that DON'T match the
milestone plan. When scoring specifically the buffer_max plan (the one Opus
audited), GPT-OSS and Opus agree on 4 of 5. On iter4, Phase 2's buffer
grades were [1×7, 0×1] mean=0.88 — the single "0" was indeed the buffer_max
plan GPT-OSS correctly flagged.

Qwen-235B is LESS aligned on this item — more lenient, calling the iter 2/4/5
plans "avoid fabrication" when Opus flags them. Cross-family doesn't
automatically mean stricter.

**Consequence**: Phase 3 proceeds without swapping binary grader.
`rubric_binary_use_alt_grader` flag retained in config for future
investigation but default False.

---

### 2026-04-23: Opus depth audit on Phase 2 milestones — CR plateaus at 17/40; Phase 3 motivated

**What ran**: 5 buffer_max milestones from `runs/2026_04_23_ger_cr_v1_smoke`
(pipeline 0.525 → 0.940) audited in parallel via 5 Claude subagents using
the standard `DEPTH_AUDIT_PROMPT` (4 dims × 1-10). See
`depth_audit/SUMMARY.json` and `analysis/ger_cr_v1_smoke_audit.ipynb`.

**Result**:

| iter | pipeline | grnd | Opus/40 | fab flag? |
|---:|---:|---:|---:|:-:|
| 0 | 0.525 | 3 | 14 | yes |
| 1 | 0.815 | 2 | 14 | yes |
| 2 | 0.880 | 3 | 17 | yes |
| 4 | 0.890 | 3 | 16 | yes |
| 5 | 0.940 | 3 | 17 | yes |

ρ(pipeline, opus) = 0.757. **Opus flagged fabricated citations on every
milestone**, including the 0.940 best plan (Kushner 1995, Dalalyan 2017,
Hernandez-Lobato 2017 all dropped as decorative).

**Initial concern** (later revised, see 2026-04-23 grader gap pre-flight entry
above): the `Avoids Fabricated Quantitative Precision` rubric item mean across
8 rollouts per iter ran 0.50-0.88, which I misread as "GPT-OSS missing
fabrication 62-88% of the time". Pre-flight on the specific buffer_max plans
showed GPT-OSS actually agrees with Opus 4/5 on the narrow "citation-attached
fabricated percentage" criterion — and the one milestone GPT-OSS binary-graded
"yes" (iter0) has no citations to fabricate percentages against anyway. So
GPT-OSS is fine for binary grading; Phase 3 can proceed without swapping
graders.

---

### 2026-04-23: GER-CR-v1 Phase 2 fork — trainer wiring only, no RL coupling

**What landed**: `src/co_scientist/grant_proposal/train_ger_cr_v1.py` forked
from `train_cr_v7.py`; new module `src/co_scientist/shared/rubric_grader.py`
(binary Yes/No grader mirroring `launch_plan_reward` async pattern); subagent
file-bus doc at `projects/grant_proposal_v2/scripts/rubric_gen_subagent.md`.

**In scope (done)**:
- Per-iter R_active binary grading on all rollouts → `record_grades` so
  `filter_and_truncate` operates on live std.
- Per-iter rubric-gen request emission (async file-bus to background Claude
  subagent). `best_vs_ref` contrast, default cadence=1.
- Every-5-iter reference-anchor check logged to `rubric_anchor_log.jsonl`
  (diagnostic only; no kill-switch).

**Deferred to Phase 3**:
- R_active aggregate → combined reward → RL advantages.
- α tuning / R_persist-vs-R_active ablation.
- Anchor drift kill-switch.
- Alt-reference consensus R_persist (needs Phase 1 Opus data first).

**Why Phase 2 first (not Phase 2+3 combined)**: single-variable diff
principle — mechanism correctness (rubric generation quality, filter
behaviour on live data, subagent file-bus latency) must be validated
before introducing a second variable (R_active → RL). Phase 2 is a no-op
for RL: `aggregate_reward` and `training_datums` construction are
untouched.

**Default toggle**: `rubric_evolution_enabled=False` → behaviour is
byte-equivalent to `train_cr_v7.py` (explicit regression verification
step planned in `plans/structured-snuggling-mccarthy.md` step 6).

---

### 2026-04-23: Path-Locking Test 2 falsified the hypothesis — R_persist reverted to option C (retain v8 signals)

**Finding**: TF-IDF cosine similarity between plans and reference proposal on the 3
v8 FoundOpt runs (276 plans after filter, iter≥3 + hard_gate passed) shows RL
moves policy LEXICALLY AWAY from reference, not toward it.

| Condition | sim_to_ref mean | Bootstrap 95% CI on Δsim vs B4 |
|---|:-:|:-:|
| B4 (no RL) | **0.0965** | — |
| MAIN_C3 (SDPO) | 0.0837 | Δ=−0.013 [−0.029, +0.004] (crosses 0) |
| MAIN_C4 (agg) | 0.0745 | Δ=**−0.022 [−0.038, −0.006]** (significant) |

Pre-registered threshold was `Δsim > +0.05 AND Opus(MAIN) < Opus(B4)` for
path-locking confirmation. Observed Δ is negative; Opus-part also fails (MAIN_C3
Opus=17.00 > B4=16.67 on n=3 raw total /20).

Supplementary: intra-run diversity (1 − mean pairwise sim) is slightly HIGHER for
MAIN runs (B4=0.872, C3=0.889, C4=0.882), so RL is not template-collapsing toward
reference either.

**Implications**:

1. **R_persist decision revised**: **option C (retain v8 signals as persistent
   layer)**, not option B (consensus from alt references). Reason: consensus was
   motivated by path-locking defense, which is no longer needed. Option C is
   zero-new-data-required and keeps v8 as the anchor that triangulation already
   validated (ρ(v8-Opus, depth-Opus)=+0.829).

2. **Supersedes this morning's option-B commitment** in the "GER-CR-v1 design
   approved" entry below. Alt reference collection cancelled. `consensus_rubric`
   build script not needed.

3. **Paper framing shift**: "RL Goodharts toward rubric-maxima templates, not
   reference imitation" replaces path-locking framing. Cleaner story — the
   evolving rubric in GER-CR-v1 counters rubric-hacking specifically, which is
   what the data says is happening.

4. **Decision queue item 1** (`Pilot Test 2 of PATH_LOCKING_FRAMING`) is **done**
   with negative verdict. `PATH_LOCKING_FRAMING.md` can be moved to
   `knowledge/archive/` with a header note recording this outcome.

**Caveats** (limits of this verdict):
- TF-IDF captures lexical overlap only. Semantic path-locking (same methodology,
  different words) would be missed. But if semantic-only, we'd expect neutral
  lexical sim, not significantly negative.
- Only 1 goal (01_foundopt). Other goals may differ, but 01_foundopt was the
  primary path-locking concern goal.
- Opus audit n=9 total (3 per run) is small; Opus-side of the verdict is
  low-power. Acceptable because the sim-side alone is decisive.

**Files**:
- Analysis notebook: `analysis/path_locking_test2/path_locking_test2_v1.ipynb`
- Executed with outputs: `analysis/path_locking_test2/path_locking_test2_v1_executed.ipynb`
- Results summary: `analysis/path_locking_test2/RESULTS_SUMMARY_v1.md`

**Revised GER-CR-v1 config** (replacing the row in plan E):

| Component | Revised | Was |
|---|---|---|
| R_persist | **v8 signal set (10 signals, Likert 1-5)** | consensus from 3 alt refs |
| Phase 1 alt-ref collection | **SKIPPED** | 4-8h Opus subagent job |
| consensus_rubric.py | **NOT NEEDED** | planned |

Phase 2 (rubric_buffer.py + rubric_gen_prompt.py + train_ger_cr_v1.py) and Phase 3
ablation structure unchanged.

---

### 2026-04-23: GER-CR-v1 design approved — 4 core decisions (synthesis from DR Tulu / OnlineRubrics / RaR / RLCF)

**Decision**: Commit to GER-CR-v1 (Grant Evolving Rubric + Critique-Revise) as next
direction. Full design synthesis in `~/.claude/plans/validated-humming-hippo.md`.
Four key choices selected after reading 4 related rubric-RL papers:

**1. R_persist source = consensus from 3 alt references per goal** (not minimal
floor, not reuse of v8 signals).

- Why: RaR Table 1 shows ref-grounded rubric +3.9 pt vs no-ref. But single-ref
  rubric encodes path-locking (our `PATH_LOCKING_FRAMING.md`). Consensus across
  3 refs retains grounding benefit while filtering path-specific items.
- Cost: ~4-8h Opus subagent work per goal to generate + human review.
- Alternatives rejected: (A) minimal structural floor was too weak per RaR
  evidence; (C) reuse v8 signals brings in known dead channels (G4, G10) and
  length bias.

**2. Phase 1 Path-Locking Test 2 runs first** as cheap falsification check.

- Why: If embedding analysis on existing v8 runs shows MAIN policy is NOT
  systematically closer to ref than B4, path-locking framing is retracted and
  R_persist defaults to option A (minimal floor). This avoids wasting Opus
  budget on alt refs if the motivating hypothesis fails.
- Approach: Use existing buffer data from 3 v8 FoundOpt runs (B4 / MAIN_C3 /
  MAIN_C4). Compute `mean_sim(plan, ref)` per condition; compare to intra-group
  diversity and Opus holistic scores.
- Decision threshold: `sim(MAIN, ref) - sim(B4, ref) > 0.05` → path-locking by
  RL confirmed → proceed to Phase 2. Otherwise → retract framing.

**3. Opus rubric generator called every iter** (~$20-40 / 35-iter run).

- Why: Matches DR Tulu + OnlineRubrics cadence. Same order of magnitude as the
  existing Opus eval agent. User `feedback_minimize_openrouter.md` requires
  consultation before large jobs — this was cleared at plan approval time.
- Alternatives rejected: Every-2-iter halves cost but halves rubric evolution
  speed (risky for 35-iter horizon); triggered-on-plateau may let Goodhart
  entrench before first evolution.

**4. Phase 3 full ablation = single goal (01_foundopt) × 3 seed**.

- Why: 4 conditions × 3 seeds = 12 runs. Mechanism study stays clean. NeurIPS
  reviewer pushback on single-goal generalization handled via held-out goal
  transfer test at end (cheaper than multi-goal training).
- Alternatives rejected: 3-goals × 3-seed (36 runs) is 3× cost for
  incremental paper-story strength; 2-goals × 2-seed weakens CI.

**Full design spec**: `~/.claude/plans/validated-humming-hippo.md` (sections E
and F). Phase structure: Phase 1 (Test 2 + alt refs + consensus script, this
week) → Phase 2 (GER-CR-v1 code + no-RL pilot, next week) → Phase 3 (4-way
ablation, ~5 days compute).

**Files to create (Phase 2)**:
- `src/co_scientist/shared/rubric_buffer.py` (DR Tulu std-filter buffer)
- `src/co_scientist/shared/rubric_gen_prompt.py` (DR Tulu + OnlineRubrics prompt mix)
- `src/co_scientist/grant_proposal/train_ger_cr_v1.py` (fork of train_cr_v7.py)
- `projects/grant_proposal_v2/dataset/goals/01_foundopt/alt_references/{1,2,3}.md`
- `projects/grant_proposal_v2/scripts/build_consensus_rubric.py`

**Literature gaps we commit to fill** (paper contribution):
1. Multi-seed + bootstrap CI (DR Tulu / OnlineRubrics / RaR HealthBench all
   single-seed)
2. Rubric drift kill-switch via reference anchor (no paper does this)
3. Non-LM-judge validation (citation verify + blind pairwise)
4. Systematic Goodhart attack surface diagnosis extended to evolving rubrics

---

### 2026-04-22: Post-v8 diagnostic — rank inversion + dead channels confirmed (anchors v9 rubric design)

Two findings from the `attack_surfaces_2026_04_22.ipynb` notebook work that
sharpen the v8 post-mortem and anchor v9 design decisions.

**1. Rank inversion at iter 10 (direct J3-level judge gap evidence)**

| Ranking basis | 1st | 2nd | 3rd |
|---|---|---|---|
| GPT-OSS buffer_max | C3 (0.875) | B4 (0.850) | C4 (0.815) |
| Opus length-normalized total /40 | **B4 (17)** | **C3 (16)** | C4 (14) |

C3 and B4 **swap position** between the two judges. Concrete rank inversion
observed within 10 iters, even before the pre-registered J1-J3 bootstrap
can be populated (audit N=2-3/run too sparse). Direction of disagreement:
GPT-OSS rewards the RL run that Opus says is worse.

**2. Dead channels (G4 + G10) — 22% of v8 reward weight contributes no gradient**

At final iter, variance of each signal across the buffer:

| Signal | Weight | Variance condition | Mechanism |
|---|---:|---|---|
| G4_focus | 0.16 | var < 0.25 on all 3 v8 runs | Rubric clusters most plans at score 3 (middle table branches too broad) |
| G10_approach_coverage | 0.06 | var < 0.25 + saturated at 5.0 on all 3 | "Any method named" counts as specific → ceiling hit by all plans |
| **Combined** | **0.22** | | **0.22 of reward weight is dead weight** |

Leave-one-out analysis on MAIN_v8_C3 at final iter shows G6_reasoning_depth
carries 22.4% of effective weight share — when G4/G10 are dead, the gradient
signal concentrates onto G6 passively. This is NOT the pre-experiment worry
("G4 at weight 0.16 will dominate"), but its complement: **dead channels
force load onto whichever signal still has variance**.

**3. Implication for v9**

v8 effectively operates on **~78% of its nominal weight budget**, with
G11/G12/G13 (length-correlated, total weight 0.32) dominating the live
portion. This is the mechanistic explanation for the earlier finding
"length hacking routes 100% via G11/G12/G13" (commit 38c50a4): dead
channels force all reward growth through the length-correlated live
channels.

v9 must therefore address BOTH attack surfaces simultaneously:
- **G4/G10** (dead) — prompt rewrite to restore variance: G4 requires
  "composition logic" for top scores (redistributes middle-clumped
  distribution); G10 tightens "specific" definition (named procedure +
  named dataset + named measurable output) to break the ceiling.
- **G11/G12/G13** (length-hacked) — prompt rewrite to verifiable tuples:
  each counted item must be a fully-specified (named + cited + used)
  tuple rather than a list entry, making fabrication and padding
  detectable.

Signal weights remain at v8 values (one variable per diff). Full v9 design
in `.claude/plans/quirky-herding-emerson.md` (approved 2026-04-22).

**Files**:
- `projects/grant_proposal_v2/analysis/attack_surfaces_2026_04_22.ipynb` —
  source of variance/LOO analysis
- `projects/grant_proposal_v2/analysis/odin_length_residual.json` —
  length-regression R² pooled across 1000+ buffer entries

---

### 2026-04-22: v8 runs killed at iter 13-14 — three distinct failure modes confirmed across v7/v8

**Decision**: Stop all 3 v8 FoundOpt runs (B4, MAIN_C3, MAIN_C4) at iter 13-14 after buffer_max plateau. Do NOT extend. Move to v9 rubric redesign.

**Trigger**: All 3 v8 runs had buffer_max flat for 3-4 consecutive iters by iter 13 (B4=0.850, C3=0.875, C4=0.815). Running 20+ more iters would not produce new quality signal, and v7 C4 precedent showed 28 extra iters after buffer saturation produced zero change.

**Ran 3 diagnostic tests on both v7 C2/C3/C4 (completed runs) and v8 B4/C3/C4 (killed at iter 13-14)**:
1. Per-signal trajectory: length-correlated (G11+G12+G13) vs discriminating (G4+G6+G9) Δ
2. Opus pairwise substance eval (iter 5 vs iter-final best plan, length-blind)
3. Plan text diff (word count, structural similarity, artifact cues)

**Finding: three distinct Goodhart failure modes**:

| Condition | Qwen/v8 reward Δ | Opus substance | Failure mode |
|---|---|---|---|
| v7 C2 (B4 Qwen self-grade) | +0.02 tiny | **clearly better** | **none** — CR pipeline works |
| v7 C3 (SDPO Qwen) | **+0.19 huge** | slight gain | **hallucination hacking** — fake theorems, fabricated arXiv, +5 "Theorem" mentions in 1% word Δ |
| v7 C4 (agg Qwen) | +0.03 tiny | **TIE identical** | **buffer saturation** — iter 6 plan duplicated to iter 34 (99.92% text similarity) |
| v8 B4 (GPT-OSS) | +0.26 | slight gain | **length hacking** — G11/G12/G13 Δ=+1.29, discrim Δ=+0.08 |
| v8 C3 SDPO (GPT-OSS) | +0.28 | slight gain | length hacking (discrim Δ NEGATIVE at −0.17) |
| v8 C4 agg (GPT-OSS) | +0.29 | clearly better | length hacking (+47% chars) |

**Key insight**: Grader choice routes Goodhart into different axes but doesn't close it.
- Lenient Qwen grader → policy learns to hallucinate rigor-looking specifics
- Strict GPT-OSS grader → policy learns to pad length (G11/G12/G13 still length-correlated)

**Length/Discriminating Δ ratio**:
- v7: 0.94-5.5 (grader gives discrim credit, Goodhart has multiple channels but a truer reward channel exists)
- v8: 3.5-∞ (grader strict on discrim, Goodhart routes 100% to length)

Cross-family grader is **necessary but not sufficient**. Length decorrelation must happen at the prompt-level.

**v9 design queue** (see `analysis/v7_vs_v8_comparison.md`):
1. Rewrite G11/G12/G13 as **count-based** (e.g., "# of derived equations" not "how formal")
2. Hard length cap on grader input (truncate >2000 words)
3. ODIN-style length residual head (fit offline on v8 buffer data)
4. Exploration diversity: critique-driven new-direction suggestions, not polish-existing
5. **SDPO hallucination hacking is RL-specific and independent of rubric** — needs verifiability at grader level (citation ground-truth check)

**Hypothesis verdicts** (from earlier DECISIONS predictions):
- H1 (B4_v8 Opus ≥ 19/40): **likely no** — stopped at iter 10 with 17/40 length-normalized; plateau before target.
- H2 (MAIN_v8 ≥ B4_v8): **no clear winner** — all 3 v8 plateaued at similar Opus quality (14-17/40 length-normalized) despite different RL algorithms.
- H3 (null — RL architecture problem): **partially supported** — v7 C4 and v8 all plateau early. But CR pipeline (B4) DOES improve substantively when grader gives discrim credit (v7 C2 case).

**Files**:
- `projects/grant_proposal_v2/analysis/v7_vs_v8_comparison.md` — full test results
- `projects/grant_proposal_v2/runs/2026_04_22_v8_foundopt_{B4,MAIN_C3,MAIN_C4}/` — v8 data frozen at iter 13-14
- `*/depth_audit_mid.jsonl` — Opus eval at iter 5/10 (length-aware + length-normalized)

---

### 2026-04-22: Single-signal saturation — variance-collapse mechanism, NOT weight-concentration

**Finding** (from `analysis/attack_surfaces_2026_04_22.ipynb`, Suite S): saturation IS present, but via a different mechanism than assumed. The original worry — "G4/G6 at weight 0.16 will dominate" — is not what the data shows.

**S3 / S4 (weight-concentration check) — NOT confirmed**:
- S3 `ρ(clipped_at_4, unclipped)` > 0.7 on all 6 runs → capping any signal at 4/5 would not materially change plan ranking. **No plan is winning by pushing a single signal to max.**
- S4 max(median|Δ_i| / median aggregate) < 0.40 on all runs → no single signal carries >40% of aggregate. **Weights are doing their job as designed.**

**S2 (variance collapse) — ✅ confirmed on 5/6 runs**:

| Run | # signals with var < 0.25 at final iter | verdict |
|---|:-:|:-:|
| B4_v8 | 0 | ❌ |
| MAIN_v8_C3 | 2 | ✅ |
| MAIN_v8_C4 | 2 | ✅ |
| B4_v7 | 3+ | ✅ |
| MAIN_v7_C3 | 3+ | ✅ |
| MAIN_v7_C4 | 3+ | ✅ |

**Interpretation**: The saturation mode in this setup is not "one signal dominates aggregate" — it's **"multiple signals collapse to the same score across all plans"**. These are *dead gradient channels*: the grader cannot discriminate plans on that dimension, so the signal contributes 0 useful gradient regardless of its weight.

**Mechanism hypothesis** (to confirm at final iter): Dead channels are likely the **easy-to-saturate signals** where every competent plan hits 4 or 5 (e.g., G1 problem_specificity, G10 approach_coverage). Weight ≠ gradient contribution once variance is 0.

**Caveat for v8 verdict**: MAIN_v8_C3/C4 showing S2 ✅ at iter 0-2 is likely small-sample noise (n=20 per run, n≤8 per iter). Re-verify at final iter (35). B4_v8 having 0 dead channels at this stage is consistent with "MAIN RL amplifies variance collapse" hypothesis — but need data.

**v9 implications**:
1. **Per-iter variance diagnostic in reward logging** — flag signals where within-iter variance drops below 0.25 for ≥3 consecutive iters, those are wasted weight
2. **Active rebalancing**: if a signal collapses, redistribute its weight to discriminating signals (ties into UCB-adaptive weights from earlier discussion)
3. **Rubric-level fix**: for signals that systematically saturate (e.g., G1 problem_specificity at 5/5 for all serious plans), **tighten the rubric** so the 4→5 gap is meaningful, OR remove the signal entirely

**What NOT to conclude**: Don't conclude weights are wrong (S4 rules this out). Don't conclude single-signal Goodhart is the risk (S3 rules this out). The design concern is discrimination power per signal, not allocation across signals.

---

### 2026-04-22: Judge gap (Suite J) — INSUFFICIENT DATA, pending mid-run audits

**Status**: J1 (ρ drop over iters), J2 (bootstrap CI non-overlap), J3 (top-5 rank inversion) all return `None` / `—` in the verdict table. Cause: `depth_audit_mid.jsonl` and `opus_eval_log.jsonl` files do not yet exist for any of the 3 v8 runs (mid-run Opus audit script `Task 17` has not been wired into launch yet).

**Baseline to compare against** (frozen from `analysis/grader_panel_v8/TRIANGULATION_SUMMARY.md`):
- ρ(v8-Opus aggregate, depth-Opus total) = **+0.829** on 15 independent plans, pre-RL
- This is the iter-0 anchor. J1 verdict triggers if this ρ drops by ≥0.15 at final iter.

**What to do on re-run (after runs finish + mid-run audits fire)**:
1. Mid-run audit script must write to `depth_audit_mid.jsonl` with schema `{iter, plan_idx, depth, methods, feasibility, grounding, total}` to be parsed by notebook's `load_audits()` helper
2. Re-execute notebook — J1/J2/J3 cells will populate automatically
3. If verdict fires on any MAIN run: judge gap is **mechanistically confirmed**, not just hypothesized from SDPO precedent

**Pre-registered decision points**:
- If J1 ≥ 0.15 drop on MAIN but not B4 → Qwen-policy learns to game GPT-OSS specifically (classic RM overoptimization, maps to Wolf et al. 2025 iterated-RLHF findings — they recommend SFT-reset per iter to mitigate)
- If J1 drop on BOTH MAIN and B4 → GPT-OSS-Opus gap is training-agnostic (e.g., caused by buffer's increasing within-distribution concentration), which implies the 120B judge is structurally insufficient, not "gameable"
- If CI non-overlap (J2) fires while Δρ < 0.15 → low-power signal, weight J3 more heavily

**Why this matters for v9 design**: which specific judge-gap failure mode confirms determines whether the v9 fix is (a) judge refresh / distillation, (b) policy-side KL tightening, or (c) replace 120B judge with higher-tier model. No single fix dominates without knowing the mode.

---

### 2026-04-22: Length bias is encoded in v8 signal design itself (not a RL artifact)

**Finding** (from `analysis/attack_surfaces_2026_04_22.ipynb`, Phase 0 — before v8 runs complete): grader length bias shows up at **iter 0 baseline**, not only after RL. v8 signal redesign did not remove it.

**Cross-run Spearman ρ(word_count, signal) on buffer data**:

| Signal | B4_v8 | MAIN_v8_C3 | MAIN_v8_C4 | B4_v7 | Pattern |
|---|:-:|:-:|:-:|:-:|---|
| ρ(len, aggregate) | 0.67 | 0.66 | 0.47 | 0.72 | L1 threshold (0.5) exceeded on 5/6 runs |
| G12 formalism | 0.73 | 0.75 | 0.57 | 0.80 | length-correlated across all runs |
| G11 evidence_rigor | 0.75 | 0.37 | 0.48 | 0.58 | length-correlated |
| G13 risk_awareness | 0.69 | 0.18 | 0.80 | 0.74 | length-correlated |
| G3 technical_evidence (v7) | — | — | — | 0.77 | removed in v8 (correctly) |

**Interpretation**: G11 + G12 + G13 together carry weight 0.32 in v8. When ρ(len, G_i) ≈ 0.7–0.8, these signals are effectively length measurements in disguise. "Longer plan → more rigorous-sounding → higher score" without actual quality gain.

**Mechanism** (ODIN, ICML 2024): judge sees "more text about X" and interprets it as "more rigorous about X". Without length-decorrelation at the RM side, RL-driven length hacking is **inevitable regardless of RL algorithm** (SDPO, aggregate, or B4).

**L2 length-drift verdicts negative, but interpretation subtle**: v7 MAIN_C3 median length grew +70% while B4_v7 grew +26%. Both drift, so "MAIN-vs-B4 differential growth" threshold not met — but this confirms length drift is a **baseline grader property**, not a MAIN-specific artifact.

**Implication for v9 design**:
1. **ODIN-style length-residual head** on the 120B grader output: predict score from length alone, subtract before using as reward
2. **Rewrite G11/G12/G13 rubric prompts** to request count-based evidence (# of specific formulas / # of named failure modes / # of measurable milestones) rather than qualitative "how rigorous" phrasing — count is length-bounded, qualitative is length-unbounded
3. **Length-controlled eval**: depth_audit_v2 Opus prompt must explicitly instruct "ignore differences in length", and pair-wise comparisons in Opus eval must be length-matched (±15%)

**What NOT to conclude**:
- This finding does NOT invalidate v8 — the rubric-triangulation ρ=+0.829 still holds, meaning aggregate v8 scores track depth_audit_v2 at the plan level. Length is one contributor to both.
- This finding does NOT predict MAIN_v8 will fail — the question "does RL amplify pre-existing length bias" is separate and requires the final-iter verdict table (L2/L3) once the runs complete.

**Follow-up**:
- After v8 runs finish, re-run notebook — populate J1-J3 (judge gap) and L3 (ODIN R² comparison) once `depth_audit_mid.jsonl` exists
- If L2 post-run shows MAIN-vs-B4 differential length growth (was flat at iter 0-2 due to small sample), length hacking IS amplified by RL
- Decide v9 rubric revisions based on which signals remain length-correlated after Opus-judge ablation

**Files**:
- `projects/grant_proposal_v2/analysis/attack_surfaces_2026_04_22.ipynb` — the diagnostic notebook (13 experiments across 3 attack surfaces)
- `.claude/plans/generic-questing-dijkstra.md` — experimental plan with pre-registered thresholds

---

### 2026-04-22: Grader switched to GPT-OSS-120B after rubric triangulation

**Decision**: Override prior Qwen3-235B selection. Use **GPT-OSS-120B** as the v8 grader for B4_v8 / MAIN_v8 experiments.

**Context**: To guard against "v8-rubric circular reasoning" (a grader might just be learning v8's idiosyncrasies rather than real quality), ran 15 independent Opus subagents (no halo — one plan each) with a rubric-agnostic holistic `depth_audit_v2` prompt (4 dimensions × 1-10). Then re-correlated every tinker grader against BOTH v8-Opus and depth-Opus.

**Key results** (plan-level aggregate Spearman ρ):

| Model | ρ vs v8-Opus | ρ vs depth-Opus |
|---|:-:|:-:|
| **gpt_oss_120b** | **+0.971** | **+0.878** |
| qwen3_235b | +0.756 | +0.794 |
| gpt_oss_20b | +0.878 | +0.722 |
| deepseek_v3_1 | +0.694 | +0.755 |

Also: **ρ(v8-Opus, depth-Opus) = +0.829** → v8 rubric is NOT circular. It captures real quality signal, validating the v8 redesign itself.

**Why override**:
- GPT-OSS-120B wins on BOTH rubrics → not rubric-specific, more robust grader signal
- Qwen3-235B drops from #1 (per-signal mean ρ=0.741) to #2 when triangulated → over-indexed on v8-rubric surface features
- GPT-OSS-120B is cross-family from Qwen policy → lower self-preference bias risk
- Cheaper: ~$0.44/sample vs ~$1.70/sample for Qwen3-235B

**Caveat**: per-signal ρ (original panel test) had Qwen3-235B slightly ahead (0.741 vs 0.732). Plan-level aggregate ρ is the more RL-relevant metric because the training reward is aggregate.

**Validation plan**: at iter ~5 / ~10 / ~25, run depth_audit Opus on best-buffer-plan to detect v8 score drift from holistic quality.

**Files**:
- `analysis/grader_panel_v8/TRIANGULATION_SUMMARY.md` — full writeup
- `analysis/grader_panel_v8/grades/opus_depth_audit.jsonl` — 15 independent Opus depth grades
- `analysis/grader_panel_v8/analysis/triangulation.json` — correlation matrix
- `analysis/grader_panel_v8/scripts/triangulate_rubrics.py` — computation

**Implication**: B4_v8 / MAIN_v8 configs should use `grader_model_name=openai/gpt-oss-120b`. Keep Qwen3-235B as secondary cross-check.

---

### 2026-04-22: Rubric v8 + confidence-gated RL (systematic rubric redesign)

**Decision**: Replace D4-v7 (12 signals) with D4-v8 (10 signals), add multi-sample
grader confidence and RL filter, test whether Qwen self-grading becomes viable.

**Context**: SDPO experiment demonstrated Qwen 30B as same-model grader
systematically Goodharts under RL. Options considered:
- (a) Switch to Opus/GPT-5.4 grader — ruled out, novelty = 0 vs RLAIF 2022
- (b) Fix rubrics so Qwen self-grading becomes robust — this decision

**Why this might work**: systematic review showed several v7 signals are
Goodhart-able (G3 rewards hallucinated citations; G5 saturated at 99.8% 5/5;
G12 rewards equation-shaped patterns). Discriminating signals (G4, G6, G9,
G13) share features: two-dim gates (count + justification), inverse counting
(red flags), or non-counting qualitative. v8 emphasizes these.

**Confidence rationale**: *Cycles of Thought* (2024) and *Confidence Improves
Self-Consistency* (2025) show sampling variance is better-calibrated than
verbalized confidence for LLM judges. v8 uses `1 - range/(score_max-1)` over
3 samples at T=0.3 instead of asking the grader to verbalize confidence
(which EMNLP 2023 showed is systematically overconfident).

**Testable prediction**:
- If B4_v8 Opus ≥ 19/40 → rubric fix alone improves frozen grading
- If MAIN_v8 Opus ≥ B4_v8 → RL stops Goodharting under robust rubric
- Either outcome supports "policy=grader is viable with right rubric design"

**Files changed**:
- NEW `src/co_scientist/shared/grant_rubric_v8.py`
- `src/co_scientist/shared/grant_signal_reward.py`: added `confidence_over_repeats()`
- `src/co_scientist/grant_proposal/train_buffer_ttt.py`: `per_signal_confidence` field, updated `collect_gradient_signals`
- `src/co_scientist/grant_proposal/train_cr_v7.py`: `min_grader_confidence` config, per-signal filter in REINFORCE, `_confidence_stats` logging

---

### 2026-04-22: Shelve G3 ablation — not the priority

**Decision**: Start G3-disabled B4 experiment but cancel mid-run. Not pursuing isolated G3 removal.

**Why**: Experiment confounded — ran as `full_critique` mode while baseline C2 was `scores_only`. Can't compare directly. More importantly, G3 alone is not the root cause — rubric review (see memory) suggests systemic issue with counting-based rubrics across multiple signals (G1/G2/G5/G8/G10/G12 all gameable). A broader rubric redesign is more valuable than single-signal ablation.

---

### 2026-04-22: Create D4v2 investigation workspace

**Decision**: Fork D4 into D4v2 for clean investigation of Qwen-Opus quality inversion.

**Context**: D4 accumulated 58 runs with mixed signal sets, critique modes, RL methods. Configuration space too tangled for systematic analysis. D4v2 provides clean workspace with structured audit of all prior work.

**Carried forward from D4**: dataset (symlink), knowledge docs, SDPO experiment analysis.
**Left in D4**: all 58 runs (too large to copy), legacy analysis, old signal versions.

---

### 2026-04-22: Prioritize investigation over new experiments

**Decision**: Before running more experiments, systematically answer: is the ceiling from model capability, reward design, or critique mechanism?

**Rationale**: We've tried 7+ RL methods (MAIN, GAPO, SDPO, aggregate, fresh-only, CR-only, scores-only RL). All Goodhart. Trying more RL methods without understanding the root cause is wasted compute.

**Investigation priority**: (1) text comparison, (2) per-signal Goodhart, (3) 235B control, (4) multi-goal check, (5) cross-family reward.

---

### 2026-04-22: Qwen3-235B chosen as grader based on panel test vs Opus

**Decision**: Switch grader from Qwen3-30B-A3B to Qwen3-235B-A22B-Instruct-2507 for v8 experiments.

**Context**: Ran 9-grader panel test (Opus 4.7 + 8 tinker models) on 15 FoundOpt plans × 10 v8 signals. See `analysis/grader_panel_v8/FINAL_SUMMARY.md` for full results.

**Ranking by mean per-signal Spearman ρ with Opus**:
1. Qwen3-235B: **0.741** (87% within-1, 150/150 complete)
2. GPT-OSS-120B: 0.732
3. GPT-OSS-20B: 0.674
4. DeepSeek-V3.1: 0.563
5. Qwen3-30B (current): 0.399
6. Qwen3-4B: 0.385
7. Llama-3.1-8B: 0.140 (near-random)
8. Kimi-K2-Thinking: killed at 13/150 (reasoning latency too high)

**Takeaways**:
- Qwen-235B correlates with Opus much better than Qwen-30B (0.74 vs 0.40) — same family but bigger helps.
- GPT-OSS-120B nearly ties 235B — viable cross-family alternative.
- Signal-level: all reasonable graders agree well on G11/G12/G13 (rigor/formalism/risk, ρ≥0.9) but disagree on G1/G9/G10.
- Current Qwen-30B grader validates user's concern: self-model grading has low correlation with ground truth.

**Implication**: B4_v8 / MAIN_v8 experiments should use `grader_model_name=Qwen/Qwen3-235B-A22B-Instruct-2507`.
