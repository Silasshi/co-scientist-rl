# D5 Decisions

Append-only log. Newest first.

## 2026-04-29 PM — v9 Layer 1 + Layer 2 pilots completed; F18 filed; KL anchor verdict

**Context**: After ABC experiments verified H16-2 (incentive misalignment) as dominant ceiling mechanism (commit d0d94ef), ran Layer 1 (verifier-grounded reward) and Layer 2 (KL anchor toward SFT'd π_oracle) pilots in same session per pre-registered Option A path. Both completed full 16 iter.

**Decision 1 — L1 verifier-grounded path is insufficient alone**: L1 peak only 22.00 (below G+ 22.38), with mid-pilot cliff iter 9-12 (low 16.50). Custom verifier on eval plans confirms L1 Goodharts: eq=5.00 sustained but cite drops to 2.88 and emp to 0.00. Sparse per-token bonus on equation matches creates a Goodhart-able surface that the policy gradients toward at the cost of citations and empirical anchors. **Layer 1 alone NOT a deployment-grade fix.**

**Decision 2 — L2 KL anchor is a stabilizer, not a peak booster**: L2 peak 23.38 ≈ C+ peak 23.50 (within 0.12pt), but L2 maintains 23.0+ across final 3 iter while C+ cliffs to 14.75 by iter 12. Mean over 16 iter L2=20.91 highest of all cells. Multi-axis grounding shows L2 is ONLY cell sustaining all 3 axes (eq=5.62, cite=5.75, emp=0.62 at peak). σ_v8 frozen+slim baseline (25.25) still unbeaten. **F18 filed**: distributional grounding ≠ deep grounding; U3 originality stays at 2 throughout — model mimics oracle surface form (notation, citation graph, index schema) but does NOT internalize derivations.

**Decision 3 — Paper framing pivot to "stability + balance" not "ceiling"**: previous plan's F.1 success criterion (peak ≥ G+ peak + 1.5 = ≥23.88) was not met by either pilot. But L2's late-iter stability + multi-axis grounding is the real story. Honest paper claim: "KL anchor toward SFT'd π_oracle stabilizes late-iter performance and prevents axis trade-offs, but does not raise the ceiling of plan quality."

**Decision 4 — Production checkpoint policy update**: G+ iter 4 still the only pairwise-corroborated v8 cell. L2 v9_kl_anchor iter 14 (23.38) needs pairwise corroboration before promotion. Pending: pairwise L2-iter-14 vs G+-iter-4 (~$10) + L2-iter-14 vs σ_v8 (~$10).

**Decision 5 — Next direction: Self-review (Qwen3-30B as reviewer instead of Opus)**: tabling Layer 3 derivation-chain reward research path. Instead, next session will rerun all completed v8 + v9 ablations with Qwen3-30B-A3B as the audit reviewer (replacing Opus 4.7). Goal: characterize self-review vs external-review divergence as paper-grade methodological finding, AND avoid the cost/rate-limit fragility of Opus-driven evaluation. Plan filed in `knowledge/current/PLAN_self_review_ablation.md`.

## 2026-04-28 PM — Phase 2 v8-d5sdpo grid stop + ABC redirect (F15 + F16)

**Context**: 7 of 13 v8 cells run (base, G, G+, E, C+, F+, A) on `train_mu_v8_d5sdpo.py`. G+ pairwise corroboration showed 8-0 vs σ_v8, 8-0 vs μ-v4, 7-1 vs G — but **3-5 LOSING vs v7-opd-full**. The pre-registered "v8 beats v7-opd-full" headline is FALSIFIED.

**Decision 1 — STOP Phase 2 grid early (skip remaining D, B, B+, F, C cells)**: ROI for completing the factorial decomposition is low because (a) the pairwise verdict against v7-opd-full already invalidates the original "v8 supremacy" narrative; (b) the audit /45 lift over σ_v8 is corroborated only at the G+ peak iter, not across cells; (c) single-seed throughout means even completed cells can't deliver bootstrap-significant main effects. Saved cost: ~$120 + 24 hr.

**Decision 2 — Pivot to oracle-transfer mechanism diagnosis (F16 / ABC plan)**: instead of completing the factorial, use this session's runs as **diagnostic data** to test whether the lift involves model-side learning of oracle content at all, OR is purely structural reweighting. Expected outcome: paper-grade negative-result-with-mechanism narrative, more defensible than partial factorial.

**Decision 3 — F16 hypotheses + ABC experiment plan**: code-level reading of `train_mu_v8_d5sdpo.py:393-473` confirmed teacher = current LoRA-adapted student (self-distillation, NOT external-strong-teacher distillation). Oracle slim verified to contain all formulas critic flags as missing. ABC experiments (`knowledge/current/EXPERIMENT_PLAN_oracle_transfer_ABC.md`) are the minimum-cost falsifiers for three hypothesized bottlenecks: H16-1 retrieval, H16-2 incentive, H16-3 LoRA bandwidth. Total ~$43 + 7 hr.

**Decision 4 — Production checkpoint policy stays at G+ iter 4** (the only pairwise-corroborated v8 cell), NOT C+ iter 6 (highest audit peak 23.50 but no pairwise). Promotion to C+ requires C+ vs σ_v8 + C+ vs G+ pairwise (~$10).

**Decision 5 — Document falsified pre-registered claim**: G+ vs v7-opd-full = 3-5 (v7-opd-full wins) is recorded explicitly in F15 and RUN_REGISTRY so future doc updates don't accidentally restore the falsified headline.

---

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)** — historical entries below freely use
> "SDPO" / "OPD" / "option (c) HER" referring to D5 trainer behavior. After
> paper-grade verification (WebFetch of arxiv 2601.20802 Hübotter and 2601.18734
> Zhao):
> - "SDPO" in pre-2026-04-27 D5 entries almost always means the **D5 in-house
>   off-policy IS-loss variant** (μ-v2/v3/v4/v6/κ-v1 family — TEACHER samples;
>   STUDENT lp recomputed; importance_sampling loss). This is structurally
>   distinct from canonical Hübotter SDPO (student on-policy + KL).
> - "option (c) HER from Hübotter 2026" in pre-2026-04-27 entries was a D5
>   internal speculation — **the Hübotter paper does NOT discuss any such
>   variant**. Read past mentions as referring to the D5 in-house off-policy
>   IS-loss variant.
> - "OPD" in pre-2026-04-27 entries was a loose synonym for canonical on-policy
>   distillation; the closest published comparable is OPSD (Zhao 2601.18734)
>   sampled-token policy-gradient variant (their Table 3).
>
> Read historical entries through this lens. For paper-grade attribution see
> [`knowledge/current/CANONICAL_NAMING_REFERENCE.md`](knowledge/current/CANONICAL_NAMING_REFERENCE.md);
> for per-run setup truth see [`knowledge/current/RUN_REGISTRY.md`](knowledge/current/RUN_REGISTRY.md).

---

## 2026-04-27 PM: Phase 3 F8 REVISED (v2) — original 7-1 STRONG headline FALSIFIED on patched + fresh plans

**Triggered by**: ML-scientist subagent rule (`feedback_ml_scientist_subagent_before_decision.md`)
+ quality-first rule (`feedback_quality_first.md`). Routine BoN sanity check (Step 2a)
revealed two issues that invalidated F8 v1.

**Findings (Steps 2a/A/B/C/E)**:

| Step | Cost | Finding |
|---|---:|---|
| 2a | $4 | Within-instance σ_within = 1.17 (inst 0 K=8); production τ_v4 single-shot for inst 0 = 27 vs K=8 reseed mean 24.88 (lucky high draw); CITATION BUG discovered (4/8 judges flagged "Math 11"/"Methodology 24" as fake citations) |
| A | $0 | Patched `_KAPPA_PLAN_FOOTER_V4` with explicit anti-leakage instruction |
| B | $4 | τ_v4_clean (8 inst × 1 plan, PATCHED prompt) cross-instance mean **24.50** (vs claimed F8 v1 **26.50**, drop 2.0); σ across 8 = 2.18; 95% CI on mean [22.96, 26.04] CONTAINS σ baseline 25.25; tag leakage reduced ~85% (18 across 8 vs ~120 orig) but not eliminated |
| C | $2 | Inst 2 K=4 σ_within = 0.50 (vs inst 0's 1.17, ratio 2.3× heterogeneous); inst 2 production single-shot was 29 (highest of 8) vs K=4 reseed mean 23.50 (gap 5.5 — even bigger lucky-draw than inst 0) |
| E | $32 | Pairwise τ_v4_clean vs σ_v4 rerun (8 pairs, seed=52, position-randomized 4-4): **4-4 TIE** — clean falsification of Step 2.5's 7-1 STRONG |

**F8 v2 reframe** (full doc: `paper_materials/findings/F8_phase3_distillation_pathway.md`):

- **Headline change**: "τ_v4 beats σ 7-1 STRONG" → "τ_v4_clean STATISTICALLY COMPARABLE to σ baseline (4-4 TIE); 3 failure modes characterized"
- **Three characterized failure modes** (paper-publishable as negative-result-with-mechanism):
  1. Self-review degeneration (M7, Smoke A 8-0 — UNCHANGED, still load-bearing)
  2. **NEW**: Single-shot evaluation inflation at σ_within ≈ 1.17 (Step 2.5 7-1 → Step E 4-4 demonstration)
  3. **NEW**: Citation-tag leakage from multi-round distillation (~85% fixable at plan-prompt level, full fix needs distillation prompt redesign)
- **Pairwise discipline lesson** (M4 update needed): pairwise on n=8 single-shot is a SAMPLING claim, not a TECHNIQUE claim. Reliable Δ < 2/45 detection needs K=4-8 plan reseeds per instance.
- **Phase 3 status**: COMPLETE (revised). κ_opus deferred (lower priority now that τ_v4 ≈ σ; selection-bias math says BoN unlikely to push past μ-v4 28.00 ceiling).

**Code changes**:
- `kappa_prompts_v1.py:_KAPPA_PLAN_FOOTER_V4` — added Citation hygiene block
- `tau_bon_sanity_v1.py` (NEW) — K=N plan reseed on existing distillations
- `step_e_pairwise_v1.py` (NEW) — focused 1-matchup pairwise rerun

**Run dirs added**:
- `runs/2026_04_29_tau_v4_bon_sanity/` + `_audit/` — Step 2a
- `runs/2026_04_30_tau_v4_clean/` + `_audit/` — Step B
- `runs/2026_04_30_tau_v4_bon_sanity_inst2/` + `_audit/` — Step C
- `runs/2026_04_30_step_e_pairwise/` — Step E

**Total cost this session**: ~$42 ($4 + $4 + $2 + $32). Single-day full Phase 3 re-validation.

**What is NOT decided yet** (open for next session):
- Whether to re-run σ baseline with K=4-8 per instance (~$4-8) to nail down whether σ population mean is ≈ 25.25 (claim holds) or higher (would mean τ_v4 slightly underperforms)
- Whether to fix distillation prompt to eliminate tag artifacts at source (vs partial plan-prompt patch)
- Whether κ_opus full smoke is worth $60-100 given τ_v4 ≈ σ now (selection-bias math suggests no, but methodology could close Phase 3 cleanly)

---

## 2026-04-27: Phase 3 distillation pipeline finding LOCKED (F8) + Smoke A self-review dispelled + prior κ smoke INVALIDATED

> **⚠️ τ_v4 / F8 portion of this entry SUPERSEDED 2026-04-27 PM by F8 v2 — see top of file (DECISIONS.md:7-44).** On patched + fresh plans (n=8), τ_v4_clean cross-instance mean = 24.50 (not 26.50, drop 2.0 from lucky-draw artifact); Step E pairwise rerun (8 pairs, seed=52, position-balanced) returned 4-4 TIE (not 7-1 STRONG). Only the τ_v4 portion is retracted; the **Smoke A self-review dispelled** finding (M7) and **κ-v1 smoke INVALIDATED** finding remain valid.

**Phase 3 paper-grade finding** (full doc: `paper_materials/findings/F8_phase3_distillation_pathway.md`):

**Headline**: Inference-time 3-round distillation pipeline (τ_v4) achieves 26.50/45
vs σ Phase 2 baseline 25.25 — ~~**first non-fine-tuned 30B variant to beat σ**~~ (FALSIFIED — see DECISIONS.md:7-44).
~~Cross-validated 7-1 STRONG over σ_v4 (slim oracle + same plan_v4 prompt) via Step 2.5 pairwise.~~ **FALSIFIED 2026-04-27 PM** — Step E pairwise rerun (8 pairs, seed=52, position-balanced, on patched + fresh plans) = 4-4 TIE; Step 2.5's 7-1 STRONG was lucky-draw artifact.

**Trajectory** (4 prompt iterations on τ): 18.12 → 19.00 → 20.38 → **26.50** (production single-shot;~~F8 v1 endpoint~~ — REVISED to τ_v4_clean cross-instance mean **24.50** on patched + fresh, see DECISIONS.md:7-44).
Plan-prompt v4 (mandatory T1+T2+T3 instructions) is the dominant variable;
~~distillation pipeline adds ~+0.75 absolute increment (pairwise STRONG corroborated).~~ **REVISED**: distillation pipeline net contribution at single-shot n=8 NOT detectable above noise floor (F8 v2 § Limitations 5).

**Step 2.5 pairwise** (3 close-cluster matchups, 24 Opus pairwise judges, seed=51):
| Matchup | Verdict |
|---|---|
| τ_v4 vs σ_v4 | ~~τ_v4 wins **7-1 STRONG**~~ → **4-4 TIE per Step E rerun** (DECISIONS.md:7-44) |
| τ_v4 vs μ-v4-replan | tie 4-4 |
| σ_v4 vs μ-v4 (Phase 2 prod) | μ-v4 wins **7-1 STRONG** |

**Key correction**: τ_v4 vs σ_v4 absolute Δ=+0.75 was originally interpreted as
"noise floor — not worth pairwise". This was a cost-saving compromise (per
feedback_quality_first.md). Pairwise 7-1 STRONG REJECTED the noise hypothesis.
Reaffirms M4 cross-validation discipline: any close-cluster comparison MUST run
pairwise, regardless of cost concern.

**Smoke A self-review hypothesis DISPELLED** (full doc: `paper_materials/methodology/M7_self_review_failure.md`):

8 τ_v4 plans × (Qwen3-30B-base critic vs Opus critic, both with source paper as
privileged info) → 8 Opus-as-judge subagents compare. **Result: Opus 8-0 sweep**
(0 Qwen, 0 tie). A-position win rate balanced (4/8 = 50%, not position bias).

Diagnosis: **Privileged Info Comprehension gap**. Qwen3-30B mentions same
conceptual terms (PUCT, entropic objective) but cannot extract specific formulas,
numerical hparam values (LoRA rank 32, 50×512 batches), domain names (Erdős /
GPUMode TriMul / AtCoder / single-cell), or baseline numbers. Confirms Challenge 2
(privileged info comprehension); refutes implicit assumption that 30B can
self-review at SOTA quality.

**κ_self path (SDPO with 30B self-critic) NOT VIABLE** at this scale.
**κ_opus path (SDPO with Opus critic in distillation framework) STILL OPEN** —
deferred pending τ_v4 BoN ablation result (next).

**INVALIDATION of prior κ-v1 smoke (cold-start critic) verdict**:

Prior κ-v1 smoke (`runs/2026_04_28_kappa_v1_smoke/`) used DISTILLATION_COLD_START_CRITIQUE
for ALL 3 rounds to save ~$20 + 1.5 hr. Drew "FAIL — distillation+SDPO doesn't
work" verdict from audit 24.88 → 24.38 trajectory.

This verdict is **INVALIDATED** per `feedback_quality_first.md` (2026-04-27 binding
rule). Cold-start critique is constant generic text, not real reviewer feedback;
SDPO advantage = "presence of generic instruction block in teacher prompt" trains
model toward "follow generic instructions" not "absorb source paper specifics".
The smoke tested a degenerate config, not the real architecture.

The conclusion "distillation+SDPO doesn't work" is NOT supported by that data.
Smoke A directly addresses what cold-start smoke could not (real critic test).

Per quality-first rule: SMOKE TESTS MUST USE REAL COMPONENTS, not mocks. Cost
should not drive eval design. This INVALIDATION entry is the durable record so
future sessions don't cite the cold-start verdict.

**Affected docs**:
- `paper_materials/findings/F8_phase3_distillation_pathway.md` (new, primary finding)
- `paper_materials/experiments/E8_phase3_distillation_evidence.json` (new, raw data)
- `paper_materials/methodology/M7_self_review_failure.md` (new, methodology lesson)
- `paper_materials/README.md` (updated index)
- `STATUS.md` (Phase 3 status update)
- `runs/2026_04_28_kappa_v1_smoke/` retained for historical record only — DO NOT cite

**Owner**: Yuhong Shi | Session A (走线 A: Phase 3 distillation pathway).

---

## 2026-04-28 (g): D-series investigation — n=24 stress test + 7-baseline pairwise corroboration + extract_solution bug fix

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` | **Trigger**: user mandate ("把能支持或者solid我们现有结论或者能解释现有不清楚地方的run都做了吧") — comprehensive multi-front investigation before paper draft.

**6 open questions targeted**:
1. F7 inconclusive at n=8 — true null or underpowered? → Phase D1 n=24 stress test
2. extract_solution regex bug → Phase D2 fix + diagnostic
3. F1 7-baseline ranking 2-plan-only → Phase D3 σ/δ/α pairwise corroboration
4. No Phase 6 mitigations tested → Phase D4 plan-level replay buffer
5. Per-goal pairwise asymmetry meta_ttl reverses but tool_v_ttrl stays weak μ' → Phase D5 diagnostic from D1
6. M8 noise floor scaling → Phase D1 dual-purpose (resolves Q1 and Q6)

### Phase D1 — n=24 stress test (DONE 2026-04-28, 96 strict + 48 pairwise subagents)

**Pre-registered binding**:
- 95% bootstrap CI on Δ /45 EXCLUDES 0 AND pairwise ≥ 14/24 same direction → CONFIRMED at n=24
- 95% CI INCLUDES 0 AND pairwise within 12-12 ± 2 → TRUE NULL at n=24
- Strict and pairwise still diverge → noise floor revised UP, n=50 needed

**Results**:

| Goal | Strict Δ at n=8 | Strict Δ at n=24 | Pairwise n=24 | A-pos | Verdict |
|---|---:|---:|---|---:|---|
| meta_ttl | -0.12 | **-0.96** | σ' 14 / μ' 9 / 1 (58%) | 52% | **σ' DIRECTIONAL CONFIRMED** ≥14/24 |
| tool_v_ttrl | -0.63 | **-0.58** | σ' 12 / μ' 10 / 2 (50%) | 55% | **TRUE NULL** within 12-12 ± 2 |

3-protocol agreement on direction. F7 transfer is **DEFINITIVELY NULL** at n=24. Anchoring inflation in 2-plan/subagent verified persistent at n=24. Effect is real, not n=8 artifact.

### Phase D2 — extract_solution() regex fix (DONE 2026-04-28)

**Bug**: `_SOLUTION_RE` non-greedy match catches literal `<solution>...</solution>` inside `<think>` preamble. Fix: strip `<think>...</think>` BEFORE applying `_SOLUTION_RE`.

**Diagnostic**: 272 plans across 17 D5 buffers scanned post-fix; **only 1 plan ≤50 chars remaining** (μ-v4 iter 7, 12 chars — genuine generation truncation, not extraction bug). All M8 production audits affected by the bug (1 plan in tt_control, 1 plan was 9/45 → recovered to 12/45 by manual rescue at the time) now extract correctly.

Tests: `tests/test_extract_solution_regression.py` (6 cases including exact M8 regression). Commit `ea26f4e`.

### Phase D3 — 7-baseline pairwise σ/δ/α (DONE 2026-04-28, 24 subagents)

**Pre-registered binding**:
- All 3 matchups within ±5/8 → 7-baseline close-cluster IS true null
- ≥2/3 matchups break ≥6/8 in same baseline's favor → 2-plan ranking robust
- Mixed → per-matchup nuance documented

**Results** (Phase 2F audit_v3 ISOLATED reported σ 25.25 / δ 25.12 / α 25.00, near-zero gaps):

| Matchup | Pairwise | A-pos | vs 2-plan audit Δ |
|---|---|---:|---|
| σ vs δ | **σ 6 / δ 2** ≥ 6/8 | 25% (boundary) | audit said TIE +0.13; pairwise reveals σ > δ |
| σ vs α | α 5 / σ 3 | 88% (outside) | audit TIE +0.25; pairwise weak α |
| δ vs α | α 5 / δ 3 | 88% (outside) | audit TIE +0.12; pairwise weak α |

**Mixed verdict**: 1/3 matchup (σ vs δ) shows ≥6/8 directional break — audit underestimated. The 2 α-matchups have position-bias outside 75% — flagged as M8 caveat (not a clean test). M8 cross-goal effect partially generalizes to 7-baseline; standing protocol (require pairwise + strict at close-cluster) holds.

### Phase D4 — Phase 6α plan-level replay buffer (PENDING — separate session due to scope)

Code change deferred. Will train μ-v6_replay (4 Tool-V SDPO + 4 TTT-D-replay plans/iter from μ-v4 iter 2 anchor); follow-up post-training audit per M8 standard.

### Phase D5 — Per-goal asymmetry diagnostic (DONE — falls out of D1 n=24)

n=8 asymmetry (meta_ttl reverses, tool_v_ttrl stays weak μ') was a noise artifact: at n=24 BOTH goals show σ' edge / TIE. The H4-variance hypothesis was incorrect; the n=8 difference was within ±3.5 sampling noise on the Δ. Per-dim breakdowns at n=24 strict show no single dim drives the residual gap.

### Phase D6 — Doc sync + commits (DONE per F7/M8 updates)

- F7 final verdict updated 2026-04-28 with §"D1 n=24 stress test"
- M8 §"n=24 stress test corroboration" added
- DECISIONS.md (this entry)
- STATUS.md, PROJECT_OVERVIEW.md updated below

**Cost actual**: ~$45 subagent (96 strict + 48 pairwise + 24 D3 pairwise). 0 OpenRouter. ~6 hr wall-clock. Phase D4 deferred to next session.

**Final paper-impacting verdicts (post D-series)**:
1. **F2 μ-v4 +2.75 over σ on TTT-D**: UNCHANGED (Phase 2 same-protocol)
2. **F7 cross-goal forward-citation transfer**: NEGATIVE at n=24 (DEFINITIVE)
3. **F9 Phase 5 NULL**: TRIPLE-CORROBORATED (unchanged)
4. **M8 audit-pairwise divergence**: STRENGTHENED at n=24 (anchoring persists; per-goal direction reverse confirmed)
5. **F10 Phase 6α replay**: PENDING D4 execution

---

## 2026-04-26 (f): Phase 5 quality audit remediation — pairwise corroboration + strict subagent isolation re-audit (走线 B)

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` | **Trigger**: user audit (2026-04-27) — concerned this session may have compromised quality like Session A. 3 NEW hard rules added: ZERO OpenRouter, subagent task isolation, quality first (pairwise mandatory on close-cluster).

**Audit findings** (Explore agent + my own analysis):

✅ **Clean (4 dimensions)**: audit prompt fidelity (full 9-dim DEPTH_AUDIT_PROMPT_V3 verbatim w/ 4 OpenReview reviewer quotes); daemon protocol (5-XML-block critic + 4-dim in-loop audit per spec); audit response substantive (claim/concern lists evidenced); ZERO OpenRouter.

⚠️ **Gaps requiring remediation**:
1. **Phase 5 v3 iter 0 vs μ-v4 isolated /45 Δ +0.62** — close-cluster, no pairwise (CLEAR violation of new rule)
2. **Phase 4a meta_ttl Δ +2.12 + tt_control Δ -0.88** — borderline close-cluster, no pairwise (rule not retroactive but paper claims hold to standard)
3. **`audit_v3_isolated_2way.py` uses 1-subagent-per-2-plans** — violates strict task isolation (inherited Phase 2F design; balanced anchoring tradeoff)

**Locked remediation (BINDING — user said "全做吧")**:

### Phase A.1 — Pairwise μ-v4 vs μ-v5_v3 iter 0 on Tool-V
- 8 anonymized pairs, position-randomized, 1 subagent per pair (strict task isolation)
- Reuses `OpusPairwiseClient` + Phase 2 `mu_v4_pairwise_v1.py` template
- Pre-registered pass/fail:
  - μ-v4 wins ≥6/8 → Phase 5 NULL CORROBORATED (μ-v4 strictly better)
  - μ-v5_v3 wins ≥6/8 → Phase 5 marginal positive REVEALED (audit underestimated; halt + brief user before continuing)
  - 4-5 split → noise floor confirmed; verdict inconclusive

### Phase A.3 — Pairwise on Phase 4a borderline (meta_ttl + tt_control)
- 2 goals × 8 pairs = 16 subagent calls
- meta_ttl: σ' vs μ' (μ-v4 + slim oracle); tt_control: same
- tool_v_ttrl Δ +3.00 NOT borderline (skipped)
- Pass/fail per matchup:
  - μ' wins ≥6/8 → F7 verdict robust
  - σ' wins ≥6/8 → F7 verdict flips (Strong-mixed → Mixed/Weak)
  - 4-5 split → borderline as expected; F7 documents

### Phase B — Strict 1-plan-per-subagent re-audit on 5 isolated /45 audits
- Re-run all 5 isolated audits at strict 1-plan/subagent: meta_ttl / tt_control / tool_v_ttrl / Phase 5 v3 / Phase 5 α
- 5 audits × 16 plans = 80 subagent calls
- Output: `runs/2026_04_28_*_audit_strict/` dirs
- Compare strict vs original 2-plan:
  - Δ direction stable AND |strict mean - original mean| ≤ 2.0 → methodology robust
  - Δ direction flips OR mean delta > 2.0 → original was anchoring artifact, strict authoritative

### Phase C — Document + commit
- F9 (Phase 5) + F7 (Phase 4a) updated with corroboration sections
- DECISIONS (f) entry locks design + final verdicts
- STATUS.md if any verdict flipped
- Single commit

**Cost estimate**: ~$31-40 subagent compute (24 pairwise + 80 strict-audit), ~7-8 hr wall-clock. ZERO OpenRouter.

**Mid-run safety gate**: if A.1 pairwise FLIPS Phase 5 verdict (μ-v5_v3 wins ≥6/8), halt before B + brief user (paper main negative finding would change).

**Owner**: Yuhong Shi | Session B (走线 B)

**Artifacts**:
- Code: `phase5_remediation_pairwise.py` (Phase A unified — 4 matchups), `audit_v3_isolated_2way_strict.py` (Phase B strict 1-plan fork)
- Run dirs: `runs/2026_04_28_phase5_remediation_pairwise/` (32 pairwise subagents), `runs/2026_04_28_*_audit_strict/` × 5 (80 strict subagents)
- Doc updates: F9, F7, M8 methodology, this DECISIONS entry closing section

### 2026-04-27 closing — final verdict

**Phase A pairwise (DONE 2026-04-27 morning)**:
| Matchup | Pairwise tally | Direction match? |
|---|:---:|:---:|
| Phase 5 v3 vs μ-v4 (Tool-V) | μ-v4 4 / μ-v5_v3 4 | TIE — Phase 5 NULL **CORROBORATED** |
| Phase 4a meta_ttl σ' vs μ' | σ' 5 / μ' 3 | **REVERSED** original audit Δ +2.12 |
| Phase 4a tt_control σ' vs μ' | σ' 5 / μ' 3 | reinforces audit -0.88 |
| Phase 4a tool_v_ttrl σ' vs μ' | μ' 4 / σ' 3 / 1 TIE | weak μ', within noise |

None of 3 Phase 4a goals reaches ≥6/8 directional threshold. Phase 4a "Strong-mixed 2/3 pass" verdict is NOT corroborated.

**Phase B strict 1-plan/subagent re-audit (DONE 2026-04-27 evening)**:
| Goal | Orig 2-plan Δ | Strict 1-plan Δ | \|shift\| | H1 verdict |
|---|---:|---:|---:|---|
| meta_ttl | +2.13 | -0.12 | 2.25 | **CONFIRMED** (matches pairwise σ') |
| tt_control | -0.88 | -0.63 | 0.25 | stable (3-protocol convergence) |
| tool_v_ttrl | +3.00 | -0.63 | 3.63 | **CONFIRMED** (collapse to TIE) |
| phase5_v3 | +0.62 | +1.87 | 1.24 | stable within noise |
| phase5_alpha | -2.63 | +0.26 | 2.89 | strict softens α verdict to TIE |

**H1 (balanced batching anchoring) CONFIRMED on 2/4 close-cluster goals where pairwise is available**. The 2-plan/subagent protocol systematically inflates Δ at borderline cases by ~2-3 points.

**Final paper-impacting verdicts**:
1. **F2 μ-v4 iter-4 +2.75 over σ**: UNCHANGED (intra-Phase-2 same-protocol comparison; no anchoring confound)
2. **F7 Phase 4a Strong-mixed (2/3 pass)**: **DOWNGRADED** to "all 3 inconclusive within audit noise; methodology divergence (M8) is the surviving contribution"
3. **F9 Phase 5 NULL across 4-cell grid**: **STRENGTHENED** by 3-protocol agreement (in-loop trajectory + isolated /45 audit + pairwise + strict)
4. **F9 anchor_ce REFINED**: original "α plans WORSE (-2.62)" softens to "α TIE (+0.26 strict)"; trajectory cliff at iter 2 remains the load-bearing failure signal
5. **NEW M8 methodology contribution**: `paper_materials/methodology/M8_audit_pairwise_divergence.md` — 2-plan/subagent batch anchoring inflates close-cluster Δ; strict 1-plan + pairwise should be required for borderline cases

**Cost actual**: ~$36 subagent (32 pairwise + 80 strict-audit + 1 batch_13 JSON-fix). ZERO OpenRouter calls. ~7 hr wall-clock.

**Standing protocol going forward**: at close-cluster (2-plan Δ < 5/45), MUST run pairwise OR strict corroboration before publishing directional verdict. Add to MEMORY.md feedback memories.

---

## 2026-04-26 (e): Phase 5 COMPLETE — NULL across 4-cell grid; Phase 4b formally DEPRECATED (走线 B)

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` | **Finding**: [`paper_materials/findings/F9_continual_sdpo_field_expert.md`](paper_materials/findings/F9_continual_sdpo_field_expert.md)

**Result**: Continual SDPO chain hypothesis tested across 4-cell grid (anchor saturation × Adam state × anchor_ce regularization). **All 4 cells fail H1 forward learning**. Per pre-registered matrices in DECISIONS 2026-04-26 (c) and (d), Phase 5 verdict is **NULL**.

**Trajectory across 4 grid cells (in-loop /20)**:

| Run | Anchor | Adam | anchor_ce | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 | Failure |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| cliff_v1 | μ-v4 iter 4 | full | 0 | 8.38 | 4.88 | (kill) | — | — | iter-1 cliff -3.50 |
| cliff_v2 | μ-v4 iter 4 | fresh | 0 | 9.00 | 6.875 | (kill) | — | — | iter-1 cliff -2.125 |
| v3 (iter2) | μ-v4 iter 2 | fresh | 0 | 9.375 | 9.000 | 8.750 | 7.750 | 6.67 | 4-iter slow decline -2.71 |
| **α** | **μ-v4 iter 2** | **fresh** | **0.1** | **10.375** | 9.625 | **6.625** | (kill) | — | **iter-2 cliff -3.0** |

**Best v5 vs μ-v4 isolated /45 on Tool-V** (n=8 each, 8 audit batches):
- v3 iter 0: 22.25 vs 21.62 → **Δ +0.62** (within n=8 noise; H1 fails +1.0 threshold)
- α iter 0: 19.25 vs 21.88 → **Δ -2.62** (negative; H1 fails hard)

**3-axis combined finding** (paper-grade): Adam state controls cliff magnitude (-3.5 → -2.1) but never prevents cliff. Anchor saturation controls cliff TIMING (iter 1 vs iter 4) but never prevents degradation. anchor_ce regularization at weight 0.1 ACCELERATES collapse (iter 2 vs iter 4 of v3) — per-token CE anchor pulls toward TTT-D prose but dilutes Tool-V task-specific reproducibility, two competing gradients fight rather than consolidate.

**Phase 4b cross-domain stretch test**: formally **DEPRECATED**. A scientist deeply in TTT-RL doesn't switch fields overnight; the field-expert metaphor breaks. Removed from active scope.

**Implications for paper**:
1. Paper has 3 cohesive findings: F2/F4 (μ-v4 +2.75 single-paper SDPO works), F7 (Phase 4a forward-citation transfer Strong-mixed 2/3), F9 (continual NULL with 4-cell diagnostic)
2. Phase 6 future-work directions explicit: replay buffer (plan-level), EWC (Fisher importance), LoRA-per-paper merge — per-token CE anchor proven WRONG abstraction
3. Paper narrative: "Single-paper SDPO + privileged-info reviewer internalizes preferences that transfer to follow-up papers; chained SDPO requires plan-level consolidation, not output regularization"
4. Phase 7 paper draft is next active phase (start fresh plan)

**Operational results (final, all via Claude subagent — 0 OpenRouter)**:
- Total subagent compute Phase 5: ~$18 (5 critic + 4 in-loop /20 audit sessions across 4 runs + 16 isolated /45 audit batches across 2 v5 candidates)
- Total Tinker: ~2.5 hr (all free)
- Total wall-clock: ~6 hr from T0 (pre-registration) to T6 (final commit)

**Owner**: Yuhong Shi | Session B (走线 B)

**Artifacts (all preserved)**:
- F9 finding (canonical): [`paper_materials/findings/F9_continual_sdpo_field_expert.md`](paper_materials/findings/F9_continual_sdpo_field_expert.md)
- Run dirs: `runs/2026_04_28_mu_v5_tool_v_cliff_v1/`, `_cliff_v2/`, `_iter2anchor_5iter/`, `_anchor_ce_v1/`
- v3 + α inference + audit dirs: `runs/2026_04_28_mu_v5_*_inference/`, `runs/2026_04_28_phase5_audit_v3_iter0/`, `runs/2026_04_28_phase5_audit_anchor_ce/`
- Code commits: `f3b9485` init_state_path; `d2d112c` reset_optimizer_state; `a984640` anchor_ce regularization
- Memory: `project_d5_phase5_continual_sdpo.md` updated with NULL final verdict

---

## 2026-04-26 (d): Phase 5α — anchor_ce regularization mitigation test (binding pre-reg)

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` | **F9 section TBD**

**Trigger**: Phase 5 main result NULL (3 runs failed: cliff_v1, cliff_v2, iter2anchor_v3 all show plan-quality degradation under naive continual SDPO). Per user 2026-04-26 decision ("更全面一些"), test ONE mitigation before declaring final NULL — anchor_ce regularization (CL textbook recommends EWC/L2 before replay buffer).

**Locked design**:
- Mitigation: anchor_ce regularization. Per iter: SDPO chunked grad steps + 1 extra forward_backward call with constant-positive-advantage SDPO datum on (TTT-D goal+oracle | TTT-D ref_plan), advantage = anchor_ce_weight = 0.1 → effective loss = 0.1 × CE(model, TTT-D ref_plan).
- Anchor weights: μ-v4 iter 2 (`weights/000002_2026_04_25`) — best surviving config from v3.
- Optimizer: fresh Adam (`reset_optimizer_state=true`).
- Source paper (critic): Tool-V (current task, same as v3).
- Anchor goal (CE prompt context): TTT-D `dataset/research_goal.txt` (different from current task — anchors prior task's distribution).
- anchor_ce_weight: 0.1 (Kirkpatrick EWC convention).
- n_iter=5, audit_drop_threshold=2.0.

**Pre-registered pass/fail criteria (BINDING — no post-hoc edits)**:

| α run outcome | F9 paper claim |
|---|---|
| **No cliff/decline** (no inter-iter drop ≥ 2.0) **AND** best-iter Δ ≥ +1.0 vs μ-v4 isolated /45 on Tool-V | **SUCCESS**: "anchor_ce regularization rescues continual SDPO; recipe = μ-v4 iter 2 anchor + fresh Adam + anchor_ce_weight=0.1" |
| Cliff or decline persists OR best-iter Δ < +1.0 | **FAIL**: "Naive AND anchor_ce regularization fail; consolidation requires full replay/EWC. Future work." |

**Pre-registered weight escalation**: if smoke (1-iter) shows mean_adv collapse < 0.05 at iter 0 due to anchor_ce dominating SDPO signal → reduce weight to 0.05 (single retry); if still collapse → 0.01 (single retry); if still collapse → declare "weight 0.01-0.1 all fail" and write FAIL with weight-sensitivity finding.

**Implementation**: ~30-line diff to `train_mu_v4.py`:
1. Add `anchor_goal_path: str = ""` config field (after `reference_plan_path`)
2. Pre-tokenize anchor prompt + ref_plan once at trainer init (when anchor_ce_weight > 0)
3. After SDPO chunked grad loop: 1 extra forward_backward call with anchor SDPO datum (constant +weight advantage)

Exploits the existing `loss_fn="importance_sampling"` form `-mean(advantage × log_pi)`: with constant +w advantage, becomes w × CE. Zero new loss-fn code.

**Cost (estimate)**: ~$9 subagent (1 critic + 5 in-loop /20 audits across 5 iters + 8 isolated /45 audit batches at end) + ~2 hr Tinker + ~5-6 hr wall-clock.

**Out of scope (still deferred)**:
- True replay buffer (Phase 6α-A)
- EWC penalty with Fisher Information (Phase 6α-C)
- LoRA-per-paper merge (Phase 6β)
- Higher anchor_ce_weight sweeps (only escalate down per pre-reg)
- μ-v6 chain step (deferred until α verifies v5 is viable)

**Owner**: Yuhong Shi | Session B (走线 B)

**Artifacts (TBD)**:
- Run dir: `runs/2026_04_28_mu_v5_anchor_ce_v1/`
- α best-iter inference + audit: `runs/2026_04_28_mu_v5_anchor_ce_v1_inference/`, `runs/2026_04_28_phase5_audit_anchor_ce/`
- F9 update: `paper_materials/findings/F9_continual_sdpo_field_expert.md` (α-B section)
- Memory: `project_d5_phase5_continual_sdpo.md` final verdict

---

## 2026-04-26 (c): Phase 5 design LOCKED — Continual SDPO chain (field-expert hypothesis test) (走线 B)

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` | **F9 (TBD)**: `paper_materials/findings/F9_continual_sdpo_field_expert.md`

**Reframe (user 2026-04-26)**: D5's research narrative is **domain-expert via continual SDPO with privileged-info reviewers**, NOT "cross-goal generalization". A scientist deepens preferences within their subfield over time; SDPO + Opus reviewer fast-forwards this accumulation into LoRA weights. Phase 4a established that single-paper SDPO transfers preference signature to follow-ups (Strong-mixed 2/3 + aggregate Δ +1.42); Phase 5 tests the actual continual-learning hypothesis.

**Phase 4b cross-domain stretch test**: **DEPRECATED**. A scientist deeply in TTT-RL doesn't suddenly switch to ViT/World Models — the field-expert metaphor breaks. Removed from PHASE_PLAN_v2.md (action item in T6.4).

### 3 user-confirmed choices (A/A/A locked)

| # | Choice | Locked | Rejected alternatives |
|---|---|---|---|
| 1 | **Chain composition** | 2-step: μ-v5 = μ-v4 + Tool-V SDPO; μ-v6 = μ-v5 + Meta-TTL SDPO. tt_control held out as expansion test | B: 3-step with tt_control in chain (trivial lift expected; doesn't test expansion). C: 3-step with 4th forward citation (longer chain, +50% cost, no clear added value) |
| 2 | **Forgetting mitigation** | NAIVE (no replay/EWC/LoRA-merge). Test what breaks; "naive sufficient" is itself a finding | B: Replay buffer (50/50 mix old plan, +2× compute per iter). C: LoRA-per-paper + weighted merge (Phase 6 territory) |
| 3 | **Audit n** | n=8 per condition (consistent with Phase 4a). Bootstrap 90% CI on Δ as secondary signal | B: retention-only n=16. C: all n=16 (×2 cost, marginal CI tightening) |

**Order rationale**: Tool-V first because Phase 4a Δ +3.00 (largest forward transfer → most receptive substrate for next SDPO step); Meta-TTL second (next-strongest at +2.12); tt_control HELD OUT because Phase 4a Δ -0.88 (the failing goal — adversarial test for H3 expansion).

### Pre-registered hypotheses + thresholds (BINDING — no post-hoc edits)

| ID | Hypothesis | Quantitative test | PASS criterion |
|---|---|---|---|
| **H1 Forward learning** | Each chain step lifts on its newly-trained paper | Δ(v5−v4 on Tool-V); Δ(v6−v5 on Meta-TTL) | each ≥ +1.0 /45 |
| **H2 Retention** | No catastrophic forgetting at chain end | Δ(v5−v4 on TTT-D); Δ(v6−v4 on TTT-D); Δ(v6−v5 on Tool-V) | each ≥ -2.0 /45 AND lower 90% bootstrap CI ≥ -5 |
| **H3 Expansion (held-out)** | Broader expertise fixes prior 4a-failing goal | Δ(v6−v4 on tt_control) | ≥ +1.0 /45 |
| **H4 Aggregate field-expert** | μ-v6 net better across all 4 papers | mean(v6−v4) over {TTT-D, Tool-V, Meta-TTL, tt_control} | ≥ +1.0 /45 |

T1-T4 rubric dims pre-registered as **non-load-bearing** (T1-T4 calibrated to TTT-RL; new papers may legitimately omit those mechanisms). Verdict uses total /45 + U1-U5 only; T-dims reported for transparency.

### Decision matrix → paper claim

| H1 | H2 | H3 | H4 | Paper claim |
|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | **STRONG**: "Naive continual SDPO transfers AND retains AND expands." |
| ✅ | ✅ | ❌ | ✅ | **EXPECTED MODAL**: "Continual SDPO transfers + retains, but doesn't solve oracle-coverage failures. Section 5 motivates retrieve-then-generate (Session A's Phase 3)." |
| ✅ | ❌ | * | * | **PARTIAL**: "Continual SDPO learns each step but forgets prior domains. Replay/EWC needed; future work." |
| ❌ | * | * | * | **NULL**: "SDPO doesn't transfer step-by-step at lr=5e-5 from a v4 anchor; Hübotter §4 instability triggered. Reset-Adam follow-up needed." |

**H3 is adversarial-by-construction**: tt_control failed in 4a because slim oracle didn't cover differentiable-LQR-adapter mechanism (architectural-mechanism gap, not parameter shortage). Tool-V/Meta-TTL SDPO won't inject that coverage. Failing H3 is the prior; passing H3 = strong-positive surprise. F9 narrative planned around (✅✅❌✅) modal outcome.

### Cross-session namespace (Phase 5, Session B)

- Greek: μ-v5, μ-v6 (chain extensions of existing μ-v4)
- Run prefixes: `runs/2026_04_28_mu_v5_*`, `runs/2026_04_29_mu_v6_*`, `runs/2026_04_28_phase5_audit_*`
- Code: train_mu_v4.py extension (`init_state_path` field, ~9 line diff)
- Reused: `audit_v3_isolated_2way.py` (already exists from 4a)
- Paper slot: F9 (F8 reserved for Session A's Phase 3 retrieval pathway)

No conflicts with Session A's Phase 3 namespace (τ/κ).

### Risk-mitigation pre-registration

1. **Adam state inheritance amplifies Hübotter cliff**: μ-v4's iter-5 cliff was at lr=5e-5. Continual from v4-iter-4 with inherited Adam moments could drift faster on new loss surface. Mitigation A: tighten audit_drop_threshold to 2.0 (from 3.0), iter-1 mean_adv < 0.1 OR pos_frac < 0.5 → halt. Mitigation B (B-plan if A fails): re-run with fresh-Adam state via a future `reset_optimizer_state` config flag (deferred).
2. **n=8 noise floor inside H2 threshold**: 95% CI on Δ ≈ ±3.5 /45. Use bootstrap **90% CI lower bound ≥ -5** as secondary discrimination signal. If borderline, paper writes "retention is at noise floor; bigger n future work."
3. **Cross-paper reviewer signal disagreement**: each chain step has different Opus reviewer (different privileged source paper). This IS the field-expert metaphor (multi-expert composite). F9 will report inter-reviewer ρ on n=8 plans rated by both reviewers as bonus diagnostic.

### Operational

- 7 audits × 8 batches = 56 audit subagent calls + ~10 critic subagent calls = ~66 Opus subagent calls total (all via Claude `Agent(general-purpose, model=opus)`; **0 OpenRouter**)
- Estimated cost: ~$35-40 subagent compute, ~4 hr Tinker (free), ~9 hr wall-clock (or ~5 hr if Step 3 audits pipeline behind Step 4 train)

**Owner**: Yuhong Shi | Session B (走线 B)

**Artifacts (TBD post-execution)**:
- F9: `paper_materials/findings/F9_continual_sdpo_field_expert.md`
- Run dirs: `runs/2026_04_28_mu_v5_tool_v/`, `runs/2026_04_29_mu_v6_meta_ttl/`, `runs/2026_04_28_phase5_audit_*/`
- Code: `train_mu_v4.py` extension (`init_state_path` config field)
- Memory: `project_d5_phase5_continual_sdpo.md`

---

## 2026-04-26 (b): Phase 4a verdict — Strong-mixed (2/3 forward-citation goals pass) (走线 B)

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` | **Finding**: [`paper_materials/findings/F7_cross_goal_followup_validation.md`](paper_materials/findings/F7_cross_goal_followup_validation.md)

**Result**: μ-v4 iter-4 LoRA tested on 3 forward-citation TTT-Discover follow-ups, slim oracle reused verbatim:

| Goal | σ' | μ' | Δ | Pass ≥+1.0? |
|---|---:|---:|---:|---|
| meta_ttl (Lou Meta-TTL, 2604.00830) | 17.75 | 19.88 | **+2.12** | ✅ |
| tt_control (Wang TT-Control, 2603.09221) | 20.00 | 19.12 | -0.88 | ❌ |
| tool_verification_ttrl (Liao, 2603.02203) | 20.38 | 23.38 | **+3.00** | ✅ |
| **Aggregate** (n=24 each) | 19.38 | 20.79 | **+1.42** | ✅ (>+0.5) |

**Verdict per pre-registered matrix**: **Strong-mixed (2/3 pass)** — "Mostly goal-stable; F7 documents the failing goal + per-dim diagnostic." NOT full pass (which required 3/3), NOT fail.

**Per-dim transfer signature** (avg across 3 goals) **matches Phase 2's iter-4 → σ pattern**:
- μ' wins consistently: U5 Reproducibility (+0.79), T3 Compute (+0.46), T4 Reward-hacking (+0.33), T1 Necessity (+0.21), U1 Soundness (+0.25)
- σ' wins: T2 Disentanglement (-0.29), U4 Clarity (-0.25), U2 Significance (-0.21)
- μ-v4 internalized depth (concrete numbers, hparams, compute units, Goodhart pathways) at slight cost of breadth (alternative-hypothesis coverage). Same trade Phase 2F audit revealed.

**TT-Control failure diagnosis**: μ' has wide variance (std 5.30 vs σ' std 1.41); 1 plan hit max_tokens=4096 truncation (idx=6, audit total 9). Excluding that single outlier, μ' (n=7) mean = 20.57 → Δ = +0.57 (positive, under threshold). Median μ' = 22.5 vs σ' 20.5 → median Δ = +2.0 (matches direction of other 2 goals). Goal-specific cause: TT-Control asks for an architectural mechanism (differentiable LQR adapter) that the slim oracle's TTT-RL-flavored insights don't directly cover — μ-v4 jargon-drifts and loses U4 Clarity. Genuine OOD signal, not noise. Phase 4b oracle-rebuild experiment would address.

**Bootstrap 95% CIs** (n=8 per condition, 10K resamples; n is the known small-sample limit):
- meta_ttl: [-0.75, +4.88]; tt_control: [-4.88, +2.75]; tool_v_ttrl: [-0.38, +5.88]; aggregate: [-0.75, +3.50]

n=8 per condition is too small for 95% CI to clear 0 on any single goal. Mean direction is positive on 2/3 + aggregate; CI tightening needs n≥30 per condition (deferred).

**Implications**:
1. Phase 2's +2.75 on TTT-Discover is **NOT pure overfit** — same per-dim lift signature reproduces on follow-up papers from same subfield.
2. Cross-goal claim for paper is **EVIDENCE-BASED, honestly-labeled near-OOD**: forward citations + same subfield + oracle reused. NOT cross-domain.
3. **U5 + T3 + T4** are the load-bearing dims for μ-v4's value claim; paper's contribution narrative should center these.
4. Single failing goal (TT-Control) is informative for Phase 4b stretch-test scope: oracle rebuild needed when source-paper architecture is more divergent from TTT-Discover.

**Operational results (cost/time, all via Claude subagent — 0 OpenRouter):**
- Subagent compute: 33 calls (8 ref-plan extractions + 25 audits, including 1 re-spawn for malformed JSON) ≈ ~$13-16
- Wall-clock: ~30 min from T4 to verdict (parallel subagent batches)
- Tinker: free (6 inference runs ≈ 7 min)

**Next concrete step**: Phase 5 paper-draft can now use F7 + F1-F6. Phase 4b stretch test (medium-OOD with oracle rebuild) deferred — only triggered if reviewer demands.

**Owner**: Yuhong Shi | Session B (走线 B)

**Artifacts**:
- F7 finding: [`paper_materials/findings/F7_cross_goal_followup_validation.md`](paper_materials/findings/F7_cross_goal_followup_validation.md)
- Per-goal raw plans: `runs/2026_04_28_xgoal_{meta_ttl,tt_control,tool_verification_ttrl}_{sigma,mu}/buffer.jsonl`
- Per-goal audit responses + summaries: `runs/2026_04_28_xgoal_*_audit/audit_*` (24 batch JSONs + 3 summaries)
- Inference harness: `src/co_scientist/d5_abstract_retrieve_refine/baseline_frozen_v2.py` (variant=mu_inference, commit `8dbf837`)
- Audit pipeline: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated_2way.py`
- Subagent prompt (shared): `runs/2026_04_28_xgoal_audit_subagent_prompt.md`

---

## 2026-04-27: Phase 3 design LOCKED — Q1-Q9 + 4 Decision Gates (走线 A: Path Y distillation pipeline)

**Authoritative reference**: [`knowledge/current/RETRIEVAL_DESIGN_v1.md`](knowledge/current/RETRIEVAL_DESIGN_v1.md).
**Plan**: `~/.claude/plans/peaceful-tickling-wolf.md`. **Namespace lock**: 2026-04-27 entry below.

**Trigger**: After Phase 2 COMPLETE (μ-v4 28.00 + pairwise 20/24), conversation
locked Phase 3 design across Q1-Q9. This entry is the durable record.

### Q1-Q9 lock summary

| Q | Decision |
|---|---|
| Q1 | **τ baseline first**, must pass Gate 1 before launching κ |
| Q2 | κ from **fresh base Qwen3-30B**, NOT warm-start μ-v4 (clean comparable) |
| **Q3** | **Path Y: oracle source = `oracle_v2 medium.md` (181 items, 6-cat), NOT final.md (580 items)** — see Q3 reframe rationale below |
| Q3.5 | Token budget: τ output ~2200 ≈ σ slim ~2100 (size-matched) |
| Q4 | **D-2 distillation** (model writes new abstraction list, NOT selection of item ids) |
| Q4.1 | 3 batches: **(60 / 60 / 61)** items, shuffled with `seed=42` (τ) / `seed=43` (κ) |
| Q4.2 | 3 rounds **INDEPENDENT** — round j does NOT see A_1..A_{j-1} (avoid anchoring + SDPO cross-round pollution) |
| Q4.3 | Plan generation step DOES see [A_1, A_2, A_3] concat; instruction "去重 + 整合" |
| Q5 | Critic on **each A_i** (option A); fallback to option C (dual-layer) if Gate 3 fails |
| Q5.1 | Critic format: XML (missing_critical / noise / faithfulness / improvement); Opus subagent + privileged source paper |
| Q6 | τ baseline = frozen + **NO critic** + 3 round independent + plan from concat |
| Q6.1 | τ output ~700 tokens/round; plan max_tokens=4096 (μ-v4 同款) |
| Q7 | **Single goal: TTT-Discover only** (cross-goal validation = Session B 走线 B) |
| Q8 | Budget: ~15 hr Tinker + Opus via subagent only (zero OpenRouter per `feedback_minimize_openrouter.md` 2026-04-27 strengthening) |
| Q9 | κ training: μ-v4 recipe (lr=5e-5, 4 grad steps, n_iter=20, save_every=1, eval_every=1) |
| Q9.1 | Early stop: audit-drop ≥ **2** over 2 audits (μ-v4 was ≥3, Phase 3 紧化) |
| Q9.2 | Production: best-of-early-iter (iter 3-5 highest audit) |
| **Q10** | **Word target: 600-750 (Phase 2 `_PLAN_FOOTER` verbatim) + Step 7 τ-1200 ablation** |

### Q10 word-target rationale (added 2026-04-27 mid-Step-2a)

Discovered that all Phase 2 prompts cap plan output at "Target 600 words, max
750 words" — half of reference plan's ~1200 words; **no documented rationale**
(D3 historical convention). T3 (compute) + U5 (reproducibility) audit dims
systematically capped (per N3.L2).

**Decision**: keep 600/750 in Phase 3 (τ + κ) to preserve cross-phase
comparability with σ=25.25 / μ-v4=28.00. Add **τ-1200 ablation** in Step 7:
1 extra τ run at 1200/1500 target + 1 audit ($5, ~1 hr). Verdict:
- τ_1200 ≫ τ_600 (gap > 1.5) → trigger Phase 4 baseline re-runs
- τ_1200 ≈ τ_600 (gap < 0.5) → 600/750 isn't the bottleneck, ship as-is

**Why NOT re-baseline Phase 2 now**: Phase 2 mechanism claims internally valid
at fixed target; re-baseline costs ~30 hr Tinker + ~$200 Opus + 4-5 days +
invalidates Session B's μ-v4 sampler dependency + rewrites F1-F6 docs. Net
estimated lift: σ and μ-v4 both rise in tandem, headline gap likely unchanged.

**Paper Limitations**: existing N3.L2 + new L11 stub on word-target ceiling.

### Q3 reframe — final.md (580) → medium.md (181)

**Discovered 2026-04-27 during Step 0 implementation**:

- final.md = 116K tokens; Qwen3-30B-A3B context = **40K tokens** (oracle build doc self-discloses this constraint)
- final cannot fit even one batch (200/200/180 → ~38K tokens each, no room for goal/output/critique)
- Switched to medium.md (181 items, ~37K tokens, Opus relevance ≥2 filter): 3 batches × 60 items × ~12K tokens fits 40K with 24K margin

**Why this is the more correct experimental design (not a compromise)**:

1. **Clean σ vs τ comparison**: σ uses slim (87, relevance ≥3 subset of medium); τ uses full medium (181, relevance ≥2). Both pull from the **same pool** — σ is one Opus-curated subset, τ is the 30B's dynamic subset. final would confound "selection method" with "noise filtering capability"
2. **Information theory**: 181 items × 200 tokens ≈ 37K of content; selection space C(181, 90) ≈ 10⁵³ — enough for 30B to demonstrate selection capability
3. **Build incrementally**: Phase 3 = mechanism check (clean medium pool); Phase 4 stretch = noise filter test (final or raw papers)
4. **Methodologically honest**: paper writes "We retrieve from `oracle_v2_medium`, the largest set fitting Qwen3-30B-A3B's 40K context"

### Decision Gates (4 gates with explicit triggers)

**Gate 1 (after τ baseline + audit)**:
| τ vs σ | Verdict | Action |
|---|---|---|
| τ ≥ 26.25 | CONFIDENT | proceed κ |
| 24.25 ≤ τ ≤ 26.25 | NEUTRAL | proceed κ + mandatory pairwise |
| τ < 23.25 | STOP | DO NOT train κ; debug pipeline |

**Gate 2 (κ training mid-iter monitors)**:
- mean_adv collapse (<0.1 × 3 iters): early stop
- audit-drop ≥ 2 over 2 audits: early stop
- catastrophic crash (>5 audit drop in 1 iter): emergency stop

**Gate 3 (κ-peak verdict)**:
| κ-peak vs τ vs σ | Verdict | Action |
|---|---|---|
| κ ≥ τ+2 AND κ ≥ σ+3 | STRONG SUCCESS | proceed Gate 4 |
| τ+1 ≤ κ < τ+2 | MODERATE | proceed Gate 4 |
| τ-1 ≤ κ < τ+1 | TRAINING NEUTRAL | retry option C or pivot framing |
| κ < τ-1 | TRAINING FAILED | retry C or ship τ only |

**Gate 4 (κ pairwise corroboration)**:
| Matchup | Threshold | Verdict |
|---|---|---|
| κ vs σ (PRIMARY) | κ ≥ 6/8 | PASS — paper headline confirmed |
| κ vs σ | κ 4-5/8 | FLAG — n=16 follow-up |
| κ vs σ | κ ≤ 3/8 | INVERSION — claim retracted |
| κ vs τ | κ ≥ 6/8 | training adds value over frozen pipeline |

### Affected files (created Step 0; rest TODO per RETRIEVAL_DESIGN_v1)

- DONE: `src/co_scientist/d5_abstract_retrieve_refine/oracle_batch_helper_v1.py`
- DONE: `knowledge/current/RETRIEVAL_DESIGN_v1.md` (authoritative single-source)
- DONE: this DECISIONS entry

**Owner**: Yuhong Shi | Session A (走线 A)

---

## 2026-04-27: Phase 3 namespace LOCKED (Session A side; cross-references the 2026-04-26 entry below)

**Trigger**: Cross-session coordination with Session B (Phase 4a follow-up
validation, plan `~/.claude/plans/adaptive-churning-perlis.md`). User running
both sessions in parallel; without coordination Greek letters / run dirs /
paper slots / data paths would collide.

**This entry is Session A's anchor**. Full bilateral namespace partition,
including Session B's claim, is recorded in the 2026-04-26 entry below ("Phase
4a cross-goal validation — 3 picks ... namespace coordinated with Session A").
Read that entry for the canonical bilateral lock; this entry exists for
discoverability when readers land on Session A's deliverables (τ/κ/F8/E7/M6)
and need a back-pointer.

**Session A claim summary** (Phase 3 retrieve-then-generate on TTT-Discover):
- Symbols: **τ** = frozen 30B + per-section retrieved chunks; **κ** =
  SDPO+critic trained on retrieval-augmented prompt
- Run-dir prefixes: `runs/2026_04_28_{tau,kappa,phase3,kappa_v1_pairwise,
  retrieval_index_build}_*`
- Source files: `train_tau_v1.py`, `train_kappa_v1.py`,
  `build_retrieval_index_v1.py`, `retrieve_per_section_v1.py`,
  `kappa_pairwise_v1.py`
- Data path: `data/retrieval_index_v1/{chunks.jsonl, embeddings.npy, bm25.pkl}`
  (top-level, per Choice #1 = B)
- Paper slots: F8, E7, M6, N4 (after Session B's F7)
- Knowledge: `knowledge/current/RETRIEVAL_DESIGN_v1.md`

**Choice #1 outcome**: B (keep flat data layout). Session B inverted their
original recommendation (was A=restructure) after Session A appeared, to avoid
parallel-session dependency chain. Restructure deferred to a unified cleanup
commit AFTER both Phase 3 (走线 A) and Phase 4a (走线 B) complete.

**Session A explicit non-touch**: `data/cross_goal/`, `baseline_frozen_v2.py`,
`audit_v3_isolated.py` (uses as-is; B forks to `_2way`), `paper_retrieval.py`,
F7 or earlier slots, cross-goal experiments — all are Session B's territory.

**Session B explicit non-touch**: `data/retrieval_index_v1/`, any τ/κ artifact.

**Owner**: Yuhong Shi (running both sessions) | Session A.

---

## 2026-04-26: Phase 4a cross-goal validation — 3 picks pre-registered (forward citations only); namespace coordinated with Session A

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md`

**Question**: Does μ-v4 LoRA (production iter-4 sampler) add value when applied to forward-citation TTT-Discover follow-up papers (post-cutoff, slim oracle reused verbatim)?

**Pre-registered selection rule**: Top picks ranked by methodological proximity to TTT-Discover (test-time learning + RL/evolution + scientific discovery), drawn from S2's 32 forward citations of arXiv:2601.16175. **Forward citations only** — backward refs would leak through oracle (oracle was extracted FROM them) and likely sit in Qwen3-30B pretraining data (pre-2025-03 cutoff).

**3 locked picks** (post user-confirmation 2026-04-26, pre-execution):

| Pick | ArXiv | Date | Authors | Title | Why |
|---|---|---|---|---|---|
| A | 2604.00830 | 2026-04-01 | Lou et al. | Learning to Learn-at-Test-Time: Language Agents with Learnable Adaptation Policies (Meta-TTL) | Explicit TTT meta-learning; algorithmic angle |
| C | 2603.09221 | 2026-03-10 | Wang et al. | Beyond Test-Time Training: Learning to Reason via Hardware-Efficient Optimal Control (TT-Control) | Explicit TTT extension via LQR; engineering angle |
| D | 2603.02203 | 2026-03-02 | Liao et al. | Tool Verification for Test-Time Reinforcement Learning | Explicit TT-RL; verification/eval angle |

All 3 are post-2026-03 (clean post-Qwen3-30B cutoff), zero external citations (avoids picking-by-fame), diverse angle coverage.

**Rejected alternatives** (anti-cherry-pick documentation):
- B (Chen et al. Self-Evolve, 2603.18620) — RL-heavy but redundant with D's RL focus
- E (Arjmandi Sensi, 2603.17683) — game-agent specific, narrower scope
- F (Nie et al. Iterative GenOpt challenges, 2603.23994) — meta-investigation; too aligned with D5's framing (circular-validation risk)
- G (Feng et al. Aletheia, 2602.10177) — high-citation autonomous math; eval methodology divergent
- H (Qu et al. CORAL, 2604.01658) — multi-agent open-ended; less direct TTT alignment
- I (Du et al. Kernel-Smith, 2603.28342) — kernel optimization; narrower scope
- J (Bicker Aster, 2602.07040) — claims 20× faster; redundant angle with G
- SimpleTES (2604.19341) — flagged by user but absent from S2's citation list; deferred pending PDF cite-verification, NOT in current locked set

**Pass criterion**: μ' > σ' + 1.0 pt mean on **all 3 goals** AND aggregate Δ > 0.5 pt → "μ-v4 LoRA adds value on follow-up work" claim valid.

**Sub-decisions locked**:
- Choice #1 (data folder restructure): **B (keep flat)**. Reversed from initial recommendation A after Session A (Phase 3 retrieve-then-generate, parallel) coordination prompt — restructure now creates sequential dependency between sessions; deferred to post-Phase-4a/Phase-3 cleanup commit.
- Choice #2 (δ' baseline): **A (default no, diagnostic-only)**. Trigger: σ' < 20 on any goal → spawn ad-hoc δ' for that goal to disambiguate "oracle off-domain" from "30B incompetent".
- Choice #3 (3 picks): **A** = Meta-TTL / TT-Control / Tool-Verification.

**Cross-session namespace claim (Phase 4a, this session = "Session B")**:
- Greek letters: σ', μ', δ'
- Run prefix: `runs/2026_04_28_xgoal_{name}_{sigma|mu|audit}`
- New files: `audit_v3_isolated_2way.py`, `cross_goal_candidate_scan.py`
- Modified: `baseline_frozen_v2.py` (add `variant=mu_inference` + `sampler_path` config), `paper_retrieval.py` (add `fetch_paper_incoming_citations`)
- New data: `data/cross_goal/{name}/{source_paper.md, research_goal.txt, reference_solution.txt}`
- Paper slot: F7 (`paper_materials/findings/F7_cross_goal_followup_validation.md`)

**Session A (Phase 3 retrieve-then-generate, parallel) claims**: τ / κ Greek letters; `runs/2026_04_28_{tau,kappa,phase3}_*` run prefixes; `train_tau_v1.py`, `train_kappa_v1.py`, `build_retrieval_index_v1.py`, etc. as new files; `data/retrieval_index_v1/` (top-level, per Choice #1=B); F8 / E7 / M6 / N4 paper slots; `RETRIEVAL_DESIGN_v1.md` knowledge doc. **No conflicts** with Session B's claim.

**Owner**: Yuhong Shi (D5 lead) | Session B

**Artifacts**:
- Plan file: `~/.claude/plans/adaptive-churning-perlis.md`
- Candidate scan source: S2 query 2026-04-26 (32 results; persistence pending T2.2)
- Memories: `feedback_cross_goal_forward_citations.md`, `feedback_full_paper_extraction.md`, `feedback_plan_walkthrough.md`, `project_d5_phase4a_followup_validation.md`

---

## 2026-04-26: μ-v4 +2.75 absolute lift CORROBORATED by pairwise tournament (20/24 = 83.3%)

**Run**: `runs/2026_04_27_mu_v4_pairwise/` — 3 matchups × 8 position-randomized pairs,
Opus 4.7 anonymized judge (seed=50).

**Trigger**: Phase 0.6 had a precedent where absolute audit overstated by 8-0
(μ-v2 vs δ inversion). Before locking μ-v4 = 28.00 / 45 (+2.75 over σ) into the
paper headline, ran an independent pairwise cross-check.

**Result**:

| Matchup | μ-v4 wins | Opp wins | Ties | A-pos win rate |
|---|---:|---:|---:|---:|
| **μ-v4 vs σ** (PRIMARY) | **6** | **2** | 0 | 50% ✓ |
| μ-v4 vs δ | 7 | 1 | 0 | 62% ✓ |
| μ-v4 vs α | 7 | 1 | 0 | 75% ✓ |

**Verdict**: PASS. audit_v3 ISOLATED's close-cluster ranking (μ-v4 ≫ {σ, δ, α})
is corroborated. Position-bias check passes for all 3 matchups. Opus rationales
cite concrete hyperparameters (β=4/8, ρ=0.2, K=100), named tools (RS-GRPO, SIFT,
TTRL, AIME-2024), and citation grounding — substance not surface artifacts.

**Reverses Phase 0.6 finding**: μ-v2 lost 0-8 to δ on pairwise; μ-v4 wins 7-1.
The broad-spectrum lift over μ-v2 (8 of 9 audit dims) transfers to pairwise
dominance.

**Affected docs**: `STATUS.md` 8-baseline table (added pairwise column),
`paper_materials/findings/F2_mu_v4_iter4_peak.md` § "Cross-validation: pairwise
corroboration", new `paper_materials/experiments/E6_mu_v4_pairwise_corroboration.json`.

**Alternatives considered**:
- PRIMARY-only scope (1 matchup vs σ) — would have caught the +2.75 cross-check
  but missed the close-cluster confirmation. User chose RECOMMENDED to also rule
  out δ/α inversion since those are within 0.25 of σ in absolute audit.
- Round-robin pairwise across all 8 baselines — overkill (~$110, 5 hr); skipped.

**Owner**: Yuhong Shi.

---

## 2026-04-27: μ-v4 SDPO at lr=5e-5 SUCCEEDS at iter 4 (+2.75 over σ); multi-round instability confirmed

**Run**: `runs/2026_04_27_mu_v4/` (COMPLETE — production = iter 4 weights)

**Verdict**: Plan-level SDPO + Opus critic at lr=5e-5 lifts Qwen3-30B-A3B EVAL plans
from σ baseline 25.25 → **28.00 / 45 at iter 4** (9-dim isolated audit, n=8 plans).
This is the **first 30B-trained variant to beat σ** in the D5 program. ε (235B+ref)
ceiling = 33.62.

**Trajectory** (9-dim isolated audit per iter on EVAL plans):

| Iter | Mean /45 | vs σ |
|---:|---:|---:|
| 0 | 24.00 | -1.25 |
| 1 | 24.75 | -0.50 |
| 2 | 24.88 | -0.38 |
| 3 | 27.25 | +2.00 |
| **4** | **28.00** | **+2.75** ← PEAK (production) |
| 5 | 15.62 | -9.62 ← cliff (-12.4 in 1 iter) |
| 6 | ~9.6 (4-dim daemon norm) | -15.7 |
| 7 | ~9.8 | -15.4 |

Pre-peak monotonic ascent; sharp cliff in 1 iter — matches Hübotter 2026 §4 multi-round
instability description.

**Per-dim diff iter 0 → iter 4** (broad-spectrum lift, 8 of 9 dims improve):

| Dim | iter 0 | iter 4 | Δ |
|---|---:|---:|---:|
| U1 Soundness | 2.9 | 3.2 | +0.3 |
| U2 Significance | 3.1 | 3.5 | +0.4 |
| U3 Originality | 2.5 | 2.9 | +0.4 |
| U4 Clarity | 3.6 | 3.5 | -0.1 |
| **U5 Reproducibility** | 2.2 | 2.9 | **+0.7** |
| T1 Necessity-of-TTT | 2.8 | 3.1 | +0.3 |
| T2 Disentanglement | 2.5 | 2.9 | +0.4 |
| T3 Compute accounting | 1.8 | 2.1 | +0.3 |
| **T4 Reward-hacking awareness** | 2.6 | 3.1 | **+0.5** |

**3-lr ablation (final, locked)**:

| Run | lr | Peak iter | Peak /45 | Collapse iter | Verdict |
|---|---:|---:|---:|---:|---|
| μ-v2 | 1e-5 | (none) | 23.88 (flat) | (none) | Under-trained |
| μ-v3 | 2e-4 | 0 | 24.88 | 6 (collapse 14.38) | Over-trained / catastrophic |
| **μ-v4** | **5e-5** | **4** | **28.00** | **5 (cliff to 15.62)** | **Sweet spot — production** |

**Critique-token-blindness diagnosis** (F4): TEACHER PICK plans absorb the critique's
content target (PUCT formula present in 8/8 PICK plans at iter 1; J_β formula in 7/8;
adv form in 1/1 at iter 1) but STUDENT EVAL plans NEVER include PUCT (0/8 across all
iters). SDPO advantage measures stylistic divergence, not content tokens — content
tokens are too rare in pretraining for per-token advantage to dominate the gradient.
The +2.75 lift comes from broad-spectrum stylistic improvements (U5 reproducibility +
T4 reward-hacking + U2 significance, etc.), NOT from inserting specific formulas.

**Early-stop guard analysis**: set at threshold 3.0 (rolling mean over 2 audits) didn't
fire because the entire collapse happened in a single iter (28.00 → 15.62 = -12.4 drop
between iter 4 and iter 5; rolling check only saw drop = 12.4 from iter 5 alone, not
combined with iter 3 → iter 4 = +0.75). **Lesson learned**: tighten guard to single-iter
drop ≥ 5 in SDPO_RECIPE_v1.md.

**Multi-round instability symptoms cataloged** (post-cliff plans, F3):
- Mixed-script bleed: Chinese, Hebrew, Cyrillic, Korean characters in English text
- XML metadata leaks: `<META>`, `<solution>`, `<structure>` tags as plan content
- Self-referential meta-narration: "The user has asked for a research plan..."
- Hallucinated jargon: "Brobdingnag super-KAGA", "Mudra method", "PROJECTIVE BIU CLAUSE"
- Token-level repetition: identical phrase 3-4 times per line
- Recursive critique: "Critique of the Improvement-Directive Critique"
- Hallucinated benchmarks: "MOSAIc", "SuperTNT", "Hydra-LLM"

**Production checkpoint**: `runs/2026_04_27_mu_v4/checkpoints.jsonl` row for batch=4.
Tinker state path; `kind: "both"` so includes sampler weights.

**SDPO recipe locked**: `knowledge/current/SDPO_RECIPE_v1.md` — reuse verbatim for Phase 3
abstraction-stage SDPO and any future plan-level SDPO experiments.

**Paper materials canonical store**: `paper_materials/` — 6 findings + 5 experiments + 5
figures CSVs + 5 exemplar plans + 5 methodology docs + 2 related-work excerpts + 3
next-steps. Self-contained; future paper-writing should grab from this folder.

**Implications**:

- **D5 paper has a clean positive result**: SDPO + critic transfers content
  successfully at early iters; production checkpoint is iter 4
- **Best-of-early-iter is the right stopping rule**: don't run past peak
- **Phase 3 retrieve-then-generate** can build on iter-4 weights using SDPO recipe v1
- **Multi-round instability is a genuine, documented failure mode** — paper Limitations
  section can cite Hübotter 2026 §4 for prior art

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_27_mu_v4/` (training run + 4-dim daemon audits + checkpoints)
- `runs/2026_04_27_mu_v4_early_iter_audit/` (9-dim isolated per-iter audit)
- `paper_materials/` (paper-writing canonical store)
- Code: `train_mu_v4.py`
- Recipe: `knowledge/current/SDPO_RECIPE_v1.md`

---

## 2026-04-26 (e): μ-v4 — keep SDPO architecture, tune lr to geometric median (5e-5)

**Run**: `runs/2026_04_27_mu_v4/` (in progress)

**Context**: μ-v2 (lr=1e-5) showed healthy advantage signal but zero policy
transfer (PUCT 0/8 student); μ-v3 (lr=2e-4) catastrophically destabilized
the model (iter-6 audit dropped from 24.88 to 14.38). Two endpoints of the
LoRA-r64 lr range tried; the middle untested.

**Decision**: keep SDPO architecture; only tune lr. Choose
`5e-5 ≈ √(1e-5 × 2e-4)` — geometric median.

**Rejected alternative**: CR-v7 + SFT pivot (proposed by Claude after μ-v3
collapse). User feedback: "我看你是想跑一个 critique revise+sft,但是这个不
是特别美丽吧,我觉得最好还是能找到一个比较美丽的比较 rl 的方式,我觉得可以
先调整下 lr 看看". The SDPO mechanism (`advantage = teacher_lp - student_lp`)
is the elegant self-distillation form the project commits to. Switching loss
form would conflate "SDPO doesn't work at any lr" with "we tried v3".

**v4 vs v3 hyperparam diff**:

| Knob | v3 | **v4** | Reason |
|---|---:|---:|---|
| `learning_rate` | 2e-4 | **5e-5** | Geometric median; LoRA-r64 commonly tolerates 1e-5 to 1e-4 |
| `save_every` | 3 | **1** | Every-iter checkpoint to allow rollback if collapse onset detected |
| `audit_drop_threshold` | (none) | **3.0** | Early-stop if rolling-mean isolated audit drops ≥ 3 between two consecutive audit points |

**Total optimization budget**:
- v2: 10 iter × 1 step × lr=1e-5 = 1e-4
- v3: 20 iter × 4 step × lr=2e-4 = 1.6e-2 (160× v2)
- v4: 20 iter × 4 step × lr=5e-5 = 4e-3 (40× v2, ¼ of v3)

**Pass / fail criteria** on final isolated 9-dim ensemble audit:
- Pass (publishable training claim): mean ≥ 26 (beats σ 25.25)
- Acceptable (training neutral): 24 ≤ mean < 26
- Fail: ≤ 22 → SDPO empirically dead at all 3 lr regimes (1e-5, 5e-5, 2e-4 spanning 20×).
  Pivot to Phase 3 retrieve-then-generate as paper's main contribution.

**Owner**: Yuhong Shi
**Artifacts**: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v4.py` (forked from v3, ~30 line diff), in-progress `runs/2026_04_27_mu_v4/`

---

## 2026-04-26 (d): μ-v3 (lr=2e-4) catastrophic collapse — KILLED at iter 12

**Run**: `runs/2026_04_27_mu_v3/`

**Setup**: 20 iter × 4 grad steps × lr=2e-4. Designed to break v2's
under-budget regime (40× more total optimization signal vs v2).

**Failure mode**: `mean_adv` 0.5 → 0.05 looked like convergence but was
distribution co-collapse. Mid-training audit on EVAL plans:
- Iter 0: mean ≈ 24.88 (baseline)
- Iter 3: mean ≈ 21.25 (already degrading)
- Iter 6: mean ≈ **14.38** (catastrophic)

Iter-6 plans showed corruption symptoms: Chinese-character bleed in English
plans, biocultural headers reappearing, thinking-mode JSON leaks. The model's
output distribution had broken; both teacher and student moved together
toward a degenerate region (mean_adv low because they collapsed *together*,
not because student approached teacher's quality).

**Diagnosis**: lr=2e-4 too aggressive for LoRA-r64 with this gradient budget.
Architecture not at fault — same SDPO loss as v2, only lr changed by 20×.

**Decision (user)**: kill at iter 12. Do NOT rely on audit_v3 ISOLATED of
v3's later iters as a verdict on SDPO mechanism — the parameter space was
destabilized, not under-trained. v3's collapse is a hyperparam result, not
a mechanism result.

**Companion finding (mid-train audit, run `runs/2026_04_27_mu_v3_midtrain_audit/`)**:
even before kill, EVAL plan content showed no PUCT/β(s) absorption, matching
the v2 student-side signature. Confirms the under-budget hypothesis was correct
at v2's lr but lr=2e-4 overcorrected past the stable regime.

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_27_mu_v3/` (12 iters, all checkpoints + critic + audit responses)
- `runs/2026_04_27_mu_v3_midtrain_audit/` (per-iter isolated audit responses)

---

## 2026-04-26 (c): μ-v2 diagnostic — SDPO signal healthy, transfer ≈ 0

**Run**: `runs/2026_04_26_mu_v2/` (committed `b10d71a`); audit
`runs/2026_04_26_phase2F_audit_isolated/`

**Setup**: 10 iter × 1 grad step × bs=8 × lr=1e-5. Plan-level SDPO + Opus
critic + oracle_v2_slim.

**isolated 9-dim audit verdict**: μ-v2 = 23.88 / 45 vs σ = 25.25 / 45 →
training **under-performs frozen+oracle by 1.37 points**. Other baselines:
δ 25.12, α 25.00, β 22.75, ξ 15.62, ε 33.62.

**Smoking-gun diagnostic** — PUCT mention pattern in critic-iteration-aware
trajectory:

| Iter | Teacher (with critique): PUCT mentions | Student EVAL (no critique): PUCT mentions |
|---:|:---:|:---:|
| 0 | 0/8 | 0/8 |
| 5 | 4/8 | 0/8 |
| 9 | 8/8 | 0/8 |

Teacher absorbed the critique (PUCT 0 → 8/8 across iters). Student EVAL
channel never picked it up. The critic kept asking for the same content
across iters because the policy never moved.

**SDPO signal health was fine**: `mean_adv` ≈ 0.4-0.5, `pos_frac` ≈ 80%
throughout. So SDPO computed advantages correctly; the policy just received
too few/too small Adam updates to actually move.

**Hypothesis**: gradient budget under-sized. `10 iter × 1 step × bs=8 × lr=1e-5`
is roughly 100× too small to move LoRA-r64 weights enough to change generation
on prompts that don't include the critique. Test via μ-v3 (20× lr). If μ-v3
still shows zero student transfer with healthy advantages, SDPO mechanism
itself is the bottleneck.

**Owner**: Yuhong Shi
**Artifacts**:
- isolated audit summary: `runs/2026_04_26_phase2F_audit_isolated/audit_v3_isolated_summary.md`
- mid-train PUCT diagnostic: notebook `runs/2026_04_27_mu_v3_midtrain_audit/` adjacent files

---

## 2026-04-26 (b): audit_v3 ISOLATED — anti-bias balanced-batch anonymized methodology

**Code**: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py`

**Trigger**: Earlier audit attempts had two failure modes — v1 baseline-batched
(7 baselines × 8 plans grouped per batch → Opus templated within batch, all
plans in same batch got identical scores); v2 shuffled-but-batched (still
session-anchored, ~50% identical-vector rate). Neither resolved within-tier
discrimination.

**Methodology**:
1. Load 56 plans (7 baselines × 8 plans).
2. Build **8 balanced batches**: each batch contains exactly 1 plan from
   each of the 7 baselines (so a batch is heterogeneous, not homogeneous).
3. Within batch: shuffle order, anonymize identifiers (`batch_NN_plan_A` ..
   `_plan_G`).
4. Dispatch 8 **parallel** Opus subagents (one per batch). Each subagent
   has independent calibration anchors.
5. Each subagent scores 7 anonymized plans on 9-dim rubric (5 universal +
   4 subfield) with strict per-plan independence instructions including a
   `key_differentiator` field forcing differentiation.
6. Collect 56 judgments, decode anonymizers via shuffle map, aggregate
   per-baseline mean.

**Verification**: each plan gets exactly one score (no double-counting), but
8 different Opus sessions provide differentiated perspectives. Within-baseline
distinct-totals rate is now high (vs v2's saturation).

**Status**: this is the canonical D5 final-evaluation methodology going forward.
All baseline rankings cited in 2026-04-26 sections of STATUS.md derive from
this audit.

**Cost / wall time**: ~$10-15 per full 56-plan audit, ~6-10 min wall (parallel).

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_26_phase2F_audit_isolated/` (8 batch requests, 8 subagent
  responses, shuffle_map.json, audit_v3_isolated_summary.md)
- `runs/2026_04_26_phase2F_audit_shuffled/` (v2 attempt, kept for diff)

---

## 2026-04-26 (a): bibliography rebuilt from LaTeX source (replaces flawed PDF parse)

**Trigger**: Phase 2A v1 PDF parsing silently dropped 17/87 references due to
"title.Venue" no-space concatenation and hyphenation in author names; OpenAlex
title-search returned wrong matches for ~22 papers. User: "你怎么犯了个这么
大的错误,直接跳过了这么多 reference".

**Decision**: switch from PDF text extraction to LaTeX source pipeline.

**Pipeline (4 stages)**:
1. `extract_cite_keys_v1.py` — regex-extract `\cite{...}` keys from `main.tex`
   → 90 unique cite keys.
2. `parse_bibtex_v1.py` — parse `main.bib` (407 BibTeX entries) and
   cross-reference cite keys → 90/90 entries resolved with author/title/year/venue.
3. `build_bibliography_v2.py` — orchestrator: S2 batch lookup (much faster
   than per-paper GET) + OpenAlex fallback. Critical fix: `_last_name()`
   handles BibTeX `LastName, FirstName` format (initial bug returned first
   names → 30+ false negatives in `_verify_match`).
4. `extract_source_paper_v2.py` — LaTeX → markdown conversion preserving
   equations, replaces corrupted PDF-extracted v1.

**Result**: 86/90 cite keys resolved with metadata; 38 of those have full
text (arxiv-bearing references), pulled via LaTeX→md and stored in
`data/bibliography/full_text/`.

**Modified shared module**: added `S2BackwardRetriever` class with
`fetch_batch()` to `src/co_scientist/shared/paper_retrieval.py`.

**Verification**: spot-checked 5 random resolutions for author/title/year
correctness — all correct. 4 unresolved entries: 2 unpublished workshop
papers without DOI/arxiv, 2 misformatted bib entries (manually noted, won't
chase).

**Owner**: Yuhong Shi
**Artifacts**:
- `data/bibliography/resolved_v2.jsonl` (86 entries)
- `data/source_paper/v2.md` (~76,300 chars clean markdown)
- `data/bibliography/full_text/*.md` (38 LaTeX-→-md converted papers)
- Code: `extract_cite_keys_v1.py`, `parse_bibtex_v1.py`,
  `build_bibliography_v2.py`, `extract_source_paper_v2.py`,
  `expand_full_text_v1.py`, `extract_arxiv_latex_v1.py`

---

## 2026-04-25: REALIGNMENT — all prior runs archived; reviewer-standards-grounded redesign

**Trigger**: User reviewed Phase 0.6 work and concluded multiple structural misalignments:

1. **Oracle abstraction was wrong**: `smoke_v3_oracle_abstraction.md` is a single-round derivation-
   scaffolding extraction from ONE reference plan, not a multi-round bibliography-grounded
   citation-cited extraction simulating "what the full D5 pipeline would produce."
2. **δ/ε mistreated as baselines**: they include the full reference plan in prompt → semi-oracle
   conditions, not fair baselines for trained models.
3. **Audit prompts had bias**: v1 surface-form bias (Pattern-X scaffolds inflate scores), v2 floor
   saturation (-1/dim cap kills within-tier discrimination).
4. **No reviewer-standard grounding**: all audit dimensions invented top-down rather than derived from
   what real ML/AI conference reviewers focus on.
5. **No user-alignment protocol**: previous runs went ahead without user reviewing exact prompts,
   pipelines, decision rules.

**Decision**: full realignment with new protocol.

### Phase 1 (research + co-design, COMPLETE 2026-04-25)

Three files created in `knowledge/current/`:

- **REVIEWER_STANDARDS_v1.md**: synthesis of 11 sources covering NeurIPS/ICLR/ICML 2025 reviewer
  forms, NSF Merit Review criteria, Stanford CS230 rubric, reviewer-bias literature (Cortes-Lawrence,
  NeurIPS 2021 Consistency, Stelmakh-Shah), and 7 OpenReview-reviewed adjacent papers in test-time-
  training / search-with-LLMs / self-improvement subfield (MTTT, ReST-MCTS, ReST-EM, Voyager, SCoRe,
  Guided-ReST, DeepEvolve). Identifies universal reviewer-focus dimensions (5) + subfield-specific
  go/no-go criteria (8).

- **ORACLE_DESIGN_v1.md**: locked design — 6 categories (Insights / Methodology / Theory / Math /
  Empirical / Failure-modes); Opus 4.7 generates as ceiling estimate; 3-round cumulative-memory
  extraction across full TTT-Discover bibliography (~30-60 papers via S2 API); per-item full
  bibliographic citation; 0-3 items per category per paper; title+abstract+S2 TLDR for all + full text
  for top-5 most-central.

- **AUDIT_RUBRIC_v3.md**: locked design — 9-dim hybrid (5 universal: Soundness/Significance/Originality/
  Clarity/Reproducibility + 4 subfield-specific: Necessity/Disentanglement/Compute/Reward-hacking);
  Option B anchors with verbatim quotes from MTTT/Voyager/Guided-ReST/SCoRe reviewers; equal-weight
  /45 raw or /20 normalized; no anti-pattern penalty (anchors encode requirements); two-pass procedure
  (claim_list + concern_list → per-dim scoring).

### Archive

All `runs/2026_04_*` and `runs/2026_04_25_*` directories moved to
`runs/_archive_pre_realignment_2026_04_25/` with explanatory README. Code (src/) NOT archived; new
v2 versions will be written as design changes. Pairwise tournament data preserved as methodology
demonstration but per-baseline rankings are NOT authoritative.

### Pre-run protocol (binding for all future runs)

`knowledge/current/RUN_CONFIRMATION_TEMPLATE.md` defines required spec: run identity, pipeline
diagram, verbatim prompt template, what model sees vs doesn't, eval metric + comparison rule,
decision rule, cost breakdown, risks. Filled out and approved by user before EVERY launch. NEVER
skip.

### Locked design choices (user-approved 2026-04-25)

- Oracle: 6 categories, Opus-generated, multi-round 3-round cumulative
- Audit: 9-dim hybrid (5 universal + 4 subfield), Option B real-reviewer-quote anchors, equal weight,
  no anti-pattern penalty
- α/β eval prompt: realigned to use `goal + new_oracle` (same as σ/μ) for fair within-pipeline comparison
- Bibliography source: real fetch via S2 API
- Greek-letter naming: ξ (frozen + goal only baseline), σ (frozen + oracle), μ (trained + oracle),
  α/β (training-method ablations), δ/ε (reference points, NOT baselines)

### Next phase

Phase 2 (build) plan to be drafted separately. Will cover: Phase 0b S2 API extension, multi-round
Opus oracle extraction script, audit_v3 prompt + parser, daemon v3 addendum, fresh runs of
ξ/σ/μ/α/β under new oracle + audit. Each individual run gated by RUN_CONFIRMATION_TEMPLATE.

**Owner**: Yuhong Shi
**Artifacts**: 3 design docs in `knowledge/current/`, archive README at `runs/_archive_pre_realignment_2026_04_25/README.md`

---

## 2026-04-25: D5 re-evaluation via FAIR pairwise — D5 training dead, inference scaffolding revived

**Run**: `runs/2026_04_25_pairwise_v1/fair_pairwise_summary.md` (5 matchups appended to existing tournament)

**Trigger**: User asked to verify whether previous "D5 dead" conclusion (based on pairwise μ losing 8-0 to δ) was reliable, and whether agents had misjudged.

**Two-part diagnosis**:

1. **Pairwise data IS reliable** (Explore agent verified): 7/7 sampled verdicts contain rationales referencing actual plan-text content, no hallucinations (FoldX/RosettaFold/CMIP6 all verified in real ε plans). Position bias mild (50-75% A-wins). Subagent reads plans carefully.

2. **Comparisons μ-vs-δ etc. were UNFAIR** (Explore agent confirmed by reading prompt-construction code):
   - δ / ε prompts include the FULL reference_solution.txt as few-shot example (semi-oracle condition)
   - μ / α / β / smoke_v3_A / smoke_v3_B all generate WITHOUT reference plan in prompt
   - "δ wins 8-0 vs μ" is "model with answer in prompt > model without answer" — not a training-method comparison

**5 fair pairwise tests** (all with NO reference plan on either side, seed=45):

| Matchup | Result | Implication |
|---|---|---|
| smoke_v3_A vs smoke_v3_B | A 8-0 | **Oracle inference scaffolding HELPS** |
| μ vs smoke_v3_A | smoke_v3_A 6-2 | SDPO training DOES NOT add value over oracle inference |
| μ vs smoke_v3_B | μ 8-0 | Training+oracle ≫ bare frozen (but lift is from oracle) |
| α vs smoke_v3_B | smoke_v3_B 5-2 | Opus distillation HURTS |
| β vs smoke_v3_B | smoke_v3_B 6-2 | Ref SFT HURTS |

**Definitive corrected narrative**:

1. **Inference-time oracle abstraction works** (smoke_v3_A 8-0 over goal-only baseline) — this is the validated D5 sub-pathway. Originally claimed via absolute audit (+1.88/+4.0/+3.13 over δ), withdrawn after δ-comparison revealed unfair, NOW REVALIDATED via fair pairwise.
2. **Plan-level SDPO training transfer ≈ 0** — confirmed across audit-time-trajectory data (iter 0 ≈ iter 8) AND now via fair pairwise (μ ≤ smoke_v3_A 2-6).
3. **Naive 30B training methods hurt**: α (Opus distillation) and β (ref SFT) both lose to frozen baseline. Model-scale gap + training instability.

**Strategic implications**:

- **D2 pivot ("inference-time scaffolding paper") IS ALIVE** — supported by 8-0 pairwise win of smoke_v3_A over smoke_v3_B
- **D5 training story is dead in current configuration** — μ doesn't internalize, α/β actively hurt
- **Cross-goal generalization is the next test** — pairwise smoke_A vs smoke_B on 2-3 additional goals would establish whether oracle scaffolding generalizes (vs just being a TTT-Discover-specific finding)
- **Audit absolute scoring is unreliable** — re-confirmed via v2 floor saturation; pairwise is the only reliable D5 metric

**Three viable paper paths**:
1. **Inference-time scaffolding** (positive single-goal evidence): "Methodological pattern scaffolds at inference lift 30B research-plan generation; no training needed; pairwise 8-0 vs baseline." Workshop-grade single-goal; full paper if cross-goal extends.
2. **Methodology** (absolute audit pitfalls): "Surface-form bias in absolute long-form audit; semi-oracle prompt unfairness; pairwise required."
3. **Combined**: "Where compute should go in long-form generation: inference scaffolding > parameter updates at 30B scale." Combines positive oracle result + negative training result + audit methodology.

**Recommended immediate next step**: cross-goal validation of oracle-scaffolding pathway. Pick 2-3 D3-style research papers (post-Qwen-cutoff), Opus-extract pattern oracles per goal, run smoke_A vs smoke_B pairwise on each. ~$30, ~1 day. If 2/3 generalize → workshop paper viable.

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_25_pairwise_v1/fair_pairwise_summary.md` — full decision-matrix report
- `runs/2026_04_25_pairwise_v1/pairwise_responses/{mu_vs_smokeA, mu_vs_smokeB, alpha_vs_smokeB, beta_vs_smokeB, smokeA_vs_smokeB}.json`
- Code: `src/co_scientist/d5_abstract_retrieve_refine/fair_pairwise_v1.py`

---

## 2026-04-25: audit prompt v2 redesign — FAILED strict gate; pairwise installed as primary metric

**Run**: `runs/2026_04_25_audit_v2_validation/`

**Motivation**: v1 absolute audit had surface-form bias confirmed by pairwise tournament (3 inversions). User asked for redesigned absolute audit that produces ground-truth-aligned scores resistant to surface-form gaming.

**v2 design**: two-pass mandatory enumeration (`claim_list` + `scaffold_list` with verbatim ≤25-word quotes) + anti-pattern penalty (-1/dim cap, floor 1) + new substance-density anchors. Backward-compatible 4×1-5 → /20 schema. Per-plan output JSON expanded with claim/scaffold lists for auditability.

**Validation methodology**: re-score 48 plans (6 baselines × 8 plans, namespaced by label since μ/β share `iter_NNN_eval_K` IDs) under v2, compare to 7-matchup pairwise ground truth via directional agreement.

**Two iteration outcomes**:

| Iter | Directional | All 3 v1-inversions flipped? | New failures |
|---|:-:|:-:|---|
| v2 try 1 | 5/7 | ✓ | mu_vs_beta inverted (β>μ); beta_vs_alpha still β>α (was tied in pairwise) |
| v2 try 2 (sharper non-Pattern-N scaffold detection: vague hparams, passive verbs, generic algo names) | 5/7 | ✓ | mu_vs_beta tied at 4.0 (both floor); beta_vs_alpha now α>β (also wrong direction) |

**Strict acceptance gate** (per user 2026-04-25 plan choice): ≥6/7 directional + 3 inversions flipped. **Both iterations: FAIL on directional (5/7), PASS on inversions.**

**Per-baseline v2-iter2 means** /20:

| Baseline | total | math | novelty | realism | rigor |
|---|---:|---:|---:|---:|---:|
| epsilon | 14.38 | 3.62 | 3.00 | 4.12 | 3.62 |
| delta | 7.38 | 2.00 | 1.50 | 2.88 | 1.00 |
| alpha | 4.62 | 1.00 | 1.00 | 1.50 | 1.12 |
| mu | 4.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| beta | 4.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| smoke_v3_A | 4.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Diagnosis**: -1/dim cap + floor=1 makes all 4 dims hit 1 when a plan has even one scaffold per dim. 30B-trained plans have multiple scaffolds across dims → all bottom out at 4/20. Mathematical ceiling on within-bottom-tier discrimination — this is a property of the schema, not Opus error. Pairwise truth says μ > β (8-0); v2 cannot capture this because both plans saturate the floor.

**Pre-registered fallback applied**: per plan `steady-tinkering-wave.md`, "If 2nd attempt fails, abandon prompt-redesign route and switch primary metric to pairwise-only." Triggered.

**Decision**:
1. **Abandon absolute-prompt-redesign as primary metric path**.
2. **Pairwise tournament installed as primary metric** for D5 within-scale comparisons going forward.
3. **v2 retained as secondary coarse tier-level signal** (works perfectly for ε ≫ δ ≫ 30B-bottom-tier discrimination at $0 marginal cost since infrastructure is built).
4. **Phase 2 (re-audit historical baselines under v2) SKIPPED** — no point re-scoring with a metric that floors the bottom tier.

**Operational implication for D5 evaluation**:
- Cross-model-scale (ε vs δ): v1 or v2 absolute acceptable
- Within-scale training-method (μ vs β, future Phase 1 ablations): pairwise REQUIRED, ~10× cost than absolute but proven reliable

**What v2 successfully demonstrates** (positive findings):
- Tier discrimination is solid: ε=14.4 ≫ δ=7.4 ≫ 30B-trained-bottom (4-5)
- All 3 v1-induced inversions (μ-δ, smokeA-δ, α-δ) flip to point correctly under v2
- Per-dim signal: math/novelty/realism all show δ > μ as expected; rigor stays flat (all 30B plans lack benchmark numbers)

**What v2 cannot do**:
- Reliably rank within the bottom tier (μ vs β vs α)
- Replace pairwise for fine-grained ablations

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_25_audit_v2_validation/audit_v2_concordance.md` (final report)
- `runs/2026_04_25_audit_v2_validation/audit_log_v2.jsonl` (per-plan v2 scores)
- `runs/2026_04_25_audit_v2_validation/audit_responses/` (iter 2 prompt responses)
- `runs/2026_04_25_audit_v2_validation/audit_responses_v2try1/` (iter 1 archive for diff)
- Code: `src/co_scientist/shared/audit_prompt_v2.py`, `src/co_scientist/d5_abstract_retrieve_refine/validate_audit_v2.py`
- Daemon spec: `projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon_v2_addendum.md`

**Methodology insight (publishable)**: A 4×1-5 absolute rubric with anti-pattern penalty cannot achieve within-tier discrimination once scaffold-density saturates. Pairwise tournaments are required for fine-grained long-form generation evaluation when comparing similarly-scaled models. Surface-form bias in absolute audits is real and reproducible (3/7 inversions on the same plans, fixed by pairwise).

---

## 2026-04-25: ε vs δ pairwise — 235B>>30B holds (8-0); scale advantage real

**Run**: appended to `runs/2026_04_25_pairwise_v1/` (matchup `delta_vs_epsilon`, seed=44)

**Motivation**: After confirming pairwise inverts μ/smoke_v3_A vs δ (absolute-grader surface-form bias), check whether the 235B-vs-30B gap (absolute +5.38 ε > δ) is also a grader artifact — same prompt format on both sides, only model size varies.

**Result**: **ε wins 8-0**. Same direction as absolute, so the +5.38 gap is **real substance**.

**Subagent rationale**:
- ε (235B): tighter formulas with named variables + decay schedules; internally consistent budget bookkeeping (eval count, batch size, LoRA rank, lr fit together); named real tools + quantitative baseline deltas (FoldX, Rosetta, NUPACK, Lean, AlphaFold2, CMIP6, GFP; merit factor 13.8→14.5, bandgap 1.32→1.25 eV); occasional algorithmic insights (rank-weighted PG, self-critique replay, recursive Φ with lineage blocking)
- δ (30B): contradictions (claims expensive eval but budgets 10k evals; invokes meta-learners despite single-problem premise); near-uniform "memory buffer + reward-weighted PG + LoRA" template; one plan truncated mid-paragraph (30B occasionally hits gen budget)

**Final 30B vs 235B picture**:
- ε ceiling = high (substantive, distinct, real)
- δ ceiling = medium (concrete because of ref content, but limited by 30B's reasoning depth)
- 30B trained = no path above δ in any of our configurations

**Updated full pairwise matrix (7 matchups)**:

| Matchup | Result | Confirms absolute? |
|---|---|---|
| ε vs δ | 8-0 ε | YES (real scale advantage) |
| ε vs μ | 8-0 ε | YES |
| δ vs μ | 8-0 δ | NO — INVERTED (absolute had μ>δ) |
| δ vs smoke_v3_A | 8-0 δ | NO — INVERTED |
| δ vs α | 8-0 δ | YES |
| μ vs β | 8-0 μ | YES |
| β vs α | 4-4 tie | partial (absolute β>α) |

**Pairwise total order**: ε ≫ δ ≫ μ ≈ smoke_v3_A > β ≈ α

Three substance tiers — clean separation:
1. ε (235B+ref) — substantively distinct, sweeps everyone
2. δ (30B+ref) — 30B's hard ceiling
3. 30B trained/scaffolded — all fail to clear δ

**Crystal-clear D5 truth**: on this task, on Qwen3-30B, with this grader (Opus pairwise), **parameter scale dominates training method**. Going from 30B to 235B at fixed prompt format buys you ~5.4 absolute points + 8/8 pairwise. None of our 30B training/scaffolding configurations buy anything over δ.

**Implications for paper direction**:
1. **Negative-result methodology paper** is well-supported now: (a) absolute audit has surface-form bias on 30B-trained variants (3/7 matchups inverted or downgraded); (b) absolute-grader matches pairwise when comparing across model scales (ε vs δ); (c) implies absolute audit is reliable for cross-scale comparison but unreliable for within-scale training-method comparison. This is a useful methodology contribution for the broader long-form RL literature.
2. **D5 cannot make a 30B training claim** with this evaluation framework. The 30B ceiling is δ; nothing inside D5's design space (oracle abstraction, plan-level SDPO, distillation) gets past it.
3. **Phase 1 multi-paper retrieval** would need to inject mechanism details NOT in `reference_solution.txt` to clear δ. But the bibliography papers (e.g. AlphaEvolve, AlphaZero, etc.) are mostly about *adjacent* methods, not the specific TTT-Discover mechanism. Likely insufficient.
4. **A1 / C2 explicitly DEAD** as competitive paths against δ — both work inside the oracle-abstraction regime that pairwise rejected.

**Surviving D5 directions**:
- (a) **Negative-result paper** focusing on grader-bias finding + "pure-prompt > training" conclusion. Workshop venue.
- (b) **Pivot to bigger model** (235B base, not 30B) — but Tinker LoRA on 235B is expensive and the "training internalizes critique" claim is even harder (235B already strong).
- (c) **Pivot off long-form research plan** entirely — back to D3-style per-goal optimization where the eval grader is verifiable (not Opus-judged), or grant-proposal generation where rubric is public.
- (d) **Pause D5 and re-survey** before more code investment.

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_25_pairwise_v1/pairwise_requests/delta_vs_epsilon.json`
- `runs/2026_04_25_pairwise_v1/pairwise_responses/delta_vs_epsilon.json`
- `runs/2026_04_25_pairwise_v1/matchups_meta.json` (extended)

---

## 2026-04-25: smoke v3 pathway claim WITHDRAWN — pairwise δ > smoke_v3_A 8-0

**Run**: appended to `runs/2026_04_25_pairwise_v1/` (matchup `smokeA_vs_delta`)

**Motivation**: μ vs δ pairwise inverted. μ uses the same oracle abstraction as smoke v3-rerun. So the smoke pathway claim (smoke_v3_A absolute audit = 9.25 > δ = 6.75 by +2.5) is suspect. Tested directly with pairwise.

**Setup**: 8 plans from `runs/2026_04_smoke_pathway_v3_rerun_div/buffer.jsonl` (A_with_abstraction) vs 8 δ plans, 8 random pairs, position-randomized, seed=43, Opus subagent picks A/B/TIE.

**Result**: **δ wins 8–0**. Same direction and same magnitude as μ vs δ.

**Subagent rationales**: δ plans (paraphrasing reference plan) consistently provided explicit equations + LoRA rank/learning rate + named benchmark numbers. smoke_v3_A plans converged to the same 5-pattern boilerplate (objective derivation / accumulated-state reuse / warm-start selection / adaptive hyperparameters / importance sampling correction) — clean structural scaffolds without mechanistic depth.

**Conclusion**: **Oracle abstraction at inference time does NOT beat reference plan in context, on this task with this grader.** All three smoke runs (v1 +1.88, v2-rerun +4.00, v3-rerun +3.13) had the same grader artifact. The smoke pathway claim must be withdrawn.

**Updated 30B configuration ranking** (pairwise, all approaches frozen Qwen3-30B unless noted):

| Approach | vs δ pairwise | Conclusion |
|---|---:|---|
| δ (frozen + ref in prompt) | — | 30B ceiling |
| smoke v3 A (frozen + oracle abs) | 0–8 (δ wins) | claim withdrawn |
| μ (SDPO + oracle, 10 iters) | 0–8 (δ wins) | training adds nothing |
| α (Opus distill, 5 epochs × 16 plans) | 0–8 (δ wins) | distillation hurts |
| β (direct ref SFT, 10 steps) | tied with α | SFT under-fits |

**Real D5 truth**: on frozen Qwen3-30B, putting `reference_solution.txt` directly in the prompt IS the ceiling. Every D5 training/scaffolding trick in Phase 0.6 fails to clear that bar. The only way 30B beats δ would be via **mechanism that injects content δ doesn't already see** — but δ already has the full reference plan; nothing else we tried adds substance Opus pairwise rewards.

**Implications for D5 paper**:
1. **D2 pivot fully dead**: "inference-time scaffolding paper" relied on oracle abstraction beating δ. Pairwise says no.
2. **A1 (span-level SDPO) and C2 (differential critique) become weaker**: both still operate inside the oracle-abstraction-prompt regime that pairwise rejected.
3. **Negative result paper viable**: "Across 5 D5-aligned configurations on 30B, none beat the simple ref-in-context baseline. We attribute the apparent absolute-score lifts to surface-form bias in the standard depth-audit grader, demonstrated by pairwise inversion." This is publishable as a methodology contribution at a workshop, with the grader-bias finding as the key insight.
4. **Genuine path forward for D5 (NeurIPS/ICLR-grade)**: needs a mechanism that injects content NOT in the reference plan — e.g.
   - **Multi-paper retrieval that supplements ref content**: bibliography papers contain mechanism details NOT in the reference plan's plan section. This is Phase 1 of the original plan, but ONLY if pairwise-evaluated and only if it provides content beyond ref.
   - **External grounding (web/code)**: not just paper bibliography but real benchmarks results + code repositories. Would actually fight the grader's "specific numerical claims" axis.
   - **Pivot to a different problem**: long-form research plan generation may be a red flag — δ proves "a 30B model + the answer in the prompt" is ceiling; this is a known result and not novel.

**Recommendation**: pause D5 framework iteration. Re-evaluate research direction with this finding before investing more.

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_25_pairwise_v1/pairwise_requests/smokeA_vs_delta.json` (request, seed=43)
- `runs/2026_04_25_pairwise_v1/pairwise_responses/smokeA_vs_delta.json` (8 verdicts)
- `runs/2026_04_25_pairwise_v1/matchups_meta.json` (extended with smokeA_vs_delta)
- `/tmp/build_smoke_vs_delta_pairs.py` (one-off generator script — could move to repo if needed)

---

## 2026-04-25: Phase 0.6 — pairwise sanity check INVERTS μ vs δ; D5 narrative needs revision

**Run**: `runs/2026_04_25_pairwise_v1/`

**Motivation**: Phase 0.6 absolute audit gave μ (8.88) > δ (6.75) by +2.13 — claimed evidence that oracle abstraction lifts 30B above ref-in-context prompting. User flagged that absolute scores may not reflect true preference. Ran 5-way pairwise tournament as sanity check.

**Setup**: 5 matchups × 8 random pairs (position-randomized), Opus picks A/B/TIE per pair, all via subagent file-bus. ~$0 OpenRouter (all subagent dispatches).

**Results**:

| Matchup | Pairwise | Absolute | Inversion? |
|---|---:|---|---|
| μ vs ε | 0–8 (ε wins) | μ=8.88 vs ε=12.13 | ✓ confirms (ε > μ) |
| **μ vs δ** | **0–8 (δ wins)** | μ=8.88 vs δ=6.75 | **🚨 INVERTED** |
| μ vs β | 8–0 (μ wins) | μ=8.88 vs β=7.75 | ✓ confirms (μ > β) |
| α vs δ | 0–8 (δ wins) | α=6.12 vs δ=6.75 | ✓ confirms (δ > α) |
| β vs α | 4–4 (tie) | β=7.75 vs α=6.12 | partial (won → tied) |

**Pairwise ranking**: ε > δ > μ > β ≈ α
**Absolute ranking**: ε > μ > β > δ > α

**Causal reading from subagent rationales**:
- μ plans inherit oracle abstraction's "Pattern X" structural framing — visually rigorous, novelty +1, but lacks concrete numerical specifics
- δ plans paraphrase the reference plan content (LoRA rank, learning rate, MAX-PUCT equation form, AHC039 / Erdős baseline numbers) — concrete substance
- Opus absolute audit gives "Pattern X" structure novelty + math credit (rubric levels 2-3); when forced to PICK, Opus consistently prefers δ's concrete specifics
- "Pattern" structure is a Goodhart axis on absolute audit but not on pairwise

**Implications for D5 paper**:

1. **D5 (μ) does NOT beat δ at substance** — earlier "+2.13 over δ" finding is grader artifact at the absolute-score boundary. Single biggest implication: oracle abstraction at inference time does NOT lift 30B above frozen+ref-in-context prompting. Smoke v3-rerun A=9.25 vs δ=6.75 likely has the same artifact.

2. **D5 training mechanism (μ vs β)** IS real — 30B trained with plan-level SDPO + oracle beats 30B SFT'd on reference (8–0 pairwise). But this is a 30B-vs-30B comparison; β is a weak baseline.

3. **α partially recovers** — pairwise α-β = 4-4 tie (vs absolute β > α by +1.63). "Opus distillation hurts the 30B model" is softened: α is comparable to β at substance level, both lose to δ.

4. **The grader-gap problem extends within Opus itself** — not just Qwen vs Opus, but Opus-absolute vs Opus-pairwise. Opus pairwise ≈ ground truth; Opus absolute on /20 has surface-form bias on this task.

5. **Phase 0.6 decision matrix needs re-reading**: μ > δ on absolute was the pillar of "D5 architecture justified". With pairwise δ > μ, the architecture's only proven win is μ > β (i.e. SDPO+oracle > naive SFT-on-ref). δ > μ means the SIMPLEST inference-time scaffolding (just put reference plan in prompt) is better than the trained system.

**Recommendations for next steps**:

- **Audit grader rebuild required**. Either (a) switch primary metric to pairwise tournament for all D5 evaluations (cost: ~10-30× more Opus calls but reliable), or (b) modify absolute rubric to penalize surface-form-without-substance (very hard to define).
- **Smoke v3-rerun re-evaluation**: re-run with pairwise to confirm δ > A pattern. If confirmed, smoke pathway claim needs withdrawal.
- **D5 mechanism question redirected**: instead of "does training internalize critique", ask "does training make 30B beat δ-in-context?" Pairwise says no for current configuration.
- **Possible pivots**:
  - **A1 span-level SDPO** still viable — narrows gradient to "concrete substance" tokens that Opus pairwise rewards
  - **C2 differential critique** — critic forces specific numbers/equations into output, addressing exactly what δ has and μ lacks
  - **D2 pivot to inference-time scaffolding paper** — REJECTED by this finding (oracle abstraction doesn't beat δ), so D2 is no longer viable as a fallback

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_25_pairwise_v1/{matchups_meta.json, pairwise_summary.md, pairwise_requests/, pairwise_responses/}`
- Code: `src/co_scientist/d5_abstract_retrieve_refine/pairwise_prefs_v1.py`, `src/co_scientist/shared/opus_pairwise_subagent.py`

---

## 2026-04-25: Phase 0.6 — α + β baselines complete, full kill matrix passes

**Runs**:
- α: `runs/2026_04_25_alpha_baseline_v1/` — 5 epochs SFT × 16 Opus plans (8 with paper, 8 without)
- β: `runs/2026_04_25_beta_baseline_v1/` — 10 SFT steps on the single reference plan

**Final baseline ranking on Opus D3-canonical /20**:

| Baseline | Score | Setup |
|---|---:|---|
| ε | 12.13 | frozen Qwen3-235B + ref in context |
| **μ** | **8.88** | plan-level SDPO + oracle v3 abstraction (10 iters, 8 plans/iter) |
| β | 7.75 | direct SFT on (goal → reference plan), 10 steps × 1 example |
| δ | 6.75 | frozen Qwen3-30B + ref in context |
| α | 6.12 | naive Opus distillation, 5 epochs × 16 plans |

**Trajectories** (per-iter audit means):
- α: 6.62 → 7.00 → 6.38 → 5.88 → 6.12 (no learning trajectory; mild oscillation)
- β: 7.88 → 8.00 → 7.75 → 7.88 → 7.63 (slight downward drift, no overfit-to-ref)
- μ: 9.13 → 8.88 → 9.38 → 8.00 → 9.25 (oscillation around 9.0)

**Kill matrix outcome**:

| Kill condition | Threshold | Actual | Triggered? |
|---|---|---|---|
| μ ≤ δ + 1 | μ ≤ 7.75 | μ = 8.88 | NO |
| α ≥ μ | α ≥ 8.88 | α = 6.12 | NO |
| β ≥ μ | β ≥ 8.88 | β = 7.75 | NO |

**ALL THREE KILL CONDITIONS PASS** — D5 architecture is the strongest 30B baseline by audit, beating both direct SFT on reference (+1.13) and Opus distillation (+2.76).

**Surprising finding: α < δ** (6.12 < 6.75). Naive Opus distillation HURT the 30B model below the frozen baseline. Hypothesis: (a) model-scale gap — Qwen3-30B can't produce Opus-quality reasoning even after SFT on Opus tokens; SFT pushes toward Opus surface form but reasoning-capacity ceiling holds; (b) Qwen3 thinking-mode bleed: 16 Opus plans (no `<think>` blocks) trained against Qwen3 chat template may have damaged the model's <think>-then-answer routine. β had similar SFT structure but only 1 example × 10 steps — less aggressive overfit pressure. α's 16 examples × 5 epochs = 80 effective updates may have moved the model further from base. Worth re-checking iter-0 vs iter-4 generations side-by-side to confirm.

**Implications for D5 paper**:
- D5 (μ) is the BEST 30B configuration in this baseline matrix → architecture is justified
- BUT μ - δ = +2.13 is largely from oracle abstraction at INFERENCE time; the 8 SDPO training iters added ≈ 0 over iter 0 audit (9.13)
- Paper claim should be: "structured oracle abstraction + inference-time critique-aware prompting lifts 30B above frozen baselines; SDPO training on top adds zero/marginal value at this configuration"
- Phase 0b/1 (multi-round retrieval + abstraction generator) needs to add ≥2 pts to be worth ICLR — currently no path to that increment

**Cost**:
- α: ~$0 (Opus plans generated by 1 main subagent in-session, no OpenRouter); ~10 min Tinker
- β: ~$0 (no Opus during training); ~5 min Tinker
- 5 audits × 2 baselines = 10 audit subagent dispatches (n=8 plans each)

**Operational notes**:
- α SFT used renderer.build_supervised_example which returns (ModelInput, weights) tuple — needed unpacking
- β trained on 1 example × 10 steps; under-fitting (no verbatim copying of ref). Trying 30 steps might overfit but unclear it helps audit.
- Both α and β dispatch all audits in 1 subagent call after collect_all blocks — saves dispatch overhead

**Next-step candidates** (per user's earlier 4-option discussion):
1. **A1 span-level SDPO** — narrow gradient to "where critique pointed at gaps" tokens; expected lift if μ ceiling is attribution mismatch
2. **C2 differential critique** — critic outputs what's MISSING vs what's WRONG; student must include K specific points next iter
3. **D2 pivot** — re-frame D5 as inference-time scaffolding paper (oracle abstraction + critique-aware prompting beat ε on cost-per-token); μ ≈ 8.88 vs ε = 12.13 at 8× lower inference compute is a defensible 3-page workshop story

**Owner**: Yuhong Shi
**Artifacts**: 
- `runs/2026_04_25_alpha_baseline_v1/{config,metrics,audit_log,eval_rollouts,run.log}.{json,jsonl}` + `audit_responses/iter_*.json`
- `runs/2026_04_25_beta_baseline_v1/...` (same structure)
- `data/plans/opus_v1.jsonl` (16 plans, ~72KB)
- Code: `train_alpha_baseline_v1.py`, `train_beta_baseline_v1.py`

---

## 2026-04-25: Phase 0.6 — μ baseline complete; MARGINAL outcome, training transfer ≈ 0

**Run**: `runs/2026_04_25_mu_baseline_v1/`. n_iter=10 × n_plans=8 × oracle v3 × anchor_ce=0.0 (pure SDPO ablation) × eval_every=2.

**Key numbers**:

| Metric | Value | Interpretation |
|---|---|---|
| μ proxy (last-3 audit means) | **8.88 /20** | Marginal (7.75 < μ < 9.00) |
| Iter 0 audit (no LoRA delta) | 9.13 /20 | Frozen + oracle + cold-start critique |
| Iter 8 audit (8 SDPO steps) | 9.25 /20 | After full training |
| **Training transfer** | **+0.12** | Within grader noise; ≈ 0 effective |
| μ − δ (vs frozen + ref-in-context) | +2.13 | Oracle abstraction itself is the lift |
| μ − ε (vs frozen 235B + ref) | −3.25 | 235B+prompt still wins |

**Audit trajectory** (n=8 each): 9.13 → 8.88 → 9.38 → 8.00 → 9.25 (iters 0/2/4/6/8). Oscillates within ±0.7 of mean ≈ 9.0.

**SDPO signal stability**: With informative critiques, mean_adv ≈ 0.45-0.58, pos_frac ≈ 0.76-0.82. With cold-start critique (iter 0, iter 3 due to iter 2 timeout), mean_adv ≈ 0.21, pos_frac ≈ 0.72. Mechanism is internally consistent — the privileged-info critique IS more informative than cold-start, and SDPO learns to match the with-critique distribution. But this learning does NOT translate into Opus-judged plan quality improvement.

**Per-dim trajectory** (math / novelty / realism / rigor):
- iter 0: 2.13 / 2.0 / 3.0 / 2.0
- iter 8: 1.38 / 3.0 / 2.88 / 2.0
- Math regressed; novelty went up; realism + rigor flat.

**Decision matrix outcome**: MARGINAL. Strict reading of plan doc (μ ≥ 9 to continue Phase 0b/1) says STOP. Loose reading (μ > δ+2) says continue with caveats. The +2.13 lift over δ comes from oracle abstraction at INFERENCE time, not from training.

**What this means for the D5 paper claim**:
- "Privileged-info SDPO at plan level + oracle abstraction" does NOT internalize information beyond what the same model achieves with oracle abstraction in the inference prompt.
- The training mechanism is mechanically working (KL distillation succeeds at the token level — pos_frac stable at 0.8) but the learned behavior does not transfer to better plans by Opus's judgment.
- This is consistent with R-1 risk in the plan: SDPO advantage may be dominated by structural critique-block presence rather than content. Even with handcrafted same-shape cold-start fallback, the model learns to reproduce critique-aware tokens but not to internalize the semantic content.

**Implications**:
1. **Plan-level SDPO with this configuration is not the right mechanism**. Either (a) SDPO needs a different objective formulation (e.g. preference-based instead of distributional KL), (b) the training signal needs to come from a stronger reviewer (Opus is fine, but plan-level KL is too weak), or (c) the fundamental ceiling is "what oracle abstraction can communicate" — and full Phase 1 retrieval+abstraction won't escape it.
2. **D5 narrative needs revision**. The pathway from smoke pathway v3 (+2.5 over δ at inference) holds, but "training internalizes critique" is unsupported. Paper claim becomes inference-time scaffolding, not parameter-update internalization.
3. **Phase 0b/1 should NOT proceed without changing the mechanism**. Building bibliography retrieval + multi-round abstraction generator on top of a non-internalizing SDPO loss adds complexity without addressing the core failure.

**Run cost**: ~$10 OpenRouter (Opus critic ~3-5 min × 10 iters + audit 8 plans × 5 evals; mostly handled by main-agent dispatched subagents) + Tinker LoRA train cost. Total wall clock ~75 min after iter 0 readiness. One iter (2) timed out at 900s.

**Operational notes (carry-forward)**:
- File-bus subagent daemon launched as background Agent did NOT persist as expected — exited after first poll cycle. Workaround for this run: main agent monitored events from training log and dispatched per-request subagents on-demand. For future runs, either (a) increase critic_timeout_sec to allow longer dispatch latency, (b) build the daemon as a Python loop using `claude -p` CLI, or (c) accept main-agent dispatch with monitor-driven flow.
- compute_logprobs_async is a coroutine, not Future — use sync `compute_logprobs(...).result()` in lists for parallel-batched server execution.
- Qwen3 renderer's parse_response returns list-of-dict content; need `_to_str` coercion before regex.

**Owner**: Yuhong Shi
**Artifacts**:
- `runs/2026_04_25_mu_baseline_v1/{audit_log.jsonl, metrics.jsonl, buffer.jsonl, critic_responses/, audit_responses/, checkpoints.jsonl, run.log}`
- Code: `src/co_scientist/d5_abstract_retrieve_refine/{train_mu_baseline_v1.py, mu_prompts_v1.py, extract_paper_v1.py}`, `src/co_scientist/shared/{opus_critic_subagent.py, opus_audit_subagent.py}`
- Daemon spec: `projects/d5_abstract_retrieve_refine/scripts/opus_critic_audit_daemon.md`
- Source paper: `projects/d5_abstract_retrieve_refine/data/source_paper/v1.md`
- Config: `projects/d5_abstract_retrieve_refine/configs/mu_baseline_v1.yaml`
- Plan doc: `~/.claude/plans/steady-tinkering-wave.md`

---

## 2026-04-25: Phase 0.6 baseline battery — δ + ε done; key findings validate D5

**Motivation**: User (2026-04-25) proposed μ baseline (oracle abstraction + plan-level SDPO) as theoretical ceiling estimator. Expanded to run all feasible pre-Phase-1 baselines (δ, ε, β, α, μ) to establish empirical anchors before infrastructure investment.

**δ + ε completed** (2026-04-25, ~2 hours total):

| Baseline | Setup | Opus /20 (D3-canonical) |
|---|---|---|
| δ | frozen Qwen3-30B + reference plan in context, no training | **6.75** |
| ε | frozen Qwen3-235B + reference plan in context, no training | **12.13** |

**Comparison with smoke runs** (same audit framework):
- smoke v1 A (pattern oracle + frozen 30B): 7.00
- smoke v3-rerun A (derivation oracle + frozen 30B): 9.25
- smoke v2-rerun A (derivation + formula hints oracle + frozen 30B): 11.25
- reference plan (gold): 20

**Critical findings**:

1. **D5 abstraction pipeline beats δ by +2.5 pts AT INFERENCE TIME** (v3-rerun A 9.25 vs δ 6.75). The "why not just prompt with reference" reviewer attack is answered: prompt with reference is meaningfully weaker than abstraction-guided inference on the same 30B model.

2. **D5 PATHWAY PREMISE VALIDATED**: before any training, abstraction-guided generation is empirically superior to reference-in-context generation. Phase 1+ training has room to lift further.

3. **ε upper bound at 12.13** (above smoke v2-rerun A 11.25 by +0.88 pts). 235B + reference beats 30B + best oracle abstraction by <1 pt. Paper reframes: "30B trained + abstraction is competitive with 235B + prompting at 8× lower inference cost".

4. **Copying behavior**: 0/16 plans literally copied reference J_β, MAX-PUCT, or specific numbers (Erdős 0.380924, AHC039 566997). All paraphrase. Reference-in-context is NOT a trivial paste attack.

5. **Strongest individual plan**: ε_6 = 15/20 — backward-induction potential Φ(s) = R(s) + γ·max_descendant Φ with lineage blocking + top-k sparsified PG. Closest spiritual echo of reference's MAX-PUCT (but novel form).

**Per-dim breakdown**:

| | math | novelty | realism | rigor | total |
|---|---|---|---|---|---|
| δ | 1.63 | 1.50 | 2.50 | 1.13 | 6.75 |
| ε | 3.00 | 2.88 | 3.75 | 2.50 | 12.13 |

**Implications for Phase 1+**:
- D5 trained 30B expected to fall in **[9.25, 12.13]** range based on this bracketing (v3-rerun A ≤ D5 trained ≤ ε).
- Realistic target: 10-11 /20 (slightly above v3-rerun A, possibly reaching v2-rerun A).
- Paper story clear: D5 is **empirically validated pathway** from 6.75 (prompt baseline) → 10-11 (trained 30B + abstraction) → 12-13 (with bibliography retrieval methodology) → 20 (reference ceiling, unreachable without specific numbers).

**Next baselines in queue**: β (Qwen SFT on reference, 1-2 days), α (naive Opus SFT, 2-3 days), μ (oracle abstraction + plan SDPO, 3-5 days, user's theoretical ceiling test).

**Owner**: Yuhong Shi

**Artifacts**:
- `runs/2026_04_baseline_delta_epsilon/{buffer.jsonl, config_delta.json, config_epsilon.json, opus_depth_audit_d3canonical.json, opus_audit_delta_epsilon_summary.md}`
- `src/co_scientist/d5_abstract_retrieve_refine/baseline_delta_epsilon.py`

---

## 2026-04-25: Session-end summary — Phase 0 + Phase 0.5 complete; ready for Phase 0b/1

**Session span**: 2026-04-23 / 24 / 25.

**5 commits** in this session:
- `26e0ba1` D5 Phase 0a: scaffolding (project dirs, harness files, DIRECTIONS.md + CLAUDE.md updates)
- `58d7f19` D5 Phase 0c smoke v1: pathway validated +9pp Opus, grader gap reproduced
- `a9b2185` D5 smoke v2: derivation+formula-hints abstraction, +5.875 originally (3× v1)
- `2181826` D5 smoke v3: derivation no formula hints, +2.50
- `ff0344f` D5 Phase 0.5: dataset migration + Qwen contamination clear + sampling fix → decomposition inverted

**5 critical findings to carry forward**:

1. **Pathway validated** on D3-canonical Opus audit, 3 independent smoke configurations:
   - Smoke v1 (pattern-only): Δ(A−B) = +1.88 / 20
   - Smoke v2 (derivation + formula hints): +4.00 (after fix; was +5.875 inflated)
   - Smoke v3 (derivation no formula hints): +3.125 (after fix)
   - All A > B per-plan, no overlap, per-dim direction preserved.

2. **Decomposition INVERTED from original claim** (key paper-level finding):
   | Component | Original | Rerun (correct) |
   |---|---|---|
   | Pattern only | +1.88 | +1.88 |
   | + Reasoning steps | +0.62 | **+1.25** |
   | + Formula hints | +3.38 | **+0.88** |
   - Reasoning scaffolding is dominant lift, not formula hints.
   - Original 57% formula-hints attribution was inflated by sample duplicates + generous calibration.
   - Implication: Phase 1+ retrieval can use abstracts-only (cheap), full-text methodology adds only +0.9 pts max.
   - n=8 caveat: gap between v2-rerun and v3-rerun is ~1.2 SE, not statistically distinguishable.

3. **Grader gap reproduced**: Qwen ten_signal self-grader gives Δ = −0.045 (says A < B) on smoke v1, while Opus gives +1.88 (says A > B). D1 rubric within-goal vs Opus = Pearson 0.58 (n=8, not significant). Grader gap is structural, motivates Opus-in-SDPO-loop.

4. **D1 rubric is topic classifier, not quality measure**: 8/10 D1 rubric items check specific framing match; 2/10 trivially pass; 0/10 reward formalism / novelty / realism / rigor. Plan with rubric 0.04 had Opus 9/20 (better than rubric 0.95 plan) — rubric penalized for "wrong framing", not lower quality. Explains D1's 0.693 ceiling: RL can't improve what reward doesn't measure. Cited as motivation for D5 design (Opus-grounded SDPO).

5. **Realistic Phase 1+ ceiling ≈ 9-10/20** (vs reference 20/20). Gap of 10-11 pts is in specific prior AI numbers + hyperparameters + full gradient derivations. Bibliography retrieval can't close this gap regardless of strategy.

**Top 5 open risks heading into Phase 1+**:

1. δ-baseline (frozen 30B + reference in context) likely matches D5 trained — MANDATORY early-warning at Phase 2 end (Decision 2026-04-25 below).
2. ε-baseline (frozen 235B + reference in context) likely beats D5 trained 30B by 2-4 pts — accept; reframe as parameter-efficient.
3. Single-goal scope risk for ICLR — mitigate via 2-3 held-out goals at Phase 4 (without retraining), per Advice 4.
4. Anti-distillation baselines (α/β/γ) must run at Phase 2 (not Phase 4) — early-warning.
5. Template collapse may return at higher entropy on full plan length — verify during Phase 1 unit tests.

**Pipeline state at session-end**:
- Plan: ~/.claude/plans/giggly-jumping-hollerith.md (full plan with 18 ML-scientist objections + 5 advices)
- D5 self-contained dataset: projects/d5_abstract_retrieve_refine/dataset/ (TTT-Discover only)
- Smoke runs: runs/2026_04_smoke_pathway_v{1,2,3}/ (originals) + _rerun_div/ versions (sampling-fixed)
- Code: smoke_pathway_v1.py + smoke_pathway_v1_score.py (sampling fixed: temp=1.0, top_p=0.95, no seed)
- Phase 1 modules: NOT YET BUILT (next session)

**Decision impact for next session**: Phase 0b is unambiguous next action (TTT-Discover bibliography fetch via S2 API extension). Phase 1 module scaffolding follows. Phase 2 must include δ-baseline early-warning checkpoint (added 2026-04-25 to plan).

**Owner**: Yuhong Shi
**Handoff prompt**: `projects/d5_abstract_retrieve_refine/knowledge/current/HANDOFF_2026_04_25.md`

---

## 2026-04-25: Phase 2 δ-baseline early-warning checkpoint added as MANDATORY

**Decision**: After Phase 2 pilot training (10 iters), before Phase 3 commitment, run δ baseline (frozen Qwen3-30B-A3B + reference plan in context, no training) on same held-out evaluation rollouts (n=8). Compare D5 trained vs δ on Opus D3-canonical.

**Decision rule**:
- D5 trained > δ + 2 pts → proceed Phase 3
- D5 trained ∈ [δ, δ+2] → marginal; investigate
- D5 trained ≤ δ → STOP. Pipeline doesn't internalize anything frozen-with-context can't do; redesign or pivot.

**Reason**: δ is the hardest baseline to beat (smoke estimates δ ≈ 8-10 /20, D5 trained ceiling ≈ 9-10 /20). Reviewer will explicitly ask "why train when prompt with reference works as well?" Catching this fail at Phase 2 end (Week 5) prevents 4 weeks wasted Phase 3 training.

**Cost**: ~$5 OpenRouter (Opus audit n=8) + 2-3 hours Tinker (frozen 30B inference).

**Owner**: Yuhong Shi

---

## 2026-04-25: Phase 0.5c — Sampling fix + smoke v2/v3 reruns; decomposition INVERTED

**Result**: Pathway VALID (Δ(A−B) > 0 in both reruns), but **decomposition inverted from original**.

**Changes**:
- SamplingParams: `temperature=0.7 + seed=42` → `temperature=1.0 + top_p=0.95 + no fixed seed`
- Sampling duplicate rate: 31-44% (orig) → 0% (rerun)

**Numbers** (Opus D3-canonical audit /20):

| | v2 orig | v2 rerun | v3 orig | v3 rerun |
|---|---:|---:|---:|---:|
| Δ(A−B) | +5.875 | **+4.00** | +2.50 | **+3.125** |
| Survival | — | 68% | — | 125% |

**Lift decomposition** (pattern → +reasoning → +formula hints):

| Component | Original | Rerun |
|---|---:|---:|
| Pattern only (v1, unchanged) | +1.88 | +1.88 |
| + Reasoning steps (v3 over v1) | +0.62 | **+1.25** |
| + Formula hints (v2 over v3) | +3.38 | **+0.88** |
| Total v2 | +5.875 | +4.00 |

**Key finding**: REASONING SCAFFOLDING is the dominant lift component, NOT formula hints. Original 58% attribution to formula hints was inflated by (a) sample duplicates clustering scores, (b) generous calibration on rigor in original v2 audit (3/5 for typed baselines without prior numbers, contradicting D3 canonical anchor).

**Statistical caveat** (Opus subagent flagged): at n=8, +0.88 gap between v2-rerun and v3-rerun is only ~1.2 SE. Decomposition magnitudes are suggestive, not confirmed. Need n ≥ 30 per condition to nail down with tight CI.

**Implications for Phase 1+ paper story (BETTER than before)**:
1. Phase 1+ trained generator producing v3-style reasoning-scaffolding abstractions from **abstracts-only retrieval** should reach ~9-10/20 ceiling (matches v3-rerun A range)
2. Methodology full-text retrieval (originally needed for formula hints) adds only +0.9 pts max — **nice-to-have, not critical**
3. Cheaper retrieval strategy is competitive
4. Paper claim updated: "abstraction lift decomposes into pattern + reasoning + formula contributions, with reasoning scaffolding as the dominant pathway component (n=8 caveat)"

**Per-dim direction**: math/novelty/realism/rigor signs preserved across reruns (no inversions).

**Owner**: Yuhong Shi
**Artifacts**: 
- `runs/2026_04_smoke_pathway_v2_rerun_div/{buffer.jsonl, opus_depth_audit_d3canonical.json}`
- `runs/2026_04_smoke_pathway_v3_rerun_div/{buffer.jsonl, opus_depth_audit_d3canonical.json}`
- `runs/sampling_fix_comparison.md`

---

## 2026-04-25: Phase 0.5b — Qwen3-30B-A3B contamination check CLEARED for TTT-Discover

**Result**: NO contamination. Smoke v1/v2/v3 results remain valid.

**Evidence**:
1. **Metadata**: Qwen3-30B-A3B official knowledge cutoff per simtheory.ai = March 2025. TTT-Discover (arxiv 2601.16175) publication = January 2026. Cutoff < publication date by ~10 months.
2. **Direct probe** (3 questions to frozen Qwen3-30B-A3B at temperature 0):
   - Probe 1 (TTT-Discover algorithm + J_β): "the year 2026 is in the future... maybe hypothetical or fictional", guesses "TTT could be Tree-based Training" (wrong; actual is "Test-Time Training")
   - Probe 2 (MAX-PUCT vs AlphaZero PUCT): "I'm not familiar with TTT-Discover" + recalls AlphaZero PUCT correctly
   - Probe 3 (entropic objective for test-time discovery): "arXiv 2601.16175 seems off, maybe a typo?"
3. None of the 3 probes produced accurate technical detail of the reference plan's actual contributions.

**Verdict**: smoke A > B deltas (+1.88 v1, +5.875 v2, +2.50 v3 on D3 canonical Opus audit) are not explained by "Qwen retrieves from in-weights knowledge of TTT-Discover". Post-cutoff confirmed.

**Owner**: Yuhong Shi
**Artifacts**: `/tmp/qwen_contamination_probe.py` (probe script), session log

---

## 2026-04-25: Phase 0.5a — Dataset migrated to self-contained D5 directory; D4 symlink removed

**Decision**: Replace D4 grant_proposal symlink with D5-owned copy of D3 canonical files.

**Trigger**: User flagged that `projects/d5_abstract_retrieve_refine/dataset` was a symlink to `projects/grant_proposal/dataset/` (12 D4 grant goals — wrong for D5's single-goal TTT-Discover scope). Plan reaffirmed single-goal commitment; symlink violated scope.

**Migration steps executed**:
1. Removed symlink (`projects/d5_abstract_retrieve_refine/dataset` no longer points to grant_proposal)
2. Created real D5 dataset directory: `projects/d5_abstract_retrieve_refine/dataset/`
3. Copied D3 canonical TTT-Discover files into D5:
   - `research_goal.txt` (831 bytes)
   - `reference_solution.txt` (6610 bytes)
   - `perturbations/` (17 perturbation variants of reference plan, ~100 KB total — for Phase 4 robustness)
4. Updated D5 code paths to use D5 dataset (was: `projects/ttt_discover/analysis/sanity_check/...`):
   - `src/co_scientist/d5_abstract_retrieve_refine/smoke_pathway_v1.py` (Config defaults)
   - `src/co_scientist/d5_abstract_retrieve_refine/smoke_pathway_v1_score.py` (Config defaults)
5. Updated D5 harness docs to remove "12-goal" / "grant_proposal symlink" references:
   - STATUS.md (Next Concrete Action section, key design commitments)
   - README.md (Dataset section)
   - CONVENTIONS.md (cross-direction rule)

**Risk mitigated**: D3 trainer still works (files COPIED not MOVED; D3's `analysis/sanity_check/` unchanged).

**Verification**: smoke v1 rerun expected to produce byte-identical output (same files, just different paths). Run after this commit.

**Owner**: Yuhong Shi

---

## 2026-04-24: D5 direction created; SDPO-style multi-round retrieve-select-abstract pipeline approved

**Decision**: Create D5 as a new research direction targeting long-form research plan generation via SDPO (Hübotter 2026) with multi-round model-selected paper retrieval and abstraction, reference paper as privileged info.

**Context**: Session dated 2026-04-23/24. User systematically ruled out 5 candidate directions (rubric-RL method novelty, novelty RM training, agentic process reward, agentic output reward long-form, selector-as-primary) on novelty/feasibility grounds. Converged on Abstraction capability + external retrieval + multi-round as core direction. Identified SDPO as the distillation mechanism through user's own description ("student has critique as context, generates again, distill between with/without critique log-probs"). Pivoted from grant proposal (D4) to research plan (D3-style) because research plan is more quantifiable and grant proposal is "水到渠成" downstream.

**Key design decisions**:
- Retrieval source: source paper's bibliography (not web search)
- Paper selection: learned action (model chooses which papers to read)
- 3 rounds, cross-round history visible
- Distill both selection + abstraction logprobs (P2 choice, ML-scientist recommendation against upstream bottleneck)
- Reviewer: Opus 4.5 via subagent initially; transition to self-review if ρ(Opus, self-review) > 0.7 on held-out abstractions
- Reviewer has source paper as privileged info (factual, distinct from SDPO's evaluative privileged info)
- Final plan: supervised CE against source paper's own plan (not SDPO)

**Venue**: ICLR 2027 main track (NeurIPS 2026 abandoned due to timeline).

**Timeline**: 12 weeks (Phase 0-5) + 3 weeks buffer.

**Budget**: projected $500-2000 OpenRouter, ceiling $8000. User explicit override of prior "minimize OpenRouter" preference, partially offset by planned transition to self-review.

**Files affected**:
- Created: `src/co_scientist/d5_abstract_retrieve_refine/` and `projects/d5_abstract_retrieve_refine/`
- To update: `DIRECTIONS.md`, `CLAUDE.md`
- Plan doc: `~/.claude/plans/giggly-jumping-hollerith.md`

**Alternatives considered**:
- Direction (2) Graph structure (claim dependency graph) — tabled: overlap with FLAN Graph (patent prediction, not generation) and graph-as-RL-reward not cleaner than SDPO; engineering complexity too high.
- Diagnostic paper from 58 D4 runs (multiple rubric-RL Goodhart taxonomy) — user rejected: wants method novelty, not diagnostic.

**Owner**: Yuhong Shi

---

## 2026-04-24: Smoke pathway v1 result — pathway validated on Opus, grader gap re-confirmed

**Run path**: `projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v1/`

**Setup**: 8 samples per condition (B_baseline = goal only; A_with_abstraction = goal + oracle_abstraction extracted from reference_solution.txt). Frozen Qwen3-30B-A3B via Tinker. Temperature 0.7, seed 42.

**Opus D3-canonical audit** (4 dims × 1-5, max 20, calibrated to TTT-Discover):
- reference_gold = 20/20 ✓ (anchor validated)
- A_with_abstraction = 7.00/20 (math 1.00, novelty 2.00, realism 2.00, rigor 2.00)
- B_baseline = 5.13/20 (math 1.13, novelty 1.00, realism 2.00, rigor 1.00)
- **Δ(A − B) = +1.88 /20, +9.4pp normalized**
- Effect on novelty (+1.0) and rigor (+1.0). Math and realism FLAT (as pre-registered — dims needing external info don't benefit from abstraction alone).

**Opus D4-style audit** (secondary prompt, 4 dims × 1-10, max 40):
- A = 15.75, B = 12.25, Δ = +3.50 /40, +8.75pp normalized
- Directionally identical to D3 canonical.

**Qwen self-grader (`ten_signal_reward`, 11 signals × 1-5)**:
- A = 0.545, B = 0.590, Δ = **−0.045** (opposite direction from Opus)
- **Grader gap re-confirmed**: Qwen ten_signal grader cannot detect pathway improvement that Opus detects. Specifically, Qwen penalizes A on S3_positioning (−1.13) and S2_rigor (−0.50), sees abstraction-guided plans as "formulaic pattern recital" without field positioning; Opus sees them as substantively better.

**Verdict**: GO to Phase 1. Criteria passed per Opus (both prompts agree +9pp). Qwen criteria failed but deprecated given grader gap evidence.

**Mandatory carry-over fixes to Phase 1**:
1. **Sampling duplicates bug**: Tinker `SamplingParams(seed=42)` collapsed 5/16 samples into byte-identical duplicates (B 2==5, B 4==6==7, A 6==7). Root cause: seed applies per-request, not per-sample; `num_samples=8` with single seed shares entropy. Fix: call `sample()` with `num_samples=1` × 8 independent calls each with a different seed; or drop seed entirely.
2. **Primary training signal must be Opus in SDPO loop**, not Qwen self-grade. Qwen is unreliable as privileged-info source for this pathway.

**Caveat from Opus subagent**: A plans import abstraction vocabulary verbatim (cumulative buffer, tree-search over past states, importance sampling, adaptive hyperparameters) but **do not derive from first principles**. None write J_β entropic objective, PUCT formula, or specific numerical hyperparameters. This explains why A=7 vs reference=20 — 13-point gap is entirely math + realism content not carried by current pattern-level abstraction.

**Implication for abstraction format**: Phase 1 should consider **v2 oracle abstraction** containing derivations (goal → objective-function reasoning) in addition to structural patterns. User's 2026-04-24 note: current abstraction improves structure, not reasoning; closing remaining gap requires abstraction that contains mechanistic derivation steps, not just pattern names.

**Alternatives considered**:
- Skip smoke v2, proceed directly to Phase 1 with pattern-level abstraction — risk: Phase 1 training ceilings at 10-12/20, reviewer asks "why didn't you test richer abstraction"
- Run smoke v2 with derivation-augmented abstraction before Phase 1 — cost ~$5, 1-2 days

**Owner**: Yuhong Shi

**Artifacts**: `analysis.md`, `opus_depth_audit_d3canonical.json`, `opus_audit_d3canonical_summary.md`, `opus_depth_audit.json` (D4-style), `metrics.jsonl` (Qwen ten_signal), `buffer.jsonl` (16 plans), `hypothesis.md` (pre-reg), `scoring_summary.json`.

---

## 2026-04-24: Smoke pathway v2 — derivation-augmented abstraction, Δ 3× larger than v1

**Run path**: `projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v2/`

**Context**: User observation from smoke v1: "提升的是 structure, 不是 reasoning". User proposed adding derivation steps to abstraction so model can derive mechanism from goal, not just adopt pattern names. Executed v2 to test.

**Setup changes vs v1**:
- Oracle abstraction: `smoke_v2_oracle_abstraction.md` — same 8 patterns, each with added "Derivation sketch" (2-6 reasoning steps). Char count 6041→9476 (~55% longer).
- Seed 42→100 to avoid byte-correlation with v1.
- Everything else identical: goal, generator (frozen Qwen3-30B-A3B), N=8 per condition, temp 0.7.

**D3 canonical Opus audit result** (4 dims × 1-5, max 20):

| | v1 | v2 |
|---|---|---|
| B_baseline total | 5.13 | 5.38 |
| A_with_abstraction total | 7.00 | 11.25 |
| Δ(A−B) total | +1.88 | **+5.875** |
| Δ math | −0.13 | **+1.25** (was flat) |
| Δ novelty | +1.00 | +1.88 |
| Δ realism | 0.00 | **+1.25** (was flat) |
| Δ rigor | +1.00 | +1.50 |

**Zero per-plan overlap** in v2: all A ≥ 11/20, all B ≤ 6/20. v1 had overlap.

**Equation reproduction**: 2 of 8 A plans (idx 12, 13) wrote the reference's entropic objective `J = log Σ_a exp(β R(a)) · π(a|s)` in LaTeX. v1 zero plans wrote any equation.

**Leakage honesty**: v2 Pattern 2 Step 3 contains "consider log-sum-exp with inverse temperature β". This is ~75% of the reference equation form. The remaining 25% (the π weighting, β → ∞ limit) is completed by the model. So +5.875 is a **joint** effect of (a) more leaky hints AND (b) model's derivation-completion ability from those hints. Both are relevant for Phase 1+ training claims.

**Updated ceiling analysis**:
- v1 pattern-only: A ceiling ~7/20. Closes ~5% of 15-pt gap to reference.
- v2 derivation-scaffolded: A ceiling ~12/20. Closes ~40% of 15-pt gap.
- Remaining 8-pt gap: specific prior AI numbers, specific hyperparameters, full gradient derivation details — none in v2 abstraction. Phase 1+ training cannot close this without **retrieving full paper methodology content** (not just abstracts).

**Decision impact on Phase 1+**:
1. **Abstraction generator training target** updated: must produce derivation-scaffolded abstractions (Pattern + Derivation sketch), not pattern-only.
2. **Retrieval content requirement** updated: for the generator to learn derivation sketches, retrieval must include methodology sections from the reference paper's bibliography, not just abstracts. Update Phase 0b bibliography fetch spec accordingly.
3. **Realistic Phase 1+ target**: A ≈ 10-12 /20 (matching v2 oracle ceiling). Not 20/20 matching reference. Paper framing must acknowledge ceiling.

**Sampling duplicates re-occurred in v2** (seed 100): B 2==3 byte-identical; A 11==14==15 byte-identical. Effective distinct n: B=7, A=6. Same fix required: drop seed or per-sample seeding.

**Alternatives considered**:
- Continue to smoke v3 with even richer abstraction (e.g., include partial derivations of all 8 patterns, not just Pattern 2) — deferred; v2 already validates pathway decisively, further smoke runs have diminishing return
- Skip v2 and go to Phase 1 — rejected; v2 confirms Phase 1 training must target derivation-scaffolded abstractions, which reshapes infrastructure design

**Owner**: Yuhong Shi

**Artifacts**: `runs/2026_04_smoke_pathway_v2/{analysis.md, opus_depth_audit_d3canonical.json, opus_audit_d3canonical_summary.md, buffer.jsonl, config.json}`, `data/oracles/smoke/v2.md`.

---

## 2026-04-24: Smoke pathway v3 — derivation scaffolding WITHOUT formula hints; decomposes abstraction lift

**Run path**: `projects/d5_abstract_retrieve_refine/runs/2026_04_smoke_pathway_v3/`

**Purpose**: v2 gave +5.875 but leaked formula forms ("log-sum-exp with β", "Q+c·P form"). v3 keeps same reasoning-step structure but removes specific functional-form hints. Tests whether derivation scaffolding transfers capability independent of formula leakage.

**Setup changes vs v2**:
- Oracle abstraction: `smoke_v3_oracle_abstraction.md` — same 8 patterns + derivation sketches, but Pattern 2/4/5/6 no longer contain specific functional forms. Models must derive forms themselves from goal-level reasoning.
- Seed 100→200. Template collapse unchanged.

**D3 canonical Opus audit**:

| | v1 | v2 | **v3** |
|---|---|---|---|
| B_baseline | 5.13 | 5.38 | 6.75 |
| A_with_abstraction | 7.00 | 11.25 | **9.25** |
| Δ(A−B) | +1.88 | +5.875 | **+2.50** |
| Equations in A | 0/8 | 2/8 (reference form) | 2/8 (generic forms) |

**Lift decomposition** (this is the paper-level finding):
- Pattern structure alone (v1): **+1.88** (32% of v2 lift)
- + Derivation reasoning steps (v3 over v1): **+0.62** marginal (11%)
- + Specific formula hints (v2 over v3): **+3.38** marginal (58%)

**Implications**:

1. **Most of v2's measured abstraction lift came from formula leakage, not pure reasoning scaffolding**. Pure derivation scaffolding adds only +0.62 over pattern-only.

2. **Phase 1+ realistic ceiling** depends on retrieval strategy:
   - Abstracts-only retrieval (cheapest): Phase 1 trained generator will produce v3-quality abstractions. Expected A ceiling ≈ **9/20**.
   - Methodology full-text retrieval: possible to learn formula hint production. Expected ceiling ≈ **11/20**.
   - Reference gold = 20/20; neither retrieval strategy closes full gap (need specific prior AI numbers + hyperparameters).

3. **Paper narrative upgraded** from "abstraction helps" to "abstraction lift decomposes into pattern / reasoning / formula, with formula hints being the dominant contributor". This is testable and defensible against reviewer pushback.

**Retrieval strategy decision for Phase 0b / Phase 1**: fetch **abstracts + methodology sections** (not just abstracts) from bibliography papers. Methodology sections likely contain the formula forms that Phase 1+ trained generator needs to learn to produce.

**Template collapse worsened** (v3 44% duplicates vs v1/v2 31%). Seed change (42→100→200) does not fix. Root cause is model+temperature interaction. Before Phase 1 runs:
- Test temperature=1.0 or top_p sampling
- If collapse persists, consider diversity regularizer in training signal

**Alternatives considered**:
- Skip v3, accept v2 result as Phase 1 baseline → rejected. Without v3, cannot decompose lift; reviewer would attribute v2's +5.875 entirely to formula leak with no way to argue back.

**Owner**: Yuhong Shi

**Artifacts**: `runs/2026_04_smoke_pathway_v3/{analysis.md, opus_depth_audit_d3canonical.json, opus_audit_d3canonical_summary.md, buffer.jsonl}`, `data/oracles/smoke/v3.md`.

---

## 2026-04-24: Evaluation framework finalized as 2-layer + novelty framing revised

**Decision**: Evaluation organized into 2 layers:
- **Layer 1 (mechanism evidence, secondary)**: sub-capability trajectories — selection precision, abstraction quality, pathway ablation (oracle abstractions + frozen composer vs full trained pipeline), SDPO KL trajectory
- **Layer 2 (primary metric)**: final plan Opus audit /40 on held-out goals, pairwise preference vs reference plan

**Critical baselines** that reviewer will require (and that could kill the paper if they match trained model):
- Frozen Qwen3-235B + single retrieve + reference plan as few-shot context ← probable upper-bound threat
- Pure SFT on (retrieved, reference plan) without SDPO or multi-round
- Full pipeline frozen (no SDPO) — isolates training contribution

**Required effect size**: trained model ≥ strongest baseline + 3 Opus points, bootstrap 95% CI non-overlapping.

**Novelty framing** revised: the paper must NOT frame novelty as "long-form writing is hard" or "external info injection helps open-source models" (both already established by DR Tulu/FLARE). The paper MUST frame novelty as 4-way specific mechanism combination:
1. SDPO with factual (not evaluative) privileged info
2. Learned paper selection action
3. Abstraction as iterated artifact (not raw text like FLARE)
4. Reference plan as supervised gold anchor

**Training-loop definition** (per user 2026-04-24): 1 round = full pipeline execution. Three options for multi-round training:
- (X) same instance, multiple SDPO updates (exploitation)
- (Y) different instances per pass (standard epoch)
- **(Z) same instance, run → update → re-run on updated model → update (self-distillation bootstrapping)** — user's preferred, matches "经过几轮训练 model 学会能力" intuition
- Decision: try Z in Phase 2, fallback to Y if drift/overfit.

**Alternatives considered**:
- Pure final-plan metric (rejected: needs mechanism evidence for paper narrative)
- Ablation-heavy without head-to-head baselines (rejected: reviewer will run frozen 235B and match)

**Affected files**: plan doc, STATUS.md, Phase 4 evaluation section

**Owner**: Yuhong Shi
