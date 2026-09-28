# D5 Abstract-Retrieve-Refine STATUS

> **⚠ READ FIRST (2026-04-27 PM, REVISED)**: terminology audit verified that
> all "SDPO" references in older D5 docs / log messages refer to a **D5
> in-house off-policy IS-loss variant** — neither canonical Hübotter SDPO
> (arxiv 2601.20802; student on-policy + KL loss) nor canonical Zhao OPSD
> (arxiv 2601.18734; student on-policy + JS or sampled-token reverse-KL).
> The earlier "option (c) HER from Hübotter 2026" framing in this project's
> docs was D5-internal speculation with no basis in the actual paper.
>
> Two ground-truth references:
> - [`knowledge/current/CANONICAL_NAMING_REFERENCE.md`](knowledge/current/CANONICAL_NAMING_REFERENCE.md) — paper attribution
> - [`knowledge/current/RUN_REGISTRY.md`](knowledge/current/RUN_REGISTRY.md) — per-run setup truth table

*Last updated: 2026-04-29 PM (**v9 Layer 1 + Layer 2 pilots COMPLETE; F18 filed; KL anchor regularizer not booster**)*

## 2026-04-29 PM — v9 Layer 1 + Layer 2 pilots (verifier-grounded + KL anchor)

Two same-session 16-iter pilots evaluating reward function modifications above v8-d5sdpo Gplus.

**Pilots completed**:
- L1 `train_mu_v9_grounded`: sparse per-token bonus on gold equation/citation matches (+0.1/+0.05). Peak 22.00 @ iter 14, mean 19.38, mid-pilot cliff iter 9-12 (low 16.50).
- L2 `train_mu_v9_kl_anchor`: L1 + per-token KL term toward π_oracle SFT'd on 30 grounded plans (β=0.05). Peak 23.38 @ iter 14, mean **20.91**, NO cliff. Final 3 iter: 21.62, 23.38, **23.12**.

**5-way comparison**:

| cell | peak | mean | final 3 iter | drop |
|---|---:|---:|---|---:|
| σ_v8 (frozen+slim) | **25.25** | — | static | — |
| C+ (critique-only) | 23.50 | 20.11 | 19.25, 17.50, 14.75 | -8.75 cliff |
| **L2 v9_kl_anchor** | 23.38 | **20.91** | 21.62, 23.38, 23.12 | -0.26 stable |
| G+ baseline | 22.38 | 19.37 | 19.38, 14.75, 12.50 | -9.88 cliff |
| L1 v9_grounded | 22.00 | 19.38 | 18.75, 22.00, 21.75 | -0.25 stable |

**Multi-axis grounding (custom verifier on gold = 8 eqs + 13 cites + 7 emp anchors, peak iter)**:

| cell | eq/8 | cite/13 | emp/7 |
|---|---:|---:|---:|
| G+ | 3.38 | **7.38** | 0.38 (cite-skewed) |
| C+ | **6.12** | 4.25 | 0.12 (eq-skewed) |
| L1 | 5.00 | 2.88 | 0.00 (verifier-Goodhart) |
| **L2** | 5.62 | 5.75 | **0.62** (only balanced cell) |

**F18 (new finding)**: KL anchor toward SFT'd π_oracle is a **regularizer** (stability + multi-axis balance), NOT a peak booster. C+ peak ≥ L2 peak by 0.12pt, but C+ cliffs to 14.75 by iter 12 while L2 sustains 23.0+ across final iter. Distributional grounding ≠ deep grounding (U3 originality stays at 2 throughout — model is mimicking surface form: J_RS notation, Hubert/Georgiev/Zuo citations, Math/Methodology/Insight index labels — not internalizing derivations).

σ_v8 frozen+slim baseline (25.25) still unbeaten by all RL cells.

**Files added**:
- Trainers: `src/co_scientist/d5_abstract_retrieve_refine/{train_mu_v9_grounded, train_mu_v9_kl_anchor, train_pi_oracle_sft, verifier_grounded_reward_v1}.py`
- Dataset: `dataset/{oracle_grounded_sft.jsonl (30 plans × 3 anchor groups), v9_gold_equations.json (8 eqs + 13 cites)}`
- Pilots: `runs/2026_04_29_mu_v9_{grounded,kl_anchor}_pilot/` + `runs/2026_04_29_pi_oracle_sft/`
- RESULTS: `knowledge/current/RESULTS_layer1_vs_layer2.md`
- Finding: `paper_materials/findings/F18_kl_anchor_stabilizes_not_lifts.md`

**Commits**: `4e301aa` (main) + `8babd17` (re-aggregate) + `bef3985` (F18)

---

## 2026-04-28 PM — Phase 2 v8-d5sdpo grid + oracle transfer ceiling diagnosis (F15 + F16)

This session ran a partial 7-cell single-seed factorial probe of (mask × α × loss) on
the μ-v8-d5sdpo trainer (`train_mu_v8_d5sdpo.py`), generated paper-grade Opus
critique on every iter via file-bus subagents, and re-read the trainer + oracle
to diagnose why oracle slim's content does not transfer into student outputs.

**Verified per-cell trajectories (audit_v3 9-dim mean /45, n=8, σ_v8 anchor=19.38)**:

| Cell | mask | α | loss | iter 0 | Peak /45 | Peak iter | Cliff iter | Final | Status |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---|
| base | F | 0 | IS | 19.38 | 20.00 | 3-4 | 6 | 10.00 (i=6) | KILLED |
| G | T | 0.05 | PPO | 19.13 | 20.00 | 3 | 8 | 10.25 (i=9) | KILLED |
| G+ | T | 0.1 | PPO | 19.50 | 22.38 | 4-5 | 11 | 12.50 (i=12) | DONE |
| E | T | 0 | PPO | 19.75 | 22.75 | 5 | 7 | 9.75 (i=9) | KILLED |
| C+ | T | 0.1 | IS | 19.75 | **23.50** | 6 | 10 | 14.75 (i=12) | KILLED |
| F+ | F | 0.1 | PPO | 19.88 | 23.00 | 5-6 | 8 | 13.38 (i=9) | KILLED |
| A | T | 0 | IS | 19.38 | 19.38 | 0 | (failed iter 1) | 19.13 (i=1) | FAILED |

**Pairwise (G+ peak, n=8 per matchup, Opus judge)**:
- G+ vs σ_v8: **8-0 G+** (audit lift corroborated)
- G+ vs v7-opd-full: **3-5 v7-opd-full** (**v7-opd-full WINS — pre-registered "v8 beats v7-opd-full" headline FALSIFIED**)
- G+ vs μ-v4: 8-0 G+ (6/8 A-position, M8-bias borderline)
- G+ vs G: 7-1 G+ (α=0.1 dominates α=0.05)

**Phase 2 findings (F15 — `paper_materials/findings/F15_v8_d5sdpo_phase2_attribution.md`)**:
1. **Cliff iter monotonic in α at fixed (mask=T, PPO)**: base i5-6 → G α=0.05 i7-8 → G+ α=0.1 i11. Hübotter SDPO Appendix A.2 trust-region works as predicted.
2. **Peak insensitive to α**: cross-α peak Δ ≤ 1.5 audit pts. α is **cliff-dampener, not peak-driver**.
3. **Mask + PPO contributes most to peak**: E (mask=T, PPO, α=0) peak 22.75 vs base (mask=F, IS, α=0) peak 20.00 = +2.75 within α=0 row.
4. **G+ does NOT beat v7-opd-full** — 3-5 pairwise. v8 redesign is "competitive with v7-opd-full, not a superset".

**F16 — Oracle transfer ceiling diagnosis (`paper_materials/findings/F16_oracle_transfer_ceiling_diagnosis.md`)**:

Code reading of `train_mu_v8_d5sdpo.py:393-473` confirms architecture is **self-distillation** (teacher and student are both the current LoRA-adapted π_θ; only the prompts differ — teacher gets goal+oracle+critique, student gets goal+oracle). frozen base only enters via trust-region α-interpolation at lines 327-329. Oracle is in the prompt (`build_student_prompt_v8`, `build_teacher_prompt_v8`) — never in any loss term.

Direct read of oracle slim (`data/oracles/oracle_v2_2026_04_26_build/slim.md`, 10,265 words / 87 items / 17 papers) confirms it contains the formulas Opus critique repeatedly flags as missing in plans (Math 5 J_RS = (1/β)·log E[e^(βr)], Math 7 GRPO RHS, Math 12 TTRL gradient, Methodology 9 RS-GRPO advantage; Empirical 4 AIME 15.6→77.9, Empirical 16 +211% TTRL).

Direct read of `critic_responses/iter_NNN.json` across G+ peak / E peak / C+ peak / E cliff confirms Opus critic produces paper-grade substantive critique citing exact formulas + named methods + benchmark identity.

**Three hypothesized bottlenecks (each falsifiable by ABC experiments)**:
- H16-1: long-context retrieval bottleneck (Qwen3-30B can't reliably copy from 10K oracle)
- H16-2: critique-conditioned advantage rewards critique-conformity, not oracle-fidelity
- H16-3: LoRA bandwidth insufficient for formula-length internalization

**ABC experiment plan (`knowledge/current/EXPERIMENT_PLAN_oracle_transfer_ABC.md`)**:
- Exp A: frozen Qwen3-30B verbatim retrieval probe (5 queries × 3 samples) — ~$3, 30 min
- Exp B: 4-gram overlap student-output vs oracle math/methodology trajectories on 6 existing cells — $0, 15 min
- Exp C: 1-cell RAG smoke (top-K oracle items vs full oracle, on G+ config) — ~$40, 6 hr

Total ABC = ~$43 + ~7 hr in one session.

**Open**: Phase 2 cells D, B, B+, F, C not run. Single-seed throughout. v7-opd-full
beats v8 G+ in pairwise (3-5). Production checkpoint policy: **G+ iter 4 remains
the only pairwise-corroborated v8 checkpoint**; C+ peak 23.50 is paper-honest "highest v8 audit peak but uncorroborated".

---

## 2026-04-27 PM — Phase 3 F8 v2 revision (7-1 STRONG falsified)

Routine BoN sanity check (Step 2a, $4) discovered TWO issues that invalidated F8 v1
"τ_v4 26.50 + 7-1 STRONG over σ_v4" headline:

1. **Lucky-draw inflation**: τ_v4 production single-shot at instance 0 = 27 vs K=8 reseed mean = 24.88 (instance 2: production 29 vs K=4 reseed 23.50, gap 5.5). Cross-instance mean 26.50 was driven by lucky single-shot draws.
2. **Citation-tag leakage**: Distillation step organizes oracle items by index labels ("Methodology 17", "Math 11"); plan generator copied them as if real citations. Judges penalized.

**Step A**: Patched `_KAPPA_PLAN_FOOTER_V4` with anti-leakage instruction (~85% effective).

**Step B** (8 inst × 1 plan, PATCHED, $4): τ_v4_clean cross-instance mean = **24.50** (vs claimed 26.50, drop 2.0). σ across 8 = 2.18; 95% CI [22.96, 26.04] CONTAINS σ baseline 25.25.

**Step C** (inst 2 K=4 σ spot-check, $2): σ_within = 0.50 (vs inst 0's 1.17, ratio 2.3× heterogeneous).

**Step E** (pairwise τ_v4_clean vs σ_v4 rerun, 8 pairs, seed=52, $32): **4-4 TIE** (50% A-position balanced — no position bias). Step 2.5's 7-1 STRONG was a sampling artifact.

**F8 v2 reframe**: "open-source 30B inference-time pipeline produces plans STATISTICALLY COMPARABLE to hand-curated baseline (no fine-tuning) + 3 characterized failure modes (self-review degeneration, single-shot evaluation inflation, citation-tag leakage)". Negative-result-with-mechanism — ICLR-publishable.

**Phase 3 status**: COMPLETE. κ_opus deferred (low-priority now that τ_v4 ≈ σ; BoN selection-bias math says BoN unlikely to reach μ-v4 28.00 ceiling).

**Open for next session**:
- σ baseline K=4-8 reseed (~$4-8) to nail population mean (does claim hold or does τ_v4 slightly underperform?)
- Distillation-prompt fix for full tag elimination (vs current ~85% partial fix)
- κ_opus full smoke (~$60-100, low-priority)
- Phase 4/5 hand-off to Session B's track (μ-v5 Tool-V continual chain using D5 in-house off-policy IS-loss variant)

---

*Previous Last updated: 2026-04-27 AM (**Phase 5 + quality audit remediation COMPLETE**; F7 DOWNGRADED to inconclusive; F9 STRENGTHENED via 3-protocol agreement; M8 NEW methodology contribution)*

**Read this first**: [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) for executive summary, then
[`DECISIONS.md`](DECISIONS.md) 2026-04-26 (f) "2026-04-27 closing" section for the post-corroboration verdict,
then [`paper_materials/findings/F9_continual_sdpo_field_expert.md`](paper_materials/findings/F9_continual_sdpo_field_expert.md)
(Phase 5 NULL, triple-corroborated) and [`paper_materials/methodology/M8_audit_pairwise_divergence.md`](paper_materials/methodology/M8_audit_pairwise_divergence.md)
(NEW methodology finding — 2-plan/subagent anchoring inflates close-cluster Δ).

## 2026-04-28 — D-series investigation complete (D1+D2+D3 done; D4 deferred)

User mandate "把能支持或者solid我们现有结论或者能解释现有不清楚地方的run都做了吧" → comprehensive multi-front investigation. 6 open questions targeted; 5 resolved this session (D4 Phase 6α replay buffer pending separate session).

**D1 n=24 stress test (96 strict + 48 pairwise subagents)**:
- meta_ttl n=24: strict Δ -0.96, pairwise σ' 14/9/1 (58%) → **σ' DIRECTIONAL CONFIRMED** at n=24
- tool_v_ttrl n=24: strict Δ -0.58, pairwise σ' 12/10/2 (50%) → **TRUE NULL** at n=24
- F7 "cross-goal transfer" claim **DEFINITIVELY NULL** at n=24 across both goals; no μ' lift emerges with 3× sample size

**D2 extract_solution regex fix**: 1-line fix (strip `<think>` before `_SOLUTION_RE`) + 6 regression tests. Diagnostic on 272 plans across 17 buffers: 0 extraction-bug residuals (1 short plan is genuine generation truncation). Commit `ea26f4e`.

**D3 7-baseline pairwise σ/δ/α (24 subagents)**: σ vs δ shows σ 6/2 ≥6/8 (audit underestimated); 2 α-matchups had position-bias outside 75% (M8 caveat). Mixed verdict — M8 cross-goal effect partially generalizes.

**D4 Phase 6α plan-level replay**: deferred to next session (Tinker training + audit ~$25 + 4 hr).

**D5 per-goal asymmetry**: resolved via D1 n=24 — n=8 asymmetry was sampling noise within ±3.5; n=24 shows uniform σ'-edge / TIE on both goals.

## 2026-04-27 — Phase 5 quality audit remediation complete

User-flagged quality concern (3 NEW hard rules: ZERO OpenRouter, subagent task isolation, quality first) triggered comprehensive corroboration of all close-cluster verdicts via pairwise + strict 1-plan/subagent re-audits.

**Cross-protocol comparison** (5 goals × 3 protocols where pairwise available; 80 strict subagents + 32 pairwise subagents + 1 batch_13 JSON-fix):

| Goal | Orig 2-plan Δ | Strict 1-plan Δ | Pairwise tally | Verdict |
|---|---:|---:|:---:|---|
| meta_ttl | +2.13 | -0.12 | σ' 5/3 | **2-plan inflated; pairwise+strict say σ' edge / TIE** |
| tt_control | -0.88 | -0.63 | σ' 5/3 | **3-protocol convergence** (σ' edge stable) |
| tool_v_ttrl | +3.00 | -0.63 | μ' 4/3/1 TIE | **2-plan inflated; pairwise+strict ≈ TIE** |
| Phase 5 v3 | +0.62 | +1.87 | TIE 4/4 | NULL (within-noise across 3 protocols) |
| Phase 5 α | -2.63 | +0.26 | n/a | NULL (strict softens "α worse" verdict) |

**Phase 4a F7 verdict DOWNGRADED**: "Strong-mixed 2/3 pass" → "all 3 inconclusive within audit noise; meta_ttl direction-reverses under pairwise; methodology lesson (M8) is the surviving contribution".

**Phase 5 F9 verdict STRENGTHENED**: NULL corroborated by all 3 protocols (in-loop trajectory + audit /45 + pairwise + strict). The 4-cell grid failure is the most rigorously validated finding in the project.

**NEW M8 methodology contribution**: 2-plan/subagent batching introduces systematic anchoring that inflates Δ at borderline cases by ~2-3 points. Strict 1-plan + pairwise are required for verdicts at Δ < 5/45.

## Phase 5 — COMPLETE: NULL across 4-cell grid (2026-04-26)

Tested **field-expert via continual SDPO** hypothesis. **All 4 grid cells fail H1 forward learning**:

| Run | Anchor | Adam | anchor_ce | Best in-loop /20 | Failure mode |
|---|---|---|---:|---:|---|
| cliff_v1 | μ-v4 iter 4 | full | 0 | 8.38 | iter-1 cliff -3.50 |
| cliff_v2 | μ-v4 iter 4 | fresh | 0 | 9.00 | iter-1 cliff -2.125 |
| v3 (iter2anchor) | μ-v4 iter 2 | fresh | 0 | 9.375 | iter 0→4 slow decline -2.71 |
| α (anchor_ce) | μ-v4 iter 2 | fresh | **0.1** | **10.375** | iter-2 cliff -3.0 |

**Best v5 vs μ-v4 isolated /45 on Tool-V** (corroborated 2026-04-27): v3 Δ +0.62 / +1.87 strict / pairwise 4-4 TIE; α Δ -2.62 / +0.26 strict (anchor_ce verdict softened to TIE under strict, but trajectory cliff at iter 2 remains the load-bearing failure signal).

**Verdict**: NULL bin per pre-registered matrix. **Naive AND simple-regularization continual SDPO are insufficient**. F9 finding (now triple-corroborated) anchors paper narrative; M8 methodology contribution adds a methodological lesson. **Phase 4b cross-domain stretch test formally DEPRECATED**.

Paper now has cohesive 4-finding story: F2/F4 (μ-v4 +2.75 on TTT-D), F7 (Phase 4a inconclusive — DOWNGRADED, becomes auxiliary methodology evidence), F9 (continual NULL with 4-cell diagnostic + 3-protocol corroboration), M8 (audit-pairwise divergence methodology contribution).

**Plan**: `~/.claude/plans/adaptive-churning-perlis.md` (Phase B done; Phase 7 paper draft is next)

---

## Phase 4a — COMPLETE (2026-04-26)

Cross-goal validation on 3 forward-citation TTT-Discover follow-up papers, reusing the slim
oracle verbatim. **Verdict: Strong-mixed (2/3 pass)** per pre-registered matrix:

| Goal | σ' /45 | μ' /45 | Δ | Pass (≥+1.0)? |
|---|---:|---:|---:|---|
| meta_ttl (Lou 2604.00830, Meta-TTL) | 17.75 | **19.88** | **+2.12** | ✅ |
| tt_control (Wang 2603.09221, TT-Control) | 20.00 | 19.12 | -0.88 | ❌ |
| tool_verification_ttrl (Liao 2603.02203) | 20.38 | **23.38** | **+3.00** | ✅ |
| **Aggregate** (n=24 each) | 19.38 | 20.79 | **+1.42** | (>+0.5 ✅) |

Per-dim transfer signature matches Phase 2: U5 +0.79, T3 +0.46, T4 +0.33, T1 +0.21 (avg) lift;
U2/U4/T2 slight regress (depth-over-breadth bias). tt_control failure correlates with
oracle-coverage gap (architectural-mechanism goal not in slim oracle scope). See
[`F7_cross_goal_followup_validation.md`](paper_materials/findings/F7_cross_goal_followup_validation.md)
for full per-goal + per-dim breakdown + bootstrap CIs.

## Phase 2 — COMPLETE (2026-04-27)

## Phase 2 — COMPLETE (2026-04-27)

Phase 2 produced a **clean positive result**: μ-v4 plan-level **D5 in-house off-policy IS-loss variant** (TEACHER samples; STUDENT lp recomputed under non-critique input; importance_sampling loss — historically labeled "SDPO" but neither canonical Hübotter SDPO 2601.20802 nor canonical OPSD 2601.18734) + Opus critic at lr=5e-5 lifts Qwen3-30B-A3B from σ baseline 25.25 → **28.00 / 45 at iter 4** (+2.75 over σ), the **first 30B-trained variant to beat the σ frozen+oracle baseline**. Multi-round instability of this in-house variant prevents sustained gains past iter 5; iter-4 checkpoint is the **production μ-v4** for the D5 paper. See `knowledge/current/RUN_REGISTRY.md` for per-run setup truth and `knowledge/current/CANONICAL_NAMING_REFERENCE.md` for paper attribution.

- **Phase 2A-bis bibliography rebuild** — DONE. PDF parsing originally dropped 17/87
  references silently; switched to LaTeX source pipeline (main.tex + main.bib).
  Result: 90 cite keys → 86 resolved metadata, 38 with full text. Files:
  `data/bibliography/resolved_v2.jsonl`, `data/source_paper/v2.md`,
  `data/bibliography/full_text/*.md` (28 LaTeX-→-md converted papers).
- **Phase 2B oracle build** — DONE (commit `67ce1b9`). 4-round Opus extraction
  over 62 papers, 580 typed items in `oracle_v2.md`. Slim variant `oracle_v2_slim.md`
  for context-fit at 4096-token policy budget (commit `3f1e848`).
- **Phase 2D realigned trainers** — DONE (commit `3f1e848`). v2 versions of
  α/β/μ/σ/ξ/δ/ε trainers (`train_*_v2.py`).
- **Phase 2E baseline battery (7 baselines)** — DONE (commit `b10d71a`).
- **Phase 2F audit_v3 ISOLATED** — DONE. New methodology: 8 balanced batches × 8
  parallel anonymized Opus subagents, per-plan independent 9-dim scoring (5 universal
  + 4 subfield). Code: `audit_v3_isolated.py`, runs: `runs/2026_04_26_phase2F_audit_isolated/`.
- **Phase 2 lr ablation (μ-v2 / v3 / v4 — all D5 in-house off-policy IS-loss variant; see RUN_REGISTRY.md)** — DONE.
  - μ-v2 lr=1e-5: flat 23.88 (no learning)
  - μ-v3 lr=2e-4: peak 24.88 → collapse 14.38 at iter 6 (catastrophic)
  - **μ-v4 lr=5e-5: peak iter 4 = 28.00 (+2.75 over σ)**, cliff iter 5 = 15.62
  - Production checkpoint: `runs/2026_04_27_mu_v4/checkpoints.jsonl` row batch=4

### 8-baseline ranking under audit_v3 ISOLATED (2026-04-27)

| Baseline | Setup | Mean /45 | vs σ | Pairwise vs μ-v4 |
|---|---|---:|---:|:---:|
| ε | frozen Qwen3-235B + reference plan in context | 33.62 | +8.37 | (gap clear, n/a) |
| **μ-v4** (iter 4) | **trained 30B in-house off-policy IS-loss + Opus critic, lr=5e-5, 5 iters** | **28.00** | **+2.75** ✅ | **—** |
| **σ** | **frozen Qwen3-30B + slim oracle (no training)** | **25.25** | **—** | **loses 2-6** |
| δ | frozen Qwen3-30B + reference plan in context | 25.12 | -0.13 | loses 1-7 |
| α | Opus distillation 5 epochs × 16 plans | 25.00 | -0.25 | loses 1-7 |
| μ-v2 | plan-level in-house off-policy IS-loss + critic, lr=1e-5, 10 iters | 23.88 | -1.37 | (not tested) |
| β | direct SFT on reference plan | 22.75 | -2.50 | (gap clear, n/a) |
| ξ | frozen Qwen3-30B + goal only (no oracle) | 15.62 | -9.63 | (gap clear, n/a) |

**Pairwise corroboration (2026-04-26)**: 3 matchups × 8 position-randomized pairs,
Opus 4.7 anonymized judge. μ-v4 wins **20/24 (83.3%)** across {σ, δ, α} — vs σ 6-2,
vs δ 7-1, vs α 7-1. Position-bias check passes (50/62/75% A-pos rates). The PRIMARY
matchup (vs σ) clears the ≥6/8 PASS threshold. Confirms audit_v3 ISOLATED ranking is
NOT an absolute-audit surface-form artifact. Source: `runs/2026_04_27_mu_v4_pairwise/`,
`paper_materials/experiments/E6_mu_v4_pairwise_corroboration.json`,
`paper_materials/findings/F2_mu_v4_iter4_peak.md` § "Cross-validation: pairwise corroboration".

### μ-v4 trajectory (9-dim isolated; per-iter)

```
| Iter | Mean /45 | vs σ |
|------|---------:|----:|
| 0    |    24.00 | -1.25 |
| 1    |    24.75 | -0.50 |
| 2    |    24.88 | -0.38 |
| 3    |    27.25 | +2.00 |  ← first to beat σ
| 4    |    28.00 | +2.75 |  ← PEAK (production)
| 5    |    15.62 | -9.62 |  ← cliff (-12.4 in 1 iter)
| 6    |    ~9.6  | -15.7 |  ← collapse
| 7    |    ~9.8  | -15.4 |
```

See [`paper_materials/findings/F2_mu_v4_iter4_peak.md`](paper_materials/findings/F2_mu_v4_iter4_peak.md)
for full details + per-dim breakdown.

### Next phase

**Phase 3: retrieve-then-generate** — replace static slim oracle with per-section
dynamic retrieval over the 86-paper bibliography. Reuse SDPO recipe v1 verbatim. See
[`paper_materials/next_steps/N1_phase3_retrieve_then_generate.md`](paper_materials/next_steps/N1_phase3_retrieve_then_generate.md).

## Phase 3 update (2026-04-27): inference-time pipeline COMPLETE; SDPO+critic deferred to BoN-conditional κ_opus

Phase 3 reframed from "retrieve-then-generate" to "**dynamic distillation pipeline**"
(Path Y, locked 2026-04-27 via Q3 reframe). Production deliverable: τ_v4 inference-
time 3-round distillation pipeline.

| Run | Setup | Mean /45 | Pairwise vs σ_v4 |
|---|---|---:|:---:|
| ~~**τ_v4** (production, F8 v1)~~ | ~~base + medium oracle 3-batch + plan_v4~~ | ~~**26.50**~~ | ~~**wins 7-1 STRONG**~~ |
| **τ_v4_clean** (F8 v2, patched + fresh, 2026-04-30) | base + medium oracle 3-batch + plan_v4 (citation-hygiene patched) | **24.50** (n=8, 95% CI [22.96, 26.04]) | **TIE 4-4 vs σ_v4** (Step E, seed=52, pos-balanced) |
| σ_v4 (control) | base + slim oracle + plan_v4 | 25.75 | — |
| μ-v4-replan (Phase 2 weights) | μ-v4 LoRA + slim + plan_v4 | 24.75 | (vs τ_v4 tie 4-4) |
| σ Phase 2 baseline | base + slim + plan_v3 | 25.25 | — |
| μ-v4 Phase 2 prod | μ-v4 LoRA + slim + plan_v3 | **28.00** | (vs σ_v4 wins 7-1) |

> **F8 v1 RETRACTED 2026-04-27 PM** — original τ_v4=26.50 was lucky-draw artifact (instance 0 single-shot 27 vs K=8 reseed mean 24.88; instance 2 single-shot 29 vs K=4 reseed mean 23.50). Step E pairwise rerun (8 pairs, seed=52, position-balanced) returned **4-4 TIE**, falsifying the original 7-1 STRONG verdict. See `DECISIONS.md:7-44`.

**Phase 3 paper-grade claim** (REVISED per F8 v2, 2026-04-27 PM): τ_v4_clean = 24.50/45 on patched + fresh sample (n=8, 95% CI contains σ baseline 25.25); pairwise 4-4 TIE vs σ_v4 (Step E, n=8, seed=52). **Defensible claim**: open-source 30B inference-time pipeline produces plans **statistically comparable** to hand-curated σ baseline (no fine-tuning), with 3 characterized failure modes (M7 self-review degeneration, single-shot evaluation inflation at σ_within ≈ 1.17, citation-tag leakage from multi-round distillation).

**Phase 3 self-review hypothesis DISPELLED** (per F8/M7 + Smoke A): Qwen3-30B as
self-critic 0-8 lost to Opus critic on same 8 plans (Opus-as-judge, position-
randomized). Privileged Info Comprehension gap; κ_self path NOT VIABLE.

**Phase 3 SDPO+critic path STILL OPEN**: κ_opus (real Opus critic in distillation
SDPO loop) untested. Conditional on τ_v4 BoN ablation result (next): if BoN-4
saturates near μ-v4 28.00, κ_opus marginal value low (skip); if BoN-4 still ≈
26.50, κ_opus is the only remaining lever.

**INVALIDATED**: prior κ-v1 smoke verdict (cold-start critic, FAIL at 24.38) —
cost-saving compromise; SDPO advantage on cold-start text is not the same test as
real critic. See DECISIONS.md 2026-04-27 entry + F8 § "Limitations".

See [`paper_materials/findings/F8_phase3_distillation_pathway.md`](paper_materials/findings/F8_phase3_distillation_pathway.md)
for full details + cross-validation.

### 7-baseline ranking under audit_v3 ISOLATED (2026-04-26)

| Baseline | Setup | Mean /45 | vs σ |
|---|---|---:|---:|
| ε | frozen Qwen3-235B + reference plan in context | **33.62** | +8.37 |
| **σ** | **frozen Qwen3-30B + oracle v2 slim (no training)** | **25.25** | **—** |
| δ | frozen Qwen3-30B + reference plan in context | 25.12 | −0.13 |
| α | Opus distillation (5 epochs × 16 plans) | 25.00 | −0.25 |
| **μ-v2** | plan-level in-house off-policy IS-loss + critic + oracle (10 iters, lr=1e-5) | **23.88** | **−1.37** |
| β | direct SFT on reference plan | 22.75 | −2.50 |
| ξ | frozen Qwen3-30B + goal only (no oracle) | 15.62 | −9.63 |

σ (frozen + oracle, no training) is the key reference baseline: any trained
30B variant must beat it. μ-v2 fails by 1.37 points → SDPO training adds
nothing over the inference-time oracle scaffolding.

### μ-v2 diagnostic — the smoking gun

Mid-training audit on μ-v2 EVAL plans showed PUCT mention pattern:

| Iter | Teacher (with critique): PUCT | Student EVAL (no critique): PUCT |
|---:|:---:|:---:|
| 0 | 0/8 | 0/8 |
| 5 | 4/8 | 0/8 |
| 9 | 8/8 | 0/8 |

Teacher absorbed the critique (PUCT 0 → 8/8 across iters). Student EVAL
channel never picked it up. SDPO advantage signal was healthy throughout
(`mean_adv` ≈ 0.4-0.5, `pos_frac` ≈ 80%). Hypothesis: gradient budget too
small (10 iter × 1 step × bs=8 × lr=1e-5 ≈ 100× under-budgeted for LoRA-r64).

### μ-v3 catastrophic collapse (lr=2e-4, 4 grad steps × 20 iter) — KILLED

Tried 20× higher lr to break the under-budget regime. Result: not convergence
but distribution co-collapse. `mean_adv` 0.5 → 0.05, but iter-6 isolated audit
mean dropped to **14.38** (baseline iter-0: 24.88). Iter-6 plans corrupted
with Chinese-character bleed, biocultural headers, thinking-mode JSON leaks.
lr=2e-4 too aggressive for LoRA-r64 in this setting. User: "kill 吧". Audit
v3 ISOLATED on μ-v3 not used as final verdict — the run destabilized the
parameter space, not just under-trained it.

### μ-v4 design — geometric-median lr with safety guards (COMPLETE — production iter 4)

User feedback: rejected CR-v7 + SFT pivot ("不够美丽"). Stay with the
in-house off-policy IS-loss architecture (μ-v2/v3/v4 family — historically
called "SDPO" but distinct from canonical Hübotter / OPSD; see RUN_REGISTRY.md),
only tune lr. Chose `5e-5 ≈ √(1e-5 × 2e-4)` — geometric median between v2 (under)
and v3 (over). **It worked**: peak iter 4 = 28.00 (+2.75 over σ).

| Knob | v2 | v3 | **v4** |
|---|---:|---:|---:|
| lr | 1e-5 | 2e-4 | **5e-5** |
| n_iter | 10 | 20 | **20 (peak iter 4)** |
| n_grad_steps_per_iter | 1 | 4 | **4** |
| save_every | 5 | 3 | **1 (every iter, for rollback)** |
| eval_every | 2 | 3 | **1 (full per-iter audit visibility)** |
| early-stop on audit drop | none | none | **≥3 over 2 audits** (TOO LAX — see SDPO_RECIPE_v1) |

- File: `src/co_scientist/d5_abstract_retrieve_refine/train_mu_v4.py`
- Run dir: `runs/2026_04_27_mu_v4/`
- Production weights: `runs/2026_04_27_mu_v4/checkpoints.jsonl` row batch=4
- **Reusable recipe**: `knowledge/current/SDPO_RECIPE_v1.md`
- Pass criterion: μ-v4 isolated audit mean ≥ 26 (beats σ 25.25) — **PASSED at iter 4 = 28.00** ✅

---

## 🚨 REALIGNMENT (2026-04-25) — all prior runs ARCHIVED, Phase 1 COMPLETE

**Trigger** (2026-04-25): user identified that prior Phase 0.6 work (μ baseline, α/β baselines, audit v1/v2,
multiple pairwise rounds) was MISALIGNED with the intended D5 pipeline. Specifically: (a) oracle
abstraction was a single-round derivation-scaffolding extraction from one reference plan, NOT a
multi-round bibliography-grounded citation-cited extraction; (b) δ/ε were treated as baselines but are
actually semi-oracle conditions; (c) audit prompts had surface-form bias / floor saturation; (d) no
reviewer-standard grounding for any audit dimension.

**All prior runs archived** to `runs/_archive_pre_realignment_2026_04_25/`. Their numbers (μ proxy
8.88/20, "D5 dead" verdicts, etc.) MUST NOT be cited as authoritative going forward.

**Current state** (2026-04-25): Realignment Phase 1 COMPLETE.

- ✅ `knowledge/current/REVIEWER_STANDARDS_v1.md` — synthesis of NeurIPS/ICLR/ICML/NSF reviewer guidelines
  + 7 OpenReview adjacent papers (MTTT, ReST-MCTS, ReST-EM, Voyager, SCoRe, Guided-ReST, DeepEvolve)
- ✅ `knowledge/current/ORACLE_DESIGN_v1.md` — locked design: 6 categories (Insights/Methodology/Theory/
  Math/Empirical/Failure-modes), Opus-generated as ceiling estimate, multi-round (3 rounds) with cumulative
  memory, full bibliography fetch via S2 API, per-item citations
- ✅ `knowledge/current/AUDIT_RUBRIC_v3.md` — locked design: 9 dimensions hybrid (5 universal:
  Soundness/Significance/Originality/Clarity/Reproducibility + 4 TTT-Discover-specific: Necessity/
  Disentanglement/Compute/Reward-hacking), anchor calibrated to real OpenReview reviewer quotes
- ✅ `knowledge/current/RUN_CONFIRMATION_TEMPLATE.md` — protocol for pre-run user confirmation

**Next phase**: Phase 2 (build) — Phase 0b bibliography fetch + multi-round oracle build + audit v3 prompt
implementation + re-run baselines. To be planned in a separate plan doc; gated by per-run user
confirmation per RUN_CONFIRMATION_TEMPLATE.md.

**Pre-run confirmation protocol** (binding): for every future training/eval run, an instance of
RUN_CONFIRMATION_TEMPLATE.md must be filled out and approved by user before launch. NEVER skip.

---

## (Below: pre-realignment notes, kept for historical reference; do NOT cite as authoritative)

## Current Phase (PRE-REALIGNMENT — ARCHIVED)

**Phase 0 + Phase 0.5 sanity gates COMPLETE (2026-04-25). Phase 0b (TTT-Discover bibliography fetch) is the immediate next action.**

**Phase 0c smoke pathway v1 COMPLETE — pathway validated per Opus audit.**

Result: oracle abstraction + frozen Qwen3-30B-A3B outperforms goal-only baseline by **+1.88 / 20 on D3 canonical Opus audit (+9.4pp normalized)**, **+3.50 / 40 on D4-style Opus audit (+8.75pp normalized, same direction)**. Effect concentrated on novelty + rigor; math + realism flat (as user pre-registered). Qwen self-grader says opposite direction — grader gap re-confirmed.

Reference plan anchors at 20/20 on D3 canonical, confirming prompt calibration. A = 7/20, gap to reference = 13 points. Abstraction recovers only ~12.5% of quality gap.

**Go/no-go**: proceed to Phase 1 with two mandatory fixes:
1. Sampling duplicates bug (Tinker `SamplingParams(seed=42)` collapses all 8 samples within single `sample()` call — need per-sample seed or no seed).
2. Primary training signal must be Opus (Qwen unreliable on this pathway).

Under consideration: **abstraction format v2** — include derivation / reasoning steps, not just structural patterns. Current pattern-level abstraction ceilings around 10-12/20; closing further gap requires abstraction to carry mechanistic derivations.

**Smoke v2 complete (2026-04-24)** — derivation-augmented abstraction validated:
- Δ(A−B) D3 canonical = **+5.875** (vs +1.88 for v1, 3× lift)
- All 4 dims move (v1 only moved 2; math and realism now +1.25 each, were flat in v1)
- Zero per-plan overlap (all A ≥ 11, all B ≤ 6)
- **2/8 A plans wrote the reference's entropic objective equation** (`J = log Σ_a exp(β R(a)) · π(a|s)`) — zero did in v1
- A ceiling ≈ 12/20 (v1 ceiling was 7). 8-pt gap to reference remains, attributable to specific prior AI numbers + specific hyperparameters + full derivation details NOT in v2 hints
- Leakage acknowledged: v2 Pattern 2 Step 3 hints "log-sum-exp with β" which is ~75% to reference equation. Phase 1+ training's abstraction generator must produce derivation-style hints, which requires retrieving full-text methodology sections (not just abstracts).

**Phase 0.5 sanity gates complete (2026-04-25)**:

- 0.5a: D5 dataset migrated from D4 grant_proposal symlink to self-contained copy of D3 TTT-Discover canonical files (`research_goal.txt`, `reference_solution.txt`, `perturbations/`). D5 code paths updated. Files byte-verified identical.
- 0.5b: Qwen3-30B-A3B contamination cleared. Cutoff = March 2025; TTT-Discover = Jan 2026 (post-cutoff). Direct probe (3 questions) confirmed no in-weights knowledge of TTT-Discover. Smoke results valid.
- 0.5c: Sampling duplication fixed. New params: `temperature=1.0, top_p=0.95, no fixed seed`. Smoke v2/v3 reruns: **0% duplicates** (was 31-44%).

**Smoke rerun results (2026-04-25)**:
- v2 rerun: Δ(A−B) = +4.00 /20 (was +5.875; 68% survival)
- v3 rerun: Δ(A−B) = +3.125 /20 (was +2.50; 125%, slightly grew)
- Pathway claims VALID. A > B in both, no per-plan overlap, per-dim directions preserved.
- **DECOMPOSITION INVERTED**: pattern +1.88 / reasoning steps +1.25 / formula hints **+0.88** (originally formula was claimed dominant at +3.38). Reasoning scaffolding is now the dominant lift component.
- Statistical caveat: n=8 too small to distinguish +4.00 vs +3.13 (gap ~1.2 SE). Decomposition is suggestive not confirmed; need n ≥ 30 per condition for tight CI.

**Implication for Phase 1+**: Phase 1+ trained generator producing v3-style reasoning-scaffolding abstractions from abstracts-only retrieval should reach ~9-10/20 ceiling. Methodology full-text retrieval (for formula hints) would add ~+0.9 pts max. **Cheaper retrieval strategy is competitive**.

---

**Smoke v3 complete (2026-04-24)** — derivation scaffolding WITHOUT formula hints (ORIGINAL run, before sampling fix):
- v3 removes formula hints from v2 (no "log-sum-exp", no "Q+c·P form", etc.) but keeps reasoning-step structure
- Δ(A−B) D3 canonical = **+2.50** (v3), compared to v1 +1.88 (pattern-only) and v2 +5.875 (with formula hints)
- v3 A total = 9.25 /20 vs v2 A = 11.25, v1 A = 7.00
- Removing formula hints eliminates ~57% of v2's lift
- 2/8 A plans still wrote equations but more generic forms (E[max R], softmax with τ, not the reference's log-sum-exp)
- Template collapse worse: 4/8 A plans, 3/8 B plans byte-identical (seed=200 didn't fix it — suggests Qwen3-30B temp=0.7 mode bias)

**Decomposition of "abstraction lift"**:
- Pattern structure: +1.88 (v1)
- + Derivation reasoning steps: +0.62 marginal (v3 over v1)
- + Specific formula hints: +3.38 marginal (v2 over v3)
- Total v2 lift: +5.875

**Realistic Phase 1+ trained target: A ceiling ~9-10/20** (matching v3 since learned abstractions from bibliography 大概率 retain reasoning structure but not produce perfect formula hints). If retrieval includes full paper methodology with formulas → possible ceiling ~11/20.

**Next concrete step**: Phase 1 infrastructure build (per-goal database, multi-round orchestrator, SDPO distillation, Opus reviewer subagent). Before running any training:
1. FIX sampling duplicates bug (temp=1.0 or top_p, not just seed change)
2. Decide whether bibliography retrieval should include methodology sections (for formula-hint learnability) or abstracts-only (realistic deployment)

## Direction One-Liner

Self-distillation-style training with privileged observations (reference paper as privileged info; D5 in-house off-policy IS-loss variant — see CANONICAL_NAMING_REFERENCE.md for paper attribution; structurally distinct from canonical Hübotter SDPO 2601.20802 / Zhao OPSD 2601.18734) for long-form research plan generation, via multi-round model-selected paper retrieval + abstraction.

## Target

**Venue**: ICLR 2027 main track. Deadline ~2026-10.
**Timeline**: 12 weeks active development + 3 weeks buffer.

## Pipeline summary

```
Per instance (research goal G, source paper S with bibliography B):
  Round 1/2/3:
    - Policy selects papers from titles (learned action)
    - Retrieve selected papers' full content from B
    - Policy generates abstraction A_i given (G, content, history, prior A's)
    - Reviewer (with S as privileged info) writes feedback F_i
    - Self-distillation loss (D5 in-house off-policy IS-loss variant in current code): TEACHER samples under (selection, A_i | with review); STUDENT lp recomputed under (selection, A_i | no review); A_t = clamp((t_lp − s_lp)·scale, ±5); importance_sampling loss
  Final plan:
    - Policy generates P from (G, all content, [A_1..A_3])
    - Supervised CE loss against S's own research plan (gold target)
```

## Phase status

| Phase | Week(s) | Status |
|---|---|---|
| 0a Project scaffolding | 1 | **in_progress** |
| 0b Dataset preparation | 1 | pending |
| 0c Architecture pilot | 1 | pending |
| 1 Infrastructure modules | 2-3 | pending |
| 2 Training pilot (2 goals, Opus reviewer) | 4-5 | pending |
| 3 Full training + ablations | 6-9 | pending |
| 4 Evaluation + baselines | 10-11 | pending |
| 5 Paper draft | 12 | pending |

## Phase 0.6 baseline matrix (final, 2026-04-25)

| Baseline | Setup | Score /20 | vs μ | vs δ |
|---|---|---:|---:|---:|
| ε | frozen Qwen3-235B + ref in context | 12.13 | +3.25 | +5.38 |
| **μ** | plan-level SDPO + oracle v3 abstraction (10 iters) | **8.88** | — | +2.13 |
| β | direct SFT on reference plan (10 SFT steps × 1 ex) | 7.75 | −1.13 | +1.00 |
| δ | frozen Qwen3-30B + ref in context | 6.75 | −2.13 | — |
| α | naive Opus distillation (5 epochs × 16 Opus plans) | 6.12 | −2.76 | −0.63 |

**Kill conditions** (all NOT triggered):
- μ ≤ δ+1 (7.75)? No, μ=8.88 → architecture survives
- α ≥ μ? No, α=6.12 ≪ μ → distillation alone insufficient
- β ≥ μ? No, β=7.75 < μ → architecture > direct ref imitation

**Observations**:
- α < δ (distillation FROM Opus to 30B HURTS the model below frozen baseline) — plausibly model-scale gap + Qwen3 thinking-mode bleed during 30B-SFT-on-Opus-tokens
- Absolute ranking: ε ≫ μ > β > δ > α

## 🚨 Pairwise inversion (2026-04-25 sanity check, run `2026_04_25_pairwise_v1`)

5 matchups × 8 random pairs (position-randomized), Opus picks A/B/TIE per pair:

| Matchup | Pairwise | Absolute | Status |
|---|---:|---|---|
| μ vs ε | 0–8 (ε wins all) | μ=8.88 < ε=12.13 | ✓ same direction |
| **μ vs δ** | **0–8 (δ wins all)** | μ=8.88 > δ=6.75 | **🚨 INVERTED** |
| μ vs β | 8–0 (μ wins all) | μ=8.88 > β=7.75 | ✓ same direction |
| α vs δ | 0–8 (δ wins all) | α=6.12 < δ=6.75 | ✓ same direction |
| β vs α | 4–4 (tie) | β=7.75 > α=6.12 | partial (won → tied) |

**Pairwise ranking**: ε > δ > μ ≈ smoke_v3_A > β ≈ α
**Absolute ranking**: ε > μ > β > δ > α

**Smoke pathway claim was WITHDRAWN on 2026-04-25 then REVALIDATED via fair pairwise**: original test compared smoke_v3_A (frozen 30B + oracle) vs δ (frozen 30B + reference plan in prompt). δ won 8-0 because δ has the full reference plan as few-shot example — δ is a semi-oracle condition, not a frozen baseline. After running fair pairwise tests under test-time-equivalent conditions (NO reference in either side's prompt):
- **smoke_v3_A vs smoke_v3_B (frozen 30B + goal only): A wins 8-0** ← oracle abstraction at inference DOES help
- The "+1.88 / +4.00 / +3.13 absolute audit lifts" still had surface-form bias, but the underlying pathway claim (oracle helps inference) holds under fair pairwise.

**μ training-transfer claim CONFIRMED ≈ 0 under fair test**:
- μ vs smoke_v3_A (both have oracle): smoke_v3_A wins 6-2 → **training does not add value over fixed oracle inference**
- μ vs smoke_v3_B (training+oracle vs goal-only frozen): μ wins 8-0 → all of μ's lift comes from oracle, not training
- α vs smoke_v3_B: smoke_v3_B wins 5-2 → **Opus distillation HURTS** (real, not artifact)
- β vs smoke_v3_B: smoke_v3_B wins 6-2 → **Ref SFT HURTS** (β also has truncation issues)

**δ vs ε pairwise CONFIRMS scale advantage**: same prompt format on both sides (ref-in-context), only model size differs (30B vs 235B). Result: **ε wins 8–0**. Subagent rationale: ε plans have tighter math, internally consistent budgets, named tools (FoldX/Rosetta/NUPACK/CMIP6) + concrete numerical baseline deltas (e.g. merit factor 13.8→14.5), occasional genuine algorithmic insights (rank-weighted PG, recursive Φ); δ plans contain contradictions (claims expensive eval but budgets 10k evals) and stay in generic PG-buffer template. The +5.38 absolute audit gap is REAL — ε is the genuine ceiling at the frozen-with-reference tier; pairwise corroborates absolute on this comparison.

**Final pairwise ranking** (consistent across all 7 matchups): **ε ≫ δ ≫ μ ≈ smoke_v3_A > β ≈ α**

Three substance tiers:
1. ε (235B + ref): genuine ceiling, distinct
2. δ (30B + ref): 30B's hard ceiling — 30B with reference plan in prompt is the best 30B can do
3. 30B trained/scaffolded (μ, smoke_A, α, β): all fail to clear δ — but δ is a semi-oracle (sees full reference plan), so this comparison is unfair. Under fair test-time conditions (no reference in either prompt), the picture is: smoke_v3_A (oracle inference) > μ ≈ trained variants > smoke_v3_B (bare frozen) ≈ α (Opus-distill) ≈ β (ref SFT).

## Fair pairwise re-evaluation (2026-04-25 take 2; corrects above narrative)

5 fair pairwise matchups in `runs/2026_04_25_pairwise_v1/fair_pairwise_summary.md`. All comparisons under test-time-fair conditions (NO reference plan in any side's prompt):

| Matchup | Winner | What it means |
|---|---|---|
| **smoke_v3_A vs smoke_v3_B** | smoke_v3_A 8-0 | Oracle abstraction at inference HELPS frozen 30B |
| **μ vs smoke_v3_A** | smoke_v3_A 6-2 | SDPO training does NOT add value over oracle inference |
| **μ vs smoke_v3_B** | μ 8-0 | Training+oracle ≫ goal-only frozen (but lift is from oracle, not training) |
| **α vs smoke_v3_B** | smoke_v3_B 5-2 | Opus distillation HURTS vs frozen baseline |
| **β vs smoke_v3_B** | smoke_v3_B 6-2 | Ref SFT HURTS vs frozen baseline |

**Definitive D5 truths** (corrected from earlier narrative):

1. **Inference-time scaffolding (oracle abstraction in prompt) WORKS** — pairwise 8-0 over plain baseline. This is a real, measurable lift.
2. **Plan-level SDPO training DOES NOT internalize anything** — μ ≤ smoke_v3_A, training transfer ≈ 0, weak negative.
3. **Naive distillation methods (α, β) actively HURT** — both lose to frozen baseline. Model-scale gap + training instability.
4. **D2 pivot revived**: "inference-time methodological scaffolding paper" is now backed by clean pairwise 8-0 evidence on a 30B model, against a fair frozen baseline.
5. **D5 training story dead** for current configuration — but the inference-time scaffolding sub-story is genuinely positive and publishable.

**D5 paper viable framing options**:
- (a) Inference-time scaffolding paper: oracle abstraction (Opus-extracted patterns) lifts 30B via prompt scaffolding; no training needed; quantify with pairwise. Workshop-grade single-goal proof; full paper if cross-goal extends.
- (b) Methodology paper: absolute audit's surface-form bias + fairness pitfalls (semi-oracle baselines) + pairwise as the right metric for long-form generation eval. Standalone methodology contribution.
- (c) Combined paper: positive inference-scaffolding result + negative training-internalization result, framed as "where compute should go in long-form generation: inference scaffolding > parameter updates at 30B scale".

**Recommended next step**: cross-goal validation of oracle-scaffolding pathway. Pick 2-3 D3-style research papers (post-Qwen-cutoff), extract Opus-patterns oracles, run smoke_A vs smoke_B pairwise on each. If smoke_A wins on 2/3 new goals → cross-goal generalization established → workshop-paper-grade evidence. Cost: ~$30, ~1 day work.

Implication: on this single goal, with this grader, **parameter scale dominates training method** — the 30B model cannot exceed what it produces when handed the answer in-context, regardless of training stack.

## Audit prompt v2 attempt (2026-04-25, FAILED strict gate; tier-coarse only)

Redesigned absolute audit prompt with two-pass enumeration (claim_list + scaffold_list) + anti-pattern penalty (-1/dim cap, floor 1). Two iterations:
- Iter 1: 5/7 directional, all 3 v1-inversions flipped, but `mu_vs_beta` inverted (β > μ; pairwise μ wins 8-0) and `beta_vs_alpha` β > α (pairwise tied)
- Iter 2 (sharper non-Pattern-N scaffold detection — vague hparams + passive verbs + generic algos): 5/7 directional, all 3 inversions flipped, but now `mu_vs_beta` ties at 4.0 (both floor) — over-correction floors all 30B-trained variants

**v2 outcome**: tier-level discrimination CORRECT (ε=14.4 ≫ δ=7.4 ≫ {μ=4.0, β=4.0, α=4.6, smoke=4.0}); within-bottom-tier ranking UNRELIABLE (penalty floor saturation under -1/dim cap).

**Per pre-registered fallback (plan `steady-tinkering-wave.md`)**: 2nd iteration also failed → **abandon absolute-prompt-redesign route; switch primary metric to pairwise tournament**. v2 retained only as cheap coarse tier-level secondary signal.

**Operational policy going forward**:
- Cross-model-scale comparison (ε vs δ): v2 absolute acceptable; pairwise corroborates
- Within-scale training-method comparison (μ vs β, α vs β etc.): **pairwise tournament REQUIRED**, absolute audit (v1 OR v2) unreliable
- Phase 1 retrieval ablation evaluations (if D5 continues): MUST be pairwise.

Phase 2 (re-audit historical baselines under v2) **SKIPPED** — no point re-scoring with floor-bottoming v2 for within-tier rankings; pairwise data already captures the truth.

**Key finding**: μ does NOT actually beat δ at substance. Absolute +2.13 lift was surface-form bias — μ plans inherit oracle's "Pattern X" structure (looks rigorous), but δ plans paraphrase the reference plan's concrete specs (LoRA rank, learning rate, equations, numbered baselines), and Opus consistently prefers concrete substance over structural pattern when forced to discriminate.

**Updated paper narrative implications**:
- D5's *training mechanism* (μ vs β, both 30B-trained) IS real (8–0 pairwise)
- D5's *base ingredient* (oracle abstraction in inference prompt) does NOT actually beat "reference plan in context" prompt
- Smoke v3-rerun A=9.25 over δ=6.75 likely had the same grader bias — needs pairwise verification
- "Opus distillation hurts 30B" partially holds (α loses to δ, ties with β) — distillation is at most equivalent to no-training-at-all

## Active experiments

- **2026-04-25 μ baseline v1** — `runs/2026_04_25_mu_baseline_v1/` — COMPLETE.
  - 10 iters, 8 plans/iter, eval_every=2, anchor_ce=0.0, oracle v3.
  - **μ proxy = 8.88/20** (last-3 audit means at iters 4, 6, 8).
  - Audit trajectory: iter 0=9.13, iter 2=8.88, iter 4=9.38, iter 6=8.00, iter 8=9.25.
  - SDPO signal healthy throughout (mean_adv ≈ 0.5, pos_frac ≈ 0.80 with real critiques; ≈ 0.21 / 0.72 with cold-start fallbacks).
  - Iter 2 critic timed out (900s) → fell back to cold-start, iter 3 trained on cold-start critique. All other iters had real critiques.
  - **Decision: MARGINAL** (7.75 < μ < 9.00). μ - δ = +2.13 (above STOP threshold), but iter 0 (no-LoRA) audit = 9.13 ≈ iter 8 audit = 9.25 → 8 training iters produced ≈ 0 net gain over frozen-with-oracle-and-cold-start. SDPO does not transfer the critique signal into evaluable plan-quality improvement.
  - Per-dim: math regressed (2.13→1.38), novelty oscillated (2.0↔3.0), realism flat (3.0), rigor flat (2.0).

## Next concrete action

Phase 0.5 (sanity gates) ALL COMPLETE (2026-04-25):
1. ✅ Phase 0.5a: dataset migrated, D5 self-contained
2. ✅ Phase 0.5b: Qwen contamination cleared
3. ✅ Phase 0.5c: sampling fix verified, decomposition INVERTED (reasoning > formula hints)

**IMMEDIATE NEXT actions (per Phase 0.6 added 2026-04-25)** — run baseline battery BEFORE Phase 0b infrastructure:

**Phase 0.6 baseline pilot** (~7-11 days, ~$265):
1. **δ** (frozen 30B + reference in context, no training, 1 hr, ~$5) — STARTING NOW
2. **ε** (frozen 235B + reference in context, no training, 1-2 hr, ~$10)
3. **β** (Qwen SFT on reference, 1-2 days, ~$0)
4. **α** (naive Opus SFT, 2-3 days, ~$50)
5. **μ** (oracle abstraction + plan-level SDPO, 3-5 days, ~$70-100 actual) — ceiling estimator — **IN PROGRESS** (framework built 2026-04-25, training kicking off)

**Decision matrix** (after baseline battery):
- μ ≤ δ + 1 → STOP. Plan-level SDPO doesn't internalize anything.
- α ≥ μ → reframe. Distillation is enough.
- β ≥ μ → reframe. Architecture is overhead.
- otherwise → continue Phase 0b/1 with empirical anchors

After baseline battery clears decision matrix:
6. **Phase 0b**: TTT-Discover bibliography fetch (arxiv 2601.16175 → S2 API)
7. **Phase 1 modules**: full SDPO infrastructure build
8. **Phase 2 training pilot** with mandatory δ-checkpoint at end

## 🚨 Active Risks (carry into Phase 1+)

1. **δ-baseline early-warning** (added 2026-04-25): MANDATORY check at Phase 2 end. δ = frozen 30B + reference plan in context (no training). If D5 trained ≤ δ + 1, training claim fails. Per smoke estimates: δ ≈ 8-10/20, D5 trained ceiling ≈ 9-10/20. Gap is small.
2. **ε-baseline** (frozen 235B + reference in context): D5 likely loses by 2-4 pts. Mitigation: reframe as parameter-efficient (30B trained competitive at 8× lower inference cost), not absolute SOTA.
3. **Single-goal scope**: paper accept-risk if no cross-goal evidence. Mitigation: 2-3 held-out goals at Phase 4 eval (without retraining), per Plan Advice 4.
4. **Sampling fix re-validation in training**: template collapse may return at higher entropy on full plan length; check during Phase 1 unit tests.
5. **Anti-distillation defense (α/β/γ)**: must run early — at Phase 2 not Phase 4 — to catch before sunk Phase 3 cost.

## Budget

- Opus reviewer: $200 (Phase 2), potentially transitioning to self-review in Phase 3
- Total projected: $500-800 (optimistic) to $2000 (if staying Opus)
- Hard ceiling: $8000

## Key design commitments (from 2026-04-24 plan)

1. Retrieval source: TTT-Discover paper's bibliography (single goal, not web search, not multi-goal)
2. Paper selection: learned action (not human curriculum)
3. 3 rounds with cross-round history visibility
4. Distill both selection + abstraction logprobs (P2)
5. Reviewer: Opus → self-review if ρ(Opus, self) > 0.7 on held-out
6. Reviewer has source paper as privileged info (distinct from standard SDPO's teacher-evaluative privileged info)
7. **Plan-level mechanism (revised 2026-04-25)**: SDPO at plan level (not supervised CE only). Plan critic has full source paper as privileged info; outputs structured (idea_alignment / missing_components / incorrect_assumptions / feasibility) + free-form text. Optional small-weight supervised CE on reference plan for stability.

8. **Buffer structure (revised 2026-04-25)**: typed slots (concepts / methods / assumptions / limitations / connections) with deduplication, not free-form stacked text.

9. **Extraction critic output (revised 2026-04-25)**: structured (relevance / faithfulness / missing / noise) scores + free-form text hybrid.
8. **Training-loop structure: Option Z** — same instance, run → SDPO update → re-run on updated model → update. Phase 2 tests stability; fallback to Y (different-instance-per-pass) if drift.

## Evaluation framework (2-layer)

**Primary metric (Layer 2)**: final plan quality on held-out goals.
- Opus 4.7 depth audit /40 (8 rollouts/goal, bootstrap 95% CI)
- Opus pairwise preference vs reference plan (target ≥40% preference for trained plan)
- **Critical baseline**: frozen Qwen3-235B + single retrieve + reference plan as few-shot. If this ≥ trained 30B, paper collapses.
- Required effect size: trained 30B ≥ strongest baseline + 3 points, CI non-overlapping

**Mechanism evidence (Layer 1)**: sub-capability trajectories
- Selection precision@5 per round (model-selected ∩ source citations)
- Abstraction quality (Opus scoring on specificity/grounding/applicability/novelty)
- Pathway ablation: oracle-abstraction-frozen vs trained-pipeline (isolates abstraction vs composition contribution)
- SDPO KL trajectory (with-review vs without-review) during training — should decrease

## Novelty framing (for paper; AVOID generic claims)

- ❌ Do NOT claim novelty from "long-form is hard" (DR Tulu/FLARE already attack this)
- ❌ Do NOT claim novelty from "external info injection" (DR Tulu validates this)
- ✅ Claim novelty from 4-way specific mechanism combination:
  1. SDPO with FACTUAL privileged info (reference paper content, not teacher eval)
  2. Learned paper selection action
  3. Abstraction as iterated artifact (not raw text like FLARE)
  4. Reference plan as supervised gold anchor (not learned reward)

## Plan reference

See `~/.claude/plans/giggly-jumping-hollerith.md` for the full plan document.
