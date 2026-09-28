# D5 Paper Materials

*Canonical store for paper-writing material. Last updated: 2026-04-27.*

This folder contains everything the D5 paper needs:
- Findings (key results in narrative form)
- Experiments (raw data dumps)
- Figures data (matplotlib-ready CSVs)
- Plan samples (exemplar generations for appendix)
- Methodology (audit rubric, SDPO recipe, oracle design)
- Related work excerpts
- Open questions for future work

## Reading order

1. `../PROJECT_OVERVIEW.md` — start here for context (one level up)
2. `findings/` — what we discovered, by importance:
   - **F2** μ-v4 iter-4 peak (+2.75 over σ) — main positive result
   - **F1** σ oracle inference dominant lift (+9.63 over ξ) — main scaffolding result
   - **F3** SDPO multi-round instability (3-lr ablation) — main negative result
   - **F4** critique-token-blindness diagnosis — methodology lesson
   - **F5** audit ensemble methodology — paper's audit-design contribution
   - **F6** 4-dim anchor saturation — auxiliary methodology lesson
   - **F7** Phase 4a forward-citation transfer **FINAL 2026-04-27** (after D1 n=24 stress test): forward-citation transfer is **DEFINITIVELY NULL**. meta_ttl n=24 strict Δ -0.96 + pairwise σ' 14/9/1 (σ' DIRECTIONAL CONFIRMED ≥14/24); tool_v_ttrl n=24 strict Δ -0.58 + pairwise σ' 12/10/2 (TRUE NULL). 3-stage corroboration trail: original 2-plan +2.13/+3.00 → strict+pairwise n=8 inconclusive → strict+pairwise n=24 NULL. M8 anchoring effect is the surviving methodology contribution.
   - **F9** Phase 5 continual SDPO chain NULL across 4-cell grid; per-token CE wrong abstraction sub-finding — paper's strongest negative-with-mechanism result; **TRIPLE-CORROBORATED 2026-04-27** (in-loop /20 + isolated /45 + pairwise 4-4 TIE + strict 1-plan Δ +1.87)
   - **F8 v2** Phase 3 inference-time distillation pipeline **REVISED 2026-04-27 PM**: τ_v4_clean (PATCHED + fresh) cross-instance mean **24.50/45 ≈ σ baseline 25.25**; pairwise rerun (Step E, seed=52) τ_v4_clean vs σ_v4 = **4-4 TIE** (Step 2.5's 7-1 STRONG was lucky-draw artifact). Three failure modes characterized: (1) self-review degeneration (M7 Smoke A 8-0, unchanged), (2) single-shot evaluation inflation at σ_within ≈ 1.17 (Step 2.5 7-1 → Step E 4-4 demonstration), (3) citation-tag leakage from distillation (~85% fixable at plan-prompt level). Negative-result-with-mechanism — ICLR-publishable.
3. `experiments/` — raw data per finding (referenced by findings)
   - **E1** 7-baseline ranking (audit_v3 ISOLATED)
   - **E2** μ-v4 trajectory (per-iter scores)
   - **E3** SDPO lr ablation (μ-v2 / v3 / v4)
   - **E4-E5** PICK/EVAL plan content evolution (PUCT mention diagnosis)
   - **E6** μ-v4 pairwise corroboration — Opus pairwise judge cross-checks audit_v3 ISOLATED ranking on close cluster (added 2026-04-27)
   - **E8** Phase 3 distillation evidence (τ trajectory + Step 2.5 pairwise + Smoke A self-review + **Step 2a/A/B/C/E re-validation 2026-04-27 PM**: σ_within = 1.17/0.50, τ_v4_clean = 24.50, Step E pairwise 4-4 TIE) — raw data dump for F8 v2
4. `figures_data/` — matplotlib-ready CSVs (G1 trajectory, G2 ranking, G3 lr ablation, G4 critique-vs-advantage)
5. `plan_samples/` — exemplar plans (good iter-4, iter-0 baseline, collapsed iter-6, σ control)
6. `methodology/` — copy of audit rubric, SDPO recipe, oracle design (self-contained for paper)
   - **M7** Why Qwen3-30B self-review fails (Smoke A 8-0): Privileged Info Comprehension gap; mentions same conceptual terms as Opus but lacks formula/hparam/domain specificity. SDPO+30B-self-critic NOT VIABLE; 235B+ critic needed for self-distill story (added 2026-04-27)
   - **M8** Audit-pairwise divergence: 2-plan/subagent batch anchoring inflates close-cluster Δ by 2-3 points; strict 1-plan + pairwise required at Δ < 5/45. **STRENGTHENED via D1 n=24 stress test**: anchoring effect persists at n=24, no μ' lift emerges with 3× sample size, position bias within bounds. NEW methodology contribution emerging from F7 collapse (final 2026-04-27)
7. `next_steps/` — Phase 3 open questions

## Update rule

Every new experiment that produces paper-relevant data MUST update this folder
in the same commit as the experiment. Don't let paper_materials/ drift from
runs/.

## Git provenance

This folder was first populated 2026-04-27 after Phase 2 SDPO ablation
verdict (μ-v4 iter-4 = 28.00/45). All numbers and exemplars in this folder
trace back to:
- `runs/2026_04_26_phase2F_audit_isolated/` — 7-baseline σ baseline data
- `runs/2026_04_27_mu_v4/` — μ-v4 training run
- `runs/2026_04_27_mu_v4_early_iter_audit/` — 9-dim isolated audit on iter 0-5
- `runs/2026_04_27_mu_v4_pairwise/` — pairwise corroboration (added 2026-04-27)
