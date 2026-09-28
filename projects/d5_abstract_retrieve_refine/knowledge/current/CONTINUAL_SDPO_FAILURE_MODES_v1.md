# Continual SDPO Failure Modes v1 — Developer-Facing Reference

*Last updated: 2026-04-26 (Phase 5 verdict, commit `fdf023d`)*

**Audience**: D5 developers (Session A/B + future sessions) considering continual SDPO experiments. F9 (`paper_materials/findings/F9_continual_sdpo_field_expert.md`) is the paper-facing canonical narrative; THIS doc is the one-page operational summary for development decisions.

## TL;DR

**Naive continual SDPO from a μ-v4 anchor at lr=5e-5 + n_grad_steps=4 fails across 4 grid cells**:
- Adam state full vs fresh: BOTH fail (cliff at iter 1)
- Anchor saturation iter 4 vs iter 2: BOTH fail (cliff at iter 1 vs slow decline over 4 iters)
- anchor_ce regularization 0 vs 0.1: REGULARIZATION ACCELERATES COLLAPSE (iter 2 cliff vs iter 4 of unregularized)

**Best v5 vs μ-v4 isolated /45 on Tool-V**: v3 iter 0 Δ = +0.62 (within n=8 noise); α iter 0 Δ = **-2.62** (negative — α plans WORSE than anchor).

**Verdict**: per-token CE anchor is the WRONG ABSTRACTION for continual SDPO. Mitigations are non-local (plan-level replay / EWC-Fisher / LoRA-per-paper merge), not hyperparameter tweaks.

## Empirical grid (4 cells, all NULL)

| Cell | Anchor | Adam | anchor_ce | Best in-loop /20 | Failure mode |
|---|---|---|---:|---:|---|
| cliff_v1 | μ-v4 iter 4 | full | 0 | 8.38 (iter 0) | iter-1 cliff -3.50 (audit_drop_threshold trigger) |
| cliff_v2 | μ-v4 iter 4 | fresh | 0 | 9.00 (iter 0) | iter-1 cliff -2.13 |
| v3 | μ-v4 iter 2 | fresh | 0 | 9.375 (iter 0) | 4-iter slow decline -2.71 (9.375→6.67) |
| α | μ-v4 iter 2 | fresh | **0.1** | **10.375** (iter 0) | iter-2 cliff -3.0 (regularization backfires) |

Audit_drop_threshold=2.0 fires automatically in cells where rolling drop exceeds 2.0; user halt where threshold not yet triggered but trajectory clearly bad.

Run dirs (all preserved):
- `runs/2026_04_28_mu_v5_tool_v_cliff_v1/`
- `runs/2026_04_28_mu_v5_tool_v_cliff_v2/`
- `runs/2026_04_28_mu_v5_iter2anchor_5iter/`
- `runs/2026_04_28_mu_v5_anchor_ce_v1/`

Best-iter inference + isolated /45 audits:
- `runs/2026_04_28_mu_v5_v3_iter0_inference/` + `runs/2026_04_28_phase5_audit_v3_iter0/`
- `runs/2026_04_28_mu_v5_anchor_ce_v1_inference/` + `runs/2026_04_28_phase5_audit_anchor_ce/`

## 3-axis diagnostic (decomposed)

| Axis | Effect | Mechanism |
|---|---|---|
| Optimizer state | Controls cliff MAGNITUDE | Inherited Adam moments amplify drift in stale directions; fresh moments dampen but don't prevent. |
| Anchor saturation | Controls cliff TIMING | Saturated weights (μ-v4 iter 4) sit 1 step from cliff edge; non-saturated (iter 2) have ~3 steps of room before equivalent drift accumulates. |
| anchor_ce regularization | ACCELERATES collapse | TTT-D anchor pulls toward TTT-D prose style; Tool-V SDPO pulls toward Tool-V content. Two competing gradients fight, model converges into incoherent middle ground (iter 2 jargon-salad mixing TTRL + TTT-Discover formulas). |

## Mechanistic working hypothesis

SDPO's per-token advantage = `teacher_lp - student_lp`, applied to ALL tokens in the generated plan with equal weight. New paper's critic re-shapes teacher's distribution toward Tool-V-style content; advantage signal pulls student weights in same direction. BUT:

1. **Plan tokens are heterogeneous in semantic role**: ~70% are stylistic/structural (chat scaffolding, "we propose...", section headers, generic framing) and ~30% are content-bearing (specific equations, hparams, named methods). Stylistic tokens get same per-token advantage as content tokens.

2. **Stylistic-token gradient mass dominates**: when student moves, the stylistic substrate moves first because it has more total tokens × gradient. The model's "writing style" shifts toward Tool-V's stylistic conventions before the actual content is internalized.

3. **Style shift in absence of full content rewrite = degenerate region**: model produces Tool-V-flavored text but the underlying content is still half-TTT-D-conditioned. Plans become "incoherent middle" — neither domain's coherent style, mixing terminology.

4. **Critic signal sparsity exacerbates**: only 1 plan/iter gets re-critiqued. As the population drifts (8 plans/iter), most plans train on stale critique. Drift compounds.

This hypothesis predicts that mitigations targeting **content-token-mass amplification** OR **plan-level (not token-level) regularization** OR **denser critic feedback** would help. Per-token CE anchor (α experiment) does the OPPOSITE: it locks stylistic tokens to TTT-D's stylistic distribution, while content tokens are left to be pulled by Tool-V SDPO. Result: even less coherent.

## Mitigation candidates (Phase 6 future work)

| # | Mitigation | Predicted effect | Cost to test |
|---|---|---:|---:|
| (i) | **Plan-level replay buffer**: per iter, mix 4 plans on Tool-V (current SDPO) + 4 plans replayed from TTT-D buffer (re-critiqued via TTT-D source paper). All 8 in SDPO loss. | Addresses critic sparsity (H_root_3) AND increases content-token diversity (H_root_1). Standard CL technique. **Highest expected info gain.** | ~$15-20 subagent (replay critic doubles per-iter critic calls) + ~2.5 hr Tinker |
| (ii) | **EWC penalty with Fisher Information**: compute per-parameter Fisher importance from μ-v4 EVAL plans; add `λ × Σ_i F_i (θ_i - θ_μ-v4_i)²` to loss. | Addresses anisotropic-manifold (H_root_2) at parameter level (not output). | Implementation: ~1-2 hr code. Run cost: ~$5 + 1.5 hr |
| (iii) | **LoRA-per-paper + weighted merge**: train a SEPARATE LoRA on Tool-V from base Qwen3-30B (NOT chained from μ-v4); inference-time merge weights = α × μ-v4 + (1-α) × Tool-V LoRA. | Bypasses chain entirely; Phase 4a's σ' + μ' on follow-ups is the implicit validation of the merge approach. | ~$5 + 1.5 hr to train Tool-V LoRA from base; merge logic ~30 min code |

**Recommended Phase 6α**: option (i) plan-level replay buffer. Cleanest path to "did continual rescue?" answer; standard literature technique; addresses two root-cause hypotheses simultaneously.

## Don't-Do List for Future Sessions

- ❌ Don't naively chain SDPO from μ-v4 (any iter) to new paper without explicit mitigation. 4 grid cells already exhaust the obvious knobs.
- ❌ Don't add anchor_ce regularization (any weight) — confirmed to accelerate collapse via two-gradient conflict. Output regularization is the wrong abstraction.
- ❌ Don't try `lr=1e-5` for continual — μ-v2 (lr=1e-5 from base) already showed zero learning at this lr; from a μ-v4 anchor it would be even more dampened.
- ❌ Don't propose Phase 4b cross-domain stretch — DEPRECATED 2026-04-26. Field-expert metaphor breaks for cross-domain switching.
- ❌ Don't add a "reset_optimizer_state with smaller lr" follow-up — both axes already tested; combination unlikely to rescue.

## Do-List for Future Sessions

- ✅ Use μ-v4 single-paper SDPO recipe verbatim for new TTT-family goals (per `SDPO_RECIPE_v1.md`).
- ✅ For continual experiments, implement ONE mitigation from the (i)/(ii)/(iii) menu above. Pre-register binding criteria in DECISIONS BEFORE running.
- ✅ Refer audit-noise context: n=8 audit Δ has bootstrap 95% CI ≈ ±3.5; require Δ ≥ +1.0 with lower 90% CI > -2 for robust claim.
- ✅ For Phase 7 paper write: F2/F4/F7/F9 + audit_v3_isolated methodology is the validated 3-finding cohesive narrative; Phase 6 outcome (if run) becomes F10.

## 2026-04-27 update — pairwise + strict corroboration

Phase 5 quality audit remediation (DECISIONS.md (f) closing, 2026-04-27) ran 2 corroboration protocols:

- **Pairwise tournament**: μ-v4 vs μ-v5_v3 iter 0 on Tool-V → 4-4 TIE → Phase 5 NULL CORROBORATED
- **Strict 1-plan/subagent re-audit**: Δ +1.87 (within noise); α Δ +0.26 (softens original -2.62 verdict)
- **3-protocol agreement**: in-loop /20 + isolated /45 + pairwise + strict all NULL

The α anchor_ce verdict refines: original "α plans WORSE than anchor (-2.62)" → "α plans TIE with anchor (+0.26)" under strict 1-plan calibration. The iter-2 cliff in /20 trajectory remains the load-bearing failure signal. The methodological gap between original audit and strict (|shift| 2.89 on α) is itself the H1-anchoring evidence captured in M8.

**Methodology M8** (`paper_materials/methodology/M8_audit_pairwise_divergence.md`): 2-plan/subagent batching inflates close-cluster Δ by ~2-3 points. Strict 1-plan + pairwise required at Δ < 5/45. This is now standing protocol for D5.

## 2026-04-27 update — D-series investigation outcomes (Phase 4a corroboration + 6α deferred)

D-series 6 open questions investigation (DECISIONS.md (g)) ran:
- **D1 n=24 stress test** on Phase 4a meta_ttl + tool_v_ttrl: BOTH goals confirm NULL transfer at n=24. F7 verdict: transfer DEFINITIVELY ABSENT.
- **D2 extract_solution regex bug**: fixed (strip `<think>` first); diagnostic on 272 plans → 0 extraction-bug residuals.
- **D3 7-baseline σ/δ/α pairwise**: σ vs δ pairwise 6/2 ≥6/8 (audit underestimated); 2 α matchups had position-bias outside 75% (M8 caveat).
- **D4 Phase 6α plan-level replay buffer continual SDPO**: **DEFERRED to next session**. Pre-reg locked in DECISIONS.md (g). Code change scope: add `use_replay_buffer` config + replay loader + 2nd critic call to `train_mu_v4.py`. Estimated cost: ~$25 subagent + 4 hr Tinker.

**Implication for this doc**: F9 NULL is now the strongest single negative result in D5 (3-protocol corroborated, persists at n=24-equivalent across mu_v4_pairwise's 24-plan tournament). D4 Phase 6α is the standing next experiment to test whether plan-level replay rescues continual SDPO. Pre-reg: pairwise ≥6/8 vs μ-v4 AND strict Δ ≥+1.0 → F10 positive; otherwise F10 negative (strengthens F9 "non-local mitigations only").

## Related artifacts

- **F9** (paper-facing): `paper_materials/findings/F9_continual_sdpo_field_expert.md` (now triple-corroborated NULL)
- **M8** (NEW methodology): `paper_materials/methodology/M8_audit_pairwise_divergence.md`
- **DECISIONS** entries: 2026-04-26 (c) Phase 5 design, (d) Phase 5α anchor_ce, (e) Phase 5 NULL verdict + Phase 4b deprecation, (f) **Phase 5 quality audit remediation + 2026-04-27 closing section** (pairwise + strict verdicts)
- **Code commits**: `f3b9485` init_state_path; `d2d112c` reset_optimizer_state; `a984640` anchor_ce regularization; `fdf023d` Phase 5 COMPLETE
- **Run dirs (corroboration)**: `runs/2026_04_28_phase5_remediation_pairwise/` (32 pairwise subagents); `runs/2026_04_28_*_audit_strict/` × 5 (80 strict subagents)
- **Memory**: `~/.claude/projects/-home-silas-co-scientist-project/memory/project_d5_phase5_continual_sdpo.md` (final NULL verdict)
- **Plan**: `~/.claude/plans/adaptive-churning-perlis.md` (Phase A done; Phase B done 2026-04-27)
- **PHASE_PLAN_v2.md**: §10 (Phase 5 COMPLETE), §15 (Phase 4b DEPRECATED), §16 (Phase 6 framework)
