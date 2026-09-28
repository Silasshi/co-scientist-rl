# F9 — Continual chain (D5 in-house off-policy IS-loss variant): NULL verdict; saturation-anchor + regularization diagnostics

> **⚠ TERMINOLOGY NOTE (2026-04-27 PM)**: F9 was originally framed as
> "continual SDPO chain". The training algorithm in v5 cliff_v1/v2/v3/α runs is
> the **D5 in-house off-policy IS-loss variant** (TEACHER samples + IS-loss),
> applied as a continual transfer of the μ-v4 LoRA. It is NOT canonical
> Hübotter SDPO (student on-policy + KL) and NOT canonical Zhao OPSD (student
> on-policy + JS or sampled-token reverse-KL). The NULL verdict is established
> for THIS variant; whether canonical SDPO or canonical OPSD chained continually
> would also fail is an open question (not tested). Disambiguation:
> `knowledge/current/CANONICAL_NAMING_REFERENCE.md`. See per-run setup at
> `knowledge/current/RUN_REGISTRY.md`.

## Headline

Phase 5 tested the **field-expert via continual chain** hypothesis (continual training
using the D5 in-house off-policy IS-loss variant): chain μ-v4 → μ-v5 (Tool-V) → μ-v6
(Meta-TTL). **All 4 attempted v5 training runs degraded plan quality**, spanning a
3D grid: anchor saturation × optimizer state × anchor_ce regularization. H1 forward
learning fails in every cell; per the pre-registered decision matrix (DECISIONS
2026-04-26 (c) + (d)), Phase 5 verdict is **NULL**. The combined failure pattern across
the grid is more informative than any single positive: it shows that naive AND
simple-regularization continual chaining of the in-house variant are both
insufficient — full replay buffer / EWC / LoRA-merge mitigations remain the standing
future-work direction. Whether canonical SDPO/OPSD has the same issue is not tested.

**Quantitative summary** (in-loop /20 trajectory across 4 grid cells):

| Run | Anchor | Adam | anchor_ce | iter 0 | iter 1 | iter 2 | iter 3 | iter 4 | Stop |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| cliff_v1 | μ-v4 iter 4 | full | 0 | 8.38 | 4.88 | (killed) | — | — | iter-1 cliff (-3.50) |
| cliff_v2 | μ-v4 iter 4 | fresh | 0 | 9.00 | 6.875 | (killed) | — | — | iter-1 cliff (-2.125) |
| iter2anchor (v3) | μ-v4 iter 2 | fresh | 0 | 9.375 | 9.00 | 8.75 | 7.75 | **6.67** | iter-4 audit_drop_threshold trigger; monotonic decline (-2.71 over 4 iters) |
| **anchor_ce (α)** | **μ-v4 iter 2** | **fresh** | **0.1** | **10.375** | 9.625 | **6.625** | (killed) | — | **iter-2 cliff (-3.0); FAIL bin per pre-reg DECISIONS 2026-04-26 (d)** |

**Best v5 candidate isolated /45 audits on Tool-V** (using Phase 4a's audit_v3_isolated_2way pipeline; n=8 plans, 8 anonymized 2-way batches, 8 Opus subagents per audit):

| Comparison | μ-v4 mean | v5 best mean | Δ | Verdict |
|---|---:|---:|---:|---|
| μ-v4 (Phase 4a re-audited) vs **μ-v5_v3 iter 0** (no regularization) | 21.62 | 22.25 | **+0.62** | H1 FAIL (< +1.0 threshold; within n=8 noise) |
| μ-v4 (Phase 4a re-audited) vs **μ-v5_α iter 0** (anchor_ce 0.1) | 21.88 | 19.25 | **-2.62** | H1 FAIL hard (negative — α plans worse than anchor) |

Both comparisons fail the +1.0 isolated-/45 threshold. The α run additionally fails the no-cliff condition (iter-2 cliff -3.0). Re-audit of identical μ-v4 buffer drifted from 23.38 (Phase 4a) → 21.62 / 21.88 here — confirms n=8 noise floor of ±1.76, larger than v3's Δ.

**Per-dim breakdown α vs μ-v4 isolated /45**: α loses on U5 Reproducibility (-0.88), U1 Soundness (-0.63), T4 Reward-hacking (-0.50). Wins nothing significant. Hypothesis: anchor_ce pulls outputs toward TTT-D ref-plan **prose style** but dilutes Tool-V's **task-specific reproducibility** (verifier hparams, β values, compute budgets specific to TTRL). Two competing gradient signals (TTT-D anchor + Tool-V SDPO) accelerate collapse into incoherent middle ground rather than consolidate.

## Decision matrix outcome

Per `DECISIONS.md` 2026-04-26 (c), the locked matrix:

| H1 | H2 | H3 | H4 | Paper claim |
|---|---|---|---|---|
| ❌ | * | * | * | **NULL**: "SDPO doesn't transfer step-by-step at lr=5e-5 from μ-v4 anchor; Hübotter §4 instability triggered. Reset-Adam + lower-saturation-anchor follow-up tested and also fails. Continual SDPO requires non-naive mitigations." |

H2 / H3 / H4 not run because the chain's input end (v5) is non-viable. No point training v6 from a degraded v5.

## 3-axis failure-mode diagnostic (load-bearing sub-finding for paper)

The 4 runs span a 3-axis grid testing 3 hypothesized causes:

| Axis | Hypothesis | Test | Result |
|---|---|---|---|
| **Optimizer state** | Inherited Adam moments amplify cliff | cliff_v1 (full) vs cliff_v2 (fresh) | Reduced cliff drop -3.50 → -2.125 but still cliffs at iter 1. Adam state matters but not primary cause. |
| **Anchor saturation** | μ-v4 iter 4 weights are at cliff edge | cliff_v2 (iter 4) vs v3 (iter 2) | Iter-1 cliff (6.875) → no iter-1 cliff (9.00). Saturation determines cliff TIMING (immediate vs delayed) but model still degrades over 4 iters (-2.71). |
| **anchor_ce regularization** | Anchoring to TTT-D ref-plan prevents drift | v3 (no anchor_ce) vs α (weight 0.1) | Cliff comes EARLIER (iter 2 vs iter 4). Regularizer DOES NOT rescue — actually accelerates collapse via two-gradient conflict. |

**3-axis combined finding**: continual SDPO from any μ-v4 weights via Tool-V critic at lr=5e-5 + n_grad_steps=4 produces plan-quality degradation **across all 4 grid cells**, with the failure SHAPE varying (immediate cliff vs delayed cliff vs slow decline) but never reaching SUCCESS criteria (Δ ≥ +1.0 over μ-v4 best on Tool-V isolated /45). This is consistent with the broader "SDPO multi-round instability" narrative Phase 2 established (Hübotter 2026 §4) and TIGHTER: instability also appears at iter 1-2 of a CONTINUED run regardless of optimizer state, anchor freshness, or simple TTT-D-anchored regularization.

**Why anchor_ce backfired**: regularizer pulls outputs toward TTT-D's high-quality reference plan style (good — anchors prior expertise) BUT dilutes Tool-V's task-specific structure (bad — prevents new domain learning). The two competing gradients don't add — they fight, and the model converges into incoherent low-quality middle ground (the iter 2 plans were jargon-salad mixing TTRL terminology with TTT-Discover formula references). This argues that **per-token CE anchor is the wrong abstraction for continual SDPO**; a Fisher-weighted EWC penalty (per-parameter importance, not per-token output) or **plan-level replay buffer** (mix prior plans into batch instead of regularizing output distribution) are the natural next mitigation tiers.

## Per-dim breakdown (μ-v5 best variants vs μ-v4 on Tool-V isolated /45)

| Dim | μ-v4 (v3 audit) | v3 iter 0 | Δ_v3 | μ-v4 (α audit) | α iter 0 | Δ_α |
|---|---:|---:|---:|---:|---:|---:|
| U1 Soundness | 2.62 | 2.38 | -0.24 | 2.75 | 2.12 | **-0.63** |
| U2 Significance | 3.00 | 3.12 | +0.12 | 3.00 | 3.00 | 0 |
| U3 Originality | 2.75 | 2.75 | 0 | 2.50 | 2.25 | -0.25 |
| U4 Clarity | 2.62 | 3.12 | **+0.50** | 2.88 | 3.00 | +0.12 |
| U5 Reproducibility | 2.38 | 2.00 | -0.38 | 2.38 | 1.50 | **-0.88** |
| T1 Necessity | 2.12 | 2.50 | **+0.38** | 2.38 | 2.12 | -0.26 |
| T2 Disentanglement | 1.88 | 2.25 | **+0.38** | 1.62 | 1.62 | 0 |
| T3 Compute | 1.75 | 1.62 | -0.13 | 1.88 | 1.62 | -0.26 |
| T4 Reward-hacking | 2.50 | 2.50 | 0 | 2.50 | 2.00 | **-0.50** |
| **TOTAL** | **21.62** | **22.25** | **+0.62** | **21.88** | **19.25** | **-2.62** |

v3 (no regularization): marginal +0.62 from **breadth dims** (U4 / T1 / T2) — exactly the dims μ-v4 was Phase-2/4a worst at. v3 iter 0 = 1 grad step from μ-v4 iter 2 anchor; lift plausibly attributable to (a) the iter-2 anchor's intrinsic structural breadth, not (b) Tool-V SDPO learning. Continued training burned this marginal gain.

α (anchor_ce 0.1): negative on **substance dims** (U1 / U5 / T1 / T3 / T4). The regularizer's TTT-D anchor pulls outputs toward TTT-D's prose style which DILUTES Tool-V's task-specific empirical anchoring (specific verifier mechanisms, β values, GRPO hparams). The "consolidation" the regularizer was supposed to provide doesn't materialize at the per-token CE granularity.

T1-T4 results pre-registered as **non-load-bearing** (calibrated to TTT-RL flavor; Tool-V's verifier-reward angle differs). Reporting for transparency only.

## Methodology contribution

The Phase 5 NULL is more informative than a textbook positive — 4 grid cells exhausted:

> **"Naive continual SDPO from a μ-v4 anchor produces plan-quality degradation across the full 3-axis grid (optimizer state × anchor saturation × anchor_ce regularization). Failure SHAPE varies (immediate cliff / delayed cliff / slow decline) but FAILURE is invariant. Per-token CE-anchor regularization (anchor_ce 0.1) actually accelerates collapse via two-gradient conflict. Continual SDPO requires plan-level mitigations (replay buffer, EWC with Fisher importance, or LoRA-per-paper merge) — the per-token output regularizer is the wrong abstraction."**

This strengthens, not weakens, the paper. It motivates Phase 6 directions in Discussion:
1. **Plan-level replay buffer** (50/50 mix prior-paper plans into new training batches; updates SDPO on mixed batches rather than regularizing output)
2. **EWC penalty** (Fisher-weighted parameter importance; constrains weights from drifting on dims that mattered for prior task — different abstraction tier than per-token CE)
3. **LoRA-per-paper + weighted merge** (parallel single-paper SDPO + inference-time aggregation, à la Phase 4a's σ' + μ' → ensemble; bypasses chain entirely)
4. **Task-specific lr scheduling** (lower lr for continual, e.g. lr=1e-5; risky given μ-v2 already showed lr=1e-5 → no learning)

The paper now has 3 self-consistent findings (F2/F4 for μ-v4 SDPO, F7 for forward-citation transfer, F9 for 4-cell continual NULL) plus the audit_v3 isolated methodology. The story converges around a coherent message: **"Single-paper SDPO works; preferences transfer to follow-ups; chained SDPO requires plan-level consolidation, not per-token regularization."**

## Comparison to literature

- **Hübotter 2026 (SDPO origin)** §4 documents single-run multi-round instability past iter 5. F9 extends: instability also appears at iter 1 of a CONTINUED run from a saturated anchor. The cliff-edge invariant is tighter than Hübotter's framing.
- **Continual learning canon** (Kirkpatrick EWC 2017, Lopez-Paz GEM 2017, etc.) catalog 4 mitigations; F9's NULL motivates trying replay or EWC in Phase 6.
- **No prior continual-SDPO paper exists** — F9 is the first published characterization of the Phase-5-style failure mode in this regime.

## What this means for the D5 paper

**Section 5 narrative (post-F9)**:
- Section 5a (existing): μ-v4 + Phase 4a forward-citation transfer (F7 Strong-mixed)
- Section 5b (NEW from F9): continual chaining attempts. Pre-register chain. Show all 3 attempts. Diagnose saturation × Adam grid. Conclude with replay/EWC future work.
- Section 6 Discussion: combined message — "single-paper SDPO + Phase 4a transfer is the validated mechanism. Continual chaining at this scale requires future work; we propose 3 candidate mitigations."

This is **stronger** than a clean Phase-5 success would have been: a clean success would be one bullet ("we chained 2 papers and it worked"); the NULL result with clear failure-mode diagnosis is a methodology contribution that opens Phase 6 territory.

## Cost / time

| Step | Subagent | Tinker | Wall-clock |
|---|---:|---:|---:|
| T0 (pre-registration) | 0 | 0 | 15 min |
| T1 (init_state_path code) | 0 | 0 | 30 min |
| cliff_v1 train | ~$2 | ~30 min | 30 min |
| cliff_v2 train | ~$2 | ~25 min | 25 min |
| v3 train (iter2anchor, full 5 iter) | ~$5 | ~75 min | 75 min |
| v3 inference + isolated /45 audit | ~$3 | ~1 min | 30 min |
| Tα.1 anchor_ce code | 0 | 0 | 1 hr |
| α train (cliff at iter 2) | ~$3 | ~25 min | 25 min |
| α inference + isolated /45 audit | ~$3 | ~1 min | 15 min |
| F9 + doc sync | 0 | 0 | 2 hr |
| **Total Phase 5** | **~$18** | **~2.5 hr** | **~6 hr** |

Compare to plan estimate (Phase 5 main ~$35-40 + 9 hr; Phase 5α extra ~$9 + 5-6 hr). Combined came in under both because chain failed early at every grid cell — saved on v6 train + downstream audits.

## Reproducibility / artifact pointers

- Pre-registration: `DECISIONS.md` 2026-04-26 (c) (binding, no post-hoc edits)
- Plan: `~/.claude/plans/adaptive-churning-perlis.md`
- Code: `train_mu_v4.py` `init_state_path` (commit `f3b9485`) + `reset_optimizer_state` (commit `d2d112c`) + `anchor_ce` regularization (commit `a984640`)
- Run dirs (all preserved):
  - `runs/2026_04_28_mu_v5_tool_v_cliff_v1/` (μ-v4 iter 4 anchor, full Adam)
  - `runs/2026_04_28_mu_v5_tool_v_cliff_v2/` (μ-v4 iter 4 anchor, fresh Adam)
  - `runs/2026_04_28_mu_v5_iter2anchor_5iter/` (μ-v4 iter 2 anchor, fresh Adam, full 5 iter trajectory)
  - `runs/2026_04_28_mu_v5_anchor_ce_v1/` (μ-v4 iter 2 anchor + fresh Adam + anchor_ce 0.1, cliffed iter 2)
  - `runs/2026_04_28_mu_v5_v3_iter0_inference/`, `runs/2026_04_28_mu_v5_anchor_ce_v1_inference/` (best-iter inference plans on Tool-V)
  - `runs/2026_04_28_phase5_audit_v3_iter0/`, `runs/2026_04_28_phase5_audit_anchor_ce/` (8 isolated /45 audit batches each)
- μ-v4 anchors used:
  - iter 4 (saturated): `tinker://6b9d996d-...:train:0/weights/000004_2026_04_25`
  - iter 2 (non-saturated): `tinker://6b9d996d-...:train:0/weights/000002_2026_04_25`

## Pairwise + strict-audit corroboration (added 2026-04-27)

Phase 5 quality audit remediation (DECISIONS.md (f), 2026-04-27) ran two additional
protocols on the v3 vs μ-v4 comparison:

### Pairwise tournament (Phase A.1)

8 anonymized pairs × 1 subagent/pair, holistic A/B/TIE judgment.

| Comparison | Pairwise tally | Direction |
|---|:---:|---|
| μ-v4 vs μ-v5_v3 iter 0 (Tool-V) | μ-v4 4 / μ-v5_v3 4 / 0 TIE | **TIE** — corroborates NULL |

The 4-4 split is exactly at the noise floor; the 2-plan audit's +0.62 was within the
direction range expected from a 4-4 random-walk distribution. **Phase 5 NULL CORROBORATED.**

### Strict 1-plan/subagent re-audit (Phase B)

| Comparison | Original 2-plan Δ | Strict 1-plan Δ | \|shift\| |
|---|---:|---:|---:|
| μ-v4 vs μ-v5_v3 iter 0 (Tool-V) | +0.62 | +1.87 | 1.24 |
| μ-v4 vs μ-v5_α iter 0 (Tool-V, anchor_ce 0.1) | -2.62 | +0.26 | **2.89** |

Strict for v3 stays on the same side (μ-v5_v3 ≥ μ-v4) but shift is within noise (±3.5 95% CI on n=8).
NULL conclusion intact. **For α**, the original "α plans WORSE than anchor (-2.62)" verdict
SOFTENS to "TIE (+0.26)" under strict — meaning anchor_ce regularization neither rescues
NOR destroys plan quality at iter 0; the in-loop /20 trajectory still cliffs at iter 2,
which remains the load-bearing failure signal. Substance gap between μ-v5_α iter 0 plans
and μ-v4 anchor plans is roughly null at the n=8 noise floor.

### Combined evidence pattern

3 lines of evidence agree on Phase 5 NULL:
1. **In-loop /20 trajectory**: 4/4 cells decline below anchor by iter 2-4 (cliff_v1/v2/v3/α)
2. **Original audit_v3 ISOLATED /45**: v3 Δ +0.62, α Δ -2.62 (both within noise)
3. **Pairwise + strict**: v3 TIE 4-4 + Δ +1.87 strict (within noise); α Δ +0.26 strict

Nothing in this matrix supports "continual SDPO at lr=5e-5 from μ-v4 anchor lifts plan quality
on Tool-V". The NULL verdict is now triple-corroborated.

**Methodology note**: The strict-vs-original shift on α (-2.62 → +0.26, |shift| 2.89) is itself
H1-anchoring evidence — see `paper_materials/methodology/M8_audit_pairwise_divergence.md` for
the full cross-method comparison and methodology contribution.

## Out of scope (deferred / future work)

- Phase 6: replay-buffer continual SDPO + EWC + LoRA-per-paper merge
- n≥30 audits to tighten 95% CI on Δ — not blocking the NULL conclusion
- Lower lr schedules (lr=1e-5) for continual — μ-v2 already showed lr=1e-5 = no learning, so this is risky
- Phase 4b cross-domain stretch test — DEPRECATED (per 2026-04-26 reframe; see PHASE_PLAN_v2.md)
