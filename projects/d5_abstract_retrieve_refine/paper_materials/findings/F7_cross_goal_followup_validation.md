# F7 — Cross-goal follow-up validation: μ-v4 LoRA on 3 forward-citation papers

## Headline (FINAL 2026-04-28 after n=24 stress test)

**FINAL VERDICT: μ-v4 LoRA does NOT transfer to forward-citation follow-up papers.**
At n=24 (3× the original n=8), strict 1-plan/subagent + pairwise tournament both
confirm: meta_ttl shows σ' edge (Δ -0.96 strict + pairwise σ' 14/9/1 ≥ 14/24 threshold);
tool_verification_ttrl shows true null (Δ -0.58 strict + pairwise σ' 12/10/2 = 50% — within
12-12 ± 2 tie band). The original n=8 audit's "Strong-mixed 2/3 pass" verdict was a 2-plan/
subagent anchoring artifact (M8). The corroborated finding is **negative transfer**:
μ-v4 LoRA + slim oracle does not generalize to follow-up papers in the same TTT-family
subfield. The surviving paper contributions are (a) M8 methodology (audit-pairwise
divergence), and (b) F7 itself as a NEGATIVE result properly characterized.

**Verdict history**:
- 2026-04-26 (original n=8 2-plan audit): "Strong-mixed (2/3 pass)" with Δ +2.13 / +3.00 / -0.88
- 2026-04-27 (Phase 5 quality remediation): pairwise + strict at n=8 → "all 3 inconclusive within noise"
- 2026-04-28 (D1 n=24 stress test): **definitive null on 2 borderline goals** (meta_ttl + tool_v_ttrl)

Per-goal results match Phase 2's per-dim signature: μ' wins on **U5 Reproducibility
(+0.79 avg)**, **T3 Compute (+0.46 avg)**, **T4 Reward-hacking (+0.33 avg)** — the
three dims μ-v4 was empirically trained to lift. This corroborates that the +2.75
on TTT-Discover was not pure overfit and DOES generalize to follow-up work.

```
Audit /45                         tool_v_ttrl μ'   23.38
   24 ┤                                              ●
      │                                                   meta_ttl μ'   19.88
   20 ┤   tt_control σ'   20.00   tool_v_ttrl σ' 20.38   ──┘
      │   ↘ ──── ── ── ── ──         ●                    ●
      │   tt_control μ'   19.12                          meta_ttl σ'   17.75
   18 ┤                                                  ──┘
      │                                                       ●
```

## Pre-registered protocol

Locked 2026-04-26 in `DECISIONS.md` (entry "Phase 4a cross-goal validation — 3 picks
pre-registered") **before any inference run**:

- **Selection rule**: top-3 TTT-family forward citations by qualitative judgement on
  diversity (algorithmic/engineering/verification angles). Drawn from S2's 32 incoming
  citations of arxiv 2601.16175.
- **Locked picks** (no post-hoc substitution): A=Meta-TTL (Lou 2604.00830), C=TT-Control
  (Wang 2603.09221), D=Tool-Verification-for-TT-RL (Liao 2603.02203).
- **Pass / fail criteria**:
  - **Pass**: μ' > σ' + 1.0 on **all 3** goals AND aggregate Δ > 0.5 → "LoRA adds value on follow-up papers"
  - **Strong-mixed**: 2/3 pass → "Mostly goal-stable; document failing goal + per-dim diagnostic"
  - **Weak-mixed**: 1/3 pass → "Goal-dependent transfer"
  - **Fail**: 0/3 pass → "No transfer"
- **Audit methodology**: per-goal 8 balanced batches × 2 plans (1σ' + 1μ', anonymized A/B),
  8 parallel Opus subagents per goal, 9-dim isolated rubric (5 universal + 4 TTT-subfield).

## Per-goal results

### Meta-TTL (arxiv 2604.00830, Lou 2026-04-01)

Goal: agentic cross-episode improvement under frozen-actor + prompt-space-only constraint.

| Baseline | N | Mean /45 | Std | Min | Max | 9-dim distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| σ' | 8 | 17.75 | 1.30 | 16 | 20 | 5/8 |
| **μ'** | 8 | **19.88** | 3.92 | 13 | 27 | 5/8 |

**Δ = +2.12** (n=8 each, bootstrap 95% CI [-0.75, +4.88]). **Pass.**

Per-dim Δ pattern: U5 +1.00, T3 +0.76, T4 +0.50, T1 +0.25, U1 +0.12, U3 0, U4 0, U2 -0.38, T2 -0.13.

### TT-Control (arxiv 2603.09221, Wang 2026-03-10)

Goal: differentiable optimal-control adapter for native multi-step planning during the LLM forward pass.

| Baseline | N | Mean /45 | Std | Min | Max | 9-dim distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| σ' | 8 | 20.00 | 1.41 | 17 | 22 | 5/8 |
| μ' | 8 | 19.12 | 5.30 | 9 | 26 | 8/8 |

**Δ = -0.88** (bootstrap 95% CI [-4.88, +2.75]). **Fail.**

Per-dim Δ: T4 +0.12, T1 +0.12, U5 +0.38, U1 0, U3 0, T3 0, U2 -0.25, T2 -0.50, U4 -0.75.

**Failure diagnosis**: μ' has **wide variance** (std 5.30 vs σ' std 1.41). The min=9 plan is the
sample where μ-v4 hit max_tokens=4096 before closing the `<solution>` tag (idx=6 in
`runs/2026_04_28_xgoal_tt_control_mu/buffer.jsonl`); extract_solution recovered only `...`,
audit understandably scored it as 9. Excluding that single truncated outlier, μ' (n=7) mean = 20.57,
giving Δ = +0.57 — below the +1.0 threshold but positive. Median-based comparison: μ' median 22.5 vs
σ' median 20.5 → **median Δ = +2.0** (matches the other 2 goals' positive direction). The
goal-specific failure pattern: TT-Control's research goal asks for an architectural mechanism
(differentiable LQR layer) that the oracle's TTT-RL-flavored insights don't squarely cover — μ-v4's
verbosity sometimes drifts into RL-jargon-salad and gets dinged on U4 Clarity.

### Tool-Verification-for-TT-RL (arxiv 2603.02203, Liao 2026-03-02)

Goal: external verification breaking TTRL self-consensus collapse without weight-update or labels.

| Baseline | N | Mean /45 | Std | Min | Max | 9-dim distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| σ' | 8 | 20.38 | 1.87 | 18 | 23 | 5/8 |
| **μ'** | 8 | **23.38** | 4.27 | 14 | 28 | 6/8 |

**Δ = +3.00** (bootstrap 95% CI [-0.38, +5.88]). **Pass — strongest.**

Per-dim Δ: U5 +1.00, U1 +0.62, T3 +0.62, U3 +0.38, T4 +0.38, T1 +0.25, U2 0, U4 0, T2 -0.25.

## Aggregate

Aggregate over all 24 (μ', σ') pairs (3 goals × 8 plans each):

| Pool | N | Mean /45 | Std |
|---|---:|---:|---:|
| σ' (all) | 24 | 19.38 | 1.85 |
| μ' (all) | 24 | 20.79 | 4.62 |

**Aggregate Δ = +1.42** (bootstrap 95% CI [-0.75, +3.50]). **Above the +0.5 threshold ✅**.

## Per-dim transfer signature

Average per-dim Δ across the 3 goals:

| Dim | meta_ttl | tt_control | tool_v_ttrl | **Avg** | Phase 2 sign |
|---|---:|---:|---:|---:|---|
| U1 Soundness | +0.12 | 0 | +0.62 | **+0.25** | + (Phase 2 +0.3) |
| U2 Significance | -0.38 | -0.25 | 0 | -0.21 | + (Phase 2 +0.4) |
| U3 Originality | 0 | 0 | +0.38 | +0.13 | + (Phase 2 +0.4) |
| U4 Clarity | 0 | -0.75 | 0 | -0.25 | − (Phase 2 -0.1) |
| **U5 Reproducibility** | **+1.00** | +0.38 | **+1.00** | **+0.79** | **++ (Phase 2 +0.7)** |
| T1 Necessity | +0.25 | +0.12 | +0.25 | +0.21 | + (Phase 2 +0.3) |
| T2 Disentanglement | -0.13 | -0.50 | -0.25 | -0.29 | + (Phase 2 +0.4) |
| **T3 Compute** | **+0.76** | 0 | **+0.62** | **+0.46** | + (Phase 2 +0.3) |
| **T4 Reward-hacking** | **+0.50** | +0.12 | **+0.38** | **+0.33** | **+ (Phase 2 +0.5)** |

The **three biggest cross-goal lifts (U5 / T3 / T4)** match Phase 2's three biggest. The
**three regressions (U2, U4, T2)** all relate to "structural breadth" — coverage of alternative
hypotheses + clarity. μ-v4 internalized concrete grounding (cited numbers, hparams, compute
units, reward-hacking pathways) but at the cost of slight breadth in the more open-ended
Significance/Disentanglement axes — mostly because μ-v4 plans are longer and dive deeper on
fewer threads.

## Implications for paper framing

**What this finding supports**:
1. **Cross-goal generalization claim is EVIDENCE-BASED, not pure overfit**: 2/3 forward-citation
   follow-ups show positive lift, with the per-dim transfer pattern matching Phase 2.
2. **U5 + T3 + T4 are the load-bearing dims** for μ-v4's value: reproducibility, compute
   accounting, and reward-hacking awareness consistently transfer.
3. **The pass criterion was strict on purpose**; "Strong-mixed" is a credible result that is
   honestly characterized rather than overclaimed.

**What this finding qualifies**:
1. **n=8 per condition is too small for 95% CI to clear 0** on any single goal. This is a
   single-run point estimate study, not a stat-significance study. Future work: n≥30 per goal.
2. **TT-Control's failure is informative**, not a mere outlier: μ-v4's training doesn't help
   when the goal asks for an architectural mechanism the oracle doesn't directly cover. This
   is a genuine OOD failure, not just sampling noise — Phase 4b oracle-rebuild stretch test
   would address it.
3. **The "follow-up" frame is near-OOD**, NOT cross-domain. Paper should label honestly.

**Paper-claim language (recommended)**:
> "We validate μ-v4's transfer on three TTT-Discover forward-citation follow-up papers
> (post-2026-02). On 2 of 3 goals, μ' beats σ' by ≥+1.0 / 45 (mean), with aggregate Δ +1.42
> across all 24 plan pairs. The transfer signature matches Phase 2's: U5 Reproducibility
> (+0.79), T3 Compute (+0.46), T4 Reward-hacking (+0.33) consistently lift; structural-breadth
> dims (U2, U4, T2) regress slightly, reflecting μ-v4's depth-over-breadth bias. The single
> failing goal (TT-Control) needs an oracle that covers differentiable-architecture insights —
> a Phase 4b oracle-rebuild experiment, deferred."

## Cost / time

| Phase | Subagents | Cost (Claude subagent compute) | Wall-clock |
|---|---:|---:|---:|
| Goal + ref plan extraction (T4) | 8 (6 + 2 re-spawn for 7-section fix) | ~$5-6 | 3 min |
| Smoke test on TTT-Discover (T5) | 0 (qualitative only) | $0 | 5 min |
| Cross-goal inference (T6) | 0 (Tinker only) | $0 | 7 min |
| Audits (T7) | 25 (24 + 1 re-spawn for malformed JSON) | ~$8-10 | 12 min |
| **Total** | **33** | **~$13-16** | **~30 min wall-clock** |

(All Opus calls via Claude `Agent(general-purpose, model=opus)`; **0 OpenRouter API calls** per
`feedback_opus_eval_subagent.md`.)

## Reproducibility / artifact pointers

- Pre-registration: `projects/d5_abstract_retrieve_refine/DECISIONS.md` (2026-04-26 entry)
- Inference harness: `src/co_scientist/d5_abstract_retrieve_refine/baseline_frozen_v2.py`
  (variant=mu_inference, sampler_path config field added in commit `8dbf837`)
- Audit pipeline: `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated_2way.py`
- Shared subagent prompt: `runs/2026_04_28_xgoal_audit_subagent_prompt.md`
- Per-goal raw plans: `runs/2026_04_28_xgoal_{name}_{sigma|mu}/buffer.jsonl`
- Per-goal audit responses: `runs/2026_04_28_xgoal_{name}_audit/audit_responses/batch_*.json`
- Per-goal audit summaries: `runs/2026_04_28_xgoal_{name}_audit/audit_v3_isolated_2way_summary.md`
- μ-v4 production checkpoint: `runs/2026_04_27_mu_v4/checkpoints.jsonl` row `iter=4`
  (sampler_path `tinker://6b9d996d-...iter_0004`)
- Slim oracle (reused verbatim, no rebuild): `data/oracles/oracle_v2_2026_04_26_build/slim.md`

## Pairwise + strict-audit corroboration (added 2026-04-27)

Phase 5 quality audit remediation (DECISIONS.md (f) entry, pre-registered 2026-04-27)
ran two corroborating protocols on this Phase 4a data:

### Pairwise tournament (Phase A.3)

8 anonymized pairs/goal × 1 subagent/pair, holistic A/B/TIE judgment (no rubric).

| Goal | Original Δ | Pairwise tally | Direction match? |
|---|---:|:---:|:---:|
| meta_ttl | +2.13 (μ' wins) | σ' 5 / μ' 3 | **REVERSED** (σ' actually wins) |
| tt_control | -0.88 (σ' edge) | σ' 5 / μ' 3 | matches σ' direction |
| tool_v_ttrl | +3.00 (μ' wins big) | μ' 4 / σ' 3 / 1 TIE | weak μ' but within noise; not corroborating "wins big" |

**None of 3 goals reaches the pairwise ≥ 6/8 threshold for directional confirmation.**
The original "Strong-mixed 2/3 pass" verdict is NOT corroborated.

### Strict 1-plan/subagent re-audit (Phase B)

16 batches × 1 plan/subagent, no within-batch anchoring. Tests H1 (balanced batching anchoring)
as the root cause of audit-pairwise divergence.

| Goal | Original Δ (2-plan) | Strict Δ (1-plan) | \|shift\| |
|---|---:|---:|---:|
| meta_ttl | +2.13 | -0.12 | 2.25 (FLIP) |
| tt_control | -0.88 | -0.63 | 0.25 (stable) |
| tool_v_ttrl | +3.00 | -0.63 | 3.63 (FLIP) |

**H1 CONFIRMED on meta_ttl + tool_v_ttrl**: 2-plan batching inflated Δ by 2-3 points.
Under strict 1-plan calibration the gap collapses to ~0 — matching pairwise direction
on meta_ttl, within-noise of pairwise on tool_v_ttrl.

### Revised per-goal verdict (after Phase 5 remediation)

| Goal | Original | Revised | Mechanism |
|---|---|---|---|
| meta_ttl | μ' wins by +2.13 | TIE / σ' edge | 2-plan anchoring inflated μ' |
| tt_control | σ' wins by -0.88 | σ' edge stable | 3-protocol convergence |
| tool_v_ttrl | μ' wins big +3.00 | TIE | 2-plan anchoring inflated μ' |

**F7's load-bearing claim** ("forward-citation transfer demonstrated, generalizes Phase 2
result") is **NOT supported by the corroborated data**. The methodology lesson — that
2-plan/subagent batching is unreliable for close-cluster comparisons — IS the surviving
contribution and is captured in `paper_materials/methodology/M8_audit_pairwise_divergence.md`.

## D1 n=24 stress test (added 2026-04-28)

Per DECISIONS.md (g) D-series pre-registration, n=24 inference + audits run on the 2
goals where pairwise reversed or anchoring was largest (meta_ttl + tool_v_ttrl).
n=24 collapses the 95% bootstrap CI on Δ /45 from ~±3.5 (n=8) to ~±2.0, enough to
discriminate "true null Δ ∈ [-1, +1]" from "small lift Δ ≥ +1".

### Strict 1-plan/subagent re-audit at n=24

| Goal | Strict Δ at n=8 | Strict Δ at n=24 | σ' n=24 mean | μ' n=24 mean |
|---|---:|---:|---:|---:|
| meta_ttl | -0.12 | **-0.96** | 15.83 | 14.88 |
| tool_v_ttrl | -0.63 | **-0.58** | 19.29 | 18.71 |

Both goals' strict Δ stays in the σ'-edge / TIE band as n goes from 8 to 24. **No μ' lift
emerges with more data** — rejecting the "underpowered noise" interpretation.

### Pairwise tournament at n=24

| Goal | σ' wins | μ' wins | TIE | A-pos rate | Verdict |
|---|---:|---:|---:|---:|---|
| **meta_ttl_n24** | 14 | 9 | 1 | 52% | **σ' direction CONFIRMED** (≥14/24 pre-reg) |
| **tool_v_ttrl_n24** | 12 | 10 | 2 | 55% | **TRUE NULL** (12-12 ± 2 pre-reg) |

A-position rates 52% / 55% — both within [25%, 75%] bounds, no position bias.

### 3-protocol agreement at n=24

| Protocol | meta_ttl | tool_v_ttrl |
|---|---|---|
| Original 2-plan (n=8) | μ' +2.13 | μ' +3.00 |
| Strict 1-plan (n=24) | σ' +0.96 | σ' +0.58 |
| Pairwise (n=24) | σ' 14/9/1 | σ' 12/10/2 (= TIE) |

**Original 2-plan said μ' wins; strict + pairwise BOTH say σ' edge / TIE at n=24.** The
direction reverses on meta_ttl and collapses to TIE on tool_v_ttrl — confirming that
the original audit was anchoring-inflated AND that there is no real transfer to find at
larger sample sizes.

### Final F7 verdict

**Forward-citation transfer is DEFINITIVELY ABSENT** for meta_ttl + tool_v_ttrl at n=24.
The original "Strong-mixed 2/3 pass" claim is fully retracted. Combined with the original
tt_control σ' edge (-0.88, stable across protocols), the 3-goal narrative becomes:
"applied μ-v4 LoRA to 3 forward-citation papers; observed null transfer on all 3 at n=24
or after pairwise corroboration; methodology contribution (M8) emerges as the durable result."

## What's next

1. **F7 reframes around M8 methodology contribution**: paper draft (Phase 7) writes Phase 4a
   as "we attempted forward-citation transfer; the audit said yes (+1.42); pairwise + strict
   said no; the methodology divergence itself is the contribution".
2. **Larger n (deferred)**: re-run Phase 4a with n=24 plans + strict isolation + pairwise
   corroboration on every audit. Even with 3× n, current evidence suggests gaps are within noise.
3. **Phase 4b stretch test**: deprecated 2026-04-26 (cross-domain stretch breaks oracle assumption).
4. **Phase 7 paper draft**: F2 + F4 + F9 + M8 are the validated 4-finding cohesive narrative.
   F7 becomes "negative+methodology" — it's still useful but no longer carries a "transfer works"
   claim.

## Reproducibility / artifact pointers (revised set)

Original 2-plan audit (2026-04-26):
- Inference: `runs/2026_04_28_xgoal_{meta_ttl,tt_control,tool_verification_ttrl}_{sigma,mu}/buffer.jsonl`
- Audit: `runs/2026_04_28_xgoal_{name}_audit/audit_v3_isolated_2way_summary.md`

Pairwise corroboration (Phase A.3, 2026-04-27):
- `runs/2026_04_28_phase5_remediation_pairwise/pairwise_tournament_summary.md`
- 32 subagents (4 matchups × 8 pairs); pair_id namespaced per matchup

Strict 1-plan corroboration (Phase B, 2026-04-27):
- `runs/2026_04_28_xgoal_{name}_audit_strict/audit_v3_isolated_strict_summary.md`
- 80 subagents (5 audits × 16 plans), 8 σ' + 8 μ' interleaved per goal

Code:
- `src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated_2way_strict.py` (Phase B fork)
- `src/co_scientist/d5_abstract_retrieve_refine/phase5_remediation_pairwise.py` (Phase A)
- Methodology doc: `paper_materials/methodology/M8_audit_pairwise_divergence.md`
