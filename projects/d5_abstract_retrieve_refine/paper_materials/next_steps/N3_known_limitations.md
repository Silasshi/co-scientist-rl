# N3 — Known limitations (paper "Limitations" section material)

## L1 — Single-goal evaluation

All Phase 2 results are on one research goal (TTT-Discover). Cross-goal generalization
is untested (see N2 for the planned validation experiment). Paper claim must be scoped
to "on this goal" until cross-goal data exists.

## L2 — Compute accounting (T3 dim) is the universally weakest dim

Across 7 baselines, T3 (compute accounting) has lowest mean per-dim score:
- ε (235B+ref): T3=4.00
- σ (30B+oracle): T3=2.00
- δ (30B+ref): T3=2.50
- α (Opus distill): T3=1.88
- μ-v2: T3=1.75
- β: T3=1.50
- ξ: T3=1.50

Even μ-v4 iter-4 (mean 28.00) only reaches T3=2.1. The model rarely commits to
operational units ($/GPU-hours/wall-clock). This is a structural limitation of plans
without a "give me a budget" prompt — neither σ-style oracle nor SDPO training
addresses this.

## L3 — PUCT formula in EVAL plans: 0/8 across all iters

The flagship content target the critic asked for (PUCT formula `Q + c·P·sqrt`) NEVER
appeared in any μ-v4 iter EVAL plan, despite teacher PICK plans containing it 8/8 at
iter 1. SDPO training improves overall plan quality (+2.75 over σ) but does not
transfer the specific tokens the critic targets. Mechanism is critique-token-
blindness (F4).

## L4 — 4-dim daemon audit unreliable for fine-grained trajectory

The 4-dim D3-canonical audit run by the in-loop daemon shows 7.00 mean across iter
0-4 (anchor saturation, F6). Only 9-dim isolated audit (`audit_v3_isolated.py`)
reveals the per-iter trajectory. Operationally: keep daemon 4-dim for in-loop
collapse detection, but always run 9-dim isolated for paper-grade conclusions.

## L5 — Multi-round SDPO instability ceiling

All three lr regimes tested (1e-5, 5e-5, 2e-4) eventually destabilize. The peak
+2.75 lift is achievable but cannot be extended to higher quality via more iters at
this scale. Reaching higher than +2.75 requires either:
- Different loss form (CR-v7 / SFT-on-revisions; user rejected as "not beautiful enough")
- Different content target (token-level critique with literal target tokens)
- Architecture change (Phase 3 retrieve-then-generate may help by putting content
  in student's context directly)

## L6 — Oracle is Opus-generated (ceiling estimate)

The slim oracle was built by Opus 4.7 reading 86 bibliography papers. A real
production pipeline (Phase 3) would generate oracles via Qwen3-30B itself (or a
smaller model), which would be noisier. σ as currently measured is a ceiling
estimate of what a real-pipeline σ would achieve.

## L7 — N=8 plans per condition; per-iter audit std ~3-4

The +2.75 effect at iter 4 is outside one within-condition standard deviation but
within two. Robustness check: the iter 3-4 plateau (both above σ) provides 16 plans
total above σ, which is more robust than relying on iter 4 alone.

## L8 — No external (non-Opus) auditor

All audit scoring is by Opus 4.7. Cross-auditor validation (e.g. GPT-5 / human)
would strengthen the audit_v3 9-dim methodology contribution. Future work.

## L9 — Tinker-specific implementation

The training pipeline depends on Tinker's `forward_backward` API and
`importance_sampling` loss. Reproducing without Tinker requires re-implementing
both. The recipe is portable in principle but not currently tested on other
training stacks.

## L10 — File-bus subagent latency

Each SDPO iter takes ~3 minutes wall (vs. ~30s if Opus were called via API). This is
acceptable for research but would not scale to large-scale training.

## L11 — Word target 600/750 caps T3/U5 dimensions (added 2026-04-27)

All Phase 2 + Phase 3 prompts use `Target 600 words, max 750 words` (D3 historical
convention; no documented rationale). Reference plan is ~1200 words — model is asked
to write half. Combined with L2 (T3 universally weakest dim), this is a structural
ceiling on T3 (compute accounting) and U5 (reproducibility) — both dimensions that
benefit from operational specifics requiring writing room.

**Why we did not re-baseline Phase 2 at 1200**: mechanism claims (μ-v4 +2.75 over σ,
multi-round instability, critique-token-blindness) are internally valid at fixed
target. Re-baseline cost ~30 hr Tinker + ~$200 Opus + 4-5 days + invalidates Session B
parallel work + rewrites F1-F6 docs. Estimated absolute lift: σ and μ-v4 both rise
in tandem, headline gap likely unchanged.

**What we did instead**: Phase 3 Step 7 includes a τ-1200 ablation (1 extra τ run at
1200/1500 target + 1 audit, ~$5, ~1 hr). If τ_1200 ≫ τ_600 (gap > 1.5), Phase 4
re-baselines all 30B variants at 1200; if τ_1200 ≈ τ_600 (gap < 0.5), 600/750 is not
the bottleneck and existing numbers stand.

See `knowledge/current/RETRIEVAL_DESIGN_v1.md` § "Q10 word-target rationale" for full
ML-scientist analysis.
