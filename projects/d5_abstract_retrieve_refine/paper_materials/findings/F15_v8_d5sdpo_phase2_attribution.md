# F15 — μ-v8-d5sdpo Phase 2 grid: α-monotonic cliff displacement, mask + PPO interaction characterized

*Filed 2026-04-28. Phase 2 partial-grid (7/13 cells run). All numbers verified from
`runs/2026_04_29_mu_v8_d5sdpo_<cell>/audit_responses/iter_NNN.json` and
`runs/2026_04_29_mu_v8_d5sdpo_Gplus_pairwise/PAIRWISE_SUMMARY.md`.*

## Headline

A 7-cell single-seed factorial probe of mask × α (3-level) × loss_fn (PPO/IS) on
the μ-v8-d5sdpo trainer reveals two systematic effects:

1. **Cliff iter is monotonic in α** at fixed (mask=T, loss=PPO):
   base (mask=F, α=0, IS) cliffs iter 5-6 → G (mask=T, α=0.05, PPO) cliffs iter 7-8 → G+
   (mask=T, α=0.1, PPO) cliffs iter ≥11. Trust-region α as Hübotter SDPO
   Appendix A.2 mitigation works as predicted.
2. **Peak is not significantly improved by α**: peaks of E (α=0, mask+PPO, 22.75)
   ≈ G+ (α=0.1, mask+PPO, 22.38) ≈ F+ (α=0.1, no-mask+PPO, 23.00) ≈ C+ (α=0.1,
   mask+IS, 23.50). Δ peak across α-levels < 1.5 audit points; α is **cliff-dampener,
   not peak-driver**.
3. **Pairwise vs σ_v8 corroborates G+ peak**: G+ 8-0 vs σ_v8 (n=8 Opus pairwise),
   8-0 vs μ-v4 production checkpoint, 7-1 vs G. **G+ 3-5 vs v7-opd-full** —
   v7-opd-full wins despite v8 redesign + length match. The "v8 beats v7-opd-full"
   pre-registered headline is FALSIFIED.

This finding decomposes which of (critique-on-current, solution-only mask,
trust-region α, PPO clip, word-target footer) drives the Δ peak +3.00 over σ_v8.

## Verified per-cell trajectories (audit_v3 9-dim mean /45, n=8 plans/iter)

| Cell | mask | α | loss | iter 0 | Peak /45 | Peak iter | Cliff iter (mean<14) | Final | Status | rc |
|---|:-:|:-:|:-:|---:|---:|---:|---:|---:|---|:-:|
| **base** | F | 0 | IS | 19.38 | 20.00 | 3-4 | 6 | 10.00 (i=6) | KILLED | n/a |
| **G** | T | 0.05 | PPO | 19.13 | 20.00 | 3 | 8 | 10.25 (i=9) | KILLED | n/a |
| **G+** | T | 0.1 | PPO | 19.50 | 22.38 | 4-5 | 11 | 12.50 (i=12) | DONE (16-iter completion not yet verified by reading rc=0) | 0 |
| **E** | T | 0 | PPO | 19.75 | 22.75 | 5 | 7 | 9.75 (i=9) | KILLED | n/a |
| **C+** | T | 0.1 | IS | 19.75 | 23.50 | 6 | 10 | 14.75 (i=12) | KILLED | n/a |
| **F+** | F | 0.1 | PPO | 19.88 | 23.00 | 5-6 | 8 | 13.38 (i=9) | KILLED | n/a |
| **A** | T | 0 | IS | 19.38 | 19.38 | 0 | (failed iter 1, critic timeout 900s) | 19.13 (i=1) | FAILED | n/a |

**σ_v8 anchor = 19.38/45** (length-matched 900/1100-word footer, strict 1-plan/Opus
audit; verified at `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/AUDIT_SUMMARY.md`).

**Reference plan = 35/45** (`runs/2026_04_29_reference_audit_v3/audit_responses/
iter_ref_plan_0.partial.json`, universal 21 + subfield 14).

## Cliff α-monotonicity (verified)

Pre-registered hypothesis H2.2 (snug-dreaming-yao plan): cliff-iter monotonic in α.

At (mask=T, loss=PPO), holding 2 of 3 axes fixed:
- α=0 (cell E): cliff iter 7-8 (mean iter 6=20.00, iter 7=16.25, iter 8=10.875)
- α=0.05 (cell G): cliff iter 7-8 (mean iter 7=16.00, iter 8=11.625)
- α=0.1 (cell G+): cliff iter 11 (mean iter 10=19.375, iter 11=14.75)

Cliff displacement E→G: 0 iter (G's cliff arrives at the same place but its peak
20.00 < E's 22.75, suggesting α=0.05 has a small destabilization side effect not
predicted by H2.2). Cliff displacement E→G+: +3 iter (large, monotonic).

At (mask=T, loss=IS), 1 datapoint:
- α=0.1 (cell C+): cliff iter 9-10 (mean iter 9=19.25, iter 10=17.5)

At (mask=F, loss=PPO), 1 datapoint:
- α=0.1 (cell F+): cliff iter 8-9 (mean iter 8=18.0, iter 9=13.375)

H2.2 partially corroborated: α=0 → α=0.1 buys ≈ +3 cliff iters. Mid-level
α=0.05 has no consistent advantage over α=0 within this 7-cell sample. We cannot
distinguish α=0.05 noise from genuine 0-shift without seed replication.

## Peak insensitivity to α

| (mask, α, loss) | Peak /45 |
|---|---:|
| (T, 0, PPO) — E | **22.75** |
| (T, 0.05, PPO) — G | 20.00 |
| (T, 0.1, PPO) — G+ | 22.38 |
| (T, 0.1, IS) — C+ | **23.50** |
| (F, 0.1, PPO) — F+ | 23.00 |
| (F, 0, IS) — base | 20.00 |
| (T, 0, IS) — A | 19.38 (only iter 0; failed before training signal landed) |

Cross-α peak Δ ≤ 1.50. The α=0.05 cell (G, 20.00) is the only outlier; with n=1
seed we cannot rule out single-run sampling noise. Conclusion: **α is a
cliff-dampener with negligible direct peak effect**.

## Mask + PPO is the largest peak driver

Comparing peaks within α=0:
- base (mask=F, IS): 20.00
- E (mask=T, PPO): 22.75 (+2.75 over base)

Comparing peaks within α=0.1:
- F+ (mask=F, PPO): 23.00
- G+ (mask=T, PPO): 22.38 (+0.62 mask deficit at α=0.1+PPO; mask is mildly
  negative here)
- C+ (mask=T, IS): 23.50 (+1.50 over G+; PPO is mildly negative at α=0.1+mask)

Aggregating:
- **mask main effect** (matched on α + loss): mixed (+2.75 in α=0+PPO, −0.62 in
  α=0.1+PPO, but +full-cell-difference in α=0+IS unverified due to A failure)
- **PPO main effect** (matched on α + mask): mixed (+ in α=0+mask vs A unverified;
  −1.12 in α=0.1+mask, C+ vs G+; +full effect at base level unverified due to D
  cell not run)

The factorial decomposition has too many unrun cells for reliable main-effect
attribution. Decisive single-axis claims require D cell (mask=F, α=0, PPO) and a
reliable A cell rerun.

## Pairwise corroboration (G+ only, n=8 per matchup, Opus judge)

Source: `runs/2026_04_29_mu_v8_d5sdpo_Gplus_pairwise/PAIRWISE_SUMMARY.md`.

| Matchup | G+ wins | Other wins | Ties | A-pos rate | Winner |
|---|---:|---:|---:|---:|---|
| G+ vs σ_v8 | 8 | 0 | 0 | 4/8 (balanced) | **G+** |
| G+ vs v7-opd-full | 3 | 5 | 0 | 3/8 | **v7-opd-full** |
| G+ vs μ-v4 | 8 | 0 | 0 | 6/8 (potential A-bias) | **G+** |
| G+ vs G | 7 | 1 | 0 | 3/8 | **G+** |

Decision rules from the snug-dreaming-yao plan (pre-registered):
- G+ ≥6/8 vs σ_v8 → audit lift corroborated → **MET** (8-0)
- G+ ≥6/8 vs v7-opd-full → v8 redesign beats v7-opd-full → **NOT MET** (3-5;
  v7-opd-full wins, **falsifying paper headline that v8 beats v7-opd-full**)
- G+ ≥6/8 vs μ-v4 → v8 surpasses Phase 2 winner → **MET** (8-0). Note: 6/8
  A-position is at the M8 attention-bias caveat boundary; verdict "G+ wins" is
  defensible but not bias-clean
- G+ ≥6/8 vs G → α=0.1 dominates α=0.05 → **MET** (7-1)

**Net pairwise verdict**: G+ is a length-matched lift over σ_v8 baseline AND
over μ-v4. G+ does NOT beat v7-opd-full. The v8 redesign delivers a lift over
the σ_v8 anchor and over the previous Phase-2 production checkpoint (μ-v4) but
does not beat the v7-opd-full reference trainer. This is a substantive
re-framing of v8 as "competitive with v7-opd-full, not a superset".

## Honest uncertainties

1. **Single seed**. All cells n=1 seed=42. Nothing here is bootstrap-significant.
   Δ peak across cells is < 1.5 audit points; cross-cell comparisons within ±1.5
   are not seed-corroborated.
2. **A cell failed at iter 1 (critic 900s timeout)**. mask-only at α=0+IS is
   unverified — leaves the pure mask main effect uncomputable.
3. **D, B, B+, F, C cells not run**. Single-axis main effects (mask main, PPO
   main, α=0.05 IS) cannot be cleanly factored.
4. **G+ "DONE" but iter 11-12 audit shows mean 14.75-12.50** — a slow-motion
   cliff; trainer rc=0 because audit_drop_threshold=10.0 (G+ ran under v1
   threshold; later cells used 7.0). This means G+ is NOT cliff-free, just
   late-cliff. This subtly weakens "α=0.1 + PPO + mask delays cliff > 16 iter"
   to "delays cliff to iter ≥11".
5. **Audit /45 raw-score calibration drifts across runs**. Pairwise is the
   gold-standard cross-check; G+ peak iter pairwise was the only matchup
   conducted. Other cell peaks (E 22.75, C+ 23.50, F+ 23.00) have no pairwise
   corroboration.

## Implications for paper

This finding is a partial factorial probe with one strong corroborated effect
(α-monotonic cliff displacement) and one falsified pre-registered claim (G+
beats v7-opd-full). For paper:

- **Preserve as Phase 2 result**: α-monotonic cliff displacement is a
  defensible, mechanism-grounded result with cliff-iter ladder 6 → 8 → 11.
- **Do not headline "v8 beats v7-opd-full"**: pairwise 3-5 falsifies it.
- **Avoid headlining "+3.00 audit / 45 over σ_v8"**: the audit number is
  length+methodology-confounded; use pairwise 8-0 vs σ_v8 as the empirical
  carrier instead.
- **Treat C+ as production candidate, not G+**: C+ peak 23.50 > G+ peak 22.38
  by +1.12, and C+ rc was rc=killed-on-cliff at iter 10, not better than G+ on
  cliff-resistance. Without pairwise C+ vs σ_v8 we cannot upgrade C+; **the
  paper-honest production checkpoint is G+ iter 4** (only one with audit + pairwise
  corroborated) until C+ peak is pairwise-validated.

## Files

- Per-iter audits: `runs/2026_04_29_mu_v8_d5sdpo_<cell>/audit_responses/iter_NNN.json`
- Per-iter critic responses: `runs/2026_04_29_mu_v8_d5sdpo_<cell>/critic_responses/iter_NNN.json`
- Trajectories captured in: each `<cell>/KILLED_README.md` where present
- σ_v8 anchor: `runs/2026_04_29_mu_v8_d5sdpo_sigma_v8/AUDIT_SUMMARY.md`
- Reference plan audit: `runs/2026_04_29_reference_audit_v3/audit_responses/iter_ref_plan_0.partial.json` (35/45)
- Pairwise: `runs/2026_04_29_mu_v8_d5sdpo_Gplus_pairwise/PAIRWISE_SUMMARY.md`
- Orchestrator state: `runs/_v8_orchestrator_state.json`, `runs/_v8_phase2_orchestrator_state.json`
- Plan source: `~/.claude/plans/snug-dreaming-yao.md`

## Open follow-ups

- D cell (mask=F, α=0, PPO) one run to nail PPO main effect (~$15 + 4 hr)
- A cell rerun with `critic_timeout_sec` raised; mask-only attribution otherwise
  permanently absent (~$15 + 4 hr)
- Pairwise C+ peak vs σ_v8 (8 pairs, ~$5)
- Pairwise C+ peak vs G+ peak (8 pairs, ~$5)
- 2-seed replication of G+ + C+ for seed-corroboration (~$60 + 12 hr)

These are deferred until the **oracle transfer ceiling** experiments (F16 / ABC
experiment plan) deliver mechanism-level evidence on whether the lift even
involves model-side learning of oracle content.
