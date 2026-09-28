# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-27 00:52:03*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **24.50 / 45**
- **Std**: 2.18
- **Min / Max**: 21 / 28
- **Distinct totals**: 6/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 26 |
| tau_1 | 21 |
| tau_2 | 28 |
| tau_3 | 24 |
| tau_4 | 23 |
| tau_5 | 27 |
| tau_6 | 24 |
| tau_7 | 23 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.38 |
| U2_significance | 3.12 |
| U3_originality | 2.00 |
| U4_clarity | 3.00 |
| U5_reproducibility | 2.75 |
| T1_necessity | 3.12 |
| T2_disentanglement | 3.00 |
| T3_compute | 3.12 |
| T4_reward_hacking | 2.00 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 24.50, gap = -0.75
- Verdict: **NEUTRAL** — pipeline ≈ slim. Proceed κ training, but mandate pairwise verification.

