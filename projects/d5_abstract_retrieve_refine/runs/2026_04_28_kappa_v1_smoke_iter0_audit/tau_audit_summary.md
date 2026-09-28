# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 18:34:07*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **24.88 / 45**
- **Std**: 1.17
- **Min / Max**: 23 / 27
- **Distinct totals**: 5/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 24 |
| tau_1 | 26 |
| tau_2 | 25 |
| tau_3 | 25 |
| tau_4 | 24 |
| tau_5 | 27 |
| tau_6 | 23 |
| tau_7 | 25 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.50 |
| U2_significance | 3.00 |
| U3_originality | 2.00 |
| U4_clarity | 3.00 |
| U5_reproducibility | 3.00 |
| T1_necessity | 3.38 |
| T2_disentanglement | 3.00 |
| T3_compute | 3.00 |
| T4_reward_hacking | 2.00 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 24.88, gap = -0.38
- Verdict: **NEUTRAL** — pipeline ≈ slim. Proceed κ training, but mandate pairwise verification.

