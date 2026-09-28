# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 17:49:18*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **24.75 / 45**
- **Std**: 1.98
- **Min / Max**: 22 / 28
- **Distinct totals**: 4/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 24 |
| tau_1 | 22 |
| tau_2 | 26 |
| tau_3 | 24 |
| tau_4 | 26 |
| tau_5 | 22 |
| tau_6 | 26 |
| tau_7 | 28 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.50 |
| U2_significance | 3.00 |
| U3_originality | 2.00 |
| U4_clarity | 2.75 |
| U5_reproducibility | 2.25 |
| T1_necessity | 3.50 |
| T2_disentanglement | 3.25 |
| T3_compute | 3.12 |
| T4_reward_hacking | 2.38 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 24.75, gap = -0.50
- Verdict: **NEUTRAL** — pipeline ≈ slim. Proceed κ training, but mandate pairwise verification.

