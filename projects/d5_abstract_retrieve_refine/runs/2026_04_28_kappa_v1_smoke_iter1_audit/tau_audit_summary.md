# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 18:34:09*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **24.38 / 45**
- **Std**: 1.32
- **Min / Max**: 22 / 26
- **Distinct totals**: 5/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 22 |
| tau_1 | 26 |
| tau_2 | 26 |
| tau_3 | 24 |
| tau_4 | 25 |
| tau_5 | 23 |
| tau_6 | 25 |
| tau_7 | 24 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.25 |
| U2_significance | 3.00 |
| U3_originality | 1.88 |
| U4_clarity | 3.00 |
| U5_reproducibility | 2.75 |
| T1_necessity | 3.12 |
| T2_disentanglement | 3.25 |
| T3_compute | 3.00 |
| T4_reward_hacking | 2.12 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 24.38, gap = -0.88
- Verdict: **NEUTRAL** — pipeline ≈ slim. Proceed κ training, but mandate pairwise verification.

