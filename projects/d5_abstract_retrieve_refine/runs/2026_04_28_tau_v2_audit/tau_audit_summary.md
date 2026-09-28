# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 16:45:25*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **19.00 / 45**
- **Std**: 1.00
- **Min / Max**: 17 / 20
- **Distinct totals**: 4/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 19 |
| tau_1 | 19 |
| tau_2 | 19 |
| tau_3 | 17 |
| tau_4 | 20 |
| tau_5 | 20 |
| tau_6 | 20 |
| tau_7 | 18 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 1.75 |
| U2_significance | 3.00 |
| U3_originality | 1.88 |
| U4_clarity | 2.75 |
| U5_reproducibility | 1.25 |
| T1_necessity | 2.12 |
| T2_disentanglement | 2.00 |
| T3_compute | 2.00 |
| T4_reward_hacking | 2.25 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 19.00, gap = -6.25
- Verdict: **STOP** — pipeline degraded. DO NOT train κ; debug distillation prompt / batch / shuffle seed.

