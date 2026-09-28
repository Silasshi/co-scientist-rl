# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 17:30:01*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **19.25 / 45**
- **Std**: 0.66
- **Min / Max**: 18 / 20
- **Distinct totals**: 3/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 18 |
| tau_1 | 19 |
| tau_2 | 19 |
| tau_3 | 19 |
| tau_4 | 20 |
| tau_5 | 20 |
| tau_6 | 19 |
| tau_7 | 20 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.00 |
| U2_significance | 3.00 |
| U3_originality | 2.00 |
| U4_clarity | 3.00 |
| U5_reproducibility | 1.75 |
| T1_necessity | 2.00 |
| T2_disentanglement | 1.38 |
| T3_compute | 1.88 |
| T4_reward_hacking | 2.25 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 19.25, gap = -6.00
- Verdict: **STOP** — pipeline degraded. DO NOT train κ; debug distillation prompt / batch / shuffle seed.

