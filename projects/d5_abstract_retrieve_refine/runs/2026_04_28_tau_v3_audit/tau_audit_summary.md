# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 17:04:51*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **20.38 / 45**
- **Std**: 1.65
- **Min / Max**: 18 / 24
- **Distinct totals**: 5/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 20 |
| tau_1 | 19 |
| tau_2 | 21 |
| tau_3 | 21 |
| tau_4 | 18 |
| tau_5 | 20 |
| tau_6 | 20 |
| tau_7 | 24 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.38 |
| U2_significance | 3.00 |
| U3_originality | 2.12 |
| U4_clarity | 2.88 |
| U5_reproducibility | 2.25 |
| T1_necessity | 2.12 |
| T2_disentanglement | 1.50 |
| T3_compute | 1.62 |
| T4_reward_hacking | 2.50 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 20.38, gap = -4.88
- Verdict: **STOP** — pipeline degraded. DO NOT train κ; debug distillation prompt / batch / shuffle seed.

