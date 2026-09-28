# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 16:23:08*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **18.12 / 45**
- **Std**: 1.05
- **Min / Max**: 16 / 19
- **Distinct totals**: 4/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 16 |
| tau_1 | 18 |
| tau_2 | 17 |
| tau_3 | 19 |
| tau_4 | 19 |
| tau_5 | 19 |
| tau_6 | 18 |
| tau_7 | 19 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 1.75 |
| U2_significance | 3.00 |
| U3_originality | 1.50 |
| U4_clarity | 3.00 |
| U5_reproducibility | 1.12 |
| T1_necessity | 2.12 |
| T2_disentanglement | 1.62 |
| T3_compute | 2.00 |
| T4_reward_hacking | 2.00 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 18.12, gap = -7.12
- Verdict: **STOP** — pipeline degraded. DO NOT train κ; debug distillation prompt / batch / shuffle seed.

