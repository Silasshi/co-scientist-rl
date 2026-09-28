# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 17:21:22*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **26.50 / 45**
- **Std**: 1.41
- **Min / Max**: 24 / 29
- **Distinct totals**: 5/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 27 |
| tau_1 | 25 |
| tau_2 | 29 |
| tau_3 | 26 |
| tau_4 | 27 |
| tau_5 | 27 |
| tau_6 | 27 |
| tau_7 | 24 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.75 |
| U2_significance | 3.00 |
| U3_originality | 2.00 |
| U4_clarity | 3.00 |
| U5_reproducibility | 3.12 |
| T1_necessity | 3.62 |
| T2_disentanglement | 3.50 |
| T3_compute | 3.12 |
| T4_reward_hacking | 2.38 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 26.50, gap = +1.25
- Verdict: **CONFIDENT** — distillation pipeline > Opus slim. Proceed κ training.

