# τ baseline audit_v3 ISOLATED — single-baseline summary

*Generated 2026-04-26 17:49:17*
*N = 8 plans, 8 subagents (one per plan, per-plan isolation)*

## Aggregate

- **Mean total**: **25.75 / 45**
- **Std**: 2.22
- **Min / Max**: 23 / 30
- **Distinct totals**: 4/8

## Per-plan totals

| plan_id | total |
|---|---:|
| tau_0 | 23 |
| tau_1 | 24 |
| tau_2 | 27 |
| tau_3 | 30 |
| tau_4 | 27 |
| tau_5 | 24 |
| tau_6 | 27 |
| tau_7 | 24 |

## Per-dim means

| Dim | Mean / 5 |
|---|---:|
| U1_soundness | 2.50 |
| U2_significance | 3.12 |
| U3_originality | 2.12 |
| U4_clarity | 3.12 |
| U5_reproducibility | 2.50 |
| T1_necessity | 3.62 |
| T2_disentanglement | 3.38 |
| T3_compute | 3.00 |
| T4_reward_hacking | 2.38 |

## Gate 1 verdict (vs σ = 25.25)

- τ mean = 25.75, gap = +0.50
- Verdict: **NEUTRAL** — pipeline ≈ slim. Proceed κ training, but mandate pairwise verification.

