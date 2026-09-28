# Phase 4a audit_v3 ISOLATED 2way — σ' vs μ' per goal

*Generated 2026-04-26 15:35:46*

Methodology: 8 balanced batches (1 σ' + 1 μ' each) × 8 parallel
Opus subagents. Each subagent scores 2 anonymized plans (plan_A, plan_B)
with strict per-plan independence instructions.

## Per-baseline aggregates

| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 8 | 20.38 | 1.87 | 18 | 23 | 5/8 |
| **mu_prime** | 8 | 23.38 | 4.27 | 14 | 28 | 6/8 |

**Δ (μ' − σ') = +3.00** /45


## Per-dim means matrix

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 2.12 | 3.00 | 2.25 | 3.00 | 1.75 | 2.38 | 2.25 | 1.38 | 2.25 |
| **mu_prime** | 2.75 | 3.00 | 2.62 | 3.00 | 2.75 | 2.62 | 2.00 | 2.00 | 2.62 |

## All totals per baseline

- **sigma_prime**: [19, 18, 20, 23, 21, 21, 18, 23]
- **mu_prime**: [28, 24, 26, 14, 24, 21, 28, 22]
