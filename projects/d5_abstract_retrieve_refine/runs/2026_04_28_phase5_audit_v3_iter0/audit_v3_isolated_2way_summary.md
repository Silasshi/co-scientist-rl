# Phase 4a audit_v3 ISOLATED 2way — σ' vs μ' per goal

*Generated 2026-04-26 17:36:23*

Methodology: 8 balanced batches (1 σ' + 1 μ' each) × 8 parallel
Opus subagents. Each subagent scores 2 anonymized plans (plan_A, plan_B)
with strict per-plan independence instructions.

## Per-baseline aggregates

| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 8 | 21.62 | 2.83 | 15 | 25 | 5/8 |
| **mu_prime** | 8 | 22.25 | 2.54 | 18 | 26 | 6/8 |

**Δ (μ' − σ') = +0.62** /45


## Per-dim means matrix

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 2.62 | 3.00 | 2.75 | 2.62 | 2.38 | 2.12 | 1.88 | 1.75 | 2.50 |
| **mu_prime** | 2.38 | 3.12 | 2.75 | 3.12 | 2.00 | 2.50 | 2.25 | 1.62 | 2.50 |

## All totals per baseline

- **sigma_prime**: [25, 23, 23, 22, 15, 20, 23, 22]
- **mu_prime**: [23, 23, 21, 26, 24, 24, 19, 18]
