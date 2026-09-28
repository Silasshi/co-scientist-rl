# Phase 4a audit_v3 ISOLATED 2way — σ' vs μ' per goal

*Generated 2026-04-26 15:32:04*

Methodology: 8 balanced batches (1 σ' + 1 μ' each) × 8 parallel
Opus subagents. Each subagent scores 2 anonymized plans (plan_A, plan_B)
with strict per-plan independence instructions.

## Per-baseline aggregates

| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 8 | 20.00 | 1.41 | 17 | 22 | 5/8 |
| **mu_prime** | 8 | 19.12 | 5.30 | 9 | 26 | 8/8 |

**Δ (μ' − σ') = -0.88** /45


## Per-dim means matrix

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 2.00 | 3.00 | 2.12 | 2.88 | 1.75 | 2.25 | 2.00 | 1.62 | 2.38 |
| **mu_prime** | 2.00 | 2.75 | 2.12 | 2.12 | 2.12 | 2.38 | 1.50 | 1.62 | 2.50 |

## All totals per baseline

- **sigma_prime**: [22, 20, 21, 17, 20, 21, 19, 20]
- **mu_prime**: [15, 22, 23, 24, 26, 16, 18, 9]
