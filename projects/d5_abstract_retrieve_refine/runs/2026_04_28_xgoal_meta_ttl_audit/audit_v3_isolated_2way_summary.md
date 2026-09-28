# Phase 4a audit_v3 ISOLATED 2way — σ' vs μ' per goal

*Generated 2026-04-26 15:28:59*

Methodology: 8 balanced batches (1 σ' + 1 μ' each) × 8 parallel
Opus subagents. Each subagent scores 2 anonymized plans (plan_A, plan_B)
with strict per-plan independence instructions.

## Per-baseline aggregates

| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 8 | 17.75 | 1.30 | 16 | 20 | 5/8 |
| **mu_prime** | 8 | 19.88 | 3.92 | 13 | 27 | 5/8 |

**Δ (μ' − σ') = +2.12** /45


## Per-dim means matrix

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 2.00 | 3.00 | 2.00 | 2.62 | 1.00 | 2.00 | 1.88 | 1.12 | 2.12 |
| **mu_prime** | 2.12 | 2.62 | 2.00 | 2.62 | 2.00 | 2.25 | 1.75 | 1.88 | 2.62 |

## All totals per baseline

- **sigma_prime**: [19, 20, 18, 16, 18, 18, 17, 16]
- **mu_prime**: [16, 20, 23, 20, 27, 20, 20, 13]
