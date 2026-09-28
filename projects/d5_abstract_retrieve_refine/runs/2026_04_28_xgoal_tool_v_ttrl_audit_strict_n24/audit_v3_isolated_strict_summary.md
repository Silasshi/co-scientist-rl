# Phase 5 remediation — STRICT 1-plan/subagent audit (H1 diagnostic)

*Generated 2026-04-26 20:40:11*

Methodology: 16 single-plan batches (8 σ' + 8 μ' interleaved at random)
× 16 parallel Opus subagents. Each subagent scores ONE anonymized plan in
ABSOLUTE isolation — no within-batch anchoring is possible by construction.

Comparison target: per-baseline mean from `audit_v3_isolated_2way_summary.md`
(2-plan/subagent original audit). |Δ_strict_minus_original| > 2.0 on ≥2 goals
with strict-direction matching pairwise → H1 CONFIRMED (anchoring causes
audit-pairwise divergence). |Δ| ≤ 1.5 across goals → H1 REJECTED (investigate
H2 dim-weighting / H3 surface-feature inflation).

## Per-baseline aggregates (strict)

| Baseline | N | Mean /45 | Std | Min | Max | Distinct totals |
|---|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 24 | 19.29 | 1.84 | 16 | 24 | 8/24 |
| **mu_prime** | 24 | 18.71 | 2.15 | 15 | 23 | 8/24 |

**Δ_strict (μ' − σ') = -0.58** /45


## Per-dim means matrix (strict)

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 1.96 | 3.00 | 2.17 | 2.96 | 1.38 | 2.38 | 1.88 | 1.42 | 2.17 |
| **mu_prime** | 2.00 | 2.96 | 2.12 | 2.54 | 1.75 | 2.08 | 1.54 | 1.46 | 2.25 |

## All totals per baseline

- **sigma_prime**: [17, 18, 18, 19, 18, 19, 17, 20, 20, 22, 21, 16, 18, 21, 24, 21, 20, 19, 22, 18, 18, 20, 19, 18]
- **mu_prime**: [18, 16, 15, 19, 18, 15, 19, 18, 17, 22, 22, 18, 20, 15, 22, 19, 19, 18, 20, 20, 18, 23, 19, 19]
