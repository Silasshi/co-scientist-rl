# Phase 5 remediation — STRICT 1-plan/subagent audit (H1 diagnostic)

*Generated 2026-04-26 20:38:57*

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
| **sigma_prime** | 24 | 15.83 | 1.86 | 13 | 20 | 8/24 |
| **mu_prime** | 24 | 14.88 | 2.45 | 11 | 18 | 8/24 |

**Δ_strict (μ' − σ') = -0.96** /45


## Per-dim means matrix (strict)

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 1.29 | 2.75 | 1.67 | 2.33 | 1.08 | 2.00 | 1.42 | 1.29 | 2.00 |
| **mu_prime** | 1.50 | 2.25 | 1.42 | 2.08 | 1.46 | 1.71 | 1.21 | 1.29 | 1.96 |

## All totals per baseline

- **sigma_prime**: [16, 13, 13, 15, 17, 17, 15, 14, 16, 18, 14, 17, 19, 14, 17, 17, 15, 14, 16, 20, 13, 16, 18, 16]
- **mu_prime**: [17, 16, 17, 18, 15, 13, 13, 18, 13, 15, 11, 14, 13, 18, 18, 17, 18, 17, 16, 13, 11, 12, 13, 11]
