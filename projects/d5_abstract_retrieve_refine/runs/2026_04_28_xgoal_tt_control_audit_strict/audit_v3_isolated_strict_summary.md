# Phase 5 remediation — STRICT 1-plan/subagent audit (H1 diagnostic)

*Generated 2026-04-26 19:44:21*

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
| **sigma_prime** | 8 | 16.75 | 1.56 | 14 | 19 | 6/8 |
| **mu_prime** | 8 | 16.12 | 2.57 | 12 | 19 | 4/8 |

**Δ_strict (μ' − σ') = -0.62** /45


## Per-dim means matrix (strict)

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 1.50 | 3.00 | 1.75 | 2.50 | 1.12 | 2.12 | 1.50 | 1.25 | 2.00 |
| **mu_prime** | 1.50 | 2.75 | 1.62 | 2.12 | 1.38 | 2.00 | 1.38 | 1.38 | 2.00 |

## All totals per baseline

- **sigma_prime**: [14, 17, 15, 19, 16, 17, 18, 18]
- **mu_prime**: [19, 16, 12, 16, 18, 18, 12, 18]
