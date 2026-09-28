# Phase 5 remediation — STRICT 1-plan/subagent audit (H1 diagnostic)

*Generated 2026-04-26 19:44:19*

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
| **sigma_prime** | 8 | 15.50 | 2.12 | 13 | 19 | 4/8 |
| **mu_prime** | 8 | 15.38 | 2.18 | 13 | 18 | 4/8 |

**Δ_strict (μ' − σ') = -0.12** /45


## Per-dim means matrix (strict)

| Baseline | U1 | U2 | U3 | U4 | U5 | T1 | T2 | T3 | T4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **sigma_prime** | 1.50 | 2.50 | 1.50 | 2.25 | 1.00 | 2.12 | 1.50 | 1.12 | 2.00 |
| **mu_prime** | 1.50 | 2.25 | 1.50 | 2.12 | 1.38 | 1.75 | 1.25 | 1.50 | 2.12 |

## All totals per baseline

- **sigma_prime**: [14, 14, 19, 17, 17, 13, 17, 13]
- **mu_prime**: [17, 18, 13, 17, 14, 13, 13, 18]
