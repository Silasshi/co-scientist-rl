# Grader Panel v8 — Final Summary

*Run date: 2026-04-22*

## Setup

- **15 FoundOpt plans** spanning quality range (3 reference-anchor + 4 high + 4 mid + 4 low)
- **10 D4-v8 signals** (G3 and G5 removed from v7)
- **9 graders**: Opus 4.7 (ground truth) + 8 tinker models
- **Scaffolding**: identical v8 `build_single_signal_prompt` for ALL graders
- Temperature 0.0, single-call per (plan × signal)

## Results (sorted by mean per-signal Spearman ρ with Opus)

| Rank | Model | Mean ρ | Pooled ρ | MAE | Within-1 | n / 150 |
|:-:|---|:-:|:-:|:-:|:-:|:-:|
| **1** | **Qwen3-235B-A22B-Instruct-2507** | **+0.741** | **+0.774** | 0.63 | **87%** | 150 ✓ |
| 2 | GPT-OSS-120B | +0.732 | +0.657 | 0.75 | 79% | 150 ✓ |
| 3 | GPT-OSS-20B | +0.674 | +0.659 | 0.71 | 80% | 133 |
| 4 | DeepSeek-V3.1 | +0.563 | +0.690 | 0.66 | 81% | 135 |
| 5 | Qwen3-30B-A3B (current grader) | +0.399 | +0.448 | 0.70 | 84% | 99 |
| 6 | Qwen3-4B-Instruct-2507 | +0.385 | +0.500 | 0.91 | 74% | 150 ✓ |
| 7 | Llama-3.1-8B-Instruct | +0.140 | +0.084 | 1.50 | 51% | 122 |
| — | Kimi-K2-Thinking | nan | +0.735 | 0.50 | 80% | 10 (killed — too slow) |

## Key Findings

### 1. Qwen3-235B is the clear winner
- **Mean ρ = 0.74** with Opus, **87% within-1** agreement
- On G11/G12/G13 (rigor/formalism/risk): ρ ≥ 0.90 — nearly perfect rank agreement
- Main weakness: G1 problem_specificity (ρ=0.21), because Opus rarely uses the low end of G1 scale (mean 4.13)

### 2. GPT-OSS family performs surprisingly well
- 120B: mean ρ=0.73 (~tied with 235B)
- 20B: mean ρ=0.67 — best small grader, even beats Qwen-30B
- Both reasoning models show high G11/G12/G13 agreement (ρ ≥ 0.93)

### 3. Our current Qwen-30B grader is mediocre
- Ranked 5/8, mean ρ only 0.40
- Especially weak on G6 (ρ=-0.23) and G9 (ρ=0.00) — the "asserted choices" and "scope red flags" signals that require semantic judgment
- This validates the concern that same-model grading is unreliable; a different grader is needed

### 4. Llama-3.1-8B is unusable
- Near-zero correlation on most signals
- Likely fails to follow the XML scaffolding format reliably
- Parse success rate ~75% (vs 100% for Qwen/GPT-OSS)

### 5. Signal-level patterns
- **Easy signals** (all graders ρ > 0.7): G11 evidence_rigor, G12 formalism, G13 risk_awareness
  - These are the "depth" signals where high-quality plans visibly differ
- **Hard signals** (graders disagree): G1, G9, G10
  - G10 approach_coverage: Opus has low variance → rank undefined for most graders
  - G9 scope_feasibility: requires judging resource-realism, which is subjective
  - G1: Opus uses narrow range (3-5), low statistical power

## Decision: Use Qwen3-235B as the grader for v8 experiments

**Rationale**:
- Top correlation with Opus (0.74 mean ρ, 0.77 pooled ρ)
- Low MAE (0.63) — absolute scores close to Opus
- 87% within-1 agreement (better than human inter-rater baseline ~80%)
- Same family as policy model → consistent tokenization, available on tinker

**Cost implication**: 235B sample pricing is higher than 30B (~5.7× per token). For typical grader loop (10 signals × n_fresh+n_revise × n_iters), switching from 30B to 235B increases grader cost from ~$0.30/run to ~$1.70/run — absorbable.

**Alternative for ablation**: GPT-OSS-120B if we want cross-family grader validation (different bias profile, same quality tier).

## Files

- `test_plans/plan_00.txt ... plan_14.txt` + `sources.jsonl` — 15 test proposals
- `grades/opus.jsonl` — 150 Opus ground-truth grades (merged from 5 subagent groups)
- `grades/opus_group_{0..4}.jsonl` — per-subagent files (race-safe backups)
- `grades/{model_key}.jsonl` — each tinker model's grades (parsed + raw)
- `analysis/correlation_matrix.jsonl` — per-(model, signal) metrics
- `analysis/heatmap.png` — visualization
- `analysis/overall_ranking.md` — auto-generated ranking table

## Caveats

1. **n=15 is small**. Spearman ρ SE ≈ 0.22. Differences between adjacent-ranked models (e.g., 120B vs 20B) are within noise.
2. **Kimi-K2-Thinking killed at 13/150**: reasoning-mode latency incompatible with grader-loop budgets. Cannot conclude whether Kimi would perform well.
3. **Qwen-30B partial (99/150)**: grader loops also timed out on some calls. Results are still indicative but undercount.
4. **Reference anchor plans (plan_00-02) are identical**: provides a degenerate Spearman estimate for those 3 rows (all same score → rank undefined). This is by design to measure grader consistency, not ranking.
