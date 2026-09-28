# D5 Smoke Pathway v2 — Opus 4.7 Depth Audit (D3 Canonical Prompt)

**Prompt source**: `src/co_scientist/ttt_discover/opus_eval_agent.py:DEPTH_AUDIT_PROMPT` (lines 30-77)
**Scale**: 4 dims (Math / Novelty / Realism / Rigor) × 1-5 integer, total /20
**Evaluator**: Claude Opus 4.7 (1M context)
**Date**: 2026-04-23

## Summary Table

| Condition | n | Math | Novelty | Realism | Rigor | **Total /20** |
|---|---|---|---|---|---|---|
| B_baseline | 8 | 1.00 | 1.13 | 1.75 | 1.50 | **5.375** |
| A_with_abstraction | 8 | 2.25 | 3.00 | 3.00 | 3.00 | **11.25** |
| **Δ (A−B)** | | **+1.25** | **+1.88** | **+1.25** | **+1.50** | **+5.88** |

Raw per-plan totals (sorted by total):

| idx | cond | sample_idx | M | N | R | Rg | total |
|---|---|---|---|---|---|---|---|
| 2  | B | 2 | 1 | 1 | 1 | 1 | 4 |
| 3  | B | 3 | 1 | 1 | 1 | 1 | 4 |
| 0  | B | 0 | 1 | 1 | 2 | 1 | 5 |
| 1  | B | 1 | 1 | 1 | 2 | 2 | 6 |
| 4  | B | 4 | 1 | 1 | 2 | 2 | 6 |
| 5  | B | 5 | 1 | 1 | 2 | 2 | 6 |
| 6  | B | 6 | 1 | 2 | 2 | 1 | 6 |
| 7  | B | 7 | 1 | 1 | 2 | 2 | 6 |
| 8  | A | 0 | 2 | 3 | 3 | 3 | 11 |
| 9  | A | 1 | 2 | 3 | 3 | 3 | 11 |
| 10 | A | 2 | 2 | 3 | 3 | 3 | 11 |
| 11 | A | 3 | 2 | 3 | 3 | 3 | 11 |
| 14 | A | 6 | 2 | 3 | 3 | 3 | 11 |
| 15 | A | 7 | 2 | 3 | 3 | 3 | 11 |
| 12 | A | 4 | 3 | 3 | 3 | 3 | 12 |
| 13 | A | 5 | 3 | 3 | 3 | 3 | 12 |

B floor: 4/20 (two truncated duplicates, idx 2 & 3).
A ceiling: 12/20 (two plans with written equation, idx 12 & 13).
Zero overlap between A and B per-plan totals.

## Duplicates observed

- Plans idx 2 and 3 (B_baseline, sample_idx 2 and 3): byte-identical truncated stubs. Both score 4/20.
- Plans idx 11, 14, 15 (A_with_abstraction, sample_idx 3, 6, 7): byte-identical. All score 11/20.

So effective distinct-plan n is B=7, A=6. The A > B direction is unchanged, and all 6 distinct A plans still dominate all 7 distinct B plans.

## Qualitative Verdict

(a) **A > B on v2 — large and consistent**. A_with_abstraction total mean is 11.25 / 20 versus B_baseline 5.375 / 20, a delta of +5.875 (more than doubling the B floor). The gap is wider than what would be expected from noise on an n=8 vs n=8 comparison because every single A plan (11 or 12) scores above every single B plan (4-6). There is zero overlap in the per-plan total distribution. Per-dimension deltas: math +1.25, novelty +1.875, realism +1.25, rigor +1.5. Note that 3/8 A plans are byte-identical duplicates (sample_idx 3, 6, 7 collapse to one text), and 2/8 B plans (sample_idx 2, 3) are identical truncated stubs, which inflates apparent n but does not change the direction or rough magnitude of the effect.

(b) **Math and realism both improve on v2 — unlike v1**. Math moves from a uniform 1/5 floor in B (no equations anywhere) to a 2-3/5 range in A: all A plans describe the log-sum-exp / soft-max objective concept verbally, and 2 of 6 distinct A plans (sample_idx 4 and 5, the "plan_12" and "plan_13" texts) actually write the equation J = log Σ_a exp(β R(a)) · π(a|s) inline. Realism moves from B's 1-2/5 (vague "small learning rate", arbitrary "100 steps", no consistent parameterization) to A's 3/5 (all parameters are at least conceptually named — per-state β, KL budget, importance ratio, buffer with max-reward descendants — even though few specific numbers appear). The v1 ceiling where math and realism were flat is broken.

(c) **Yes, two A plans attempt derivation/equations**. Plans at idx 12 and 13 (A_with_abstraction sample_idx 4 and 5) both write the log-sum-exp objective equation in LaTeX. They stop short of deriving the gradient or showing a convergence/β-limit sketch, so they do not reach the D3 5/5 anchor (J_β + ∇J_β + adaptive β via KL budget + MAX-PUCT). But this is a qualitative break from v1, where no A plan wrote any equation.

(d) **Best A (12/20) vs worst B (4/20) quality floor**. The best-scoring A plans (idx 12, 13) hit "one adapted formula for this plan + non-obvious combination with justification + numerically consistent + named baselines with general prior reference", which is solid 3/5 across all four dimensions plus one 3 on math for writing the equation. They do not reach the reference anchor because: no gradient derivation, no concrete MAX vs MEAN limit argument, no specific prior AI numbers (e.g., no "LeanDojo solves X%"), and no specific benchmark names with published results. The worst B plans (idx 2, 3 — identical truncated stubs at 1853 chars) score 4/20: they are cut off mid-methodology, describe SAS-GF only as a name, and contain no evaluation section, baselines, equations, or numerical claims of any kind. The v2 abstraction's derivation sketches — step-by-step "write metric → define J → log-sum-exp → derive gradient → constrain β by KL → importance-sample correction" — visibly transfer into A plans' methodology sections, even though the generator frozen Qwen3-30B-A3B only partially executes the full derivation (typically writes the objective but skips the gradient derivation).

## Reference anchor (5/5) — for context

- **Math 5/5**: Novel objective (e.g., J_β entropic with derivation of ∇J_β + adaptive β via KL budget) + convergence/limit sketch.
- **Novelty 5/5**: Novel algorithm design with rigorous justification (e.g., MAX-PUCT with one-character deviation from AlphaZero and limit proof).
- **Realism 5/5**: All numerical claims consistent, specific, and justified (e.g., gpt-oss-120b / LoRA r=32 specifics).
- **Rigor 5/5**: Specific open problems with exact prior AI numbers and testable claims (e.g., Erdős problems, TriMul, AtCoder with published baselines).

No v2 plan reaches 5/5 on any dimension. The v2 A ceiling (12/20, 3/5 × 4) is respectable given the frozen 30B-A3B generator, but leaves a clear gap to the TTT-Discover reference (which would score 20/20).
