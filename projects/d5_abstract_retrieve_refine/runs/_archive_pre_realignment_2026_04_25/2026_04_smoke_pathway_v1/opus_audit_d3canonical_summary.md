# Opus D3-canonical Depth Audit — 2026_04_smoke_pathway_v1

**Prompt source:** `src/co_scientist/ttt_discover/opus_eval_agent.py:DEPTH_AUDIT_PROMPT` (lines 30-77).
**Scale:** 4 dims x 1-5 integer (Math, Novelty, Realism, Rigor), total / 20.
**Items scored:** 1 reference_gold + 8 B_baseline + 8 A_with_abstraction = 17.

## Headline numbers

| Condition | n | Math | Novelty | Realism | Rigor | Total |
|---|---|---|---|---|---|---|
| reference_gold | 1 | 5.00 | 5.00 | 5.00 | 5.00 | **20.00** |
| A_with_abstraction | 8 | 1.00 | 2.00 | 2.00 | 2.00 | **7.00** |
| B_baseline | 8 | 1.13 | 1.00 | 2.00 | 1.00 | **5.13** |
| **Delta A - B** |  | -0.13 | +1.00 | 0.00 | +1.00 | **+1.88** |

## Per-plan scores

| # | Condition | Sample | Math | Nov | Real | Rig | Total |
|---|---|---|---|---|---|---|---|
| ref | reference_gold | - | 5 | 5 | 5 | 5 | 20 |
| 0 | B_baseline | 0 | 1 | 1 | 2 | 1 | 5 |
| 1 | B_baseline | 1 | 1 | 1 | 2 | 1 | 5 |
| 2 | B_baseline | 2 | 1 | 1 | 2 | 1 | 5 |
| 3 | B_baseline | 3 | 2 | 1 | 2 | 1 | 6 |
| 4 | B_baseline | 4 | 1 | 1 | 2 | 1 | 5 |
| 5 | B_baseline | 5 | 1 | 1 | 2 | 1 | 5 |
| 6 | B_baseline | 6 | 1 | 1 | 2 | 1 | 5 |
| 7 | B_baseline | 7 | 1 | 1 | 2 | 1 | 5 |
| 8 | A_with_abstraction | 0 | 1 | 2 | 2 | 2 | 7 |
| 9 | A_with_abstraction | 1 | 1 | 2 | 2 | 2 | 7 |
| 10 | A_with_abstraction | 2 | 1 | 2 | 2 | 2 | 7 |
| 11 | A_with_abstraction | 3 | 1 | 2 | 2 | 2 | 7 |
| 12 | A_with_abstraction | 4 | 1 | 2 | 2 | 2 | 7 |
| 13 | A_with_abstraction | 5 | 1 | 2 | 2 | 2 | 7 |
| 14 | A_with_abstraction | 6 | 1 | 2 | 2 | 2 | 7 |
| 15 | A_with_abstraction | 7 | 1 | 2 | 2 | 2 | 7 |

## Duplicate content note

Sampling collapsed — B samples 2/5, 4/6/7 and A samples 6/7 are byte-identical (md5 confirmed). Scored independently per spec.

## Verdict

**(a) Reference anchor calibration.** The reference plan scores 5/5/5/5 = 20/20 exactly as the D3 anchor predicts. It matches the anchor examples verbatim: J_beta entropic objective with derived gradient and adaptive beta via KL budget gamma=ln 2 (math=5); MAX-PUCT as one-character deviation from AlphaZero with rank prior and lineage blocking (novelty=5); gpt-oss-120b via Tinker, LoRA r=32, Adam 4e-5, 512 rollouts in 8 groups of 64, 50 steps, ~$500 (realism=5); four benchmarks with published AI priors — Erdos min-overlap < 0.380924, AHC039 > 566997, TriMul kernels on A100/H100, MAGIC 0.64 vs target ~0.71 (rigor=5). Prompt + anchor are calibrated.

**(b) A vs B.** A_with_abstraction beats B_baseline by +1.88 total, concentrated on novelty (+1.0) and rigor (+1.0). The signal is architecture vocabulary: every A plan invokes "cumulative state buffer", "tree-search over past states with low ancestry/lineage overlap", "adaptive per-state hyperparameters under a divergence budget", "importance-sampled gradients", and a "max-focused training objective" — the exact reference-plan skeleton — plus a compute-matched Best-of-N baseline. Every B plan falls back to the textbook recipe (policy-gradient / RWML + MCTS-or-beam + LoRA + memory buffer) with unnumbered generic baselines. Neither condition produces equations (A math=1.00, B math=1.13; only B plan 3 has one trivial gradient-ascent formula), and neither names a concrete open problem with a prior AI number — so realism and math are flat. Abstraction retrieval moves plans up the "names the right concepts" ladder, nothing lower.

**(c) Gap from A to reference_gold.** 13 points out of 20 (A mean 7.0 vs ref 20). A plans have the skeleton but none of the four load-bearing details: (1) entropic objective J_beta with derived policy gradient and beta->infinity limit (math 1 -> 5); (2) explicit MAX-vs-MEAN Q-value deviation from AlphaZero with mechanistic justification (novelty 2 -> 5); (3) specific base model, LoRA rank, learning rate, batch structure, step count, dollar cost (realism 2 -> 5); (4) concrete open problems with published prior numbers and testable exceed-these-numbers claims (rigor 2 -> 5). This is exactly the D5 Phase 0 hypothesis: abstraction-retrieval is necessary but not sufficient. The refine stage (not exercised in this smoke) is the one that has to carry derivations + specific numbers.

## Notes on prompt-version comparison

Score fresh from the D3 canonical prompt without looking at the prior audit (`opus_depth_audit.json` / `opus_audit_summary.md`). Those files can now be diffed against this one to isolate the prompt-version effect.
