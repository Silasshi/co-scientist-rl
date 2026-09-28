# v7 (Qwen-30B grader) vs v8 (GPT-OSS-120B grader) — Unified Analysis

*2026-04-22. v8 runs killed at iter 13-14 after plateau. v7 C* runs completed full 25/35 iters.*

## TL;DR

Both v7 and v8 graders produce Goodhart in RL, but via **different failure modes**. Neither grader alone is sufficient. v8 rubric redesign + cross-family grader did NOT remove length bias.

| Condition | Reward grew? | Opus substance (length-blind pairwise) | Primary failure mode |
|---|---|---|---|
| v7 C2 (B4, Qwen) | +0.02 tiny | **real improvement** | none (CR pipeline works) |
| v7 C3 (SDPO, Qwen) | **+0.19 huge** | slight gain | **hallucination hacking** (fake theorems/arXiv) |
| v7 C4 (agg, Qwen) | +0.03 tiny | TIE (identical) | **buffer saturation** (iter 6 plan duplicated) |
| v8 B4 (GPT-OSS) | **+0.26** | slight gain | **length hacking** (G11/G12/G13 ↑↑) |
| v8 C3 SDPO (GPT-OSS) | +0.28 | slight gain | **length hacking + minor padding** |
| v8 C4 agg (GPT-OSS) | +0.29 | clearly better | **length hacking** (+47% chars) |

## Three Tests Run

### Test 1: Per-signal trajectory (length vs discriminating Δ)

| Run | length Δ (G11+G12+G13)/3 | discrim Δ (G4+G6+G9)/3 | L/D ratio |
|---|---:|---:|---:|
| v7 C2 B4 | +1.21 | **+1.29** | 0.94 balanced |
| v7 C3 SDPO | +0.60 | +0.11 | 5.5 |
| v7 C4 agg | +0.88 | +0.79 | 1.11 balanced |
| v8 B4 | +1.29 | +0.08 | **16** |
| v8 C3 SDPO | +0.50 | **−0.17** | ∞ (discrim down) |
| v8 C4 agg | +0.88 | +0.25 | 3.5 |

**Finding**: v7 Qwen grader gives credit on discriminating signals too, so reward grows on real quality. v8 GPT-OSS strict on discriminating → reward growth 100% routed through G11/G12/G13 → L/D ratio explodes. **v8's cross-family choice narrowed the Goodhart surface but did not close it**.

### Test 2: Opus pairwise (iter 5 best vs iter 10/final best, length-blind)

| Run | Opus winner | Margin | length_verdict |
|---|---|---|---|
| v7 C2 B4 | B (iter 24) | **clearly better** | substance |
| v7 C3 SDPO | B (iter 34) | slightly better | **mixed — hallucinated numbers, fabricated arXiv, truncated §11** |
| v7 C4 agg | **TIE** | indistinguishable | **structural — duplicated plan, no change** |
| v8 B4 | B (iter 10) | slightly better | mixed |
| v8 C3 SDPO | B (iter 10) | slightly better | mixed |
| v8 C4 agg | B (iter 10) | clearly better | mixed |

**Finding**: Pipeline IS doing substantive work on top of the Goodhart — iter-final plans are genuinely better on substance pairwise. But every trained run has detectable padding/hallucination artifacts. **Reward growth overstates real quality gain by ~3-10x**.

### Test 3: Plan diff (word count, structural similarity)

| Run | words Δ | similarity | diagnostic cue |
|---|---:|---:|---|
| v7 C2 B4 | +32% | 20% | heavy rewrite, +6 \mathcal, +2 feedback artifact |
| v7 C3 SDPO | **+1%** | 17% | same-length rewrite, **0→5 Theorem mentions** (fake) |
| v7 C4 agg | **0%** | **99.92%** | **iter 6 plan literally duplicated to iter 34** |
| v8 B4 | −16% words / +12% chars | 20% | shorter but char-dense (LaTeX bloat for G12) |
| v8 C3 SDPO | +29% | 71% | +29% appended to existing sections |
| v8 C4 agg | +41% | 14% | +41% with "Revisions Based on Feedback" self-rating |

**Finding**: Three distinct failure shapes. v7 C3 shows **hallucination-at-fixed-length** (policy swaps real content for fake math), v7 C4 shows **RL unable to beat BoN**, v8 C4 shows **length-amplified padding**.

## Core Insight: Grader Choice Routes Goodhart Into Different Signals

The fundamental issue is that **v7 and v8 rubrics share G11/G12/G13 as length-correlated signals**. These account for 32% of v8 weight (14+4+14) and 34% of v7 (14+12+8 before v8 re-weighting). No matter the grader:

- **If grader is lenient (Qwen-30B)**: policy learns to hallucinate specifics that look rigorous (C3 SDPO path)
- **If grader is strict (GPT-OSS-120B)**: policy learns to pad length since hallucinated specifics won't pass (v8 path)

Cross-family grader + stricter rubric did not remove the underlying incentive. The ONLY path left is **direct length-decorrelation at the signal-prompt level** (DECISIONS.md v9 implication #2).

## Also: Buffer Saturation Is Independent of RL Algorithm

v7 C4 (aggregate) kept the same iter-6 plan as global-best through iter 34. This means the **critique-revise-BoN exploration is the actual quality ceiling**, not the RL gradient. RL cannot push past what critique-revise can generate.

This matches the known D4 finding "CR >> RL >> nothing, B4 ≥ MAIN on Opus" — RL just adds Goodhart on top of a ceiling set by generation diversity.

## v9 Design Implications

1. **Rewrite G11/G12/G13 as count-based, not qualitative.** e.g., G12 should be "# of well-defined equations that derive a result, max 5" NOT "how formal is the math?". Same for # of named risks (G13), # of citations load-bearing to specific claims (G11). Count is length-bounded; qualitative is length-unbounded.

2. **Add a hard length cap on the grader side.** Truncate plans >2000 words before grading. This forces the policy to fit substance into a fixed budget.

3. **ODIN-style length residual head on the grader output.** Regress score from word count alone, subtract. v8 runs provide enough buffer data to fit this offline.

4. **Exploration diversity is the harder problem.** Buffer saturation (v7 C4, v8 plateau at iter 10) shows RL can't fix exploration. v9 should invest in:
   - More aggressive fresh-gen diversity (temperature scheduling, GAPO-style signal-space diversity bonus)
   - Critique-driven exploration (make critique suggest NEW directions, not polish existing ones)

5. **Hallucination hacking (v7 C3 SDPO path) is RL-specific**. Under lenient grader + high optimization pressure, policy learns to fabricate. Even if we fix length, SDPO will find a new axis to hack unless grader has verifiability built in.

## Decision: Move to v9

Based on the above, v9 should:
- Rewrite G11/G12/G13 to count-based (start fresh grader panel test)
- Try length-capped grader as ablation
- Keep GPT-OSS-120B as grader family (confirmed better than Qwen on triangulation)
- Focus next experiments on exploration diversity, not new RL algorithms

v8 runs saved at iter 13-14 (killed after plateau). v8 results confirm v8 rubric is **necessary but not sufficient**.

## Appendix: Data files

- v8 runs: `projects/grant_proposal_v2/runs/2026_04_22_v8_foundopt_{B4,MAIN_C3,MAIN_C4}/`
- v7 runs: `projects/grant_proposal/runs/2026_04_22_sdpo_{C2_B4,C3_sdpo,C4_aggregate}_scores_only/`
- depth_audit mid-run: `*/depth_audit_mid.jsonl`
- Pairwise prompts+results: referenced via agent conversation log
