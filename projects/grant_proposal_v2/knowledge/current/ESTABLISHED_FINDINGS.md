# Established Findings (from D4)

*All findings below are supported by completed experiments. See RUN_AUDIT.md for run details.*

## F1: Critique is Redundant for Plan Quality

**Evidence**: B4_paper (full critique) = C2 (scores only) = **19/40 Opus** on FoundOpt.
Full critique boosts Qwen +0.115 (0.855→0.970) but adds zero Opus quality.

**Implication**: The frozen 30B model already has self-diagnostic ability from scores alone. Critique text helps the model game the Qwen grader (write to rubric keywords) but doesn't improve genuine quality.

## F2: RL Goodharts — More RL = Worse Opus

**Evidence** (D4-v7, FoundOpt, 30B, scores_only):

| Method | Qwen | Opus | RL "strength" |
|---|---|---|---|
| B4 (no RL) | 0.855 | **19** | none |
| Aggregate RL | 0.940 | 15 | medium |
| SDPO | 0.970 | 13 | high (per-token) |

Qwen and Opus are perfectly inversely correlated within this comparison.

**Implication**: Any RL method that optimizes Qwen 30B reward will Goodhart. The issue is the reward model, not the RL algorithm.

## F3: 235B >> 30B (+10-17 Opus Points)

**Evidence**: 235B policy + Better Reward rubric = 26-29/40 Opus. 30B + D4-v7 rubric = 13-19/40.
*Caveat: different rubrics used, so the gap may be partially rubric effect.*

**Implication**: Model scale is the single biggest lever for plan quality. No 30B RL trick matches a 235B zero-shot.

## F4: Cross-Domain Gap is Universal (~19 points)

**Evidence** (D4-v7 B4, 6 goals):

| Goal | Generated | Reference | Gap |
|---|---|---|---|
| FoundOpt (AI) | 17 | 34 | -17 |
| Causal Healthcare | 15 | 35 | -20 |
| Chemo Toxicity | 18 | 36 | -18 |
| Ecosystem Dynamics | 16 | 35 | -19 |
| Sentencing Disparities | 17 | 37 | -20 |
| Climate Displacement | 14 | 34 | -20 |

**Implication**: The quality gap is not domain-specific. The model's capability ceiling applies universally.

## F5: Only Frontier Models Can Judge Plan Quality

**Evidence** (10-judge pairwise on C2/C3/C4):

| Judge | Position Bias | Opus Agreement |
|---|---|---|
| **Opus 4.7** | 0/3 | gold standard |
| **GPT-5.4** | 0/3 | 2/3 |
| **o3** | 1/3 | 2/3 |
| Qwen 235B | 1/3 | 0/3 (opposite!) |
| Qwen 30B | 2/3 | 0/3 |
| GPT-4.1, o4-mini, DeepSeek, Kimi | 3/3 | 0/3 |

**Implication**: Qwen family (30B, 235B) share the same bias — they prefer RL-Goodharted plans. Only cross-family frontier models (Opus, GPT-5.4, o3) give reliable quality judgments.

## F6: SDPO Learns Efficiently but Teacher is Flawed

**Evidence**: SDPO pos_frac 12%→47% over 35 iters. mean_adv -0.20→-0.01. buf_max overtakes both C2 and C4 on Qwen.

**Implication**: SDPO as an algorithm works — it successfully distills the teacher's preferences. But the self-teacher (Qwen 30B + critique) has the same biases as the grader. Self-distillation amplifies Goodhart.

## F7: Standard Rubric > Strict Variants

**Evidence**: Multi-prompt strict rubric validation showed ALL strict variants performed worse than standard (G12 ρ 0.902→0.552 with Opus).

**Implication**: The D4-v7 signal set is already well-designed. The problem is not signal definition but signal aggregation and model capability.
