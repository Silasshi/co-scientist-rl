# Investigation Questions

*These are the open questions that D4v2 experiments should answer.*

## Q1: Is it a Model Problem or a Reward Problem?

**What we know**: 30B plans score 13-19/40 on Opus. 235B plans score 26-29/40 (different rubric).
**What we don't know**: If we run 235B with D4-v7 standard rubric, would it score 29? Or is the 235B advantage partly from the Better Reward rubric being "easier"?

**Experiment needed**: Run 235B B4 with D4-v7 standard rubric on FoundOpt. Compare Opus score directly with 30B B4 (19/40). This isolates model capability from rubric design.

**If 235B D4-v7 ≈ 29**: Model capability is the bottleneck. 30B simply can't generate deep enough content.
**If 235B D4-v7 ≈ 19**: Rubric/reward design is the bottleneck. Even 235B hits the same ceiling under D4-v7 signals.

## Q2: What Does Critique Actually Do?

**What we know**: Full critique = scores only = 19/40 Opus. Critique boosts Qwen +0.115 but not Opus.
**What we don't know**: Are the TWO plans qualitatively different (different content, same Opus score)? Or literally the same quality from different paths?

**Analysis needed**: Detailed text comparison of B4_paper best plan vs C2 best plan. Check:
- Word count difference
- Section structure similarity
- Citation overlap
- Per-signal Qwen score profiles (are different signals high/low?)
- Opus per-dimension breakdown (same D/M/F/G pattern?)

**Already available**: Both plans saved in `analysis/sdpo_experiment/`.

## Q3: Which Signals are Goodharting?

**What we know**: RL increases aggregate Qwen reward but decreases Opus quality.
**What we don't know**: Is it ALL signals that Goodhart, or specific ones? Are structural signals (G1/G2/G5/G10) fine while depth signals (G4/G6/G11/G12) Goodhart?

**Analysis needed**: Per-signal Qwen scores for C2/C3/C4 best plans, mapped to Opus 4-dimension breakdown:
- Qwen G6_reasoning_depth → Opus "Problem Depth" and "Methodological Substance"
- Qwen G12_formalism → Opus "Methodological Substance"  
- Qwen G3_technical_evidence → Opus "Scholarly Grounding"

**Already available**: Per-signal scores in buffer.jsonl, Opus scores in opus_scores.jsonl.

## Q4: Can a Cross-Family Reward Fix Goodhart?

**What we know**: Only Opus/GPT-5.4/o3 align with true quality. Qwen (30B, 235B) both have wrong preference direction.
**What we don't know**: If we use GPT-5.4 or Opus as the reward/teacher, would RL still Goodhart?

**Challenge**: Neither GPT-5.4 nor Opus provides per-token logprobs through tinker. SDPO requires teacher logprobs. Possible alternatives:
- GPT-5.4 as rejection sampling oracle (generate many plans → GPT-5.4 picks best → SFT)
- Opus pairwise preference → DPO training
- Use Opus as early-stopping criterion (Opus-in-the-loop)

## Q5: Is the Qwen-Opus Inversion Systematic?

**What we know**: On FoundOpt, RL plans that score higher on Qwen score lower on Opus. N=3 data points.
**What we don't know**: Is this consistent across all 6 goals? If we took the B4 best plan from each goal and compared to the MAIN best plan, would we see the same inversion?

**Already available**: 6-goal B4 paper runs + 8-goal d4v7 MAIN runs. Need Opus eval on MAIN plans.

## Priority Order

1. **Q2 (critique text analysis)** — Zero cost, just read two plans. Can do right now.
2. **Q3 (per-signal Goodhart)** — Low cost, data already available. Analyze existing scores.
3. **Q1 (235B with D4-v7 rubric)** — Medium cost, one training run (~3h compute).
4. **Q5 (multi-goal inversion)** — Medium cost, 6-8 Opus eval calls.
5. **Q4 (cross-family reward)** — High cost, requires API access to GPT-5.4/Opus for reward.
