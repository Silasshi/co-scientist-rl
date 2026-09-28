# Grader Panel Triangulation: v8 Rubric vs Independent Depth Audit

*2026-04-22.* Ran 15 independent Opus subagents (one plan each, no halo) with the
holistic `depth_audit_v2` prompt (4 dimensions × 1-10) to compare against the
v8-rubric Opus grades. This tests whether v8-Opus captures "real" quality or
just v8-rubric compliance.

## 1. The two Opus rubrics largely agree (Spearman ρ = +0.83)

**Takeaway**: Our v8 rubric is **not** circular. A holistic, rubric-agnostic Opus
grader ranks plans similarly. High agreement across 15 plans at different quality
levels validates v8 as a reasonable quality proxy.

| plan | v8 aggregate | depth_audit /40 | bucket |
|---|---:|---:|---|
| plan_00-02 (reference) | 0.815 | 32-33 | anchor |
| plan_03-06 (high) | 0.64-0.83 | 15-21 | high |
| plan_07-10 (mid) | 0.19-0.72 | 12-16 | mid |
| plan_11-14 (low) | 0.21-0.38 | 11-14 | low |

The **only mid-outlier**: plan_07 (v8=0.185 vs depth=14) — v8 penalized it
harshly (likely G4/G6 counting), depth_audit gave more credit. plan_08 opposite
direction (v8=0.72 vs depth=16) — v8 generous, depth strict.

## 2. Tinker model rankings — **GPT-OSS-120B is the clear winner**

| Model | ρ vs v8-Opus | ρ vs depth-Opus | n | Robust? |
|---|:-:|:-:|:-:|:-:|
| **gpt_oss_120b** | **+0.971** | **+0.878** | 15 | ★★★ |
| qwen3_235b | +0.756 | +0.794 | 15 | ★★ |
| gpt_oss_20b | +0.878 | +0.722 | 15 | ★★ |
| deepseek_v3_1 | +0.694 | +0.755 | 14 | ★ |
| qwen3_4b | +0.662 | +0.500 | 15 | - |
| qwen3_30b | +0.422 | +0.269 | 11 | - |
| llama_3_1_8b | +0.036 | +0.048 | 15 | ✗ |

Key insights:
- **GPT-OSS-120B** tops BOTH rubrics (0.97 on v8, 0.88 on depth) — not rubric-specific
- **Qwen3-235B** drops from #1 (v8: 0.74 mean-per-signal) to #2 (plan-level: 0.76/0.79) once we triangulate. Still strong but not dominant.
- **Qwen3-30B current grader** stays weak on both (0.42/0.27) — confirms we need to change
- **Llama** fails on both, not a rubric artifact

## 3. Revised recommendation: use GPT-OSS-120B, not Qwen3-235B

Previously selected Qwen3-235B based on per-signal ranking against v8-Opus alone.
After triangulation, **GPT-OSS-120B is more robust**:

- Wins on v8-Opus plan-level (0.971 vs 0.756)
- Wins on depth-Opus (0.878 vs 0.794)
- OpenAI-OSS architecture is cross-family from Qwen policy → less self-preference bias risk
- Similar tinker cost ($0.44 sample vs $1.70 sample for Qwen-235B)

**Caveat**: per-signal ρ (original panel test) had Qwen-235B slightly ahead of
GPT-OSS-120B on mean (0.741 vs 0.732). The plan-level aggregate ρ (this triangulation)
strongly favors GPT-OSS-120B. Both metrics are valid; plan-level aggregate is more
RL-relevant because the training reward is aggregate, not per-signal.

## 4. The v8 rubric itself is validated

ρ=0.83 between v8-aggregate and holistic depth audit means:
- v8 is capturing real quality signal, not just v8-specific heuristics
- Rubric redesign (remove G3/G5, stricter thresholds) is working
- **Remaining 0.17 disagreement** = where v8 misses holistic quality (likely plans with heavy math formalism but shallow mechanism — v8's G12 overweighted in those cases)

## 5. Next steps

1. **Use GPT-OSS-120B as the v8 grader** for B4_v8 and MAIN_v8 experiments
2. Keep Qwen3-235B as a secondary grader for cross-check
3. At iter ~5, ~10, ~25, run depth_audit Opus on best-buffer-plan to validate v8 score isn't drifting from holistic quality
