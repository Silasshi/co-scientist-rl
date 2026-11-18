# GRPO with an LLM rubric reward (D1 reproduction)

**Follows:** Goel et al., *Training AI Co-Scientists Using Rubric Rewards*, [arXiv:2512.23707](https://arxiv.org/abs/2512.23707), with the GRPO group-relative baseline from DeepSeekMath ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300)).

## Idea
The policy writes a research plan for a goal. A frozen grader model reads the goal, the goal-specific rubric items and a reference plan. It rates every rubric item on 7 general desiderata, at 4 levels each. The plan's score is the mean over items. Each goal gets a group of 8 sampled plans, and the group mean is the baseline: there is no value network.

## Implementation — [`src/co_scientist/rubric_reward/grpo/best_ver.py`](../../src/co_scientist/rubric_reward/grpo/best_ver.py)

| Step | Where |
|---|---|
| Policy prompt with `<think>` / `<solution>` format | [`build_research_plan_prompt` L79](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L79) |
| Grader prompt (goal + rubric + reference + plan → XML) | [`build_grader_prompt` L137](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L137) |
| XML → score; 4-level map {0, 0.2, 0.6, 1.0} averaged over desiderata and items | [`compute_rubric_reward_from_xml` L228](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L228) |
| Reward = rubric + 0.08 · Gaussian length bonus (600 ± 120 words) − format penalty | [L634–L639](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L634) |
| Group-relative advantage `r_i − mean(group)`; groups with zero variance are skipped | [L696–L704](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L696) |
| Advantage is broadcast to every generated token; prompt tokens are padded with 0 | [L728–L738](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L728) |
| PPO-clip loss (ε = 0.2, no KL term, as in the paper's appendix) through Tinker `forward_backward` | [L801–L808](../../src/co_scientist/rubric_reward/grpo/best_ver.py#L801) |

Engineering details that mattered:
- Async sampling and grading with futures. Each batch of 64 goals × 8 plans issues 512 grader calls.
- Degenerate or format-violating samples are dropped instead of being given a penalty. Without this, malformed plans that happened to score well inflated training reward to 0.84 with no gain on eval ([`reward_shaping.md`](../../projects/rubric_reward/knowledge/methods/reward_shaping.md)).

## Why the graded scale
The paper's binary per-item reward ("no violations" → 1) gave a flat curve at Goel's learning rate: 0.103 → 0.095 over 60 batches (run [`22(6)`](../../projects/rubric_reward/runs/2026_01/baseline_exploration/22%286%29/summary.md), `metrics.jsonl`). Recomputing from that run's raw training logs (not included here):
- 60–75% of samples per batch score exactly 0.
- In 28.5% of GRPO groups every sample has the same reward, so those groups give no gradient.

The 4-level graded satisfaction scale restores variance within groups. All runs are indexed in [`RUNS_INDEX.md`](../../projects/rubric_reward/runs/RUNS_INDEX.md).

## Result
Qwen3-30B-A3B, LoRA rank 64, `facebook/research-plan-gen` ML split. Evaluated on 640 held-out goals × 8 samples (5,120 plans) with the same frozen Qwen3-30B-A3B grader, using the identical protocol before and after training:

| | Rubric score |
|---|---|
| Base model (checkpoint −1) | 0.560 |
| **After 215 GRPO batches (2 epochs)** | **0.693** |
| GPT-5.4 zero-shot (640 goals × 1) | 0.826 |
| Reference plans (n = 685, graded with themselves in context) | 0.867 |

Caveats:
- Single run, single grader family.
- The grader also produced the training reward. That is why the follow-up work in [LLM-judge evaluation](06-llm-judge-evaluation.md) uses independent judges.
