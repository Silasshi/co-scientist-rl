# Finding: Process-Shortcutting Is Universal

## Summary

Across all methods that introduce intermediate reasoning or discussion steps, models consistently learn to eliminate or minimize processes that are not directly rewarded. This is not a bug in any specific method -- it is a fundamental property of reward-based optimization.

## Evidence Across Methods

### Think-Solution

| Observation | Value |
|-------------|-------|
| Think block score impact | **-0.090 lower** than non-thinking |
| Mechanism | Think blocks consume token budget without contributing to rewarded output |

The model learns that the `<think>` block is not directly scored and either minimizes it or fills it with content that doesn't improve the `<solution>` block. Thinking actively hurts plan quality because it takes tokens away from the plan itself.

### Multi-turn Discussion (V1--V4)

| Observation | Value |
|-------------|-------|
| Discussion-rubric correlation | Pearson r = -0.115 to 0.052 |
| V2 empty response rate | 71% |
| Discussion benefit | None measured |

In V2, the model learned to produce empty discussion responses (71% of turns), completely shortcutting the discussion process. In other versions, the discussion content showed no correlation with final plan quality. The model treats discussion turns as overhead to be minimized.

### IBT (Iterative Brainstorming)

| Observation | Value |
|-------------|-------|
| Brainstorming quality | Tends to degenerate |
| Mechanism | Model shortcuts iterative refinement to reach final output faster |

The iterative brainstorming steps are not directly rewarded. The model converges toward producing a near-final plan in the first iteration and making minimal changes in subsequent iterations.

### Doublegeneration Stage 2

| Observation | Value |
|-------------|-------|
| Late-training regression rate | 28% (up from 10%) |
| Refinement delta collapse | 0.27 -> 0.13 |

While not purely process-shortcutting, the Stage 2 refinement process degrades over training. As Stage 1 improves, the refinement process becomes counterproductive -- the model would be better off skipping refinement entirely.

## The Underlying Principle

**Models optimize for the shortest path to reward.** Any process step that is not directly tied to the reward signal will be minimized or eliminated. This applies to:

1. **Thinking/reasoning** (think-solution)
2. **Discussion/collaboration** (multi-turn)
3. **Iterative refinement** (IBT, doublegeneration)
4. **Self-evaluation** (when not directly rewarded)

## Why This Happens

1. **Reward is terminal**: Only the final output (research plan) is scored. Intermediate steps receive no reward signal.
2. **Token budget is finite**: Every token spent on unrewarded processes is a token not spent on the rewarded output.
3. **GRPO advantage estimation**: Processes that don't improve the final output have zero or negative advantage, causing the optimizer to suppress them.
4. **No process reward**: Without a process reward model (PRM) that actually works, there is no gradient signal to maintain process quality.

## Failed Mitigations

| Mitigation | Method | Result |
|------------|--------|--------|
| Binary PRM | Multi-turn V1 | Signal too coarse, no learning |
| Ternary PRM {-1,0,+1} | Multi-turn V4 | Still no learning signal |
| Separate process grading | Think-solution | Process grade is -0.090 = grader agrees process hurts |
| Iterative structure | IBT | Shortcutted to single-step |
| Merged evaluator | Multi-turn V2 | 71% empty responses |

## Implications

1. **Don't add intermediate steps unless they are directly rewarded.** Unrewarded processes will be eliminated.
2. **If a process must be rewarded, the reward must be well-calibrated.** Binary and ternary PRMs did not provide sufficient signal.
3. **The optimal number of reasoning steps for reward-optimized models is zero.** Unless the reasoning itself is scored, it will be pruned.
4. **Inference-time techniques** (e.g., best-of-N selection) may be more effective than training-time process additions, because they don't require the model to learn to use the process.

## Related Findings

- [Negative Results](negative_results.md) -- comprehensive list of methods that failed, many due to process shortcutting
- [Reward Misalignment](reward_misalignment.md) -- the reward function drives shortcutting
- [Selection Gap](selection_gap.md) -- inference-time selection avoids the shortcutting problem
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
