# Method: SDPO (Self-Distillation Policy Optimization)

## Motivation

After bestversion established a strong single-stage GRPO baseline (eval rubric 0.693), the question was whether the loss function itself could be improved. Standard GRPO uses only within-group advantage: a sample is "good" if its reward exceeds the group mean. But this discards information -- specifically, the *reason* a sample scored well or poorly.

SDPO hypothesized that feeding the grader's diagnostic feedback (weaknesses and suggested fixes) back to the model as a "teacher" signal would provide richer gradient information than reward-only GRPO. The model would learn not just "this sample is above average" but "here is specifically how to improve." If the self-distillation signal captures information that GRPO advantages miss, it should push the policy further.

This was tested across 20 runs (the most intensive exploration of any single method in the project), with lambda values from 0.1 to 0.45, multiple architectures, and warmstart experiments from bestversion's best checkpoint.

## Pipeline

SDPO extends the bestversion GRPO loop with a teacher-student distillation term.

### Step-by-step loop

```
For each batch of 64 goals:
  1-4. [Same as bestversion: generate 8 plans, grade, parse, compute rewards]

  5. GRPO advantage (same as bestversion):
       grpo_adv_i = reward_i - mean(rewards in group)

  6. Teacher construction (SDPO-specific):
       For each plan:
         a. Extract grader feedback: weaknesses + suggested fixes (from XML)
         b. Build "improved" prompt: original prompt + feedback
         c. Get teacher_logprobs from the current model on the improved prompt
         d. Get student_logprobs from the current model on the original prompt
         e. Interpolate teacher with frozen reference model:
              teacher_logprobs = (1 - alpha) * teacher_logprobs + alpha * ref_logprobs
              (alpha = 0.01)

  7. SDPO advantage (token-level):
       sdpo_adv = teacher_logprobs - student_logprobs
       Clipped to [-5, +5]

  8. Mixed advantage:
       final_adv = (1 - lambda) * grpo_adv + lambda * sdpo_adv

  9. PPO update with final_adv (same clip_eps=0.2, Adam lr=1e-5)
```

### Key parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| lambda (mixing) | 0.1--0.45 tested | Controls SDPO vs GRPO influence |
| Default split | 0.30 * grpo_adv + 0.70 * sdpo_adv | In runs 20--24 |
| alpha (reference interpolation) | 0.01 | Small frozen-reference regularization |
| SDPO advantage clip | [-5, +5] | Prevents extreme teacher-student gaps |
| All other params | Same as bestversion | Group size 8, LoRA 64, lr 1e-5 |

### The distillation hypothesis

The SDPO advantage `teacher_logprobs - student_logprobs` encodes: "how much more likely is this token when the model knows the fix?" If the grader says "weakness: no cost analysis" and suggests "add compute budget estimation," the teacher prompt includes this fix. Tokens related to cost analysis should have higher teacher logprobs, giving them positive SDPO advantage and reinforcing their generation.

The idea is that this token-level directional signal should be more informative than GRPO's scalar per-sample advantage.

## Runs & Results

### Full run catalog (17 runs in `runs/2026/2/SDPO/`)

| Run Path | Batches | Final Reward | Status | Notes |
|----------|---------|-------------|--------|-------|
| `2/SDPO/20` | 139 | -0.410 | Completed | First SDPO run; reward collapsed |
| `2/SDPO/21` | 107 | 0.337 | Completed | Adjusted lambda |
| `2/SDPO/22` | 29 | 0.500 | Short run | Exploratory |
| `2/SDPO/23` | 98 | -0.049 | Completed | Near-zero reward |
| `2/SDPO/24` | 24 | 0.472 | Short run | |
| `2/SDPO/24(2)` | 30 | 0.447 | Short run | |
| `2/SDPO/24(3)` | 76 | 0.069 | Completed | Near-zero |
| `2/SDPO/26` | 6 | 0.566 | Short run | |
| `2/SDPO/26(2)` | 5 | 0.338 | Short run | |
| `2/SDPO/26(2)(sanity check)` | 5 | 0.350 | Short run | GRPO-only control |
| `2/SDPO/26(3)` | 1 | 0.402 | Short run | |
| `2/SDPO/26(sanity check)` | 7 | 0.566 | Short run | GRPO-only control |
| `2/SDPO/26(test)` | 2 | 0.849 | Short run | Test harness |
| `2/SDPO/26(test2)` | 1 | 0.850 | Short run | Test harness |
| `2/SDPO/27(rich)` | 107 | 0.405 | Completed | Full run with richer feedback |
| `2/SDPO/27(rich)(sanity check)` | 107 | 0.596 | Completed | GRPO-only control: **better than SDPO** |
| `2/SDPO/28_recovery_A_warmstart_best214` | 21 | 0.673 | Short run | Warmstarted from bestversion batch 214 |

### Sanity checks (GRPO-only control)

The SDPO codebase was run with all distillation disabled (`use_sdpo=False`, `sdpo_grpo_mix_lambda=1.0`) to isolate SDPO's contribution:

| Run | SDPO reward | GRPO-only reward | Delta |
|-----|-------------|------------------|-------|
| 27(rich) vs 27(rich)(sanity check) | 0.405 | 0.596 | **SDPO hurt by -0.191** |
| 26 vs 26(sanity check) | 0.566 | 0.566 | No difference (both short) |

The 27(rich) sanity check is the most telling: 107 batches of GRPO-only in the same codebase achieved 0.596 reward, while the SDPO version achieved only 0.405. Self-distillation actively degraded performance.

### Warmstart experiment

The strongest SDPO attempt: take bestversion's best checkpoint (batch 214, reward 0.784) and continue training with SDPO loss. The hope was that starting from a strong policy would let the distillation signal push past the ceiling.

| Metric | Value |
|--------|-------|
| Run | `28_recovery_A_warmstart_best214` |
| Starting reward | 0.784 (from bestversion) |
| Final reward (21 batches) | 0.673 |
| Delta | **-0.111** |

SDPO training from the bestversion checkpoint caused **reward degradation**. The distillation signal pushed the policy away from the already-good configuration.

## What Failed and Why

### Root cause: self-distillation is redundant with GRPO advantages

The SDPO advantage `teacher_logprobs - student_logprobs` measures how much knowing the fix changes the model's distribution over tokens. But GRPO advantages already capture this information implicitly:

1. **Samples that address the weakness score higher.** Within a group of 8 plans for the same goal, plans that happen to cover cost analysis (using the earlier example) will get higher rubric scores and positive GRPO advantage. The GRPO gradient already reinforces these tokens.

2. **The teacher signal is noisy.** The "improved" prompt (original + feedback) produces logprobs over the entire sequence, not just the relevant tokens. Most tokens (e.g., formatting, common phrases) have similar logprobs with or without the feedback, creating a near-zero SDPO advantage on most tokens. The few tokens that change substantially are lost in the noise.

3. **Gradient interference.** When SDPO advantage disagrees with GRPO advantage (which happens frequently because they use different baselines), the mixed gradient points in a compromise direction that is worse than either signal alone.

### Evidence hierarchy

| Evidence | Finding |
|----------|---------|
| 20 runs, no improvement | No lambda value, architecture, or configuration outperformed bestversion |
| GRPO-only control (27 sanity check) | Same codebase without SDPO: reward 0.596 vs 0.405 with SDPO |
| Warmstart degradation | Starting from bestversion 0.784, SDPO pushed reward down to 0.673 |
| First run (20) | Reward collapsed to -0.410 with high SDPO weight |
| Run 23 | Near-zero reward (-0.049), 98 batches |

### Specific failure patterns

- **High lambda (heavy SDPO weight)**: Reward collapse (runs 20, 23). The distillation signal dominates and pushes the policy away from reward-optimizing behavior.
- **Low lambda (light SDPO weight)**: Equivalent to GRPO alone (sanity checks match). The SDPO signal is too weak to matter.
- **Medium lambda**: Worse than GRPO alone (run 27 vs sanity check). The SDPO signal adds noise without useful information.

There is no lambda sweet spot. The distillation signal is either too strong (causes collapse) or too weak (irrelevant) or in between (adds noise).

## What This Led To

1. **SDPO was abandoned after 20 runs.** This was the most thorough negative result in the project -- not a single configuration showed improvement.

2. **Confirmed simplicity principle.** The failure of SDPO reinforced what became the project's central finding: adding complexity to the loss function does not help. Single-stage GRPO with proper reward shaping is sufficient.

3. **Motivated the move to reward shaping.** Since modifying the loss function (SDPO) did not help, attention turned to modifying the reward signal. See [methods/reward_shaping.md](reward_shaping.md) for those experiments. (These also did not beat bestversion.)

4. **Established the sanity check methodology.** The GRPO-only control runs (disabling SDPO in the same codebase) became the template for isolating a method's contribution. This methodology was used in later experiments.

5. **The warmstart failure was particularly informative.** It showed that even starting from a known-good checkpoint, SDPO could not improve -- it could only degrade. This ruled out the hypothesis that SDPO just needs a better starting point.

## Lessons for Future Work

1. **Self-distillation adds no value when GRPO advantage already captures the same information.** The feedback-based teacher logprobs do not contain novel signal beyond what the reward function already provides. Future distillation approaches would need a fundamentally different source of information (e.g., a stronger external model, not the same model with feedback).

2. **Token-level advantage signals are risky.** SDPO operates at the token level, but the quality signal is at the plan level. This granularity mismatch means most tokens receive noisy or irrelevant SDPO advantages. Plan-level signals (like GRPO) are more robust.

3. **Always run a control.** The sanity check runs (GRPO-only in the SDPO codebase) were essential for understanding the failure. Without them, it would be unclear whether SDPO was slightly helping or actively hurting.

4. **20 runs is enough.** Lambda 0.1--0.45, multiple architectures, warmstart from the best checkpoint, sanity checks -- if the method worked at all, one of these configurations would have shown it. There is no reason to continue exploring SDPO variants for this task.

5. **Warmstarting is a strong test.** If a method cannot improve on a strong checkpoint (where the GRPO ceiling is already reached), it is unlikely to help from scratch. Use warmstart experiments as an efficient way to test new loss terms before running full training.

See: [methods/grpo_bestversion.md](grpo_bestversion.md) for the baseline that SDPO failed to beat | [findings/negative_results.md](../findings/negative_results.md) for the complete catalog of negative results
