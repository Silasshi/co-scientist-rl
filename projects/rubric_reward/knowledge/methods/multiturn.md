# Method: Multi-Turn Discussion (V1 through V4)

## Motivation

The hypothesis was that multi-turn collaborative discussion before plan generation could improve research plan quality. A real researcher would not produce a plan in one shot -- they would discuss ideas, receive feedback, refine their thinking, and then write. If the model could be trained to use discussion productively, it might break the bestversion ceiling (0.693 eval rubric).

The approach drew from process reward model (PRM) literature: score each intermediate turn of the discussion, not just the final plan. Combined with on-policy distillation (OPD) using evaluator hints, this would provide per-turn gradient signal to shape the discussion.

Four versions were attempted across March--April 2026, each fixing failures from the previous version. None succeeded.

---

## Pipeline

### Core Architecture (All Versions)

```
For each batch of 8 goals x 4 samples = 32 conversations:

  For each conversation:
    1. Collaborator (policy model) and Researcher (external model) discuss
    2. Evaluator (PRM) scores each collaborator turn
    3. Researcher requests a plan when ready
    4. Collaborator generates plan in <solution> tags
    5. Canonical grader scores final plan with 7-desiderata rubric

  Training:
    - Per-turn: PRM score + OPD hint --> per-turn advantages
    - Final plan: GRPO advantages across K=4 samples per goal
    - Combined PPO step
```

### Roles

| Role | Model | Purpose |
|------|-------|---------|
| Collaborator (policy) | Qwen3-30B-A3B + LoRA rank 64 | Generates discussion and plans (the model being trained) |
| Researcher | Gemini 2.0 Flash (V2--V4) or fixed prompt (V1) | Drives conversation, asks questions, requests plan |
| Evaluator (PRM) | Varied by version (see below) | Scores each collaborator turn for quality |
| Canonical Grader | Qwen3-30B-A3B base (no LoRA) | 7-desiderata rubric grading of final plan |

### Version Evolution

| Area | V1 | V2 | V3 | V4 |
|------|----|----|----|----|
| Architecture | Batch-frozen | Wave-based | Batch-frozen | Wave-based |
| PRM model | Qwen3-235B | Merged into researcher | Gemini Flash | Gemini Flash |
| PRM scale | Binary {0,1} | Merged (-3 to +3) | -3 to +3 | Ternary {-1,0,+1} |
| PRM parsing | Single regex | Single regex | Single regex | 4-pattern cascade |
| max_policy_turns | 10 | 7 | 7 | 15 |
| Discussion limits | Hard 1--5 | Hard 1--5 | Hard 1--5 | Free-form |
| Stuck detection | None | None | None | threshold=8 |
| Hint design | Only on PRM=+1 | Only on score != 0 | May output "none" | Always-hint |
| Checkpoint | save_every=20 | save_every=20 | save_every=5 | save_every=5 |

---

## Runs & Results

### V1: Binary PRM (run4, 54 batches)

| Metric | Value |
|--------|-------|
| Run path | `runs/2026/3/multiturn/run4` |
| Batches | 54 (also logged as 119 in metrics, includes restarts) |
| Best batch rubric | 0.624 |
| PRM model | Qwen3-235B on OpenRouter |
| PRM score distribution | 90%+ positive |
| Learning | None -- flat training curve |

**Failure**: Binary PRM scores were 90%+ positive. After batch whitening, advantages were near-zero. No gradient signal reached the policy.

### V2: Merged Evaluator (run1, 72 batches)

| Metric | Value |
|--------|-------|
| Run path | `runs/2026/3/multiturn_v2/run1` |
| Batches | 72 |
| Best batch rubric | 0.754 |
| Empty response rate | 71% |
| Discussion collapse | 0.7 turns --> 0.0 (immediate) |
| Zero-discussion rubric | 0.692 (matched bestversion) |

**Failure**: Merging the evaluator and researcher into one model (Gemini Flash doing both response + scoring) caused the model to return empty responses 71% of the time on plan turns. The PRM scores encoded turn type (discussion=2, plan=0), not quality. The collaborator learned to skip discussion immediately -- and its zero-discussion rubric (0.692) essentially matched bestversion (0.693).

### V3: PRM Parsing Bug (run1, 3 batches)

| Metric | Value |
|--------|-------|
| Run path | `runs/2026/3/multiturn_v3/run1` |
| Batches | 3 (crashed) |
| PRM distribution | {0: 85, 2: 86} -- bimodal, missing negatives |
| Negative scores recovered | ~250 (all lost as 0) |
| Discussion turns | 5.3 (stable before crash) |

**Failure**: The PRM parsing regex expected `<score>N</score>` but the evaluator often produced `-1</score>` (missing opening tag). All ~250 negative scores were silently mapped to 0, destroying the penalty signal. Additionally, `asyncio.run()` inside `asyncio.run()` caused a checkpoint crash at batch 3. Discovered via investigation in `analysis/prm_investigation/`.

### V4: Wave-Based + Ternary PRM (5 runs, ~0 progress)

| Metric | Value |
|--------|-------|
| Run paths | `runs/2026/3/multiturn_v4/4/` and related |
| Total runs | 5 (including smoke test) |
| PRM scale | Ternary {-1, 0, +1} |
| PRM parsing | 4-pattern fallback cascade |
| Discussion collapse | 5.6 turns --> 0.0 (gradual, over ~10 batches) |
| Best batch rubric | -- (no consistent improvement) |

V4 incorporated every fix from V1--V3:
- **Ternary PRM** {-1, 0, +1} with few-shot examples and anchored questions
- **Separated evaluator** (dedicated Gemini Flash call, not merged with researcher)
- **Robust 4-pattern parsing cascade**: `<score>N</score>` --> `N</score>` --> `<score>N` --> `Score: N`
- **Always-hint design**: Summarize for +1, correct for 0/-1, never "none"
- **Wave-based updates**: Policy updates between waves (turns), not just end-of-batch
- **Free-form discussion**: No hard min/max discussion turn limits
- **max_policy_turns=15** with stuck detection at 8 turns

Despite all fixes, no learning emerged. The collaborator still learned to collapse discussion turns to zero.

---

## What Failed and Why

### The Fundamental Finding: Discussion Does Not Help

Across ~1400+ scored conversations from V2--V4, discussion quality showed no meaningful correlation with final plan rubric score:

| Version | Conversations | Pearson r | Interpretation |
|---------|--------------|-----------|----------------|
| V2 | 369 | -0.115 | Weak negative |
| V3 | 191 | -0.335 | Moderate negative |
| V4 | 832 | 0.052 | Near zero |

Within-goal comparisons (same goal, more vs fewer discussion turns):

| Version | More disc --> better | More disc --> worse | Tie |
|---------|---------------------|---------------------|-----|
| V2 | 30% | 32% | 38% |
| V3 | 19% | 81% | 0% |
| V4 | 18% | 28% | 54% |

In V3, more discussion was 4x more likely to hurt than help. In V2, zero-discussion conversations achieved rubric 0.692 -- matching bestversion's eval score of 0.693.

### Process Shortcutting

All versions exhibit the model learning to skip discussion:

| Version | Discussion collapse pattern | Speed |
|---------|-----------------------------|-------|
| V2 | 0.7 turns --> 0.0 | Immediate (batch 0 already low) |
| V3 | 5.3 turns --> stable | Crashed at batch 3 (insufficient data) |
| V4 | 5.6 turns --> 0.0 | Gradual (over ~10 batches) |

The model rationally optimizes: discussion turns are scored by the PRM but the terminal reward (rubric) is what GRPO optimizes. Since discussion does not improve rubric score, the GRPO gradient drives the model to eliminate discussion and allocate all tokens to the plan itself.

This is not a PRM calibration issue -- it is a genuine finding that **multi-turn discussion provides no benefit to final plan quality** as measured by the 7-desiderata rubric. The rubric evaluates the plan in isolation; it does not measure whether the plan addresses topics raised during discussion.

### Why Each Fix Was Insufficient

| Fix Applied | Problem It Solved | Why It Wasn't Enough |
|-------------|-------------------|----------------------|
| Ternary PRM (V4) | Binary PRM too coarse (V1) | PRM signal orthogonal to rubric |
| Separated evaluator (V3+) | Merged evaluator returned empty (V2) | Separating roles didn't make discussion useful |
| Parsing cascade (V4) | Negative scores lost (V3) | Correct parsing still showed no learning |
| Wave-based updates (V2, V4) | Batch-frozen was off-policy (V1) | More frequent updates didn't help |
| Always-hint OPD (V4) | Some turns lacked training signal | OPD guided toward irrelevant discussion quality |
| Free-form discussion (V4) | Hard limits were arbitrary | Model chose 0 turns when free to decide |
| max_turns=15 (V4) | Conversations hit ceiling (V3) | More turns available but model used fewer |

---

## What This Led To

1. **Confirmed that process shortcutting is universal.** Multi-turn discussion is the clearest example: discussion is an unrewarded intermediate process, and the model eliminates it. This pattern also appeared in think-solution (-0.090 impact), IBT (brainstorming degeneration), and doublegeneration Stage 2 (28% regression rate). See [findings/process_shortcutting.md](../findings/process_shortcutting.md) for the broader pattern.

2. **Confirmed that PRM calibration is not the bottleneck.** Binary, merged, 7-level, and ternary PRM designs were all tried. The issue is not the PRM's scoring accuracy -- it is that discussion quality is orthogonal to plan quality in this task and evaluation setup.

3. **Raised a fundamental question about the rubric.** The 7-desiderata rubric evaluates the plan in isolation. A rubric that measured whether the plan addressed specific concerns raised during discussion might show a discussion benefit. But no such rubric exists, and creating one would change the task definition.

4. **Reinforced the pivot to selection.** If multi-turn discussion cannot improve generation quality, and training improvements are exhausted, the remaining path is inference-time selection from multiple single-turn generations.

---

## Lessons for Future Work

1. **Do not add unrewarded intermediate steps.** Any process not tied to the terminal reward will be eliminated by GRPO. Discussion, thinking, iterative refinement -- all are pruned unless they directly improve the scored output.

2. **Verify the mechanism before scaling.** None of the V1--V4 designs tested whether discussion actually helps plan quality in a non-RL setting (e.g., prompting a frozen model with vs without discussion). This should have been the first experiment.

3. **PRM design cannot fix a task-level mismatch.** If the intermediate process does not help the terminal output, no PRM design will make it trainable. The PRM scores discussion quality, but discussion quality is uncorrelated with plan quality (r ~ 0).

4. **Free-form observation is diagnostic gold.** V4's free-form discussion (no hard min/max) provided the clearest evidence: given the choice, the trained model chose zero discussion turns. This is more informative than hard limits that force discussion.

5. **Process shortcutting is a research finding, not a failure.** The consistent elimination of unrewarded processes across multiple methods (multi-turn, think-solution, IBT) is a genuine contribution to understanding reward-based optimization of LLMs.

---

## Related Documents

- [Process Shortcutting](../findings/process_shortcutting.md) -- The universal pattern across methods
- [Phase 3 Details](../phases/phase3_doublegeneration.md) -- Full phase overview including multi-turn
- [Negative Results](../findings/negative_results.md) -- Comprehensive failure catalog
- [Multi-turn Discussion Analysis](../../analysis/reports/multiturn_discussion_analysis.md) -- Detailed statistical analysis
- [V4 Design Document](../../paper/design/method_pipeline_v4_design.md) -- Full V4 architecture specification
- [Experiment Catalog](../EXPERIMENT_CATALOG.md) -- Master experiment reference

## Source Code

| File | Purpose |
|------|---------|
| `src/co_scientist/trainers/multiturn/train_multiturn_v4.py` | V4 training pipeline (1269 lines) |
| `src/co_scientist/trainers/multiturn/openrouter_client.py` | OpenRouter HTTP client for Gemini Flash |
| `src/co_scientist/trainers/multiturn/eval_multiturn.py` | Multi-turn evaluation script |
| `analysis/prm_investigation/` | PRM prompt investigation and V3 parsing bug discovery |
| `paper/design/method_pipeline_v4_design.md` | V4 design document with full failure history |

## Run Paths

| Version | Path | Status |
|---------|------|--------|
| V1 | `runs/2026/3/multiturn/run4` | Complete (no learning) |
| V2 | `runs/2026/3/multiturn_v2/run1` | Complete (merged PRM failed) |
| V3 | `runs/2026/3/multiturn_v3/run1` | Complete (parsing bug + crash) |
| V4 | `runs/2026/3/multiturn_v4/4/` | Complete (no learning despite all fixes) |
| V4 smoke | `runs/2026/3/multiturn_v4/smoke_test` | Test run |
