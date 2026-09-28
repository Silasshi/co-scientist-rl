# Method: GRPO Bestversion

## Motivation

Phase 1 (January 2026) tested three model architectures (Qwen3-30B-A3B, gpt-oss-20b, Llama-3.1-8B) across 16 exploratory runs and established two facts: Qwen3-30B-A3B was the best model for research plan generation, and dense rubric scoring outperformed sparse binary scoring on this task. The first major training run (`success_dense_score`, 189 batches, reward 0.587) proved GRPO could learn, but left substantial headroom below the reference solution score of 0.86.

Bestversion was the result of systematically improving the Phase 1 GRPO baseline with two modifications (internally labeled A1 and A2): LoRA rank 64 with a properly calibrated reward formula including length bonus and format penalty. The goal was to establish the strongest possible single-stage GRPO baseline before attempting more complex multi-stage or multi-loss approaches.

It succeeded: bestversion achieved eval rubric **0.693** after 2 epochs on the ML split. This became the ceiling that no subsequent method -- across 60+ additional runs -- has been able to break.

## Pipeline

The training loop has three phases per batch: **sample, grade, update**.

### Step-by-step loop

```
For each batch of 64 goals:
  1. Snapshot current LoRA weights, create sampler
  2. For each goal: generate 8 plans (async, temperature=1.0, max_tokens=2048)
  3. For each plan: launch grader (async, temperature=0.0, max_tokens=8192)
     - Grader prompt: scenario + rubric items + reference solution + proposed plan
     - Grader outputs structured XML with per-item desiderata levels (0-3)
  4. Parse grader XML -> per-item desiderata scores -> rubric_score -> final reward
  5. GRPO advantage: per-goal group of 8 samples
       advantage_i = reward_i - mean(rewards in group)
  6. Build training datums (token sequences, old logprobs, advantages)
  7. PPO-style update: forward_backward(loss_fn="ppo", clip_eps=0.2)
     + optimizer step (Adam, lr=1e-5)
```

### Reward formula

```
reward = rubric_score + 0.08 * length_bonus - format_penalty
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| rubric_score | Mean across all rubric items. Per-item: mean of 7 desiderata, mapped {0->0.0, 1->0.2, 2->0.6, 3->1.0} | Core quality signal. Non-linear mapping creates a big jump from level 1 to 2, encouraging the model to push past mediocre scores. |
| length_bonus | `exp(-((word_count - 600) / 120)^2)` -- Gaussian centered at 600 words, sigma=120 | Guides plan length toward an informative but concise range. Coefficient 0.08 keeps it subordinate to rubric score. |
| format_penalty | `0.2 + 0.0005 * max(0, word_count - 750)` if non-compliant (missing `<solution>` tags or word count > 750); 0.0 otherwise | Enforces structured output. Harsh enough to achieve ~99% compliance, light enough not to dominate gradient. |

Format compliance requires `<solution>` tags and word count <= 750.

### GRPO advantage estimation

Group Relative Policy Optimization (GRPO) computes advantages within each goal's group of 8 samples. For a goal with rewards `[r_1, ..., r_8]`:

```
advantage_i = r_i - mean(r_1, ..., r_8)
```

No value function is needed -- the group mean serves as the baseline. This makes GRPO simpler and more stable than PPO with a learned critic. Group size 8 provides a reliable mean estimate while keeping compute manageable (8 generations + 8 gradings per goal per batch).

### PPO clipped update

The policy gradient uses the PPO clipped objective with `clip_eps=0.2`:

```
ratio = exp(logprob_new - logprob_old)
clipped_ratio = clip(ratio, 1 - 0.2, 1 + 0.2)
loss = -min(ratio * advantage, clipped_ratio * advantage)
```

Optimizer: Adam with lr=1e-5, applied to LoRA rank-64 parameters only.

### Key design choices (A1 + A2 improvements)

- **LoRA rank 64**: Sufficient capacity for the task without overfitting
- **Group size 8**: Stable advantage estimation with manageable compute
- **Temperature 1.0 for generation**: Ensures diverse samples within each group -- critical for GRPO to have meaningful variance to learn from
- **Temperature 0 for grading**: Deterministic scoring eliminates noise in the reward signal
- **Drop non-compliant samples**: Removed from advantage computation to avoid poisoning the gradient
- **Format retry**: Failed format attempts get one retry before being dropped
- **Min words warmup**: Early batches have a relaxed minimum word requirement

## Runs & Results

### Primary bestversion runs

| Run Path | Batches | Final Reward | Eval Rubric | Domain | Notes |
|----------|---------|-------------|-------------|--------|-------|
| `2/withA1,A2/2(ml)` | 215 | 0.784 | **0.693** | ML | **Primary baseline. Nothing has beaten this.** |
| `2/withA1,A2/3(arxiv)` | 204 | 0.726 | -- | arXiv | Cross-domain transfer |
| `2/withA1,A2/3(pubmed)` | 200 | 0.853 | -- | PubMed | Higher reward (easier domain) |

### Async variant (relaxed length)

| Run Path | Batches | Final Reward | Notes |
|----------|---------|-------------|-------|
| `4/best_ver_async/2` | 29 | 0.560 | Short run |
| `4/best_ver_async/3` | 27 | 0.596 | Short run |
| `4/best_ver_async/4` | 106 | 0.778 | Best async run; 850 max words, 3072 max tokens |

### Pre-bestversion baseline (older codebase, before A1+A2)

| Run Path | Batches | Final Reward | Notes |
|----------|---------|-------------|-------|
| `2/std_stand+band_bonus/8(ml)` | 96 | 0.840 | Highest raw training reward across all runs |
| `2/std_stand+band_bonus/9(ml)` | 214 | 0.821 | Band bonus variant |

These pre-bestversion runs achieved higher *training* rewards than bestversion (0.84 vs 0.78), but they lacked the A1+A2 improvements (drop_noncompliant_samples, format_retry, min_words_warmup). No eval was conducted, so their actual generalization is unknown. The lesson: training reward is not a reliable proxy for eval performance.

### Training dynamics (run 2(ml))

| Metric | Value |
|--------|-------|
| Reward trajectory | 0.65 -> 0.78--0.83 (peaks 0.83 at batch 139) |
| Batches to convergence | ~140 |
| Total batches (2 epochs) | 214 |
| Format compliance | ~99% (from ~90% at start) |
| Format penalty (mean) | 0.004 (negligible after training) |
| Avg rubric (batch 107) | 0.443 |
| Avg reward (batch 107) | 0.527 |

### Eval details (checkpoint batch 214)

| Metric | Value |
|--------|-------|
| Rubric mean | **0.6926** |
| Reward mean | 0.7498 |
| Format penalty | 0.0013 |
| Valid samples | 5,120 / 5,120 (100%) |

## What Failed and Why

**Bestversion itself did not fail -- it is the project's strongest result.** The failure is that no method surpassed it. The 0.693 ceiling proved robust across 60+ subsequent runs testing 10+ approaches.

### Why the ceiling exists

1. **Selection gap dominates (91%)**: Oracle best-of-8 achieves 0.85 -- nearly matching reference solutions (0.86). The model CAN produce excellent plans. It just produces 8 diverse plans per goal and most don't match the rubric's expected methodology. The gap from 0.70 (mean) to 0.85 (oracle) is selection, not capability.

2. **Method-specific rubric items**: 80.6% of model-low scoring items occur when the reference scores HIGH on the same criterion. The rubric checks for the reference's specific methodology (e.g., "uses federated learning"). The model proposes different valid approaches that score low on method-specific criteria. This is an evaluation bias, not a generation weakness.

3. **Reward misalignment**: The reward formula penalizes length (>750 words) but the grader values detail. GPT-5.4 achieves rubric 0.843 without any RL training, just by being a more capable model generating longer, more detailed plans. The reward function may suppress the very quality the grader wants.

### Why nothing beat it

Every alternative method introduced new failure modes without compensating benefits:

| Method | Failure mode | Reference |
|--------|-------------|-----------|
| SDPO (20 runs) | Self-distillation redundant with GRPO advantages | [methods/sdpo.md](sdpo.md) |
| Doublegeneration | Stage 2 OOD collapse after batch ~130 | [findings/negative_results.md](../findings/negative_results.md) |
| Rubric dropout | Splitting signal between visible/hidden hurt training | [findings/negative_results.md](../findings/negative_results.md) |
| Multi-turn V1--V4 | Discussion shows zero correlation with rubric (r ~ 0) | [findings/negative_results.md](../findings/negative_results.md) |
| Think-solution | Thinking hurts plan quality (-0.090) | [findings/process_shortcutting.md](../findings/process_shortcutting.md) |
| Reward shaping (6 variants) | Higher training reward does not predict higher eval | [methods/reward_shaping.md](reward_shaping.md) |

## What This Led To

1. **Bestversion became the universal comparison point.** All subsequent methods report their eval rubric relative to 0.693.

2. **Phase 2 ablations**: Before moving to architectural changes, 11 reward shaping runs and 20 SDPO runs tested whether modifying the reward or loss function could help. None did. See [methods/reward_shaping.md](reward_shaping.md) and [methods/sdpo.md](sdpo.md).

3. **Phase 3 multi-stage methods**: Doublegeneration (two-stage refinement) and rubric dropout both tried to break the ceiling through architectural complexity. Both performed worse.

4. **Gap analysis**: The realization that oracle best-of-8 (0.85) nearly matches reference (0.86) redirected attention from generation improvement to selection improvement. The self-selector approach targets the 91% selection gap -- the most promising remaining direction.

5. **Cross-domain transfer**: The ArXiv (0.726 reward) and PubMed (0.853 reward) runs showed bestversion transfers to other domains. PubMed's higher reward suggests easier grading in that domain.

## Lessons for Future Work

1. **The pipeline works.** Single-stage GRPO with dense rubric scoring is a proven, reliable training method. Do not add complexity unless you have strong evidence it helps.

2. **Group size 8 is the right balance.** Smaller groups give noisier advantages. Larger groups increase compute without proportional benefit.

3. **Training reward != eval performance.** The pre-bestversion baseline hit 0.84 training reward, higher than bestversion's 0.78. Always evaluate on held-out data.

4. **The 0.693 ceiling is a selection problem.** Training the generator harder will not break it. The model already generates 0.85-quality plans -- it just can't tell which ones are good. Focus on inference-time selection (best-of-N with a trained reranker).

5. **Reward misalignment matters.** The length penalty (>750 words) may actively suppress quality. GPT-5.4 scores 0.843 without any RL, partly by generating longer, more detailed plans. If revisiting reward design, consider relaxing or removing the length penalty.

6. **Format penalty is essential.** The 0.2 + 0.0005*excess penalty achieves ~99% compliance. Without it, the grader produces unreliable scores on malformatted plans.

See: [findings/what_works.md](../findings/what_works.md) for the full bestversion recipe | [EXPERIMENT_CATALOG.md](../EXPERIMENT_CATALOG.md) for the complete experiment timeline
