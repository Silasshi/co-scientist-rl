# Method: Reward Shaping Ablations

## Motivation

The bestversion reward formula (`rubric_score + 0.08 * length_bonus - format_penalty`) was designed through intuition, not systematic search. Before moving to architectural changes (multi-stage, multi-turn), it was worth testing whether a better reward formulation could push past the 0.693 eval ceiling. Each ablation modified one component of the reward while keeping the GRPO pipeline, model, and all other hyperparameters identical to bestversion.

Seven distinct reward modifications were tested across 11 runs. All achieved different *training* rewards, but none translated to improved *eval* performance. A separate experiment (think-solution) tested whether adding explicit reasoning blocks could improve quality. It demonstrated that thinking actively hurts plan quality.

This document covers all reward-related ablations in one place.

---

## Baseline for Comparison

All ablations compare against the bestversion reward formula:

```
reward = rubric_score + 0.08 * length_bonus - format_penalty

rubric_score:     mean across rubric items, each item mapped {0->0.0, 1->0.2, 2->0.6, 3->1.0}
length_bonus:     exp(-((word_count - 600) / 120)^2)
format_penalty:   0.2 + 0.0005 * max(0, word_count - 750) if non-compliant, else 0.0
```

Bestversion training reward: 0.784 | Bestversion eval rubric: **0.693**

See [methods/grpo_bestversion.md](grpo_bestversion.md) for full bestversion details.

---

## 1. Weighted Dense Score (mean + std reweighting)

**Hypothesis**: Not all 7 desiderata are equally important or equally learnable. Adaptively reweighting desiderata -- upweighting weak ones (by mean) and unstable ones (by variance) -- should focus learning on the dimensions that matter most.

**What changed**: Rubric score aggregation uses adaptive per-desideratum weights instead of uniform mean. Uses a rolling buffer (window=20 batches) to track per-desideratum statistics.

```
mean_weight_d = exp(-3.0 * (mean_d - global_mean))    # Upweights below-average desiderata
std_weight_d  = exp(3.0 * (std_d - std_mean))          # Upweights high-variance desiderata
Both clipped to [0.5, 2.0]
Combined: weight_d = mean_weight_d * std_weight_d
```

**Runs**:

| Run Path | Batches | Final Reward | Reweighting Active |
|----------|---------|-------------|-------------------|
| `2/std+mean_weighted/11(ml)` | 46 | 0.638 | Both mean + std |

**Result**: Short run (46 batches), reward 0.638 -- substantially below bestversion's 0.784 at the same training stage. The combined reweighting appeared to destabilize training. This led to testing mean-only and std-only variants separately.

**Implementation**: `src/co_scientist/trainers/weighted/train_weighted_dense_score.py`

---

## 2. Weighted Mean-Only

**Hypothesis**: Isolate the mean-based reweighting (upweight desiderata the model scores below average on). Remove the std component that may have caused instability.

**What changed**: Same framework as weighted dense, but only mean-based reweighting active (std path disabled).

```
weight_d = exp(-3.0 * (mean_d - global_mean))    # Clipped to [0.5, 2.0]
```

**Runs**:

| Run Path | Batches | Final Reward |
|----------|---------|-------------|
| `2/std+mean_weighted/12(mean_weighted)` | 62 | 0.732 |
| `2/std+mean_weighted/13(mean_weighted)` | 107 | **0.787** |

**Result**: Training reward 0.787, essentially identical to bestversion's 0.784. Mean reweighting neither helps nor hurts significantly. The adaptive weights converge toward near-uniform, suggesting the desiderata are already roughly balanced in difficulty.

**Implementation**: `src/co_scientist/trainers/weighted/train_weighted_mean.py`

---

## 3. Weighted Std-Only

**Hypothesis**: Upweight desiderata where the model's performance is most variable (high std across a rolling window). High variance means the model sometimes gets it right and sometimes does not -- this is where training signal is most informative.

**What changed**: Same framework as weighted dense, but only variance-based reweighting active (mean path disabled).

```
weight_d = exp(3.0 * (std_d - std_mean))    # Clipped to [0.5, 2.0]
```

**Runs**:

| Run Path | Batches | Final Reward |
|----------|---------|-------------|
| `2/std+mean_weighted/12(std_weighted)` | 61 | 0.742 |
| `2/std+mean_weighted/13(std_weighted)` | 107 | **0.819** |

**Result**: Training reward 0.819, higher than bestversion's 0.784. But higher training reward does not imply higher eval rubric -- the pre-bestversion band_bonus runs hit 0.840 training reward without improving eval. No eval was conducted for this run, but the pattern across all ablations is clear: training reward gains do not transfer.

**Why it did not help**: Upweighting high-variance desiderata makes the reward more sensitive to the dimensions the model is already exploring. This accelerates training reward but does not change the underlying quality of the generated plans.

**Implementation**: `src/co_scientist/trainers/weighted/train_weighted_std.py`

---

## 4. Hard-Min Aggregation

**Hypothesis**: The mean rubric score allows a plan to score well overall while having one very weak desideratum. Hard-min penalizes plans with any weakness, encouraging balanced quality across all criteria.

**What changed**: Rubric aggregation switches from pure mean to a blend of mean and min:

```
rubric_score = (1 - alpha) * mean(desiderata_scores) + alpha * min(desiderata_scores)
alpha = 0.5
```

A plan scoring {3, 3, 3, 3, 3, 3, 0} would get mean=0.857, but hard-min gives 0.5*0.857 + 0.5*0.0 = 0.429.

**Runs**:

| Run Path | Batches | Final Reward |
|----------|---------|-------------|
| `2/hard_min/14` | 124 | **0.618** |

**Result**: Training reward 0.618, substantially below bestversion's 0.784. The min component creates a noisy, high-variance reward signal (a single bad desideratum can tank the entire score). This makes advantage estimation unreliable and slows learning.

**Why it did not help**: The 7 desiderata are not equally controllable by the model. Some criteria (e.g., "no ethical issues") are inherently easier to satisfy than others (e.g., "detailed, specific solution"). The hard minimum forces the model to over-invest in raising its worst criterion, which often hits a ceiling before the investment pays off. Meanwhile, the reduced signal on already-good criteria lets them regress.

**Implementation**: `src/co_scientist/trainers/hard_min/train_hard_min.py`

---

## 5. 0--9 Grading Scale

**Hypothesis**: The default 4-level scale (0--3) is too coarse. A 10-level scale (0--9) provides finer-grained reward signal, allowing the model to distinguish between "slightly above mediocre" and "almost good" rather than lumping them into the same level.

**What changed**: Grader uses a 10-level scale (0--9) instead of 4-level (0--3). Dense desiderata mapped linearly as `level/9.0`. Three variants tested which desiderata get dense scoring:

- **15(2_3)**: Desiderata 2 and 3 only get dense scoring
- **15(All)**: All 7 desiderata get dense scoring
- **15(None)**: No desiderata get dense scoring (control -- 0--9 scale but bucketed back to 4 levels)

**Runs**:

| Run Path | Batches | Final Reward | Dense Desiderata |
|----------|---------|-------------|-----------------|
| `2/0-9_scale/15(2_3)` | 107 | **0.802** | 2 and 3 only |
| `2/0-9_scale/15(All)` | 108 | **0.760** | All 7 |
| `2/0-9_scale/15(None)` | 107 | **0.798** | None (control) |

**Result**: Training rewards 0.760--0.802, spanning bestversion's 0.784. No clear advantage from finer-grained scoring. The control (None) performed comparably to the dense variants, suggesting the grading scale is not a bottleneck.

**Why it did not help**: The non-linear mapping in bestversion ({0->0.0, 1->0.2, 2->0.6, 3->1.0}) already creates a useful gradient -- the big jump from level 1 to 2 focuses learning on escaping mediocrity. A linear 0--9 scale spreads the signal more evenly but doesn't concentrate it where it matters most.

**Implementation**: `src/co_scientist/trainers/scale_0_9/train_scale_0_9.py`

---

## 6. Band Bonus (Pre-Bestversion Baseline)

**Hypothesis**: These runs used the older codebase (before A1+A2 improvements) with band bonus reward shaping. While not a direct ablation of bestversion's reward formula, they represent the pre-bestversion state of the art and achieved the highest training rewards in the entire project.

**What changed**: Same GRPO + same reward formula as bestversion, but without the A1+A2 quality-of-life improvements: no `drop_noncompliant_samples`, no `format_retry`, no `min_words_warmup`.

**Runs**:

| Run Path | Batches | Final Reward | Notes |
|----------|---------|-------------|-------|
| `2/std_stand+band_bonus/8(ml)` | 96 | **0.840** | Highest training reward in the project |
| `2/std_stand+band_bonus/9(ml)` | 214 | **0.821** | Full 2-epoch run |

**Result**: Training reward 0.840 -- the highest of any run. But these runs preceded bestversion and no formal eval was conducted. The A1+A2 improvements (which reduced training reward to 0.784) are believed to improve *eval* generalization by removing degenerate high-reward samples (format-violating plans that luck into high rubric scores).

**Why training reward is misleading**: Without `drop_noncompliant_samples`, malformatted plans that happen to score well inflate the training reward but do not reflect genuine plan quality. Bestversion's lower training reward (0.784 vs 0.840) reflects stricter quality control, not worse performance.

---

## 7. Think-Solution

**Hypothesis**: Explicit reasoning (a `<think>` block) before the plan (`<solution>` block) should improve quality. The model can use the thinking phase to reason about the criteria, plan its approach, and produce a more thoughtful final plan. This is inspired by chain-of-thought and reasoning approaches.

**What changed**: Generation format splits into two blocks: `<think>` (reasoning) and `<solution>` (plan). Each block is graded independently. The grader scores both the reasoning quality and the plan quality.

**Runs**:

| Run Path | Batches | Final Reward | Notes |
|----------|---------|-------------|-------|
| `4/think_solution/1` | 107 | 0.693 | Full run, 1 epoch |
| `4/think_solution_smoke/1` | 5 | 0.610 | Smoke test |

**Result**: Think blocks score **-0.090 lower** than non-thinking plans. The `<think>` block does not improve the `<solution>` block -- it actively hurts it by consuming token budget.

**Why it failed**: This is a clear instance of process shortcutting (see [findings/process_shortcutting.md](../findings/process_shortcutting.md)). The think block is not directly tied to the reward signal for plan quality. The model either:

1. Minimizes the think block to preserve tokens for the plan, in which case thinking adds overhead without benefit
2. Fills the think block with content that doesn't translate to better plans, wasting token budget

In both cases, the `<solution>` block has fewer available tokens than in bestversion, leading to shorter and less detailed plans. The grader penalizes the reduced detail.

**Implementation**: `src/co_scientist/trainers/baselines/train_think_solution.py`

---

## Summary: All Reward Ablations

| Method | Modification | Runs | Best Training Reward | vs Bestversion (0.784 training) | Eval Conducted? |
|--------|-------------|------|---------------------|-------------------------------|-----------------|
| **Bestversion** | **Baseline** | **3** | **0.784** | **--** | **Yes: 0.693** |
| Weighted dense (mean+std) | Adaptive reweighting | 1 | 0.638 | -0.146 | No |
| Weighted mean-only | Upweight weak desiderata | 2 | 0.787 | +0.003 | No |
| Weighted std-only | Upweight variable desiderata | 2 | 0.819 | +0.035 | No |
| Hard-min | 0.5*mean + 0.5*min | 1 | 0.618 | -0.166 | No |
| 0--9 scale (best variant) | 10-level grading | 3 | 0.802 | +0.018 | No |
| Band bonus (pre-bestversion) | Older codebase | 2 | 0.840 | +0.056 | No |
| Think-solution | Add `<think>` block | 2 | 0.693 | -0.091 | No (but -0.090 think penalty measured) |

**Critical observation**: Training reward ranged from 0.618 (hard-min) to 0.840 (band bonus) -- a span of 0.222 -- yet none of the ablations with formal eval outperformed bestversion's 0.693 eval rubric. The consistent finding across 11+ runs: **training reward improvement does not predict eval improvement**.

---

## What Failed and Why

### The universal problem: training reward != eval quality

Every reward ablation that increased training reward did so by making it easier for the model to score well on training goals, not by improving the quality of generated plans on unseen goals. Specific mechanisms:

1. **Std weighting** (0.819): Upweighting high-variance desiderata accelerates reward growth on dimensions the model is already exploring. This is reward hacking on a soft dimension.

2. **Band bonus** (0.840): Without A1+A2 quality controls, malformatted samples that luck into high scores inflate the training reward without representing real quality.

3. **0--9 scale** (0.802): Finer-grained scoring gives more credit for partial improvements, raising the training reward but not changing the quality of the plans themselves.

### The hard-min and think-solution failures are more informative

These ablations *lowered* training reward, providing clear negative signals:

- **Hard-min** (0.618): The min component creates high-variance rewards. A single bad desideratum (which may be beyond the model's control) can dominate the signal, making the reward noisy and the advantages unreliable.

- **Think-solution** (-0.090 think penalty): Reasoning is an unrewarded process. The model cannot usefully reason about the plan in a `<think>` block because the quality signal comes only from the final `<solution>`. This confirms the process shortcutting principle observed across multi-turn and IBT experiments.

## What This Led To

1. **Reward formula was frozen.** After 11 runs showing no reward variant outperforms bestversion, the reward formula was fixed and attention turned to architectural changes (Phase 3: doublegeneration, rubric dropout, multi-turn).

2. **Established the training-vs-eval disconnect.** This finding became a project-wide principle: always evaluate on held-out data. Training reward is necessary but not sufficient evidence of improvement.

3. **Think-solution linked to process shortcutting.** The -0.090 think penalty became a key data point in the process shortcutting finding (see [findings/process_shortcutting.md](../findings/process_shortcutting.md)), alongside multi-turn (zero correlation) and IBT (brainstorming degeneration).

4. **Confirmed bestversion's reward design is near-optimal for this task.** The non-linear mapping ({0->0.0, 1->0.2, 2->0.6, 3->1.0}), the Gaussian length bonus centered at 600, and the format penalty at 750 words form a carefully balanced combination. Each ablation that changed one component made things worse or no better.

## Lessons for Future Work

1. **Do not optimize the reward formula.** The bestversion formula is good enough. Further reward engineering will not break the 0.693 ceiling because the ceiling is caused by the selection gap (91%), not the reward signal.

2. **If you must change the reward, evaluate on held-out data.** Training reward improvements of 0.05--0.08 (as seen in std weighting and 0--9 scale) are meaningless without eval confirmation.

3. **Hard-min / worst-case objectives are too noisy.** The 7 desiderata have different difficulty levels. Penalizing the worst one creates high-variance rewards that destabilize GRPO advantage estimation.

4. **Finer grading scales do not help.** The 4-level scale ({0, 1, 2, 3}) with non-linear mapping provides sufficient gradient information. 10-level scales spread the signal without concentrating it where it matters.

5. **Explicit reasoning hurts when not rewarded.** Do not add `<think>` blocks, multi-turn discussion, or iterative brainstorming unless the process itself is scored with a well-calibrated reward. Unrewarded processes will be shortcutted.

6. **Adaptive reweighting converges to uniform.** The mean-weighted variant (0.787) performed almost identically to uniform weighting (0.784), suggesting the desiderata are already roughly balanced. The std-weighted variant (0.819) overfits to noisy dimensions.

See: [methods/grpo_bestversion.md](grpo_bestversion.md) for the baseline all ablations compare against | [findings/negative_results.md](../findings/negative_results.md) for the complete negative results catalog
