# Co-Scientist Project: Comprehensive Conclusion

## Project Objective

Train an LLM (Qwen/Qwen3-30B-A3B, LoRA rank 64) to generate structured research plans given a (research_goal, research_target) pair. A grader model scores outputs using a 7-desiderata rubric per criterion. The goal is to maximize rubric score on held-out test goals.

---

## Dataset

**Source:** `facebook/research-plan-gen` (ML split)
- Train split: ~6,848 goals (107 batches of 64)
- Test split: 685 goals (10 batches of 64 + remainder)
- Each goal has: Goal text, Rubric items (method-specific criteria), Reference solution

---

## Unified Reward Formula

All training and evaluation pipelines use the same reward computation:

```
rubric_score_per_item = mean of 7 desiderata levels, mapped {0→0.0, 1→0.2, 2→0.6, 3→1.0}
rubric_score = mean across all rubric items

length_bonus = exp(-((word_count - 600) / 120)^2)     # Gaussian centered at 600 words
format_penalty = 0.0 if compliant else 0.2 + 0.0005 * max(0, word_count - 750)
final_reward = rubric_score + 0.08 * length_bonus - format_penalty
```

Format compliance requires `<solution>` tags and word count <= 750.

---

## 7 General Desiderata

Every rubric item is scored on these 7 desiderata (levels 0-3):

1. **HANDLES ALL CRITERIA** — Does the plan satisfy what the criterion requires?
2. **DETAILED, SPECIFIC SOLUTION** — Are implementation details concrete, not vague?
3. **NO OVERLOOKED FLAWS** — Are there important weaknesses that undermine this?
4. **WELL JUSTIFIED RATIONALE** — Is the approach motivated and justified?
5. **COST AND EFFORT EFFICIENT** — Is the approach efficient, without unnecessary complexity?
6. **NO ETHICAL ISSUES** — Are there potential negative consequences?
7. **CONSISTENT WITH OVERALL PLAN** — Does this part cohere with the rest?

---

## Phase 1: Paper Reproduction

Reproduced the baseline method from the reference paper (arXiv:2512.xxxxx). Strict reproduction did NOT improve task success rate in this setting.

---

## Phase 2: Bestversion (Single-Stage GRPO)

### Pipeline (`src/co_scientist/trainers/baselines/best_ver.py`)

```
For each batch of 64 goals:
  1. Save current LoRA weights → create sampler
  2. For each goal: generate 8 plans (async, temp=1.0, max_tokens=2048)
  3. For each plan: launch grader (async, temp=0.0, max_tokens=8192)
     - Grader sees: scenario + rubric items + reference solution + proposed plan
     - Grader outputs: structured XML with per-item desiderata levels
  4. Parse grader XML → rubric_score → final_reward
  5. GRPO advantages: per-goal group of 8, advantage_i = reward_i - mean(group)
  6. Build training datums (tokens, logprobs, advantages)
  7. forward_backward(loss_fn="ppo", clip_eps=0.2) + optim_step(Adam, lr=1e-5)
```

**Key design choices (a1 + a2 improvements):**
- LoRA rank 64
- Group size 8 for GRPO
- Length bonus Gaussian (target 600, sigma 120)
- Format penalty with hard cap at 750 words

### Training Results

- Run: `runs/2026/2/withA1,A2/2(ml)/`
- 215 batches (2 epochs)
- Training reward: 0.65 → 0.78-0.83 (peaks 0.83 at batch 139)
- Format compliance: ~99%

### Eval Result

| Metric | Value |
|--------|-------|
| **Rubric mean** | **0.6926** |
| Reward mean | 0.7498 |
| Format penalty | 0.0013 |
| Valid samples | 5,120 / 5,120 |
| Checkpoint | batch 214 (end of epoch 2) |

**Status:** Performance ceiling reached. Further training extensions failed to surpass this.

---

## Phase 2b: Ablations (Reward, Loss, Data, Scale)

Before moving to architectural changes (Phase 3+), several ablations were run to test whether modifying the reward signal, loss function, grading scale, or data domain could improve over bestversion. All used the same single-stage GRPO pipeline.

### Sanity Checks

**GSM8K** (`train_baseline_gsm8k.py`, runs: `1/25(gsm8k_…)`, `1/25(rl_loop_gsm8k)`)
Completely different task: binary math reward (correct/incorrect). Validates the GRPO training pipeline works on a known-learnable RL task.

**Paper-original sparse scoring** (`train_baseline_sparse_qwen3.py`, runs: `1/22(1)`–`1/22(7)`, `1/23()`, etc.)
Reproduces the original paper's reward: binary per-rubric-item (errors="none" → 1, else → 0), reward = satisfied/total - format_penalty. Multiple iteration runs in month 1. This is Phase 1's reproduction method.

**GRPO-only control** (`train_grpo_sanity.py`, runs: `2/SDPO/` subset)
SDPO codebase with all distillation terms disabled (`use_sdpo=False`, `sdpo_grpo_mix_lambda=1.0`). Isolates SDPO's contribution vs pure GRPO in the same code path.

### Reward Shaping

**Hard-min aggregation** (`train_hard_min.py`, run: `2/hard_min/14`)
Changes rubric aggregation from mean to `(1 - alpha) * mean + alpha * min` across the 7 desiderata, with `alpha=0.5`. Penalizes plans with even one weak desideratum. All other settings identical to bestversion.

**0–9 grading scale** (`train_scale_0_9.py`, runs: `2/0-9_scale/15(2_3)`, `15(All)`, `15(None)`)
Grader uses a 10-level scale (0–9) instead of 4-level (0–3), providing finer-grained reward signal. Dense desiderata mapped linearly as `level/9.0`; non-dense bucketed back to 4 coarse levels. Three variants tested different subsets of desiderata with dense scoring.

**Weighted mean+std** (`train_weighted_dense_score.py`, run: `2/std+mean_weighted/11(ml)`)
Adaptive per-desideratum reweighting using a rolling buffer (window=20). Two multiplicative terms: mean-weight (`exp(-3.0 * (mean_d - global_mean))`, upweights below-average desiderata) and std-weight (`exp(3.0 * (std_d - std_mean))`, upweights high-variance desiderata). Both clipped to [0.5, 2.0].

**Weighted mean-only** (`train_weighted_mean.py`, runs: `2/std+mean_weighted/12,13(mean_weighted)`)
Same framework but only mean-based reweighting active (std path disabled). Upweights desiderata scoring below average.

**Weighted std-only** (`train_weighted_std.py`, runs: `2/std+mean_weighted/12,13(std_weighted)`)
Same framework but only variance-based reweighting active (mean path disabled). Upweights desiderata with unstable (high-variance) scores.

### Loss Function

**SDPO — Self-Distillation Policy Optimization** (`train_sdpo.py`, runs: `2/SDPO/20`–`24`)
Adds a teacher-student distillation term on top of GRPO. Grader feedback (weaknesses + fixes) is fed back to the model to get improved "teacher" logprobs. Token-level SDPO advantage = `teacher_logprobs - student_logprobs`, mixed with GRPO: `final_adv = 0.30 * grpo_adv + 0.70 * sdpo_adv`. Teacher logprobs interpolated with frozen reference (`alpha=0.01`). SDPO advantages clipped to [-5, +5]. Multiple runs explored hyperparameter variations.

### Data Domain

**ArXiv transfer** (`train_baseline_arxiv.py`, run: `2/withA1,A2/3(arxiv)`)
Identical bestversion pipeline trained on the arXiv split instead of ML.

**PubMed transfer** (`train_baseline_pubmed.py`, run: `2/withA1,A2/3(pubmed)`)
Identical bestversion pipeline trained on the PubMed split instead of ML.

### Pre-Bestversion Baseline

**Older codebase** (`train_baseline_best.py`, runs: `2/std_stand+band_bonus/8,9`)
Same GRPO + same reward formula, but on the older codebase before a1+a2 improvements (no `drop_noncompliant_samples`, `format_retry`, `min_words_warmup`, etc.).

### Ablation Outcome

None of the reward shaping, loss function, or data domain ablations surpassed bestversion's eval score of 0.693. The consistent finding was that bestversion's simple, direct GRPO with the nonlinear 4-level scoring was the most effective configuration.

---

## Phase 3: Doublegeneration (Two-Stage GRPO)

### Pipeline (`src/co_scientist/trainers/refinement/train_double_generation.py`)

```
For each batch of 64 goals:
  Stage 1 — Initial Generation:
    1. Generate 4 plans per goal (async)
    2. Grade each with DETAILED grader (structured XML with weaknesses/fixes)
    3. Extract feedback: weaknesses, fixes, desiderata profiles
    4. Compute Stage 1 GRPO advantages (within 4 initial samples)

  Stage 2 — Refinement:
    5. For each initial plan: build refinement prompt (original plan + feedback)
    6. Generate 4 refined plans (async)
    7. Grade refined plans with detailed grader
    8. Compute Stage 2 GRPO advantages (within 4 refined samples)

  Combined Optimization:
    9. Merge datums from both stages (weighted equally)
    10. Single forward_backward + optim_step
```

**Compute-matched design:** 4 initial + 4 refined = 8 total samples per goal (same as bestversion).

### Training Results (Epoch 2: `runs/2026/3/refinement/8/`)

- 179 batches
- Refined reward: rises 0.55→0.66 (peaks batch ~130-139), then FALLS to 0.47
- Refinement delta shrinks: 0.27 (early) → 0.13 (late)
- Regression rate rises: ~10% (early) → 28% (late)
- Root cause: Stage 1 improvement causes distribution shift in Stage 2 input; Stage 2's learned refinement strategy becomes OOD

### Eval Results

**Single-stage eval** (Stage 1 only, `eval_only.py`, batch 133):

| Metric | Value |
|--------|-------|
| Rubric mean | 0.6556 |
| Reward mean | 0.6926 |
| Format penalty | 0.0104 |

**Two-stage eval** (Stage 1 → feedback → Stage 2, `eval_only_double.py`, batch 133):

| Metric | Stage 1 | Stage 2 (Refined) | Delta |
|--------|---------|-------------------|-------|
| Rubric mean | 0.6611 | 0.6536 | **-0.0075** |
| Reward mean | 0.6923 | 0.6651 | -0.0272 |
| Improvement rate | — | 60.6% | — |

**Conclusion:** Stage 2 refinement actually **hurt** performance at this checkpoint. The doublegeneration approach failed to outperform bestversion.

### Critical Note: Grader Prompt Scale Mismatch

- `best_ver.py` uses `build_grader_prompt` (simple) → batch 0 rubric ~0.61
- `train_double_generation.py` uses `build_detailed_grader_prompt` (structured XML) → batch 0 rubric ~0.24
- These scales are NOT comparable. All evals use the simple grader for scoring to ensure comparability.

---

## Gap Analysis (Bestversion, 640 goals x 8 samples)

Before attempting new training approaches, we analyzed where the gap between model performance (0.69) and reference performance (0.86) comes from.

### Decomposition

| Gap Component | Size | % of Total |
|--------------|------|-----------|
| **Selection gap** (mean → oracle best-of-8) | 0.157 | **91%** |
| **Capability gap** (oracle → reference) | 0.015 | **9%** |

### Best-of-N Ceiling

| N | Oracle Score |
|---|-------------|
| 1 | 0.70 |
| 2 | 0.77 |
| 4 | 0.82 |
| 8 | 0.85 |

### Root Cause

80.6% of model-low scoring items: reference scores HIGH on the same rubric item. The rubric items check the reference's specific methodology (e.g., "uses federated learning", "addresses cold-start with method X"). The model proposes different valid approaches that don't match these method-specific criteria.

**Implication:** The model CAN produce excellent plans (oracle 0.85), but most samples use a different approach than what the rubric expects. The gap is primarily about **selection** (picking the right sample), not **capability** (producing good plans).

---

## Reference Evaluation

### Pipeline (`src/co_scientist/trainers/baselines/eval_reference.py`)

```
For each reference solution in the test set:
  1. Use reference text as both proposed_plan AND reference_solution in grader
  2. Grade with simple grader (same as all evals)
  3. Record rubric score
```

### Result

| Metric | Value |
|--------|-------|
| **Rubric mean** | **0.8665** |
| Rubric std | 0.1269 |
| Rubric median | 0.8971 |
| N examples | 685 |

This establishes the performance ceiling: even reference solutions don't score 1.0 because some desiderata naturally cap below maximum.

---

## Phase 4: Rubric Dropout (Criteria-Aware Generation)

### Motivation

Target the 9% capability gap. Hypothesis: if the model sometimes sees rubric items during training, it will learn what good criteria look like and generalize to inferring criteria when rubric is hidden (as in eval).

### Pipeline (`src/co_scientist/trainers/refinement/best_ver_rubric_dropout.py`)

```
Same as bestversion, except:

For each goal in the batch:
  1. Coin flip with probability p → decide rubric visibility
  2. If visible: inject rubric items in prompt, think block reviews criteria
     If hidden: think block infers criteria autonomously
  3. Generate 8 plans, grade all (grader always sees rubric)
  4. Track rewards by visibility: reward_visible_batch, reward_hidden_batch

After each batch:
  5. Update RubricScheduler:
     - EMA of hidden reward (alpha=0.1)
     - p = p_min + (p_max - p_min) * (target - hidden_ema) / (target - baseline_hidden)
     - As hidden reward improves → p decreases → less rubric shown

At eval time: rubric is NEVER shown (matches real task).
```

### Scheduler Design

```
p starts at p_max (0.80)
baseline_hidden recorded on first batch with hidden samples
As training progresses:
  hidden_ema rises → gap to target shrinks → p falls
  p_min = 0.05 (floor), p_max = 0.80 (ceiling)
  target = 0.86 (reference score)
```

### Scheduler Issues Discovered and Fixed

1. **Original design (p = gap/reference_gap):** Collapsed to p=1.0 immediately because the visible-hidden gap couldn't change after just 1 batch. Created a starvation feedback loop where no hidden samples were generated.

2. **Fix 1 — p_max ceiling:** Added p_max=0.80 to guarantee hidden samples. Prevented starvation.

3. **Fix 2 — Absolute hidden reward signal:** Changed from relative gap to tracking hidden_ema toward target. Different goals per batch made the relative gap noisy. Absolute hidden reward is robust to goal variation.

4. **Fix 3 — Scheduler state persistence:** Added state_dict/load_state_dict so p doesn't reset on resume.

### Training Results (`runs/2026/3/rubric_dropout/13/`, 107 batches = 1 epoch)

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| hidden_ema | 0.546 | 0.729 | +0.183 |
| visible_ema | 0.824 | 0.955 | +0.131 |
| p | 0.800 | 0.363 | -0.437 |
| gap to target | 0.314 | 0.131 | closed 58% |
| dropped samples | 0-1 | 0-1 | stable |

The scheduler worked correctly: p decreased as hidden reward improved. The model did learn to score better without rubric during training.

### Eval Result (Checkpoint batch 105)

| Metric | Bestversion | Rubric Dropout | Delta |
|--------|------------|----------------|-------|
| **Rubric mean** | **0.6926** | **0.6569** | **-0.0357** |
| Reward mean | 0.7498 | 0.7092 | -0.0406 |
| Format penalty | 0.0013 | 0.0086 | +0.0073 |

**Rubric dropout underperformed bestversion by 0.036.** Splitting training signal between visible and hidden samples hurt more than criteria-awareness transfer helped. The visible-sample learning (80%→36% of training) didn't transfer to eval, and hidden samples were fewer than bestversion's 100% no-rubric training.

**Status:** Abandoned.

---

## Phase 5: Best-of-N Self-Selection (Eval-Time)

### Motivation

Target the 91% selection gap with no training. If the model can evaluate its own plans and pick the best one, it could close the gap from 0.69 (mean) toward 0.85 (oracle).

### Pipeline (`src/co_scientist/trainers/baselines/eval_bon.py`)

```
For each test goal:
  1. Generate 8 plans from finetuned policy (same as normal eval)
  2. Grade all 8 with standard grader (rubric + reference → rubric_score)
  3. Self-evaluate all 8 with a separate prompt:
     - Self-evaluator sees ONLY scenario + plan (NO rubric items, NO reference)
     - Must infer evaluation criteria, then score against 7 desiderata
     - Output format matches grader XML (reuses scoring function)
  4. Select: pick plan with highest self-eval score
  5. Report:
     - mean-of-8: average grader score across 8 samples
     - oracle: grader score of best plan (upper bound)
     - self-selected: grader score of plan the self-evaluator picked
     - gap_closed_pct: how much of (oracle - mean) did self-selection capture
     - rank_corr: Spearman correlation between grader and self-eval rankings
```

### Finetuned Model Results (Bestversion, batch 214)

| Metric | Value |
|--------|-------|
| mean-of-8 | 0.681 |
| oracle best-of-8 | 0.849 |
| **self-selected** | **0.694** |
| **gap closed** | **7.5%** |
| rank correlation | 0.063 |
| self picked oracle | 12.9% |

### Base Model Results (No finetuning, 5 batches)

| Metric | Value |
|--------|-------|
| mean-of-8 | 0.562 |
| oracle best-of-8 | 0.709 |
| self-selected | 0.565 |
| gap closed | 2.4% |
| rank correlation | 0.050 |

### Comparison: Base vs Finetuned

| Metric | Base | Finetuned | Delta |
|--------|------|-----------|-------|
| mean-of-N | 0.562 | 0.681 | +0.119 |
| oracle | 0.709 | 0.849 | +0.140 |
| self-selected | 0.565 | 0.694 | +0.128 |
| gap closed % | 2.4% | 7.5% | +5.0% |
| rank corr | 0.050 | 0.063 | +0.014 |

### Self-Eval Score Analysis

| Metric | Base | Finetuned |
|--------|------|-----------|
| Self-eval score (model's own assessment) | 0.724 | 0.827 |
| Grader score (real quality) | 0.563 | 0.681 |

The self-evaluator sees 87% of the grader's improvement — confirming that by the model's own standards, training improved output quality. However, self-eval scores are inflated (model is always generous with itself).

### Why Self-Selection Failed

The self-evaluator ranks the oracle sample at **mean rank 4.4 out of 8** — a nearly **uniform distribution** (10-15% at each rank position). This is indistinguishable from random.

Key evidence:
- Self-eval gives its picks score 0.94, but grader gives those same plans only 0.69
- Oracle plans score 0.85 by grader but only 0.83 by self-eval
- Pearson correlation between grader and self-eval: 0.061 (essentially zero)

**Root cause:** The self-evaluator infers general quality criteria ("is the plan detailed?") but cannot infer the method-specific rubric items ("does the plan use federated learning?"). It confidently scores against wrong criteria.

**Status:** Self-selection approach does not work without access to actual rubric items.

---

## Summary: All Eval Results

| Method | Eval Rubric | vs Bestversion | Notes |
|--------|------------|----------------|-------|
| **Reference solutions** | 0.867 | +0.174 | Ceiling |
| **Bestversion** | **0.693** | **baseline** | **Best result** |
| BoN self-selected (finetuned) | 0.694 | +0.001 | Self-selection barely works |
| Doublegeneration (Stage 1) | 0.656 | -0.037 | Single-stage underperforms |
| Doublegeneration (Stage 2) | 0.654 | -0.039 | Refinement hurts |
| Rubric dropout | 0.657 | -0.036 | Criteria-awareness doesn't transfer |
| BoN self-selected (base) | 0.565 | -0.128 | Base model, no training |
| Base model mean | 0.562 | -0.131 | Base model, no training |

### Phase 2b Ablations (no formal eval — training-time comparison only)

| Method | Ablation type | Runs | What it tested |
|--------|--------------|------|----------------|
| Hard-min aggregation | Reward shaping | `2/hard_min/14` | `0.5*mean + 0.5*min` across desiderata |
| 0–9 grading scale | Reward shaping | `2/0-9_scale/15(*)` | 10-level vs 4-level grading (3 variants) |
| Weighted mean+std | Reward shaping | `2/std+mean_weighted/11` | Adaptive reweighting by mean and variance |
| Weighted mean-only | Reward shaping | `2/std+mean_weighted/12,13(mean)` | Upweight below-average desiderata |
| Weighted std-only | Reward shaping | `2/std+mean_weighted/12,13(std)` | Upweight high-variance desiderata |
| SDPO | Loss function | `2/SDPO/20–24` | Per-token self-distillation from grader feedback |
| GRPO-only control | Loss function | `2/SDPO/` subset | SDPO codebase with distillation disabled |
| ArXiv transfer | Data domain | `2/withA1,A2/3(arxiv)` | Same method, arXiv data |
| PubMed transfer | Data domain | `2/withA1,A2/3(pubmed)` | Same method, PubMed data |
| Blended generation | Architecture | `3/blended/1,18` | Relabel refined outputs as initial-prompt |
| Pre-bestversion baseline | Historical | `2/std_stand+band_bonus/8,9` | Before a1+a2 improvements |
| GSM8K sanity check | Pipeline validation | `1/25(gsm8k_*)` | Binary math reward, validates GRPO works |
| Paper-original (sparse) | Reproduction | `1/22(*)`, `1/23(*)` | Original paper's binary rubric reward |

---

## Key Learnings

### 1. The Gap is Selection, Not Capability
The model produces 0.85-quality plans in its best-of-8 samples, but averages 0.69. 91% of the gap to reference (0.87) is about consistently picking the right approach, not about generating better content.

### 2. Method-Specific Rubric Items Are the Bottleneck
80.6% of scoring failures occur because rubric items check the reference's specific methodology. The model proposes different valid approaches that score low on these criteria. This is arguably an evaluation bias, not a model limitation.

### 3. Training Signal Splitting Hurts
Both rubric dropout and doublegeneration split training signal across multiple objectives. In both cases, the split hurt the primary task more than the secondary objective helped. Bestversion's simple single-stage GRPO with 100% of signal on the eval-relevant task remained strongest.

### 4. Self-Evaluation Cannot Replace the Rubric
The model's self-evaluation (0.061 correlation with grader) is effectively random for ranking. It cannot infer method-specific rubric items from the scenario alone. Any selection mechanism that doesn't access the actual rubric will likely fail.

### 5. Training Improves Generation Quality
Bestversion training clearly improved both grader scores (+0.12) and self-eval scores (+0.10 by the model's own assessment). The 87% alignment between self-eval and grader improvements confirms the quality gain is real, not just rubric-hacking.

### 6. Epoch 2 Requires Careful Monitoring
Doublegeneration showed severe degradation in epoch 2 (reward collapsed from 0.66 to 0.47). The root cause was distribution shift: Stage 1 improved, making Stage 2's learned strategy OOD. Bestversion was more stable across epochs but also plateaued.

---

## Possible Future Directions

1. **Trained reranker:** Use the ~100k+ (goal, plan, grader_score) pairs from training to train a lightweight model predicting grader score from (goal, plan). A learned reranker could capture subtle statistical patterns that prompt-based self-eval cannot.

2. **Increase N with weak selection:** Generate 16-32 samples. Oracle best-of-16 would exceed 0.89. Even a modest selector becomes useful at higher N.

3. **Consensus selection:** Instead of scoring individually, identify the majority approach among N samples and select the best representative. If most samples converge on a similar method, it may align with what the rubric expects.

4. **Challenge the evaluation methodology:** Report the finding that 80.6% of the gap comes from method-specific rubric bias. The model's best-of-8 (0.85) nearly matches the reference (0.87), suggesting the model's capability is much higher than the mean score suggests.

---

## File Reference

### Training Code
| File | Purpose |
|------|---------|
| `src/co_scientist/trainers/baselines/best_ver.py` | Bestversion single-stage GRPO |
| `src/co_scientist/trainers/baselines/train_baseline_best.py` | Pre-bestversion baseline (older codebase) |
| `src/co_scientist/trainers/baselines/train_baseline_arxiv.py` | ArXiv domain transfer |
| `src/co_scientist/trainers/baselines/train_baseline_pubmed.py` | PubMed domain transfer |
| `src/co_scientist/trainers/baselines/train_baseline_gsm8k.py` | GSM8K pipeline sanity check |
| `src/co_scientist/trainers/baselines/train_baseline_sparse_qwen3.py` | Paper-original sparse binary reward |
| `src/co_scientist/trainers/hard_min/train_hard_min.py` | Hard-min rubric aggregation |
| `src/co_scientist/trainers/scale_0_9/train_scale_0_9.py` | 0–9 grading scale |
| `src/co_scientist/trainers/weighted/train_weighted_dense_score.py` | Adaptive mean+std reweighting |
| `src/co_scientist/trainers/weighted/train_weighted_mean.py` | Mean-only reweighting |
| `src/co_scientist/trainers/weighted/train_weighted_std.py` | Std-only reweighting |
| `src/co_scientist/trainers/sdpo/train_sdpo.py` | SDPO (self-distillation + GRPO) |
| `src/co_scientist/trainers/sdpo/train_grpo_sanity.py` | GRPO-only control for SDPO |
| `src/co_scientist/trainers/refinement/train_double_generation.py` | Two-stage GRPO with refinement |
| `src/co_scientist/trainers/refinement/train_blended_generation.py` | Blended generation (prompt relabeling) |
| `src/co_scientist/trainers/refinement/best_ver_rubric_dropout.py` | Rubric dropout training |

### Evaluation Code
| File | Purpose |
|------|---------|
| `src/co_scientist/trainers/baselines/eval_only.py` | Single-pass eval (shared by all) |
| `src/co_scientist/trainers/baselines/eval_only_double.py` | Two-stage eval (Stage 1 + refinement) |
| `src/co_scientist/trainers/baselines/eval_reference.py` | Grade reference solutions |
| `src/co_scientist/trainers/baselines/eval_bon.py` | Best-of-N with self-selection |
| `src/co_scientist/trainers/baselines/eval_only_rubric.py` | Eval with rubric visibility flag |

### Run Paths
| Run | Path |
|-----|------|
| Paper reproduction (sparse) | `runs/2026/1/22(1)` – `1/23(2)` |
| GSM8K sanity check | `runs/2026/1/25(gsm8k_…)`, `1/25(rl_loop_gsm8k)` |
| Dense score baseline | `runs/2026/1/success_dense_score` |
| Pre-bestversion baseline | `runs/2026/2/std_stand+band_bonus/8,9` |
| Bestversion (2 epochs) | `runs/2026/2/withA1,A2/2(ml)/` |
| ArXiv transfer | `runs/2026/2/withA1,A2/3(arxiv)` |
| PubMed transfer | `runs/2026/2/withA1,A2/3(pubmed)` |
| Hard-min | `runs/2026/2/hard_min/14` |
| 0–9 scale (3 variants) | `runs/2026/2/0-9_scale/15(2_3)`, `15(All)`, `15(None)` |
| Weighted mean+std | `runs/2026/2/std+mean_weighted/11(ml)` |
| Weighted mean-only | `runs/2026/2/std+mean_weighted/12,13(mean_weighted)` |
| Weighted std-only | `runs/2026/2/std+mean_weighted/12,13(std_weighted)` |
| SDPO (5 runs) | `runs/2026/2/SDPO/20` – `24` |
| Doublegeneration epoch 2 | `runs/2026/3/refinement/8/` |
| Blended generation | `runs/2026/3/blended/1`, `18` |
| Rubric dropout | `runs/2026/3/rubric_dropout/13/` |
| BoN finetuned eval | `runs/2026/2/withA1,A2/2(ml)/eval_bon/` |
| BoN base model eval | `eval_bon_base_model/` |
| Reference eval | `runs/2026/2/withA1,A2/2(ml)/reference_plan_score/` |

### Analysis Notebooks
| Notebook | Location |
|----------|----------|
| Rubric dropout diagnosis | `analysis/notebooks/active/rubric_dropout_diagnosis.ipynb` |
| Doublegeneration diagnosis | `analysis/notebooks/active/double_gen_diagnosis.ipynb` |
| Method overview | `analysis/notebooks/active/method_overview.ipynb` |
| Robust diagnosis tools | `analysis/notebooks/diagnosis/` |
| Cross-method comparisons | `analysis/notebooks/comparison/` |
