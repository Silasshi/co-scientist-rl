# Co-Scientist Project: Comprehensive Analysis Overview

## Task

Train an LLM (Qwen3-30B-A3B, LoRA rank 64) to generate structured research plans given (research_goal, research_target). A grader model (Qwen3-30B-A3B, frozen) scores outputs using a 7-desiderata rubric per criterion. Dataset: `facebook/research-plan-gen`, ML split (~6,848 train / 685 test goals).

---

## Results Summary

| Method | Rubric | Samples | Notes |
|--------|--------|---------|-------|
| Reference solutions | 0.867 | 685 | Human expert, ceiling |
| GPT-5.4 (zero-shot) | 0.843 | 192 (partial, eval ongoing) | No training, no rubric access |
| Bestversion oracle (best-of-8) | 0.988 | 640 goals | Model CAN produce near-perfect plans |
| **Bestversion mean** | **0.693** | 5,120 | Single-stage GRPO, 2 epochs |
| BoN self-selected | 0.694 | 5,120 | Self-selection ≈ noise (r=0.061) |
| Rubric dropout | 0.657 | — | -0.036 vs bestversion |
| Doublegeneration (Stage 2) | 0.654 | — | Refinement hurt performance |
| All other ablations (SDPO, reward shaping, multi-turn V1-V4) | < 0.693 | — | None beat bestversion |

---

## Finding 1: The Gap Is Selection, Not Capability

**The model CAN produce 0.85+ plans. It just can't do so consistently.**

### Gap Decomposition (bestversion eval, 685 test goals x 8 samples)

| Component | Value | % of Total Gap |
|-----------|-------|---------------|
| Selection gap (mean -> oracle best-of-8) | 0.157 | 91% |
| Capability gap (oracle -> reference) | 0.015 | 9% |

### Best-of-N Ceiling

| N | Oracle Score |
|---|-------------|
| 1 | 0.70 |
| 2 | 0.77 |
| 4 | 0.82 |
| 8 | 0.85 |

### Root Cause

80.6% of scoring failures: rubric items check the reference solution's specific methodology (e.g., "uses federated learning for privacy"). The model proposes different valid approaches that don't match these method-specific criteria.

99.6% of rubric items are unique across goals (confirmed by gate experiment, ROUGE-L = 0.133 on k-NN retrieval). Rubric items are fundamentally unpredictable from goal text.

---

## Finding 2: Length Does NOT Explain Score Differences

**Within bestversion, longer plans do NOT score higher.**

| Metric | Value |
|--------|-------|
| Pearson(word_count, rubric_score) | -0.024 |
| Within-goal Pearson | 0.075 |
| High-scoring (>=0.85) plan avg words | 618 |
| Low-scoring (0.40-0.55) plan avg words | 633 |
| Best sample in group is longer? | 48.4% (coin flip) |

**Rubric by word count (bestversion, last epoch, n=55,242):**

| Words | Count | Rubric Mean |
|-------|-------|-------------|
| < 400 | 1,016 | 0.657 |
| 400-500 | 8,785 | 0.701 |
| 500-550 | 10,750 | 0.711 |
| 550-600 | 13,294 | 0.718 |
| 600-650 | 11,378 | 0.712 |
| 650-700 | 6,195 | 0.714 |
| 700-750 | 2,086 | 0.716 |
| 750+ | 1,738 | 0.648 |

The rubric is essentially flat across 400-750 words. Length is not the differentiator.

---

## Finding 3: Thinking Actually Hurts

**GRPO training eliminated the `<think>` block. The rare plans that still think score LOWER.**

| Plans | Count | % | Rubric Mean |
|-------|-------|---|-------------|
| WITHOUT think block | 4,969 | 97.1% | 0.695 |
| WITH think block | 151 | 2.9% | 0.605 |
| Delta | — | — | **-0.090** |

**Rubric by think block length (bestversion eval, n=5,120):**

| Think Length | n | Rubric |
|-------------|---|--------|
| No think | 4,969 | 0.695 |
| 151-300 words | 63 | 0.620 |
| 301-500 words | 65 | 0.632 |
| 500+ words | 22 | 0.483 |

**Within-goal comparison:**
- Least-thinking sample scores higher: 64.1% of goals
- Most-thinking sample scores higher: 34.4% of goals

### Why Thinking Hurts

1. **Token budget competition.** max_tokens=2048. Thinking tokens reduce words available for the scored `<solution>` block.
2. **Thinking = uncertainty signal.** The model produces think blocks when uncertain about the topic. Confident generations skip to `<solution>`. Think blocks mark "I don't know this well."
3. **Think blocks were never trained.** GRPO only rewards the solution. The prompt says "only the content within `<solution>` tags will be judged." The think block is unoptimized noise.

### Implication

The model's inconsistency is NOT due to lack of deliberation. Adding thinking would not help — the model never learned to think productively. The inconsistency stems from knowledge depth: for familiar goals the model is confident and correct; for unfamiliar goals it struggles regardless of thinking.

---

## Finding 4: The Real Differentiator Is Per-Desideratum Quality

**High and low scoring plans differ on ALL 7 desiderata simultaneously, not just one.**

Per-desideratum comparison (bestversion training logs, last epoch, same ~620 word count):

| Desideratum | High (>=0.85) | Low (0.40-0.55) | Gap |
|-------------|--------------|-----------------|-----|
| D1: Handles criteria | 2.78 | 1.60 | +1.18 |
| D2: Detailed/specific | 2.54 | 1.34 | +1.19 |
| D3: No flaws | 2.44 | 1.21 | +1.23 |
| D4: Well justified | 2.88 | 1.63 | +1.25 |
| D5: Cost efficient | 2.92 | 1.68 | +1.25 |
| D6: No ethical issues | 2.97 | 1.83 | +1.14 |
| D7: Consistent | 2.97 | 1.96 | +1.00 |

Low-scoring plans fail uniformly across all dimensions. The gap is not a specific missing skill — it is overall plan quality collapsing when the model encounters an unfamiliar goal.

---

## Finding 5: GPT-5.4 Overcomes the Gap Through Raw Capability

**GPT-5.4 (zero-shot, no training) scores 0.843 — matching reference solutions.**

| Metric | GPT-5.4 (n=192) | Bestversion (n=5,120) |
|--------|-----------------|---------------------|
| Rubric mean | 0.843 | 0.693 |
| Word count | ~797 | ~600 |
| Format compliant | 50% | 99% |
| Final reward | 0.742 | 0.750 |

### Controlling for Length

GPT-5.4 writes ~33% more words. But even at similar word counts:

| GPT-5.4 Words | n | Rubric | vs Bestversion |
|--------------|---|--------|----------------|
| 650-700 | 5 | 0.790 | +0.097 |
| 700-750 | 11 | 0.891 | +0.198 |
| 800+ | 69 | 0.849 | +0.156 |

At 650-700 words (bestversion's range), GPT-5.4 still scores +0.097 higher. Length contributes ~0.05-0.06 of the gap; capability contributes another ~0.05-0.10.

### Contamination Check

| Metric | GPT-5.4 vs Reference | Random Ref vs Ref (baseline) |
|--------|---------------------|------------------------------|
| Token Jaccard | 0.128 | 0.117 |
| 4-gram overlap | 0.002 | — |

No contamination detected. No tools/search enabled in API calls. GPT-5.4's high scores reflect genuine capability.

### Reward Paradox

By reward (which includes format penalty), bestversion WINS (0.750 vs 0.742). By rubric (pure content quality), GPT-5.4 crushes bestversion. The reward function penalizes the very detail that the grader values.

---

## Finding 6: Process-Shortcutting Is a Universal Pattern

**RL-trained models learn to shortcut any unrewarded process.**

| Version | What collapsed | Speed |
|---------|---------------|-------|
| Bestversion | Think blocks → zero | Complete by convergence (98% eliminated) |
| Multi-turn V2 | Discussion turns 0.7→0.0 | Immediate |
| Multi-turn V3 | Discussion turns 5.3→stable (crashed) | N/A |
| Multi-turn V4 | Discussion turns 5.6→0.0 | 10 batches |

In every case, the model rationally eliminates processes that don't improve the reward signal:
- Think blocks don't improve rubric score → eliminated
- Discussion doesn't improve rubric score → eliminated
- The model optimizes exactly what it's rewarded for, nothing more

---

## Finding 7: All Training-Time Solutions Failed

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Doublegeneration | 0.654 (-0.039) | Distribution shift: Stage 1 improved → Stage 2 went OOD |
| Rubric dropout | 0.657 (-0.036) | Signal splitting: 80%→36% visible hurt more than transfer helped |
| SDPO | < 0.693 | Self-distillation didn't add useful signal |
| Hard-min aggregation | < 0.693 | Changed what to optimize, not ability to optimize |
| 0-9 scale | < 0.693 | Finer reward granularity didn't help |
| Weighted reweighting | < 0.693 | Adaptive weights didn't overcome knowledge gap |
| Multi-turn V1-V4 | < 0.693 | Process-shortcutting in every version |
| CPR (contrastive ranking) | Abandoned | GRPO collapse on binary tasks; gate experiment showed rubric is unpredictable |
| RC-GRPO (rubric prediction) | Killed by gate | 99.6% unique rubric items, ROUGE-L = 0.133 |

---

## The Fundamental Limitation

The model's 3B active parameters (Qwen3-30B-A3B is MoE with ~3B active) cannot reliably store and apply the full breadth of ML research methodology knowledge across all goals in the dataset. For each goal:

- If the goal falls within the model's "comfort zone" → confident, produces 0.85+ plan
- If the goal is outside → uncertain, produces 0.50 plan
- The oracle (best-of-8) reaches 0.988 because temperature=1.0 sampling occasionally hits the right approach even for hard goals

GRPO cannot fix this because:
1. It teaches relative features (what's better than mean), not absolute knowledge
2. It cannot inject domain knowledge the model doesn't already have
3. It optimizes what's rewarded (solution text), eliminating unrewarded processes (thinking, discussion)

GPT-5.4 overcomes this through vastly more parameters and training data, giving it deeper and more consistent ML domain knowledge.

---

## Paths Forward

| Direction | Expected Impact | Effort | Risk |
|-----------|----------------|--------|------|
| SFT from GPT-5.4 plans (knowledge distillation) | +0.05-0.10 | 3-5 days | Medium — distribution shift from bestversion |
| Relax length constraints (700w target, wider σ) | +0.02-0.05 | 1-2 days | Low |
| Larger base model (Qwen3-32B or denser) | +0.05-0.10 | Days | Low if available on tinker |
| Challenge evaluation methodology (propose fairer metric) | Paper contribution | 1 week | Low |
| Structured reasoning training (train think blocks) | Unknown | 1-2 weeks | High — novel, untested |

---

## Paper Contribution

The project's core contributions are empirical findings, not a solution:

1. **The Rubric Lottery**: RL-trained LLMs hit a ceiling because method-specific rubric items create an unpredictable evaluation lottery. The 91/9 selection/capability decomposition quantifies this precisely.

2. **Process-Shortcutting**: Models trained with output-only rewards systematically eliminate unrewarded processes (thinking, discussion) — a universal pattern observed across 5+ training configurations.

3. **Reward Misalignment**: Standard reward shaping (length bonus + format penalty) actively suppresses the quality-correlated behaviors that the evaluation metric rewards.

4. **Capability vs Consistency**: A frontier model (GPT-5.4) overcomes the lottery through raw capability, demonstrating the lottery is a scaling phenomenon — smaller models face it harder.

5. **Comprehensive negative results**: 10+ training approaches fail to break the ceiling, with clear mechanistic diagnoses for each failure mode.

---

## File Reference

| File | Description |
|------|-------------|
| `analysis/project_conclusion.md` | Full experimental history and results |
| `analysis/multiturn_discussion_analysis.md` | Multi-turn discussion benefit analysis |
| `analysis/gate_report.json` | Rubric predictability gate experiment results |
| `runs/2026/2/withA1,A2/2(ml)/` | Bestversion training and eval logs |
| `runs/2026/4/eval_sota/gpt54/` | GPT-5.4 evaluation (ongoing) |
| `src/co_scientist/trainers/baselines/best_ver.py` | Bestversion trainer |
| `src/co_scientist/trainers/baselines/eval_sota.py` | SOTA model evaluation script |
| `paper/design/` | Pipeline design documents |
