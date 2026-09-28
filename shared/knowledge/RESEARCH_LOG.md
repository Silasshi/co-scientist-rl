# Research Log: Co-Scientist Project

## Overview

This project set out to train an LLM to generate structured research plans using reinforcement learning. Over four months and 112 runs, we discovered that the simplest approach -- single-stage GRPO -- reaches a ceiling that no amount of training-time complexity can break. Every multi-stage, multi-turn, and multi-loss variant we tried performed worse. The breakthrough insight came not from trying harder at generation, but from measuring *where* the gap actually lies: 91% of the performance gap between our model and reference solutions is a selection problem, not a capability problem. The model already produces excellent plans; it just cannot tell which ones are excellent. This realization pivoted the project from generation training to selection training, which is where it stands today.

---

## Chapter 1: Finding Our Footing (January 2026)

We began with a blank slate: no preferred model, no validated reward formulation, no evidence that GRPO could even work for research plan generation.

The first month was pure exploration. Across 16 runs, we tested three model architectures -- Qwen3-30B-A3B, gpt-oss-20b, and Llama-3.1-8B -- on the facebook/research-plan-gen dataset. The task: given a research goal and target, generate a structured plan that would score well on a 7-desiderata rubric covering thoroughness, specificity, soundness, justification, efficiency, ethics, and coherence.

Two findings emerged quickly. First, Qwen3-30B-A3B significantly outperformed the other models. Second, dense rubric-based scoring (mapping each desideratum to a 0-3 scale) provided far better training signal than sparse binary scoring. An instructive contrast: on GSM8K (math), sparse scoring worked beautifully (reward 0.961 in 23 batches), but research plan generation is too nuanced for binary correctness signals. The 7-desiderata rubric gave GRPO the gradient landscape it needed.

The first substantial training run (`success_dense_score`) reached reward 0.587 over 189 batches. Not impressive by later standards, but it proved the concept: RL training could meaningfully improve research plan quality. The architecture was set -- Qwen3-30B-A3B with dense scoring -- and we moved on to optimizing the method.

*For details, see [Phase 1: Baselines](phases/phase1_baselines.md).*

---

## Chapter 2: The Bestversion Ceiling (February 2026)

February was the most productive and most frustrating month of the project. Productive because we found the best method. Frustrating because nothing we tried afterward could beat it.

Bestversion (internally called "withA1,A2") combined two improvements over the Phase 1 baseline: LoRA rank 64 with a carefully calibrated reward formula incorporating a Gaussian length bonus (centered at 600 words) and a format penalty (harsh cliff at 750 words). The training loop was GRPO at its simplest -- 64 goals per batch, 8 samples per goal, grade all, compute group-relative advantages, PPO-style update.

After 2 epochs (214 batches), bestversion reached eval rubric **0.693** on the ML test split. Training reward grew from 0.65 to 0.83, format compliance hit 99%, and the model learned to produce well-structured, informative plans. We also trained cross-domain variants: PubMed (0.853 reward) and arXiv (0.726 reward), confirming the approach generalizes.

Thinking we could do better, we launched two massive explorations:

**20 SDPO runs** tested self-distillation policy optimization across lambda values from 0.1 to 0.45, including recovery experiments warmstarted from the bestversion checkpoint at batch 214. The hypothesis was that distilling from the model's own best outputs would provide a richer learning signal than GRPO's group-relative advantages. It did not. The best SDPO run (`28_recovery_A_warmstart`, 244 batches) reached reward 0.673 -- below bestversion. Self-distillation adds a loss term that is redundant with what GRPO already captures.

**11 reward shaping ablations** tested wider scales (0-9 instead of 0-3), hard minimum constraints, weighted reward variants, and band bonuses. The std-weighted variant achieved the highest raw training reward (0.819), but higher training rewards did not translate to higher eval rubric scores. This was an early warning sign: the reward function and the eval metric are not perfectly aligned.

By the end of February, we had our answer. Bestversion -- the simplest method tested -- was the best. Twenty SDPO runs and eleven reward ablations collectively proved that adding complexity to single-stage GRPO does not help. The 0.693 eval ceiling was established.

*For details, see [Phase 2: Bestversion](phases/phase2_bestversion.md) and [Method: GRPO Bestversion](methods/grpo_bestversion.md).*

---

## Chapter 3: The Gap Analysis Revelation (Late February -- Early March 2026)

Before charging into Phase 3, we paused to understand *why* bestversion plateaued. The answer reshaped the entire project.

We ran a comprehensive gap analysis on bestversion's eval output: 552 goals, 8 samples each. For each goal, we compared the model's mean score, its oracle best-of-8 score (if we could magically pick the best sample), and the reference solution score.

The numbers were striking:

| Metric | Score |
|--------|-------|
| Model mean (N=1) | 0.70 |
| Oracle best-of-2 | 0.77 |
| Oracle best-of-4 | 0.82 |
| Oracle best-of-8 | 0.85 |
| Reference solutions | 0.86 |

The **selection gap** (mean to oracle) was 0.157, accounting for **91% of the total gap** to reference. The **capability gap** (oracle to reference) was just 0.015, only 9%. The model was already generating plans nearly as good as human references -- it just could not identify them.

We tested self-evaluation: ask the model to score its own outputs without rubric access. Correlation with actual rubric score: r ~ 0.061. Essentially random. The model has no ability to distinguish its best work from its worst.

We also examined the 80.6% of rubric items where the model scored low: in those cases, the reference solution scored *high* on the same criterion. The rubric checks for the reference's specific methodology. When the model proposes a different (but potentially valid) approach, it fails method-specific criteria. This is partly a rubric design issue, not a pure capability issue.

This analysis was the turning point. It told us that further generation training was the wrong lever. The selection gap -- the model's inability to pick its own best work -- was where almost all the opportunity lay. But we did not pivot immediately. Instead, we spent March trying to break the ceiling anyway.

*For details, see [Finding: Selection Gap](findings/selection_gap.md).*

---

## Chapter 4: Trying to Break the Ceiling Anyway (March 2026)

March was the month of ambitious failures. Armed with the gap analysis but not yet ready to abandon generation improvement, we tried four fundamentally different approaches. All failed.

### Doublegeneration: Two-Stage Refinement

The most carefully designed method of the project. Stage 1 generates 4 initial plans, the grader provides structured XML feedback (weaknesses, suggested fixes), and Stage 2 refines each plan using that feedback. Compute-matched to bestversion: 4+4 = 8 total samples per goal. Independent GRPO advantages per stage, combined in one optimizer step.

It worked -- at first. Refined plans improved over initial plans 83-89% of the time, and the refinement delta started strong at 0.27. But then the collapse began. As Stage 1 improved during training, the distribution of inputs to Stage 2 shifted. Stage 2 had learned to fix weak plans; when handed increasingly good plans, its refinement strategy became out-of-distribution. After batch 130, the refined reward peaked at 0.66 and fell to 0.47 by batch 178. The regression rate -- how often Stage 2 *worsened* a plan -- climbed from 10% to 28%.

Eval rubric: **0.654**. Not only below bestversion (0.693), but the internal collapse made the method fundamentally unstable.

*This was our first encounter with a principle that would recur: coupled training stages create distribution shift that eventually destroys the downstream stage.*

### Rubric Dropout: Stochastic Criteria Visibility

A simpler idea: randomly hide some rubric criteria during training, forcing the model to learn to infer hidden criteria. The hypothesis was that this would improve generalization to criteria the model has not seen.

Eval rubric: **0.657**. Splitting the training signal between visible and hidden criteria hurt more than awareness transfer helped. The model needs all criteria visible to optimize effectively.

### Multi-turn Discussion: V1 through V4

Four versions of a multi-turn discussion approach where the model plays Collaborator, Researcher, and Evaluator roles in a structured research discussion before producing a final plan. Each version attempted to fix the previous one's failure:

- **V1** used a binary PRM (Process Reward Model) to score discussion quality. Signal too coarse -- no learning.
- **V2** merged the evaluator role into the student. The model learned to produce empty responses (71% empty rate), completely shortcutting the discussion.
- **V3** fixed the role architecture but had a PRM parsing bug: all negative scores mapped to 0, destroying the reward signal. Only 3 batches before discovery.
- **V4** introduced wave-based training, ternary PRM ({-1, 0, +1}), few-shot examples, and robust parsing. Despite all fixes, 5 runs produced no meaningful progress. Multi-turn discussion showed no correlation with final plan quality (Pearson r = -0.115 to 0.052).

The multi-turn saga taught us something important: models treat discussion as overhead to be minimized. If the final plan is all that is scored, discussion turns are unrewarded process that the optimizer will eliminate. This became the process-shortcutting finding.

### The Pattern

By the end of March, a clear pattern had emerged: **every method that adds complexity performs worse than simple bestversion.** Two-stage refinement collapses from distribution shift. Rubric dropout splits the signal. Multi-turn discussion gets shortcutted. The generator ceiling is real, and more complex training cannot break through it.

*For details, see [Phase 3: Doublegeneration](phases/phase3_doublegeneration.md), [Method: Doublegeneration](methods/doublegeneration.md), and [Finding: Negative Results](findings/negative_results.md).*

---

## Chapter 5: The Process-Shortcutting Discovery (March -- April 2026)

As evidence accumulated from multi-turn, doublegeneration, and the first IBT experiments, we recognized a universal principle operating across all failed methods.

**Think-solution** made it most visible. We separated generation into explicit `<think>` (reasoning) and `<solution>` (plan) blocks, with independent grading. Result: think blocks scored **-0.090 lower** than non-thinking plans. Reasoning actively hurt. The model learned that the think block consumes token budget without contributing to the rewarded output, so it either minimized thinking or filled it with counterproductive content.

**Multi-turn V2** demonstrated the extreme case: 71% of discussion responses were empty. The model had discovered that the shortest path to reward was to skip the discussion entirely.

**IBT** showed the same dynamic in a different form: the model converged toward producing near-final plans at turn 0 and making minimal use of subsequent brainstorming turns.

We named this **process shortcutting**: models optimize for the shortest path to reward. Any intermediate step -- thinking, discussion, brainstorming, refinement -- that is not directly tied to the reward signal will be minimized or eliminated. This is not a bug in any specific method; it is a fundamental property of reward-based optimization with terminal-only reward.

The implications were profound. It meant that any method relying on intermediate reasoning, iterative improvement, or collaborative discussion would face the same failure mode unless those processes were themselves rewarded -- and our attempts at process reward models (binary PRM, ternary PRM) had failed to provide sufficient signal.

*For details, see [Finding: Process Shortcutting](findings/process_shortcutting.md).*

---

## Chapter 6: Pivoting to Selection (April 2026)

April brought two final pieces of evidence and a strategic pivot.

### GPT-5.4: Capability Without Training

We evaluated GPT-5.4 on the same test set with zero-shot generation (no RL training, no rubric access). Rubric score: **0.843**. For context, our RL-trained bestversion scored 0.693 and reference solutions score 0.860. GPT-5.4 nearly matched reference quality without any training at all.

This result had two implications. First, raw model capability can overcome what RL training struggles with. Second, and more importantly, it suggested that the reward function itself might be suppressing quality. The length penalty (0.2 + 0.0005 * excess for plans over 750 words) and the Gaussian length bonus (centered at 600 words) actively discourage the longer, more detailed plans that the grader values. Within plans, length correlates with rubric score at only r = 0.075 -- the reward function is optimizing a dimension that barely matters. See [Finding: Reward Misalignment](findings/reward_misalignment.md).

### IBT: Iterative Brainstorming on a Small Model

IBT shifted to a 4B-parameter model (Qwen3-4B-Instruct-2507) for faster iteration, training with K-turn iterative improvement where each turn receives the grader's hint from the previous turn. Across 27 runs in 6 rounds, the results were mixed at best. Single-chain (1 sample/turn) reached reward 0.894, but the hint vs no-hint comparison was ambiguous: a clear benefit early on (0.842 vs 0.737 in round 2) shrank to nearly zero by round 5 (0.868 vs 0.864). OPD variants consistently hurt. IBT V2 (self-distillation signal types) failed to produce any batches.

The most informative IBT experiment was self-calibration: have the model self-evaluate blind (no rubric), compare to actual rubric scores, and amplify the GRPO advantage where the model overestimates its quality. Result: a nearly uniform 1.08x weight across all samples. Since self-evaluation is random (r ~ 0.061), the calibration weights carry no signal. Self-calibration was yet another confirmation that the model cannot judge its own work.

*For IBT details, see [Method: IBT](methods/ibt.md).*

### The Selector: Targeting the 91% Gap Directly

Having exhausted generation-improvement approaches, we pivoted to the selection gap.

The first attempt -- CPR (Contrastive Plan Ranking) -- was deliberately minimal: a pairwise prompt ("Answer with just A or B"), max_tokens=32, temperature=1.5. Only 12 batches in one run. It served as a proof of concept.

The self-selector improved on CPR with structured prompts (7 generic quality dimensions), chain-of-thought reasoning (max_tokens=512), position debiasing (test both orderings), and validation every 50 batches. Training data: 111K preference pairs extracted from bestversion's training logs (score gap >= 0.05). The full evaluation pipeline generates 8 plans per goal, runs a pairwise tournament (28 pairs, or 56 with debiasing), and selects the winner.

The self-selector is implemented but not yet trained. It is the most promising remaining direction.

*For selector details, see [Method: CPR & Self-Selector](methods/selector.md).*

---

## Key Decision Points

1. **Choose Qwen3-30B-A3B over gpt-oss-20b and Llama-3.1-8B** (January). Justified by superior performance in Phase 1 exploration. All subsequent work built on this model.

2. **Choose dense rubric scoring over sparse binary scoring** (January). Sparse scoring worked for math (GSM8K) but not for the nuanced research plan task. The 7-desiderata rubric provided the gradient landscape GRPO needed.

3. **Adopt bestversion as the permanent baseline** (February). After 20 SDPO runs and 11 reward ablations failed to beat it, we accepted that single-stage GRPO with the A1+A2 formula was optimal for this task. Every subsequent method was evaluated against bestversion's 0.693 eval rubric.

4. **Conduct gap analysis before Phase 3** (late February). This analysis -- showing 91% selection gap, 9% capability gap -- fundamentally reframed the problem. Without it, we might have spent much longer trying to break the generation ceiling.

5. **Pursue generation improvement in Phase 3 despite the gap analysis** (March). In hindsight, we could have pivoted to selection immediately. But the generation approaches (doublegeneration, multi-turn) needed to be tried to confirm the ceiling. The negative results strengthened the case for selection.

6. **Recognize process shortcutting as a universal failure mode** (March-April). This stopped us from proposing yet another multi-stage or multi-process training method. Any future method must either reward intermediate steps effectively or avoid them entirely.

7. **Pivot to selection training** (April). After 80+ runs confirmed the generation ceiling and the gap analysis pointed to selection as the lever, CPR and then the self-selector became the primary focus. This was the most consequential strategic decision of the project.

---

## Open Questions for Future Work

1. **Can a trained selector close the 91% gap?** The self-selector is implemented but untrained. If it can reliably pick the best plan from 8 candidates, it would reach ~0.85 eval rubric -- matching reference solutions.

2. **Is the rubric itself the fundamental problem?** 80.6% of model-low items fail because the rubric checks the reference's specific methodology. A more method-agnostic rubric might reveal that the model's alternative approaches are valid.

3. **Would a more capable base model overcome the ceiling?** GPT-5.4 scores 0.843 without RL. If compute were not a constraint, using a larger base model might bypass the entire generation ceiling.

4. **Could DPO work better than GRPO for the selector?** The pairwise ranking task is a natural fit for Direct Preference Optimization, which trains directly from preference pairs without reward modeling.

5. **What if we fix the reward function?** Removing the length penalty and aligning the reward more closely with the eval metric (pure rubric score) might raise the generation ceiling. The best_ver_async experiment (850 words, 3072 tokens) was inconclusive because the reward penalty had already shaped the model's behavior.

6. **Does the selector generalize across generators?** If trained on bestversion plans, will the selector work when paired with a different generator (e.g., a future model fine-tuned with a better reward function)?

---

## Reading Guide

This research log provides the narrative arc. For details on specific topics:

- **Method implementations**: `shared/knowledge/methods/` -- [GRPO Bestversion](methods/grpo_bestversion.md), [Doublegeneration](methods/doublegeneration.md), [IBT](methods/ibt.md), [Self-Selector](methods/selector.md)
- **Key findings**: `shared/knowledge/findings/` -- [Selection Gap](findings/selection_gap.md), [Reward Misalignment](findings/reward_misalignment.md), [Process Shortcutting](findings/process_shortcutting.md), [Negative Results](findings/negative_results.md), [What Works](findings/what_works.md)
- **Phase summaries**: `shared/knowledge/phases/` -- [Phase 1](phases/phase1_baselines.md), [Phase 2](phases/phase2_bestversion.md), [Phase 3](phases/phase3_doublegeneration.md), [Phase 4](phases/phase4_advanced.md)
- **Run data**: [Experiment Catalog](EXPERIMENT_CATALOG.md) for numbers and tables; [Runs Catalog](../runs/RUNS_CATALOG.md) for the full 112-run inventory
- **Source code**: `src/co_scientist/rubric_reward/` (edit here); `src/methods/` (symlinks, do not edit)
- **Analysis notebooks**: `analysis/notebooks/`
