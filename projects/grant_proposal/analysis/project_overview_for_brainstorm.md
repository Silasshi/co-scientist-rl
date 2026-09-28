# Project Overview: Making RL Work for Research Plan Generation

## 1. What We're Building

We're training a language model (Qwen3-30B) to generate high-quality research proposals (grant-style plans) for any given research goal. The model should produce plans with methodological depth, evidence grounding, feasibility, and goal-specificity.

Target venue: NeurIPS 2026. RL must be a core contribution — without a clear RL result, the paper doesn't stand.

## 2. Pipeline Architecture (CR-v7)

Each training iteration:

1. **Fresh Generation**: Model generates N=4 fresh research plans from scratch (temperature=1.0)
2. **Grading**: A Qwen3-30B grader scores each plan on 12 signals (1-5 scale each), e.g., reasoning depth, evidence rigor, formalism, risk awareness, etc.
3. **Hard Gates**: Two binary checks — (G1) goal-contrast: plan must be specific to target goal, not generic; (G2) claim verification: no fabricated citations. Plans failing hard gates get reward=0.
4. **Critique-Revise (CR)**: For each plan in the buffer, the grader produces per-signal critiques. A revision prompt shows the plan + all signal scores + critique text. The model rewrites the plan addressing the critiques. The revised plan is re-graded.
5. **Buffer**: Keeps the best plans found so far (by Qwen aggregate reward).
6. **RL Update (REINFORCE)**: Per-signal HER-style gradient — for each signal, compute log π(revision | single-signal context) weighted by per-signal reward delta. KL-regularized against the base model.

Key config flags:
- `skip_rl_update`: If True, model weights are frozen (no RL training) → "B4" condition
- `train_on_fresh`: If True, fresh plans also contribute to RL gradient
- `n_revise`: Number of revisions per iteration (0 = no CR)
- `scores_only`: If True, revision prompt shows signal names + scores but NO critique text
- `fresh_use_context`: If True, fresh generation prompt includes buffer plans as examples

## 3. Evaluation

- **Qwen reward**: Aggregate of 12 signal scores (weighted mean, 0-1 scale). Used as training reward. Known to saturate at high scores and be Goodhart-able.
- **Opus eval (ground truth)**: Independent Claude Opus 4.7 evaluation on 4 dimensions (depth, methods, feasibility, grounding), each 1-10 scale, total /40. Reference proposals score 34-37/40.

## 4. What We've Found

### 4.1 RL vs No-RL with full pipeline (CR + full critique)

Tested on 8 diverse research goals (AI, biomedical, ecology, social science), 25 iterations each.

| Metric | RL (MAIN) | No-RL (B4) | Δ |
|---|---|---|---|
| Fresh plan mean (last 5 iters) | 0.632 | 0.641 | **-0.009 ± 0.041** |
| Buffer max (final) | 0.953 | 0.969 | **-0.016 ± 0.014** |
| Opus eval (best plan /40) | 14 | 15 | **-1** |

**RL provides zero benefit when CR is present.** B4 is slightly better on every metric.

### 4.2 Why RL doesn't help

1. **CR masks RL's contribution**: CR is a plan-level improvement mechanism that doesn't depend on model quality. A frozen model + CR reaches buffer_max = 1.000. RL improves the model's average generation quality, but CR compensates for bad generation.

2. **Template collapse**: RL concentrates the policy on a single template that scores well on Qwen. This reduces diversity without improving actual quality.

3. **Reward saturation**: Qwen grader can't distinguish quality above ~0.9. RL gradient becomes pure noise at high scores.

4. **Best plans come from early iterations**: Buffer max reaches 95% of final value within 5-10 iterations, regardless of RL. The "best plan" is found by sampling + CR, not by gradual RL improvement.

### 4.3 RL without CR (fresh generation only)

| Condition | Hard gate pass rate | Fresh mean | Buffer max |
|---|---|---|---|
| Frozen model (no RL, no CR) | **8%** | 0.047 | 0.825 |
| RL trained (no CR) | **80%** | 0.406 | 0.625 |

RL dramatically improves hard gate pass rate (8% → 80%). But:
- The improvement is mostly "learning to not fail hard gates", not improving plan substance
- Plans that DO pass hard gates score similarly (frozen: ~0.59, RL: ~0.51)
- Buffer max is actually LOWER with RL (0.625 vs 0.825) due to template collapse — RL narrows the distribution

### 4.4 Scores-only revision (no critique text)

Tested on 235B model, different goal version (01_foundopt_v2, 1-10 scale rubric):

| Condition | Qwen reward | Opus eval /40 |
|---|---|---|
| Full critique + B4 | 0.953 | **29** |
| Full critique + RL | 0.964 | **26** |
| Scores-only + B4 | 0.987 | **12** |
| Scores-only + RL | 1.000 | **12** |

Removing critique causes extreme Goodhart: Qwen gives perfect scores but Opus reveals plans are hollow. The model games the grader when it doesn't know what specifically to fix.

### 4.5 GAPO (diversity RL)

Added nearest-neighbor distance bonus in signal space to encourage diverse fresh plans. Result: Opus 17/40 (same as B4 median). Diversity doesn't help.

### 4.6 Cross-domain generalization

~19 Opus-point gap between AI goals and non-AI goals, consistent across all conditions. Grounding is the universal weakness. This gap is goal-specific, not method-specific.

## 5. The Core Problem

**RL is redundant in the current architecture.** The critique-revise pipeline is so effective that model quality doesn't matter — CR can fix any plan to high quality. RL's contribution (improving the generator) is fully compensated by CR.

For RL to matter, we need one of:
- A setup where CR isn't available or is weaker
- A way to show RL + CR > CR alone (currently not the case)
- RL operating on a different part of the pipeline (not generation)
- A fundamentally different RL approach (not REINFORCE)

## 6. Constraints

- **Single-goal training**: Each run trains on one research goal. Single-goal RL is likely overfitting to that goal's preferences, unlikely to transfer to new goals. Multi-goal training is architecturally possible but adds significant complexity.
- **Model**: Qwen3-30B-A3B (policy and grader). 235B available but expensive.
- **Reward**: Qwen 30B grader with 12 signals. Known saturation and Goodhart issues. Opus eval is ground truth but too expensive for training reward.
- **Compute**: Each 10-iteration run takes ~30-50 minutes. Can run experiments quickly.
- **Template collapse**: REINFORCE on long text generation is inherently prone to mode collapse. KL regularization (budget=0.693) doesn't fully prevent it.

## 7. What We Need

A concrete, **reasonable** experimental setup where:
1. RL training produces a clear, visible improvement curve over iterations
2. The improvement is genuine (confirmed by Opus eval, not just Qwen gaming)
3. The setup is defensible in a paper (not artificially designed to make RL look good)
4. Ideally, RL shows improvement even in the presence of CR (RL + CR > CR alone)

## 8. Ideas Already Considered

| Idea | Status | Problem |
|---|---|---|
| RL + full CR pipeline | Tested (8 goals) | B4 ≥ MAIN everywhere |
| RL-only, no CR | Tested (v7-FreshOnly) | Improves hard gates but not substance; template collapse |
| GAPO diversity bonus | Tested | No improvement over B4 |
| Scores-only CR (weaker CR) | Partially tested | Causes Goodhart on 235B; running on 30B now |
| Multi-goal RL | Not tested | Adds complexity; 12 goals might not be enough diversity |
| Buffer context for fresh plans | Partially tested in D3 | D3 v9 showed RL helps at BoN=1; not replicated in D4 |
| RL for revision (not generation) | Not tested | Would need architecture changes |
| DPO/preference-based RL | Not tested | More stable than REINFORCE; needs code changes |
| Harder/better reward | Not tested | Could use Opus-in-the-loop or ranking-based reward |

## 9. Open Questions

1. Is there a reward design that gives RL clear gradient signal without being Goodhart-able?
2. Can RL operate on a narrower subtask (e.g., just methodology section, or just revision) where the output space is smaller and REINFORCE is more stable?
3. Is there a way to make RL + CR synergistic rather than redundant? (e.g., RL improves the model's ability to USE critiques effectively)
4. Should we abandon REINFORCE for DPO or other preference-based methods?
5. Is "RL teaches hard gate compliance" a publishable result on its own, or is it too narrow?
6. Can we reframe the paper so RL's contribution is about something other than final plan quality? (e.g., sample efficiency, robustness, cross-goal transfer)
