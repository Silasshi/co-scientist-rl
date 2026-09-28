# Method: IBT (Iterative Brainstorming Training) & Self-Calibration

## Motivation

After the bestversion ceiling (eval rubric 0.693) proved impervious to every training-time modification on the 30B model -- doublegeneration, rubric dropout, multi-turn discussion, SDPO -- the project shifted to two new angles. First, could iterative improvement with grader hints yield learning that single-pass GRPO could not? Second, could a smaller, cheaper model (4B parameters) learn effectively with the right training structure, opening the door to more affordable experimentation?

IBT was designed around the insight that human researchers improve plans iteratively: draft, get feedback, revise. Rather than the two-stage structure of doublegeneration (which suffered OOD collapse as Stage 1 improved), IBT uses a K-turn loop where each turn builds on the previous turn's grader hint. This avoids the stage-coupling problem by treating each turn as an independent RL update.

Self-calibration came later as a separate attempt: instead of iterating over turns, it asked whether the model could learn to judge its own blind spots and use that self-awareness to weight the training signal.

---

## Pipeline

### IBT V1 (`src/co_scientist/trainers/ibt/train_ibt.py`)

```
For each goal (processed sequentially, 20-25 goals per run):

  Turn 0:
    prompt = research_goal (no hint)

  Turn k (k = 1 ... K-1):
    prompt = research_goal + hint from turn k-1

  Each turn:
    1. Policy generates plan(s)
       - single_chain: 1 sample, temperature=1.0
       - mini_grpo: G=4 samples, temperature=1.0
    2. Grader (Qwen3-30B-A3B, temp=0.0) scores plan and generates improvement hint
    3. Compute advantage:
       - single_chain: reward - EMA_baseline (decay=0.9)
       - mini_grpo: group-relative advantage over G samples
    4. PPO-style update (lr=1e-5, clip_eps=0.2)
    5. Extract hint from grader output -> feed into next turn's prompt

  After K turns: move to next goal (weights carry over)
```

**Models**: Policy = Qwen3-4B-Instruct-2507 (LoRA rank 32), Grader = Qwen3-30B-A3B.

**Key design choices**:
- **Sequential goal processing**: Unlike bestversion's 64-goal batches, IBT processes one goal at a time through K turns before moving on. Weights accumulate across goals.
- **Hint accumulation**: Each turn sees only the latest hint (not all previous hints). The grader produces both a score and actionable improvement suggestions.
- **Two modes**: single_chain is cheaper (1 sample/turn, 5 turns = 5 total generations per goal). mini_grpo is more expensive but provides group-relative signal (4 samples/turn, 5 turns = 20 total generations per goal).

### OPD Variant (Online Policy Distillation)

Both modes support an OPD flag. When enabled, the per-token advantage is a weighted sum of the RL scalar advantage and a token-level OPD signal:

```
advantage_token = w_rl * scalar_advantage + w_opd * clip(log(pi_hint/pi_base))
```

This attempts to distill hint-aware knowledge into the policy at the token level. In practice, OPD variants consistently underperformed their non-OPD counterparts (reward 0.584 vs 0.894 for single_chain, 0.699 vs 0.877 for mini_grpo in round 2).

### Baselines

| Baseline | Description | Script |
|----------|-------------|--------|
| **Training-free** | Same K-turn loop with hints, but NO weight updates. Tests whether in-context hints alone improve quality. | `baseline_training_free.py` |
| **Single-pass GRPO** | Standard GRPO (G=8, no iteration) on the 4B model. Tests whether iterating helps vs single-shot. | `baseline_grpo.py` |

### IBT V2 (`src/co_scientist/trainers/ibt/train_ibt_v2.py`)

Three signal types, all with NO hints in the model's input prompt. Hints are used only for computing supervision signals:

| Setup | Signal computation | Rationale |
|-------|-------------------|-----------|
| **self_distill_ratio** | Per-token advantage = clip(pi(t\|goal+hint) / pi(t\|goal)) * scalar_advantage | Hint signal as multiplicative weight on RL advantage |
| **self_distill_additive** | Per-token advantage = w_rl * scalar_advantage + w_sd * clip(log_hint - log_base) | OPD-style additive, but self-distillation (same model) |
| **dense_grpo** | Accumulate G samples across all turns, compute GRPO advantage over full buffer | Importance-weighted; stale samples dropped if too old (>3 turns) |

### Self-Calibration (`src/co_scientist/trainers/ibt/train_self_calibration.py`)

A separate approach applied at bestversion scale (64-goal batches, Qwen3-30B-A3B, LoRA rank 64):

```
For each batch of 64 goals:
  1. Generate 8 plans per goal (standard bestversion)
  2. Grade each plan with rubric grader -> rubric_eval scores
  3. Self-evaluate each plan (blind: no rubric, no reference) -> self_eval scores
  4. Compute calibration gap per desideratum: self_eval - rubric_eval
  5. Compute calibration weight: amplify advantage where model overestimates
  6. GRPO update with calibration-weighted advantages
```

The hypothesis: where the model thinks it did well but the rubric says otherwise, there is a blind spot. Amplifying the gradient in those cases should teach the model to address its weaknesses.

---

## Runs & Results

### IBT V1 Rounds

| Round | Runs | Key variants | Batches | Notes |
|-------|------|-------------|---------|-------|
| **1** | 4 | single_chain, mini_grpo, +OPD | 125 each | First round. single_chain reward 0.894; OPD hurts (-0.144 for SC, -0.109 for MG) |
| **2** | 8 | +baselines, +no-hint ablation | 250-306 | mini_grpo reward 0.877; single_chain_nohint 0.737 vs with-hint 0.842 |
| **3** | 4 | training-free baselines | Eval-only | Testing hint accumulation without training |
| 4 | -- | -- | -- | Missing round |
| **5** | 4 | single_chain, nohint, baselines | 300-303 | Hint vs no-hint gap narrows: 0.868 vs 0.864 |
| **6** | 3 | V2 variants (self_distill, dense_grpo) | 0 each | All failed to produce batches |
| **7** | 1 | Self-calibration (30B model) | 119 | Only completed run at bestversion scale |

**Total**: 27 runs across 6 rounds (round 4 missing). Most V2 runs (round 6) produced 0 batches.

### Key Numbers

| Variant | Best reward | Batches | Round |
|---------|-------------|---------|-------|
| single_chain | 0.894 | 125 | 1 |
| single_chain (round 5) | 0.868 | 300 | 5 |
| mini_grpo | 0.877 | 250 | 2 |
| single_chain_nohint | 0.864 | 303 | 5 |
| single_chain_opd | 0.750 | 125 | 1 |
| mini_grpo_opd | 0.699 | 250 | 2 |
| self_calibration (run 7) | 0.725 | 119 | 7 |

### Self-Calibration Result

Self-calibration produced a nearly uniform 1.08x weight across all samples. The model's self-evaluation correlation with rubric: r ~ 0.061 (effectively random). Since the calibration weights are essentially uniform, the weighted GRPO update is indistinguishable from standard GRPO. No improvement over bestversion.

---

## What Failed and Why

### OPD Consistently Hurts

OPD variants underperformed across both modes (single_chain, mini_grpo) in every round tested. The cross-model distillation signal (30B grader's hint-aware probabilities imposed on 4B policy) creates a distribution mismatch. The 4B model cannot reproduce the 30B model's token-level patterns, and the OPD gradient overwhelms the RL signal.

### V2 Runs Failed to Launch

All three V2 setups (self_distill_ratio, self_distill_additive, dense_grpo) in round 6 produced 0 batches. The more complex training signal computations introduced implementation or stability issues that prevented training from starting.

### Hint Value Is Ambiguous

The hint vs no-hint comparison tells a mixed story:
- Round 2: hint helps (0.842 vs 0.737, delta = 0.105)
- Round 5: hint barely helps (0.868 vs 0.864, delta = 0.004)

This suggests that early in training, grader hints provide useful guidance, but the benefit vanishes as training progresses. The model may learn to produce good plans regardless of hints, or the hints themselves become less informative as plans improve (the same dynamic that caused doublegeneration's feedback degradation).

### Process Shortcutting

Like multi-turn and think-solution, IBT is susceptible to process shortcutting. The iterative brainstorming steps are not directly rewarded -- only the final plan is scored. The model converges toward producing near-final plans at turn 0 and making minimal use of subsequent turns.

### Self-Calibration: Self-Eval Is Random

The fundamental obstacle to self-calibration is that the model cannot meaningfully evaluate its own outputs. With self-eval correlation r ~ 0.061 with rubric scores, the calibration gap (self_eval - rubric_eval) is noise, not signal. Amplifying noise in the gradient produces uniform-ish weights and no learning benefit.

---

## What This Led To

IBT's mixed results reinforced two lessons:

1. **Process steps get shortcutted.** Whether it is think blocks, discussion turns, or brainstorming iterations, any process not directly tied to reward will be minimized. See [Process Shortcutting](../findings/process_shortcutting.md).

2. **Self-evaluation is unreliable.** The self-calibration failure is the clearest demonstration. The model cannot tell which of its outputs are good, which is exactly the selection gap identified in the [gap analysis](../findings/selection_gap.md).

These findings strengthened the case for pivoting to selection-focused approaches (CPR, self-selector) rather than continuing to search for better generation training methods.

---

## Lessons for Future Work

1. **OPD across model scales is counterproductive.** Distilling from a 30B model into a 4B model via token-level signals creates harmful distribution mismatch. If distillation is needed, it should be at the reward level (scalar), not the token level.

2. **Hints have diminishing returns.** Grader hints help early but lose value as the policy improves. This mirrors doublegeneration's feedback degradation pattern. Any method relying on iterative feedback should expect diminishing returns.

3. **Small-model experiments are useful for speed but not directly comparable.** The 4B model runs complete faster and allow more variants, but the 4B/30B capability gap makes it hard to attribute results to the method vs the model.

4. **Self-evaluation requires training, not just prompting.** Prompting the model to self-evaluate produces random signal. A trained selector (see [Self-Selector](selector.md)) may succeed where prompted self-evaluation fails.

---

## Implementation Files

| File | Purpose |
|------|---------|
| `src/co_scientist/trainers/ibt/train_ibt.py` | IBT V1 (single_chain, mini_grpo) |
| `src/co_scientist/trainers/ibt/train_ibt_v2.py` | IBT V2 (self_distill_ratio, self_distill_additive, dense_grpo) |
| `src/co_scientist/trainers/ibt/train_self_calibration.py` | Self-calibration (bestversion scale) |
| `src/co_scientist/trainers/ibt/baseline_training_free.py` | Baseline A: no weight updates |
| `src/co_scientist/trainers/ibt/baseline_grpo.py` | Baseline B: single-pass GRPO on 4B |
| `src/co_scientist/trainers/ibt/create_shared_init.py` | Shared initialization checkpoint creation |

## Run Paths

- Round 1: `runs/2026/4/ibt/1/`
- Round 2: `runs/2026/4/ibt/2/`
- Round 3: `runs/2026/4/ibt/3/`
- Round 5: `runs/2026/4/ibt/5/`
- Round 6 (V2): `runs/2026/4/ibt/6/`
- Round 7 (self-calibration): `runs/2026/4/ibt/7/`

## Related

- [Negative Results](../findings/negative_results.md) -- IBT and self-calibration cataloged here
- [Process Shortcutting](../findings/process_shortcutting.md) -- IBT exhibits the same pattern
- [Selection Gap](../findings/selection_gap.md) -- self-calibration failure confirms self-eval is unreliable
- [Self-Selector](selector.md) -- the trained alternative to prompted self-evaluation
- [Back to Catalog](../EXPERIMENT_CATALOG.md)
