# Multi-signal reward with vector advantages and hard gates (D3)

**Motivation:** goal-specific rubrics mostly check whether a plan uses the *reference's* method. In 80.6% of low-scoring items the reference scored high on that same item ([`selection_gap.md`](../../projects/rubric_reward/knowledge/findings/selection_gap.md)). This direction asks whether a **goal-agnostic** set of plan-quality signals can replace them. It follows the "decompose, don't collapse" line of work: Rubrics as Rewards [arXiv:2507.17746](https://arxiv.org/abs/2507.17746), Fine-Grained RLHF [arXiv:2306.01693](https://arxiv.org/abs/2306.01693), ArmoRM [arXiv:2406.12845](https://arxiv.org/abs/2406.12845).

## Signals — [`shared/ten_signal_reward.py`](../../src/co_scientist/shared/ten_signal_reward.py), [`shared/seven_signal_reward.py`](../../src/co_scientist/shared/seven_signal_reward.py)
- **Hard gates** zero the reward ([`HARD_GATES` L98](../../src/co_scientist/shared/seven_signal_reward.py#L98)):
  - *Goal contrast*: the grader must prefer the plan for its own goal over 2 distractor goals.
  - *Claim verification*: claims are extracted, then checked by a program ([L533](../../src/co_scientist/shared/seven_signal_reward.py#L533)).
- **Gradient signals** (v9): depth, rigor, formalism, arithmetic consistency, positioning, feasibility, risk awareness, specificity, scope, focus. Each is a separate grader call with its own rubric spec ([`SignalSpec` L76](../../src/co_scientist/shared/ten_signal_reward.py#L76)). Weights sum to 1 and an assertion enforces it ([L856–L885](../../src/co_scientist/shared/ten_signal_reward.py#L856)).

## Vector advantage — [`ttt_discover/train_7signal_vector.py`](../../src/co_scientist/ttt_discover/train_7signal_vector.py)
Instead of normalising a scalar sum, the trainer normalises **each signal within the group** and then takes the weighted sum ([`compute_vector_advantages` L347](../../src/co_scientist/ttt_discover/train_7signal_vector.py#L347)):

```
adv_k,i = (s_k,i − mean_k) / (std_k + ε)
A_i     = Σ_k w_k · adv_k,i
```

With a scalar sum, a signal with low variance gets drowned out. With per-signal normalisation, a plan that is better on a rarely-varying signal still gets credit for it.

## Signal validation before training
Every signal set was validated **before** any RL run. Validation checked:
- detection of controlled perturbations (60 reference plans + 162 perturbed versions, AUC 0.767 for v8.1)
- agreement with an independent cross-family judge (Claude Opus)

This caught one problem. Within a tight score band (Qwen grader 0.78–0.82), the v8.1 reward ranked plans *against* Opus (ρ = −0.16), because the top of the scale was saturated. The v9 "signal hardening" added formula-depth counting and an arithmetic-consistency check graded by a larger model. The changes were:

| Change | Before | After | Source |
|---|---|---|---|
| Training-reward ceiling (Qwen grader) | 0.975 | 0.950 | [`projects/ttt_discover/STATUS.md`](../../projects/ttt_discover/STATUS.md) |
| Independent Opus score | 7–8 / 20 | 9 / 20 | same |
| New signals' rank agreement with Opus | — | ρ = 1.0 (N = 3) and 5/5 perturbations detected | [`SIGNAL_SET_v9_FULL_REFERENCE.md`](../../projects/ttt_discover/knowledge/current/SIGNAL_SET_v9_FULL_REFERENCE.md) |

These are small-N checks on a single goal. They show what the validation procedure catches, and they are not claims of general validity.
