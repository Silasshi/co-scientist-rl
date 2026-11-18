# Selection-gap analysis with best-of-N and self-selection

**Question:** after GRPO training plateaued at 0.693, is the policy unable to write good plans, or does it write them and fail to pick them?

## Implementation — [`src/co_scientist/eval/eval_bon.py`](../../src/co_scientist/eval/eval_bon.py)
For each held-out goal:
1. Generate N = 8 plans.
2. Grade all 8 with the real grader (rubric + reference).
3. Self-evaluate all 8 with a prompt that sees **only the goal and the plan**. The model has to infer the criteria itself ([`build_self_eval_prompt` L103](../../src/co_scientist/eval/eval_bon.py#L103)).
4. Report, per goal ([`compute_goal_metrics` L202](../../src/co_scientist/eval/eval_bon.py#L202)):
   - mean-of-N
   - oracle best-of-N
   - the grader score of the plan the model picked
   - Spearman rank correlation between self-scores and grader scores ([L177](../../src/co_scientist/eval/eval_bon.py#L177))

The script reports BoN-8. The BoN-2 and BoN-4 points below were computed afterwards from the same per-sample logs, using the exact combinatorial expected-max estimator, so the curve needs no extra sampling.

## Result
Trained checkpoint, 640 goals × 8 samples, same grader as training ([`project_conclusion.md`](../../shared/analysis/reports/project_conclusion.md)):

| BoN-1 (mean) | BoN-2 | BoN-4 | BoN-8 (oracle) | Self-selected | Reference |
|---|---|---|---|---|---|
| 0.681 | 0.754 | 0.808 | 0.849 | 0.694 | 0.867 |

- **90.8%** of the gap between the mean and the reference lies between the mean and the oracle. That part is a *selection* problem, not a capability problem.
- Self-selection closes only **7.5%** of that gap. Its rank correlation with the grader is **0.063**, roughly chance.
  - Cause: the model scores against general quality ("is it detailed?"). The rubric checks the *reference's specific method*, which the model cannot infer.

**Caveat.** The maximum of 8 noisy grader scores is biased upward, so part of the oracle gap is grader noise, not hidden capability. For that reason the finding is stated as an upper bound and was **not** used as evidence that the policy "can already do it".
