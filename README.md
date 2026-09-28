# co-scientist-rl

RL post-training of LLMs for open-ended **research-plan generation**. This project is a follow-up to
*Training AI Co-Scientists Using Rubric Rewards* (Goel et al., [arXiv:2512.23707](https://arxiv.org/abs/2512.23707)).

Given a research goal, the policy writes a research plan. A frozen copy of the model then grades the plan against goal-specific rubric items, and that grade is the reward. The work here reproduces the rubric-reward GRPO recipe at 30B scale and tries several ways to go beyond it.

Author: Yuhong (Silas) Shi, Torr Vision Group, University of Oxford (Dec 2025 – Apr 2026).

## Results at a glance

| | |
|---|---|
| Policy | Qwen3-30B-A3B (MoE), LoRA rank 64, trained through the [Tinker](https://thinkingmachines.ai/tinker/) API |
| Data | `facebook/research-plan-gen` (ML split; not redistributed here) |
| **Reproduction** | Held-out rubric score **0.562 → 0.693** (+23% relative) on 640 held-out goals × 8 samples, frozen Qwen3-30B-A3B grader |
| Extensions | 80+ runs: self-distillation (in-house SDPO-style variants), reward shaping, two-stage refinement, rubric dropout, multi-turn, think-then-solve, critique–revise + RL. **None surpassed the reproduced baseline.** |
| Main diagnosis | Best-of-8 sampling closes ~91% of the gap between the average plan (0.681) and the reference plan (0.867). The bottleneck is **selection and reward fidelity, not generation**. |
| Reward hacking | When the policy is optimised against an LLM grader, the proxy reward rises while independent-judge quality falls (grant-proposal study: Qwen reward 0.855 → 0.970, Opus-judged score 19 → 13 / 40). |

Status: **concluded**. The reproduction worked. The extensions were negative or unstable, and the lasting output is the set of diagnostic findings about rubric rewards. See [`shared/analysis/reports/project_conclusion.md`](shared/analysis/reports/project_conclusion.md).

## Directions

See [`DIRECTIONS.md`](DIRECTIONS.md) for the full map and [`shared/knowledge/EXPERIMENT_CATALOG.md`](shared/knowledge/EXPERIMENT_CATALOG.md) for every experiment.

| Direction | Idea | Outcome |
|---|---|---|
| D1 Rubric reward | GRPO against a goal-specific rubric grader, plus extensions | Reproduction 0.693; no extension beat it |
| D2 Iterative brainstorming | Dialogue-style idea refinement | Inconclusive |
| D3 / D4 Critique–revise + RL | Multi-signal rewards; grant-proposal writing with OpenAlex retrieval | Most of the gain comes from critique itself; RL adds little |
| D5 Per-goal self-distillation | Privileged-information self-distillation with an external critic | Small, fragile gains (single goal, few seeds) |

## Code map

```
src/co_scientist/
  rubric_reward/grpo/best_ver.py        D1 reproduction (GRPO loop adapted from tinker-cookbook rl_loop.py)
  rubric_reward/sdpo/train_sdpo.py      D1 self-distillation variant
  shared/rubric_grader.py               rubric-based reward
  eval/eval_bon.py                      best-of-N evaluation (selection-gap analysis)
  ttt_discover/train_7signal_vector.py  D3 multi-signal vector GRPO
  grant_proposal/train_cr_v7.py         D4 critique–revise + RL on grant proposals
  d5_abstract_retrieve_refine/          D5 self-distillation variants (train_mu_v*.py)
projects/<direction>/                   configs, run summaries, analysis notebooks, decision logs
shared/                                 docs, analysis reports, tools
tests/                                  unit tests
```

## Not included

Datasets, model checkpoints, raw rollouts and generated plans, paper drafts, third-party papers and API credentials are not included. `tinker://` checkpoint URIs in configs need a Tinker account. The Python environment is listed in `shared/env/pip_freeze_tinker.txt` (Python 3.13, tinker 0.7.0).

## Attribution

The training loop is adapted from [tinker-cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) (Apache-2.0). See [`NOTICE`](NOTICE) and `THIRD_PARTY_NOTICES/`. The task and rubric-reward method follow Goel et al. (2025).
