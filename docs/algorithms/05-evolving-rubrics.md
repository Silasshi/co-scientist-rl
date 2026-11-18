# Evolving rubrics: generate, grade, filter by variance (GER-CR)

**Motivation:** a fixed rubric saturates. Once every plan in a group passes an item, that item carries no gradient. The fix tried here is to keep generating *new* rubric items that discriminate between the current rollouts, and to keep only the items that still vary. The generator prompt ([`shared/rubric_gen_prompt.py` L8–L12](../../src/co_scientist/shared/rubric_gen_prompt.py#L8)) borrows from two papers:
- the discriminative-power criteria in DR Tulu's evolving rubrics ([arXiv:2511.19399](https://arxiv.org/abs/2511.19399), App. A.1)
- the pairwise-contrast elicitation in OnlineRubrics ([arXiv:2510.07284](https://arxiv.org/abs/2510.07284), Fig. 2)

## Components
| Component | Role | Code |
|---|---|---|
| `RubricItem`, `RubricBuffer` | Items with stable hashed IDs and a per-iteration grade history. `filter_and_truncate` first drops items with zero variance (after a minimum number of iterations scored), then keeps the top K by the latest standard deviation | [`shared/rubric_buffer.py` L48–L162](../../src/co_scientist/shared/rubric_buffer.py#L48) |
| Contrast-pair selection | Picks a high-scoring and a low-scoring rollout, and asks the generator what separates them | [`select_contrast_pair` L173](../../src/co_scientist/shared/rubric_evolution.py#L173) |
| Generation step | Builds the prompt, calls a **pluggable** `GenClient` protocol (OpenRouter, Tinker, or a stub for tests), parses the JSON, inserts the items into the buffer. It has no API imports of its own | [`run_rubric_evolution_step` L83](../../src/co_scientist/shared/rubric_evolution.py#L83) |
| Binary per-item grader | One short yes/no call per item. A parse failure counts as 0. Qwen `<think>` and GPT-OSS reasoning channels are stripped | [`shared/rubric_grader.py` L41–L194](../../src/co_scientist/shared/rubric_grader.py#L41) |
| Trainer integration | A background rubric generator talks to the trainer through a file bus. Every N iterations it grades the reference against the active items as a drift diagnostic. **Off by default and byte-identical to CR-v7 when disabled** | [`grant_proposal/train_ger_cr_v1.py`](../../src/co_scientist/grant_proposal/train_ger_cr_v1.py) |

## Design choices worth noting
- **Mechanism before coupling.** In this phase the evolving rubric is logged but does *not* feed the RL advantage. I wanted to know whether the items stay discriminative and whether reference scores drift, before risking a new Goodhart surface.
- **Testable without the network.** `make_stub_gen_client` and `_run_sanity_checks` let the whole loop run offline.

## Status
The mechanism is implemented and unit-tested. The project concluded before the rubric was coupled into the reward. This page documents a design, not a result.
