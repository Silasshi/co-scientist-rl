# Algorithms

Each page covers one method I implemented. It states the idea and the paper it follows, links to the key functions by line, and gives what was measured with its protocol and caveats.

| # | Method | Main code |
|---|---|---|
| 1 | [GRPO with an LLM rubric reward](01-grpo-rubric-reward.md) | `rubric_reward/grpo/best_ver.py` |
| 2 | [Self-distillation from privileged feedback (SDPO / OPD)](02-self-distillation.md) | `rubric_reward/sdpo/train_sdpo.py`, `d6_grant_proposal/train_opd.py`, `d5_.../train_mu_v4.py` |
| 3 | [Multi-signal reward with vector advantages and hard gates](03-multi-signal-vector-reward.md) | `shared/ten_signal_reward.py`, `ttt_discover/train_7signal_vector.py` |
| 4 | [Test-time RL with an entropic objective, plus critique–revise](04-critique-revise-rl.md) | `ttt_discover/train_buffer_ttt.py`, `ttt_discover/train_critique_revise.py` |
| 5 | [Evolving rubrics: generate, grade, filter by variance](05-evolving-rubrics.md) | `shared/rubric_buffer.py`, `shared/rubric_evolution.py` |
| 6 | [LLM-as-judge evaluation that resists its own biases](06-llm-judge-evaluation.md) | `d5_.../audit_v3_isolated.py`, `shared/opus_pairwise_subagent.py` |
| 7 | [Selection-gap analysis with best-of-N](07-selection-gap-bon.md) | `eval/eval_bon.py` |
| 8 | [Literature retrieval and verifier-grounded rewards](08-retrieval-and-grounding.md) | `shared/paper_retrieval.py`, `d6_grant_proposal/grounding.py` |

All training runs used **Qwen3-30B-A3B** (MoE) with LoRA rank 64, trained through the [Tinker](https://thinkingmachines.ai/tinker/) API.
