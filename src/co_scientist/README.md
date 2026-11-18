# src/co_scientist — featured implementations

This package holds the implementations documented in [`docs/algorithms/`](../../docs/algorithms/), plus the modules they import. Superseded trainer versions, ablations and one-off probes are in [`experiments/`](../../experiments/). Nothing here imports from there.

| Directory | Featured modules |
|---|---|
| `rubric_reward/` (D1) | `grpo/best_ver.py` (GRPO + rubric reward), `sdpo/train_sdpo.py` (SDPO + GRPO mix), `selector/`, `refinement/train_double_generation.py`, `multiturn/train_multiturn_v4.py` |
| `ttt_discover/` (D3) | `train_buffer_ttt.py` (entropic test-time RL), `train_critique_revise.py`, `train_cr_v7.py`, `train_7signal_vector.py` (vector advantages) |
| `grant_proposal/` (D4) | `train_cr_v7.py` (per-signal REINFORCE + diversity), `train_ger_cr_v1.py` (evolving rubrics), `train_buffer_ttt.py`, `train_critique_revise.py` |
| `d5_abstract_retrieve_refine/` (D5) | `train_mu_v4.py` (off-policy self-distillation), `audit_v3_isolated.py` (isolated judge audit), `verifier_grounded_reward_v1.py`, prompt and oracle modules |
| `d6_grant_proposal/` (D6) | `train_opd.py` (on-policy distillation), `train_grpo.py` (reward-only control), `grounding.py`, `signals.py`, `audit.py` |
| `ibt/` (D2) | `train_ibt.py` |
| `shared/` | rewards (`ten_signal_reward`, `seven_signal_reward`, `grant_signal_reward`), graders (`rubric_grader`, `eval_core`), rubric evolution (`rubric_buffer`, `rubric_evolution`, `rubric_gen_prompt`), judge clients (`opus_*_subagent`), `paper_retrieval`, `api_profiles` |
| `eval/` | `eval_bon.py` (best-of-N / selection gap) |

Imports are absolute (`from co_scientist.shared.ten_signal_reward import ...`) with `PYTHONPATH=src`.
