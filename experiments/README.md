# experiments/ — research iteration history

These 94 scripts are the **iteration trail** behind the featured implementations in `src/co_scientist/`. They cover ablations, superseded trainer versions, baselines, one-off probes and evaluation harnesses. They are kept for provenance, so that every number cited in `projects/<direction>/` can be traced to the code that produced it.

Nothing in `src/` imports from here. The featured, documented implementations are listed in the top-level [README](../README.md#skills-demonstrated) and explained in [`docs/algorithms/`](../docs/algorithms/).

Run a script from the repo root with `PYTHONPATH=src python experiments/<path>.py key=value ...`. The scripts use [chz](https://github.com/thinking-machines-lab/chz) configs and need a Tinker account. Paths resolve relative to the repo root.

| Folder | Count | What is in it |
|---|---|---|
| `rubric_reward/` (D1) | 15 | Reward-shaping ablations (hard-min, weighted mean/std, 0–9 scale, dense score), rubric-conditioned GRPO, think/solution split advantages, multi-turn GRPO with a process reward model (v1–v3), rubric dropout, blended generation |
| `eval/` | 10 | Evaluation harnesses: reference-solution and frontier-LLM ceilings, contrastive plan ranking, self-selector, RAG, multi-turn, methodology-agnostic grader validation |
| `ibt/` (D2) | 5 | Iterative-brainstorming baselines (training-free prompting, single-pass GRPO), shared LoRA init for fair comparison, self-calibration |
| `ttt_discover/` (D3) | 1 | Pure-revision trainer with UCB plan selection |
| `grant_proposal/` (D4) | 1 | LSE-inspired RL trainer |
| `d5_abstract_retrieve_refine/` (D5) | 53 | Self-distillation trainer versions μ-v2…v9 (HER relabelling, replay buffer, on-policy IS-loss OPD, verifier-grounded reward, KL anchor), α/β/κ/σ/τ baselines, oracle and bibliography construction from LaTeX/S2, isolated / shuffled / strict LLM-judge audits, pairwise tournaments used to corroborate close verdicts |
| `d6_grant_proposal/` (D6) | 6 | Frozen baseline, gold/oracle extraction, deferred OPD-grounded and KL-anchor variants (`train_opd_kl_anchor.py` is a scaffold only) |
| `shared/` | 3 | Superseded rubric / audit prompt versions |

Known pre-existing breakage: `eval/eval_multiturn.py` imports a name (`ConversationResult`) that no longer exists in `train_multiturn_v4.py`. It was already broken in the original snapshot (`v0-snapshot` tag).
