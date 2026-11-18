# Self-distillation from privileged feedback (SDPO / OPD family)

**Follows:** Hübotter et al., *Reinforcement Learning via Self-Distillation* (SDPO), [arXiv:2601.20802](https://arxiv.org/abs/2601.20802); Zhao et al., *Self-Distilled Reasoner: On-Policy Self-Distillation for LLMs* (OPSD), [arXiv:2601.18734](https://arxiv.org/abs/2601.18734).

## Idea
A scalar reward says *how good* a plan was, but not *which tokens* were wrong. Self-distillation turns feedback into a dense per-token signal:
- The **teacher** is the same model, conditioned on extra privileged context: the grader's critique, a reviewer's notes, or an "oracle" summary of the source paper.
- The **student** sees only the normal prompt.
- On each sampled token, the per-token advantage is `log p_teacher − log p_student`. Tokens the informed model would have written more confidently get pushed up.

The goal is for the policy to internalise feedback it will not see at test time.

I implemented three variants. They differ in **who samples** and **what the privileged context is**, and that difference turned out to matter.

| Variant | Sampler | Privileged context | Loss | Code |
|---|---|---|---|---|
| SDPO + GRPO mix (D1) | student | grader weaknesses / fixes for that plan | PPO-clip, `A = 0.3·A_GRPO + 0.7·clip(Δlp, ±5)` | [`rubric_reward/sdpo/train_sdpo.py`](../../src/co_scientist/rubric_reward/sdpo/train_sdpo.py) |
| On-policy distillation, OPD (D6) | student | external reviewer's (Claude Opus) critique, written with the reference proposal visible | PPO-clip on `clip(Δlp, ±5)`, solution tokens only | [`d6_grant_proposal/train_opd.py`](../../src/co_scientist/d6_grant_proposal/train_opd.py) |
| Off-policy relabelling, μ-v4 (D5) | **teacher** | oracle abstraction + previous critique | importance-sampling loss | [`d5_abstract_retrieve_refine/train_mu_v4.py`](../../src/co_scientist/d5_abstract_retrieve_refine/train_mu_v4.py) |

## Implementation notes
- **D1 SDPO.** The grader's XML is parsed into per-plan weaknesses and fixes ([`extract_weaknesses_from_grader` L1488](../../src/co_scientist/rubric_reward/sdpo/train_sdpo.py#L1488)), then turned into a self-teacher prompt ([`build_self_teacher_prompt` L1688](../../src/co_scientist/rubric_reward/sdpo/train_sdpo.py#L1688)).
  - Per-token Δlp is masked to `<solution>` tokens and clipped ([L3495–L3527](../../src/co_scientist/rubric_reward/sdpo/train_sdpo.py#L3495)), then mixed with the GRPO advantage ([L3529–L3533](../../src/co_scientist/rubric_reward/sdpo/train_sdpo.py#L3529)).
  - The teacher is regularised toward the initial policy (SDPO appendix interpolation variant, [L2362](../../src/co_scientist/rubric_reward/sdpo/train_sdpo.py#L2362)).
- **D6 OPD.**
  - The student samples, and an external reviewer writes a critique of that very rollout.
  - Teacher log-probs are computed on the student's tokens under `goal + oracle + critique` ([L425–L439](../../src/co_scientist/d6_grant_proposal/train_opd.py#L425)).
  - The advantage is clipped and solution-masked ([L462–L483](../../src/co_scientist/d6_grant_proposal/train_opd.py#L462)), then trained with PPO-clip ([L508–L527](../../src/co_scientist/d6_grant_proposal/train_opd.py#L508)).
  - A reward-only GRPO control ([`train_grpo.py`](../../src/co_scientist/d6_grant_proposal/train_grpo.py)) isolates the reviewer's contribution.
- **D5 μ-v4.** Historically labelled "SDPO" inside the code, but the teacher samples and the student is scored off-policy, so it is *not* canonical SDPO. The docstring and [`CANONICAL_NAMING_REFERENCE.md`](../../projects/d5_abstract_retrieve_refine/knowledge/current/CANONICAL_NAMING_REFERENCE.md) record this correction. I found the mislabel while checking the implementation against the papers.

## What was measured
- **D1.** SDPO runs (5 configurations) were screened on training reward only, and none beat GRPO. A GRPO-only control in the same code path ([`experiments/rubric_reward/sdpo/train_grpo_sanity.py`](../../experiments/rubric_reward/sdpo/train_grpo_sanity.py)) rules out code-path differences.
- **D5.** μ-v4 scored 28.00 / 45 against 25.25 for the frozen-model-plus-oracle baseline σ on an isolated Opus audit (8 plans each, one goal). Pairwise judging corroborated it 20 / 24 ([`DECISIONS.md`](../../projects/d5_abstract_retrieve_refine/DECISIONS.md)).
  - A later in-house variant lost 3–5 pairwise to the on-policy OPD variant, which falsified that pre-registered claim. The falsification is recorded in the same log.
  - Read this as a small, single-goal effect.
- **Mechanism finding (D5 F16).** Reading the trainer confirmed that the teacher is the current LoRA student. The signal therefore rewards *conformity to the critique*, not fidelity to the oracle. This motivated the [verifier-grounded reward](08-retrieval-and-grounding.md).
