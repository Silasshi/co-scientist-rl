# LLM-as-judge evaluation that resists its own biases

**Motivation:** when the training grader also does the evaluation, the reported gain can be a grader artefact. Every direction after D1 therefore paired the training reward with an **independent, cross-family judge** (Claude Opus). Each judge protocol was hardened against biases documented in the literature:
- Panel of LLM evaluators (PoLL): [arXiv:2404.18796](https://arxiv.org/abs/2404.18796)
- Position bias: Zheng et al., MT-Bench, [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Self-preference: Panickssery et al., [arXiv:2404.13076](https://arxiv.org/abs/2404.13076)

## Protocols
**1. Isolated, balanced, anonymised absolute audit** — [`d5_abstract_retrieve_refine/audit_v3_isolated.py`](../../src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py)

v1 batched plans by method, and the judge gave identical scores to whole batches ("templating"). v2 shuffled the order but was still anchored within a session. v3 does the following:
- Splits 56 plans (7 methods × 8) into 8 **balanced** batches, each holding exactly one plan per method ([L151–L154](../../src/co_scientist/d5_abstract_retrieve_refine/audit_v3_isolated.py#L151)).
- Anonymises plans within each batch (plan_A … plan_G).
- Scores each batch in a separate judge session.
- Requires a per-plan `key_differentiator` field, so the judge has to say how each plan differs.

**2. Pairwise preference with position swap** — [`shared/opus_pairwise_subagent.py`](../../src/co_scientist/shared/opus_pairwise_subagent.py)
- A and B are randomly swapped, with the flag stored outside the judge's view. Verdicts are decoded back after judging.
- Standing rule: any absolute-score gap below 5 / 45 must be confirmed pairwise, or re-audited strictly with one plan per judge session ([`experiments/.../audit_v3_isolated_2way_strict.py`](../../experiments/d5_abstract_retrieve_refine/audit_v3_isolated_2way_strict.py)).

**3. Asynchronous judge clients over a file bus** — [`shared/opus_audit_subagent.py`](../../src/co_scientist/shared/opus_audit_subagent.py), [`shared/opus_critic_subagent.py`](../../src/co_scientist/shared/opus_critic_subagent.py)
- The trainer writes request JSON files and keeps training. Claude Code subagents fan out one judge per request and write response files.
- The audit judge never sees the privileged source paper that the critic sees, so the evaluation cannot leak training-time information.

## What the protocols caught
- **A retraction.** A 7–1 pairwise "strong" headline for an inference-time pipeline fell to **4–4** once the plans were regenerated with a bug fix and judged again with the same protocol. It was retracted ([`D5 DECISIONS.md`, 2026-04-27 PM](../../projects/d5_abstract_retrieve_refine/DECISIONS.md)).
- **A falsified pre-registered claim.** "The v8 variant beats on-policy OPD" lost 3–5 pairwise. It was recorded as falsified rather than quietly dropped.
- **A reversed ranking.** The D3 training reward ranked plans against Opus within a saturated score band (see [multi-signal reward](03-multi-signal-vector-reward.md)).
