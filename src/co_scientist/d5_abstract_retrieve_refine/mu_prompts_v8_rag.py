"""D5 μ-v8 RAG prompt builders — Exp C (oracle-transfer ABC).

Thin wrapper around mu_prompts_v8: the prompt body is identical (same footer,
same scaffolding), but the `oracle_abstraction` argument is expected to be a
top-K retrieved subset (~500-1000 words) instead of the full slim oracle
(~10K words). Function signatures are unchanged so the trainer can switch
between full-oracle and RAG via a config flag at the callsite.

Used by `train_mu_v8_d5sdpo.py` when `config.use_rag=True`. The retriever
that produces `oracle_retrieved` is `oracle_retriever_v1.OracleRetrieverV1`.

This file exists primarily to document the difference at the import level
(grep-discoverable that a callsite intentionally uses RAG vs full oracle)
and to give a future researcher a single place to diverge prompt phrasing
if needed (e.g., "# Top-K methodological patterns relevant to this scenario").
For now, the v8 prompts are reused verbatim because the bge retriever returns
items whose section headers (### Math 5: ...) already self-label as patterns.
"""
from __future__ import annotations

from co_scientist.d5_abstract_retrieve_refine.mu_prompts_v8 import (
    build_student_prompt_v8,
    build_teacher_prompt_v8,
)


def build_student_prompt_v8_rag(goal: str, oracle_retrieved: str) -> str:
    """STUDENT context for v8 RAG: goal + retrieved top-K items, NO critique.

    `oracle_retrieved` is the concatenation of top-K oracle items returned by
    `OracleRetrieverV1.retrieve(goal, k=...)`. Body and footer are identical
    to `build_student_prompt_v8` — only the input substring changes.
    """
    return build_student_prompt_v8(goal, oracle_retrieved)


def build_teacher_prompt_v8_rag(
    goal: str, oracle_retrieved: str, critique_xml: str
) -> str:
    """TEACHER context for v8 RAG: goal + retrieved top-K items + critique.

    `oracle_retrieved` should be retrieved against `(goal + critique_xml)` so
    that the teacher's retrieval reflects critique-addressed needs (per
    EXPERIMENT_PLAN_oracle_transfer_ABC.md L131-133). Body and footer
    identical to `build_teacher_prompt_v8`.
    """
    return build_teacher_prompt_v8(goal, oracle_retrieved, critique_xml)
