# prompts

**Source:** `src/co_scientist/d6_grant_proposal/prompts.py` (~120 lines)
**Stage:** Infrastructure (cross-cutting)

## Purpose

The Python-facing payload builders that the trainers and audit scripts call. Each builder is a thin wrapper around `prompt_loader.render_prompt(...)` plus the SDPO context formatting that D5's `mu_prompts.py` defined. Re-exports the D5 helpers `extract_solution`, `parse_critique_xml`, `build_sdpo_datum` so D6 trainers don't depend directly on D5 internals.

## Key callables

| Callable | Purpose |
|---|---|
| `build_student_prompt(goal, oracle_abstraction, external_knowledge="") -> str` | STUDENT context: goal + slim_oracle [+ optional RAG slot] |
| `build_teacher_prompt(goal, oracle_abstraction, critique_xml, external_knowledge="") -> str` | TEACHER context: STUDENT context + previous-iter `<critique>` |
| `build_critic_request_payload(goal, reference_proposal, plan_text) -> dict` | Reviewer (Opus) request: goal + privileged `reference_proposal.md` + the policy's plan |
| `build_audit_request_payload(goal, plans, domain) -> str` | Audit prompt: anonymized batch of plans + 12-signal scoring instructions (calls `audit/absolute_score.md` via `prompt_loader`) |
| `COLD_START_CRITIQUE_NONE` | Iter-0 placeholder for teacher's `critique_xml` slot (loaded from `generation/cold_start_critique.md`) |

## Templates loaded

- `generation/student.md` (target 1500w, max 2000w, 8 grant headings)
- `generation/teacher.md` (student template + `<improvement_directive>` conditioning channel)
- `generation/cold_start_critique.md`
- `review/critic_instructions.md` (8-child `<critique>` schema: directive + clarity + strengths + weaknesses + ...)
- `audit/absolute_score.md` (12-signal, 1–5, isolated batch)

## Dependencies

- `co_scientist.d6_grant_proposal.prompt_loader.load_prompt / render_prompt`
- `co_scientist.d5_abstract_retrieve_refine.mu_prompts` — re-exports `extract_solution`, `parse_critique_xml`, `build_sdpo_datum` (these are SDPO mechanics, identical across D5/D6)

## Capability

Pure infrastructure — does not train, evaluate, or score. Owned by every trainer + the audit script.

## Backward-compatibility note

A 24-test byte-equality suite (since replaced by structural invariants in `test_d6_prompt_loader.py`) verified that the loader-backed builders return identical bytes to the pre-refactor hardcoded versions. Do not modify a builder's call signature without re-running the test suite.
