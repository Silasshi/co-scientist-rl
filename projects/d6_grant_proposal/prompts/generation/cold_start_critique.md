<!-- Source: src/co_scientist/d6_grant_proposal/prompts.py, COLD_START_CRITIQUE_NONE -->

# D6 Generation — Cold-start critique (iter 0)

**Role**: Iter-0 critique placeholder for the teacher prompt's `{critique_xml}`
slot, before any reviewer rollout has been run.

**Template variables**: none — `load_prompt` only.

**Children**: matches the 8-child schema enforced by `review/critic_instructions.md`.
Strengths/weaknesses are placeholders at iter 0; they only carry signal
once a real reviewer rollout exists.

---

<critique>
<idea_alignment>This is the first iteration; no prior proposal exists to align against. Focus on producing a well-structured proposal that addresses the methodological patterns provided.</idea_alignment>
<missing_components>To be evaluated against the reference proposal after generation. Ensure each pattern's key insights are followed, not skipped.</missing_components>
<incorrect_assumptions>To be evaluated against the reference proposal after generation. State assumptions explicitly so they can be checked.</incorrect_assumptions>
<feasibility>State a concrete scope, a realistic timeline framing, and specific evaluation criteria even if estimated. Avoid hand-waving on claims about feasibility or impact.</feasibility>
<clarity>Use the 8-heading reference template (Specific Aims / Research Strategy → Significance, Approach, Evaluation Plan, Timeline, Deliverables, Risk Mitigation). Define jargon on first use; do not assume reviewer familiarity with the sub-area.</clarity>
<strengths>To be filled by the reviewer after the proposal is written.</strengths>
<weaknesses>To be filled by the reviewer after the proposal is written.</weaknesses>
<improvement_directive>Name the specific problem with a concrete limitation (not just 'X is important'). State 2-3 measurable specific aims with expected outcomes. Cite at least 2 pieces of specific technical evidence (named methods, quantitative findings, or named systems). Justify each design choice with a mechanistic sentence (why this approach for this problem). Name 3+ specific deliverables. Identify 2-3 concrete failure modes and the key fragile assumption. Convey clear scientific impact: who benefits and how the field changes if successful.</improvement_directive>
</critique>
