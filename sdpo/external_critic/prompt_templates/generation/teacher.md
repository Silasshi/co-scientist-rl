<!-- Source: src/co_scientist/d6_grant_proposal/prompts.py, build_teacher_prompt() -->

# D6 Generation — Teacher

**Role**: TEACHER context (with critique conditioning). Used by the OPD teacher
to compute the on-policy distillation logprob target.

**Template variables**: `{goal}`, `{oracle_abstraction}`, `{external_knowledge}`,
`{critique_xml}`

**Note for OPD distillation**: only the critique's `<improvement_directive>`
field functions as the conditioning signal that shifts the teacher's logprob
distribution; the other `<critique>` children are debug context.

---

I will provide you a research challenge, a set of methodological patterns extracted from a high-quality grant proposal on this same challenge, (optionally) external evidence, and a structured critique of a prior proposal attempt. Use the patterns to guide structure and the critique's improvement_directive to address concrete weaknesses. Do not copy phrasing literally.

Research Challenge: {goal}

# Methodological patterns from a reference proposal

{oracle_abstraction}

# External evidence

{external_knowledge}

# Critique of prior attempt (to address)

{critique_xml}

Write your grant proposal inside <solution>...</solution> tags. Target 1500 words, max 2000 words. Use this exact section structure:

## Specific Aims
## Research Strategy
### Significance
### Approach
### Evaluation Plan
### Timeline
### Deliverables
### Risk Mitigation

Be specific: name techniques, cite evidence, state measurable aims. Each section should carry concrete content — avoid hand-waving. Convey clear scientific impact: who benefits and how the field changes if the work succeeds.
