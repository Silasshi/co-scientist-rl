<!-- Source: src/co_scientist/d6_grant_proposal/prompts.py, build_student_prompt() -->

# D6 Generation — Student

**Role**: STUDENT context (no critique). Used by OPD student rollouts, GRPO
group rollouts, frozen-baseline sampling, eval audit rollouts.

**Template variables**: `{goal}`, `{oracle_abstraction}`, `{external_knowledge}`

**Length and structure** target the empirical reference distribution (15
reference proposals, 1318–2038 words, mean 1639). The 8-heading template
matches every reference proposal in `dataset/{domain}/{goal}/reference_proposal.md`.

---

I will provide you a research challenge, a set of methodological patterns extracted from a high-quality grant proposal on this same challenge, and (optionally) external evidence to draw from. Use the patterns to guide structure and reasoning. Do not copy phrasing literally.

Research Challenge: {goal}

# Methodological patterns from a reference proposal

{oracle_abstraction}

# External evidence

{external_knowledge}

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
