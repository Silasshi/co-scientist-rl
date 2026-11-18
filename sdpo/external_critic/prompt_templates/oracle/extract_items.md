<!-- Source: src/co_scientist/d6_grant_proposal/extract_oracle.py, _EXTRACT_PROMPT -->

# D6 Oracle — Extract Items

**Role**: Sent to Opus to extract structured oracle items from a reference
grant proposal. Output is parsed into JSONL by `_parse_items()` in
`extract_oracle.py`.

**Template variables**: `{goal}`, `{reference_proposal}`

---

You are an expert research grant evaluator. You will read a high-quality grant proposal
and extract structured oracle items that capture its key patterns.

Extract items across these 6 categories:
1. **Insights**: High-level conceptual takeaways — what makes this proposal's framing strong
2. **Methodology**: Concrete algorithmic, analytical, or procedural components proposed
3. **Evidence**: Named citations, quantitative claims, named systems or datasets cited
4. **Structure**: How the proposal organizes its argument (section flow, logical dependencies)
5. **Deliverables**: Specific named outputs (papers, datasets, tools, guidelines, policies)
6. **Risk**: Failure modes, scope boundaries, or fragile assumptions explicitly stated

For each item, output:
<item>
<category>one of: Insights | Methodology | Evidence | Structure | Deliverables | Risk</category>
<text>The extracted pattern, written as a general principle (not quoting verbatim). 1-3 sentences.</text>
</item>

Extract 15-25 items total. Prioritize items that are most likely to help a writer
produce a high-quality proposal on the SAME topic. Do not include items that are
too topic-specific (e.g., specific numbers that only apply to this proposal).

# Research Challenge
{goal}

# Reference Proposal
{reference_proposal}

Now extract oracle items:
