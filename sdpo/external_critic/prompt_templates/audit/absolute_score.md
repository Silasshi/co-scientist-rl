<!-- Source: src/co_scientist/d6_grant_proposal/audit.py, _build_audit_prompt() -->

# D6 Audit — Absolute Score (multi-plan batch)

**Role**: Multi-plan, isolated 12-signal audit prompt. Each Opus subagent
receives a balanced batch of anonymized plans and scores every plan
independently on the active signals (G12 → G12a for social-science domain).

**Score scale**: 1–5 per signal (integer); aggregate = weighted mean (0.0–1.0).
NOT D5's /45.

**Template variables**: `{goal}`, `{n_plans}`, `{plans_text}`, `{n_signals}`,
`{signal_names}`, `{dim_lines}`. The caller computes `dim_lines` as
`"".join(f'  <dim id="{{s.id}}"><reasoning>...</reasoning><score>1-5</score></dim>\n' for s in active_signals)`.

---

You are an expert grant proposal evaluator. Score each plan independently on all 12 evaluation dimensions.

CRITICAL RULES:
- Score each plan INDEPENDENTLY. Do not anchor scores to other plans in this batch.
- Use the FULL scoring range (1-5). Do not cluster at 3.
- After scoring each plan, state 1 key differentiator: what makes this plan distinctly stronger or weaker than average on its highest or lowest signal.

# Research Challenge
{goal}

# Plans to evaluate ({n_plans} plans)

{plans_text}

# Evaluation Dimensions
Score each plan on these {n_signals} dimensions:
{signal_names}

For each plan, output:

<plan id="PLAN_ID">
<evaluation>
{dim_lines}
</evaluation>
<aggregate_reward>FLOAT (weighted mean, 0.0-1.0)</aggregate_reward>
<key_differentiator>1-2 sentences on what distinguishes this plan.</key_differentiator>
</plan>

Score EVERY plan. Output nothing after the final </plan> tag.
