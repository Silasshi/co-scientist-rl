# Canonical Prompts — D4v2

*Locked: 2026-04-22. All future D4v2 runs MUST use these exact prompts for comparability.*

**Rule**: Do NOT modify any prompt below without creating a new version entry in DECISIONS.md and re-running baselines.

---

## 1. Policy: Fresh Plan Generation

**Source**: `src/co_scientist/grant_proposal/train_buffer_ttt.py:build_research_plan_prompt` (lines 241-288)

```
# Research Goal
{goal}

[# Relevant Published Papers                          ← only if use_retrieval=True
Use these real papers to ground your proposal with accurate citations.

{retrieved_papers}]

[# Past Attempts                                      ← only if fresh_use_context=True
Below are {N} past attempts at this goal with their evaluation scores (1-5 per dimension).

## Past Attempt 1  (aggregate reward: 0.XXX)
Signal scores: G1=X G2=X ...

{plan_text (truncated to 8000 chars)}

---]

# Output Format
Write a research plan for the goal above.

<think>
...your reasoning...
</think>
<solution>
...your research plan...
</solution>
```

---

## 2. Policy: Revision (Whole-Plan Rewrite)

**Source**: `src/co_scientist/grant_proposal/train_cr_v7.py:build_whole_plan_revision_prompt` (lines 366-432)

```
# Research Goal
{goal}

[# Relevant Published Papers                          ← only if retrieval active
Use these real papers to improve grounding and citations.

{retrieved_papers}]

# Current Plan
{plan_text}

# Per-signal feedback
{feedback_block}                                      ← mode-dependent, see below

# Output Format
Rewrite the plan addressing the feedback above.

<think>
...your reasoning...
</think>
<solution>
...your rewritten plan...
</solution>
```

### Feedback block modes

**Mode A: full_critique** (`scores_only=False, strip_critiques=False`)
```
- Problem Specificity (4/5): The proposal clearly names the gap between...
- Reasoning Depth (2/5): The proposal lacks multi-step causal chain reasoning...
- Evidence Rigor (3/5): Two citations provided but both are generic survey references...
...
```

**Mode B: scores_only** (`scores_only=True`)
```
- Problem Specificity (4/5): Does the outline name a SPECIFIC technical problem?
- Reasoning Depth (2/5): Does the proposal build multi-step causal chains?
- Evidence Rigor (3/5): Does the outline cite SPECIFIC technical evidence?
...
```

**Mode C: strip_critiques** (`strip_critiques=True`)
```
- Overall score: 0.725/1.0
```

---

## 3. Grader: Per-Signal Evaluation

**Source**: `src/co_scientist/shared/grant_signal_reward.py:build_single_signal_prompt` (lines 939-997)

### Preamble (shared across all signals)

```
You are an expert-level research plan evaluator. Your job is to assess a
research plan on a specific, well-defined evaluation dimension.

RULES:
- Evaluate based on what is ACTUALLY in the plan, not what it claims about itself.
- Follow the rubric strictly. Each score level has specific criteria.
- Be skeptical of surface-level plausibility — check whether claims are
  supported by concrete content.
- Do not demand rigor-theater (long lists of statistical procedures, exhaustive
  failure modes) when the plan is concise and focused. Judge the substance of
  what's there.
- Concise methodology that delegates boilerplate to citations is acceptable if
  the core is clearly described.
- If a signal has "Required reasoning" scaffolding, you MUST do the reasoning
  BEFORE assigning a score.
- Integer scores only: 1 through {score_max}.
```

### Per-signal template

```
{preamble}

You will evaluate the following research plan on ONE dimension: **{signal.name}**.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimension

## Signal: {signal.name} (id={signal.id})

**Question**: {signal.question}

**Required reasoning before scoring**:
{signal.cot_scaffolding}

**Scoring rubric**:
{signal.scoring_rubric}

---

# Output Format

<evaluation>
    <dim id="{signal.id}">
        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>
        <score>INTEGER 1-{signal.score_max}</score>
        <critique>In 1-3 sentences, explain WHAT content is missing, weak, or
        unconvincing for {signal.name} per the rubric above. Be actionable and
        specific; describe the deficiency conceptually. Do NOT quote plan text
        verbatim.</critique>
    </dim>
</evaluation>

Begin your evaluation now.
```

*Note: `<critique>` block only emitted when `emit_critique=True`. Locus block omitted (not used in D4v2).*

---

## 4. Hard Gates

**Source**: `src/co_scientist/shared/seven_signal_reward.py`

### Goal Contrast (lines 111-147)

Evaluates whether the plan is specifically tailored to the goal (not generic).
- Extracts 5-7 goal-specific elements
- Classifies each as DIRECT (2pts) / INDIRECT (1pt) / ABSENT (0pt)
- Computes score = earned / max_possible
- Pass threshold: margin ≥ 0.10 (goal_contrast_target - 0.5 * mean(alt_goals))

### Claim Verification (lines 150-186)

Extracts verifiable factual claims from the plan and classifies them.
- Categories: DATASET_REFERENCE, PAPER_CITATION, METHOD_NAME, BENCHMARK_NUMBER, etc.
- Role: SUPPORTING vs NAME_DROPPED
- Relevance: RELEVANT vs IRRELEVANT
- *Note: Disabled for D4 (71% false rejection rate). May re-enable in D4v2 with tuning.*

---

## 5. Opus Eval: Depth Audit

**Source**: `src/co_scientist/grant_proposal/opus_eval_agent.py:DEPTH_AUDIT_PROMPT` (lines 25-98)

**Model**: `claude-opus-4-7-20250430` (Opus 4.7)
**Label**: `standardized_depth_audit_v2_1to10`
**Scale**: 4 dimensions × 1-10 = /40 total

```
You are an expert grant proposal reviewer (NIH study section level). Score the
following research proposal on 4 independent dimensions, each 1-10 (integer).
Evaluate SUBSTANCE, not surface structure. A proposal that names many methods
or cites many references but lacks depth should score LOW.

Use the FULL 1-10 range. A mediocre proposal should score 4-5, not 6-7.
Reserve 9-10 for proposals that would impress a domain expert.

# Research Goal
{goal}

# Research Proposal to Evaluate
{plan}

# Scoring Dimensions

## 1. Problem Depth (1-10)
- 1-2: Restates the goal without analysis; no independent thinking
- 3-4: Names the problem with generic background; sub-problems identified but
  at a textbook level without revealing why they are hard
- 5-6: Shows domain knowledge; identifies specific sub-problems with concrete
  examples; begins to articulate why existing approaches fall short
- 7-8: Demonstrates genuine understanding; reveals specific tensions or
  trade-offs between competing objectives; explains the mechanism behind
  each gap, not just the gap itself
- 9-10: Reveals non-obvious insights that a non-expert would miss; reframes
  the problem in a way that opens new solution directions

## 2. Methodological Substance (1-10)
- 1-2: Lists method names without explaining how they apply
- 3-4: Describes standard procedures generically; equations are textbook
  restated without adaptation to this specific problem
- 5-6: Explains how specific methods address specific aims; some justification
  for method choices but alternatives not seriously considered
- 7-8: Justifies method choices with domain-appropriate reasoning; addresses
  limitations; explains why alternatives were rejected; formulas are
  adapted or derived for this problem, not just cited
- 9-10: Demonstrates methodological innovation with clear rationale; methods
  are tightly coupled to the problem structure in a non-obvious way;
  a reviewer would learn something from reading the methodology

## 3. Feasibility & Specificity (1-10)
- 1-2: Vague scope with no timeline or resource plan
- 3-4: Has structure but commitments are unrealistic or contradictory;
  arbitrary quantitative targets without justification
- 5-6: Reasonable scope with basic timeline and named datasets; some
  quantitative targets that are plausible but not well-justified
- 7-8: Specific, achievable milestones with acknowledged limitations;
  contingency plans for identified risks; resource allocation justified
- 9-10: Detailed work plan with fallback strategies; success criteria are
  operationalized with specific thresholds and statistical justification;
  team composition and timeline are realistic given the scope

## 4. Scholarly Grounding (1-10)
- 1-2: No real citations; fabricated references or completely generic
- 3-4: A few real citations but used as decoration, not integrated into
  the argument; some citations appear fabricated or misattributed
- 5-6: Several real citations properly contextualized; demonstrates
  awareness of the field but gaps in coverage of key prior work
- 7-8: Strong command of relevant literature; citations are integrated
  to build the argument; identifies a genuine gap in prior work
- 9-10: Synthesizes literature to identify a non-obvious gap; all citations
  verifiable and precisely attributed; positions the work relative to
  the state of the art with specific comparisons

IMPORTANT: For Dimension 4, be skeptical of citations. Check whether cited
papers actually exist and whether the attributed claims are plausible.
Fabricated references should score 1-4 regardless of other merits.

# Output Format
Respond with ONLY this JSON (no other text):
{"depth": <int>, "methods": <int>, "feasibility": <int>, "grounding": <int>}
```

---

## 6. Pairwise Preference (Judge Comparison)

**Source**: `projects/grant_proposal/analysis/sdpo_opus_eval/run_pairwise.py`

```
You are an expert grant proposal reviewer (NIH study section level). You will
see two research proposals (Plan A and Plan B) addressing the same research goal.

Choose which proposal is BETTER overall. Consider:
1. Problem depth and genuine understanding (not surface-level restating)
2. Methodological substance (adapted methods, not textbook descriptions)
3. Feasibility and specificity (realistic scope, concrete milestones)
4. Scholarly grounding (real citations, properly integrated)

Focus on SUBSTANCE over surface formatting. A longer proposal is not
necessarily better.

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

Which plan is better overall? Respond with ONLY a JSON object:
{"winner": "A" or "B", "reason": "one sentence justification"}
```

**Protocol**: Always run BOTH orders (AB, BA) to detect position bias. A judge with position bias on ≥2/3 pairs is unreliable.

**Validated judges** (0-1 position bias, ≥2/3 Opus agreement): Opus 4.7, GPT-5.4, o3.
