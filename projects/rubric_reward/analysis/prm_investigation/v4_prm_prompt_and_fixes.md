# PRM Prompt Investigation Results & V4 Fixes

## Critical Finding: Parsing Bug

### The Bug

The PRM (evaluator) was producing negative scores all along, but a parsing bug silently converted them to 0.

**What happens:** When Gemini Flash gives a negative score, it often omits the opening `<score>` tag and starts the response directly with the score:

```
-1</score>
<hint>The response is too generic...</hint>
```

But our regex requires the opening tag:
```python
re.search(r"<score>\s*([+-]?\d+)\s*</score>", text)
```

This doesn't match `-1</score>` (no opening tag), so `parse_prm_response` defaults to score=0.

Positive scores work fine because the model outputs the full `<score>+1</score>` format.

### Impact on V3 Training

The bimodal distribution {0: 491, 2: 489} across 6 batches was actually:
- ~250 responses were truly scored **-1** by the evaluator (parsed as 0)
- ~240 responses were genuinely scored **0**
- ~489 responses were scored **+2**

This means:
- Per-turn advantages for ~250 turns were 0.0 instead of -0.33 (the model missed negative training signal)
- OPD hints from these turns still worked (they were extracted correctly from the `<hint>` tag)
- GRPO was unaffected (uses rubric scores, not PRM)

### Fix

Add a fallback regex in `parse_prm_response`:
```python
def parse_prm_response(text: str) -> tuple[int, str | None]:
    score = 0
    # Primary: <score>-1</score>
    score_match = re.search(r"<score>\s*([+-]?\d+)\s*</score>", text)
    if not score_match:
        # Fallback: response starts with or contains score without opening tag
        score_match = re.search(r"([+-]?\d+)\s*</score>", text)
    if score_match:
        try:
            score = max(-3, min(3, int(score_match.group(1))))
        except ValueError:
            score = 0
    # ... hint parsing unchanged
```

---

## Prompt Investigation Results

Tested 5 prompt variants on 15 conversations (5 bad, 5 mediocre, 5 good tutor responses):

| Variant | Negative Scores | Bad Mean | Mediocre Mean | Good Mean | Assessment |
|---------|----------------|----------|---------------|-----------|------------|
| A: Baseline | 1/15 (but 10 hidden by parsing bug) | 0.00 (actually -1.00) | 0.00 (actually -1.00) | 0.40 | **Good with parsing fix** |
| B: Few-shot examples | 3/15 | 0.00 | -0.40 | 0.80 | **Best spread** |
| C: Ternary (-1/0/+1) | 0/15 | 0.00 | 0.00 | 0.20 | Collapsed to 0 |
| D: Forced distribution | 0/15 | 0.00 | 0.00 | 0.00 | Complete collapse |
| E: Anchored questions | 2/15 | -0.60 | 0.00 | 1.20 | **Best bad-response detection** |

**Variant C (ternary) and D (forced distribution) performed worst** — simplifying the scale or instructing about distributions caused score collapse.

**Variants B and E performed best.** B produces more negative scores through examples; E produces correct extreme negatives through critical thinking questions.

---

## Recommended V4 Evaluator Prompt (Combined B + E)

```
You are evaluating a research collaborator's response in a discussion
with a researcher. You will see the full conversation.

Rate the collaborator's MOST RECENT response on a scale of -3 to +3.

### Before Scoring, Answer These Questions
1. Did the collaborator provide ANY specific method, algorithm, or
   approach? If no → score ≤ 0.
2. Did the collaborator explain WHY their suggestion fits THIS specific
   problem? If no → score ≤ +1.
3. Did the collaborator give actionable details (parameters, steps,
   metrics)? If no → score ≤ +1.
4. Could this advice apply to ANY research problem without modification?
   If yes → score ≤ 0.
5. Does the response contain filler phrases like "that's a great question"
   or "there are many approaches"? If yes → score -1.

### Scoring Scale
+3 EXCEPTIONAL: Breakthrough insight with specific, actionable details.
+2 VERY HELPFUL: Substantive with specific methods and reasoning.
+1 GOOD: Useful but somewhat generic. Right direction.
 0 NEUTRAL: Restates known info. No new actionable insight.
-1 WEAK: Vague, off-topic, or generic platitudes.
-2 POOR: Technical inaccuracies or ignores the question.
-3 HARMFUL: Fundamentally wrong advice. Filler text.

### Example Evaluations

Example 1 — Score: +2
Collaborator: "For handling class imbalance, I recommend focal loss with
gamma=2.0, combined with stratified sampling. Focal loss downweights easy
examples, which is ideal for your 1:50 imbalance ratio."
→ Specific method, parameter, justified for the problem.

Example 2 — Score: -1
Collaborator: "That's a great question! There are many approaches to
handle this. You might want to look into various techniques that have
been proposed in the literature."
→ No specific advice. Generic filler.

Example 3 — Score: 0
Collaborator: "So you want to improve the model's performance on rare
classes. That's indeed an important challenge in machine learning."
→ Restates the problem. No guidance.

Example 4 — Score: -2
Collaborator: "You should use PCA to reduce dimensionality before
applying your transformer model."
→ PCA destroys sequential structure. Technically misleading.

### Output Format
<score>INTEGER from -3 to +3</score>
<hint>1-3 sentences: what the collaborator SHOULD have said. "none" if excellent.</hint>
```

A separate plan-evaluation variant should be written following the same pattern but with plan-specific anchoring questions and examples.

---

## Implementation Checklist for V4

1. **Fix `parse_prm_response`**: Add fallback regex for missing opening `<score>` tag
2. **Replace PRM prompts**: Use combined B+E prompt (above) for discussion turns
3. **Write plan variant**: Same structure with plan-specific questions and examples
4. **Verify with test**: Re-run investigation script to confirm negative scores are parsed
5. **Consider**: Log raw PRM responses for the first few batches to catch any remaining parsing issues

---

## Files

- Investigation script: `analysis/prm_prompt_investigation.py`
- Raw results: `analysis/prm_investigation/prm_investigation_results.jsonl`
- Summary: `analysis/prm_investigation/prm_investigation_summary.txt`
- This document: `analysis/prm_investigation/v4_prm_prompt_and_fixes.md`
