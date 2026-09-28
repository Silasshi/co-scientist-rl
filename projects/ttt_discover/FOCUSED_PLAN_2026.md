# Focused Research Plan: Single-Goal × (Meta-Harness ∨ TTT-Discover) × Multi-Signal Critique

**Date**: 2026-04-09
**Owner**: Yuhong Shi
**Scope**: A narrowly-focused research plan for the specific direction of per-goal exploration/training with Meta-Harness or TTT-Discover, using a multi-signal self-critique reward with light RAG augmentation.

**Companion document**: `RESEARCH_PROGRAM_2026.md` (the comprehensive landscape map, used as reference)

---

## 0. Core Thesis

> **Can we close the co-scientist selection gap by abandoning large-scale full-dataset training entirely, and instead running focused per-goal exploration using search (Meta-Harness) or per-problem training (TTT-Discover), with a robust multi-signal self-critique reward that is partially anchored in external retrieval?**

The bet underlying this plan is that:
- The bestversion 0.693 ceiling reflects **full-dataset GRPO's limits**, not the base model's capacity (oracle = 0.85)
- Better leverage comes from **per-problem deep exploration** than from continued full-dataset training
- A **robust reward function** (multi-signal, partially grounded) is the prerequisite for using Meta-Harness or TTT-Discover safely (without Goodharting)
- **5-10 parallel single-goal experiments** give us enough statistical power to tell if the method works while keeping compute tractable

If this thesis is correct, we should see:
1. Meta-Harness finding inference-time harnesses that beat bestversion on individual goals
2. TTT-Discover training finding per-goal policies that beat bestversion on those goals
3. The multi-signal reward correlating with GPT-5.4 arena judgment (validating it's not Goodharted)
4. Per-goal improvements being consistent enough across 5-10 goals to rule out luck

If it's wrong, we'll learn why, and the comprehensive plan (`RESEARCH_PROGRAM_2026.md`) has fallback options.

---

## 1. Decisions Already Made

From our conversation, these are locked in:

| Decision | Choice |
|----------|--------|
| Experimental protocol | 5-10 parallel independent single-goal experiments |
| Goal domain | RL / project-related (not physics) |
| Budget philosophy | Quality-first, don't cap artificially |
| Reward function | 7-signal, 3-layer design (structural + rubric + coherence) with constrained aggregation |
| External grounding | Light RAG for Signal 3 (Methodology Convention Check); cached per goal |
| Base model | Frozen bestversion checkpoint (Qwen3-4B-Instruct-2507) for initial experiments |
| Training option | TTT-Discover-style LoRA training per goal (kept lightweight) |
| Search option | Meta-Harness-style code-space search per goal |

---

## 2. Phase Structure

```
Phase A: Reward Design & Validation          (1.5-2 weeks)
  ↓  [blocking gate: signals must be orthogonal and correlated with GPT-5.4 arena]
Phase B: Goal Selection & Baseline            (3-5 days)
  ↓
Phase C: Meta-Harness Track                   (2-3 weeks)  ┐
                                                           ├─ can run in parallel
Phase D: TTT-Discover Track                   (3-4 weeks)  ┘
  ↓
Phase E: Combination & Comparison             (1-2 weeks)
  ↓
Phase F: (OPTIONAL) Recursive Meta-Goal       (exploratory)
```

Phases C and D can run in parallel once Phase B is done. Phase A is a **hard gate** — if the signals don't validate, we rethink before building C/D.

---

## 3. Phase A — Reward Design & Validation (3-Layer Structural Design)

**Purpose**: Design and validate a robust, non-gameable reward function. Phase A is a **hard gate** — no training experiment proceeds until signals pass template-hack tests AND correlate with external truth.

**Revised from original 5-signal design** (see Section 3.0 below for why). The new design uses **7 signals across 3 layers**: structural tests (non-LLM), universal-rubric signals (LLM-judged + programmatic), and a coherence probe. Template-hack resistance is **structural**, not statistical.

### 3.0 — Why this differs from the original plan

The original plan had 5 LLM-judge signals: Goal-Method Alignment, Gap Detection, Convention Check (RAG), Decision Value, Specificity. After a 4-agent critique round (documented in conversation history), we found:

- The 5 signals had effective rank ~2 (not 5) — pairwise correlations of 0.5-0.65 on multiple pairs
- Template plans could score >0.82 average on all 5 signals while being scientifically worthless
- Using 5 different evaluator models doesn't fix the bias — RLHF'd frontier models share biases toward structure, length, confident claims
- Real scientific peer review has **zero inter-rater reliability** on NIH grants (Pier et al. PNAS 2018) — averaging LLM judges inherits this problem

The fix is to introduce **signals that aren't LLM judgments at all**:
- **Contrast-based signals** (plan scored against its goal vs. off-goals; templates score equally on all, so margin ≈ 0 → fail)
- **Verifier-based signals** (claims extracted and checked against databases; fabricated specifics fail)
- **Consistency signals** (cross-section checks; bag-of-sections plans fail)

Plus fewer, better-targeted LLM judge signals on dimensions where human reviewers actually have measurable agreement (soundness, specificity, feasibility — per Agent 2's NIH/NSF/NeurIPS literature review).

### The 3-layer design

```
┌─────────────────────────────────────────────────────────┐
│ LAYER 1: STRUCTURAL TESTS (not LLM judgments)            │
│   1.1  Goal-Contrast Margin      (contrast)              │
│   1.2  Claim Verification        (programmatic)          │
│   1.3  Internal Consistency      (schema checks)         │
├─────────────────────────────────────────────────────────┤
│ LAYER 2: UNIVERSAL RUBRIC (from real scientific review)  │
│   2.1  Methodological Soundness  (RAG, deviation-inv.)   │
│   2.2  Feasibility & Resources   (RAG, comparative)      │
│   2.3  Specificity (programmatic, not LLM-judged)        │
├─────────────────────────────────────────────────────────┤
│ LAYER 3: COHERENCE (process-shortcutting defense)        │
│   3.1  Narrative/Logical Coherence (forced citation)     │
└─────────────────────────────────────────────────────────┘
Aggregation: weighted mean - floor penalties (constrained)
```

Only 4 of the 7 signals involve LLM judgment; the other 3 are programmatic or contrast-based.

---

### A.0 — Template sanity check (✅ COMPLETE — PASSED)

**Status**: PASS as of 2026-04-10 after one v1→v2 prompt iteration.
**Full record**: [`analysis/_archive/phase_a0_jan2026/phase_a0_results.md`](analysis/_archive/phase_a0_jan2026/phase_a0_results.md) (archived)
**Runner**: `tools/_archive/phase_a0_sanity_check.py` (Qwen3-30B-A3B grader via Tinker, profile `new`) — archived; superseded by `src/co_scientist/shared/ten_signal_reward.py`
**Output JSON**: `analysis/_archive/phase_a0_jan2026/phase_a0_sanity_check_results.json`

**Purpose**: Before implementing any signal infrastructure, verify the design defeats known adversarial templates. If any template scores ≥ 0.30 aggregate, iterate before building.

**What we did**: Built a runner that exercises all 7 signals against 5 hack templates (3 from the original plan + 2 negative controls). Each signal makes ~9 grader calls per template; 25 minutes total runtime.

**The 5 templates** (full text in `docs/plans/signal_design/sanity_check_templates_v1.md`):

1. **Canonical Fortress** — generic-but-rigorous-looking, real ML name-drops, field-agnostic
2. **Reviewer Cosplay** — tiny fake plan + 10 "anticipated objections" pre-emption
3. **Specificity Bomb** — fabricated datasets/citations/benchmarks (should trigger fabrication hard gate)
4. **Empty Plan** (negative control) — 45 words of nothing
5. **Verbose Vacuum** — 460 words, 8 sections, every sentence a platitude

**Final v2 results** (all 5 < 0.30 aggregate):

| Template | g_c | c_v | i_c | snd | feas | spec | coh | **agg** |
|---|---|---|---|---|---|---|---|---|
| canonical_fortress | 0.143 | 0.800 | 1.000 | 0.150 | 0.440 | 0.440 | 0.500 | **0.079** |
| reviewer_cosplay | 0.400 | 0.733 | 0.750 | 0.000 | 0.580 | 0.556 | 0.200 | **0.000** |
| specificity_bomb | 0.143 | **0.000** ⚡ | 0.800 | 0.700 | 0.860 | 1.000 | 0.600 | **0.000** |
| empty_plan | 0.000 | 0.300 | 0.000 | 0.200 | 0.300 | 0.000 | 0.000 | **0.000** |
| verbose_vacuum | 0.000 | 0.300 | 0.000 | 0.300 | 0.300 | 0.000 | 0.200 | **0.000** |

⚡ = hard gate triggered (fabricated_ratio 25% > 10% threshold).

**v1 → v2 fixes** (5 root causes; full analysis in record doc):
1. Signal 1.1 rescale bug (`margin + 0.5` inflated 0.4 → 0.9) → use `min(target_score, 2·margin)`
2. Signal 1.1 DIRECT criterion too lenient → require methodological commitment + substitutability test
3. Signal 1.2 no relevance check → add RELEVANT/IRRELEVANT classification + fabricated_ratio > 10% hard gate
4. Signal 1.3 no VACUOUS handling → add VACUOUS as fail category
5. Signal 2.1 PURE_MIMICRY too lenient → require 3+ explicit goal-specific adaptations
6. Signal 3.1 generic links pass → add goal-specificity test ("could this link exist in a different goal's plan?")

**Key insight**: For canonical_fortress (the hardest template), no single signal caught it — the **aggregation did**. Floor-penalty compound effect on goal_contrast (0.143) + soundness (0.15) subtracted 0.389 from a weighted_mean of 0.468, producing 0.079. This validates the 3-layer defense-in-depth philosophy: the attack surface shrinks geometrically with layer count, not linearly.

**Known issues from v2 run** (queued for Phase A.0.5 / A.1):
- **#7.1**: Signal 2.1 has JSON parse brittleness on long thinking blocks (max_tokens or extraction fragility)
- **#7.2**: Signal 1.2 still gives canonical_fortress 0.8 because the model labels HumanEval/MMLU as "RELEVANT" (same broad field). Stress test will probe sharper relevance.
- **#7.3**: Signal 2.1 alone is fooled by fabricated specifics — the hard gate is the only defense for specificity_bomb-class templates. **Single biggest reliability risk.** Add Semantic Scholar lookup in A.1.
- **#7.5**: NO POSITIVE CONTROL yet — we've only verified the signals reject bad plans, not that they accept good ones. **Critical for A.0.5.**

---

### A.1 — Implement Layer 1: Structural Tests

This is the **most novel and highest-leverage** layer. These signals are what makes the design fundamentally different from prior art.

#### A.1.1 — Goal-Contrast Margin (Signal 1.1)

**Purpose**: A good plan is *goal-specific*. It scores high on its actual goal and LOW on different goals. Template plans are goal-agnostic and score equally on any goal → margin ≈ 0 → fail.

**Mechanism**:
```python
def goal_contrast_margin(plan, goal, target, alt_goals):
    """
    plan: the research plan being evaluated
    goal: the plan's actual research goal
    alt_goals: 2-3 alternative goals the plan should NOT be suitable for
    
    Returns: margin = score(plan|goal) - mean(score(plan|alt_goal))
    """
    judge = claude_sonnet_45  # or any frontier model
    score_target = judge.score_plan_against_goal(plan, goal, target)
    alt_scores = [judge.score_plan_against_goal(plan, alt_g.goal, alt_g.target)
                  for alt_g in alt_goals]
    return score_target - np.mean(alt_scores)
```

**Key design choices**:
- **How to pick alt_goals?** They must be different enough that a *specific* plan would score low on them, but similar enough (same domain) that the judge takes them seriously. Algorithm: sample from the eval set with **distance ≥ 0.5 in BERT embedding space** from the target goal.
- **How many alt_goals?** 2-3. More increases variance; fewer is cheap but less robust.
- **What does the judge prompt look like?**
  ```
  Research goal: {goal}
  Research target: {target}
  
  Here is a research plan. Score from 0-1 how well this plan ADDRESSES 
  THIS SPECIFIC RESEARCH GOAL. A plan that could equally well address 
  any similar goal should score low; a plan that is tailored to answer 
  precisely this question should score high.
  
  Plan:
  {plan}
  
  Score 0-1:
  ```
- **Expected template behavior**: All 3 attack templates are goal-agnostic, so score_target ≈ alt_scores → margin ≈ 0.

**Compute**: 3 judge calls per plan evaluation (1 target + 2 alt goals).

**Files**:
- `src/co_scientist/signals/layer1_goal_contrast.py`
- `src/co_scientist/retrieval/alt_goal_sampler.py`

#### A.1.2 — Claim Verification (Signal 1.2)

**Purpose**: Extract factual claims from the plan, verify each against external sources (databases, prior publications). Fabricated specifics fail.

**Mechanism**:
```python
def claim_verification(plan, retrieved_context):
    """
    Returns: fraction of extracted claims that verify against external sources.
    """
    # Stage 1: extract claims with schema-constrained small LM
    claims = extract_claims_schema(plan, schema={
        'dataset_references': 'named datasets (e.g., ImageNet, HumanEval)',
        'baseline_results': 'numerical claims about prior work performance',
        'paper_citations': 'citations to prior work (by name or arxiv ID)',
        'sample_size_claims': 'stated N with or without power calculation',
        'hyperparameter_values': 'specific numerical values with context',
    })
    
    verified = 0
    total = len(claims)
    if total == 0:
        return 0.3  # suspicious: no verifiable claims at all
    
    for claim in claims:
        if claim.type == 'dataset_reference':
            # Check against Hugging Face / Semantic Scholar / static list
            verified += check_dataset_exists(claim.value)
        elif claim.type == 'baseline_result':
            # Check against Papers with Code, cross-reference with S2
            verified += check_baseline_number(claim.value, claim.context)
        elif claim.type == 'paper_citation':
            verified += check_paper_exists(claim.value)  # S2/OpenAlex
        elif claim.type == 'sample_size_claims':
            # Programmatic: does the stated N match the stated power + effect size?
            verified += check_power_calculation_consistency(claim)
        elif claim.type == 'hyperparameter_values':
            # Is the value in a sane range for the claimed method?
            verified += check_hyperparameter_sanity(claim)
    
    return verified / total
```

**Key design choices**:
- **What counts as a "claim"?** Initial taxonomy: 5 types (dataset, baseline, citation, sample size, hyperparameter). Iterate based on what the extractor finds in real plans.
- **Extractor model**: Small schema-constrained model. Candidates: a fine-tuned Qwen3-1.5B or Llama 3.2 1B with structured outputs. Must be FAST (~100ms per plan) and RELIABLE (schema-valid output).
- **Verification sources**:
  - Datasets: Hugging Face Hub + manual list of canonical datasets
  - Baselines: Papers with Code API + Semantic Scholar
  - Citations: Semantic Scholar (free, 200M papers) + OpenAlex fallback
  - Sample size: programmatic power calculation check (scipy.stats.power)
  - Hyperparameters: heuristic sanity check against common ranges

**Handling of "no claims extracted"**: Returns 0.3 (not 1.0!) because a plan with zero verifiable claims is suspicious — it's either too vague to verify or avoiding specifics to escape detection.

**Compute**: 1 small-LM call for extraction + 5-15 API lookups per plan. ~0.5s latency. Essentially free ($).

**Files**:
- `src/co_scientist/signals/layer1_claim_verification.py`
- `src/co_scientist/verifiers/dataset_checker.py`
- `src/co_scientist/verifiers/citation_checker.py`
- `src/co_scientist/verifiers/power_calc_checker.py`
- `src/co_scientist/verifiers/hyperparameter_checker.py`

#### A.1.3 — Internal Consistency (Signal 1.3)

**Purpose**: Cross-section consistency checks. A plan where Methods describes X but Expected Results assumes Y is internally inconsistent, regardless of how rigorous each section looks.

**Mechanism**:
```python
def internal_consistency(plan):
    """
    Returns: fraction of consistency checks that pass.
    """
    sections = parse_sections(plan)  # robust markdown parser
    
    checks = [
        check_methods_to_expected_results(sections),
        check_sample_size_to_power_claim(sections),
        check_timeline_to_scope(sections),
        check_hypothesis_to_analysis_plan(sections),
        check_background_claims_to_methods(sections),
        check_control_to_intervention(sections),
    ]
    
    return sum(c.passed for c in checks) / len(checks)
```

Each check is a small LLM call with strict schema. Example:
```python
def check_methods_to_expected_results(sections):
    prompt = f"""
    Methods section: {sections['methods']}
    Expected Results section: {sections['expected_results']}
    
    Does the Methods section, if executed, produce the measurements/outputs
    described in Expected Results? Answer YES or NO, and cite the specific
    mismatch if NO.
    
    Output: {{"consistent": bool, "evidence": str}}
    """
    return small_llm_schema(prompt)
```

**Key design choices**:
- **Section parsing**: Must be robust to markdown variations. Fall back to heuristic splits if headers are missing.
- **Check evaluator**: Small LLM (4B-class), schema-enforced. Fast, cheap, deterministic.
- **What checks to include**: Start with 6 checks (listed above). Add more based on observed failure modes during validation.

**Compute**: 6 small-LM calls per plan, ~2s total. Near-free.

**Files**:
- `src/co_scientist/signals/layer1_internal_consistency.py`
- `src/co_scientist/parsers/section_parser.py`

---

### A.2 — Implement Layer 2: Universal Rubric Signals

These are LLM-judge signals, but on dimensions where the Agent 2 literature review showed humans have measurable inter-rater agreement.

#### A.2.1 — Methodological Soundness (RAG, deviation-inverted) (Signal 2.1)

**Purpose**: Score methodological soundness with reference to canonical field practice, but with an **inversion** from the original: pure mimicry of retrieved papers is PENALIZED, not rewarded. Justified deviation with reasoning is rewarded as much as or more than following convention.

**Mechanism**:
```python
def methodological_soundness(plan, goal, target, retrieved_context):
    """
    Key inversion: convention mimicry = penalty, justified novelty = bonus.
    """
    judge = gemini_3_pro
    prompt = f"""
    Research goal: {goal}
    Research target: {target}
    
    Retrieved related papers (showing field conventions):
    {retrieved_context}
    
    Submitted plan: {plan}
    
    Assess the plan's methodological soundness. Consider:
    
    1. SOUNDNESS: Is the methodology scientifically valid for this question?
    2. SIMILARITY TO RETRIEVED PAPERS: Is the plan copying retrieved methodology
       verbatim, or adapting it to the specific goal?
    3. DEVIATION FROM CONVENTION: If the plan deviates from canonical
       approaches, is the deviation explicitly justified with reasoning?
    
    Scoring:
    - Pure mimicry of a retrieved paper (plan ≈ copy): 0.3
    - Convention applied with minor adaptation to the goal: 0.6
    - Convention applied with goal-specific thought: 0.8
    - Justified deviation from convention: 0.8-1.0
    - Unjustified deviation: 0.2-0.4
    - Methodologically unsound (regardless of novelty): 0.0-0.2
    
    Score 0-1 and cite specific evidence.
    """
    return judge.score(prompt)
```

**Key change from original**: Convention mimicry is penalized, not rewarded. This prevents plans from scoring high by copying retrieved abstracts.

**Compute**: 1 judge call per plan.

#### A.2.2 — Feasibility & Resource Realism (Signal 2.2)

**Purpose**: Can the plan actually be executed with the stated resources/timeline? Compare to retrieved similar studies.

**Mechanism**:
```python
def feasibility(plan, goal, target, retrieved_context):
    """
    Compare stated resources (team, timeline, compute) to retrieved similar studies.
    Flag >2x mismatches.
    """
    judge = claude_sonnet_45
    prompt = f"""
    Research goal: {goal}
    
    Retrieved similar studies (for resource comparison):
    {retrieved_context}
    
    Submitted plan: {plan}
    
    Assess feasibility:
    
    1. Extract stated resources from the plan: team size, timeline,
       compute budget, sample size, data requirements.
    2. Estimate what similar studies in the retrieved literature typically required.
    3. Flag mismatches:
       - Plan claims to complete in 1/2 the time of typical studies → suspicious
       - Plan requires resources ≥ 2x beyond typical → infeasible
       - Plan omits key resource requirements → penalty for vagueness
    4. Judge: is this plan executable as stated?
    
    Score 0-1 where 1.0 = "clearly executable" and 0.0 = "impossible".
    """
    return judge.score(prompt)
```

**Compute**: 1 judge call per plan.

#### A.2.3 — Specificity via Programmatic Check (Signal 2.3)

**Purpose**: Measure specificity WITHOUT using an LLM judge, which is hackable. Use schema-constrained extraction to count concrete vs. vague claims.

**Mechanism**:
```python
def specificity(plan):
    """
    Programmatic specificity check. NOT LLM-judged.
    """
    extraction = small_llm_schema(plan, schema={
        'concrete_claims': 'claims with named entities, specific numbers, or concrete parameters',
        'vague_claims': 'claims using general/abstract language without specifics',
        'missing_elements': 'places where specifics would be expected but absent',
    })
    
    n_concrete = len(extraction['concrete_claims'])
    n_vague = len(extraction['vague_claims'])
    n_missing = len(extraction['missing_elements'])
    total = n_concrete + n_vague + n_missing
    
    return n_concrete / max(total, 1)
```

**Critical**: This signal is WORTHLESS on its own (can be gamed by fabricated specifics). Its value comes from **combining with Signal 1.2 (Claim Verification)** — specificity matters only if the specifics are also VERIFIED. The aggregation will down-weight Signal 2.3 when Signal 1.2 is low.

**Compute**: 1 small-LM call per plan.

---

### A.3 — Implement Layer 3: Coherence Probe

#### A.3.1 — Narrative/Logical Coherence (Signal 3.1)

**Purpose**: This is the ONE signal that directly attacks the documented "process shortcutting" failure mode (plans as bag-of-sections with no inter-dependencies).

**Mechanism**:
```python
def coherence(plan):
    """
    Forces the judge to cite SPECIFIC cross-section dependencies.
    Plans where no dependencies can be cited score 0.
    """
    judge = gpt_54
    prompt = f"""
    Plan: {plan}
    
    Analyze the narrative and logical coherence of this plan. 
    
    REQUIRED: For each of the following dependency types, you MUST cite
    a specific passage from one section that depends on a specific passage
    in another section. If you cannot find concrete citations, that type
    gets 0 for this plan.
    
    Dependency types to check:
    1. Does Methods depend on Hypothesis? (specific mechanism reflects hypothesis)
    2. Does Expected Results depend on Methods? (outputs match procedures)
    3. Does Analysis Plan depend on Hypothesis? (tests measure the claim)
    4. Does Timeline depend on Methods? (duration matches complexity)
    5. Does Sample Size depend on Statistical Power claims?
    
    For each dependency:
    - Cite the specific text from each section
    - Explain how one depends on the other
    - Mark PRESENT, WEAK, or ABSENT
    
    Score = fraction marked PRESENT.
    """
    return judge.score(prompt)
```

**Why "forced citation" matters**: Without the citation requirement, the judge will say "yes the plan is coherent" for almost any plan. Forcing specific text citations makes it impossible to fake — if there are no dependencies, the judge has nothing to cite.

**Compute**: 1 judge call per plan.

---

### A.4 — Aggregation with Floor Penalties

**NOT a simple weighted mean**. Use constrained aggregation inspired by Agent 4's matrix design.

```python
def aggregate_reward(signals: dict) -> float:
    """
    signals: {signal_name: score in [0, 1]}
    """
    weights = {
        'goal_contrast_margin': 0.15,
        'claim_verification':   0.15,
        'internal_consistency': 0.10,
        'soundness':            0.15,
        'feasibility':          0.10,
        'specificity':          0.10,
        'coherence':            0.25,  # highest weight - process shortcutting defense
    }
    
    floors = {
        'goal_contrast_margin': 0.1,   # margin must be positive
        'claim_verification':   0.4,   # must verify at least 40% of claims
        'internal_consistency': 0.5,   # must pass at least half of checks
        'soundness':            0.3,
        'feasibility':          0.3,
        'specificity':          0.2,   # low floor because it's combined with verification
        'coherence':            0.3,   # at least some dependencies must be cited
    }
    
    weighted_mean = sum(weights[k] * signals[k] for k in signals)
    
    # Floor penalties: 5x penalty for every point below floor
    penalty = 0
    for k, score in signals.items():
        if score < floors[k]:
            penalty += 5 * (floors[k] - score) * weights[k]
    
    return max(0, weighted_mean - penalty)
```

**Effect**: A plan that aces 6 signals but has internal_consistency = 0.2 (below floor of 0.5) gets hit with penalty = 5 × (0.5 - 0.2) × 0.10 = 0.15. Even if its weighted mean was 0.8, final score = 0.65. This catches plans that cheat on specific dimensions.

**Aggregation file**: `src/co_scientist/signals/aggregate_constrained.py`

---

### A.5 — Build Retrieval Infrastructure (for Signals 2.1 and 2.2)

Same as original plan: Semantic Scholar API, cached per-goal, used by signals 2.1 and 2.2 (both need retrieved canonical approaches).

**No change from original A.2** — this part of the original plan still holds. 5-10 S2 queries per goal, cached once.

**Files**:
- `src/co_scientist/retrieval/semantic_scholar_client.py`
- `src/co_scientist/retrieval/goal_context_cache.py`
- `data/goal_retrieval_cache/`

---

### A.6 — Validate All 7 Signals on Existing Data

**What**: Using existing bestversion eval outputs, validate the full 7-signal pipeline.

**Validation tests**:

1. **Template hack test (CRITICAL)**: Implement the 3 attack templates (Canonical Fortress, Reviewer Cosplay, Specificity Bomb). Run them through the 7-signal pipeline. **Pass criterion: all 3 templates score < 0.3 aggregate.** If any scores > 0.4, iterate on signal design.

2. **Orthogonality**: Pairwise correlation matrix of 7 signals on 500 existing plans.
   - **Pass**: All off-diagonal correlations < 0.7
   - **Expected**: Layer 1 signals should be the MOST independent (they measure structural properties, not "quality"); Layer 2 signals may correlate slightly more; Coherence should be independent of everything

3. **Correlation with GPT-5.4 arena**: For 200 plans with known GPT-5.4 arena scores, compute Spearman correlation per signal.
   - **Pass**: Aggregate ≥ 0.7 correlation with GPT-5.4 arena
   - **Per-signal**: Layer 1 structural signals may have LOWER correlation with GPT-5.4 (that's OK — they measure different things). Layer 2 + Coherence should correlate ≥ 0.4.

4. **Rank-ordering on known pairs**: On 50 plan-pairs with clear GPT-5.4 preferences, does the aggregate agree? **Pass ≥ 75%.**

5. **Computed scores on bestversion outputs**: Distribution should NOT be degenerate. Median aggregate should be in [0.4, 0.7]. If all plans score near 1.0 → signals are too lenient. If all near 0.0 → too strict.

**Decision point**: If Phase A.6 fails on the template test, we STOP and iterate. Signals must defeat known templates before we trust them for training.

**Compute**: ~500 plans × 7 signals = ~3,500 signal evaluations (~3,500 LLM calls + claim verification API calls). Estimated $80-150.

**Files**:
- `scripts/validate_7_signals.py`
- `scripts/run_template_hack_test.py`
- `analysis/signal_validation_report.md`

---

### A.7 — Document the Validated Signal Design

**What**: Once A.6 passes all tests, freeze the design in `docs/signal_design_v1.md`. This becomes the locked reward for Phases C-E.

**Include**:
- Full specification of each signal (prompts, schemas, thresholds)
- Validated correlations and template-hack test results
- Aggregation formula
- Known limitations
- Cost per evaluation (~$0.15-0.25 per plan, dominated by Layer 2 LLM calls)
- Fallback procedure if any signal fails during training runs

**Output**: Frozen `docs/signal_design_v1.md`.

---

### Phase A compute summary

| Component | One-time | Per-plan | Notes |
|-----------|----------|----------|-------|
| Template sanity check (A.0) | ~0 (paper) | — | Hand-computed |
| Retrieval infrastructure setup (A.5) | ~100 LLM calls | — | One-time per goal, cached |
| Per-goal retrieval (A.5) | ~1 LLM call + 5 S2 calls | — | Cached, amortized |
| Signal validation (A.6) | ~3,500 LLM calls | — | One-time validation run |
| **Training-time per-plan cost** | — | 4 LLM judges + 2 small-LM calls + API lookups | ~$0.15-0.25/plan |

**Total Phase A cost**: ~$150-300 (validation) + $0 ongoing infrastructure. Reasonable.

**Phase A total time**: 1.5-2 weeks (more than original 1 week, because Layer 1 infrastructure is novel).

---

## 4. Phase B — Goal Selection & Baseline

**Purpose**: Pick the 5-10 research goals, establish baselines, prepare the experimental infrastructure.

### B.1 — Goal selection

**Criteria**:
- **RL/ML-related** (not physics, per your decision)
- **Diverse within the domain**: 2-3 goals each from roughly:
  - RL training methods (e.g., "how to improve PPO on sparse reward environments")
  - Alignment/RLHF (e.g., "how to reduce reward hacking in self-rewarding LLMs")
  - Evaluation/benchmarking (e.g., "how to design a benchmark for agent long-horizon planning")
  - Specific technical challenges (e.g., "how to train value functions for long-horizon credit assignment")
- **Moderate difficulty**: choose goals where bestversion's current rubric score is in the 0.5-0.8 range (not already saturated, not too hard)
- **Available reference plan**: each goal must have a reference plan in the dataset (for baseline comparison)

**Process**:
1. Filter the full eval set (552 goals) by domain (title/target contains RL/ML keywords)
2. Among those, rank by "how much room to improve" (1 - current bestversion rubric score)
3. Select 10 goals stratified across the 4 sub-areas above

**Output**: `data/focused_goals.json` with 10 goals + metadata.

**Compute**: None (reading existing data).

**Time**: 1 day.

### B.2 — Build per-goal retrieval contexts

**What**: For each of the 10 goals, run the retrieval pipeline from A.2 to build and cache the retrieval context. This happens ONCE.

**Output**: `data/goal_retrieval_cache/{goal_id}.json` per goal.

**Compute**: 10 goals × 5-10 S2 queries × 1 LLM call = ~10 LLM calls. Trivial cost.

**Time**: 1 day.

**Manual check**: Eyeball 3-5 goal contexts. Are the retrieved papers actually relevant? If not, adjust retrieval prompt.

### B.3 — Baseline: bestversion on each goal

**What**: For each of the 10 goals, generate K=32 plans from the frozen bestversion checkpoint (no training). Grade them using:
- Current rubric (for historical comparison)
- Reference-anchored rubric (if Phase 0 of comprehensive plan showed it works)
- 7-signal aggregate (our validated signal)
- GPT-5.4 arena (external anchor)

For each goal, record:
- Mean and best-of-32 under each metric
- The actual best plan text (will be "reference best" for later comparisons)

**Output**: `runs/focused/baseline/{goal_id}/` with 32 plans per goal + all metrics.

**Compute**: 10 goals × 32 plans × (1 generation + 4 grading systems × 1 call each) = 1,600 LLM calls. ~$30-60.

**Time**: 2-3 days with parallelism.

---

## 5. Phase C — Meta-Harness Track

**Purpose**: Test whether Meta-Harness-style code-space search, applied per-goal with the 7-signal reward, can produce inference-time improvements over bestversion.

### C.1 — Build Meta-Harness scaffold for single-goal

**What**: Create a minimal Meta-Harness infrastructure. The "harness" is a Python file defining how bestversion is wrapped to generate a plan for a specific goal.

**Harness interface**:
```python
# harness_template.py
from co_scientist.signals import evaluate_4_signal
from co_scientist.policies import frozen_bestversion

def generate_plan(goal, target, retrieved_context):
    """Return the best plan for (goal, target) using frozen bestversion."""
    # Harness code goes here
    ...

def evaluate_plan(goal, target, plan, retrieved_context):
    """Return 7-signal score."""
    return evaluate_4_signal(goal, target, plan, retrieved_context)
```

The **Proposer** is Claude Code (Opus 4.6) launched via the Agent tool. Its job: read past harness candidates and their scores, propose a new harness variant, run it, evaluate, and iterate.

**Architecture per goal**:
```
experiments/meta_harness/{goal_id}/
├── baseline_harness.py        ← starts as wrapper around bestversion
├── candidates/
│   ├── cand_001/
│   │   ├── harness.py
│   │   ├── score.json          ← 7-signal score on this goal
│   │   ├── traces/            ← plan generations + signal evaluations
│   │   └── analysis.md        ← Proposer's reasoning for this candidate
│   ├── cand_002/
│   └── ...
└── best_so_far.json
```

**Files**:
- `src/co_scientist/meta_harness/proposer.py` (wraps Claude Code)
- `src/co_scientist/meta_harness/evaluator.py` (runs harness + 7-signal)
- `src/co_scientist/meta_harness/archive.py` (filesystem management)

**Time**: 1 week to build and test on 1 goal.

### C.2 — Launch parallel single-goal Meta-Harness runs

**What**: Run the Meta-Harness loop on each of the 10 goals independently. Each goal gets its own archive and its own Proposer session.

**Per-goal budget**: 20 Meta-Harness iterations = up to 40 candidate harnesses. Each candidate evaluated on 32 generations (same as baseline K=32). So per goal: ~40 × 32 = 1,280 plan generations + 1,280 × 4 signal calls = ~6,400 LLM calls.

**Per goal cost**: ~$100-200.
**Total for 10 goals**: ~$1,000-2,000.

**Launch strategy**: Because Proposer runs are long (~8-12 hours each with Opus 4.6), run them sequentially (one goal per day) rather than truly parallel. 10 goals in 10-15 days.

Alternative: Use smaller goal batches (5 goals first, analyze, then 5 more) to catch infrastructure bugs early.

**Files**:
- `scripts/run_meta_harness_single_goal.py`
- `scripts/run_meta_harness_batch.py` (orchestrates 10 parallel goals)

**Expected output per goal**:
- Best harness found
- 7-signal score of best harness vs. baseline
- Best harness's plan for the goal (for manual inspection)
- Full archive for post-hoc analysis

### C.3 — Cross-goal analysis

**Questions to answer**:
1. **Does Meta-Harness actually improve over bestversion on individual goals?** Measure: what fraction of 10 goals see improvement? If 8-10/10 → method works. If 3-5/10 → method is lucky. If 0-2/10 → method doesn't work.
2. **Does the same harness pattern win across goals?** Compare the best harness from different goals. If they're structurally similar → there's a transferable pattern. If radically different → per-goal is fundamentally the right unit.
3. **Are Meta-Harness's gains correlated with GPT-5.4 arena?** Critical Goodharting check. If 7-signal rises but GPT-5.4 doesn't → we're gaming the signal.
4. **What did the Proposer actually do?** Qualitative read of 2-3 Proposer trajectories. Is it finding real improvements or exploiting reward quirks?

**Output**: `analysis/phase_c_meta_harness_results.md`.

**Time**: 2-3 days of analysis after runs complete.

### C.4 — Decision point

After Phase C:
- **If Meta-Harness works (8+/10 goals improved, GPT-5.4 confirms)**: Promote to primary recommendation. Document discovered harness patterns. Consider scaling to full eval set.
- **If partial (3-7/10 goals)**: Analyze which goals work and why. Might need per-domain specialization.
- **If fails (0-2/10)**: Meta-Harness isn't the right tool for this task. Fall back to Phase D (TTT-Discover) results, consider why.

---

## 6. Phase D — TTT-Discover Track

**Purpose**: Test whether per-goal LoRA training with entropic objective and PUCT reuse can produce goal-specific improvements.

### D.1 — TTT-Discover scaffold (lightweight variant)

**Key simplifications from the original paper**:

- **Rollout count**: Original TTT-Discover uses 512 rollouts × 50 steps = 25,600 per problem. For us: **64 rollouts × 15 steps = 960 per goal**. This is ~27× cheaper.
- **LoRA rank**: Paper uses rank 32. We use rank 16 (cheaper, still meaningful).
- **Entropic objective**: Exactly as paper. Adaptive β via KL budget γ = ln(2).
- **PUCT reuse**: Exactly as paper. Archive caps at 100 states per goal.
- **Reward**: 7-signal aggregate (from Phase A).

**Mathematical detail** (entropic objective):
```
J_β(θ) = E[ log E[ e^(β · R(s,a)) ] ]
w_β(a) = e^(β · R(s,a)) / Z
gradient: ∇J = E[ w_β(a) · ∇ log π_θ(a|s) ]
```

Where `R(s,a)` is the 7-signal aggregate reward and β is adapted per-state so KL(q_β || π_θ) ≈ ln(2).

**Files**:
- `src/co_scientist/trainers/ttt_discover/entropic_loss.py`
- `src/co_scientist/trainers/ttt_discover/puct_buffer.py`
- `src/co_scientist/trainers/ttt_discover/adaptive_beta.py`
- `src/co_scientist/trainers/ttt_discover/train_single_goal.py`

**Time**: 2 weeks to build and test on 1 goal.

### D.2 — Single-goal TTT-Discover runs (parallel)

**Per-goal protocol**:
1. Start from frozen bestversion as base policy
2. Initialize LoRA adapter (rank 16) on top
3. For 15 iterations:
   - PUCT sample a batch of 64 warm-start states from the archive
   - Generate 64 candidate plans (policy with current LoRA)
   - Compute 7-signal reward for each
   - Store in archive
   - Compute entropic objective gradient
   - LoRA update (Adam, lr 1e-5)
4. Final output: the best plan in the archive (by 7-signal)

**Compute per goal**: 960 plan generations + 960 × 4 signal calls = 4,800 LLM calls. Plus LoRA training overhead (~1-2 GPU-hours per goal with a 4B policy).

**Per goal cost**: ~$150-300 + compute.
**Total for 10 goals**: ~$1,500-3,000 + 10-20 GPU-hours.

**Launch strategy**: Run 2-3 goals in parallel on separate GPUs (if available) to speed up wall-clock. Total: ~1 week for all 10 goals.

**Critical monitoring** (per goal, per iteration):
- Reward trajectory (does it climb?)
- Policy entropy (does it collapse? Warning if drops >50% from start)
- Diversity of sampled plans (Self-BLEU; warning if rises sharply)
- Correlation between 7-signal and GPT-5.4 on a held-out sample (every 5 iterations)

If any warning triggers → stop that goal's run, investigate.

### D.3 — Cross-goal analysis

**Same questions as Phase C.3**:
1. Does per-goal training improve over baseline on individual goals?
2. Are the improvements consistent across goals?
3. Does GPT-5.4 confirm the improvements?
4. Does the 7-signal reward remain stable (not collapsed)?

**Additionally**:
5. **Compare to Phase C**: For goals where both Meta-Harness and TTT-Discover were run, which wins? Is one consistently better or is it goal-dependent?

**Output**: `analysis/phase_d_ttt_discover_results.md`.

### D.4 — Decision point

- **If TTT-Discover works**: Quantify the improvement and stability. Major finding.
- **If partial**: Analyze failure modes. Was it entropy collapse? Reward hacking? Insufficient iterations?
- **If fails**: TTT-Discover may not transfer to subjective tasks even with a good reward. Document why.

---

## 7. Phase E — Combination & Comparison

**Purpose**: Test whether combining Meta-Harness and TTT-Discover gives a multiplicative effect, or whether one dominates.

### E.1 — Meta-Harness + TTT-Discover (sequential combination)

**Experiment variant 1**: Meta-Harness first, then TTT-Discover on the best harness.
1. Take the best harness from Phase C for each goal
2. Use that harness's generation strategy as the base policy for TTT-Discover
3. Run TTT-Discover for 10 iterations
4. Measure: does this beat Meta-Harness alone?

**Experiment variant 2**: TTT-Discover first, then Meta-Harness on the trained LoRA.
1. Take the best LoRA from Phase D for each goal
2. Feed this as the base policy to Meta-Harness
3. Run Meta-Harness for 10 iterations
4. Measure: does this beat TTT-Discover alone?

**Questions**:
- Does one combination order work and the other not?
- Is the combination substantially better than either alone?
- What's the marginal gain from the second stage?

**Compute**: ~2× the per-goal cost of C or D (for the second stage). Cap at 5 goals (the best performers from C and D) to keep total cost manageable.

**Output**: `analysis/phase_e_combination_results.md`.

### E.2 — Final analysis and paper-worthy findings

Identify the concrete finding to report:

- "Per-goal X improves over baseline by Y% on Z/10 goals" is our main claim
- The multi-signal reward methodology is the secondary claim
- The comparison between search-based (Meta-Harness) and training-based (TTT-Discover) is the tertiary claim

**Manual quality check**: Read through 5 best plans across all 10 goals. Are they actually good? Would you trust them as starting points for research?

---

## 8. Phase F — OPTIONAL Recursive Meta-Goal

**Only attempt if Phase C or D shows clear positive signal.**

**Purpose**: Test your original recursive idea — set a research goal to "how to make AI generate better research plans", use the method to produce a methodology, and apply it back to the original 10 RL goals.

### F.1 — Run Meta-Harness/TTT-Discover on the meta-goal

Use the same infrastructure from Phases C/D, but with:
- Goal: "Design a methodology for generating high-quality research plans in AI/ML"
- Target: "A system that produces plans scoring high on rubric-style evaluations"
- Retrieval context: papers on LLM plan generation, self-critique, rubric rewards

**Expected output**: A research plan describing a methodology for better plan generation.

### F.2 — Apply the methodology

Interpret the meta-plan as concrete modifications to the plan-generation process:
- Does it describe new prompting strategies?
- Does it describe new self-critique patterns?
- Does it describe new training techniques?

Implement one or two of its suggestions and test on the original 10 RL goals.

### F.3 — Analysis

- Is the meta-plan actually useful?
- Does implementing its suggestions improve results on real goals?
- Is this a viable recursive improvement loop or does it degenerate?

**Risk**: This phase is highly exploratory. The meta-plan might be vague or unactionable. Budget at most 1 week for this phase.

---

## 9. Cross-Cutting Infrastructure

### 9.1 — Signal monitoring during all runs

Every Phase C and Phase D run must log:
- Per-signal scores over iterations (4 or 5 separate traces)
- Pairwise signal correlations every 3 iterations (watch for collapse)
- GPT-5.4 arena spot-check every 5 iterations (n=8 plans, correlation with 7-signal aggregate)
- Plan length distribution (catch length gaming)
- Policy entropy (catch mode collapse, TTT-Discover only)

### 9.2 — Anti-hacking safeguards

Before each Phase C/D run:
- **CREAM-style consistency check**: run the 7-signal grading twice on the same plan with reshuffled prompts. Discard training examples where signals disagree significantly across runs.
- **Correlation floor**: If 7-signal's correlation with GPT-5.4 on held-out samples drops below 0.5 during training → STOP the run and analyze.
- **Bounded rewards**: Apply sigmoid-saturating transformation to 7-signal aggregate to prevent unbounded optimization targets.

### 9.3 — Validation suite (end of every experiment)

Every candidate (best harness from C, best LoRA from D, best combined from E) must pass:

1. **7-signal aggregate improvement** over baseline — primary metric
2. **GPT-5.4 arena winrate** over baseline's best-of-32 — independent check
3. **Reference-anchored rubric** — cross-check against existing grader framework
4. **Manual inspection** of 3 best plans — you read them, sanity check

Only candidates passing all 4 checks go into the "results" set.

### 9.4 — Budget summary (rough estimates)

| Phase | Compute | $ Cost |
|-------|---------|--------|
| A (signal validation) | ~2,500 LLM calls | ~$80 |
| B (baselines) | ~1,600 LLM calls | ~$60 |
| C (Meta-Harness × 10) | ~64,000 LLM calls | ~$1,500 |
| D (TTT-Discover × 10) | ~48,000 LLM calls + ~20 GPU-hours | ~$2,000 + GPU |
| E (combinations × 5) | ~30,000 LLM calls + ~10 GPU-hours | ~$1,000 + GPU |
| F (optional) | ~10,000 LLM calls | ~$300 |
| **Total** | **~150,000 LLM calls + ~30 GPU-hours** | **~$5,000 + GPU** |

Compared to a bestversion full GRPO run (~110,000 rollouts), this is roughly **1.5× compute cost** but gives per-goal insights and a fundamentally different kind of result.

---

## 10. Key Decision Points

These are the moments where we pause, analyze, and potentially pivot:

1. **After Phase A.0** (template sanity check): if templates score >0.4 on hand-computed signals, iterate on design before building infrastructure
2. **After Phase A.6** (full validation): signals must pass template hack test AND correlate ≥ 0.7 with GPT-5.4 arena → proceed; fail → redesign (stop the plan)
3. **After Phase B.2** (retrieval contexts): manual inspection → if retrieval quality is poor, fix before Phase C
4. **After Phase C** (Meta-Harness results): does it work? If yes, document as primary; if no, focus on D
5. **After Phase D** (TTT-Discover results): does it work? Compare to C
6. **After Phase E**: decide whether to attempt F (only if C or D showed clear positive)

At each decision point, the plan document is updated with actual results and the next phase is adjusted based on what we learned.

---

## 11. What's Deliberately NOT in This Plan

To stay focused, these are explicitly out of scope:

- ❌ Full-dataset GRPO variants (see `RESEARCH_PROGRAM_2026.md` Phase 1, 3)
- ❌ Large-scale self-critique training (see Phase 4)
- ❌ Orthogonal grounding methods like citation verification or checklist compliance (see Phase 5) — except the light RAG for Signal 3
- ❌ Adversarial reward model hardening (see Phase 6 E6.3)
- ❌ Aggressive variants beyond single-goal TTT/Meta-Harness (see Phase 6)

If any of these become relevant later, we re-open `RESEARCH_PROGRAM_2026.md`.

---

## 12. Immediate Next Step

**Phase A is now a 7-signal, 3-layer structural design** (see Section 3 for full details). The sequence is:

**Step 1 — A.0: Template sanity check (0.5 days, paper exercise)**
Hand-compute the 3 hack templates (Canonical Fortress, Reviewer Cosplay, Specificity Bomb) against the 7 signals. Verify all score < 0.3 aggregate. If not, iterate on signal design before writing any code.

**Step 2 — A.1: Build Layer 1 (structural tests) [1 week]**
This is the most novel infrastructure:
- 1.1 Goal-Contrast Margin (alt-goal sampler + margin computation)
- 1.2 Claim Verification (claim extractor + dataset/citation/power-calc checkers)
- 1.3 Internal Consistency (section parser + 6 consistency checks)

These can be built in parallel. Layer 1 is the HARDEST because it requires new infrastructure (claim verification, goal contrast judge).

**Step 3 — A.2+A.3: Layers 2 and 3 (in parallel with A.1) [3-4 days]**
- 2.1 Soundness (deviation-inverted) — single judge call, prompt engineering
- 2.2 Feasibility — single judge call, prompt engineering
- 2.3 Specificity (programmatic) — reuses claim extractor from 1.2
- 3.1 Coherence — single judge call with forced citation requirement

**Step 4 — A.5: Retrieval infrastructure [1 day]**
Semantic Scholar client + per-goal cache (shared by 2.1 and 2.2).

**Step 5 — A.6: Full validation [3-5 days]**
Run the 7-signal pipeline on:
- The 3 hack templates (must all score < 0.3 aggregate)
- 500 existing bestversion plans (orthogonality check, distribution check)
- 200 plans with GPT-5.4 arena scores (correlation ≥ 0.7)
- 50 known-preference pairs (rank ordering ≥ 75%)

**Step 6 — A.7: Freeze signal design in `docs/signal_design_v1.md`.**

Phase B can begin once A.7 is complete.

---

**What I need from you next**:

1. **Should I start writing the actual implementation code now?** Specifically:
   - Prompt templates for each of the 4 LLM-judge signals
   - The claim verification taxonomy (what counts as a "verifiable claim")
   - The alt-goal sampler logic

2. **Or do you want to first do the A.0 template sanity check** as a conversation? We could walk through each template and each signal together on paper, see if the expected scores line up.

3. **Or something else** (pause, read the updated plan in full, push back on specific signals, etc.)
