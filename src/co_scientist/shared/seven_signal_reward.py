"""7-signal reward function for the Co-Scientist project.

This module is the source-of-truth for the v2 7-signal reward design that
passed Phase A.0 sanity check (archived:
projects/ttt_discover/analysis/_archive/phase_a0_jan2026/phase_a0_results.md).

NOTE: Current signal definitions live in ten_signal_reward.py; this module
retains the hard gates (G1 Goal-Contrast, G2 Claim Verification) only.

Architecture:
  Layer 1 (structural tests):
    - 1.1 Goal-Contrast Margin: contrast 3 calls (target + 2 alt goals)
    - 1.2 Claim Verification: extraction + programmatic verification
    - 1.3 Internal Consistency: 6 forced-citation cross-section checks
  Layer 2 (universal rubric):
    - 2.1 Methodological Soundness: 5x3 classification matrix
    - 2.2 Feasibility: 5-resource categorical
    - 2.3 Specificity: CONCRETE/VAGUE/MISSING counts
  Layer 3 (coherence):
    - 3.1 Narrative Coherence: 5 forced-citation dependencies

The library exposes a single high-level entry point:

    reward = compute_seven_signal_reward(
        plan=plan_text,
        target_goal=goal,
        target_target=target,
        alt_goals=[(alt_goal_1, alt_target_1), (alt_goal_2, alt_target_2)],
        sampling_client=tinker_sampling_client,
        renderer=tinker_renderer,
        tokenizer=tinker_tokenizer,
    )

The returned `SignalReward` carries:
  - aggregate (scalar, for monitoring/eval)
  - per_signal (dict, for vector-advantage GRPO training)
  - hard_gate_triggered + hard_gate_name
  - diagnostic + raw_outputs (for inspection)

For training-time use prefer per_signal; aggregate is for human monitoring.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from tinker import types

logger = logging.getLogger(__name__)

# =============================================================================
# Sampling defaults
# =============================================================================

DEFAULT_MAX_TOKENS = 12288
DEFAULT_TEMPERATURE = 0.0


# =============================================================================
# Aggregation configuration
# =============================================================================

SIGNAL_NAMES = (
    "goal_contrast_margin",
    "claim_verification",
    "internal_consistency",
    "soundness",
    "feasibility",
    "specificity",
    "coherence",
)

SIGNAL_WEIGHTS = {
    "goal_contrast_margin": 0.18,
    "claim_verification": 0.12,
    "internal_consistency": 0.12,
    "soundness": 0.15,
    "feasibility": 0.10,
    "specificity": 0.08,
    "coherence": 0.25,
}

SIGNAL_FLOORS = {
    "goal_contrast_margin": 0.45,
    "claim_verification": 0.40,
    "internal_consistency": 0.50,
    "soundness": 0.30,
    "feasibility": 0.30,
    "specificity": 0.20,
    "coherence": 0.40,
}

# Hard gates: if violated, the entire reward is zeroed.
# These are structural failures that no amount of compensatory quality can fix.
HARD_GATES = {
    "claim_verification": 0.15,    # significant fraction of claims fabricated
    "goal_contrast_margin": 0.00,  # plan less goal-specific than random alt
}

# Penalty multiplier on floor shortfall (used in scalar aggregation only).
FLOOR_PENALTY_MULTIPLIER = 5.0


# =============================================================================
# Signal prompts (v2)
# =============================================================================

SIGNAL_1_1_SCORING_PROMPT = """You are evaluating whether a research plan is SPECIFICALLY tailored to a given research goal. Be EXTREMELY STRICT: a plan that merely mentions goal-adjacent keywords without technical commitments should score low. A plan that could be copied to a different research goal in the same area with only minor word substitution should score near zero.

RESEARCH GOAL: {goal}
RESEARCH TARGET: {target}

PLAN TO EVALUATE:
{plan}

INSTRUCTIONS - follow exactly. Do NOT give a holistic 0-1 score.

Step 1. Extract 5-7 SPECIFIC elements from the research goal. Each element must be TECHNICALLY SPECIFIC - not generic concepts like "training" or "evaluation", but concrete technical aspects that DIFFERENTIATE this goal from other research in the same area.

Good examples of specific elements:
- "no ground truth exists" (a specific constraint that rules out standard supervised approaches)
- "resistant to hacking templates like reviewer cosplay" (a specific adversarial threat)
- "TTT-Discover-style per-goal optimization" (a specific training paradigm)

Bad examples (too generic, reject these):
- "training LLMs" (applies to any LLM work)
- "evaluation" (applies to anything)
- "multi-signal" (just a buzzword unless elaborated)

Step 2. For each element, classify how the plan addresses it:
- DIRECT (2 points): plan contains a SPECIFIC METHODOLOGICAL COMMITMENT that addresses this exact element. A commitment is a concrete choice (named method, specific algorithm, quantified constraint, explicit design decision). Merely RESTATING the goal in the plan text is NOT a commitment.
- INDIRECT (1 point): plan contains related content but no specific methodological commitment
- ABSENT (0 points): plan does not address this element at all, OR only restates the goal without adding methodological detail

CRITICAL: If the cited passage is a MERE RESTATEMENT of the goal (e.g., goal says "multi-signal reward" and plan says "we will use a multi-signal reward"), that is NOT DIRECT. It must add a methodological commitment beyond the restatement.

CRITICAL: Before calling any element DIRECT, apply the substitutability test: "If I replaced this goal with a different research goal in the same area, would this cited passage still apply unchanged?" If yes, downgrade to INDIRECT or ABSENT.

You MUST cite the exact passage (>= 10 words) for DIRECT and INDIRECT, AND explicitly state what methodological commitment (beyond restatement) it adds.

Step 3. Compute: T = number of elements; D = count of DIRECT; I = count of INDIRECT; A = count of ABSENT. earned = 2*D + 1*I; max_possible = 2*T; score = earned / max_possible.

Output ONLY this JSON (no prose before or after):
{{"elements": [{{"id": 1, "quote_from_goal": "...", "specificity_reason": "..."}}], "addressed": [{{"element_id": 1, "status": "DIRECT", "cited_passage": "...", "methodological_commitment_beyond_restatement": "...", "substitutability_test": "..."}}], "T": 0, "D": 0, "I": 0, "A": 0, "score": 0.0}}"""


SIGNAL_1_2_EXTRACTION_PROMPT = """Extract VERIFIABLE FACTUAL CLAIMS from this research plan, and classify each claim's role and relevance.

RESEARCH GOAL: {goal}

PLAN:
{plan}

A "verifiable factual claim" is a specific factual statement that can be checked against an external source. You are NOT judging plan quality - you are extracting claims and classifying them.

Categories:
1. DATASET_REFERENCE: a named dataset (e.g., "ImageNet", "MMLU", "PubMedQA")
2. PAPER_CITATION: a reference to prior work by name or paper title (e.g., "Chen et al. 2024", "the GPT-3 paper")
3. METHOD_NAME: a named method or technique (e.g., "LoRA", "PPO", "GRPO")
4. BENCHMARK_NUMBER: a numerical claim about prior work (e.g., "achieves 78% on MMLU")
5. SAMPLE_SIZE_CLAIM: a stated sample size with optional power justification
6. HYPERPARAMETER: a specific numerical hyperparameter value
7. RESOURCE_CLAIM: a specific stated resource (e.g., "4 H100 GPUs", "200 hours")

Role classification (how the claim functions in the plan):
- SUPPORTING: the plan uses this claim as foundation/evidence for its approach (e.g., "we build on [X]", "baseline achieves [Y]%", "we use [Z] as our training framework")
- NAME_DROPPED: the claim is listed without being actively used (e.g., "benchmarks include X, Y, Z" where X/Y/Z are just enumerated without tie to methodology)

Relevance classification (is this claim appropriate for the research goal above?):
- RELEVANT: the claim is a natural, common, appropriate choice for research on THIS specific goal (not just any LLM/RL research - this specific problem)
- IRRELEVANT: the claim is from a different subfield or task type and would be inappropriate foundation for this goal. Example: citing HumanEval/MBPP (code generation benchmarks) in a plan about reward function design for plan generation is IRRELEVANT - code benchmarks don't measure plan quality.

INSTRUCTIONS:
- Extract every claim that fits the categories
- Quote the exact text from the plan (>= 5 words context)
- Extract the canonical form for verification
- Classify role and relevance for each claim
- If the plan contains zero claims, return empty list with explanation
- DO NOT make up claims
- DO NOT include vague statements without specific values

Output ONLY this JSON:
{{"claims": [{{"category": "DATASET_REFERENCE", "exact_quote": "...", "canonical": "...", "context": "...", "role": "SUPPORTING", "relevance": "RELEVANT", "relevance_reason": "..."}}], "total_claims": 0, "explanation_if_empty": ""}}"""


SIGNAL_1_3_PROMPT = """You are checking the INTERNAL CONSISTENCY of a research plan. You are NOT judging quality - you are checking whether the parts of the plan make specific, mutually consistent commitments.

PLAN:
{plan}

Perform the following 6 consistency checks. For each, you MUST:
1. Quote the SPECIFIC passage from each side (minimum 8 words; must contain a concrete commitment, not a generic statement)
2. Make a YES/NO/VACUOUS/N/A judgment
3. Explain your reasoning briefly

Judgment definitions:
- YES: both sides have SPECIFIC content, and they are mutually consistent
- NO: both sides have specific content, and there is a concrete contradiction
- VACUOUS: at least one side is too generic to quote a specific commitment (e.g., "we will use rigorous methodology" is not a specific commitment). VACUOUS counts as a FAILED check because a plan with no concrete commitments cannot be internally consistent.
- N/A: one side is entirely absent from the plan (not just vague)

CRITICAL RULES:
- A quote must contain at least one concrete choice (named method, specific number, named dataset, specific procedure). Generic phrases like "appropriate methods", "suitable tools", "rigorous methodology", "careful experimentation", "established metrics", "state-of-the-art libraries" are NOT specific commitments - if you can only quote such phrases, answer VACUOUS.
- If the plan merely names sections without filling them with specific commitments, mark VACUOUS for any check touching those sections.
- Do NOT reward plans for having section headers; only specific content earns YES.

CHECK 1: Methods -> Expected Results (would the specific methodology produce the specific measurements in expected results?)
CHECK 2: Hypothesis -> Analysis Plan (does the specific analysis plan actually test the specific hypothesis?)
CHECK 3: Sample Size -> Statistical Power (are the sample size and power claims numerically and appropriately consistent for this task type?)
CHECK 4: Timeline -> Scope (is the specific timeline plausible for the specific scope described?)
CHECK 5: Background -> Methods (does the method address the specific gap identified in the background?)
CHECK 6: Controls -> Intervention (is the specific control appropriate for isolating the specific effect?)

Compute: pass_count = YES answers; vacuous_count = VACUOUS answers; fail_count = NO answers; na_count = N/A answers; applicable_count = 6 - na_count; if applicable_count == 0 then score = 0.0 else score = pass_count / applicable_count. (VACUOUS and NO both count against the score.)

Output ONLY this JSON:
{{"checks": [{{"check_id": 1, "name": "Methods -> Expected Results", "from_quote": "...", "to_quote": "...", "answer": "YES", "evidence_or_reason": "..."}}], "pass_count": 0, "vacuous_count": 0, "fail_count": 0, "na_count": 0, "applicable_count": 0, "score": 0.0}}"""


SIGNAL_2_1_PROMPT = """You are evaluating the methodological soundness of a research plan, with reference to canonical practice in the field.

RESEARCH GOAL: {goal}
RESEARCH TARGET: {target}

PLAN TO EVALUATE:
{plan}

INSTRUCTIONS - follow exactly. Do NOT give a free-form score.

Step 1. Using your knowledge of the field, identify the 2-3 most common methodological approaches that researchers would use for problems like this goal. For each: name the approach, its key technical components.

Step 2. List the GOAL-SPECIFIC ADAPTATIONS the plan contains. A goal-specific adaptation is a methodological choice (method, dataset, metric, architecture, training regime, evaluation protocol, constraint) that:
  (a) is CONCRETE (named method, specific number, specific choice, not a vague phrase)
  (b) would be INAPPROPRIATE if the plan were applied to a different research goal in the same broad area (i.e., swapping the goal statement would make this choice wrong)
  (c) is ACTUALLY USED in the methodology, not merely listed

For each adaptation you identify, quote the exact passage (>=10 words) and state WHY it is goal-specific (how it would be wrong for a different goal in the same area).

Step 3. Classify the plan as ONE of:
- PURE_MIMICRY: plan follows a standard template structure but has FEWER THAN 3 goal-specific adaptations (per your Step 2 list). This includes plans that restate the goal in the methods section without adding new methodological commitments.
- CONVENTION_APPLIED: plan uses a standard approach AND has 3 OR MORE goal-specific adaptations (from Step 2)
- JUSTIFIED_DEVIATION: plan deviates from standard approaches AND explicitly justifies the deviation with goal-specific reasoning
- UNJUSTIFIED_DEVIATION: plan deviates but provides no justification
- UNCONVENTIONAL: plan does something radically different

CRITICAL: Use of a popular framework (GRPO, LoRA, Qwen3-30B-A3B, AdamW) alone does NOT count as a goal-specific adaptation. Those are generic ML choices that would apply to any LLM research. A goal-specific adaptation must tie to the PARTICULAR research question.

Example of goal-specific adaptation for the goal "reward function for plan generation where no ground truth exists":
  - Adaptation: "Stage 2 uses a held-out set of plans annotated by domain experts, since no automatic ground truth is available"
  - Why goal-specific: Because the goal explicitly rules out ground truth, any plan that ignores this constraint is wrong.

Example that does NOT count:
  - "We use Qwen3-30B-A3B with LoRA rank 32" - this is a generic ML setup, applies to any LLM RL research.

Step 4. INDEPENDENTLY assess methodological SOUNDNESS:
- SOUND: the methodology is internally valid and would answer the research question
- UNSOUND: the methodology has a fundamental flaw
- UNCERTAIN: cannot determine from the plan (e.g., plan too vague)

Step 5. The score is determined by this deterministic matrix:

                           SOUND  UNSOUND  UNCERTAIN
PURE_MIMICRY               0.30   0.05     0.20
CONVENTION_APPLIED         0.70   0.15     0.50
JUSTIFIED_DEVIATION        0.85   0.20     0.55
UNJUSTIFIED_DEVIATION      0.30   0.05     0.20
UNCONVENTIONAL             0.60   0.10     0.40

You do NOT choose the score. You make the two classifications and look up the score in the matrix.

Output ONLY this JSON:
{{"canonical_approaches": [{{"name": "...", "components": ["..."]}}], "goal_specific_adaptations": [{{"quote": "...", "why_goal_specific": "..."}}], "num_adaptations": 0, "classification": "PURE_MIMICRY", "classification_evidence": "...", "soundness": "SOUND", "soundness_evidence": "...", "score": 0.0}}"""


SIGNAL_2_2_PROMPT = """You are evaluating whether a research plan is FEASIBLE given its stated resources.

RESEARCH GOAL: {goal}

PLAN TO EVALUATE:
{plan}

INSTRUCTIONS:

Step 1. Extract resource claims from the plan for each of 5 resource types. Report exact value or "ABSENT":
1. Team size / personnel
2. Timeline / duration
3. Compute budget
4. Sample size / data requirements
5. Equipment / infrastructure

Step 2. Using your knowledge of typical research in this area, estimate what each resource would typically need.

Step 3. For each resource type, judge the plan vs typical baseline:
- PLAUSIBLE: plan resource is within ~0.5x-2x of typical
- SUSPICIOUSLY_LOW: plan claims significantly less than typical
- SUSPICIOUSLY_HIGH: plan claims much more than typical
- ABSENT: plan does not state this resource

Step 4. Compute score:
- PLAUSIBLE: +1.0 each
- ABSENT: +0.3 each
- SUSPICIOUSLY_LOW: +0.2 each
- SUSPICIOUSLY_HIGH: +0.2 each
- score = sum / 5.0

Output ONLY this JSON:
{{"plan_resources": {{"team": "...", "timeline": "...", "compute": "...", "sample": "...", "equipment": "..."}}, "baseline_estimates": {{"team": "...", "timeline": "...", "compute": "...", "sample": "...", "equipment": "..."}}, "judgments": [{{"resource": "team", "judgment": "PLAUSIBLE", "comparison": "..."}}], "score": 0.0}}"""


SIGNAL_2_3_PROMPT = """You are measuring the specificity of claims in a research plan.

PLAN:
{plan}

Classifications:
- CONCRETE: contains at least one of: a named entity (dataset, model, library, tool, method), a specific numerical value with units, or a specific procedure with named components
- VAGUE: describes something in general but unparameterized terms (e.g., "we will tune hyperparameters" with no method)
- MISSING: a place where the plan SHOULD give specifics but doesn't

INSTRUCTIONS:
1. Iterate through the plan paragraph by paragraph
2. For each major content statement, classify and provide exact text
3. Do NOT double-count
4. Do NOT inflate counts by listing trivial sub-statements

Then compute: C = CONCRETE count; V = VAGUE count; M = MISSING count; score = C / (C + V + M) if total > 0 else 0.

Output ONLY this JSON:
{{"items": [{{"text": "...", "classification": "CONCRETE", "details": "..."}}], "C": 0, "V": 0, "M": 0, "score": 0.0}}"""


SIGNAL_3_1_PROMPT = """You are evaluating the narrative and logical coherence of a research plan. A coherent plan is one where each section logically depends on the previous ones IN A WAY THAT IS SPECIFIC TO THE RESEARCH GOAL.

RESEARCH GOAL: {goal}

PLAN:
{plan}

INSTRUCTIONS - follow exactly. You MUST cite specific passages to support each judgment. Do NOT make up dependencies.

For each of the 5 dependency types, find a specific passage in the FROM section, a specific passage in the TO section, articulate the logical link, and mark PRESENT/WEAK/ABSENT.

CRITICAL GOAL-SPECIFICITY RULE:
A dependency is PRESENT only if the logical link is SPECIFIC TO THIS RESEARCH GOAL. Before marking any dependency as PRESENT, apply this test:
"Could this exact logical link exist in a research plan for a completely different goal in the same broad area?"
If yes, the dependency is WEAK at best (generic chain-of-reasoning, not goal-specific structure). If the link exists only in implicit form or depends on generic scaffolding, mark WEAK or ABSENT.

CRITICAL NOVEL-CONTENT RULE:
A dependency is PRESENT only if BOTH sides contribute ORIGINAL CONTENT. If the TO section merely restates, reformulates, or summarizes what the FROM section already said (e.g., "Building on the background's observation that X, we hypothesize X"), the dependency is WEAK even if the link is goal-specific. PRESENT requires the TO section to add a NOVEL technical commitment that was not already in the FROM section.

CRITICAL CITATION RULES:
- If you cannot find specific quotes from BOTH sides, mark ABSENT
- A quote must contain at least one CONCRETE COMMITMENT (named method, specific number, specific choice). Generic statements like "rigorous methodology" or "appropriate baselines" cannot be used as the basis for a PRESENT judgment.
- WEAK is for dependencies that exist but are either implicit or generic (not goal-specific)
- PRESENT requires explicit, citable, GOAL-SPECIFIC logical connection

DEPENDENCY 1: Background -> Hypothesis (the background must motivate THIS specific hypothesis, not merely set general context)
DEPENDENCY 2: Hypothesis -> Methods (the methods must be the natural way to test THIS specific hypothesis)
DEPENDENCY 3: Methods -> Expected Results (the expected results must be what THESE specific methods would produce)
DEPENDENCY 4: Expected Results -> Interpretation (the interpretation must derive from THESE specific results and this goal's claims)
DEPENDENCY 5: Sample Size -> Statistical Power (the sample size must be appropriate for THIS specific measurement, not a generic power calc)

For each dependency, explicitly answer the goal-specificity test in your reasoning. If the logical link is generic, downgrade the status.

Score: PRESENT = 1.0, WEAK = 0.5, ABSENT = 0.0, score = sum / 5.

Output ONLY this JSON:
{{"dependencies": [{{"id": 1, "name": "Background -> Hypothesis", "from_quote": "...", "to_quote": "...", "logical_link": "...", "goal_specificity_test": "...", "status": "PRESENT"}}], "present_count": 0, "weak_count": 0, "absent_count": 0, "score": 0.0}}"""


# =============================================================================
# Public dataclass
# =============================================================================


@dataclass
class SignalReward:
    """Result of computing the 7-signal reward on a single plan.

    Attributes:
        aggregate: scalar reward in [0, 1] for monitoring/eval. Computed via
            weighted_mean - floor_penalties, with hard gates zeroing it out.
        per_signal: dict of 7 raw signal scores (each in [0, 1]) for vector
            training. THIS is the canonical training signal.
        hard_gate_triggered: True if either claim_verification or
            goal_contrast_margin tripped a hard gate. Vector consumers should
            zero out the entire per_signal vector when this is True.
        hard_gate_name: which gate fired (or None).
        diagnostic: per-signal sub-counts and intermediate values (parse
            failures, fabrication ratios, classification labels, etc.) for
            inspection and debugging.
        raw_outputs: per-signal raw model responses (truncated) for debugging.
    """

    aggregate: float
    per_signal: dict[str, float]
    hard_gate_triggered: bool
    hard_gate_name: str | None
    diagnostic: dict
    raw_outputs: dict


# =============================================================================
# JSON extraction + model call helpers
# =============================================================================


def extract_json(text: str) -> dict | None:
    """Extract the first JSON object from text, tolerating <think> blocks and prose prefixes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def call_grader(
    sampling_client,
    renderer,
    tokenizer,
    prompt: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """Synchronous call to the grader model. Returns decoded text."""
    rendered = renderer.build_generation_prompt([{"role": "user", "content": prompt}])
    future = sampling_client.sample(
        rendered,
        num_samples=1,
        sampling_params=types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    result = future.result()
    tokens = result.sequences[0].tokens
    return tokenizer.decode(tokens)


@dataclass
class _SignalCallResult:
    """Internal: result of one signal call (JSON-extracted)."""

    score: float = 0.0
    parsed: dict | None = None
    raw_response: str = ""
    error: str | None = None


def _run_signal_call(
    sampling_client,
    renderer,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> _SignalCallResult:
    """Run one grader call and parse the JSON output. Errors yield score=0.0."""
    try:
        response = call_grader(sampling_client, renderer, tokenizer, prompt, max_tokens, temperature)
    except Exception as e:
        return _SignalCallResult(error=f"API error: {e}")

    parsed = extract_json(response)
    if parsed is None:
        return _SignalCallResult(raw_response=response[:1500], error="JSON parse failed")

    raw_score = parsed.get("score", 0.0)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    return _SignalCallResult(score=score, parsed=parsed, raw_response=response[:1500])


# =============================================================================
# Programmatic Stage-2 verifier for Signal 1.2
# =============================================================================

# Known-fabricated names from the original Specificity Bomb template.
# This is a heuristic for the sanity-check templates; production should query
# Semantic Scholar / OpenAlex / HF Hub for real-time verification.
_FABRICATED_NAMES = {
    "researchplanbench", "meridian-3", "meridian 3", "flammable",
    "flammable-v2", "rubricnet", "patel & martinez", "chen et al. 2024",
    "chen et al 2024", "patel and martinez", "patel & martinez 2023",
    "patel and martinez 2023",
}

_REAL_NAMES = {
    "imagenet", "mmlu", "big-bench", "big bench", "humaneval", "mbpp",
    "gsm8k", "math", "pubmedqa", "qwen3-30b-a3b", "qwen3", "lora",
    "ppo", "grpo", "adamw", "cohen (1988)", "cohen 1988", "cream", "mona",
    "bonferroni", "osf", "wilcoxon", "h100",
}

# When the fraction of fabricated claims exceeds this threshold, the verifier
# returns 0.0, which triggers the aggregator's claim_verification hard gate.
FABRICATION_HARD_GATE_RATIO = 0.10


def programmatic_claim_verification(claims_parsed: dict | None) -> tuple[float, dict]:
    """Stage 2 of Signal 1.2: programmatic claim verification.

    Returns (score, diagnostic).

    Score is in [0, 1] except 0.0 from the fabrication hard gate.
    Diagnostic includes per-claim counts and the hard-gate flag.
    """
    diag = {
        "num_claims": 0,
        "num_fabricated": 0,
        "num_real_relevant": 0,
        "num_real_irrelevant": 0,
        "num_unknown": 0,
        "fabricated_ratio": 0.0,
        "hard_gate_triggered": False,
    }

    if claims_parsed is None:
        return 0.3, diag

    claims = claims_parsed.get("claims", [])
    diag["num_claims"] = len(claims)

    if not claims:
        return 0.3, diag  # default for zero claims

    total_credit = 0.0
    total = len(claims)
    for c in claims:
        canonical = c.get("canonical", "").lower().strip()
        exact = c.get("exact_quote", "").lower()
        category = c.get("category", "")
        role = c.get("role", "UNKNOWN")
        relevance = c.get("relevance", "UNKNOWN")

        is_fabricated = any(fab in canonical or fab in exact for fab in _FABRICATED_NAMES)
        is_real = any(real in canonical or real in exact for real in _REAL_NAMES)

        if is_fabricated:
            diag["num_fabricated"] += 1
            credit = 0.0
        elif is_real:
            if relevance == "RELEVANT":
                credit = 1.0
                diag["num_real_relevant"] += 1
            elif relevance == "IRRELEVANT":
                credit = 0.1
                diag["num_real_irrelevant"] += 1
            else:
                credit = 0.6
                diag["num_real_relevant"] += 1
        else:
            diag["num_unknown"] += 1
            # Unknown claims: PAPER and DATASET citations should be externally
            # verifiable; if we don't recognize them, they're suspicious.
            # HYPERPARAMETER and RESOURCE are locally verifiable (no lookup needed).
            if category in ("PAPER_CITATION", "DATASET_REFERENCE"):
                credit = 0.05  # can't verify → near-zero credit
            elif category == "BENCHMARK_NUMBER":
                credit = 0.1
            elif category == "SAMPLE_SIZE_CLAIM":
                credit = 0.2
            elif category == "HYPERPARAMETER":
                credit = 0.5
            elif category == "RESOURCE_CLAIM":
                credit = 0.5
            elif category == "METHOD_NAME":
                credit = 0.3
            else:
                credit = 0.15

        if role == "NAME_DROPPED" and credit > 0.3:
            credit = 0.3

        total_credit += credit

    diag["fabricated_ratio"] = diag["num_fabricated"] / total

    # Hard gate 1: detected fabrication ratio
    if diag["fabricated_ratio"] > FABRICATION_HARD_GATE_RATIO:
        diag["hard_gate_triggered"] = True
        return 0.0, diag

    # Count unknown citations (PAPER_CITATION + DATASET_REFERENCE not in known lists)
    unknown_citation_count = sum(
        1 for c in claims
        if c.get("category") in ("PAPER_CITATION", "DATASET_REFERENCE")
        and not any(fab in c.get("canonical", "").lower() for fab in _FABRICATED_NAMES)
        and not any(real in c.get("canonical", "").lower() for real in _REAL_NAMES)
    )
    diag["num_unknown_citations"] = unknown_citation_count
    unknown_citation_ratio = unknown_citation_count / total
    diag["unknown_citation_ratio"] = unknown_citation_ratio

    base_score = total_credit / total

    # Penalty: if >15% of claims are unverifiable citations, apply multiplier
    if unknown_citation_ratio > 0.15:
        penalty_mult = max(0.0, 1.0 - 2.0 * unknown_citation_ratio)
        diag["unknown_citation_penalty"] = penalty_mult
        base_score *= penalty_mult

    return base_score, diag


# =============================================================================
# Aggregation
# =============================================================================


def aggregate_reward(per_signal: dict[str, float]) -> tuple[float, dict]:
    """Compute scalar aggregate with hard gates and floor penalties.

    The vector consumers (vector-advantage GRPO) should NOT use this — they
    should normalize each signal's group statistics separately. This is for
    eval, monitoring, and the sanity-check verdict.

    Returns (aggregate, diagnostic).
    """
    diagnostic: dict = {}

    for name, threshold in HARD_GATES.items():
        if per_signal.get(name, 0.0) < threshold:
            diagnostic["hard_gate_failed"] = name
            diagnostic["hard_gate_value"] = per_signal.get(name, 0.0)
            return 0.0, diagnostic

    weighted_mean = sum(SIGNAL_WEIGHTS[k] * per_signal[k] for k in per_signal)
    diagnostic["weighted_mean"] = weighted_mean

    penalty = 0.0
    below_floor = []
    for k, score in per_signal.items():
        if score < SIGNAL_FLOORS[k]:
            p = FLOOR_PENALTY_MULTIPLIER * (SIGNAL_FLOORS[k] - score) * SIGNAL_WEIGHTS[k]
            penalty += p
            below_floor.append({"signal": k, "score": score, "floor": SIGNAL_FLOORS[k], "penalty": p})
    diagnostic["penalty"] = penalty
    diagnostic["below_floor"] = below_floor

    return max(0.0, weighted_mean - penalty), diagnostic


# =============================================================================
# Top-level entry point
# =============================================================================


def compute_seven_signal_reward(
    plan: str,
    target_goal: str,
    target_target: str,
    alt_goals: list[tuple[str, str]],
    sampling_client,
    renderer,
    tokenizer,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
) -> SignalReward:
    """Compute the full 7-signal reward on one plan.

    Args:
        plan: the research plan text to grade.
        target_goal: the actual research goal.
        target_target: the actual research target.
        alt_goals: list of 2 (alt_goal, alt_target) pairs from the same domain
            but with fundamentally different constraint structure (used by
            Signal 1.1 Goal-Contrast Margin).
        sampling_client: a tinker.SamplingClient bound to the grader model.
        renderer: a tinker_cookbook renderer for the grader model.
        tokenizer: a tinker_cookbook tokenizer for the grader model.
        max_tokens: max generation tokens per grader call.
        temperature: sampling temperature for grader (default 0.0).

    Returns:
        SignalReward with both scalar aggregate and per_signal vector.
    """
    if len(alt_goals) != 2:
        raise ValueError(f"alt_goals must contain exactly 2 (goal, target) pairs, got {len(alt_goals)}")

    per_signal: dict[str, float] = {}
    diagnostic: dict = {}
    raw_outputs: dict = {}

    # ---------- Signal 1.1: Goal-Contrast Margin (3 calls) ----------
    target_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_1_1_SCORING_PROMPT.format(goal=target_goal, target=target_target, plan=plan),
        max_tokens, temperature,
    )
    alt1_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_1_1_SCORING_PROMPT.format(goal=alt_goals[0][0], target=alt_goals[0][1], plan=plan),
        max_tokens, temperature,
    )
    alt2_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_1_1_SCORING_PROMPT.format(goal=alt_goals[1][0], target=alt_goals[1][1], plan=plan),
        max_tokens, temperature,
    )
    target_score = target_res.score
    alt_scores = [alt1_res.score, alt2_res.score]
    margin = target_score - sum(alt_scores) / len(alt_scores)
    if margin <= 0:
        signal_1_1 = 0.0
    else:
        signal_1_1 = max(0.0, min(1.0, min(target_score, 2.0 * margin)))
    per_signal["goal_contrast_margin"] = signal_1_1
    diagnostic["1.1_target_score"] = target_score
    diagnostic["1.1_alt_scores"] = alt_scores
    diagnostic["1.1_margin"] = margin
    raw_outputs["1.1_target"] = {"score": target_res.score, "error": target_res.error, "raw": target_res.raw_response[:1000]}
    raw_outputs["1.1_alt1"] = {"score": alt1_res.score, "error": alt1_res.error}
    raw_outputs["1.1_alt2"] = {"score": alt2_res.score, "error": alt2_res.error}

    # ---------- Signal 1.2: Claim Verification (extraction + programmatic) ----------
    cv_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_1_2_EXTRACTION_PROMPT.format(goal=target_goal, plan=plan),
        max_tokens, temperature,
    )
    cv_score, cv_diag = programmatic_claim_verification(cv_res.parsed)
    per_signal["claim_verification"] = cv_score
    diagnostic["1.2_verification"] = cv_diag
    raw_outputs["1.2"] = {**cv_diag, "error": cv_res.error, "raw": cv_res.raw_response[:1000]}

    # ---------- Signal 1.3: Internal Consistency ----------
    ic_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_1_3_PROMPT.format(plan=plan),
        max_tokens, temperature,
    )
    per_signal["internal_consistency"] = ic_res.score
    raw_outputs["1.3"] = {
        "score": ic_res.score,
        "pass_count": (ic_res.parsed or {}).get("pass_count"),
        "vacuous_count": (ic_res.parsed or {}).get("vacuous_count"),
        "fail_count": (ic_res.parsed or {}).get("fail_count"),
        "na_count": (ic_res.parsed or {}).get("na_count"),
        "error": ic_res.error,
        "raw": ic_res.raw_response[:1000],
    }

    # ---------- Signal 2.1: Methodological Soundness ----------
    ms_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_2_1_PROMPT.format(goal=target_goal, target=target_target, plan=plan),
        max_tokens, temperature,
    )
    per_signal["soundness"] = ms_res.score
    raw_outputs["2.1"] = {
        "score": ms_res.score,
        "classification": (ms_res.parsed or {}).get("classification"),
        "soundness": (ms_res.parsed or {}).get("soundness"),
        "num_adaptations": (ms_res.parsed or {}).get("num_adaptations"),
        "error": ms_res.error,
        "raw": ms_res.raw_response[:1000],
    }

    # ---------- Signal 2.2: Feasibility ----------
    fe_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_2_2_PROMPT.format(goal=target_goal, plan=plan),
        max_tokens, temperature,
    )
    per_signal["feasibility"] = fe_res.score
    raw_outputs["2.2"] = {"score": fe_res.score, "error": fe_res.error}

    # ---------- Signal 2.3: Specificity ----------
    sp_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_2_3_PROMPT.format(plan=plan),
        max_tokens, temperature,
    )
    per_signal["specificity"] = sp_res.score
    raw_outputs["2.3"] = {"score": sp_res.score, "error": sp_res.error}

    # ---------- Signal 3.1: Coherence ----------
    co_res = _run_signal_call(
        sampling_client, renderer, tokenizer,
        SIGNAL_3_1_PROMPT.format(goal=target_goal, plan=plan),
        max_tokens, temperature,
    )
    per_signal["coherence"] = co_res.score
    raw_outputs["3.1"] = {
        "score": co_res.score,
        "present_count": (co_res.parsed or {}).get("present_count"),
        "weak_count": (co_res.parsed or {}).get("weak_count"),
        "absent_count": (co_res.parsed or {}).get("absent_count"),
        "error": co_res.error,
        "raw": co_res.raw_response[:1000],
    }

    # ---------- Aggregate ----------
    aggregate, agg_diag = aggregate_reward(per_signal)
    diagnostic["aggregate"] = agg_diag

    hard_gate_triggered = "hard_gate_failed" in agg_diag
    hard_gate_name = agg_diag.get("hard_gate_failed")

    return SignalReward(
        aggregate=aggregate,
        per_signal=per_signal,
        hard_gate_triggered=hard_gate_triggered,
        hard_gate_name=hard_gate_name,
        diagnostic=diagnostic,
        raw_outputs=raw_outputs,
    )
