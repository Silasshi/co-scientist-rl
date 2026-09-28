"""Multi-signal reward implementation for buffer-conditioned TTT training.

V9 SIGNAL HARDENING (2026-04-18): Added S2a_formalism (formula-depth counting)
and SA_arithmetic (arithmetic consistency via Qwen3-235B grader) after Opus
cross-family audit revealed training grader ρ = -0.16 with external depth
judge. Rebalanced weights: depth signals (S2a + SA + S1) = 0.25 combined.

v8.1 CALIBRATION (2026-04-15): S4 GATE loosened after human review showed
37% of refs failing gate (vs 30% of perturbations) — the gate was
over-triggering on good refs with real but concisely-worded Problem sections.
v8.1 accepts "ML researchers", "removes bottleneck X", and named
benchmarks/methods as GATE PASS evidence.

V8 CALIBRATION (2026-04-15): Refined v7 rubrics to fix regressions:
- S4: added PROBLEM-SECTION GATE (capped at 2 if Problem is generic filler).
  v7 S4 detection was 17% because rubric checked whole plan.
- S1: restricted to TOP 3-5 load-bearing design choices (not every mention).
  v7 over-counted hardware/datasets as "design choices".
- S9: restricted to Core Idea techniques only with caps at 8. v7 grader
  counted 40+ "techniques" on some refs, tanking Opus correlation.

V7 STRUCTURAL UPDATE (2026-04-15): Weak-signal rubrics rewritten with
EXPLICIT COUNTING rules to prevent grader rationalization. Previous version
(v6) had detection rates of 24-47% on perturbations because qualitative
rubrics let the grader override targeted degradation. v7 adds deterministic
count → score tables for S1, S2, S3, S4, S7, S8, S9 (S5/S6 unchanged —
already 88-89% detection).

Signal set v9 (2026-04-18, after Opus cross-family hardening):
  Layer 0 (Hard Gates, programmatic — inherit from seven_signal_reward.py):
    G1. Goal-Contrast Margin
    G2. Claim Verification

  Layer 1 (Gradient Signals, 1-5 scale):
    S1.  Reasoning Depth             (asserted/total ratio table)
    S2.  Evidence Rigor              (FAIR/STRAWMAN counting + criterion check)
    S2a. Mathematical Formalism      (v9: non-trivial formula counting)
    S3.  Positioning                 (named/insufficient counting table)
    S4.  Significance                (DISABLED — grader can't isolate section)
    S5.  Feasibility Evaluability    (rubric bands — strong signal already)
    S6.  Mature Risk Awareness       (rubric bands — strong signal already)
    S7.  Implementation Specificity  (vague/specific marker counting)
    S8.  Scope-Generalization        (overclaim trigger / domain bucket counting)
    S9.  Research Focus              (unjustified-technique counting)
    SA.  Arithmetic Consistency      (v9: numerical claim validation, 235B grader)

Dropped from earlier drafts:
  - S1_coherence (halo magnet, mean_r = 0.52)
  - Risk Awareness (no grant rubric scores it separately)
  - Old Methodological Soundness (absorbed into S2_rigor)
  - Old Feasibility (degenerate in GRPO validation, absorbed into S7)

Supports two grader-call modes:
  - single_call: all 8 signals scored in one grader prompt (cheaper, halo risk)
  - separate_call: 8 separate grader prompts, one per signal (chosen — lower halo)

Both modes support N-median aggregation across multiple grader repeats for reliability.
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# =============================================================================
# Signal definitions
# =============================================================================

@dataclass
class SignalSpec:
    id: str
    name: str
    question: str
    cot_scaffolding: str | None = None  # Optional pre-scoring reasoning steps
    scoring_rubric: str = ""  # Optional rubric expansion
    # CR-v6 locus attribution: when set, build_single_signal_prompt appends a
    # <locus> output block asking the grader to quote (verbatim) the spans
    # where this signal is weak, for revision targeting. None = legacy path
    # (no locus emission). S9_focus stays None (whole Core Idea is local).
    # Phase 0 validation: projects/ttt_discover/analysis/signal_validity/
    # reports/v8_1/locus_accuracy_S3.md
    locus_directive: str | None = None
    grader_model_override: str | None = None  # v9: route to a different grader model


SIGNALS: list[SignalSpec] = [
    SignalSpec(
        id="S1_depth",
        name="Reasoning Depth",
        question=(
            "For each major design choice in the plan, does the author explain WHY "
            "it was made? Depth means explicit derivation: problem → insight → each "
            "choice justified by reference to that insight. Shallow plans list "
            "techniques with citations but don't explain the logical connection."
        ),
        cot_scaffolding=(
            "Do EXPLICIT CLASSIFICATION with STRICT definitions:\n"
            "\n"
            "STEP 1 — State the core problem (from Problem section) and KEY "
            "INSIGHT (from Core Idea / Motivation) in one sentence each.\n"
            "\n"
            "STEP 2 — Identify the TOP 3-5 LOAD-BEARING DESIGN CHOICES — "
            "the decisions that DISTINGUISH this plan from a standard approach. "
            "A load-bearing design choice is:\n"
            "  - a named ALGORITHMIC innovation (e.g., 'tree-guided evolution', "
            "'soft parallel decoding', 'on-policy uniform training')\n"
            "  - a novel TRAINING procedure (e.g., 'RL with self-correction reward')\n"
            "  - a novel ARCHITECTURE feature (e.g., 'structure-aware 441-token vocab')\n"
            "  - a novel DATA or OBJECTIVE choice (e.g., 'hybrid on-policy + SFT loss')\n"
            "\n"
            "DO NOT count as design choices:\n"
            "  - Hardware (GPUs, memory, compute)\n"
            "  - Dataset names (unless the plan is about building a new dataset)\n"
            "  - Standard optimizer/hyperparameter values (lr, batch size, "
            "temperature, seed) unless explicitly argued as novel\n"
            "  - Evaluation benchmarks\n"
            "  - Boilerplate tools (git, logging, monitoring)\n"
            "\n"
            "IMPORTANT: Cap the list at 5 items. If you identify more than 5, "
            "pick the 5 MOST central to the plan's contribution.\n"
            "\n"
            "STEP 3 — For EACH of the 3-5 load-bearing choices, classify as:\n"
            "  * JUSTIFIED: plan contains a sentence stating WHY this choice "
            "addresses the insight/problem (causal or mechanistic).\n"
            "  * ASSERTED: plan names the choice but gives no 'why' sentence.\n"
            "\n"
            "STEP 4 — Count (N_choices, N_asserted). Note: N_choices ∈ [3, 5].\n"
            "\n"
            "STEP 5 — Apply this TABLE (pick row matching N_asserted):\n"
            "  N_asserted = 0 → score 5\n"
            "  N_asserted = 1 → score 4\n"
            "  N_asserted = 2 → score 3\n"
            "  N_asserted = 3 → score 2\n"
            "  N_asserted ≥ 4 → score 1\n"
            "\n"
            "STRICT RULE: 'inspired by', 'following', 'as in [cite]' are NEVER "
            "justifications. But a citation PLUS a mechanistic sentence ('we use "
            "X because it provides property Y that addresses Z') IS justified."
        ),
        scoring_rubric=(
            "Use the explicit counting from STEP 5 on the TOP 3-5 load-bearing "
            "choices only (not everything mentioned in the plan).\n"
            "1: ≥4 load-bearing choices asserted (list of techniques, no argument)\n"
            "2: 3 asserted\n"
            "3: 2 asserted\n"
            "4: 1 asserted (mostly justified)\n"
            "5: 0 asserted — every load-bearing choice has explicit why-sentence"
        ),
        locus_directive=(
            "Quote each load-bearing design choice (named algorithmic "
            "innovation, training procedure, architecture feature, or "
            "data/objective choice) that you classified as ASSERTED — named "
            "without a mechanistic 'why' sentence explaining how it addresses "
            "the plan's insight. Prefer the SHORTEST sentence containing the "
            "asserted choice."
        ),
    ),
    SignalSpec(
        id="S2_rigor",
        name="Evidence Rigor",
        question=(
            "Can the proposed evidence support the claim? For EMPIRICAL work: are "
            "baselines FAIR (appropriate, modern, matched in scale — not strawman) "
            "and is there an OPERATIONALIZED success criterion (specific metric)? "
            "For THEORETICAL work: are assumptions stated and is a proof "
            "sketch/strategy given (not just 'we prove X')?"
        ),
        cot_scaffolding=(
            "Do EXPLICIT CLASSIFICATION (no interpretation):\n"
            "\n"
            "STEP 1 — Identify if the work is EMPIRICAL or THEORETICAL.\n"
            "\n"
            "STEP 2 (EMPIRICAL branch) — List every baseline in the Evaluation "
            "section. For each baseline, classify as:\n"
            "  * FAIR: a modern, appropriate-scale method from the same "
            "problem space (e.g., comparing a new LLM method to GPT-4o, Claude, "
            "or a strong recent approach).\n"
            "  * STRAWMAN: obviously weak baseline unlikely to be competitive. "
            "Signs: 'random baseline', 'untrained model', 'hand-coded heuristic "
            "from 201X', a method 5+ years old in a fast-moving field, a method "
            "from a different problem regime, or a method the plan already "
            "argues is inadequate.\n"
            "Count N_fair and N_strawman.\n"
            "\n"
            "STEP 2 (THEORETICAL branch) — Mark each of the following as present "
            "(1) or absent (0):\n"
            "  * Explicit assumptions stated\n"
            "  * Proof strategy / sketch given (not just 'we will prove')\n"
            "  * Tight bounds discussed OR key lemma stated\n"
            "Count T = sum (0-3).\n"
            "\n"
            "STEP 3 — Check the SUCCESS CRITERION. Is there a specific measurable "
            "threshold or operationalized metric?\n"
            "  * OPERATIONALIZED: named metric with benchmark ('pass@1 on MATH-"
            "500 above 0.5', 'perplexity on WikiText2 below X').\n"
            "  * VAGUE: 'we expect improvement', 'our method should outperform', "
            "'we aim to demonstrate'. Mark VAGUE if criterion uses words like "
            "'expect', 'should', 'aim to' without a threshold.\n"
            "\n"
            "STEP 4 — Apply this TABLE (no exceptions):\n"
            "  EMPIRICAL:\n"
            "    N_strawman ≥ 2 → score 1 (or 2 if a fair baseline also present)\n"
            "    N_strawman ≥ 1 AND criterion VAGUE → score 1\n"
            "    N_strawman ≥ 1 AND N_fair ≤ 1 → score 2\n"
            "    N_fair ≥ 2 AND criterion VAGUE → score 2\n"
            "    N_fair ≥ 2 AND criterion OPERATIONALIZED → score 4\n"
            "    N_fair ≥ 3 AND criterion OPERATIONALIZED AND targeted ablations "
            "→ score 5\n"
            "    Otherwise (e.g., N_fair = 1) → score 3\n"
            "\n"
            "  THEORETICAL:\n"
            "    T = 0 or 1 → score 1-2\n"
            "    T = 2 → score 3\n"
            "    T = 3 + operationalized criterion → score 4-5\n"
            "\n"
            "STRICT RULE: A strawman baseline is a strawman even if the plan "
            "calls it 'standard'. Judge baselines by whether they would be a "
            "meaningful comparison, not by what the plan calls them."
        ),
        scoring_rubric=(
            "Use the explicit classification rule from STEP 4. The counts "
            "determine the score — do NOT override based on overall impression.\n"
            "1: ≥2 strawman baselines, OR 1+ strawman + vague criterion\n"
            "2: 1 strawman with only 1 fair baseline, OR fair baselines but vague "
            "criterion\n"
            "3: 1 fair baseline with operationalized criterion, or theoretical "
            "T=2\n"
            "4: ≥2 fair baselines + operationalized criterion\n"
            "5: ≥3 fair baselines + operationalized criterion + targeted ablations"
        ),
        locus_directive=(
            "Quote each STRAWMAN baseline (weak, obsolete, or off-regime "
            "method proposed as comparison) OR each VAGUE success-criterion "
            "sentence (uses 'expect', 'should', 'aim to' without a "
            "measurable threshold). For theoretical work, quote spans where "
            "assumptions are missing or where a proof strategy is promised "
            "but not sketched."
        ),
    ),
    SignalSpec(
        id="S3_positioning",
        name="Positioning",
        question=(
            "Does the plan name SPECIFIC prior methods and explain WHY each is "
            "insufficient for this problem? Positioning is about argument, not "
            "citation count, but it requires NAMED prior work — not vague references "
            "like 'prior work', 'existing methods', 'some approaches'."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING (no interpretation):\n"
            "\n"
            "STEP 1 — List every NAMED prior method/work in the plan. A NAMED method "
            "has one of: (a) a proper-noun name (e.g., 'GEPA', 'TextGrad', 'ESM-2', "
            "'AlphaEvolve'), (b) a specific citation (e.g., 'Huang 2024', 'Vaswani "
            "et al.'), (c) a concrete descriptor that identifies a specific body of "
            "work (e.g., 'DPR retrieval', 'LoRA fine-tuning'). "
            "DO NOT count vague references: 'prior work', 'existing methods', "
            "'some approaches', 'recent work', 'alternative approaches', "
            "'other methods'. These are NOT named.\n"
            "\n"
            "STEP 2 — For EACH named method from STEP 1, ask: does the plan contain "
            "an explicit sentence stating what is INSUFFICIENT about that method FOR "
            "THIS PROBLEM? Mark each as:\n"
            "  * WITH_INSUFFICIENCY: plan states a specific limitation (e.g., "
            "'needs thousands of mutations', 'requires labeled data', 'scales "
            "poorly to long context').\n"
            "  * NAMED_ONLY: method is named but no insufficiency sentence.\n"
            "\n"
            "STEP 3 — Count (N_named, N_with_insufficiency).\n"
            "\n"
            "STEP 4 — Apply this TABLE (no exceptions):\n"
            "  N_named = 0 → score 1\n"
            "  N_named = 1, N_with_insufficiency = 0 → score 2\n"
            "  N_named = 1, N_with_insufficiency = 1 → score 3\n"
            "  N_named ≥ 2, N_with_insufficiency ≤ 1 → score 3\n"
            "  N_named ≥ 2, N_with_insufficiency ≥ 2 → score 4\n"
            "  N_named ≥ 3, N_with_insufficiency ≥ 3 AND forms explicit chain-of-"
            "reasoning ('X does A, Y extends to B, both fail on C, we target C') "
            "→ score 5\n"
            "\n"
            "STRICT RULE: 'inspired by X' or 'we use X' is NOT insufficiency. "
            "Insufficiency must be a sentence explicitly stating a limitation."
        ),
        scoring_rubric=(
            "Use the explicit counting rule from STEP 4. The count determines the "
            "score — do NOT override based on overall impression.\n"
            "1: 0 named prior methods\n"
            "2: 1 named without insufficiency\n"
            "3: 1 named with insufficiency, OR 2+ named but ≤1 have insufficiency\n"
            "4: 2+ named, 2+ have insufficiency\n"
            "5: 3+ named with full chain-of-reasoning about gap"
        ),
        locus_directive=(
            "Quote spans where a named prior method is NAMED_ONLY (no "
            "insufficiency sentence explaining what's inadequate about it "
            "for this problem), OR where 'prior work', 'existing methods', "
            "'some approaches', 'other methods' appear as generic "
            "references without naming specific methods. NOTE: CR-v6 "
            "default skips S3 from revision targeting due to goal-specific "
            "grader priors — see locus_accuracy_S3.md Phase 0 findings."
        ),
    ),
    SignalSpec(
        id="S4_significance",
        name="Significance",
        question=(
            "Does the plan make the STAKES of the problem concrete? Specifically, "
            "does it name: (a) WHO benefits if this succeeds (specific users, "
            "domains, or communities), (b) WHAT changes (measurable improvement, new "
            "capability, or removed constraint), and (c) WHY NOW (connection to the "
            "goal's motivating problem). Generic 'improves performance' or 'advances "
            "the field' is NOT significance — it's filler."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING with LOCALIZED checks:\n"
            "\n"
            "STEP 0 — PROBLEM-SECTION GATE. Read ONLY the '## Problem' section.\n"
            "  * GATE PASS: Problem section contains ANY ONE of these:\n"
            "    - a specific named bottleneck, limitation, or concrete failure mode\n"
            "    - a named task, benchmark, or application domain\n"
            "    - a specific technical distinction (e.g., 'binary vs distribution', "
            "'sequence length', 'backpropagation-free')\n"
            "    - a named prior method or set of methods as the incumbent\n"
            "    - a concrete stakeholder context (even 'ML researchers', "
            "'practitioners', 'the field', IF accompanied by a specific activity)\n"
            "\n"
            "  * GATE FAIL (ONLY if ALL the above are missing): Problem section is "
            "pure filler — e.g., 'X is important. Various challenges exist. More "
            "work is needed. This is an active area of research.'\n"
            "\n"
            "IF GATE FAIL → score caps at 2. Skip to scoring.\n"
            "IF GATE PASS → proceed to STEP 1.\n"
            "\n"
            "STEP 1 — WHO benefits? Look at the WHOLE plan (not just Problem).\n"
            "  * YES: plan names any of: (a) a specific user group ('roboticists', "
            "'radiologists', 'ML researchers doing X', 'practitioners deploying "
            "LLMs'), (b) a specific research community or subfield, (c) a specific "
            "downstream application domain.\n"
            "  * NO: only 'the field', 'users', 'people' without qualifier.\n"
            "\n"
            "STEP 2 — WHAT changes with concrete impact?\n"
            "  * YES: plan names any of: (a) a measurable outcome (numeric or "
            "categorical), (b) a removed bottleneck ('removes the need for N', "
            "'eliminates manual labeling'), (c) a new capability that didn't "
            "exist before ('enables X-shot learning on Y'), (d) a concrete "
            "impact on cost/time/throughput/scope.\n"
            "  * NO: only 'improves performance', 'advances SOTA', 'better results' "
            "without specifying what dimension.\n"
            "\n"
            "STEP 3 — WHY NOW: does the plan connect to the motivating problem?\n"
            "  * YES: plan explicitly references the challenge or gap (either in "
            "Problem, Motivation, or Core Idea).\n"
            "  * NO: plan pursues its approach disconnected from why it matters.\n"
            "\n"
            "STEP 4 — Count YES marks (0, 1, 2, or 3).\n"
            "\n"
            "STEP 5 — Apply this TABLE:\n"
            "  GATE FAIL + 0 YES → score 1\n"
            "  GATE FAIL → score 2 (capped)\n"
            "  GATE PASS + 0 YES → score 2\n"
            "  GATE PASS + 1 YES → score 3\n"
            "  GATE PASS + 2 YES → score 4\n"
            "  GATE PASS + 3 YES → score 4 (or 5 with proportional stakes re-articulation)\n"
            "\n"
            "PRINCIPLE: In a research plan for an ML paper, 'ML researchers' IS "
            "a specific audience. 'Removes a bottleneck' IS concrete impact even "
            "without a number. The WHO/WHAT tests are about whether the plan "
            "articulates stakes AT ALL, not whether it writes them like a grant "
            "proposal for a non-technical reviewer."
        ),
        scoring_rubric=(
            "Use GATE + YES-count scoring from STEP 5.\n"
            "1: GATE FAIL AND no stakes elsewhere\n"
            "2: GATE FAIL, OR GATE PASS with no stakes content\n"
            "3: GATE PASS + 1 YES (partial stakes)\n"
            "4: GATE PASS + 2-3 YES (clear stakes)\n"
            "5: GATE PASS + 3 YES + proportional re-articulation"
        ),
        locus_directive=(
            "Quote sentences from the '## Problem' section that fail the "
            "GATE (pure filler lacking specific bottleneck, benchmark, "
            "technical distinction, or stakeholder) OR spans claiming "
            "'improves performance' / 'advances SOTA' / 'better results' "
            "without specifying a measurable change, removed bottleneck, "
            "or new capability. NOTE: S4 is in the default disabled_signals "
            "set (weight=0) but the directive is provided for consistency "
            "and future enablement."
        ),
    ),
    SignalSpec(
        id="S5_feasibility",
        name="Feasibility Evaluability",
        question=(
            "Can a domain expert read this plan and judge whether it is "
            "implementable? Specifically: is the CORE ALGORITHM specified, are KEY "
            "DEPENDENCIES (datasets, models, tools) named, are CRITICAL parameters "
            "given? A good plan makes the methodology SELF-CONTAINED enough that a "
            "reader in the field can assess feasibility without needing additional "
            "context. "
            "\n\n"
            "DO NOT penalize plans for omitting: cloud providers, exact hardware "
            "specs, cost estimates, batch sizes, parallelization strategies, or "
            "other deployment-level details. These belong in grant proposals, not "
            "research plans. Published paper methodologies ROUTINELY omit these."
        ),
        scoring_rubric=(
            "1: Not evaluable. Core algorithm is vague ('we tune hyperparameters "
            "appropriately', 'we use standard techniques') or key dependencies "
            "(datasets, models) are unnamed. A domain expert cannot judge if the "
            "approach is implementable.\n"
            "2: Marginal. Core algorithm stated but 2+ key design decisions are "
            "hand-waved. Reader can guess but not confidently judge feasibility.\n"
            "3: Adequate. Core algorithm described and most dependencies named, but "
            "one critical aspect (algorithm detail, key parameter, or dependency) "
            "is missing or vague.\n"
            "4: Feasibility-judgeable. Core algorithm is clearly described, "
            "key dependencies (datasets, models, tools) are named, critical "
            "parameters appear at methodology level (no need for exhaustive "
            "implementation details). A domain expert could confidently judge "
            "whether this approach works.\n"
            "5: Exceptionally clear. All critical design choices are specified AND "
            "briefly justified. Reader understands not just WHAT the plan does but "
            "WHY each choice is made. Methodology is self-contained enough that a "
            "skilled implementer could execute it."
        ),
        locus_directive=(
            "Quote spans where the core algorithm is VAGUE (e.g., 'we tune "
            "hyperparameters appropriately', 'we use standard techniques', "
            "'we apply common tricks'), OR where a KEY DEPENDENCY — "
            "dataset, model, or critical parameter that a domain expert "
            "would need to judge feasibility — is unnamed or hand-waved. "
            "Do NOT quote routine omissions (hardware, batch size, cost "
            "estimates) per the rubric's scope guidance."
        ),
    ),
    SignalSpec(
        id="S6_risk_awareness",
        name="Mature Risk Awareness",
        question=(
            "Does the plan demonstrate mature awareness of its own boundaries, "
            "risks, and failure modes? A good plan names 2-3 SPECIFIC failure modes, "
            "identifies which assumptions are most fragile, states what the plan "
            "does NOT claim (scope boundary), and has a coherent view of how "
            "failures would be recognized. This is NOT about enumerating 10 "
            "failure modes (dilutive theater) — it's about showing the author "
            "understands the risk structure."
        ),
        cot_scaffolding=(
            "Before scoring:\n"
            "1. Does the plan identify 2-3 specific ways it could fail? "
            "(specific failure = specific assumption breaking, specific resource "
            "running out, specific domain mismatch)\n"
            "2. Does the plan state which assumptions are most fragile "
            "(the 'key bet')?\n"
            "3. Does the plan acknowledge scope boundaries (what it does NOT claim)?\n"
            "4. Is there a coherent risk structure, or just a laundry list?\n"
            "5. Assign a score."
        ),
        scoring_rubric=(
            "1: No risk awareness. Plan assumes success. No discussion of when/why "
            "the method might fail, no scope boundaries, no acknowledgment of "
            "fragile assumptions.\n"
            "2: Generic disclaimers. Plan says 'there might be limitations' or "
            "'further research is needed' without specificity.\n"
            "3: Partial risk awareness. Plan identifies 1 failure mode OR mentions "
            "scope boundary, but risk structure is incomplete. Doesn't name fragile "
            "assumptions.\n"
            "4: Mature risk awareness. Plan names 2-3 "
            "specific failure modes (e.g., 'fails on insufficiently capable base "
            "models', 'assumes tree-guided evolution outperforms linear chain'). "
            "Identifies fragile assumptions. States scope boundary. Example: "
            "IntrinsicSelfCritique noting 'Gemma-2 27B barely moves — confirms the "
            "method requires a sufficiently capable base model.'\n"
            "5: Exceptional risk awareness. Plan has a coherent model of its own "
            "risk structure: names specific failure modes, articulates which "
            "assumptions are load-bearing, states scope boundary explicitly, AND "
            "sketches a fallback direction. Demonstrates the author has thought "
            "through 'what could go wrong' without padding with generic risks."
        ),
        locus_directive=(
            "Quote the COMPLETE SENTENCE containing each GENERIC risk "
            "disclaimer (e.g., 'there might be limitations', 'further "
            "research is needed', 'various challenges exist'). If the "
            "plan lacks specific failure modes / fragile assumptions / "
            "scope boundaries, quote the LAST SENTENCE of the plan and "
            "set your 'why' to: 'No risk content exists — use EXPAND to "
            "add specific failure modes and scope boundaries after this "
            "sentence.' This tells the reviser to INSERT, not REPLACE."
        ),
    ),
    SignalSpec(
        id="S7_specificity",
        name="Implementation Specificity",
        question=(
            "Does the plan make SPECIFIC commitments, or does it rely on vague "
            "placeholders? Specificity means concrete, named commitments — models, "
            "datasets, hyperparameters, metrics, theorems. Vagueness shows up as "
            "filler words that look like specificity but don't commit to anything: "
            "'appropriately tuned', 'standard techniques', 'suitable baselines'."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING of two lists (no interpretation):\n"
            "\n"
            "STEP 1 — Count VAGUE MARKERS in the Core Idea + Methodology + "
            "Evaluation sections. A VAGUE MARKER is any of:\n"
            "  - 'appropriately', 'appropriate' (tuned, sized, chosen)\n"
            "  - 'standard' (techniques, optimizer, practice, procedure, baselines)\n"
            "  - 'typical' (setup, values, range)\n"
            "  - 'suitable', 'reasonable' (parameters, baselines, threshold)\n"
            "  - 'we will tune', 'we will explore', 'we will investigate' "
            "(without naming the search space)\n"
            "  - 'various', 'several', 'many' (without enumeration)\n"
            "  - 'some' as a quantifier (e.g., 'some regularization')\n"
            "  - 'further details', 'exact values TBD', 'to be determined'\n"
            "  - 'state-of-the-art' used as a substitute for a named method\n"
            "\n"
            "Go line-by-line and COUNT occurrences. Write V = the count.\n"
            "\n"
            "STEP 2 — Count SPECIFIC COMMITMENTS in the same sections. A SPECIFIC "
            "COMMITMENT is any of:\n"
            "  - a named model (e.g., 'GPT-4o', 'Qwen3-30B', 'LLaMA-7B')\n"
            "  - a named dataset (e.g., 'MATH-500', 'WikiText2', 'PACS')\n"
            "  - a numeric hyperparameter (lr=3e-5, batch=32, seeds=5, α=2.0)\n"
            "  - a named loss/objective (e.g., 'GRPO loss', 'cross-entropy')\n"
            "  - a named metric (e.g., 'perplexity on WikiText2', 'pass@1')\n"
            "  - a named theorem/lemma or formally stated proposition\n"
            "\n"
            "Count occurrences. Write C = the count.\n"
            "\n"
            "STEP 3 — Apply this TABLE (no exceptions):\n"
            "  V ≥ 6 → score 1 (pervasive vagueness)\n"
            "  V ≥ 3 and C ≤ 3 → score 2\n"
            "  V ≥ 3 and C ≥ 4 → score 3 (vagueness offset by some specificity)\n"
            "  V ≤ 2 and C ≥ 5 → score 4 (specific with minor vagueness)\n"
            "  V ≤ 1 and C ≥ 8 → score 5 (exceptional specificity, almost no "
            "filler language)\n"
            "  Otherwise → score 3\n"
            "\n"
            "STRICT RULE: Do NOT rationalize vague markers as 'acceptable "
            "delegation to prior work'. If the plan uses 'standard optimizer' "
            "without naming it, that is vague — even if 'standard' means Adam in "
            "practice. The question is whether the plan MADE the commitment, not "
            "whether the reader could guess."
        ),
        scoring_rubric=(
            "Use the explicit V/C counting rule from STEP 3. The counts determine "
            "the score.\n"
            "1: V ≥ 6 (pervasive vagueness, even if some specifics present)\n"
            "2: V ≥ 3 and C ≤ 3\n"
            "3: V ≥ 3 and C ≥ 4, OR anything that doesn't fit the other bands\n"
            "4: V ≤ 2 and C ≥ 5\n"
            "5: V ≤ 1 and C ≥ 8"
        ),
        locus_directive=(
            "Quote the COMPLETE SENTENCE (not just the vague word) "
            "containing each VAGUE MARKER from STEP 1 ('appropriately', "
            "'standard', 'typical', 'suitable', 'reasonable', 'various', "
            "'several', 'some', 'we will tune/explore/investigate' "
            "without a search space, 'state-of-the-art' as a method "
            "name, 'TBD', 'further details'). The sentence must be "
            "copy-pasteable from the plan text above. If multiple vague "
            "markers appear in one sentence, quote that sentence once. "
            "Prefer the 3 sentences with the highest density of vague "
            "markers."
        ),
    ),
    SignalSpec(
        id="S8_scope",
        name="Scope-Generalization Alignment",
        question=(
            "Does the plan's evaluation COVER what the plan claims? "
            "IMPORTANT: This is NOT about breadth (broad is NOT better than narrow). "
            "This is about whether the evaluation would SUPPORT the plan's claims "
            "if successful. A narrow claim with narrow matching evaluation is a "
            "HIGH SCORE (4-5)."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING (no interpretation):\n"
            "\n"
            "STEP 1 — COUNT OVERCLAIM TRIGGER WORDS in the Problem + Motivation + "
            "Core Idea sections. A trigger word is any of:\n"
            "  - 'universal', 'universally'\n"
            "  - 'general', 'generally', 'general-purpose'\n"
            "  - 'any task', 'any domain', 'any field', 'any research'\n"
            "  - 'all domains', 'all fields', 'all scientific'\n"
            "  - 'across domains', 'across fields', 'across all'\n"
            "  - 'every discipline', 'every field'\n"
            "  - 'foundation', 'foundational' (used as a claim, not a citation)\n"
            "Write OC = the count.\n"
            "\n"
            "STEP 2 — COUNT EVALUATION BENCHMARKS / DOMAINS in the Evaluation "
            "section. Group related benchmarks into DOMAIN BUCKETS. For example, "
            "GSM8K + MATH-500 + Minerva are ONE bucket (math). HumanEval + MBPP "
            "are ONE bucket (code). PACS + DomainNet are ONE bucket (vision). "
            "Write D = number of distinct domain buckets evaluated.\n"
            "\n"
            "STEP 3 — Apply this TABLE (no exceptions; pick the single matching "
            "row):\n"
            "\n"
            "  OC ≥ 4 AND D ≤ 2 → score 1 (severe overclaim: many universal "
            "trigger words but narrow eval)\n"
            "  OC ≥ 2 AND D ≤ 2 → score 2 (moderate overclaim)\n"
            "  OC ≥ 2 AND D ≥ 3 → score 3 (broad claim partly substantiated)\n"
            "  OC ≤ 1 AND D = 1 → score 3 (narrow claim, narrow eval)\n"
            "  OC ≤ 1 AND D ≥ 2 AND plan claims transfer/OOD → must include "
            "OOD evaluation: yes → 4-5, no → 2\n"
            "  OC ≤ 1 AND D ≥ 2 AND plan claims a specific problem only → 4\n"
            "  OC = 0 AND eval matches each claim exactly → 5\n"
            "\n"
            "STRICT RULE: 'universal' or 'across all domains' in the Problem "
            "section IS an overclaim trigger, even if it's prose framing. Count "
            "every occurrence. Do NOT discount these as 'general motivation'."
        ),
        scoring_rubric=(
            "Use the explicit OC/D counting from STEP 3. The counts determine the "
            "score — do NOT override based on overall impression.\n"
            "1: OC ≥ 4 AND D ≤ 2 (severe overclaim)\n"
            "2: OC ≥ 2 AND D ≤ 2 (moderate overclaim)\n"
            "3: OC ≥ 2 AND D ≥ 3 (broad claim partly substantiated), OR narrow "
            "claim with single benchmark\n"
            "4: OC ≤ 1 AND D ≥ 2 with specific claim, OR narrow claim with 2+ "
            "benchmarks\n"
            "5: OC = 0 AND eval exactly matches claim scope"
        ),
        locus_directive=(
            "Quote the COMPLETE SENTENCE (not just the trigger word) "
            "containing each OVERCLAIM TRIGGER from STEP 1 ('universal', "
            "'general', 'any task/domain/field', 'across domains/fields', "
            "'all domains/fields', 'foundation' used as a claim). The "
            "sentence must be copy-pasteable from the plan text above. "
            "If the scope mismatch is on the evaluation side, quote "
            "instead a COMPLETE SENTENCE from the Evaluation section "
            "whose scope is narrower than what Problem/Motivation/Core "
            "Idea claimed."
        ),
    ),
    SignalSpec(
        id="S9_focus",
        name="Research Focus",
        question=(
            "Does the plan STAY focused on a small number of core techniques, or "
            "does it STACK unrelated methods without justifying each? A focused "
            "plan has 1-3 core techniques that clearly compose. A stacked plan "
            "lists many techniques (e.g., 'Bayesian optimization + meta-learning + "
            "evolutionary algorithms + curiosity bonuses') without explaining why "
            "each is necessary."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING with STRICT definitions:\n"
            "\n"
            "STEP 1 — Identify the CORE IDEA techniques. Read ONLY the '## Core "
            "Idea' section. List the distinct METHODOLOGICAL techniques named "
            "in the Core Idea as part of the central approach. A 'technique' is "
            "a named algorithmic or training component that is FUNDAMENTAL to "
            "what the plan proposes.\n"
            "\n"
            "DO NOT count as techniques:\n"
            "  - standard infrastructure (GPUs, distributed training, logging)\n"
            "  - standard datasets (used for evaluation, not as a method)\n"
            "  - standard optimizer/learning-rate choices\n"
            "  - Components of a single multi-stage pipeline where the stages "
            "are obviously interdependent (e.g., an encoder + decoder counts as "
            "ONE architecture, not two)\n"
            "  - Evaluation metrics or benchmarks\n"
            "  - Citations of prior methods (unless the plan USES them as part of "
            "its method, not just for positioning)\n"
            "\n"
            "Cap the list at 8 items maximum. If more than 8, that itself is a "
            "signal of stacking — pick the 8 most central.\n"
            "\n"
            "STEP 2 — For EACH technique, mark as exactly ONE of:\n"
            "  * JUSTIFIED: plan has a sentence explaining WHY this technique "
            "is needed for the core idea's insight (a mechanistic/causal "
            "sentence). Being part of a sentence like 'X combined with Y enables "
            "Z' counts IF Z is explained.\n"
            "  * UNJUSTIFIED: technique is named but no explanation of why "
            "it's needed for THIS plan's insight.\n"
            "\n"
            "STEP 3 — Count (N_total, N_unjustified).\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_total ≤ 3 AND N_unjustified = 0 → score 5 (focused, justified)\n"
            "  N_total ≤ 3 AND N_unjustified ≥ 1 → score 4\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≤ 1 → score 4 (ok focus)\n"
            "  N_total ∈ [4, 6] AND N_unjustified ∈ [2, 3] → score 3\n"
            "  N_total ∈ [4, 6] AND N_unjustified ≥ 4 → score 2\n"
            "  N_total ≥ 7 AND N_unjustified ≤ 2 → score 3 (many but mostly justified)\n"
            "  N_total ≥ 7 AND N_unjustified ≥ 3 → score 2\n"
            "  N_total ≥ 7 AND N_unjustified ≥ 5 → score 1 (classic stacking)\n"
            "\n"
            "STRICT RULE: 'inspired by X' is NOT justification. Justification "
            "must explain the CAUSAL role of the technique for this plan's "
            "insight. But techniques that share a sentence explaining a joint "
            "mechanism (e.g., 'A and B together enable C because...') both "
            "count as JUSTIFIED if C is explained."
        ),
        scoring_rubric=(
            "Use the explicit counting from STEP 4. Focus is a function of "
            "BOTH technique count and justification ratio. Few techniques, "
            "justified = 5. Many techniques, unjustified = 1.\n"
            "1: N_total ≥ 7 AND N_unjustified ≥ 5 (severe stacking)\n"
            "2: moderate stacking or many unjustified\n"
            "3: borderline (2-3 unjustified of moderate total)\n"
            "4: focused or mostly justified\n"
            "5: tight focus (≤3 techniques) with every one justified"
        ),
    ),

    # =========================================================================
    # v9 signals (2026-04-18): formula-depth and arithmetic consistency
    # =========================================================================

    SignalSpec(
        id="S2a_formalism",
        name="Mathematical Formalism",
        question=(
            "Does the plan contain NON-TRIVIAL mathematical content? "
            "Non-trivial means: a named equation written in symbolic form that "
            "is NOT tautological (L = -R is tautological), NOT just hyperparameter "
            "values (lr=3e-5), and NOT standard definitions. Count DISTINCT "
            "equations/formulas that contribute to the plan's core methodology."
        ),
        cot_scaffolding=(
            "Do EXPLICIT COUNTING (no interpretation):\n"
            "\n"
            "STEP 1 — Scan the ENTIRE plan for EQUATIONS written in symbolic form.\n"
            "An EQUATION contains:\n"
            "  - mathematical symbols (=, +, -, *, /, ^, log, exp, sum, integral, E[])\n"
            "  - AND at least one NAMED variable or function beyond simple constants\n"
            "\n"
            "List each equation found. Write down the actual formula.\n"
            "\n"
            "STEP 2 — For EACH equation, classify as:\n"
            "  * TAUTOLOGICAL: L = -R (loss is negative reward), y = f(x) without\n"
            "    defining f, or any formula that restates the problem without adding\n"
            "    analytical content.\n"
            "  * HYPERPARAMETER-ONLY: Just lr=3e-5, batch=32, rank=8, etc.\n"
            "    These are parameter settings, NOT formulas.\n"
            "  * NON-TRIVIAL: A formula that adds analytical content. Examples:\n"
            "    - A policy gradient: nabla J = E[R * nabla log pi]\n"
            "    - An objective: J_beta = E[log E[exp(beta * R)]]\n"
            "    - A bound: Q(s) + c * P(s) * sqrt(1+T)/(1+n(s))\n"
            "    - A regularizer with specific form: KL(pi || pi_ref) <= delta\n"
            "    - A proof sketch inequality or lemma statement\n"
            "\n"
            "STEP 3 — Count N_nontrivial = number of NON-TRIVIAL formulas.\n"
            "\n"
            "STEP 4 — Apply this TABLE (no exceptions):\n"
            "  N_nontrivial = 0 → score 1\n"
            "  N_nontrivial = 1 AND formula is a standard textbook result\n"
            "    (plain REINFORCE, plain cross-entropy, plain KL) → score 2\n"
            "  N_nontrivial = 1 AND formula is adapted/novel for this plan → score 3\n"
            "  N_nontrivial = 2 AND at least 1 non-textbook → score 4\n"
            "  N_nontrivial >= 3 AND at least 1 non-textbook AND formulas form a\n"
            "    coherent derivation chain → score 5\n"
            "\n"
            "STRICT RULE: Hyperparameter values (lr=3e-5, beta=0.1, batch=32) are\n"
            "NEVER equations. Section headers with math words ('Mathematical\n"
            "Formulation') are NOT formulas. LaTeX formatting of a number\n"
            "($\\eta = 10^{-5}$) is NOT a formula."
        ),
        scoring_rubric=(
            "Use the explicit counting from STEP 4. The count of NON-TRIVIAL\n"
            "formulas determines the score.\n"
            "1: Zero non-trivial formulas (only prose, hyperparams, or tautologies)\n"
            "2: One standard textbook formula (plain REINFORCE, standard loss)\n"
            "3: One adapted/novel formula\n"
            "4: Two formulas with at least one non-textbook\n"
            "5: Three+ formulas forming a coherent derivation chain"
        ),
        locus_directive=(
            "Quote the COMPLETE SENTENCE where a mathematical claim is made "
            "in prose form without an accompanying symbolic equation (e.g., "
            "'we use a policy gradient objective' without writing "
            "nabla J = ...). If score >= 4, output empty array."
        ),
    ),
    SignalSpec(
        id="SA_arithmetic",
        name="Arithmetic Consistency",
        question=(
            "Are the NUMERICAL CLAIMS in the plan internally consistent? "
            "Check: (1) parameter counts consistent with stated model sizes, "
            "(2) compute budgets consistent with model forward/backward pass costs, "
            "(3) citations resolve to plausible papers (not hallucinated arXiv IDs), "
            "(4) named quantities use correct units and magnitudes."
        ),
        cot_scaffolding=(
            "Do EXPLICIT CHECKING (line by line):\n"
            "\n"
            "STEP 1 — List every NUMERICAL CLAIM in the plan involving:\n"
            "  - parameter counts (e.g., '3.2B parameters', '10% of layers')\n"
            "  - compute budgets (FLOPs, GPU-hours, wall-time, dollar cost)\n"
            "  - LoRA/adapter sizing (rank, which layers, added param count)\n"
            "  - speed/throughput claims (tokens/sec, samples/iter)\n"
            "  - dataset sizes or sample counts\n"
            "  - threshold values with units (e.g., 'QED >= 0.85', '1,000 GFLOPs')\n"
            "Cap at 8 claims. Pick the most load-bearing ones.\n"
            "\n"
            "STEP 2 — For EACH numerical claim, perform a QUICK CONSISTENCY CHECK:\n"
            "  * Does the number's ORDER OF MAGNITUDE make sense?\n"
            "    - A 7B model has ~7e9 params; 10% of layers ~ 0.7B params, not 3.2B\n"
            "    - A 7B forward pass ~ 2*N*T FLOPs per token; 1000 tokens ~ 1.4e13 FLOPs\n"
            "    - LoRA rank r on a layer with hidden dim d adds ~2*r*d params per layer\n"
            "    - arXiv IDs have format YYMM.NNNNN; the year+month should be plausible\n"
            "  * Mark each as:\n"
            "    - CONSISTENT: magnitude and units are plausible\n"
            "    - INCONSISTENT: off by >=3x, or units wrong, or internally contradictory\n"
            "    - UNVERIFIABLE: cannot check without external lookup (acceptable)\n"
            "\n"
            "STEP 3 — Count N_inconsistent and N_claims (CONSISTENT + INCONSISTENT only).\n"
            "\n"
            "STEP 4 — Apply this TABLE:\n"
            "  N_inconsistent >= 3 → score 1 (pervasive arithmetic errors)\n"
            "  N_inconsistent = 2 → score 2\n"
            "  N_inconsistent = 1 → score 3\n"
            "  N_inconsistent = 0 AND N_claims >= 5 → score 5\n"
            "  N_inconsistent = 0 AND N_claims >= 3 → score 4\n"
            "  N_claims <= 2 (too few to verify) → cap at score 3\n"
            "\n"
            "NOTE: UNVERIFIABLE claims do NOT count as inconsistent. Only flag\n"
            "claims where you can compute or estimate the correct answer."
        ),
        scoring_rubric=(
            "Use the explicit counting from STEP 4.\n"
            "1: 3+ arithmetic errors (parameter counts, FLOPs, units inconsistent)\n"
            "2: 2 errors\n"
            "3: 1 error, OR too few numerical claims to verify\n"
            "4: 0 errors with 3+ verifiable claims\n"
            "5: 0 errors with 5+ verifiable claims, all consistent"
        ),
        locus_directive=(
            "Quote the COMPLETE SENTENCE containing each INCONSISTENT numerical "
            "claim (parameter count off by >=3x, FLOP budget wrong order of "
            "magnitude, hallucinated citation, unit-inconsistent threshold)."
        ),
        grader_model_override="Qwen/Qwen3-235B-A22B-Instruct-2507",
    ),
]


# =============================================================================
# Signal weights (for scalar aggregation in the entropic objective)
# =============================================================================
# Set after sanity check + deep review. Higher weight for signals with:
#   (a) strong literature grounding (NIH/NSF/ERC)
#   (b) low halo / high independence observed in sanity check
#   (c) unique coverage (captures something no other signal does)

SIGNAL_WEIGHTS: dict[str, float] = {
    # v9 weights (2026-04-18): signal hardening after Opus cross-family audit.
    #
    # Changes from v8.1:
    # - S2a_formalism (NEW, 0.10): formula-depth counting — targets the #1
    #   Opus gap (-3.75/5 on math formalism). S2_rigor only checks baselines.
    # - SA_arithmetic (NEW, 0.08): arithmetic consistency — no existing signal
    #   validates whether numerical claims (param counts, FLOPs) are correct.
    #   Graded by Qwen3-235B (needs stronger math reasoning).
    # - S2_rigor reduced 0.13 → 0.08: still measures baseline fairness but
    #   coverage is now shared with S2a_formalism.
    # - S1_depth up 0.03 → 0.07: addresses novelty gap (Opus -3.375/5).
    # - S8/S9 reduced: were overweighted relative to depth signals.
    # - Depth signals (S2a + SA + S1) combined = 0.25 vs S8+S9 = 0.27.
    #   Previously depth = 0.16 vs S8+S9 = 0.37.
    "S1_depth":          0.07,
    "S2_rigor":          0.08,
    "S2a_formalism":     0.10,
    "S3_positioning":    0.10,
    "S4_significance":   0.00,  # Still disabled — grader can't isolate section
    "S5_feasibility":    0.10,
    "S6_risk_awareness": 0.10,
    "S7_specificity":    0.10,
    "S8_scope":          0.13,
    "S9_focus":          0.14,
    "SA_arithmetic":     0.08,
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-6, \
    f"Weights must sum to 1.0, got {sum(SIGNAL_WEIGHTS.values())}"


def normalize_score(score: int | None) -> float:
    """Map 1-5 integer score to [0, 1]:  1→0.0, 2→0.25, 3→0.5, 4→0.75, 5→1.0."""
    if score is None:
        return 0.0
    return (score - 1) / 4.0


def aggregate_reward(signal_scores: dict[str, int | None]) -> float:
    """Scalar aggregate across gradient signals using SIGNAL_WEIGHTS.

    Missing signals contribute 0. Does NOT apply hard gates (those come from
    seven_signal_reward.py Goal-Contrast and Claim Verification).
    """
    total = 0.0
    for sid, weight in SIGNAL_WEIGHTS.items():
        total += weight * normalize_score(signal_scores.get(sid))
    return total


# =============================================================================
# Prompt builders
# =============================================================================

def _format_signal_block(spec: SignalSpec, include_cot: bool = True) -> str:
    block = f"## Signal: {spec.name} (id={spec.id})\n\n**Question**: {spec.question}\n"
    if include_cot and spec.cot_scaffolding:
        block += f"\n**Required reasoning before scoring**:\n{spec.cot_scaffolding}\n"
    if spec.scoring_rubric:
        block += f"\n**Scoring rubric**:\n{spec.scoring_rubric}\n"
    return block


SHARED_PREAMBLE = """You are an expert-level research plan evaluator. Your job is to assess a research plan on a specific, well-defined evaluation dimension.

RULES:
- Evaluate based on what is ACTUALLY in the plan, not what it claims about itself.
- Follow the rubric strictly. Each score 1-5 has specific criteria.
- Be skeptical of surface-level plausibility — check whether claims are supported by concrete content.
- Do not demand rigor-theater (long lists of statistical procedures, exhaustive failure modes) when the plan is concise and focused. Judge the substance of what's there.
- Concise methodology that delegates boilerplate to citations is acceptable if the core is clearly described.
- If a signal has "Required reasoning" scaffolding, you MUST do the reasoning BEFORE assigning a score.
- Integer scores only: 1, 2, 3, 4, or 5."""


def build_single_call_prompt(goal: str, plan: str) -> str:
    """Build a single prompt that asks the grader to score ALL 8 signals at once."""
    signal_blocks = "\n\n---\n\n".join(_format_signal_block(s) for s in SIGNALS)

    output_template = "\n".join(
        f"    <dim id=\"{s.id}\">\n"
        f"        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>\n"
        f"        <score>INTEGER 1-5</score>\n"
        f"    </dim>"
        for s in SIGNALS
    )

    return f"""{SHARED_PREAMBLE}

You will evaluate the following research plan on {len(SIGNALS)} independent dimensions.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimensions

Below are the {len(SIGNALS)} dimensions to evaluate. For EACH dimension, apply the scoring rubric strictly.

{signal_blocks}

---

# Output Format

Evaluate ALL {len(SIGNALS)} dimensions in the following XML structure:

<evaluation>
{output_template}
</evaluation>

Begin your evaluation now."""


MAX_PLAN_CHARS = 15000  # ~3750 tokens — keeps grading prompts within 32K context


def _truncate_plan(plan: str) -> str:
    """Truncate plan text to fit within context window."""
    if len(plan) > MAX_PLAN_CHARS:
        return plan[:MAX_PLAN_CHARS] + "\n\n[... truncated for context limit ...]"
    return plan


def _build_locus_output_block(signal: SignalSpec) -> str:
    """Build the <locus> output instruction appended to per-signal prompts.

    Modeled on the validated Phase 0 prompt in
    `projects/ttt_discover/analysis/signal_validity/scripts/grade/grade_locus_s3_pilot.py`
    (Phase 0 literal-verbatim rates: 7/8 signals ≥90%; S3 is goal-specific
    outlier at 16% on TTT-Discover and is skipped by default in CR-v6).

    Parameterized by `signal.name` (appears in the directive for the grader)
    and `signal.locus_directive` (what spans to quote — signal-specific).
    Returns empty string if the signal has no locus_directive.
    """
    if not signal.locus_directive:
        return ""
    return f"""

---

# Additional Output: Locus Attribution

After the <evaluation> block, output a separate <locus> block containing a
JSON array with up to 3 entries identifying WHERE in the Research Plan the
**{signal.name}** weakness appears. {signal.locus_directive}

Each entry has exactly these three keys:

  - "section_hint": name of the plan section the span belongs to
     (e.g., "Problem", "Motivation", "Related Work", "Core Idea",
     "Methodology", "Evaluation", "Risks"). Use "" if unclear.
  - "quote": a VERBATIM substring copied from the Research Plan text above
     (≤ 250 characters).
  - "why": 1-2 sentences explaining why this span hurts the {signal.name}
     signal per the rubric above. Cite the specific rubric language.

**VERBATIM RULE (STRICT)**: The "quote" MUST be a literal copy-paste from the
"# Research Plan" text above. DO NOT paraphrase, DO NOT merge non-adjacent
sentences, DO NOT fix typos, DO NOT insert words. If you cannot find a clean
contiguous substring that captures the weakness, use the single BEST
available substring and describe the missing content in "why". Before
emitting, mentally Ctrl-F the quote against the plan text — if not an exact
substring, rewrite it so it is.

Choose spans where the weakness is CONCENTRATED — the 1-3 places a reviser
would most naturally edit. If the plan has NO {signal.name} weakness
(score 5), output an empty JSON array.

If the weakness is ABSENCE of content (e.g., "no named prior methods
anywhere"), quote the SHORTEST sentence/phrase where a reviser would most
naturally ADD the missing content — but still only VERBATIM from the plan.

**CONCISENESS**: Keep your <reasoning> concise (≤ 400 words). The locus and
score are the critical outputs. Avoid long exposition.

Format (output exactly this structure after the evaluation block):
<locus>
[
  {{"section_hint": "Motivation", "quote": "...", "why": "..."}}
]
</locus>
"""


def build_single_signal_prompt(
    goal: str, plan: str, signal: SignalSpec,
    *, emit_locus: bool = False, emit_critique: bool = False,
) -> str:
    """Build a prompt that asks the grader to score ONE signal.

    Args:
        emit_locus: When True AND `signal.locus_directive` is set, append a
            <locus> output block requesting verbatim weak-span quotes. Default
            False: standard grading pass (Pass 1) does not request loci. Only
            the CR-v6 bottleneck re-grading pass (Pass 2) sets emit_locus=True.
        emit_critique: When True, require a <critique> block with 1-3
            sentences describing what content is missing, weak, or
            unconvincing for this signal. Actionable and specific, but NOT
            verbatim quotes (separate from locus, which IS verbatim). Used
            by CR-v7 per-signal context REINFORCE to generate a per-signal
            advice string that is stored in the buffer and used at both
            revision-sampling time (all 8 critiques concatenated) and
            loss-computation time (single critique per context_i).
    """
    signal_block = _format_signal_block(signal)
    plan = _truncate_plan(plan)
    locus_block = _build_locus_output_block(signal) if emit_locus else ""
    critique_output_block = ""
    if emit_critique:
        critique_output_block = (
            f"        <critique>In 1-3 sentences, explain WHAT content is "
            f"missing, weak, or unconvincing for {signal.name} per the rubric "
            f"above. Be actionable and specific; describe the deficiency "
            f"conceptually. Do NOT quote plan text verbatim.</critique>\n"
        )

    return f"""{SHARED_PREAMBLE}

You will evaluate the following research plan on ONE dimension: **{signal.name}**.

# Research Goal
{goal}

# Research Plan
{plan}

# Evaluation Dimension

{signal_block}

---

# Output Format

<evaluation>
    <dim id="{signal.id}">
        <reasoning>Your analysis (do required reasoning steps if any).</reasoning>
        <score>INTEGER 1-5</score>
{critique_output_block}    </dim>
</evaluation>{locus_block}

Begin your evaluation now."""


# =============================================================================
# Parsing
# =============================================================================

def parse_scores(xml_text: str) -> dict[str, dict]:
    """Parse evaluation XML into {signal_id: {score, reasoning, critique}}.

    Robust to malformed XML from weaker graders (e.g., 4B models).
    Logs a warning if fewer than 50% of signals are successfully parsed.

    The `critique` field is the optional prose emitted when
    `build_single_signal_prompt(..., emit_critique=True)` was used; empty
    string when not present (CR-v7 per-signal context REINFORCE).
    """
    results = {}
    dim_blocks = re.findall(
        r'<dim\s+id="([^"]+)">(.*?)</dim>',
        xml_text,
        flags=re.DOTALL,
    )
    for dim_id, body in dim_blocks:
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", body, re.DOTALL)
        score_match = re.search(r"<score>\s*(\d+)\s*</score>", body, re.DOTALL)
        critique_match = re.search(r"<critique>(.*?)</critique>", body, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        score = int(score_match.group(1)) if score_match else None
        critique = critique_match.group(1).strip() if critique_match else ""
        if score is not None and 1 <= score <= 5:
            results[dim_id] = {"score": score, "reasoning": reasoning, "critique": critique}
        else:
            results[dim_id] = {"score": None, "reasoning": reasoning, "critique": critique}

    n_parsed = sum(1 for v in results.values() if v["score"] is not None)
    n_dims_found = len(dim_blocks)
    # In separate_call mode only 1 dim is expected; warn only when parsing
    # fails relative to how many dims the grader actually returned.
    if n_dims_found == 0 or (n_dims_found > 1 and n_parsed < n_dims_found * 0.5):
        logger.warning(
            f"parse_scores: {n_parsed}/{n_dims_found} dims parsed successfully. "
            f"Grader output may be malformed (first 200 chars: {xml_text[:200]!r})"
        )
    return results


def aggregate_critique_across_repeats(
    parsed_per_repeat: list[dict[str, dict]],
    median_scores: dict[str, int | None],
) -> dict[str, str]:
    """Pick one critique per signal from `grader_repeats` parsed outputs.

    Strategy: for each signal, prefer the critique from the repeat whose
    score equals the median used downstream. Ties broken by first non-empty
    critique. If no repeat has the median score or all critiques are empty,
    return the first non-empty critique encountered; fall back to "".

    Used by CR-v7 to land a single critique string per signal in buffer
    entries, consistent with the score actually used in the aggregate.

    Args:
        parsed_per_repeat: one `parse_scores` output per grader repeat.
            Each element maps signal_id -> {score, reasoning, critique}.
        median_scores: the median score per signal_id (as used for
            aggregate_reward). Values may be None when all repeats failed.

    Returns:
        dict signal_id -> critique string (possibly empty).
    """
    out: dict[str, str] = {}
    # Collect every signal_id mentioned in any repeat
    all_sids: set[str] = set()
    for parsed in parsed_per_repeat:
        all_sids.update(parsed.keys())

    for sid in all_sids:
        target_score = median_scores.get(sid)
        # Candidate critiques from repeats matching the median score
        median_matches = []
        non_empty = []
        for parsed in parsed_per_repeat:
            entry = parsed.get(sid)
            if not entry:
                continue
            crit = entry.get("critique", "") or ""
            crit = crit.strip()
            if crit:
                non_empty.append(crit)
                if target_score is not None and entry.get("score") == target_score:
                    median_matches.append(crit)
        if median_matches:
            out[sid] = median_matches[0]
        elif non_empty:
            out[sid] = non_empty[0]
        else:
            out[sid] = ""
    return out


# =============================================================================
# Locus parsing (CR-v6)
# =============================================================================

@dataclass
class LocusEntry:
    """A single locus entry parsed from a grader's <locus> block.

    section_hint: Section name hint (e.g., "Methodology"); "" if grader unsure
    quote: Verbatim substring of the plan body (≤ 250 chars by prompt
        convention; hard invariant: `quote in plan_body` is True)
    why: 1-2 sentence explanation of why the span hurts the signal
    """
    section_hint: str
    quote: str
    why: str


_LOCUS_BLOCK_RE = re.compile(r"<locus>\s*(.*?)\s*</locus>", re.DOTALL | re.IGNORECASE)


def _find_best_verbatim_match(
    approximate_quote: str, plan_body: str, min_ratio: float = 0.50,
) -> str | None:
    """Find the plan substring that best matches an approximate grader quote.

    Qwen3-30B grader frequently paraphrases: condenses markdown lists into
    single sentences, strips **bold** formatting, truncates with "...". The
    SEMANTIC target is usually correct but the literal text doesn't match.

    Strategy: slide a LINE-BASED window over the plan (natural boundaries),
    normalize both sides (strip markdown + collapse whitespace), and score
    with SequenceMatcher. Return the raw (un-normalized) plan span.

    Returns the best-matching plan substring, or None if no match exceeds
    `min_ratio`. The returned string is guaranteed `in plan_body`.

    Discovered in Phase 1.5 smoke test: 0/10 plans had literal-verbatim
    locus quotes despite all 10 having semantically correct loci.
    """
    from difflib import SequenceMatcher

    def _norm(s: str) -> str:
        s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)       # strip bold
        s = re.sub(r"^\s*[-*+]\s+", " ", s, flags=re.MULTILINE)  # strip list bullets
        return " ".join(s.split()).lower()

    q_norm = _norm(approximate_quote)
    if len(q_norm) < 10:
        return None

    lines = plan_body.split("\n")
    best_ratio = 0.0
    best_span: str | None = None

    # Try all contiguous line windows up to 15 lines (covers most sections)
    max_window = min(15, len(lines))
    for start in range(len(lines)):
        for end in range(start + 1, min(start + max_window + 1, len(lines) + 1)):
            candidate_raw = "\n".join(lines[start:end])
            c_norm = _norm(candidate_raw)
            if not c_norm:
                continue
            # Quick length filter: avoid comparing tiny/huge candidates
            len_ratio = len(c_norm) / max(len(q_norm), 1)
            if len_ratio < 0.2 or len_ratio > 6.0:
                continue
            r = SequenceMatcher(None, q_norm, c_norm).ratio()
            if r > best_ratio:
                best_ratio = r
                best_span = candidate_raw.strip()

    if best_ratio < min_ratio or best_span is None:
        return None

    # Verify the span is a verbatim substring of plan_body
    if best_span in plan_body:
        return best_span

    # strip() may have broken the exact-substring property for edge lines;
    # try the raw (un-stripped) version
    for start in range(len(lines)):
        for end in range(start + 1, min(start + max_window + 1, len(lines) + 1)):
            raw = "\n".join(lines[start:end])
            if raw.strip() == best_span and raw in plan_body:
                return raw

    return None


def parse_locus(response: str, plan_body: str) -> list[LocusEntry] | None:
    """Parse `<locus>...</locus>` from grader response with fuzzy verbatim matching.

    The grader frequently paraphrases plan text (condenses markdown lists,
    strips bold formatting, truncates). When the literal quote doesn't match,
    we use `_find_best_verbatim_match` to find the closest actual plan
    substring. The returned LocusEntry.quote is ALWAYS a verbatim substring
    of plan_body (guaranteed by the fuzzy-match → exact-recovery pipeline).

    Returns:
        list[LocusEntry]: parsed entries with corrected verbatim quotes.
            Empty list = grader emitted `<locus>[]</locus>` (score 5).
        None: no `<locus>` block, malformed JSON, or no entry could be
            matched (even fuzzily) to plan_body text. Callers fall back
            to legacy revision path.
    """
    match = _LOCUS_BLOCK_RE.search(response)
    if not match:
        return None

    payload = match.group(1).strip()
    try:
        raw = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(raw, list):
        return None

    entries: list[LocusEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        quote = item.get("quote")
        section_hint = item.get("section_hint", "")
        why = item.get("why", "")
        if not isinstance(quote, str) or not quote:
            return None
        if not isinstance(section_hint, str) or not isinstance(why, str):
            return None
        # Try exact match first (cheapest)
        if quote in plan_body:
            matched_quote = quote
        else:
            # Fuzzy match: find the best actual plan substring
            matched_quote = _find_best_verbatim_match(quote, plan_body)
            if matched_quote is None:
                return None  # Can't locate this locus at all
        entries.append(LocusEntry(
            section_hint=section_hint,
            quote=matched_quote,
            why=why,
        ))

    return entries


# =============================================================================
# Aggregation
# =============================================================================

def median_over_repeats(repeat_scores: list[dict[str, dict]]) -> dict[str, int | None]:
    """Take the median score for each signal across multiple grading repeats."""
    if not repeat_scores:
        return {s.id: None for s in SIGNALS}

    out = {}
    for spec in SIGNALS:
        vals = []
        for r in repeat_scores:
            info = r.get(spec.id) or {}
            score = info.get("score")
            if score is not None:
                vals.append(score)
        if len(vals) == 0:
            out[spec.id] = None
        else:
            # Use integer median (round down for even counts to be deterministic)
            out[spec.id] = int(statistics.median_low(vals))
    return out


# =============================================================================
# Sanity prints
# =============================================================================

if __name__ == "__main__":
    from pathlib import Path
    # Sanity fixtures live under the project's sanity_check dir after the
    # 2026-04-10 restructure (CONVENTIONS.md load-bearing paths).
    repo_root = Path(__file__).resolve().parents[3]
    sanity_dir = repo_root / "projects" / "ttt_discover" / "analysis" / "sanity_check"
    ref = (sanity_dir / "perturbations" / "00_reference.txt").read_text()
    goal = (sanity_dir / "research_goal.txt").read_text()

    prompt_single = build_single_call_prompt(goal, ref)
    prompt_one_signal = build_single_signal_prompt(goal, ref, SIGNALS[0])

    print(f"Signals: {len(SIGNALS)}")
    for s in SIGNALS:
        has_locus = s.locus_directive is not None
        print(f"  {s.id}: {s.name} (cot={bool(s.cot_scaffolding)}, locus={has_locus})")
    print()
    print(f"Single-call prompt: {len(prompt_single)} chars (~{len(prompt_single)//4} tokens)")
    print(f"Per-signal prompt (S1, locus on): {len(prompt_one_signal)} chars "
          f"(~{len(prompt_one_signal)//4} tokens)")

    # Contrast: S9 has no locus_directive, so no locus block appended
    s9 = next(s for s in SIGNALS if s.id == "S9_focus")
    prompt_s9 = build_single_signal_prompt(goal, ref, s9)
    print(f"Per-signal prompt (S9, locus off): {len(prompt_s9)} chars "
          f"(~{len(prompt_s9)//4} tokens)")
    print(f"Locus block overhead (S1 − S9): {len(prompt_one_signal) - len(prompt_s9)} chars")
    print()
    print(f"Total per-signal cost: {len(prompt_one_signal) * len(SIGNALS)} chars "
          f"(~{len(prompt_one_signal) * len(SIGNALS) // 4} tokens) for {len(SIGNALS)} calls")
