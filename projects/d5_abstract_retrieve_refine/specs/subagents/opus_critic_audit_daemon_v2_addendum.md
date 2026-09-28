# D5 audit-prompt v2 — daemon branching addendum

This file SUPPLEMENTS `opus_critic_audit_daemon.md`. It does NOT replace.

## When to use v2

When an audit request JSON contains the field `"prompt_version": "v2"`, the
daemon (or a one-off main-agent dispatched subagent processing that request)
MUST use the v2 per-plan subagent prompt below INSTEAD of the v1 audit prompt
described in the main daemon spec.

If `prompt_version` is missing or set to `"v1"`, behavior is unchanged.

## v2 changes versus v1

1. **Two-pass mandatory enumeration**: subagent must extract `claim_list` and
   `scaffold_list` BEFORE scoring 4 dims.
2. **New anchors**: keyed to substance density (RHS equations, named
   hyperparameters, benchmark numbers) not feature presence.
3. **Anti-pattern penalty**: -1/dim, capped, floor 1.
4. **Output JSON has 6 fields** (not 4): `claim_list`, `scaffold_list`,
   `per_dim_raw`, `per_dim_penalty`, `per_dim_scores`, `total`.
5. **`max_tokens` raised** from 256 → 1500 to accommodate the lists.

`per_dim_scores` keeps the v1-shape `{math, novelty, realism, rigor}` keys
so trainer-side `OpusAuditClient.collect_all()` aggregation works unchanged.

## v2 per-plan subagent prompt (verbatim)

When fanning out a Task subagent to score one plan under v2, use this prompt
template (substitute `<goal>` and `<plan>` from the audit request):

```
You are an expert research methodology reviewer auditing a research plan.
Your task is to score the plan on 4 dimensions, each 1-5 (integer), for a /20 total.

Reviewers like you have a known failure mode: rewarding STRUCTURAL SCAFFOLDING
("Pattern 1 / Pattern 2", numbered methodology lists, headed subsections)
even when the scaffolding is empty — no inline equation RHS, no specific
hyperparameter numbers, no concrete operator definitions. You must avoid this
failure mode by following a TWO-PASS procedure: first commit to a list of
verifiable claims AND a list of hollow scaffolds, THEN score using anchors
that explicitly reference your enumeration.

# Research Goal
<goal>

# Research Plan to Evaluate
<plan>

# Pass 1 — Mandatory Claim Extraction (do this BEFORE scoring)

Extract two disjoint lists.

A) `claim_list`: every VERIFIABLE specific factual claim. Include only if:
   - Equations have a full right-hand side
     (`L(θ) = -E[(r/max r) · log π_θ(a|s)]` ✓; `J(θ) is derived` ✗)
   - Hyperparameters have numbers (`LoRA rank 64`, `learning rate 2e-5`,
     `10K compute budget`, `Llama-3-8B`). "Adaptive temperature" ✗.
   - Named real benchmarks/datasets with cited prior numbers
     (`MATH benchmark, GPT-4 baseline 50.4%` ✓; `mathematical theorem proving` ✗)
   - Deterministic operators
     (`score(s) = α·R(s) + (1-α)·log(1+n(s))` ✓; `score combines value and
     uncertainty` ✗)

B) `scaffold_list`: every STRUCTURAL SCAFFOLD that lacks an inline mechanism.
   IMPORTANT: scaffolds are NOT just "Pattern N" labels. ANY vague mechanistic
   gesture WITHOUT a verifiable claim is a scaffold:

   - "Pattern N", "Step N", numbered methodology bullets without RHS/operator/value
   - Placeholder equations (`J(θ) is derived`, `we minimize cross-entropy`)
   - Domains named without benchmark numbers
   - **Vague prose hyperparameters with no numeric value**
     (`use a small learning rate`, `appropriate batch size`, `moderate temperature`)
   - **Mechanisms in passive voice / generic verbs**
     (`update parameters via gradient ascent`, `select high-quality samples`,
     `apply policy gradient`, `rewards are weighted appropriately`)
   - **Generic algorithm names without instantiation**
     (`use REINFORCE`, `apply DPO`, `LoRA fine-tuning` without rank,
     `standard backprop` with no learning rate)

CRITICAL: a plan can have NO "Pattern N" labels yet still be all scaffold.
A plan that says "we use REINFORCE with a small learning rate" has 2
scaffolds (REINFORCE without hyperparam, "small learning rate" unnamed)
and 0 verifiable claims.

Each list entry MUST quote ≤25 words from the plan verbatim.

# Pass 2 — Score 4 Dimensions, 1-5 Each

## 1. Mathematical Formalism (1-5)
- 1: claim_list contains 0 equations with RHS
- 2: 1 standard formula RHS, no derivation
- 3: ≥1 adapted/novel formula RHS specific to this plan
- 4: full objective + a stated gradient or update rule (both with RHS)
- 5: 4 + a limit/convergence sketch or a one-character deviation from a named
     classical objective with justification
  Reference anchor (5/5): D3-canonical TTT-Discover plan with J_β entropic
  objective, ∇J_β policy gradient, adaptive β via KL budget, MAX-PUCT one-
  character deviation from AlphaZero scores 5/5/5/5.

## 2. Algorithmic Novelty (1-5)
- 1: standard toolkit named without specific operator
- 2: known techniques with minor adaptation; ≥1 deterministic operator in claim_list
- 3: non-obvious combination with ≥2 deterministic operators + clear justification
- 4: substantive modification with at least one novel-looking operator
- 5: novel algorithm design with rigorous justification

## 3. Implementation Realism (1-5)
- 1: 0 specific hyperparameters in claim_list; multiple fatal errors
- 2: 1 specific hyperparameter; otherwise vague
- 3: 2-3 specific hyperparameters; one error or vague on key details
- 4: ≥4 specific hyperparameters covering model size, optimizer, compute
     budget, and an algorithm-specific knob (e.g., LoRA rank). All numerical
     claims internally consistent.
- 5: 4 + every key parameter justified or tied to a published baseline

## 4. Empirical Rigor (1-5)
- 1: 0 named real benchmarks in claim_list
- 2: ≥1 named real benchmark, no prior number
- 3: ≥1 named real benchmark with at least one prior baseline number cited
- 4: ≥2 named real benchmarks with published baseline numbers and comparison axis
- 5: 4 + a testable falsification criterion or open problem with prior AI numbers

# Anti-Pattern Penalty Rule (mandatory; applied AFTER raw scoring)

For each scaffold entry occupying a structural position in a dimension's
evidence, subtract -1 from that dimension's raw score (cap penalty at -1
per dimension; floor each final dim at 1):

- "Pattern N" / "Step N" mechanism in scaffold_list (not claim_list) on
  the math axis: -1 to Mathematical Formalism
- Pattern label whose operator is in scaffold_list (passive / no RHS):
  -1 to Algorithmic Novelty
- Hyperparameter named without a number ("adaptive temperature"): -1 to
  Implementation Realism
- Domain listed without a benchmark number: -1 to Empirical Rigor

The penalty is a sentinel (max -1/dim, total cap -4/20), not a multiplier.
Final dim score = max(1, raw - penalty).

# Output Format

Respond with ONLY this JSON (no other text, no markdown fence):

{"claim_list": [
    {"kind": "equation"|"hparam"|"benchmark"|"operator", "quote": "<≤25 words>"}
  ],
  "scaffold_list": [
    {"kind": "pattern_label"|"placeholder_eq"|"unnamed_domain"|"passive_mechanism",
     "quote": "<≤25 words>",
     "penalized_dim": "math"|"novelty"|"realism"|"rigor"|null}
  ],
  "per_dim_raw":     {"math":<int>, "novelty":<int>, "realism":<int>, "rigor":<int>},
  "per_dim_penalty": {"math":<0|1>, "novelty":<0|1>, "realism":<0|1>, "rigor":<0|1>},
  "per_dim_scores":  {"math":<int>, "novelty":<int>, "realism":<int>, "rigor":<int>},
  "total": <int 4-20>}
```

## Aggregation back into batch response

After all per-plan v2 subagents return, the daemon aggregates into
`audit_responses/iter_NNN.json` with the SAME outer schema as v1, except
each plan's verdict is the full v2 JSON object (claim_list etc. preserved):

```json
{
  "iter": <int>,
  "kind": "audit",
  "prompt_version": "v2",
  "completed_at": "<ISO-8601 UTC>",
  "judgments": {
    "<plan_id>": {
      "claim_list": [...],
      "scaffold_list": [...],
      "per_dim_raw": {...},
      "per_dim_penalty": {...},
      "per_dim_scores": {"math":3,"novelty":2,"realism":3,"rigor":2},
      "total": 10
    },
    ...
  }
}
```

## Per-plan max_tokens

v1 used 256 tokens (sufficient for the 4-int JSON). v2 needs ~1500 tokens
because of `claim_list` + `scaffold_list` enumerations. If a v2 subagent
runs out of tokens mid-output, the response will be truncated JSON and the
trainer-side parser will fall back to null scores. Use `max_tokens=1500`.

## Failure handling

If a per-plan v2 subagent returns malformed JSON (e.g., truncated or
prose preamble), write the placeholder verdict for that plan_id:
```json
{"per_dim_scores": {"math": null, "novelty": null, "realism": null, "rigor": null},
 "total": null,
 "_raw": "<truncated subagent output>",
 "_warning": "v2_parse_failure"}
```
and continue with the rest of the batch. Log to `critic_audit_daemon.log`.
