# D5 Pairwise Tournament v2 — 3 prompt variants per matchup

*Generated 2026-04-26 02:31:27, seed=100, n_pairs=8*

## Prompt variants used

### PV1: minimal — A/B/TIE + 1-sentence rationale

```
You are evaluating two research plans on the same scientific problem. Pick the better
one or call it a TIE.

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

Output JSON only:
{{"winner": "A|B|TIE", "rationale": "<1 sentence>"}}

```

### PV2: reviewer-lite — soundness/novelty/feasibility/empirical-rigor checklist

```
You are evaluating two research plans on the same scientific problem. Pick the better
one or call it a TIE. Consider:
- Soundness: are claims supported by math or empirical evidence?
- Novelty: is the proposed approach distinct from cited prior work?
- Feasibility: are compute, hyperparameters, and evaluation realistic?
- Empirical rigor: are baselines named with prior numbers; are stats/seeds specified?

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

Output JSON only:
{{"winner": "A|B|TIE", "rationale": "<1-2 sentences referencing the criteria>"}}

```

### PV3: v3-rubric verdict — 9-dim score per side then verdict

```
You are an expert ML/AI conference reviewer comparing two research plans on the same
scientific problem. First score each plan on 9 dimensions (1-5 each), then output a
verdict based on the per-dim differences.

# Dimensions

UNIVERSAL (1-5 each):
- U1 Soundness: claims supported by evidence
- U2 Significance: real problem with real impact
- U3 Originality: novel beyond pretraining-knowledge
- U4 Clarity: well-reasoned and structurally clear
- U5 Reproducibility: compute / hparams / stats specified

SUBFIELD (1-5 each, for test-time-search / search-with-LLMs papers):
- T1 Necessity: shows test-time RL is necessary
- T2 Disentanglement: separates parameter updates from search alone
- T3 Compute accounting: latency / cost honesty
- T4 Reward-hacking awareness: probes for shortcuts/Goodhart

# Research Goal
{goal}

# Plan A
{plan_a}

# Plan B
{plan_b}

# Output

Output JSON only with this schema:
{{
  "scores_a": {{"U1":<1-5>, "U2":<1-5>, ..., "T4":<1-5>}},
  "scores_b": {{"U1":<1-5>, ..., "T4":<1-5>}},
  "winner": "A|B|TIE",
  "rationale": "<2-3 sentences referencing the most decisive per-dim differences>"
}}

```

## Results table

| Matchup | la | lb | PV | la-wins | lb-wins | ties | Winner |
|---|---|---|---|---:|---:|---:|---|
| sigma_vs_xi | sigma | xi | PV1 | 8 | 0 | 0 | **sigma** |
| sigma_vs_xi | sigma | xi | PV2 | 8 | 0 | 0 | **sigma** |
| sigma_vs_xi | sigma | xi | PV3 | 8 | 0 | 0 | **sigma** |
