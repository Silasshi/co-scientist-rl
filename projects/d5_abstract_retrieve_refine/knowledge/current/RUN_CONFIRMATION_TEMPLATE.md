# Run Confirmation Template — D5

**Mandatory for every training and eval run after 2026-04-25 realignment.**

The user has explicitly required pre-run confirmation. Before launching ANY run, fill out
this template, present to user, get explicit "yes go" before executing.

NEVER skip this confirmation.

---

## Run identity

- **Run name**: <e.g. `μ_v2_dryrun_2026_05_NN` or `σ_v2_eval_2026_05_NN`>
- **Greek-letter label**: <ξ / σ / μ / α / β / δ / ε>
- **Run dir**: `projects/d5_abstract_retrieve_refine/runs/<YYYY_MM_DD>_<run_name>/`
- **Estimated wall-clock**: <X minutes / hours>
- **Estimated cost**: <Tinker compute hours + Opus subagent calls equivalent in USD>

## Pipeline diagram

```
[Goal] ─────┐
            ├──> [Prompt template <name>] ──> [Model: <Qwen3-30B-A3B / 235B / Opus>] ──> [Plan output]
[Oracle?] ──┤                                                                       │
[Critique?] ┘                                                                       ├──> [Eval]
                                                                                    │
                                                                                    └──> [Audit]
```

(Customize for the specific run — show subagents, file-bus, training loop, etc.)

## Verbatim prompt template

The exact string fed to the model. Copy-paste from code, no paraphrasing.

```
<system or user message text, with {goal}, {oracle}, {critique} markers shown>
```

If multiple prompts are involved (teacher / student / critic / audit), show ALL of them.

## What model sees, what it doesn't

| Variable | In prompt? | Source |
|---|---|---|
| goal | yes/no | `dataset/research_goal.txt` |
| oracle abstraction | yes/no | `data/oracles/oracle_v2_2026_04_26_build/final.md` |
| reference plan (`reference_solution.txt`) | yes/no | `dataset/reference_solution.txt` |
| previous-iter critique | yes/no | `critic_responses/iter_N.json` |
| source paper full text | yes/no | `data/source_paper/v1.md` |

## Eval metric + comparison rule

- Metric: <pairwise tournament / audit v3 absolute / both>
- Comparison: <which existing data point are we comparing against?>
- Sample size: <n=8 plans / n=16 plans>
- Statistical aggregation: <mean / median / per-dim breakdown>

## Decision rule

- PASS if: <specific numerical threshold + which data point it must beat>
- FAIL if: <specific numerical threshold + redirect to next path>
- MARGINAL if: <range + investigation steps>

## Cost breakdown

- Tinker compute: <X GPU-hours × $Y/hr = $Z>
- Opus subagent calls: <K calls × ~$N/call = $M>
- Total: <$T>

## Risks identified

- <risk 1 + mitigation>
- <risk 2 + mitigation>
- ...

## User confirmation

Send the above to user via chat. Wait for explicit "yes go" or specific changes requested.

---

## Worked example: σ_v2 evaluation

Below is what a filled-out spec looks like.

### Run identity
- Run name: `sigma_v2_eval_2026_05_NN`
- Greek-letter label: σ
- Run dir: `projects/d5_abstract_retrieve_refine/runs/2026_05_NN_sigma_v2_eval/`
- Estimated wall-clock: ~10 min Tinker + ~30 min Opus audit subagents
- Estimated cost: ~$5 ($2 Tinker + $3 Opus equivalent via subagent)

### Pipeline diagram

```
goal (research_goal.txt) ────┐
                             ├──> [build_prompt_with_oracle_v2(goal, oracle)] ──> [frozen Qwen3-30B-A3B] ──> [8 plans]
oracle_v2 (Opus-generated)───┘                                                                                   │
                                                                                                                  ├──> Pairwise: σ_v2 vs ξ + audit_v3 absolute scoring
                                                                                                                  │
                                                                                                                  └──> 8 plans × audit_v3 (9 dims) via Opus subagent
```

### Verbatim prompt template

```
I will provide you a research scenario and a structured set of methodological insights
extracted from a high-quality research plan's bibliography. Use these as grounding for
your plan, but do not copy phrasing literally.

# Scenario
{goal}

# Methodological insights from prior work
{oracle_abstraction}

Write your research plan inside <solution>...</solution>. Target 600 words, max 750.
Use sections: Problem Statement / Background / Hypothesis / Methodology / Evaluation /
Limitations.
```

### What model sees / doesn't

| Variable | In prompt? | Source |
|---|---|---|
| goal | YES | `dataset/research_goal.txt` |
| oracle abstraction (v2) | YES | `data/oracles/oracle_v2_2026_04_26_build/final.md` |
| reference plan | NO | (not used for σ) |
| previous-iter critique | NO | (frozen, no training) |
| source paper full text | NO | (not used for σ) |

### Eval metric + comparison rule

- Pairwise: 8 random-pair matchups vs ξ baseline (frozen 30B + goal only)
- Audit v3: 9-dim score per plan, mean across 8 plans
- Compare to v1+v2 archived numbers as sanity, but treat v3 as authoritative

### Decision rule

- PASS if: σ wins ≥6/8 pairwise vs ξ AND σ audit_v3 mean ≥ 12/20 normalized
- FAIL if: σ ties or loses pairwise vs ξ
- MARGINAL: σ wins 5/8

### Cost: ~$5 total
### Risks: minimal (frozen run, no training)
### Awaiting user "yes go"
