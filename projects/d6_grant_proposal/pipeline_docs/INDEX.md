# D6 Pipeline — Module Index

One markdown doc per source module under `src/co_scientist/d6_grant_proposal/`. Each doc states the module's pipeline stage, purpose, key callables, dependencies, and the capability it trains or evaluates.

## Pipeline data flow

```
                    research_goal.md   reference_proposal.md
                              │              │
                              ▼              ▼
                      ┌─────────────────────────────┐
                      │  EXTRACT (one-time / goal)  │
                      │  extract_oracle   ──► oracle.jsonl
                      │  build_slim_oracle ──► slim.md   (~3k tok, formatted)
                      │  extract_gold     ──► gold.json (OPD-grounded only)
                      └──────────────┬──────────────┘
                                     │
                                     ▼
                      ┌─────────────────────────────┐                                                          ──►  ►  REFINE (training loop)        │
                      │  train_baseline   [frozen baseline — no grad updates]
                      │  train_opd      [OPD  — Opus reviewer + sol-mask + PPO-clip]
                      │  train_grpo    [GRPO    — Qwen-30B 12-signal reward, no reviewer]
                      │  train_opd_grounded / train_opd_kl_anchor*    [OPD-grounded / OPD-KL-anchor — OPD + grounding bonus / KL anchor]
                      └──────┬──────────────────┬───┘
                             │                  │
                             │     calls        │
                             ▼                  ▼
              shared/grant_signal_reward.py    Opus 4.7 file-bus subagent
                  (12-signal grader,            using review/critic_instructions.md
                   Qwen-30B-A3B via tinker)     privileged: reference_proposal.md
                                     │
                                     ▼
                      ┌─────────────────────────────┐
                      │  FEEDBACK / EVAL            │
                      │  audit  (12-signal       │  ◄── audit/absolute_score.md
                      │             isolated batch) │      Opus 4.7 subagents
                      └─────────────────────────────┘

INFRASTRUCTURE (cross-cutting, used by all stages):
   prompt_loader.py  — load/render .md templates from projects/d6_grant_proposal/prompts/
   prompts.py     — build_student/teacher/critic_request/audit payloads
   signals.py     — re-export D4 12-signal rubric
   grounding.py   — match equations/citations vs gold (OPD-grounded only)
```

## Module table

| Stage | Module | Purpose (one line) |
|---|---|---|
| Infrastructure | [prompt_loader](prompt_loader.md) | Load + render `.md` prompt templates |
| Infrastructure | [prompts](prompts.md) | Build SDPO / audit payloads from templates |
| Infrastructure | [signals](signals.md) | Re-export D4's 12-signal rubric |
| Extract | [extract_oracle](extract_oracle.md) | Opus extracts typed oracle items from `reference_proposal.md` |
| Extract | [extract_oracle_slim](extract_oracle_slim.md) | Truncate oracle to ~3k tokens, render as markdown |
| Extract | [extract_gold](extract_gold.md) | Extract equations + citations from reference (OPD-grounded) |
| Retrieve | [grounding](grounding.md) | Match policy spans against gold (equation Jaccard, citation regex) |
| Refine | [train_baseline](train_baseline.md) | Frozen-model baseline; sample + grade, no training |
| Refine | [train_opd](train_opd.md) | **Main contribution.** SDPO option-(c) HER + sol-mask + PPO-clip (privileged-Opus reviewer) |
| Refine | [train_grpo](train_grpo.md) | GRPO ablation with Qwen-30B 12-signal reward, no reviewer |
| Refine | [train_opd_grounded](train_opd_grounded.md) | OPD-grounded: OPD + per-token citation/equation grounding bonus |
| Refine | [train_opd_kl_anchor](train_opd_kl_anchor.md) | OPD-KL-anchor scaffold (NotImplementedError; needs oracle SFT first) |
| Feedback | [audit](audit.md) | Isolated 12-signal scoring via parallel Opus subagents |

## Conventions

- Config dataclasses use [chz](https://github.com/openai/chz) (`@chz.chz`) — invoke with `python -m co_scientist.d6_grant_proposal.<module> goal_domain=ai goal_name=02_foundational_rl log_path=...`
- All prompts loaded via `prompt_loader.load_prompt` / `render_prompt` from `projects/d6_grant_proposal/prompts/`
- All goal paths use `{goal_domain}/{goal_name}/` structure; no `goal_id` slugs (see CONVENTIONS.md)
- Score scale: D4 12-signal weighted mean (0.0–1.0). Do NOT mix with D5's /45 scale
- Pilot goal: `ai/02_foundational_rl` (locked 2026-04-30; see DECISIONS.md)

## Read order for new contributors

1. This INDEX
2. `prompt_loader.md` + `prompts.md` (how prompts are wired)
3. `train_opd.md` (the main trainer)
4. `audit.md` (how outputs are scored)
5. The baseline / GRPO / deferred-trainer docs as needed
