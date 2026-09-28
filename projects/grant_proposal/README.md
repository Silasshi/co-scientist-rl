# Direction 4: Grant Proposal Generation

**Status**: Initial setup
**Target**: NeurIPS 2026
**Code**: `src/co_scientist/grant_proposal/`

## Overview

Using the CR-v7 critique-revise + multi-signal reward + RL pipeline (proven in D3) to generate research grant proposals. The pipeline is forked from D3 (ttt_discover) with independent signal evolution.

## Motivation

- Grant proposals have **explicit public rubrics** (NSF Intellectual Merit + Broader Impacts, NIH 5-criteria, ERC evaluation criteria) — more grounded than self-defined signal rubrics
- **Larger revision search space**: proposals are 5-15 pages vs single-page research plans, with more structured sections (Specific Aims, Research Strategy, Budget Justification)
- Emphasis on **structural soundness** over genuine scientific novelty — better fit for the pipeline
- Suggested by the project advisor

## Relationship to D3

- **Forked signal system**: `grant_signal_reward.py` (independent from `ten_signal_reward.py`)
- **Same pipeline architecture**: CR-v7 with per-signal REINFORCE
- **Same shared infrastructure**: api_profiles, eval_core, seven_signal_reward (hard gates)
- **Independent runs/analysis/knowledge**: clean separation from D3 data

## Code

| File | Purpose |
|---|---|
| `src/co_scientist/shared/grant_signal_reward.py` | Forked signal definitions (adapt to grant rubrics) |
| `src/co_scientist/grant_proposal/train_cr_v7.py` | CR-v7 trainer (primary) |
| `src/co_scientist/grant_proposal/train_buffer_ttt.py` | Base buffer training framework |
| `src/co_scientist/grant_proposal/train_critique_revise.py` | Shared utilities + legacy paths |
| `src/co_scientist/grant_proposal/opus_eval_agent.py` | Background Opus depth evaluation |

## Dataset Plan

- ~Dozens of winning proposals from [Open Grants](https://www.ogrants.org/)
- NSF solicitations as instructions (extractable research goals + explicit review criteria)
- Diversity: multiple disciplines and funding agencies

## Key Differences from D3

| Dimension | D3 (Research Plans) | D4 (Grant Proposals) |
|---|---|---|
| Output length | ~750 words | 5-15 pages |
| Evaluation rubric | Self-defined signals | Agency-published review criteria |
| Reference data | 60 arxiv-derived plans | Open Grants winning proposals |
| Novelty requirement | High (scientific contribution) | Moderate (soundness + framing) |
| Target venue | ICML AI4Science | NeurIPS 2026 |
