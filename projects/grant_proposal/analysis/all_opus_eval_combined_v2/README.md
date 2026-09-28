# All Opus Eval — Re-scored with depth_audit_v2 Template

*Created 2026-04-23.* This directory contains re-evaluated Opus scores for
all 37 entries in `../all_opus_eval_combined.jsonl` using a **single consistent
template** so that cross-run / cross-rubric comparison is valid.

## Template used: `depth_audit_v2` (length-normalized holistic)

Located in `projects/grant_proposal_v2/scripts/mid_run_depth_audit.py`.

4 dimensions × 1-10 each = 40 total:

1. **DEPTH** — mechanism-level reasoning, beyond surface description
2. **METHODS** — named algorithms, specific techniques, measurable procedures
3. **FEASIBILITY** — resource-realistic, Monday-morning actionable
4. **GROUNDING** — relevant load-bearing citations, correct literature engagement

**Critical length-normalization instruction**:
> Score the plan's *density* and *specificity*, NOT its verbosity. Imagine
> every plan is rewritten to exactly 1500 words; score based on the per-word
> quality that would result. A short dense plan and a long padded plan should
> receive identical scores if the underlying commitments are equivalent.
> Extra length without extra substance is NOT rigor.

## Why re-score

The original `all_opus_eval_combined.jsonl` mixed evaluations from two
different rubrics:
- **"Better Reward (R1-R8, 1-10)"**: 8 dims × 10 = 80 max (used for
  BR_B4_235B, BR_MAIN_235B, SO_*). Scale is incompatible with...
- **"D4-v7 standard (G1-G13, 1-5)"**: 12 signals × 5 = 60 max, weighted. Tool
  for tracking training dynamics, not absolute quality.

Neither was length-normalized, so longer plans got systematically inflated.
All comparisons in this v2 dataset use the same rubric-free length-normalized
holistic template, giving a single /40 scale.

## Source of each re-score

Each entry has a `source` field indicating how the v2 score was produced:

- `"v2 spawn this session"` = Opus subagent spawned fresh today with
  `depth_audit_v2` prompt against the plan file listed in the manifest
- `"session run <run_name> iter N"` = plan came from training-buffer best-plan
  at iter N of a run we tracked this session; same `depth_audit_v2`
  prompt applied

Reuse logic: if we already ran `depth_audit_v2` on the best-buffer-plan of
the same run this session, the score is borrowed from there. For the other
24 entries, fresh Opus subagents were spawned with prompts in
`/tmp/reeval_{row}_{tag}.prompt`.

## Files

- `opus_scores_v2.jsonl` — 37 entries with D/M/F/G/total + `prev_total` for
  comparison with the original rubric-specific scores
- `manifest.jsonl` — mapping from original row index → tag → plan_path → goal
  → status (pending/already_done/missing_file)

## Key findings from v2 re-score

### References are 29-33/40 (all 5 goals)
The reference proposals retrieved from each goal's `reference_proposal.md`
are consistently strong when length-normalized:
- 01_foundopt_reference: 32
- 05_causal_healthcare_reference: 33
- 07_chemo_toxicity_reference: 30
- 08_ecosystem_dynamics_reference: 29
- 09_sentencing_disparities_reference: 32
- 12_climate_displacement_reference: 31

**Exception**: `ablation_reference` scored 4/40 — this refers to
`sanity_check/reference_solution.txt` which is a TODO placeholder (not a
real reference), confirming that the previous eval (total=37) used a
different file.

### Generated plans cluster at 11-21/40
Training-buffer best plans score well below references:
- Peak: `SO_MAIN_235B` at 26/40 (Qwen3-235B, 46 iters, scores_only)
- Peak excluding 235B: `SO_B4_235B` at 24, `BR_B4_235B` at 21,
  `08_ecosystem_generated` at 21
- Median: 14-18
- Worst: ablation_worst series 11-12 (and C3_sdpo final 11)

Reference-to-generated gap: ≥10 points on all 5 cross-domain goals (29-33 ref
vs 17-21 generated). This is the cross-domain training ceiling DECISIONS.md
describes as the "19-point gap".

### Model scaling dominates
- SO_MAIN_235B (Qwen3-235B): 26
- SO_MAIN_30B (Qwen3-30B-A3B): 16
- 10-point gap from 235B → 30B alone, matching established finding.

### v7 C* vs D4v1 same-model (Qwen3-30B) comparison
- v7 C2 B4 final iter 24: 16
- v7 C3 SDPO final iter 34: 11 (worst MAIN, confirms SDPO Goodhart)
- v7 C4 aggregate final iter 34: 14
- d4v7_ablation_MAIN iter 6: 18 (D4v1 best, very early iter)
- d4v7_ablation_B4 iter 7: 18

On 30B, D4v1 early peaks (~18) > v7 final (~11-16). Longer training did not
improve quality on 30B when the grader=policy and the rubric stays fixed.

## How to reproduce

```bash
cd /home/silas/co-scientist-project
python projects/grant_proposal/analysis/_tmp_build_eval_bundles.py
# Spawns `/tmp/reeval_{row}_{tag}.prompt` for the 24 pending entries
# Then spawn Opus subagents pointing at each prompt:
#   Agent(subagent_type='general-purpose', prompt='Read /tmp/reeval_XX_TAG.prompt ...')
# and parse the <depth_audit>...</depth_audit> XML from each response.
```
