# M7 — Why Qwen3-30B self-review fails as SDPO critic (methodology lesson)

## Context

Phase 3 explored whether the Opus reviewer in the SDPO loop could be replaced by
the same Qwen3-30B model that's being trained — making the entire pipeline
(distillation + critic + plan generation) open-source-only without dependence on
a SOTA proprietary model. This would be a strong reproducibility/cost story.

We pre-validated the hypothesis with a cheap test (Smoke A) before committing to a
full κ_self training run. Result: 0-8 dispelled. This methodology doc documents
WHY for future open-source-self-distill work.

## Test design (Smoke A)

8 τ_v4 plans (varied quality) × two critiques each:
- Qwen3-30B-A3B base model via Tinker, given source paper as privileged info
- Opus 4.7 via subagent, given the same source paper as privileged info

Both critiques follow the same XML schema (`<distill_critique>` with
missing_critical / noise / faithfulness / improvement_directive children).

Then 8 Opus-as-judge subagents compare per-pair (position-randomized) which
critique is more useful: more specific named items, more accurate ground-truth
reference, more actionable improvement directive.

Cost: ~$13 Opus subagent + ~5 min Tinker. Wall ~30 min.

## Result

| | Count |
|---|---:|
| Opus critique preferred | **8/8** |
| Qwen3 critique preferred | 0/8 |
| Tie | 0/8 |
| A-position win rate | 4/8 (50% — not position bias) |

## Diagnosed mechanism: Privileged Info Comprehension gap

Across all 8 judge rationales, the explanation converges:

> "Qwen3 critique is directionally correct (mentions same conceptual terms — entropic
> objective, PUCT, LoRA) but stays at generic-bullet level. Opus critique provides
> the concrete formula/hparam/domain/baseline-number specificity Qwen3 lacks."

Specific examples (judge rationales):
- Opus extracts `J_β` formula, `KL budget γ=ln 2`, `PUCT c·P(s)·√(1+T)/(1+n(s))`,
  `LoRA rank 32`, `50 steps × 512 rollouts`, `~$500/run`, named domains
  (Erdős minimum-overlap, GPUMode TriMul, AtCoder, single-cell denoising),
  baseline numbers (Erdős value `0.380924→0.380876`)
- Qwen3 says "the source paper uses an entropic objective and PUCT-style search
  with LoRA fine-tuning" — correct but not actionable

The 30B model **can locate the relevant section** of source paper (it knows
PUCT exists) but **cannot quote the formula** or **distinguish source's adaptive
β(s) from a constant β=8** (a common mis-attribution in Qwen3 distillations).

## Implications for SDPO mechanism

The SDPO advantage on distillation tokens is `clamp(scale * (teacher_lp -
student_lp))`, where teacher sees the prior round's critique and student does not.
For SDPO to push the policy toward "absorb source paper specifics", the critique
must contain those specifics — which then make teacher's logprob on the next
distillation's correct-content tokens HIGHER than student's logprob on the same
tokens.

If critique is content-blind generic ("you should mention more about PUCT"), then
teacher's prompt-conditioned distribution differs from student only in surface
text alignment. SDPO gradient pushes the policy toward "produce text matching the
critique's surface style" — NOT toward "produce content matching source paper
ground truth". This is exactly F4 (critique-token-blindness) but worse, because
the critique itself is content-blind.

**The chain breaks at critique→content, not at student→teacher transfer.**

## Two interpretations(both important for paper)

1. **Capability gap**: 30B base model lacks the depth to extract precise technical
   details from a long technical paper. Larger models (235B Opus class) bridge
   this. → "Self-review needs ≥ 235B for SOTA-quality SDPO critique."

2. **Format gap (less likely)**: 30B was given the same prompt template designed
   for Opus. Maybe a different prompt template (e.g., "first list 5 specific
   formulas you find in the paper, then critique") would unlock comprehension.
   → "Self-review COULD work with prompt-engineering tailored to 30B reading
   patterns." Untested in this iteration.

We claim interpretation 1 in the paper (more conservative) and flag interpretation
2 as future work.

## What this rules in / out

**Rules out**:
- κ_self path (SDPO with Qwen3-30B as critic)
- "Pure open-source self-distillation" Phase 3 framing as primary claim

**Rules in** (Phase 3 contribution remains):
- τ_v4 inference-time distillation pipeline (no training, no critic in pipeline at
  inference time) — **Phase 3's actual contribution**
- κ_opus path (Opus critic in SDPO loop) — **conditional on τ_v4 BoN ablation**
  showing inference-time scaling has not saturated; otherwise κ_opus marginal value
  is low

## Open questions for future work

- N1: Would Qwen3-235B-as-critic close the gap with Opus? If yes, this
  significantly cheapens the pipeline (235B is open weights, no Opus dependency)
- N2: Multi-step critic (e.g., "first list 5 formulas, then critique gaps")
  prompt design for 30B
- N3: Critic ensemble (3 × 30B critique with majority voting on identified items) —
  does ensembling lift specificity?
- N4: Mixed-scale critic (30B for round 1 cheap critique, Opus for round 3
  high-stakes critique only) — saves cost while preserving SDPO signal quality

## Data pointers

- Smoke A run: `runs/2026_04_28_smoke_a_critique_corr/`
  - `qwen_critiques.jsonl` (8 Qwen3 critiques, 4180-6816 chars)
  - `opus_critic_responses/*.json` (8 Opus critiques, 2725-3865 chars)
  - `opus_judge_responses/*.json` (8 Opus-as-judge verdicts)
- Code: `smoke_a_critique_correlation_v1.py` (orchestrator)
- Self-critic prompt: `kappa_prompts_v1.py` `build_self_review_critic_prompt`
- Aggregation: 8/8 Opus preference computed via `opus_judge_requests/*.json` swap
  flag decoding (per pairwise corroboration discipline M4)
