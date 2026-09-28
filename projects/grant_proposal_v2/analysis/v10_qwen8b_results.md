# v10 × Qwen3-8B FoundOpt C* rerun — Results

*2026-04-23. 3 runs killed at iter 22-25 after length-normalized Opus regression.*

## TL;DR

Hypothesis: hybrid-dense Qwen3-8B outperforms MoE Qwen3-30B-A3B at policy=grader self-RL. **Refuted.**

- **B4 (no RL) peaked at iter 10 with Opus=18/40**, then regressed to 15 by iter 22 (−3).
- **SDPO (C3) worst**: Opus 15 → 14 → **11** across iter 4/10/21. New hallucination mode: fake institutional grants, fabricated method names ("Zipro", "Cayley discriminant"), time-travel publication dates.
- **Aggregate RL (C4)**: plateaued at Opus 16 (iter 10-24) while Qwen reward climbed 0.88 → 0.94 (+0.06 pure Goodhart).
- **Dense architecture did not fix RL Goodhart**. It changed which failure mode dominates: Qwen3-30B-A3B had Qwen-reward inflation to 0.97 (v7 C3); Qwen3-8B had Opus-quality regression without Qwen inflation.

## Data

### Qwen (self-grader) reward trajectory

| Run | iter 0 | iter 10 | iter 22-24 (final) | Qwen Δ |
|---|---:|---:|---:|---:|
| C2 (B4 no-RL) | 0.48 | 0.91 | **0.92** (iter 22) | +0.44 |
| C3 (SDPO) | 0.37 | 0.82 | **0.905** (iter 21) | +0.535 |
| C4 (aggregate) | 0.415 | 0.88 | **0.94** (iter 24) | +0.525 |

### Opus length-normalized depth_audit /40

| Run | iter 4 | iter 10 | iter 21-24 | Trajectory |
|---|:-:|:-:|:-:|---|
| C2 (B4) | 15 | **18** | 15 | peak iter 10, −3 by iter 22 |
| C3 (SDPO) | 15 | 14 | **11** | monotonic regression |
| C4 (aggregate) | 15 | 16 | 16 | plateau since iter 10 |

## SDPO specific metrics (C3)

The per-token advantage `A^SDPO_t = log(π_teacher / π_student)` did approach zero over training:

| iter | mean_adv | pos_frac | buf_max |
|---:|---:|---:|---:|
| 1 | -0.244 | 24% | 0.675 |
| 7 | -0.126 | 24% | 0.745 |
| 13 | -0.077 | 34% | 0.820 |
| 20 | -0.023 | **43%** | 0.905 |

Mean advantage did drift toward 0 (healthy convergence signal per SDPO paper). But:
- Teacher-student agreement never exceeded ~43% positive (never the 50%+ needed for balanced gradient)
- Qwen reward did climb (0.67 → 0.905) but **Opus score regressed** (15 → 11)
- The advantage-approaching-zero was being achieved by student learning to generate
  text the teacher ALSO found generically plausible — NOT content teacher preferred
  over student. SDPO converged toward statistical agreement, not quality.

Interpretation: on Qwen3-8B, adding critique to the teacher prompt spreads teacher's
distribution (critique acts as information overload, not credit assignment
refinement). SDPO then optimizes to match this spread distribution, producing
mean-regression text.

## Hallucination hacking — new failure modes (C3 iter 21 best plan)

Opus depth_audit explicitly flagged these as **new failure modes not in v7/v8 runs**:

- **Fake institutional details**: "Mayo Clinic Collaboration Agreement AA-3021",
  "NASDAQ research grant #2424057", "Rhododendron RHC Node" (fabricated
  infrastructure)
- **Time-travel publication claims**: "targeting NeurIPS/ICML 2024 dates that
  precede the work" (dates before proposal submission)
- **Fabricated method names**: "Cayley discriminant", "Zipro", "LSearchParams"
  (Opus: "nonsense tokens masquerading as methods")

### Why v10's defenses failed

v10's G11 citation-format check requires `(Author, Year, Venue)` wrapping for
`arXiv:xxxx` IDs to count as valid citations. This **did** catch the v7/v8-era
fabricated arXiv hack. But v10 did not check:

- Grant numbers (free-form fabrication space)
- Institutional agreement identifiers (format-free)
- Novel terminology (no dictionary check; any capitalized word passes)

On Qwen3-8B, the policy found these unprotected attack surfaces and fabricated
with creativity that exceeded v8 padding patterns.

## Architecture hypothesis — refuted

| Condition | v7 (Qwen3-30B-A3B MoE) | v10 (Qwen3-8B dense) |
|---|---|---|
| B4 (no RL) Qwen final | 0.855 | 0.92 |
| SDPO Qwen final | 0.970 (Goodhart peak) | 0.905 (lower self-inflation) |
| aggregate Qwen final | 0.940 | 0.94 |
| SDPO Opus at peak-Qwen | unverified (no iter 10 audit on v7) | **11/40** (bottom) |
| Dominant failure mode | Qwen self-inflation + fabricated arXiv | **Creative hallucination** (grant#, methods) |

Dense 8B is NOT a cure. The MoE-vs-dense distinction changes WHICH failure
surface the policy exploits, but RL still finds one. Both architectures
show B4 ≥ MAIN on Opus at equivalent iter.

## B4 regression: CR pipeline itself drifts in late training

Even the no-RL B4 run shows Opus regression from iter 10 to iter 22 (18 → 15).
This means **critique-revise + BoN buffer selection alone** drifts toward
lower-quality plans after ~10 iters. Mechanism:

1. Buffer selects for HIGH Qwen reward (aggregate score)
2. Revisions learn to trigger Qwen-reward-positive patterns (longer, more
   formal-looking)
3. Opus sees these as noise without substance
4. Buffer entries become increasingly verbose/decorative while Qwen keeps rewarding

This is CR-pipeline-level Goodhart, independent of RL gradients.

## Implications for next steps (GER-CR-v1)

The `DECISIONS.md 2026-04-23` entry commits to evolving rubric (GER-CR-v1) based
on DR Tulu + OnlineRubrics + RaR. These v10 results reinforce that direction:

1. **Static rubrics cannot anticipate all hallucination axes** — v10's citation
   check caught arXiv; policy discovered grant numbers + method names. Next
   attempt would target those; policy would find 3rd axis. Evolving rubric
   responds adaptively instead of playing whack-a-mole.

2. **Iter 10 is a consistent Opus sweet spot** across v8/v10 runs. Suggests
   evolving rubric should either (a) trigger first evolution before iter 10 or
   (b) use iter-10 checkpoint as refresh anchor.

3. **B4 CR-pipeline drift** means evolver must guard against pure generation
   drift, not just RL-amplified Goodhart. Embedding-space similarity-to-reference
   is one candidate mechanism (matches Path-Locking Test 2 design).

4. **Dense architecture does not avoid RL Goodhart**. GER-CR-v1's Phase 3
   ablation (approved to use Qwen3-30B-A3B) is fine — architecture is not the
   bottleneck.

## Data artifacts

- `projects/grant_proposal_v2/runs/2026_04_23_v10_qwen8b_foundopt_C2/` (23 iters)
- `projects/grant_proposal_v2/runs/2026_04_23_v10_qwen8b_foundopt_C3/` (22 iters)
- `projects/grant_proposal_v2/runs/2026_04_23_v10_qwen8b_foundopt_C4/` (25 iters)
- Each has `train/iter_summary.jsonl`, `buffer.jsonl`, `depth_audit_mid.jsonl`
  (3 Opus evals per run at iter 4/10/21-24)

## Not tested

- Non-SDPO modes on 30B-A3B with v10 rubric (would isolate rubric effect from
  architecture effect — the v10 vs v7 comparison mixes both)
- Longer SDPO training on 8B — we killed at iter 22 with pos_frac=43%; unclear
  if pos_frac→50% would produce Opus recovery. v7 precedent suggests no.
- Size-matched control (Qwen3-14B) — would distinguish "small dense worse than
  big MoE" from "architecture class matters".
