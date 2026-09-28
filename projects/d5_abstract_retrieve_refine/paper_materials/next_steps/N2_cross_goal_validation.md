# N2 — Cross-goal validation: does μ-v4 transfer?

## Motivation

Phase 2 trained μ-v4 on a single goal (TTT-Discover; arxiv:2601.16175). For paper
publication, we need cross-goal evidence:

1. **Production use case**: D5 should generate plans for arbitrary research goals,
   not just TTT-Discover-style. Does iter-4 weights transfer to other goals?
2. **Scientific claim**: "the SDPO+critic mechanism teaches transferable abstraction"
   requires testing on goals the model didn't train on.

## Candidate goals (post-2026 follow-up papers)

Identified during Phase 1 as candidates with similar setup to TTT-Discover:

| Paper | arxiv | Setup | Why a good test |
|---|---|---|---|
| TTRL | 2502.10517 | test-time RL on single problem | Closest sibling; tests "same family" transfer |
| MiGrATe | 2503.NNNNN | test-time RL with gradient on inputs | Tests gradient-input variation |
| ThetaEvolve | 2504.20571 | evolution-based test-time | Tests non-RL variant |
| 1-shot RLVR | 2504.05108 | one-shot RL with verifiable rewards | Tests verifier-driven RL |
| TTC-RL | 2503.NNNNN | test-time compute scaling RL | Tests scaling-direction variation |
| LatentSeek | 2505.13308 | latent-space search at test-time | Tests representation-axis variation |

(Some arxiv IDs are placeholders — fill in when the paper is identified for use.)

## Experimental design

For each candidate goal:

1. **Extract goal text** from paper's intro / related-work to construct a goal
   prompt (mimicking how TTT-Discover's `research_goal.txt` was extracted)
2. **Retrieve goal-specific oracle**: rerun oracle build pipeline with new
   bibliography (post-cutoff papers) and new source paper (the candidate paper itself)
3. **Compare 4 conditions** (8 plans each):
   - ξ' (frozen 30B + goal only)
   - σ' (frozen 30B + new oracle)
   - μ-v4-iter-4 weights + new oracle (zero-shot transfer)
   - μ' (μ-v4-iter-4 weights, additional 5 SDPO iters on new goal)
4. **9-dim isolated audit**: same methodology as Phase 2F

## Hypotheses to test

- **H1**: μ-v4 weights add +X over σ' on new goal (zero-shot transfer)
- **H2**: Additional 5 SDPO iters on new goal lifts further (in-distribution
  fine-tuning), with same iter-4 peak pattern
- **H3**: σ' lift over ξ' is consistent across goals (≈ +9 like TTT-Discover)
- **H4**: D5 architecture's value is the abstraction-mechanism, transferable; or
  alternatively, training is goal-specific

## Cost estimate

Per goal:
- Bibliography rebuild: ~$5 in S2/OpenAlex (free) + Opus oracle build ~$10
- σ' baseline: ~$0 (frozen Qwen3-30B, ~10 min)
- μ-v4 zero-shot: ~$0
- μ' SDPO 5 iters: ~$15 in Opus critic+audit
- 9-dim isolated audit ×4 conditions: ~$2

Total: ~$30 per goal × 3 goals = ~$90. Wall: ~1 day per goal.

## Decision deferred to a separate plan after Phase 3 retrieve-then-generate.

If Phase 3 produces strong σ' / μ', then N2 cross-goal validation is the natural
follow-up to demonstrate generalization.
