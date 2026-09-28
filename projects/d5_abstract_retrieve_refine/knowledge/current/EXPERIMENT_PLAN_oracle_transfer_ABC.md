# Experiment plan — Oracle transfer ABC

*Filed 2026-04-28. Three falsification experiments to diagnose why μ-v8-d5sdpo
critique-conditioned self-distillation does not transfer oracle slim's
formula/number content into student outputs (F16 hypothesis).*

## Why these three experiments

F16 names three hypothesized bottlenecks:
- H16-1: long-context retrieval bottleneck (Qwen3-30B can't reliably copy from 10K oracle)
- H16-2: critique-conditioned advantage rewards critique-conformity, not oracle-fidelity
- H16-3: LoRA bandwidth insufficient for formula-length internalization

ABC are the minimum experiments that distinguish these three hypotheses. Each
costs <$30 and runs in <1 day, so all three together fit one session.

| Hyp | Falsifying experiment | If hyp true → outcome | If hyp false → outcome |
|---|---|---|---|
| H16-1 | Exp A: prompt frozen Qwen3-30B with oracle + verbatim-quote instruction | Cannot quote Math 5 verbatim | Quotes correctly → bottleneck is downstream |
| H16-2 | Exp B: n-gram overlap student-output vs oracle math/methodology, by iter | Overlap flat/decreasing across iters | Overlap rises monotonically → distill DOES push toward oracle |
| H16-3 | Exp C: 1-cell smoke replacing full oracle with RAG top-K oracle items | RAG ≈ full-oracle → bandwidth/incentive is bottleneck | RAG > full-oracle → attention is bottleneck |

The three are complementary. A and B are zero-training observations on existing
data + one frozen-inference call. C requires a 1-cell training run (~$30 + 6 hr).

## Experiment A — Long-context retrieval probe

### Question
Can Qwen3-30B at full oracle context retrieve a specific formula verbatim when
explicitly instructed?

### Setup
- Backbone: Qwen3-30B-A3B (frozen, no LoRA), thinking=False
- Prompt: full oracle slim (10,265 words) + query
- Query 1: "Quote verbatim the formula labeled 'Math 5' from the oracle, including
  source attribution."
- Query 2: "Quote verbatim the formula labeled 'Math 7' from the oracle (GRPO
  loss with clipped policy ratio)."
- Query 3: "Quote verbatim the formula labeled 'Methodology 9' from the oracle
  (RS-GRPO advantage)."
- Query 4: "Quote the AIME 2024 pass@1 trajectory numbers from Empirical 4
  (initial → final percentages and step count)."
- Query 5: "Without consulting the oracle, write the J_RS formula from
  risk-sensitive RL." (control — what does Qwen3-30B know baseline-without-oracle)
- Sampling: temperature 0.0, max_tokens 1024, n=3 samples per query (consistency
  check)

### Implementation
New script: `src/co_scientist/d5_abstract_retrieve_refine/exp_A_oracle_retrieval.py`
- chz config with oracle_path, query_set
- Tinker sampling client with frozen base
- Write `runs/2026_04_29_exp_A_oracle_retrieval/responses/query_N_sample_K.json`

### Expected runtime / cost
~$2-3 (15 inference calls × 30s × frozen sampling). Wall <30 min.

### Falsification rule (pre-registered)
- 0/4 verbatim-correct → H16-1 strongly corroborated (retrieval bottleneck)
- 1-2/4 → ambiguous, needs follow-up (likely retrieval is partial bottleneck)
- 3-4/4 → H16-1 falsified; bottleneck is downstream (H16-2 or H16-3)

"Verbatim-correct" = formula RHS character-by-character match modulo whitespace
and equivalent unicode/LaTeX rendering, AND correct source attribution
(paper / ref number).

### Why this is decisive
If frozen Qwen3-30B can retrieve when explicitly asked, then in our training
setting the issue is not "can't copy" but "doesn't know it should copy" → H16-2
or H16-3. If it can't retrieve even when explicitly asked, the formulas can
NEVER reach student output regardless of what the gradient signal does.

## Experiment B — n-gram overlap trajectory

### Question
Does student output n-gram overlap with oracle math/methodology sections
increase monotonically across training iters?

### Setup
- Inputs: existing buffer.jsonl files from `runs/2026_04_29_mu_v8_d5sdpo_<cell>/`
  for cells where we have iter 0 + peak + cliff data: G+ (13 iters), C+ (12),
  E (10), F+ (10), G (10), base (7)
- Reference corpus: oracle slim Math sections (Math 1-12) + Methodology sections
  (Methodology 1-17), concatenated
- Metric: 4-gram Jaccard overlap between each plan's `<solution>` content and
  the reference corpus; n-gram tokenization on whitespace-split words after
  lowercasing
- Per iter: report mean ± SE across 8 plans. Plot 6 trajectories on one chart.

### Implementation
New script: `src/co_scientist/d5_abstract_retrieve_refine/exp_B_ngram_overlap.py`
- chz config with run_dirs list, reference_path
- Read each `buffer.jsonl`, extract `<solution>` per plan
- Compute 4-gram Jaccard vs reference corpus
- Write `runs/2026_04_29_exp_B_ngram_overlap/per_cell_per_iter.csv`
- Generate matplotlib figure `figures/exp_B_overlap_trajectories.png`

### Expected runtime / cost
~$0 (pure offline analysis on existing data). Wall <15 min.

### Falsification rule (pre-registered)
- All 6 cells show flat or decreasing overlap across iters → H16-2 strongly
  corroborated (distillation does NOT push toward oracle)
- Mixed pattern (some up, some down) → ambiguous; report case-by-case
- All 6 cells show monotonic increase → H16-2 falsified; the oracle pathway
  IS active and the failure is in retrieval/bandwidth (H16-1/H16-3)

### Secondary metric (companion, free)
Per-iter unique 4-gram count and total 4-gram count. If unique/total decreases,
plans are becoming more repetitive (template collapse). If both rise, plans are
diversifying. This contextualizes the overlap result.

### Why this is decisive
The oracle is a 10K-word fixed text. Student plans are ~800-1100 words. If
distillation is teaching the model to use oracle, n-grams should drift from
unconditional baseline (~iter 0) toward oracle vocabulary as training progresses.
A flat curve directly falsifies "oracle is being learned".

## Experiment C — RAG vs full-oracle 1-cell smoke

### Question
Does replacing full oracle (10K words) with embedding-similarity-retrieved
top-K oracle items (~500-1000 words) change peak audit / cliff iter / output
n-gram overlap with oracle?

### Setup
- New prompt builder: `mu_prompts_v8_rag.py` with `build_student_prompt_v8_rag`,
  `build_teacher_prompt_v8_rag` that take `oracle_retrieved` (top-K items) instead
  of full oracle
- Retrieval: per (goal + critique_xml) embed via `text-embedding-3-large`-equivalent
  (or local SBERT model for free). Score each oracle item by cosine similarity
  to (goal + last critique). Take top K=5 items (~500 words combined).
- Trainer cell: clone of G+ config (mask=T, α=0.1, PPO, 16 iter,
  audit_drop_threshold=7.0) but with RAG prompts. Run dir
  `runs/2026_04_29_mu_v8_d5sdpo_Gplus_rag/`
- Compare to G+ baseline (already complete)

### Implementation
New script:
- `src/co_scientist/d5_abstract_retrieve_refine/mu_prompts_v8_rag.py`
- `src/co_scientist/d5_abstract_retrieve_refine/oracle_retriever_v1.py` (sbert
  embed + cosine top-K)
- New trainer entrypoint or chz override on `train_mu_v8_d5sdpo.py` to swap
  prompt builders (cleanest: add `config.use_rag: bool = False` flag and switch
  in `build_student_prompt_v8` / `build_teacher_prompt_v8` callsites)

### Expected runtime / cost
~$30 trainer + ~$10 audit + ~$0 retrieval (local SBERT) = ~$40
Wall: ~6 hr (16 iter @ ~22 min/iter)

### Decision matrix (pre-registered)
| RAG peak vs G+ peak | RAG cliff iter vs G+ cliff iter | Verdict |
|---|---|---|
| RAG peak ≥ G+ peak by ≥1 audit pt | similar or later cliff | H16-1 corroborated (attention bandwidth was bottleneck); RAG is the production direction |
| RAG peak ≈ G+ peak (Δ < 1) | similar cliff | H16-1 falsified; bottleneck is H16-2 (incentive) or H16-3 (LoRA bandwidth) |
| RAG peak < G+ peak by ≥1 | earlier cliff | full oracle is providing some retrieval signal even if imperfect; RAG-only too narrow |

Companion metrics: rerun Experiment B's 4-gram overlap on the RAG run. If
overlap is significantly higher (RAG > G+) but peak audit doesn't change, the
oracle pathway is opening but the gradient signal still doesn't ground formula
learning → strongly suggests H16-2 + H16-3 dominate.

### Why this is decisive
Three orthogonal outcomes (peak / cliff / overlap) on one experiment, each tied
to a specific hypothesis. Combined with Experiments A and B, the joint outcome
of all three uniquely identifies which 1-2 of {H16-1, H16-2, H16-3} are the
dominant bottlenecks.

## Total cost / wall

| Exp | Cost | Wall | Type |
|---|---:|---:|---|
| A | ~$3 | ~30 min | Inference probe |
| B | $0 | ~15 min | Offline analysis |
| C | ~$40 | ~6 hr | 1-cell training |
| **Total** | **~$43** | **~7 hr** | Single session |

## Pre-registered combined verdict matrix

| A result | B result | C peak vs G+ | Most likely diagnosis |
|---|---|---|---|
| Can retrieve | Flat | RAG ≈ G+ | H16-2 (incentive) AND H16-3 (LoRA bandwidth) |
| Can retrieve | Rising | RAG > G+ | H16-1 (attention bandwidth) only |
| Can retrieve | Rising | RAG ≈ G+ | H16-2 only (oracle reachable, distill pulls toward it, but advantage isn't grounded enough to bake formulas in) |
| Cannot retrieve | Flat | RAG ≈ G+ | H16-1 + H16-2 + H16-3 (oracle never reaches output regardless) — paper-publishable strong negative |
| Cannot retrieve | Rising | * | Contradiction — investigate B's metric specification |

## Out-of-scope reminders

- ❌ Not running multi-seed (same single-seed=42 limitation as Phase 2)
- ❌ Not changing critic source (still Opus subagent file-bus)
- ❌ Not changing reward / verifier — this experiment plan diagnoses the
  current architecture's ceiling; verifier-grounded reward (sympy, executor)
  is a separate next-stage plan
- ✅ Just: 3 minimum-cost falsifiers for the 3 hypothesized bottlenecks

## Verification before launching

1. Confirm oracle slim path resolves: `ls projects/d5_abstract_retrieve_refine/data/oracles/oracle_v2_2026_04_26_build/slim.md`
2. Confirm at least 6 cells have buffer.jsonl: `for c in base G Gplus E Cplus Fplus; do ls runs/2026_04_29_mu_v8_d5sdpo_$c/buffer.jsonl; done`
3. Confirm Tinker API access: `source shared/tools/use_api_profile.sh new && python -c "from co_scientist.shared.api_profiles import create_service_client; print('OK')"`
4. Confirm SBERT availability for offline embedding (or fall back to a tiny
   instruction-tuned model on the same Tinker base)

## Risks

1. **Exp A might pass but be misleading**: if Qwen3-30B cites oracle correctly
   when explicitly asked, that doesn't prove it can do so during natural plan
   generation. Mitigation: add an "implicit retrieval" query like "Write a
   research plan that uses the J_RS formulation from the oracle"; compare
   verbatim-formula reproduction rate to explicit-quote rate
2. **Exp B might be confounded by template collapse**: cliff iters produce
   word-salad output (per E iter 9 KILLED_README); 4-gram overlap with oracle
   collapses to ~0 mechanically. Mitigation: report overlap pre-cliff only
   (iter 0 to peak); cliff data is separate
3. **Exp C RAG retriever quality**: SBERT cosine on raw oracle items may be
   too noisy. Mitigation: include ablation with manual top-K selection
   (researcher reads goal + picks 5 most relevant items) as a "ceiling" RAG
4. **Cost overrun**: if Exp C cliffs early (which is fine — RAG might destabilize
   training in unforeseen ways), still write up the early-stop run; report the
   trajectory's first 3-4 useful iters
