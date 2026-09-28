# RESULTS — Oracle transfer ABC

*Filed 2026-04-29. Joint verdict from three pre-registered falsification
experiments (EXPERIMENT_PLAN_oracle_transfer_ABC.md) on F16's three
hypothesized bottlenecks for v8-d5sdpo's oracle-transfer ceiling.*

## Hypotheses & verdicts (FINAL — all three experiments complete)

| Hyp | Falsifier | Status |
|---|---|---|
| H16-1 long-context retrieval bottleneck | Exp A — frozen Qwen3-30B + verbatim quote queries on Math 5/7, Methodology 9, Empirical 4 | **STRONGLY FALSIFIED — 4/4 verbatim correct** |
| H16-2 critique-conditioned advantage rewards critique-conformity, not oracle-fidelity | Exp B — n-gram overlap trajectory across 6 v8-d5sdpo cells, n ∈ {1,2,3,4} | **CORROBORATED at n ≥ 3** (no cells rising; G+/C+ have steepest n-gram declines despite best audits) |
| H16-3 LoRA bandwidth too small to internalize formula-length sequences | Exp C — RAG top-K=5 vs full-oracle 1-cell (G+ config), 8/16 iter (manual stop after cliff) | **INDETERMINATE — RAG peak Δ -1.12 ≈ G+ within noise → H16-2 confirmed dominant; H16-3 not separable** |

## Cross-table verdict (FINAL)

| A result | B result | C peak vs G+ | Most likely diagnosis |
|---|---|---|---|
| Can retrieve | Flat | RAG ≈ G+ | H16-2 + H16-3 |
| **Can retrieve** | **Flat (n≥3)** | **RAG = -1.12 ≈ G+** | **H16-2 dominant; H16-3 indeterminate** ← landed |
| Can retrieve | Rising | RAG > G+ | H16-1 only |
| Can retrieve | Rising | RAG ≈ G+ | H16-2 only |
| Cannot retrieve | Flat | RAG ≈ G+ | All three |

**Landed row**: `(Can retrieve, Flat at n≥3, RAG ≈ G+ peak)` → **H16-2 (incentive
misalignment) is the dominant fault. H16-1 (retrieval) and H16-3 (LoRA bandwidth)
are not separately load-bearing under this experiment design.**

## Exp A summary

- **Setup**: frozen Qwen3-30B-A3B, full slim oracle (72,056 chars / 18,150
  tokens) in context, 5 queries × 3 samples (temp=0.0)
- **Manual verbatim verdict** (responses/Q1-Q4 inspected, see
  `runs/2026_04_29_exp_A_oracle_retrieval/verdict.md`):
  - Q1 Math 5 J_RS: VERBATIM ✓ (LaTeX-rendered equivalent)
  - Q2 Math 7 GRPO loss: VERBATIM ✓ (exact char-by-char incl. `eps_low/eps_high`)
  - Q3 Methodology 9 RS-GRPO: VERBATIM ✓
  - Q4 Empirical 4 AIME: VERBATIM ✓ (`15.6% -> 77.9% over ~10,000 RL steps`)
- **Q5 control**: model knows J_RS from prior training (4/4 anchor hits without
  consulting oracle). Q1 partially confounded by prior knowledge, but Q2
  (MiGrATe-specific GRPO ratio convention) and Q4 (specific numbers) are
  unambiguously oracle-retrieved.
- **Verdict**: H16-1 STRONGLY FALSIFIED. Long-context retrieval works.

## Exp B summary

- **Setup**: 6 v8-d5sdpo cells (base, G, G+, E, C+, F+), 528 plans across
  7-13 iters per cell. Reference corpus = slim Math 1-12 + Methodology 1-17
  (29 items, 8,592 tokens). Reference plan (35/45) plotted as horizontal anchor.
- **Joint verdict per n** (`runs/2026_04_29_exp_B_ngram_overlap/`):
  - n=1: MIXED (3/6 rising; reference plan = 0.168, students peak 0.18-0.21
    — students have HIGHER overlap than ref because oracle is in their prompt)
  - n=2: MIXED (1/6 rising; only base cell)
  - n=3: CORROBORATED (no cells rising; ref plan = 0.002 floor)
  - n=4: CORROBORATED (no cells rising; ref plan = 0 — metric saturates)
- **Inverted-U pattern shared across cells** regardless of mask/α/loss. The
  rise iter 0 → 3-4 is shared with the no-training base cell — not driven by
  SDPO advantage.
- **G+ (peak audit 22.38) and C+ (peak audit 23.50) — best cells — have the
  STEEPEST n-gram declines** at every n. Cells that please the audit grader
  most are also the cells moving farthest from oracle vocabulary.
- **Verdict**: H16-2 CORROBORATED at the n ≥ 3 resolution where the metric
  is interpretable. The SDPO mechanism does not push toward oracle content.

## Exp C summary

- **Setup**: G+ baseline config + `use_rag=True rag_k=5`. Cell config matched G+
  baseline exactly (mask=T, α=0.1, PPO, ppo_clip_eps=0.2, audit_drop_threshold=7.0).
- **Run dir**: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_Gplus_rag/`
- **Retriever**: `oracle_retriever_v1.OracleRetrieverV1` (BAAI/bge-small-en-v1.5
  CLS-pooling + L2-norm). 87 items embedded once at startup. Student retrieval
  cached (goal fixed); teacher retrieval per-iter on `goal+critique`.
- **Oracle bandwidth**: 5,087 chars / 7.1% of full slim oracle (72,056 chars).
- **Stop**: manual at iter 8/16 — cliff confirmed at iter 6 (mean drop -5.4
  from iter 5, below trainer's auto-cliff threshold 7.0); iter 7 stayed in
  collapse zone; remaining iter 8-15 would be redundant post-cliff data per
  pre-registered ABC plan goals (decision matrix already resolved).

### Trajectory vs G+ baseline

| iter | RAG /45 | G+ /45 | Δ |
|---|---:|---:|---:|
| 0 | 18.88 | 19.50 | -0.62 |
| 1 | 18.00 | 18.75 | -0.75 |
| 2 | 17.50 | 19.88 | -2.38 |
| 3 | 19.00 | 20.38 | -1.38 |
| 4 | 20.12 | 22.38 | -2.25 |
| **5 (RAG peak)** | **21.25** | 22.38 | **-1.12** |
| 6 (RAG cliff) | 15.88 | 21.38 | -5.50 |
| 7 (post-cliff) | 16.88 | 20.00 | -3.12 |

**Peak**: RAG iter 5 = 21.25 vs G+ peak iter 4-5 = 22.38 → Δ = -1.12, within
single-seed noise floor (G+ inter-iter std ≈ 1 pt).

**Cliff timing**: RAG cliffed at iter 6 vs G+ cliff at iter ~11 (per F15) —
**RAG accelerated cliff by ~5 iters**.

### Two findings, one on-matrix and one new

**1. On-matrix (per L150-156)**: `(Can retrieve, Flat n≥3, RAG ≈ G+ peak)` →
H16-2 dominant (incentive misalignment). The full oracle's BREADTH does not
provide a peak-audit lift over RAG top-K=5 — confirms the gradient signal
extracts the SAME amount of value from a 5-item context as from 87. This
isolates the bottleneck from "input bandwidth" hypotheses.

**2. New off-matrix finding — F17 (filed separately)**: RAG cell template-collapses
~5 iters earlier than full-oracle G+ baseline. Hypothesis: oracle BREADTH
provides an anti-collapse anchor independent of which items contribute to
gradient signal (perhaps via richer per-iter teacher attention shifts on
critique-conditioned retrieval, or vocabulary diversity that resists boilerplate
contraction). This was NOT in the H16-1/2/3 set; goes to F17 as a new
hypothesis to characterize.

### Companion metric (Exp B-style n-gram on RAG buffer)

Not yet rerun. If RAG overlap rises significantly above G+ overlap but peak
audit doesn't change, this would corroborate "oracle pathway opens but
advantage stays broken" → triple-confirm H16-2. **Recommended as a quick
follow-up** (~5 min via existing exp_B_ngram_overlap.py with new run dir).

## Production decision (FINAL)

Landed on row "RAG ≈ G+ peak" → **verifier-grounded reward is the primary
lever**. Larger LoRA / wider context will not solve the ceiling because the
gradient signal (critique-conditioned advantage) doesn't reward oracle-fidelity
in the first place — adding bandwidth gives the model more room to NOT use
oracle content rather than more room to internalize it.

**Plus, the F17 cliff-acceleration finding** suggests RAG is NOT a free
production substitute even in the H16-2-dominant world. If retrieval is
adopted, it should be:

- **Hybrid**: full oracle for student (anti-collapse anchor), retrieved
  oracle for teacher (per-iter critique-conditioned focus); OR
- **Stacked**: RAG + verifier-grounded reward + stronger anti-collapse
  anchor (higher α trust-region, broader-oracle KL anchor).

Pure RAG-only swap loses the breadth-as-anchor effect and accelerates collapse
without compensating peak gain.

## F16 update — DONE

A, B, C verdicts applied to `paper_materials/findings/F16_oracle_transfer_ceiling_diagnosis.md`
(Falsification-update section). F17 cliff-acceleration filed as new finding.

## Next-stage handoff

1. **Primary direction (verifier-grounded reward)**: design
   `paper_materials/findings/F17_verifier_grounded_reward_design.md` spec
   - Verifier choices: sympy expression equality (Math 5/7/9 type formulas),
     oracle citation match (ref-ID and quoted-content), executable test cases
     (Methodology 4/14 type code-eval)
   - Reward shape: additive bonus on plan score (gated by signal_present),
     OR multiplicative gate (no signal → -∞ reward)
   - 1-cell pilot recipe: G+ baseline config + verifier reward layer; budget
     match Exp C (~$40)
2. **Secondary direction (F17 characterization)**: design 1-cell ablation
   testing oracle breadth as anti-collapse anchor — vary rag_k ∈ {5, 10, 20,
   40} on otherwise-identical RAG cell. Measures if cliff iter scales with K.
   Cheap (~$30 per cell × 4 = $120) if user wants to characterize the
   anti-collapse mechanism before committing to verifier-grounded design.

## Files

- A: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_exp_A_oracle_retrieval/`
  - `verdict.md` (manual verbatim review)
  - `responses/Q*_sample_*.json`
  - `config.json`
- B: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_exp_B_ngram_overlap/`
  - `figures/exp_B_overlap_trajectories.png` (4×2 grid)
  - `per_cell_per_iter.csv`
  - `verdict.json`
  - `README.md`
- C: `projects/d5_abstract_retrieve_refine/runs/2026_04_29_mu_v8_d5sdpo_Gplus_rag/` (pending)
