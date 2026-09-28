# N1 — Phase 3: retrieve-then-generate

## Motivation

Phase 2 used a STATIC slim oracle (Opus-extracted from full bibliography, ~4096 tokens)
as the inference-time scaffold for σ baseline and as the prompt input for μ-v4 SDPO
training. Phase 3 replaces this with DYNAMIC per-section retrieval over the same
86-paper bibliography.

## Key questions

1. **Does dynamic retrieval beat static slim oracle?** — measured as σ' (frozen + retrieved
   chunks per section) vs σ (frozen + slim oracle). Hypothesis: slightly worse (noisier
   retrieval signal) but more scalable across goals.

2. **Does per-section retrieval enable plan structure that static oracle prevents?** —
   E.g., can retrieved Methodology-section chunks for the Methodology section give
   the model targeted content the slim oracle's mixed-section format couldn't?

3. **Does SDPO recipe v1 transfer to retrieval-augmented training?** — Same lr=5e-5,
   same 4 grad steps, same iter-4 peak expectation? Or does retrieval noise change
   the dynamics?

4. **Can we close the σ' → ε gap (current 8.37 points)?** — If retrieval gives the model
   bibliography-grounded content beyond what slim oracle has, μ' (trained + retrieval)
   might exceed μ-v4's 28.00.

## Architecture sketch

```
research goal ──┐
                ├──> retrieve top-K chunks per section
bibliography ───┤    (sections: Insights / Methodology / Math / Empirical / etc.)
                │
                ▼
        ┌────────────────────┐
        │ retrieved chunks   │
        └────────┬───────────┘
                 ▼
        ┌────────────────────┐
        │ student prompt =   │
        │ goal + retrieved   │ ← this is σ' baseline
        │ chunks (no critic) │
        └────────────────────┘
                 │
                 ▼ (training: add critique slot for teacher)
        ┌────────────────────┐
        │ teacher prompt =   │ ← this is μ' training
        │ goal + retrieved + │
        │ critique           │
        └────────────────────┘
```

## Open design choices

- **Retrieval scope per section**: 3 chunks per section (12 total) vs 5 chunks
  (20 total) vs adaptive
- **Retrieval method**: dense embedding similarity vs BM25 vs hybrid
- **Embedding model**: open-weight (e.g. all-MiniLM, BGE) vs Opus-generated
  query-time embeddings
- **Section-categorization**: use the audit_v3 9-dim categories as section types?
  Or simpler 3-section structure (Methodology / Empirical / Other)?

## Reuse from Phase 2

- **SDPO recipe v1** (`knowledge/current/SDPO_RECIPE_v1.md`) — same hyperparameters
- **9-dim isolated audit** (`paper_materials/methodology/M4_audit_isolated_methodology.md`)
  — same eval methodology
- **σ baseline as the comparison anchor** — must show σ' vs σ explicitly to
  prove retrieval doesn't regress
- **Best-of-early-iter stop criterion** — likely peak at iter 3-5 again

## Estimated effort

- Implementation: 1-2 weeks (retrieval index + per-section query + prompt assembly)
- Experiments: ~3 days for σ' baseline + 1 SDPO run (μ')
- Evaluation: ~1 day (audit_v3 isolated methodology)

## Decision deferred to a separate plan after Phase 2 documentation pass.
