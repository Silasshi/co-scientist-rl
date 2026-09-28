# D5 File Layout Schema

*Single source of truth for "where does this file go?" in D5. Read together with `CONVENTIONS.md` (which governs **naming**; this file governs **placement**). Any file produced by D5 work — by hand or by an agent — MUST land in a slot defined here. If a new file type does not fit any slot, propose an addition to this schema before creating a new top-level directory.*

**Last updated**: 2026-04-26 (Phase 0a). Supersedes the flat `data/` layout that accumulated through the smoke / mu / pairwise pre-realignment work.

---

## 0. Design principles

1. **Group by file lifecycle, not by file extension.** A `.jsonl` may be a derived dataset, a run output, an analysis intermediate, or a spec — these belong in different slots.
2. **Two-level minimum where one source produces multiple artifacts.** If one logical thing (e.g. *bibliography*) produces ≥ 3 files, those files share their own subdirectory.
3. **Versioning lives on filenames, not directory names.** Exception: dated multi-file builds (e.g. `oracle_v2_2026_04_26_build/`) where the directory groups one coherent build's intermediates.
4. **`_archive/` is the only legal way to retire content.** Never delete; never rename in place.
5. **One canonical name per concept.** See `CONVENTIONS.md` § Canonical Spellings for prose↔code mapping.
6. **Determinism**: for any plausible new file an agent might create, the decision tree in § 3 must terminate at a unique slot.

---

## 1. Top-level tree

```
projects/d5_abstract_retrieve_refine/
├── PROJECT_OVERVIEW.md        Executive summary — read FIRST (added 2026-04-27)
├── README.md                  Project overview (slow-changing)
├── STATUS.md                  Live state — read second
├── DECISIONS.md               Append-only decision log
├── CONVENTIONS.md             Naming rules
├── SCHEMA.md                  THIS FILE — placement rules
│
├── dataset/                   ► Canonical, hand-curated task inputs (read-only after creation)
│   ├── research_goal.txt
│   ├── reference_solution.txt
│   ├── perturbations/         18 perturbation variants of reference solution
│   └── held_out/              (Phase 4) per-goal subdirs for cross-goal eval
│
├── data/                      ► Derived data assets (rebuildable from src/ + external APIs)
│   ├── source_paper/          Source paper full text — TTT-Discover
│   ├── bibliography/          Bibliography parsing pipeline outputs
│   │   ├── full_text/         Per-cited-paper full text (.md per arxiv id, PDF-extracted)
│   │   ├── full_text_latex/   Per-cited-paper full text (.md per arxiv id, LaTeX-extracted, added 2026-04-27)
│   │   └── faiss_index/       (Phase 1) FAISS embeddings for retrieval
│   ├── oracles/               Oracle abstractions — ceiling estimates
│   │   ├── smoke/             Hand-coded smoke v1/v2/v3 oracles
│   │   └── <build_name>/      Multi-round Opus build outputs (intermediate + final)
│   ├── plans/                 External-LLM plan generations (distillation/SFT pools)
│   └── _archive/              Retired data assets
│
├── configs/                   ► YAML training/eval configs (one config = one launch profile)
│
├── specs/                     ► Markdown specifications (NOT executable code)
│   ├── subagents/             Subagent prompt templates (Opus oracle, critic-audit daemon)
│   └── runs/                  Pre-launch run specs (filled-out RUN_CONFIRMATION_TEMPLATE)
│
├── scripts/                   ► Ad-hoc utilities (.py / .sh) — housekeeping, migrations
│
├── runs/                      ► Training and evaluation outputs (one dir per run)
│   ├── _archive_pre_realignment_2026_04_25/   Pre-realignment runs (frozen, do NOT cite)
│   └── <YYYY_MM_DD>_<intent>/                 Active runs — see § 2.6 for inner schema
│
├── analysis/                  ► Offline cross-run analysis
│   └── <purpose>/             One subdir per analysis project
│       ├── scripts/           Analysis scripts (.py)
│       ├── data/              Intermediate jsonl/csv/json
│       └── reports/           Final markdown / ipynb / png
│
├── knowledge/                 ► Long-form design docs
│   ├── current/               Active, authoritative
│   └── archive/               Superseded
│
├── paper_materials/           ► Paper-writing canonical store (added 2026-04-27, see § 2.13)
│   ├── findings/              Substantive finding write-ups (FN_<topic>.md)
│   ├── experiments/           Raw data dumps per finding (EN_<short>.{json,md})
│   ├── figures_data/          matplotlib-ready CSVs (GN_<short>.csv)
│   ├── plan_samples/          Exemplar plans with provenance headers
│   ├── methodology/           Self-contained method docs (M1-M5; copies of knowledge/current/ ok)
│   ├── related_work_excerpts/ Verbatim quotes from cited papers
│   └── next_steps/            Open questions / future-work skeletons (NN_<topic>.md)
│
└── paper/                     ► (Phase 5) ICLR 2027 LaTeX (does not exist yet)
```

Code lives outside this tree:
- D5 production code: `src/co_scientist/d5_abstract_retrieve_refine/*.py`
- Cross-direction shared modules: `src/co_scientist/shared/*.py`
- Unit tests: `tests/d5/*.py` (under repo root)

---

## 2. Slot specifications

### 2.1 `dataset/`

**Purpose**: human-curated, frozen-once-created inputs that define the task.

**Goes here**:
- `research_goal.txt` — TTT-Discover Goel-style research goal.
- `reference_solution.txt` — Gold reference plan.
- `perturbations/` — 18 perturbation variants. Naming `P_<dim>_<short>.txt` or `M<NN>_<intent>.txt`.
- (Phase 4) `held_out/<goal_name>/{research_goal,reference_solution}.txt` — held-out goals for cross-goal eval.

**Does NOT go here**: LLM-generated plans (→ `data/plans/`), bibliography (→ `data/bibliography/`), oracle abstractions (→ `data/oracles/`), analysis outputs (→ `analysis/`).

**Lifecycle**: write-once, read-many. Editing requires a `DECISIONS.md` entry.

### 2.2 `data/source_paper/`

**Purpose**: full text of the source paper (TTT-Discover, arxiv 2601.16175) in markdown form.

**Goes here**:
- `v1.md` — earlier extraction (PDF→md, has artifacts).
- `v2.md` — current canonical (LaTeX→md, clean).
- Future `v<N>.md` for re-extractions.

**Naming**: `v<N>.md`. The directory name carries the source-paper identity; do NOT prefix files with `ttt_discover_`.

### 2.3 `data/bibliography/`

**Purpose**: all artifacts from parsing TTT-Discover's bibliography and fetching cited papers.

**Goes here** (one slot per pipeline stage; all `_v<N>` versioned):
- `cite_keys_v<N>.json` — extracted citation keys from main.tex.
- `cited_bibtex_v<N>.jsonl` — structured BibTeX entries per cite key.
- `raw_refs_v<N>.jsonl` — raw references parsed from source paper.
- `resolved_v<N>.jsonl` — S2/OpenAlex-resolved entries with metadata + `full_text_path`.
- `unresolved_v<N>.jsonl` — entries that failed resolution.
- `REDO_SUMMARY_v<N>.md` — per-version accounting report.
- `full_text/<arxiv_id>.md` — per-paper full text (currently 38 papers).
- (Phase 1) `faiss_index/{embeddings.npy, index.faiss, manifest.json}` — vector store for retrieval.

**Naming**: stage-noun + version. Avoid the `ttt_discover_` prefix; the directory name is the source.

### 2.4 `data/oracles/`

**Purpose**: oracle abstractions used as ceiling estimates and as scaffolding for inference-time experiments.

**Subdirs**:
- `smoke/` — hand-coded smoke oracles. Files `v1.md`, `v2.md`, `v3.md`, etc.
- `<build_name>_<YYYY_MM_DD>_build/` — multi-round Opus-generated oracles. Each build dir contains:
  - `final.md` — the assembled oracle (the artifact training/eval reads).
  - `round0_relevance.jsonl` — relevance filter outputs.
  - `round1/`, `round2/`, `round3/` — per-paper extraction JSONs per round (one file per surviving paper).
  - `manifest.json` — paper-list + round-status snapshot.
  - `prompts/` — exact prompt strings used per round (for reproducibility).

**Naming**: smoke files versioned `v<N>.md`; build dirs include date and intent. The `<build_name>` token reflects the design version (e.g. `oracle_v2_2026_04_26_build`).

### 2.5 `data/plans/`

**Purpose**: external-LLM-generated plans, used as distillation / SFT data or as eval references.

**Goes here**:
- `opus_v<N>.jsonl` — Opus-generated plans (one row per plan).
- `gpt_v<N>.jsonl` — if GPT-N plans are added later.
- `frozen_qwen30b_v<N>.jsonl` / `frozen_qwen235b_v<N>.jsonl` — frozen-model rollouts produced as a reusable dataset.

**Distinction**: a plan dataset becomes a `data/plans/` artifact when it is **reused across runs**. One-off rollouts produced inside a single run stay in that run's `train/eval_rollouts.jsonl`.

### 2.6 `runs/<YYYY_MM_DD>_<intent>/` (inner schema)

**Purpose**: one directory per training or evaluation execution.

**Recommended inner layout** (matches archive practice; subgroup directories optional but **strongly encouraged** for new runs to keep run dirs navigable as artifact count grows):

```
<YYYY_MM_DD>_<intent>/
├── config.json                     The launch config (frozen at start)
├── code.diff                       git diff vs main at launch
├── hypothesis.md                   Pre-registered hypothesis + decision rule
│
├── train/                          Training-loop artifacts
│   ├── buffer.jsonl
│   ├── metrics.jsonl
│   ├── checkpoints.jsonl
│   ├── eval_rollouts.jsonl
│   └── iter_summary.jsonl          (optional)
│
├── critic/                         Critic-step artifacts (SDPO critique)
│   ├── log.jsonl
│   ├── requests/                   Per-iter request JSONs
│   └── responses/                  Per-iter response JSONs
│
├── audit/                          Opus depth-audit artifacts
│   ├── log.jsonl
│   ├── log_v<N>.jsonl              (only if multiple audit prompt versions co-exist)
│   ├── requests/
│   ├── responses/
│   └── responses_v<N>try<M>/       (when iterating on audit prompt)
│
├── reviewer/                       (Phase 2+) Reviewer subagent artifacts (privileged-info SDPO reviewer)
│   ├── log.jsonl
│   ├── requests/
│   └── responses/
│
├── pairwise/                       Pairwise eval artifacts (only present for pairwise runs)
│   ├── matchups_meta.json
│   ├── requests/
│   └── responses/
│
├── logs/                           Process logs
│   ├── run.log
│   ├── critic_daemon.log
│   ├── audit_daemon.log
│   └── reviewer_daemon.log
│
└── analysis/                       This run's post-hoc analysis (NOT cross-run)
    ├── analysis.md                 Headline summary + key tables
    ├── opus_audit_summary.md
    ├── opus_depth_audit.json
    ├── pairwise_summary.md         (pairwise runs)
    ├── fair_pairwise_summary.md    (pairwise runs, fair-condition variant)
    └── scoring_summary.json
```

**Notes**:
- The pre-realignment archive uses a **flatter** layout (buffer.jsonl etc. at run-dir root). **Do not modify archived dirs**; they are frozen.
- For new runs, the train/critic/audit/reviewer/pairwise/logs/analysis subgrouping is the norm. If the trainer hardcodes flat paths and switching is too costly, document that exception in `hypothesis.md`.
- **Cross-run analysis NEVER goes inside a single run's `analysis/`**. It belongs in `analysis/<purpose>/`.

### 2.7 `analysis/<purpose>/`

**Purpose**: offline analysis crossing runs or analyzing datasets independent of any single run.

**Inner layout** (matches D4 grant_proposal precedent):
```
<purpose>/
├── scripts/                Analysis scripts (.py)
├── data/                   Intermediate jsonl/csv/json
└── reports/                Final outputs — markdown summaries, ipynb, figures
    └── v<N>/               Per-revision sub-dir when the analysis is iterated
```

**Naming for `<purpose>`**: `<verb>_<noun>` or `<noun>_<noun>` — e.g. `pairwise_tournament`, `audit_v3_calibration`, `cross_goal_pilot`, `ablation_round_count`. Full English words; no abbreviations.

**Notebook placement**: `analysis/<purpose>/reports/<name>_v<N>.ipynb` for analyses with supporting files. For very lightweight one-off notebooks with no companions, `analysis/<name>_v<N>.ipynb` at the analysis root is acceptable.

### 2.8 `knowledge/`

**Purpose**: design docs that guide the project (oracle design, audit rubric, reviewer standards). Slow-changing; not run-specific.

- `current/` — authoritative as of now.
- `archive/` — superseded docs. Always preserve.

**Naming**: `<TOPIC>_v<N>.md` with **uppercase prefix** for visibility, e.g. `ORACLE_DESIGN_v1.md`, `AUDIT_RUBRIC_v3.md`. Templates: `<NAME>_TEMPLATE.md`. Handoff notes: `HANDOFF_<YYYY_MM_DD>.md`.

### 2.9 `specs/`

**Purpose**: markdown specifications for processes that will be executed elsewhere.

- `subagents/` — Subagent prompt templates. Examples: `opus_oracle_subagent.md`, `opus_critic_audit_daemon.md`, `opus_critic_audit_daemon_v2_addendum.md`.
- `runs/` — Pre-launch run specs (filled-out `RUN_CONFIRMATION_TEMPLATE`). One file per planned run. Naming: `<run_name>.md` matches the run dir that will be created.

**Distinction from `scripts/`**: `specs/` contains *human/agent instructions in prose*; `scripts/` contains *executable code*.

### 2.10 `scripts/`

**Purpose**: ad-hoc utility scripts that don't belong in `src/co_scientist/d5_abstract_retrieve_refine/` — housekeeping, one-off cleanups, data migrations, schema validations.

**Currently empty** (post-migration). Production code (build_*, train_*, fetch_*, extract_*, parse_*) lives under `src/`.

### 2.11 `configs/`

**Purpose**: YAML configs for launches. One config = one runnable profile.

**Naming**: `<method>_<intent>_v<N>.yaml`, e.g. `mu_baseline_v1.yaml`, `phase2_pilot_opus_reviewer_v1.yaml`.

### 2.12 `paper/` (Phase 5+)

**Purpose**: ICLR 2027 LaTeX source. Created in Phase 5.

**Recommended layout** (when created):
```
paper/
├── main.tex
├── sections/                    Per-section .tex files
├── figures/                     PDFs / PNGs embedded in main.tex
│   └── source/                  Notebook scripts that generate the figures
├── tables/                      .tex tables
├── references.bib
└── build/                       Auxiliary build artifacts (.aux, .log, .pdf)
```

**Workflow rule** (per user CLAUDE.md): when generating a figure, add `\includegraphics` to `main.tex` in the same step and verify it appears in the compiled PDF.

### 2.13 `paper_materials/` (added 2026-04-27, post-Phase-2)

**Purpose**: paper-writing canonical store. Distinct from `knowledge/current/` (design
docs, evolving) and `analysis/` (cross-run analysis): paper_materials is **frozen
output** ready to drop into a manuscript. Read `paper_materials/README.md` for the
intra-folder index.

**Goes here** (one file per finding / experiment / figure / exemplar / method):
- `findings/F<N>_<topic>.md` — narrative write-up of one finding (1-2 pages)
- `experiments/E<N>_<short_name>.{json,md}` — raw data dump per finding
- `figures_data/G<N>_<short_name>.csv` — matplotlib-ready CSV (one per figure)
- `plan_samples/exemplar_<context>.md` — exemplar plan with provenance header
- `methodology/M<N>_<short_name>.md` — self-contained methodology doc; OK to be a
  copy of `knowledge/current/<DOC>.md` so paper_materials/ is portable
- `related_work_excerpts/<short_name>.md` — verbatim quotes + citations from cited papers
- `next_steps/N<N>_<topic>.md` — open questions / future-work skeletons

**Does NOT go here**: in-progress design docs (→ `knowledge/current/`), training run
outputs (→ `runs/`), cross-run analysis (→ `analysis/`), source paper / oracles / data
(→ `data/`).

**Update rule** (per CONVENTIONS.md): every experiment that produces paper-relevant
data must update `paper_materials/` in the SAME commit as the experiment artifacts.
Don't let `paper_materials/` drift from `runs/`.

**Naming**: `F<N>_`, `E<N>_`, `G<N>_`, `M<N>_`, `N<N>_` prefix indicates type and
ordering within type (F1 = first finding, F2 = second, etc.). Free-form short_name
after prefix.

**Lifecycle**: append-mostly. Old findings can be superseded by new ones (rename
`F2_<topic>.md` → `F2_<topic>_archive.md` and add a successor `F2_<new_topic>.md`).
Don't delete findings — paper-writing may revisit them.

---

## 3. Decision tree — "where does my new file go?"

Walk top-down; pick the first match.

```
1.  Canonical task input (research goal, reference solution, perturbation, held-out goal)?
    → dataset/

2.  Output written by a TRAINING / EVAL RUN during its execution?
    → runs/<YYYY_MM_DD>_<intent>/<subgroup>/
       (subgroup ∈ {train, critic, audit, reviewer, pairwise, logs, analysis})

3.  Output of a DATA-PREP PIPELINE — one-time, reused across runs?
    3a. Source paper full text?            → data/source_paper/
    3b. Bibliography parse / fetch / FAISS? → data/bibliography/
    3c. Oracle abstraction?                  → data/oracles/{smoke, <build_name>}/
    3d. External-LLM plan dataset?           → data/plans/

4.  OFFLINE ANALYSIS that reads runs/ or data/ and writes reports?
    → analysis/<purpose>/{scripts, data, reports}/

5.  CONFIG for launching a run?
    → configs/<method>_<intent>_v<N>.yaml

6.  SUBAGENT PROMPT TEMPLATE or PRE-LAUNCH RUN SPEC?
    → specs/{subagents, runs}/<name>.md

7.  DESIGN / KNOWLEDGE document referenced across sessions?
    → knowledge/current/<TOPIC>_v<N>.md

8.  AD-HOC UTILITY CODE (one-off cleanup, migration script)?
    → scripts/<verb>_<noun>.{py, sh}

9.  PRODUCTION TRAINING / DATA-PREP CODE?
    → src/co_scientist/d5_abstract_retrieve_refine/<verb>_<noun>_v<N>.py
       (NOT under projects/d5_abstract_retrieve_refine/)

10. PAPER MATERIAL (.tex, figures for the paper)?
    → paper/   (Phase 5+)

11. None of the above?
    → STOP. Propose an addition to this schema before creating the file.
```

---

## 4. Migration plan — current state → schema

This section enumerates files that violate the schema and proposes moves. **Each move requires a corresponding code update in `src/`** because production scripts hardcode current paths. Do not execute a move without updating the listed line numbers in the same commit.

### 4.1 `data/` flat dump → nested layout

| Current path | Proposed path | `src/` references to update |
|---|---|---|
| `data/source_paper/v1.md` | `data/source_paper/v1.md` | `train_mu_baseline_v1.py:94`, `extract_paper_v1.py:22`, `extract_source_paper_v2.py:3-4` |
| `data/source_paper/v2.md` | `data/source_paper/v2.md` | `extract_source_paper_v2.py:28` (+ any consumer of v2) |
| `data/ttt_discover_raw_refs.jsonl` | `data/bibliography/raw_refs_v1.jsonl` | `fetch_bibliography_v1.py:38` |
| `data/bibliography/cite_keys_v1.json` | `data/bibliography/cite_keys_v1.json` | `extract_cite_keys_v1.py:23`, `parse_bibtex_v1.py:37` |
| `data/bibliography/cited_bibtex_v1.jsonl` | `data/bibliography/cited_bibtex_v1.jsonl` | `parse_bibtex_v1.py:38`, `build_bibliography_v2.py:52` |
| `data/ttt_discover_unresolved_refs.jsonl` | `data/bibliography/unresolved_v1.jsonl` | (grep before move) |
| `data/bibliography/unresolved_v2.jsonl` | `data/bibliography/unresolved_v2.jsonl` | (grep before move) |
| `data/bibliography/resolved_v1.jsonl` | `data/bibliography/resolved_v1.jsonl` | `fetch_bibliography_v1.py:37`, `build_oracle_v2.py:68` |
| `data/bibliography/resolved_v2.jsonl` | `data/bibliography/resolved_v2.jsonl` | `build_bibliography_v2.py:53` |
| `data/bibliography/REDO_SUMMARY_v2.md` | `data/bibliography/REDO_SUMMARY_v2.md` | `build_bibliography_v2.py:54` |
| `data/bibliography/full_text/` | `data/bibliography/full_text/` | `build_bibliography_v2.py:8,26` (`full_text_path` field) + every `full_text_path` value inside `resolved_v*.jsonl` |
| `data/bibliography/full_text_latex/` | `data/bibliography/full_text_latex/` | (added 2026-04-27; LaTeX-extracted full text — search src/ for `full_text_latex` references) |
| `data/oracles/smoke/v1.md` | `data/oracles/smoke/v1.md` | `smoke_pathway_v1.py:48` |
| `data/oracles/smoke/v2.md` | `data/oracles/smoke/v2.md` | (grep) |
| `data/oracles/smoke/v3.md` | `data/oracles/smoke/v3.md` | `train_mu_baseline_v1.py:91` |
| `data/plans/opus_v1.jsonl` | `data/plans/opus_v1.jsonl` | `train_alpha_baseline_v1.py:75` |
| `data/plans/opus_v2.jsonl` | `data/plans/opus_v2.jsonl` | (added 2026-04-27; search src/ + train_alpha_v2.py) |
| `data/oracles/oracle_v2_2026_04_26_build/final.md` | `data/oracles/oracle_v2_2026_04_26_build/final.md` | `build_oracle_v2.py:71` |
| `data/oracles/oracle_v2_2026_04_26_build/medium.md` | `data/oracles/oracle_v2_2026_04_26_build/medium.md` | (grep — no hardcoded references; consumed via slim) |
| `data/oracles/oracle_v2_2026_04_26_build/slim.md` | `data/oracles/oracle_v2_2026_04_26_build/slim.md` | `train_mu_v2.py`, `train_mu_v3.py`, `train_mu_v4.py`, `baseline_*_v2.py`, `train_alpha_v2.py`, `train_beta_v2.py`, `train_sigma_v2.py` |
| `data/oracles/oracle_v2_2026_04_26_build/` | `data/oracles/oracle_v2_2026_04_26_build/` (round_0..3 + requests/responses) | `build_oracle_v2.py:70` (`out_dir`) |
| `data/expand_full_text_v1_log.jsonl` | (DELETE — Q3 user-confirmed; one-off run log) | (no consumers) |
| `data/extract_arxiv_latex_v1_log.jsonl` | (DELETE — Q3 user-confirmed; one-off run log) | (no consumers) |

### 4.2 `scripts/` → `specs/subagents/`

The current `scripts/` directory contains **only markdown** (subagent specs, not executable code). Move to `specs/`:

| Current path | Proposed path |
|---|---|
| `scripts/opus_oracle_subagent.md` | `specs/subagents/opus_oracle_subagent.md` |
| `scripts/opus_critic_audit_daemon.md` | `specs/subagents/opus_critic_audit_daemon.md` |
| `scripts/opus_critic_audit_daemon_v2_addendum.md` | `specs/subagents/opus_critic_audit_daemon_v2_addendum.md` |

Post-migration `scripts/` is empty and reserved for ad-hoc executable utilities.

### 4.3 Misplaced run spec

| Current path | Proposed path |
|---|---|
| `runs/run_spec_oracle_build_v2.md` | `specs/runs/oracle_build_v2_2026_04_26.md` |

A run spec is a **pre-launch** document, not a run output. It belongs in `specs/runs/`. When the run launches, a separate `runs/2026_04_26_oracle_build_v2/` directory is created for actual outputs.

### 4.4 Flat `analysis/d1_rubric_validation/`

Currently 4 jsons + 2 mds at the same level. Schema requires `scripts/` + `data/` + `reports/` subdirs.

| Current file | Proposed slot |
|---|---|
| `d1_8plans_opus_audit.json`, `d1_8plans_same_goal_query_policy.json`, `d1_5plans_group215_2.json`, `d1_opus_audit.json` | `analysis/d1_rubric_validation/data/` |
| `d1_8plans_opus_audit_summary.md`, `d1_opus_audit_summary.md` | `analysis/d1_rubric_validation/reports/` |

No Python in this analysis; `scripts/` stays empty.

### 4.5 Migration policy

- **Do not execute any move without explicit user confirmation.** The migrations are mechanical but require atomic commits that update both files and `src/` references.
- Suggested order:
  1. Commit current state under existing layout (so HEAD is clean).
  2. Execute migration in a single commit titled `D5 SCHEMA migration: data/ → nested layout`.
  3. Update `STATUS.md` "Load-bearing paths" with the new canonical paths.
  4. Append a `DECISIONS.md` entry documenting the migration.

---

## 5. Naming patterns (cross-references CONVENTIONS.md)

These rules govern file/directory **names**; placement is in § 2.

- **Verb prefix on production code**: `build_`, `fetch_`, `extract_`, `parse_`, `train_`, `validate_`, `analyze_`, `grade_`, `regrade_`, `test_`, `run_`.
- **Version suffix**: `_v<N>` on filenames, NOT directory names. Exception: dated multi-file builds like `oracle_v2_2026_04_26_build/`.
- **Run dir**: `<YYYY_MM_DD>_<intent>`.
- **Archive**: any retired content moves into the nearest `_archive/` subdir; never deleted.
- **Prohibited dir/file names**: `tmp`, `misc`, `stuff`, `new_*`, `final_*`, `old_*`, leading-digit-only.
- **Canonical spellings** (paper concepts): see `CONVENTIONS.md` § Canonical Spellings.

---

## 6. Phase forecast — anticipated future artifacts

Predicted files for upcoming phases, with their schema-determined slots. This is *forecast*, not commitment.

### Phase 0b (immediate — bibliography fetch + oracle v2 build)

- `data/bibliography/full_text/<arxiv_id>.md` — newly fetched papers (currently 38; expect up to ~50 after retries).
- `data/oracles/oracle_v2_2026_04_26_build/{final.md, round0_relevance.jsonl, round1/, round2/, round3/, manifest.json, prompts/}` — multi-round Opus build outputs.
- `specs/runs/oracle_build_v2_2026_04_26.md` — run spec (currently misplaced at `runs/run_spec_oracle_build_v2.md`).

### Phase 1 (infrastructure modules)

Most code lands in `src/`, not `projects/`:
- `src/co_scientist/shared/per_goal_database.py` — FAISS + content loader.
- `src/co_scientist/shared/paper_selection.py` — selection action head + parser.
- `src/co_scientist/shared/abstraction_extractor.py` — abstraction generator.
- `src/co_scientist/shared/reviewer_subagent_client.py` — Opus reviewer file-bus.
- `src/co_scientist/shared/sdpo_distillation.py` — SDPO loss.
- `src/co_scientist/shared/final_plan_supervised.py` — CE loss vs reference plan.
- `src/co_scientist/d5_abstract_retrieve_refine/train_d5_sdpo_v1.py` — main trainer.
- `src/co_scientist/d5_abstract_retrieve_refine/pipeline_orchestrator.py` — multi-round loop.
- `tests/d5/test_*.py` — unit tests under repo-root `tests/`.

In `projects/d5_abstract_retrieve_refine/`:
- `data/bibliography/faiss_index/{embeddings.npy, index.faiss, manifest.json}` — vector store.
- `configs/d5_sdpo_v1_*.yaml` — Phase-1 launch configs.
- `knowledge/current/SDPO_LOSS_DESIGN_v1.md`, `RETRIEVAL_PIPELINE_v1.md` — design docs.

### Phase 2 (training pilot, 4-5 weeks out)

- `runs/2026_05_<NN>_phase2_pilot_<variant>/` — multiple run dirs with full inner schema (§ 2.6).
- `specs/runs/phase2_pilot_<variant>.md` — pre-launch specs (one per planned variant).
- `analysis/phase2_pilot_summary/` — cross-variant comparison.

### Phase 3 (full training + ablations, 6-9 weeks out)

- Many run dirs with intent tokens like `ablation_round_count`, `ablation_selection_random`, `ablation_no_sdpo`, `ablation_self_review`.
- `analysis/ablation_round_count/`, `analysis/ablation_selection/`, `analysis/sdpo_ablation/`.
- `data/plans/frozen_qwen30b_no_pipeline_v1.jsonl`, `data/plans/frozen_qwen235b_with_ref_v1.jsonl` — fixed baseline-rollout pools (if reused across ablations).

### Phase 4 (evaluation + baselines, 10-11 weeks out)

- `dataset/held_out/<goal_name>/{research_goal.txt, reference_solution.txt}` for 2-3 cross-goal validation goals.
- `runs/2026_<NN>_eval_held_out_<goal>/` — held-out eval runs.
- `runs/2026_<NN>_baseline_<name>/` — baseline runs (frozen-235B + ref, pure SFT, full-pipeline-no-SDPO).
- `analysis/cross_goal_eval/`, `analysis/baseline_comparison/`.

### Phase 5 (paper, week 12)

- `paper/{main.tex, sections/, figures/, tables/, references.bib}` — see § 2.12.
- Figures generated by `analysis/<purpose>/scripts/build_figure_*.py` are copied/symlinked into `paper/figures/`. Source script paths recorded in `paper/figures/source/<figname>.txt` for reproducibility.

---

## 7. Schema evolution

**Add a new top-level slot ONLY when**:

1. The new file type is produced regularly (not one-off).
2. It does not fit any existing slot under the § 3 decision tree.
3. The proposed slot has > 3 expected files in its first month of use.

Process:
- Append a section to § 2 (Slot specifications) with the new slot's purpose, contents, naming.
- Add a row to the § 3 decision tree.
- Append a `DECISIONS.md` entry recording the addition.
- Update the top-level tree in § 1.

**Retire a slot ONLY when**:

1. It has been empty for > 1 month, AND
2. No code references the path.

Process:
- Move the empty directory into `_archive/`.
- Update the schema with a strikethrough on the retired slot.
- Append a `DECISIONS.md` entry recording the retirement.
