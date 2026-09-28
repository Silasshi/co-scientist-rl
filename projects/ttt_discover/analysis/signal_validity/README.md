# Signal Validity — Perturbation-Based Signal Validation Pipeline

This directory holds all data, scripts, and reports for validating the
multi-signal reward system via targeted perturbations.

## Canonical current state

**v8.1-minimal** (2026-04-16). See `reports/v8_1/RESULTS_SUMMARY_v8_1.md`
for the current canonical findings. Key numbers:

- 60 reference plans + 162 perturbations
- AUC P(ref > pert) = 0.767
- 84% of perturbations below ref P50
- Detection rates per signal: S2 100%, S8 94%, S7 93%, S9 88%, S6 88%, S5 83%, S3 61%, S1 55%, S4 disabled

## Layout

```
signal_validity/
├── README.md                    ← this file
│
├── data/
│   ├── refs/                    ← reference-plan .jsonl files
│   │   ├── references.jsonl             (v1, legacy)
│   │   ├── references_v2.jsonl          (current 60-ref dataset)
│   │   ├── grading_refs_v7.jsonl        (v7 rubric scores on 60 refs)
│   │   └── grading_refs_v8.jsonl        (v8 rubric scores on 60 refs)
│   ├── perturbations/           ← perturbation .jsonl files
│   │   ├── perturbations.jsonl               (162 perturbed plans)
│   │   ├── grading_perturbations_v6.jsonl    (v6 rubric scores)
│   │   ├── grading_perturbations_v7.jsonl
│   │   ├── grading_perturbations_v8.jsonl
│   │   └── s4_regraded_v8_1.jsonl           (S4-only re-grade after v8.1 gate fix)
│   └── human_ratings/           ← Opus-as-human proxy ratings
│       ├── human_ratings.jsonl  (v1)
│       └── human_ratings_v2.jsonl
│
├── perturbations/               ← 162 .txt files of perturbed plan content
│                                   (named {base}_{perturbation_type}.txt)
│
├── per_signal/                  ← 9 per-signal sample review .md files
├── sample_refs/                 ← 5 reference plan .txt exports
├── human_check/                 ← 10 per-plan human review notes
│
├── scripts/
│   ├── build/                   ← dataset construction
│   │   ├── build_reference_dataset.py
│   │   ├── build_references_v2.py
│   │   ├── build_perturbations.py
│   │   ├── build_s1s2_perturbations.py
│   │   ├── rebuild_from_full_papers.py
│   │   ├── enhance_specificity.py
│   │   ├── expand_via_semantic_scholar.py
│   │   └── auto_pipeline.sh             (legacy v6 orchestrator)
│   ├── grade/                   ← LLM-graded scoring runs
│   │   ├── grade_perturbations_v6.py    (renamed from grade_perturbations.py)
│   │   ├── grade_perturbations_v7.py
│   │   ├── grade_perturbations_v8.py
│   │   ├── grade_refs_v7.py
│   │   ├── grade_refs_v8.py
│   │   ├── regrade_s4_v8_1.py           (partial re-grade of S4 only)
│   │   ├── run_reference_grading.py
│   │   ├── run_v2_grading.py
│   │   ├── run_grading_with_reasoning.py
│   │   ├── run_235b_subset.py
│   │   └── human_review_all.py
│   └── analyze/                 ← read .jsonl, print summary stats
│       ├── analyze_v6.py                (renamed from analyze_final.py)
│       ├── analyze_v7.py
│       ├── analyze_v8.py
│       ├── analyze_v8_1.py              ← current
│       ├── test_aggregation_formulas.py (ablation on aggregation method)
│       └── test1_finer_scale.py         (grader-scale ablation)
│
├── reports/                     ← human-readable findings
│   ├── v6/
│   ├── v7/
│   ├── v8/
│   └── v8_1/                    ← current canonical
│       ├── FINAL_ANALYSIS_v8_1.txt
│       └── RESULTS_SUMMARY_v8_1.md
│
├── logs/                        ← 30+ .log files from grading runs
│
└── archive/                     ← pre-v6 grading data (unreferenced)
    ├── grading_results.jsonl
    ├── grading_results_v2.jsonl
    ├── grading_results_v5.jsonl
    ├── grading_235b_subset.jsonl
    └── grading_235b_subset.jsonl.bak
```

## How scripts resolve paths

All scripts under `scripts/<verb>/` compute:

```python
BASE = Path(__file__).resolve().parents[2]   # signal_validity/
PROJECT_ROOT = Path(__file__).resolve().parents[6]  # co-scientist-project/
```

Data is then referenced as `BASE / "data" / "refs" / "references_v2.jsonl"`, etc.

## When adding new content

- New reference dataset version → `data/refs/references_v<N>.jsonl` + script in `scripts/build/`
- New perturbation batch → write to `data/perturbations/perturbations.jsonl` or a versioned variant
- New grading run → `scripts/grade/grade_<what>_v<N>.py` writing to `data/<refs|perturbations>/grading_<what>_v<N>.jsonl`
- New analysis → `scripts/analyze/analyze_v<N>.py` reading from data/, writing text output to `reports/v<N>/`

See `../../CONVENTIONS.md` for project-wide naming rules.
