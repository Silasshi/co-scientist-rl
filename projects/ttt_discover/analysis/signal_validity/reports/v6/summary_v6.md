# Signal Validity Test Results

Grader: Qwen/Qwen3-30B-A3B
Total references graded: 162

## Distribution of aggregate scores (higher is better)
- Mean: 0.549
- Median: 0.546
- Std: 0.074
- Min: 0.355
- Max: 0.730

## Pass-rate thresholds
- Score >= 0.9 : 0.0% (0/162)
- Score >= 0.8 : 0.0% (0/162)
- Score >= 0.7 : 1.9% (3/162)
- Score >= 0.6 : 27.2% (44/162)
- Score >= 0.5 : 79.0% (128/162)

## By source

| Source | Count | Mean | Median | Std | ≥0.7 | ≥0.8 |
|---|---|---|---|---|---|---|
| facebook_arxiv | 50 | 0.552 | 0.549 | 0.065 | 0% | 0% |
| facebook_ml | 50 | 0.532 | 0.540 | 0.073 | 0% | 0% |
| facebook_pubmed | 50 | 0.546 | 0.549 | 0.074 | 0% | 0% |
| papers_analysis | 12 | 0.611 | 0.616 | 0.087 | 25% | 0% |

## Per-signal distribution

| Signal | Weight | Mean | Median | ≥4 count | ≥3 count |
|---|---|---|---|---|---|
| S1_depth | 0.10 | 3.36 | 3.0 | 36% | 100% |
| S2_rigor | 0.12 | 3.12 | 3.0 | 24% | 90% |
| S3_positioning | 0.10 | 2.35 | 2.5 | 19% | 50% |
| S4_significance | 0.13 | 3.52 | 4.0 | 52% | 100% |
| S5_stability | 0.08 | 2.82 | 3.0 | 2% | 79% |
| S6_failure_interp | 0.11 | 3.02 | 3.0 | 4% | 99% |
| S7_specificity | 0.08 | 3.26 | 3.0 | 26% | 99% |
| S8_scope | 0.15 | 3.43 | 4.0 | 63% | 85% |
| S9_focus | 0.13 | 3.78 | 4.0 | 77% | 100% |

## Interpretation

**If signals are well-calibrated:** reference plans (which are human-written methodologies from accepted papers) should score >= 0.7 on aggregate.

**Signals are systematically underrating references if:**
- Median aggregate is << 0.7
- Many individual signals have median score < 4
- The distribution clusters in the 0.5-0.7 range
