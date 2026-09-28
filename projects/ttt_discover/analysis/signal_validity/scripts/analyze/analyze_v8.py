#!/usr/bin/env python3
"""V8 analysis: did the S4 gate fix work? did S1/S9 calibration improve Opus correlation?

Compares v8 to v7 baseline.
"""

import json
import statistics
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
SIGS = ['S1_depth','S2_rigor','S3_positioning','S4_significance',
        'S5_feasibility','S6_risk_awareness','S7_specificity','S8_scope','S9_focus']

TARGET = {
    'P_S1_asserted': 'S1_depth',
    'P_S2_strawman': 'S2_rigor',
    'P_S3_no_positioning': 'S3_positioning',
    'P_S4_weak_problem': 'S4_significance',
    'P_S5_vague_method': 'S5_feasibility',
    'P_S6_no_risk': 'S6_risk_awareness',
    'P_S7_no_specifics': 'S7_specificity',
    'P_S8_overclaim': 'S8_scope',
    'P_S9_stacked': 'S9_focus',
}

# V7 baselines (for comparison)
V7_DETECTION = {
    'P_S1_asserted': 0.70, 'P_S2_strawman': 1.00,
    'P_S3_no_positioning': 0.56, 'P_S4_weak_problem': 0.17,
    'P_S5_vague_method': 0.89, 'P_S6_no_risk': 0.88,
    'P_S7_no_specifics': 0.64, 'P_S8_overclaim': 1.00, 'P_S9_stacked': 0.41,
}


def load():
    with open(BASE / 'data' / 'refs' / 'grading_refs_v8.jsonl') as f:
        refs = {json.loads(l)['source_id']: json.loads(l) for l in f}
    with open(BASE / 'data' / 'perturbations' / 'grading_perturbations_v8.jsonl') as f:
        perturbs = [json.loads(l) for l in f]
    return refs, perturbs


def main():
    refs, perturbs = load()
    print(f"References (v8): {len(refs)}, Perturbations (v8): {len(perturbs)}\n")

    # Q1: Detection accuracy per signal
    print("=" * 75)
    print("Q1: DETECTION ACCURACY (v8) vs v7 — did S4 improve, did others hold?")
    print("=" * 75)
    by_type = defaultdict(list)
    for p in perturbs:
        by_type[p['perturbation_type']].append(p)

    print(f"\n{'Perturbation':<25} {'target':<20} {'v8 det%':>8} {'v7 det%':>8} {'Δ':>6}")
    print('-' * 75)
    for ptype, target in TARGET.items():
        pert_list = by_type.get(ptype, [])
        if not pert_list:
            continue
        target_drops = []
        for p in pert_list:
            if p['base_id'] not in refs:
                continue
            ref_score = refs[p['base_id']]['signals_median'].get(target)
            pert_score = p['signals_median'].get(target)
            if ref_score is None or pert_score is None:
                continue
            target_drops.append(ref_score - pert_score)

        if not target_drops:
            continue
        det_v8 = sum(1 for d in target_drops if d >= 1) / len(target_drops)
        det_v7 = V7_DETECTION.get(ptype)
        delta_str = f"{(det_v8-det_v7)*100:+.0f}%" if det_v7 else "  --"
        v7_str = f"{det_v7:>7.0%}" if det_v7 else "    N/A"
        print(f"  {ptype:<23} {target:<20} {det_v8:>7.0%} {v7_str} {delta_str:>6}")

    # Q2: Aggregate separation
    print("\n" + "=" * 75)
    print("Q2: AGGREGATE SCORE — refs vs perturbations (v8)")
    print("=" * 75)
    ref_aggs = [r['aggregate'] for r in refs.values()]
    pert_aggs = [p['aggregate'] for p in perturbs]
    print(f"  Refs:  n={len(ref_aggs)}, mean={statistics.mean(ref_aggs):.3f}, "
          f"range {min(ref_aggs):.3f}-{max(ref_aggs):.3f}")
    print(f"  Perts: n={len(pert_aggs)}, mean={statistics.mean(pert_aggs):.3f}, "
          f"range {min(pert_aggs):.3f}-{max(pert_aggs):.3f}")
    print(f"  Gap (ref μ - pert μ): {statistics.mean(ref_aggs) - statistics.mean(pert_aggs):+.3f}")
    above = 0; total = 0
    for r in ref_aggs:
        for p in pert_aggs:
            if r > p: above += 1
            total += 1
    print(f"  AUC P(ref>pert): {above/total:.3f}")
    sorted_ref = sorted(ref_aggs)
    p50 = statistics.median(ref_aggs)
    below_50 = sum(1 for p in pert_aggs if p < p50)
    print(f"  Perts below ref median: {below_50}/{len(pert_aggs)} ({below_50/len(pert_aggs):.0%})")

    # v7 baseline for comparison
    print("\n  V7 BASELINE: gap=0.064, AUC=0.685, perts below ref median=81%")

    # Signal means
    print("\n  Signal means (refs vs perts):")
    for sid in SIGS:
        rvals = [r['signals_median'].get(sid) for r in refs.values()
                 if r['signals_median'].get(sid) is not None]
        pvals = [p['signals_median'].get(sid) for p in perturbs
                 if p['signals_median'].get(sid) is not None]
        if rvals and pvals:
            print(f"    {sid:<22} ref_μ={statistics.mean(rvals):.2f}  "
                  f"pert_μ={statistics.mean(pvals):.2f}  Δ={statistics.mean(rvals) - statistics.mean(pvals):+.2f}")

    # Q3: Correlation with Opus
    print("\n" + "=" * 75)
    print("Q3: CORRELATION with OPUS ratings")
    print("=" * 75)
    opus_path = BASE / "data" / "human_ratings" / "human_ratings_v2.jsonl"
    if opus_path.exists():
        with open(opus_path) as f:
            opus = {json.loads(l)['source_id']: json.loads(l) for l in f}
        xs, ys = [], []
        for sid, r in refs.items():
            if sid in opus:
                xs.append(r['aggregate'])
                ys.append(opus[sid]['overall'])
        if xs:
            r = pearsonr(xs, ys).statistic
            print(f"  V8 refs vs Opus: r={r:.3f} (n={len(xs)})")
            print(f"  V7 baseline:      r=0.158  (dropped from v6=0.395)")
            print(f"  Goal: recover v6 level (>0.3)")

    # Q4: Inter-signal
    print("\n" + "=" * 75)
    print("Q4: INTER-SIGNAL CORRELATION")
    print("=" * 75)
    all_entries = list(refs.values()) + perturbs
    X = []
    for e in all_entries:
        sig = e['signals_median']
        X.append([sig.get(sid) if sig.get(sid) is not None else 3 for sid in SIGS])
    X = np.array(X, dtype=float)
    print(f"\nMatrix (n={len(X)}):")
    print(" " * 22 + "  ".join(f"{s[:3]}" for s in SIGS))
    for i, si in enumerate(SIGS):
        row = [f"{pearsonr(X[:,i], X[:,j]).statistic:+.2f}" for j in range(len(SIGS))]
        print(f"  {si:<22} " + " ".join(row))


if __name__ == "__main__":
    main()
