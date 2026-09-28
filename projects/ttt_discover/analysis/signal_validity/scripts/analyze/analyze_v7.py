#!/usr/bin/env python3
"""V7 analysis: signal validity with explicit-counting rubrics.

Reads grading_refs_v7.jsonl + grading_perturbations_v7.jsonl.
Compares v7 detection rates against v6 baseline.
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

# Map perturbation type to target signal
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

# v6 detection rates from FINAL_ANALYSIS.txt
V6_DETECTION = {
    'P_S1_asserted': None,  # not tested in v6
    'P_S2_strawman': None,
    'P_S3_no_positioning': 8/18,
    'P_S4_weak_problem': 8/18,
    'P_S5_vague_method': 16/18,
    'P_S6_no_risk': 15/17,
    'P_S7_no_specifics': 8/17,
    'P_S8_overclaim': 7/16,
    'P_S9_stacked': 4/17,
}


def load():
    with open(BASE / 'data' / 'refs' / 'grading_refs_v7.jsonl') as f:
        refs = {json.loads(l)['source_id']: json.loads(l) for l in f}
    with open(BASE / 'data' / 'perturbations' / 'grading_perturbations_v7.jsonl') as f:
        perturbs = [json.loads(l) for l in f]
    return refs, perturbs


def main():
    refs, perturbs = load()
    print(f"References (v7): {len(refs)}, Perturbations (v7): {len(perturbs)}\n")

    # Q1: Detection accuracy per signal
    print("=" * 75)
    print("Q1: DETECTION ACCURACY (v7) — does each signal detect its perturbation?")
    print("=" * 75)
    by_type = defaultdict(list)
    for p in perturbs:
        by_type[p['perturbation_type']].append(p)

    print(f"\n{'Perturbation':<25} {'target':<20} {'v7 det%':>8} {'v6 det%':>8} {'Δ':>6}")
    print('-' * 75)
    for ptype, target in TARGET.items():
        pert_list = by_type.get(ptype, [])
        if not pert_list:
            continue
        target_drops = []
        all_signal_drops = defaultdict(list)
        for p in pert_list:
            if p['base_id'] not in refs:
                continue
            ref_score = refs[p['base_id']]['signals_median'].get(target)
            pert_score = p['signals_median'].get(target)
            if ref_score is None or pert_score is None:
                continue
            drop = ref_score - pert_score
            target_drops.append(drop)
            for sid in SIGS:
                r = refs[p['base_id']]['signals_median'].get(sid)
                q = p['signals_median'].get(sid)
                if r is not None and q is not None:
                    all_signal_drops[sid].append(r - q)

        if not target_drops:
            continue
        det_v7 = sum(1 for d in target_drops if d >= 1) / len(target_drops)
        det_v6 = V6_DETECTION.get(ptype)
        v6_str = f"{det_v6:>7.0%}" if det_v6 is not None else "    N/A"
        delta = (det_v7 - det_v6) if det_v6 is not None else None
        delta_str = f"{delta:+.0%}" if delta is not None else "  --"
        print(f"  {ptype:<23} {target:<20} {det_v7:>7.0%} {v6_str} {delta_str}")

    # Per-signal halo detail
    print("\n--- Halo detail (drops on non-target signals > 0.3) ---")
    for ptype, target in TARGET.items():
        pert_list = by_type.get(ptype, [])
        if not pert_list:
            continue
        all_signal_drops = defaultdict(list)
        for p in pert_list:
            if p['base_id'] not in refs:
                continue
            for sid in SIGS:
                r = refs[p['base_id']]['signals_median'].get(sid)
                q = p['signals_median'].get(sid)
                if r is not None and q is not None:
                    all_signal_drops[sid].append(r - q)
        halos = []
        for sid in SIGS:
            if sid == target:
                continue
            if all_signal_drops.get(sid):
                m = statistics.mean(all_signal_drops[sid])
                if abs(m) > 0.3:
                    halos.append(f"{sid}({m:+.2f})")
        if halos:
            print(f"  {ptype} → {', '.join(halos)}")

    # Q2: Aggregate separation
    print("\n" + "=" * 75)
    print("Q2: AGGREGATE SCORE — refs vs perturbations (v7)")
    print("=" * 75)
    ref_aggs = [r['aggregate'] for r in refs.values()]
    pert_aggs = [p['aggregate'] for p in perturbs]
    print(f"  Refs:  n={len(ref_aggs)}, mean={statistics.mean(ref_aggs):.3f}, "
          f"range {min(ref_aggs):.3f}-{max(ref_aggs):.3f}")
    print(f"  Perts: n={len(pert_aggs)}, mean={statistics.mean(pert_aggs):.3f}, "
          f"range {min(pert_aggs):.3f}-{max(pert_aggs):.3f}")
    overlap = sum(1 for p in pert_aggs if p >= min(ref_aggs))
    print(f"  Perts above min ref: {overlap}/{len(pert_aggs)} ({overlap/len(pert_aggs):.0%})")
    # AUC: P(random ref > random pert)
    above = 0
    total = 0
    for r in ref_aggs:
        for p in pert_aggs:
            if r > p: above += 1
            total += 1
    print(f"  AUC (P(ref > pert)): {above/total:.3f}")

    # Per-signal mean across refs vs perturbations
    print("\n  Signal means (refs vs all perts):")
    for sid in SIGS:
        rvals = [r['signals_median'].get(sid) for r in refs.values()
                 if r['signals_median'].get(sid) is not None]
        pvals = [p['signals_median'].get(sid) for p in perturbs
                 if p['signals_median'].get(sid) is not None]
        if rvals and pvals:
            print(f"    {sid:<22} ref_μ={statistics.mean(rvals):.2f}  "
                  f"pert_μ={statistics.mean(pvals):.2f}  Δ={statistics.mean(rvals) - statistics.mean(pvals):+.2f}")

    # Q3: Correlation with Opus (if available)
    print("\n" + "=" * 75)
    print("Q3: CORRELATION with OPUS ratings")
    print("=" * 75)
    opus_path = BASE / "data" / "human_ratings" / "human_ratings_v2.jsonl"
    if not opus_path.exists():
        print("  No Opus ratings available")
    else:
        with open(opus_path) as f:
            opus = {json.loads(l)['source_id']: json.loads(l) for l in f}
        xs, ys = [], []
        for sid, r in refs.items():
            if sid in opus:
                xs.append(r['aggregate'])
                ys.append(opus[sid]['overall'])
        if xs:
            r = pearsonr(xs, ys).statistic
            print(f"  Refs: r={r:.3f} (n={len(xs)})")

    # Q4: Inter-signal correlation
    print("\n" + "=" * 75)
    print("Q4: INTER-SIGNAL CORRELATION (combined refs+perts)")
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
    print("\nRedundant pairs (|r| > 0.7):")
    redundant = False
    for i in range(len(SIGS)):
        for j in range(i+1, len(SIGS)):
            r = pearsonr(X[:,i], X[:,j]).statistic
            if abs(r) > 0.7:
                print(f"  {SIGS[i]} + {SIGS[j]}: r={r:+.3f}")
                redundant = True
    if not redundant:
        print("  (none)")

    # Q5: Discriminative power
    print("\n" + "=" * 75)
    print("Q5: SIGNAL DISCRIMINATIVE POWER (variance across all entries)")
    print("=" * 75)
    for i, sid in enumerate(SIGS):
        vals = X[:, i]
        print(f"  {sid:<22} mean={vals.mean():.2f}, std={vals.std():.2f}, "
              f"range {int(vals.min())}-{int(vals.max())}")


if __name__ == "__main__":
    main()
