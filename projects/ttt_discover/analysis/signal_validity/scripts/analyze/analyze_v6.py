#!/usr/bin/env python3
"""Final analysis: signal validity via reference + perturbation data.

Answers:
1. Does each signal correctly detect its targeted perturbation? (detection accuracy)
2. Halo effects: does perturbing S3 also drop S1, S9, etc.?
3. Correlation: with full spread (refs + perturbs), what's correlation with Opus?
4. Occam's razor: which signals are necessary? which are redundant?
"""

import json
import statistics
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
SIGS = ['S1_depth','S2_rigor','S3_positioning','S4_significance',
        'S5_feasibility','S6_risk_awareness','S7_specificity','S8_scope','S9_focus']

# Map perturbation type to target signal
TARGET = {
    'P_S3_no_positioning': 'S3_positioning',
    'P_S4_weak_problem': 'S4_significance',
    'P_S5_vague_method': 'S5_feasibility',
    'P_S6_no_risk': 'S6_risk_awareness',
    'P_S7_no_specifics': 'S7_specificity',
    'P_S8_overclaim': 'S8_scope',
    'P_S9_stacked': 'S9_focus',
}

def load():
    with open(BASE / 'archive' / 'grading_results_v5.jsonl') as f:
        refs = {json.loads(l)['source_id']: json.loads(l) for l in f}
    if not (BASE / 'data' / 'perturbations' / 'grading_perturbations_v6.jsonl').exists():
        print("grading_perturbations.jsonl not found — grade perturbations first")
        return refs, []
    with open(BASE / 'data' / 'perturbations' / 'grading_perturbations_v6.jsonl') as f:
        perturbs = [json.loads(l) for l in f]
    return refs, perturbs

def main():
    refs, perturbs = load()
    if not perturbs:
        return

    print(f"References: {len(refs)}, Perturbations: {len(perturbs)}")
    print()

    # === Question 1: Does each signal detect its targeted perturbation? ===
    print("=" * 70)
    print("Q1: DETECTION ACCURACY — does each signal detect its perturbation?")
    print("=" * 70)
    by_type = defaultdict(list)
    for p in perturbs:
        by_type[p['perturbation_type']].append(p)

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
            # Halo: did OTHER signals also drop?
            for sid in SIGS:
                r = refs[p['base_id']]['signals_median'].get(sid)
                q = p['signals_median'].get(sid)
                if r is not None and q is not None:
                    all_signal_drops[sid].append(r - q)

        if not target_drops:
            continue
        print(f"\n{ptype} (target: {target}):")
        print(f"  Target signal drop: mean={statistics.mean(target_drops):+.2f}, n={len(target_drops)}")
        print(f"  Detected (drop ≥ 1): {sum(1 for d in target_drops if d >= 1)}/{len(target_drops)}")
        print(f"  Halo on OTHER signals:")
        for sid in SIGS:
            if sid == target: continue
            drops = all_signal_drops.get(sid, [])
            if drops:
                m = statistics.mean(drops)
                if abs(m) > 0.3:
                    marker = '⚠' if m > 0.3 else ''
                    print(f"    {sid:<22} {m:+.2f} {marker}")

    # === Question 2: Aggregate detection — do perturbations lower overall score? ===
    print("\n" + "=" * 70)
    print("Q2: AGGREGATE SCORE — refs vs perturbations")
    print("=" * 70)
    ref_aggs = [r['aggregate'] for r in refs.values()]
    pert_aggs = [p['aggregate'] for p in perturbs]
    print(f"  Refs: n={len(ref_aggs)}, mean={statistics.mean(ref_aggs):.3f}, range {min(ref_aggs):.3f}-{max(ref_aggs):.3f}")
    print(f"  Perts: n={len(pert_aggs)}, mean={statistics.mean(pert_aggs):.3f}, range {min(pert_aggs):.3f}-{max(pert_aggs):.3f}")
    # Separation
    overlap = sum(1 for p in pert_aggs if p >= min(ref_aggs))
    print(f"  Perturbations above min ref score: {overlap}/{len(pert_aggs)} ({overlap/len(pert_aggs):.0%})")

    # === Question 3: Correlation analysis with Opus ===
    print("\n" + "=" * 70)
    print("Q3: CORRELATION with OPUS ratings (combined refs + perturbations)")
    print("=" * 70)
    opus_path = BASE / "data" / "human_ratings" / "human_ratings_v2.jsonl"
    if not opus_path.exists():
        print("  No Opus ratings available")
    else:
        with open(opus_path) as f:
            opus = {json.loads(l)['source_id']: json.loads(l) for l in f}
        # On refs only
        xs, ys = [], []
        for sid, r in refs.items():
            if sid in opus:
                xs.append(r['aggregate'])
                ys.append(opus[sid]['overall'])
        r_refs = pearsonr(xs, ys).statistic
        print(f"  Refs only: r={r_refs:.3f} (n={len(xs)})")

        # For perturbations, we don't have Opus ratings — would need to generate
        print(f"  Perturbations: (no Opus ratings available yet — generate separately if needed)")

    # === Question 4: Signal redundancy (inter-signal correlation) ===
    print("\n" + "=" * 70)
    print("Q4: INTER-SIGNAL CORRELATION (Occam's razor)")
    print("=" * 70)
    # Use combined data (refs + perturbs) for spread
    all_entries = list(refs.values()) + perturbs
    X = []
    for e in all_entries:
        sig = e['signals_median']
        X.append([sig.get(sid) if sig.get(sid) is not None else 3 for sid in SIGS])
    X = np.array(X, dtype=float)
    print("\nCorrelation matrix (combined n={}):".format(len(X)))
    print(" " * 22 + "  ".join(f"{s[:3]}" for s in SIGS))
    for i, si in enumerate(SIGS):
        row = [f"{pearsonr(X[:,i], X[:,j]).statistic:+.2f}" for j in range(len(SIGS))]
        print(f"  {si:<22} " + " ".join(row))

    # Redundancy: pairs with |r| > 0.7
    print("\nHighly redundant pairs (|r| > 0.7):")
    redundant = []
    for i in range(len(SIGS)):
        for j in range(i+1, len(SIGS)):
            r = pearsonr(X[:,i], X[:,j]).statistic
            if abs(r) > 0.7:
                redundant.append((SIGS[i], SIGS[j], r))
                print(f"  {SIGS[i]} + {SIGS[j]}: r={r:+.3f}")
    if not redundant:
        print("  (none)")

    # === Question 5: Signal discriminative power ===
    print("\n" + "=" * 70)
    print("Q5: SIGNAL DISCRIMINATIVE POWER (variance across all entries)")
    print("=" * 70)
    for i, sid in enumerate(SIGS):
        vals = X[:, i]
        print(f"  {sid:<22} mean={vals.mean():.2f}, std={vals.std():.2f}, range {int(vals.min())}-{int(vals.max())}")

if __name__ == "__main__":
    main()
