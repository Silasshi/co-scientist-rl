#!/usr/bin/env python3
"""v8.1 analysis: did loosening the S4 GATE fix the over-triggering on refs?

Merges s4_regraded_v8_1.jsonl back into v8 results and recomputes
detection/aggregate.
"""

import json
import statistics
import sys
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from co_scientist.shared.ten_signal_reward import (
    SIGNALS, SIGNAL_WEIGHTS, aggregate_reward, normalize_score,
)

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
SIGS = [s.id for s in SIGNALS]
TARGET = {
    'P_S1_asserted': 'S1_depth', 'P_S2_strawman': 'S2_rigor',
    'P_S3_no_positioning': 'S3_positioning', 'P_S4_weak_problem': 'S4_significance',
    'P_S5_vague_method': 'S5_feasibility', 'P_S6_no_risk': 'S6_risk_awareness',
    'P_S7_no_specifics': 'S7_specificity', 'P_S8_overclaim': 'S8_scope',
    'P_S9_stacked': 'S9_focus',
}
# Frozen v8 weights for v8-vs-v8.1 comparison in S4 distribution section
V8_WEIGHTS = {
    "S1_depth": 0.10, "S2_rigor": 0.12, "S3_positioning": 0.10,
    "S4_significance": 0.13, "S5_feasibility": 0.08, "S6_risk_awareness": 0.11,
    "S7_specificity": 0.08, "S8_scope": 0.15, "S9_focus": 0.13,
}


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_merged():
    refs = {r['source_id']: r for r in _load_jsonl(BASE / 'data' / 'refs' / 'grading_refs_v8.jsonl')}
    perts = _load_jsonl(BASE / 'data' / 'perturbations' / 'grading_perturbations_v8.jsonl')
    s4_new = {r['id']: r for r in _load_jsonl(BASE / 'data' / 'perturbations' / 's4_regraded_v8_1.jsonl')}

    # Save original S4 for v8-vs-v8.1 comparison before mutating
    v8_s4_refs = {sid: r['signals_median'].get('S4_significance') for sid, r in refs.items()}

    # Override S4 in refs and perts, recompute aggregate with canonical weights
    for sid, r in refs.items():
        if sid in s4_new and s4_new[sid]['S4_median'] is not None:
            r['signals_median']['S4_significance'] = s4_new[sid]['S4_median']
            r['aggregate'] = aggregate_reward(r['signals_median'])

    for p in perts:
        pid = p['perturbation_id']
        if pid in s4_new and s4_new[pid]['S4_median'] is not None:
            p['signals_median']['S4_significance'] = s4_new[pid]['S4_median']
            p['aggregate'] = aggregate_reward(p['signals_median'])

    return refs, perts, v8_s4_refs


def main():
    refs, perts, v8_s4_orig = load_merged()
    print(f"Refs: {len(refs)}, Perturbations: {len(perts)}\n")

    # S4 distribution check — did the loosening fix ref over-triggering?
    print("=" * 75)
    print("S4 DISTRIBUTION: v8 vs v8.1")
    print("=" * 75)
    v8_s4_refs = [v for v in v8_s4_orig.values() if v is not None]
    v81_s4_refs = [r['signals_median']['S4_significance'] for r in refs.values()
                   if r['signals_median'].get('S4_significance') is not None]

    print(f"  Refs S4 mean:   v8={statistics.mean(v8_s4_refs):.2f}  v8.1={statistics.mean(v81_s4_refs):.2f}")
    print(f"  Refs S4 ≤ 2:    v8={sum(1 for s in v8_s4_refs if s<=2)}/{len(v8_s4_refs)}  "
          f"v8.1={sum(1 for s in v81_s4_refs if s<=2)}/{len(v81_s4_refs)}")

    # Detection analysis
    print("\n" + "=" * 75)
    print("DETECTION per perturbation type (v8.1)")
    print("=" * 75)
    by_type = defaultdict(list)
    for p in perts:
        by_type[p['perturbation_type']].append(p)

    # v8 baseline for comparison
    V8_DET = {'P_S1_asserted': 0.55, 'P_S2_strawman': 1.00,
              'P_S3_no_positioning': 0.61, 'P_S4_weak_problem': 0.83,
              'P_S5_vague_method': 0.83, 'P_S6_no_risk': 0.88,
              'P_S7_no_specifics': 0.93, 'P_S8_overclaim': 0.94, 'P_S9_stacked': 0.88}

    print(f"\n{'Perturbation':<25} {'target':<18} {'v8.1 det%':>10} {'v8 det%':>8} {'Δ':>6}")
    print('-' * 75)
    for ptype, target in TARGET.items():
        pert_list = by_type.get(ptype, [])
        if not pert_list: continue
        drops = []
        for p in pert_list:
            if p['base_id'] not in refs: continue
            r_s = refs[p['base_id']]['signals_median'].get(target)
            p_s = p['signals_median'].get(target)
            if r_s is None or p_s is None: continue
            drops.append(r_s - p_s)
        if not drops: continue
        det = sum(1 for d in drops if d >= 1) / len(drops)
        v8 = V8_DET.get(ptype)
        delta = f"{(det - v8)*100:+.0f}pp" if v8 else "---"
        v8_str = f"{v8:.0%}" if v8 else "n/a"
        print(f"  {ptype:<23} {target:<18} {det:>9.0%} {v8_str:>8} {delta:>6}")

    # Aggregate
    print("\n" + "=" * 75)
    print("AGGREGATE (v8.1)")
    print("=" * 75)
    ref_aggs = [r['aggregate'] for r in refs.values()]
    pert_aggs = [p['aggregate'] for p in perts]
    gap = statistics.mean(ref_aggs) - statistics.mean(pert_aggs)
    above = sum(1 for r in ref_aggs for p in pert_aggs if r > p)
    total = len(ref_aggs) * len(pert_aggs)
    auc = above / total
    sorted_refs = sorted(ref_aggs)
    p50 = statistics.median(ref_aggs)
    below_p50 = sum(1 for p in pert_aggs if p < p50)
    print(f"  Refs:  mean={statistics.mean(ref_aggs):.3f}  range {min(ref_aggs):.3f}-{max(ref_aggs):.3f}")
    print(f"  Perts: mean={statistics.mean(pert_aggs):.3f}  range {min(pert_aggs):.3f}-{max(pert_aggs):.3f}")
    print(f"  Gap: {gap:+.3f}  AUC: {auc:.3f}  Perts below ref P50: {below_p50}/{len(pert_aggs)} ({below_p50/len(pert_aggs):.0%})")
    print(f"  V8 BASELINE: gap=0.065, AUC=0.742, perts below P50=86%")

    # Opus correlation
    try:
        from scipy.stats import pearsonr
        opus = {r['source_id']: r for r in _load_jsonl(BASE / 'data' / 'human_ratings' / 'human_ratings_v2.jsonl')}
        xs, ys = [], []
        for sid, r in refs.items():
            if sid in opus:
                xs.append(r['aggregate'])
                ys.append(opus[sid]['overall'])
        r = pearsonr(xs, ys).statistic
        print(f"\n  Opus correlation (v8.1): r={r:.3f}  (v8: 0.079, v7: 0.158, v6: 0.395)")
    except Exception as e:
        print(f"  [Opus correlation unavailable: {e}]")


if __name__ == "__main__":
    main()
