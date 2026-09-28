#!/usr/bin/env python3
"""Test different aggregation formulas on existing data.

Which formula best separates refs from perturbations?
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

HERE = Path(__file__).parent
BASE = Path(__file__).resolve().parents[2]
SIGS = ['S1_depth','S2_rigor','S3_positioning','S4_significance',
        'S5_feasibility','S6_risk_awareness','S7_specificity','S8_scope','S9_focus']
WEIGHTS = {'S1_depth':0.10,'S2_rigor':0.12,'S3_positioning':0.10,'S4_significance':0.13,
           'S5_feasibility':0.08,'S6_risk_awareness':0.11,'S7_specificity':0.08,
           'S8_scope':0.15,'S9_focus':0.13}


def load():
    with open(BASE / 'archive' / 'grading_results_v5.jsonl') as f:
        refs = [json.loads(l) for l in f]
    with open(BASE / 'data' / 'perturbations' / 'grading_perturbations_v6.jsonl') as f:
        perts = [json.loads(l) for l in f]
    return refs, perts


def get_signals(e):
    """Get signal vector as dict."""
    if 'signals_median' in e:
        return e['signals_median']
    return e['signals']


def aggregates(entries, formula):
    """Apply formula to list of entries. Returns list of aggregate scores."""
    out = []
    for e in entries:
        sigs = get_signals(e)
        normed = {k: (v-1)/4.0 if v is not None else 0.5 for k, v in sigs.items()}
        if formula == 'weighted_mean':
            out.append(sum(WEIGHTS[s] * normed[s] for s in SIGS if s in normed))
        elif formula == 'geometric_mean':
            # product of s^w, need positive base
            prod = 1.0
            total_w = 0.0
            for s in SIGS:
                if s in normed:
                    val = max(normed[s], 0.01)  # avoid zero
                    prod *= val ** WEIGHTS[s]
                    total_w += WEIGHTS[s]
            out.append(prod ** (1/total_w) if total_w > 0 else 0)
        elif formula == 'min':
            out.append(min(normed[s] for s in SIGS if s in normed))
        elif formula == 'soft_min':
            # weighted min: aggregate = mean - alpha * (mean - min)
            # alpha=0.5 blends mean and min
            vals = [normed[s] for s in SIGS if s in normed]
            if not vals:
                out.append(0)
                continue
            m = sum(WEIGHTS[s] * normed[s] for s in SIGS if s in normed)
            mn = min(vals)
            out.append(0.5 * m + 0.5 * mn)
        elif formula == 'gated':
            # Hard gate: if any normed < 0.5 (score <= 3), penalize heavily
            vals = [normed[s] for s in SIGS if s in normed]
            base = sum(WEIGHTS[s] * normed[s] for s in SIGS if s in normed)
            if min(vals) < 0.5:  # any signal ≤ 3
                out.append(base * 0.5)
            else:
                out.append(base)
        elif formula == 'weakest_link':
            # Take min of 3 critical signals; average the rest
            # Critical = strongest discriminators from perturbation analysis
            critical = ['S5_feasibility', 'S6_risk_awareness', 'S8_scope']
            crit_min = min(normed[s] for s in critical if s in normed)
            other = [normed[s] for s in SIGS if s in normed and s not in critical]
            other_mean = sum(other) / len(other) if other else 0
            out.append(0.4 * crit_min + 0.6 * other_mean)
    return out


def evaluate(refs, perts, formula):
    """Evaluate formula: does it separate refs from perturbations?"""
    ref_aggs = aggregates(refs, formula)
    pert_aggs = aggregates(perts, formula)
    r_mean = sum(ref_aggs) / len(ref_aggs)
    p_mean = sum(pert_aggs) / len(pert_aggs)
    # Separation: how much perturbation below min ref?
    min_ref = min(ref_aggs)
    below = sum(1 for p in pert_aggs if p < min_ref)
    # AUC-like: how often a random ref > random perturbation?
    above = 0
    total = 0
    for r in ref_aggs:
        for p in pert_aggs:
            if r > p: above += 1
            total += 1
    auc = above / total if total > 0 else 0
    return {
        'formula': formula,
        'ref_mean': r_mean,
        'pert_mean': p_mean,
        'gap': r_mean - p_mean,
        'min_ref': min_ref,
        'max_pert': max(pert_aggs),
        'pert_below_min_ref': below,
        'pert_total': len(pert_aggs),
        'separation_pct': below / len(pert_aggs),
        'auc': auc,
    }


def main():
    refs, perts = load()
    print(f'Refs: {len(refs)}, Perturbations: {len(perts)}\n')
    print(f'{"Formula":<20} {"ref_μ":>6} {"pert_μ":>6} {"gap":>6} {"sep%":>6} {"AUC":>6}')
    print('-' * 60)
    formulas = ['weighted_mean', 'geometric_mean', 'min', 'soft_min', 'gated', 'weakest_link']
    results = []
    for f in formulas:
        r = evaluate(refs, perts, f)
        results.append(r)
        print(f'{f:<20} {r["ref_mean"]:>6.3f} {r["pert_mean"]:>6.3f} {r["gap"]:>6.3f} '
              f'{r["separation_pct"]:>5.0%} {r["auc"]:>6.3f}')
    print()
    print('Best by separation:', max(results, key=lambda x: x['separation_pct'])['formula'])
    print('Best by AUC:', max(results, key=lambda x: x['auc'])['formula'])


if __name__ == "__main__":
    main()
