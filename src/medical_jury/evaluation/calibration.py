"""Threshold calibration, reliability and risk-coverage analysis (DEV SPLIT ONLY for fitting).

Rows: {split, group_id, p_met, verdict, confidence, labels:[bool]}. Labels enter here only as a
metric target on dev; ``choose_threshold`` refuses any other split.
"""
from __future__ import annotations

from . import agreement as A

DEFAULT_GRID = [round(0.30 + 0.05 * i, 2) for i in range(9)]  # 0.30 .. 0.70


def _pairs(rows, threshold=None):
    out = []
    for r in rows:
        pred = (r['p_met'] >= threshold) if threshold is not None else (r['verdict'] == 'met')
        out.extend((bool(pred), l) for l in r['labels'])
    return out


def choose_threshold(rows: list[dict], grid: list[float] = DEFAULT_GRID, objective: str = 'cohen_kappa') -> dict:
    if any(r.get('split') != 'dev' for r in rows):
        raise ValueError('Thresholds may only be fitted on dev rows (PROTOCOL-V2 §4).')
    if objective not in {'cohen_kappa', 'gwet_ac1', 'macro_f1'}:
        raise ValueError('Unknown objective.')
    table = []
    for t in grid:
        s = A.binary_summary(_pairs(rows, t))
        table.append({'threshold': t, objective: s[objective], 'agreement': s['agreement'],
                      'predicted_met_rate': s['predicted_met_rate']})
    valid = [row for row in table if row[objective] is not None]
    best = max(valid, key=lambda row: (row[objective], -abs(row['threshold'] - 0.5))) if valid else None
    return {'objective': objective, 'chosen_threshold': best['threshold'] if best else None, 'grid': table,
            'note': 'Fitted on dev only; freeze before validation. Ties prefer thresholds closer to 0.5.'}


def reliability(rows: list[dict], bins: int = 5) -> list[dict]:
    """Stated confidence vs. observed share of physician labels agreeing with the verdict."""
    def index(conf):
        c = min(1.0, max(0.5, conf))
        return min(bins - 1, int((c - 0.5) / 0.5 * bins))
    edges = [0.5 + 0.5 * i / bins for i in range(bins + 1)]
    out = []
    for b, (lo, hi) in enumerate(zip(edges, edges[1:])):
        members = [r for r in rows if index(r['confidence']) == b]
        agree = [((r['verdict'] == 'met') == l) for r in members for l in r['labels']]
        out.append({'bin': [round(lo, 2), round(hi, 2)], 'n_items': len(members),
                    'mean_confidence': (sum(r['confidence'] for r in members) / len(members)) if members else None,
                    'observed_agreement': (sum(agree) / len(agree)) if agree else None})
    return out


def risk_coverage(rows: list[dict], steps: int = 10, threshold: float | None = None) -> list[dict]:
    """Auto-handle the most confident x% of items; route the rest to physicians."""
    ranked = sorted(rows, key=lambda r: -abs(r['p_met'] - 0.5))
    out = []
    for k in range(1, steps + 1):
        n = max(1, round(len(ranked) * k / steps))
        s = A.binary_summary(_pairs(ranked[:n], threshold))
        out.append({'coverage': round(n / len(ranked), 3) if ranked else 0, 'n_items': n,
                    'cohen_kappa': s['cohen_kappa'], 'gwet_ac1': s['gwet_ac1'], 'agreement': s['agreement']})
    return out
