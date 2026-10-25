"""Pure metric functions, independent of model input and transport."""
from __future__ import annotations

import random
from fractions import Fraction


def binary_metrics(pairs):
    if not pairs:
        return {'n': 0, 'agreement': None, 'macro_f1': None, 'cohen_kappa': None,
                'confusion': {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}}
    if any(type(p) is not bool or type(g) is not bool for p, g in pairs):
        raise ValueError('Binary labels must be original booleans.')
    tp = sum(p and g for p, g in pairs)
    tn = sum(not p and not g for p, g in pairs)
    fp = sum(p and not g for p, g in pairs)
    fn = sum(not p and g for p, g in pairs)
    n = len(pairs)
    a = Fraction(tp + tn, n)
    chance = Fraction((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp), n * n)
    fs = [Fraction(2 * z, 2 * z + fp + fn) if 2 * z + fp + fn else None for z in [tp, tn]]
    return {'n': n, 'agreement': float(a), 'macro_f1': float(sum(fs) / 2) if None not in fs else None,
            'cohen_kappa': float((a - chance) / (1 - chance)) if chance < 1 else None,
            'confusion': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}}


def score(items, arm, *, bootstrap=3000):
    groups = []
    planned = sum(len(i['labels']) for i in items)
    undecided, failed = 0, 0
    for item in items:
        p = item['arms'][arm]
        decision = p.get('decision')
        if p['status'] != 'Completed':
            failed += 1
            continue
        if decision not in {'met', 'not_met'}:
            undecided += 1
            continue
        groups.append((item['topic'], [(decision == 'met', label) for label in item['labels']]))
    pairs = [p for _, values in groups for p in values]
    metrics = binary_metrics(pairs)
    # Conditional diagnostics are NOT the all-planned primary performance when coverage is incomplete.
    primary = metrics if len(groups) == len(items) else None
    result = {'planned_cases': len(items), 'planned_physician_comparisons': planned,
              'decided_cases': len(groups), 'undetermined_cases': undecided, 'failed_or_not_run_cases': failed,
              'decision_coverage': len(groups) / len(items) if items else None,
              'primary_metrics': primary, 'conditional_decided_metrics': metrics,
              'primary_withheld_reason': None if primary else 'Coverage incomplete; do not cherry-pick decided cases.',
              'delivered_label_match_fraction': ((metrics['confusion']['TP'] + metrics['confusion']['TN']) / planned)
                     if planned else None,
              'delivered_match_warning': 'Engineering delivery diagnostic, NOT standard accuracy on missing predictions.'}
    if groups:
        strata = {}
        for t, p in groups:
            strata.setdefault(t, []).append(p)
        rng = random.Random(1520261025)
        draws = {k: [] for k in ['agreement', 'macro_f1', 'cohen_kappa']}
        for _ in range(bootstrap):
            chosen = [p for rows in strata.values() for values in rng.choices(rows, k=len(rows)) for p in values]
            m = binary_metrics(chosen)
            for k in draws:
                if m[k] is not None:
                    draws[k].append(m[k])
        result['conditional_group_stratified_bootstrap_95ci'] = {
            k: [sorted(v)[int((len(v) - 1) * .025)], sorted(v)[int((len(v) - 1) * .975)]] if v else None
            for k, v in draws.items()}
        result['ci_warning'] = 'Exploratory percentile intervals with only two groups/topic; missing groups remain excluded, so conditional.'
    return result
