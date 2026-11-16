"""Independent scorer for Jury v2 predictions (PROTOCOL-V2 §2-3).

Called only AFTER predictions are sealed. Inputs are plain records so the same scorer serves
dev runs, the one-shot validation run and the final test run.
"""
from __future__ import annotations

from collections import defaultdict

from . import agreement as A

COVERAGE_MAX_FAILURE_SHARE = 0.05
NON_INFERIORITY_MARGIN = 0.10


def _kappa(pairs):
    return A.cohen_kappa([p for p, _ in pairs], [g for _, g in pairs]) if pairs else None


def _delta(rows):
    jh = [(r['pred'], l) for r in rows for l in r['labels']]
    hh = A.pairwise_human_pairs([r['labels'] for r in rows])
    a, b = _kappa(jh), _kappa(hh)
    return None if a is None or b is None else a - b


def score_predictions(predictions: list[dict], references: dict[str, dict], *, n_boot: int = 2000,
                      seed: int = 0, threshold: float | None = None, alt_epsilon: float = 0.2,
                      alt_min_items: int = 30) -> dict:
    """``predictions``: [{example_id, status, verdict?, p_met?}], one per PLANNED item.
    ``references``: example_id -> {labels: [bool], physician_ids: [str], group_id, topic}.
    ``threshold``: if given (frozen from dev), predict met when p_met >= threshold; else use verdict.
    """
    planned = len(predictions)
    failed = [p['example_id'] for p in predictions if p.get('status') != 'Completed']
    rows = []
    for p in predictions:
        if p.get('status') != 'Completed':
            continue
        ref = references[p['example_id']]
        pred = (p['p_met'] >= threshold) if threshold is not None else (p['verdict'] == 'met')
        rows.append({'example_id': p['example_id'], 'pred': bool(pred), 'labels': list(ref['labels']),
                     'physician_ids': list(ref.get('physician_ids', [])), 'group_id': ref['group_id'],
                     'topic': ref.get('topic', 'other'), 'p_met': p.get('p_met')})
    failure_share = (len(failed) / planned) if planned else None
    coverage_ok = failure_share is not None and failure_share <= COVERAGE_MAX_FAILURE_SHARE

    jh_pairs = [(r['pred'], l) for r in rows for l in r['labels']]
    hh_pairs = A.pairwise_human_pairs([r['labels'] for r in rows])
    groups = defaultdict(list)
    for r in rows:
        groups[r['group_id']].append(r)
    delta = _delta(rows)
    delta_ci = A.group_bootstrap_ci(groups, _delta, n_boot, seed)
    majority_rows = [(r['pred'], A.majority_label(r['labels'])) for r in rows
                     if len(r['labels']) >= 3 and A.majority_label(r['labels']) is not None]
    loo_same_items = A.leave_one_out_majority_pairs([r['labels'] for r in rows if len(r['labels']) >= 3], 3)
    alt = A.alt_test([{'llm': r['pred'], 'humans': dict(zip(r['physician_ids'], r['labels']))}
                      for r in rows if len(r['physician_ids']) == len(r['labels'])],
                     epsilon=alt_epsilon, min_items_per_annotator=alt_min_items)
    baselines = A.constant_baselines(jh_pairs) if jh_pairs else None
    # Gate uses the GROUP-level permutation (criteria of one dialogue are correlated); item-level is reported too.
    perm = A.permutation_test_kappa([(r['pred'], r['labels']) for r in rows], 2000, seed,
                                    groups=[r['group_id'] for r in rows]) if jh_pairs else None
    perm_item = A.permutation_test_kappa([(r['pred'], r['labels']) for r in rows], 2000, seed) if jh_pairs else None
    by_topic = {}
    for topic in sorted({r['topic'] for r in rows}):
        sub = [r for r in rows if r['topic'] == topic]
        by_topic[topic] = {'n_items': len(sub),
                           'judge_vs_physician': A.binary_summary([(r['pred'], l) for r in sub for l in r['labels']]),
                           'physician_vs_physician_kappa': _kappa(A.pairwise_human_pairs([r['labels'] for r in sub]))}

    checks = {
        'coverage_ok': coverage_ok,
        'non_inferior': (delta_ci['lower'] is not None and delta_ci['lower'] >= -NON_INFERIORITY_MARGIN),
        'alt_test_passed': bool(alt['passed']),      # acceptance unchanged; see alt_test['status'] for why
        # Macro-F1, not raw agreement: with ~77% 'met', a real physician's agreement does not significantly
        # beat 'always met' (validated on HealthBench: 0.775 vs 0.752, McNemar p=0.25). McNemar stays reported.
        'beats_constant_baselines_macro_f1': bool(baselines and A.macro_f1(jh_pairs) is not None and all(
            A.macro_f1(jh_pairs) > (v['macro_f1'] or 0.0) for v in baselines.values())),
        'permutation_p_below_0_05': bool(perm and perm['p_value'] is not None and perm['p_value'] < 0.05),
    }
    primary = None
    if coverage_ok:
        primary = {'judge_vs_physician': A.binary_summary(jh_pairs),
                   'physician_vs_physician_same_items': A.binary_summary(hh_pairs),
                   'delta_kappa': delta, 'delta_kappa_group_bootstrap_95ci': delta_ci}
    return {
        'planned_items': planned, 'completed_items': len(rows), 'failed_or_not_run': len(failed),
        'failure_share': failure_share,
        'primary_metrics': primary,
        'primary_withheld_reason': None if primary else
            f'Failure share {failure_share} exceeds {COVERAGE_MAX_FAILURE_SHARE}; conditional diagnostics only.',
        'conditional_completed_metrics': {
            'judge_vs_physician': A.binary_summary(jh_pairs),
            'physician_vs_physician_same_items': A.binary_summary(hh_pairs),
            'delta_kappa': delta, 'delta_kappa_group_bootstrap_95ci': delta_ci},
        'secondary_vs_majority_min3': {'judge_vs_majority': A.binary_summary(majority_rows),
                                       'physician_loo_vs_majority_same_items': A.binary_summary(loo_same_items)},
        'alt_test': alt,
        'constant_baselines': baselines,
        'permutation_test': perm,
        'permutation_test_item_level': perm_item,
        'by_topic': by_topic,
        'acceptance_checks': checks,
        'accepted': all(checks.values()),
        'decision_source': 'threshold' if threshold is not None else 'verdict',
        'threshold': threshold,
        'limitations': ['Public benchmark; physicians are reference opinions, not ground truth.',
                        'Pretraining contamination of public data cannot be excluded.',
                        'Acceptance = non-inferior to inter-physician agreement on these items, not clinical validity.'],
    }
