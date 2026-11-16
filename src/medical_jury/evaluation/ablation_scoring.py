"""Scoring for the step-6 rubric-selection ablation (PROTOCOL-V3-RAG-ABLATION section 4).

Called only after every evaluator part is sealed. Input rows, one per sampled example:
    {'example_id', 'group_id', 'topic', 'verdicts': {arm: 'met' | 'not_met' | None},
     'rag': {'router_status', 'selected', 'gold', 'assessed'}, 'labels': [bool], 'physician_ids': [...]}
A verdict of None means the evaluator call failed; that example is left out of that arm (and of any paired
comparison involving the arm) and counts against the arm's coverage.
"""
from __future__ import annotations

from collections import defaultdict

from . import agreement as A

COVERAGE_MAX_FAILURE_SHARE = 0.05
PRIMARY = ('rag', 'single')
SECONDARY = (('oracle', 'rag'), ('oracle', 'single'))


def _kappa(pairs):
    return A.cohen_kappa([p for p, _ in pairs], [g for _, g in pairs]) if pairs else None


def _jh(rows, arm):
    return [(r['verdicts'][arm] == 'met', l) for r in rows for l in r['labels']]


def _delta_vs_physicians(arm):
    def fn(rows):
        a, b = _kappa(_jh(rows, arm)), _kappa(A.pairwise_human_pairs([r['labels'] for r in rows]))
        return None if a is None or b is None else a - b
    return fn


def _paired_delta(arm_b, arm_a):
    def fn(rows):
        x, y = _kappa(_jh(rows, arm_a)), _kappa(_jh(rows, arm_b))
        return None if x is None or y is None else y - x
    return fn


def _groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[r['group_id']].append(r)
    return out


def _arm_rows(rows, *arms):
    return [r for r in rows if all(r['verdicts'].get(a) is not None for a in arms)]


def _unanimous_correct(rows, arm):
    return [r['verdicts'][arm] == ('met' if r['labels'][0] else 'not_met') for r in rows]


def score_ablation(rows: list[dict], arms=('single', 'rag', 'oracle'), n_boot: int = 2000, seed: int = 0) -> dict:
    planned = len(rows)
    per_arm = {}
    for arm in arms:
        done = _arm_rows(rows, arm)
        failed = planned - len(done)
        share = failed / planned if planned else None
        per_arm[arm] = {
            'completed': len(done), 'failed': failed, 'failure_share': share,
            'coverage_ok': share is not None and share <= COVERAGE_MAX_FAILURE_SHARE,
            'judge_vs_physician': A.binary_summary(_jh(done, arm)),
            'physician_vs_physician_kappa_same_items': _kappa(A.pairwise_human_pairs([r['labels'] for r in done])),
            'delta_kappa_vs_physicians': _delta_vs_physicians(arm)(done),
            'delta_kappa_vs_physicians_ci95': A.group_bootstrap_ci(_groups(done), _delta_vs_physicians(arm), n_boot, seed),
        }
    comparisons = {}
    for b, a in (PRIMARY, *SECONDARY):
        if a not in arms or b not in arms:
            continue
        both = _arm_rows(rows, a, b)
        unanimous = [r for r in both if len(set(r['labels'])) == 1]
        comparisons[f'{b}_minus_{a}'] = {
            'items': len(both), 'kappa_a': _kappa(_jh(both, a)), 'kappa_b': _kappa(_jh(both, b)),
            'delta_kappa': _paired_delta(b, a)(both),
            'delta_kappa_ci95': A.group_bootstrap_ci(_groups(both), _paired_delta(b, a), n_boot, seed),
            'verdicts_differ': sum(r['verdicts'][a] != r['verdicts'][b] for r in both),
            'mcnemar_unanimous_items': {'items': len(unanimous), **A.exact_mcnemar(
                _unanimous_correct(unanimous, a), _unanimous_correct(unanimous, b))},
        }
    rag = [r['rag'] for r in rows]
    routed = [x for x in rag if x['router_status'] == 'Completed']
    router = {
        'planned': planned, 'completed': len(routed), 'failed_or_not_run': planned - len(routed),
        'gold_selected': sum(x['selected'] == x['gold'] for x in routed),
        'accuracy_on_completed': (sum(x['selected'] == x['gold'] for x in routed) / len(routed)) if routed else None,
        'none_selected': sum(x['selected'] is None for x in routed),
        'rag_assessed_share': sum(bool(x['assessed']) for x in rag) / planned if planned else None,
    }
    by_topic = {}
    for topic in sorted({r['topic'] for r in rows}):
        sub = [r for r in rows if r['topic'] == topic]
        by_topic[topic] = {'items': len(sub),
                           'router_accuracy': (sum(r['rag']['selected'] == r['rag']['gold'] for r in sub
                                                   if r['rag']['router_status'] == 'Completed')
                                               / max(1, sum(r['rag']['router_status'] == 'Completed' for r in sub))),
                           **{f'kappa_{arm}': _kappa(_jh(_arm_rows(sub, arm), arm)) for arm in arms}}
    primary = comparisons.get(f'{PRIMARY[0]}_minus_{PRIMARY[1]}')
    ci = (primary or {}).get('delta_kappa_ci95') or {}
    if not all(per_arm[a]['coverage_ok'] for a in PRIMARY if a in per_arm):
        reading = 'coverage rule failed: conditional diagnostics only'
    elif ci.get('lower') is not None and ci['lower'] > 0:
        reading = 'rag better than single (95% CI above 0)'
    elif ci.get('upper') is not None and ci['upper'] < 0:
        reading = 'rag worse than single (95% CI below 0)'
    else:
        reading = 'no evidence of a difference (95% CI includes 0)'
    return {'planned_items': planned, 'arms': per_arm, 'comparisons': comparisons, 'router': router,
            'by_topic': by_topic, 'primary': f'{PRIMARY[0]}_minus_{PRIMARY[1]}', 'primary_reading': reading}
