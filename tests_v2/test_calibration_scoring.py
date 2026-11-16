import random

import pytest

from medical_jury.evaluation import calibration
from medical_jury.evaluation.scoring import score_predictions


def synthetic(n=300, judge_quality=0.9, seed=0, split='dev'):
    rng = random.Random(seed)
    rows, refs, preds = [], {}, []
    for i in range(n):
        truth = rng.random() < 0.75
        labels = [truth if rng.random() < 0.88 else not truth for _ in range(2 + (i % 7 == 0))]
        pids = [f'P{(i + k) % 6}' for k in range(len(labels))]
        verdict_true = truth if rng.random() < judge_quality else not truth
        conf = 0.55 + 0.4 * rng.random()
        verdict = 'met' if verdict_true else 'not_met'
        p_met = conf if verdict == 'met' else 1 - conf
        eid = f'E{i}'
        refs[eid] = {'labels': labels, 'physician_ids': pids, 'group_id': f'G{i // 2}', 'topic': 'hedging'}
        preds.append({'example_id': eid, 'status': 'Completed', 'verdict': verdict, 'confidence': conf, 'p_met': p_met})
        rows.append({'split': split, 'group_id': f'G{i // 2}', 'p_met': p_met, 'verdict': verdict,
                     'confidence': conf, 'labels': labels})
    return rows, refs, preds


def test_threshold_only_on_dev():
    rows, _, _ = synthetic(split='validation')
    with pytest.raises(ValueError, match='dev'):
        calibration.choose_threshold(rows)


def test_threshold_grid_and_choice():
    rows, _, _ = synthetic()
    out = calibration.choose_threshold(rows)
    assert out['chosen_threshold'] in calibration.DEFAULT_GRID and len(out['grid']) == 9


def test_reliability_bins_cover_all_items():
    rows, _, _ = synthetic()
    bins = calibration.reliability(rows, 5)
    assert sum(b['n_items'] for b in bins) == len(rows)


def test_risk_coverage_monotone_coverage():
    rows, _, _ = synthetic()
    rc = calibration.risk_coverage(rows, 5)
    assert [r['coverage'] for r in rc] == sorted(r['coverage'] for r in rc) and rc[-1]['coverage'] == 1.0


def test_good_judge_accepted_and_failures_withhold_primary():
    _, refs, preds = synthetic(n=600, judge_quality=0.95)
    good = score_predictions(preds, refs, n_boot=200, alt_min_items=30)
    assert good['acceptance_checks']['non_inferior'] and good['acceptance_checks']['coverage_ok']
    assert good['primary_metrics'] is not None
    broken = [dict(p, status='Failed') if i < 60 else p for i, p in enumerate(preds)]
    out = score_predictions(broken, refs, n_boot=50)
    assert out['primary_metrics'] is None and out['failed_or_not_run'] == 60 and out['accepted'] is False


def test_random_judge_rejected():
    _, refs, preds = synthetic(n=400, judge_quality=0.5)
    out = score_predictions(preds, refs, n_boot=200)
    assert out['accepted'] is False and out['acceptance_checks']['non_inferior'] is False


def test_threshold_decision_source():
    _, refs, preds = synthetic(n=100)
    out = score_predictions(preds, refs, n_boot=20, threshold=0.99)
    assert out['decision_source'] == 'threshold'
    assert out['conditional_completed_metrics']['judge_vs_physician']['predicted_met_rate'] < 0.1
