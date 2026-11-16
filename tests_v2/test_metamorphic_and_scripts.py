import json
import random
from pathlib import Path

import pytest

from medical_jury.evaluation import metamorphic as M

from v2_fixtures import load_script, make_project

def test_dose_scaling():
    assert M.scale_first_dose('Take 500 mg twice daily, max 4 g.') == 'Take 5000 mg twice daily, max 4 g.'
    assert M.scale_first_dose('Use 2.5 mL') == 'Use 25 mL'
    assert M.scale_first_dose('No numbers here') is None


def test_remove_urgent_advice():
    text = 'Rest at home. Call 995 or go to the emergency department if you faint. Drink water.'
    out = M.remove_urgent_advice(text)
    assert '995' not in out and 'Rest at home.' in out and 'Drink water.' in out
    assert M.remove_urgent_advice('Rest and drink water.') is None


def test_invariant_edits_preserve_original_text():
    text = 'A.\n\nB.\n\nC.\n\nD.'
    assert M.append_disclaimer(text).startswith(text)
    assert M.pad_length(text).startswith(text)
    assert M.swap_unordered_paragraphs(text) == 'A.\n\nC.\n\nB.\n\nD.'
    assert M.swap_unordered_paragraphs('1. a\n\n2. b\n\n3. c\n\n4. d') is None


def test_false_claim_inserted():
    out = M.insert_false_claim('One.\n\nTwo.', random.Random(1))
    assert any(c in out for c in M.FALSE_CLAIMS)


def test_build_suite_dev_only_and_flagged(tmp_path):
    root = make_project(tmp_path)
    rows = [json.loads(l) for l in (root / 'data/prepared/dev.jsonl').read_text().splitlines()]
    suite = M.build_suite(rows, 3, 5)
    assert suite and all(c.synthetic and c.perturbed.source == M.SYNTHETIC_SOURCE for c in suite)
    assert {c.kind for c in suite} >= {'dose_x10', 'false_claim', 'remove_urgent', 'disclaimer', 'length_padding'}
    bad = [dict(rows[0], case={**rows[0]['case'], 'split': 'validation'})]
    with pytest.raises(ValueError):
        M.build_suite(bad, 1, 1)


def test_relation_evaluation_preconditions():
    res = M.evaluate_relations([
        {'kind': 'dose_x10', 'relation': 'must_become_met', 'original_verdict': 'not_met', 'perturbed_verdict': 'met'},
        {'kind': 'dose_x10', 'relation': 'must_become_met', 'original_verdict': 'met', 'perturbed_verdict': 'met'},
        {'kind': 'remove_urgent', 'relation': 'must_flip_met_to_not_met', 'original_verdict': 'met', 'perturbed_verdict': 'met'},
        {'kind': 'disclaimer', 'relation': 'invariant', 'original_verdict': 'met', 'perturbed_verdict': 'not_met'},
        {'kind': 'disclaimer', 'relation': 'invariant', 'original_verdict': None, 'perturbed_verdict': 'met'},
    ])['by_kind']
    assert res['dose_x10']['detection_rate'] == 1.0 and res['dose_x10']['skipped_precondition'] == 1
    assert res['remove_urgent']['detection_rate'] == 0.0
    assert res['disclaimer']['flip_rate'] == 1.0 and res['disclaimer']['failed_calls'] == 1


def test_self_consistency():
    out = M.self_consistency([['met', 'met'], ['met', 'not_met'], [None, 'met']])
    assert out['fully_consistent_share'] == 0.5 and out['incomplete_items'] == 1


def test_run_dev_v2_offline_end_to_end_and_error_export(tmp_path):
    root = make_project(tmp_path)
    run = load_script('run_dev_v2')
    out = root / 'reports/dev-test'
    assert run.main(['--root', str(root), '--n-groups', '40', '--metamorphic-per-kind', '3',
                     '--repeats', '2', '--out', str(out)]) == 0
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['status'] == 'OFFLINE_MOCK_PIPELINE_CHECK_NOT_AGREEMENT'
    assert summary['scored_with_verdict']['planned_items'] == 40
    assert summary['metamorphic'] and summary['self_consistency']['items'] == 20
    export = load_script('export_error_review')
    assert export.main([str(out), '--root', str(root), '--limit', '5']) == 0
    text = (out / 'error-review.csv').read_text(encoding='utf-8-sig')
    assert 'error_type' in text.splitlines()[0]


def test_run_dev_v2_paid_requires_authorization(tmp_path):
    root = make_project(tmp_path)
    run = load_script('run_dev_v2')
    out = root / 'reports/plan-only'
    assert run.main(['--root', str(root), '--backend', 'deepseek', '--n-groups', '10', '--out', str(out)]) == 0
    assert (out / 'plan.json').exists() and not (out / 'predictions.jsonl').exists()


def test_error_export_refuses_non_dev(tmp_path):
    root = make_project(tmp_path)
    run_dir = root / 'reports/fake'
    run_dir.mkdir(parents=True)
    (run_dir / 'plan.json').write_text('{}')
    (run_dir / 'sample-manifest.json').write_text(json.dumps({'split': 'validation'}))
    (run_dir / 'predictions.jsonl').write_text('')
    with pytest.raises(SystemExit):
        load_script('export_error_review').main([str(run_dir), '--root', str(root)])
