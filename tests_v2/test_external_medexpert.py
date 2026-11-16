import json

import pytest

from medical_jury.datasets import medexpert as MX
from medical_jury.evaluation.external import score_external

from v2_fixtures import load_script

SEV = ['Mild - no action is required', 'Moderate - may negatively impact the patients health if no action is taken',
       'Severe - may require medical intervention by a doctor']


def raw_rows(n=30, double=4):
    rows = []
    for i in range(n):
        base = {'domain': 'PC' if i % 2 else 'MH', 'base_question_id': f'q{i // 5}', 'model': f'm{i % 5}',
                'question': f'Question {i // 5}?', 'response': f'Answer {i}. Take 500 mg.', 'title': f't{i}',
                'annotator': f'A{i % 3}', 'omission_severity': None if i % 3 == 0 else SEV[i % 3],
                'highest_fact_severity': ['No Error', 'Mild', 'Moderate'][i % 3], 'iaa': i < double}
        rows.append(base)
        if i < double:
            rows.append({**base, 'annotator': 'A9', 'title': f't{i}-iaa', 'response': base['response'].replace('.', '. '),
                         'omission_severity': SEV[1], 'highest_fact_severity': 'No Error'})
    return rows


def test_build_two_criteria_per_pair_and_split_labels():
    public, private, manifest = MX.build(raw_rows())
    assert manifest['unique_pairs'] == 30 and manifest['items'] == 60 and manifest['double_annotated_pairs'] == 4
    assert manifest['iaa_copies_differing_only_in_punctuation'] == 4
    assert not any(k in p for p in public for k in ('labels', 'ordinal', 'physician_ids'))
    by = {r['example_id']: r for r in private}
    assert sum(len(r['labels']) == 2 for r in private) == 8
    for r in private:
        assert r['labels'] == [lv >= 2 for lv in r['ordinal']]
    assert {p['rubric']['rubric_id'] for p in public} == {'MEX-R-OMIT', 'MEX-R-FACT'}
    assert set(by) == {p['example_id'] for p in public}


def test_build_rejects_real_text_conflicts():
    rows = raw_rows(10, 2)
    rows[1]['response'] = 'Completely different text'
    with pytest.raises(ValueError, match='Inconsistent'):
        MX.build(rows)


def test_external_scorer_reports_per_criterion_and_ceiling():
    public, private, _ = MX.build(raw_rows())
    labels = {r['example_id']: r for r in private}
    preds = [{'example_id': p['example_id'], 'status': 'Completed',
              'verdict': 'met' if labels[p['example_id']]['labels'][0] else 'not_met'} for p in public]
    out = score_external(preds, labels, n_boot=50)
    assert set(out['by_criterion']) == {'OMIT', 'FACT'}
    omit = out['by_criterion']['OMIT']
    assert omit['completed'] == 30 and omit['double_annotated_subset']['items'] == 4
    assert omit['defect_sensitivity'] is not None


def _snapshot(root):
    # SQLite sidecar files (-shm/-wal) appear on any read-only open of the audit DB; they are not data writes.
    return sorted(str(p.relative_to(root)) for p in (root / 'data').rglob('*')
                  if p.is_file() and not p.name.startswith('jury.sqlite'))


def test_runner_offline_lock_and_paid_gate(tmp_path):
    raw = tmp_path / 'data/external/medexpert/medexpert-all.jsonl'
    raw.parent.mkdir(parents=True)
    raw.write_text('\n'.join(json.dumps(r) for r in raw_rows()) + '\n', encoding='utf-8')
    before = _snapshot(tmp_path)
    run = load_script('run_external_medexpert')
    assert run.main(['--root', str(tmp_path), '--judge', 'mock']) == 0
    with pytest.raises(SystemExit, match='one-shot'):
        run.main(['--root', str(tmp_path), '--judge', 'mock'])
    out = next((tmp_path / 'reports/external-runs').glob('medexpert-*/'))
    summary = json.loads((out / 'summary.json').read_text())
    assert summary['plan']['status'] == 'PRELIMINARY_NOT_FROZEN' and summary['mock_warning']
    assert summary['status_counts'] == {'Completed': 60}
    # nothing written under data/ except the audit database
    assert [p for p in _snapshot(tmp_path) if p not in before] == []
    # paid judge without --authorized: plan file only, no lock, no data writes
    (tmp_path / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'ds', 'kind': 'deepseek'}]}))
    snap = _snapshot(tmp_path)
    assert run.main(['--root', str(tmp_path), '--judge', 'ds']) == 0
    assert _snapshot(tmp_path) == snap
    assert list((tmp_path / 'reports/external-plans').glob('*/plan.json'))
    assert not list((tmp_path / 'reports/external-runs').glob('*prelim*ds*'))
    with pytest.raises(SystemExit, match='smoke'):
        run.main(['--root', str(tmp_path), '--judge', 'ds', '--limit', '5'])
