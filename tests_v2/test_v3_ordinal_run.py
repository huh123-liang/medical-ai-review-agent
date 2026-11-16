"""Step 5 (offline part): ordinal run harness and v3 freeze. Synthetic data, mock / scripted judges only."""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
import shutil
from pathlib import Path

import pytest

from medical_jury.evaluation import ordinal_run as R
from medical_jury.judge_v2 import BackendError, Completion
from medical_jury.v3.evaluator import mock_responder_v3

from v2_fixtures import load_script

REPO = Path(__file__).resolve().parents[1]
SEV = ['Mild', 'Moderate', 'Severe', 'Life-threatening']
LABELS = {'Scientific_Consensus': 'Aligned with consensus', 'Inappropriate_Content': 'No',
          'Missing_Content': 'Yes little clinical significance', 'Extent_of_Harm': 'No harm',
          'Likelihood_of_Harm': 'Low', 'Bias': 'No', 'Empathy': 'Moderate empathy', 'Grammaticality': 'Yes'}


def _csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def make_root(tmp_path) -> Path:
    root = tmp_path / 'proj'
    shutil.copytree(REPO / 'kb' / 'v3', root / 'kb' / 'v3')
    (root / 'docs' / 'project').mkdir(parents=True)
    shutil.copy(REPO / 'docs/project/PROTOCOL-V3-ORDINAL.md', root / 'docs/project/PROTOCOL-V3-ORDINAL.md')
    rows = []
    for i in range(12):
        for m in ('m1', 'm2'):
            om = {'Mild': 'Mild - no action is required', 'Severe': 'Severe - may require medical intervention by a doctor'}
            rows.append({'domain': 'd', 'base_question_id': f'q{i}', 'model': m, 'annotator': 'A', 'title': f'q{i}-{m}',
                         'question': f'Question {i}?', 'response': f'Answer {i} {m}.',
                         'omission_severity': om['Mild'] if i % 2 else None, 'highest_fact_severity': 'No Error'})
    mx = root / 'mx.jsonl'
    mx.write_text('\n'.join(json.dumps(r) for r in rows))
    repo = root / 'bionlp'
    qs = [[f'Q{k}', f'Patient question number {k} about headaches?', '', '', '', '', ''] for k in range(30)]
    _csv(repo / 'output/rephrase_results/combined_rephrased_TREC_temp_0.1.csv',
         ['questionid', 'original', 'v1', 'v2', 'v3', 'v4', 'v5'], qs)
    _csv(repo / 'output/answer_results/TREC_M_temp_0.1.csv', ['M_answer_original'], [[f'Answer {k}.'] for k in range(30)])
    _csv(repo / 'data/annotated_TRECLiveQA.csv', ['generated_answer', *LABELS, 'unique_id', 'annotator', 'model_name'],
         [[f'Answer {k}.', *LABELS.values(), str(k), 'ABC'[k % 3], 'M'] for k in range(30)])
    (repo / '.git').mkdir()
    from medical_jury.datasets.ordinal_v3 import BIONLP_COMMIT
    (repo / '.git/HEAD').write_text(BIONLP_COMMIT)
    assert load_script('prepare_ordinal_v3').main(['--root', str(root), '--medexpert', str(mx),
                                                    '--bionlp-repo', str(repo)]) == 0
    return root


def runner():
    return load_script('run_ordinal_v3')


def test_subset_takes_whole_groups_reproducibly():
    rows = [{'item_id': f'{g}-{k}', 'group_id': g} for g in 'abcdefgh' for k in range(3)]
    a, b = R.subset(rows, 7, 's'), R.subset(list(reversed(rows)), 7, 's')
    assert {r['item_id'] for r in a} == {r['item_id'] for r in b} and len(a) == 9
    assert all(sum(r['group_id'] == g for r in a) in (0, 3) for g in 'abcdefgh')


def test_inputs_with_label_fields_are_refused(tmp_path):
    root = make_root(tmp_path)
    path = root / 'data/prepared-v3/bionlp-trec/dev.jsonl'
    rows = path.read_text().splitlines()
    bad = json.loads(rows[0])
    bad['scores'] = {'empathy': [2]}
    path.write_text('\n'.join([json.dumps(bad)] + rows[1:]))
    with pytest.raises(ValueError, match='labels must never be in inputs'):
        R.load_items(root, 'bionlp-trec', 'dev')


def test_dev_run_seals_predictions_before_metrics(tmp_path, capsys):
    root = make_root(tmp_path)
    assert runner().main(['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'dev', '--judge', 'mock',
                          '--bootstrap', '50']) == 0
    out = next((root / 'reports').glob('ordinal-v3-bionlp-trec-dev-*'))
    seal = json.loads((out / 'seal.json').read_text())
    assert seal['predictions_sha256'] == hashlib.sha256((out / 'predictions.jsonl').read_bytes()).hexdigest()
    preds = [json.loads(l) for l in (out / 'predictions.jsonl').read_text().splitlines()]
    assert preds and all('scores' not in p and 'raters' not in p for p in preds)
    metrics = json.loads((out / 'metrics.json').read_text())
    assert metrics['label'].startswith('DEV') and metrics['coverage_ok'] is True
    assert set(metrics['primary_dimensions']) == {'missing_content', 'inappropriate_content', 'harm_extent', 'empathy'}
    joined = [json.loads(l) for l in (out / 'joined.jsonl').read_text().splitlines()]
    assert all('scores' in j for j in joined)
    assert (out / 'seal.json').stat().st_mtime_ns <= (out / 'metrics.json').stat().st_mtime_ns


def test_held_out_runs_need_a_matching_freeze_and_run_once(tmp_path, capsys):
    root = make_root(tmp_path)
    base = ['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'validation', '--judge', 'mock',
            '--bootstrap', '20']
    with pytest.raises(SystemExit, match='need --freeze'):
        runner().main(base)
    with pytest.raises(SystemExit, match='dev only'):
        runner().main(base + ['--n', '5'])
    assert load_script('freeze_v3').main(['--root', str(root), '--judge', 'mock', '--call-mode', 'single']) == 0
    freeze = next((root / 'reports/protocol-freeze-v3').glob('*/freeze.json'))
    with pytest.raises(SystemExit, match='call_mode'):
        runner().main(base + ['--freeze', str(freeze), '--call-mode', 'per_dimension'])
    assert runner().main(base + ['--freeze', str(freeze)]) == 0
    with pytest.raises(SystemExit, match='only once'):
        runner().main(base + ['--freeze', str(freeze)])
    m = json.loads(next((root / 'reports').glob('ordinal-v3-bionlp-trec-validation-*/metrics.json')).read_text())
    assert m['freeze']['freeze_sha256'] and m['label'].startswith('HELD-OUT')


def test_freeze_detects_a_changed_profile(tmp_path):
    root = make_root(tmp_path)
    assert load_script('freeze_v3').main(['--root', str(root), '--judge', 'mock', '--call-mode', 'single']) == 0
    freeze = next((root / 'reports/protocol-freeze-v3').glob('*/freeze.json'))
    prof = root / 'kb/v3/profile-medexpert-severity-5level.json'
    prof.write_text(prof.read_text().replace('Severity of the most serious', 'Severity of the worst'))
    with pytest.raises(SystemExit, match='profiles_sha256'):
        runner().main(['--root', str(root), '--dataset', 'medexpert', '--split', 'test', '--judge', 'mock',
                       '--freeze', str(freeze)])


def test_unknown_split_for_a_dataset_is_refused(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(ValueError, match='no .dev. split'):
        runner().main(['--root', str(root), '--dataset', 'medexpert', '--split', 'dev', '--judge', 'mock'])


def test_immediate_stop_marks_the_rest_not_run_and_withholds_primary(tmp_path, capsys, monkeypatch):
    root = make_root(tmp_path)
    mod = runner()
    rr = importlib.import_module('run_review_v3')

    class Dies:
        name, model, billable = 'dies', 'SYNTHETIC', False

        def __init__(self):
            self.calls = 0

        def complete(self, messages, max_tokens, job_id):
            self.calls += 1
            if self.calls > 2:
                raise BackendError('PROVIDER_HTTP_401')
            return Completion(mock_responder_v3(messages), 'stop', {}, 'SYNTHETIC')
    backend = Dies()
    monkeypatch.setattr(rr, 'make_backend', lambda *a, **k: (backend, False, None))
    assert mod.main(['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'dev', '--judge', 'mock',
                     '--threads', '1', '--bootstrap', '20']) == 0
    m = json.loads(next((root / 'reports').glob('ordinal-v3-bionlp-trec-dev-*/metrics.json')).read_text())
    assert m['stop']['stop_reason'] == 'auth_or_quota' and m['coverage_ok'] is False
    assert m['mean_primary_qwk'] is None and backend.calls == 3
    assert any(k.startswith('NotRun') for k in m['status_counts'])


def test_paid_plan_only_writes_nothing_and_counts_calls(tmp_path, capsys, monkeypatch):
    root = make_root(tmp_path)
    capsys.readouterr()                       # drop the preparation summary
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'deepseek-pro', 'kind': 'deepseek',
                                                               'model': 'deepseek-v4-pro', 'price_in': 13.2,
                                                               'price_out': 39.6}]}))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('JURY_PRICING_VALID_UNTIL', '2099-01-01')
    base = ['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'dev', '--judge', 'deepseek-pro']
    assert runner().main(base + ['--n', '6']) == 0
    out = capsys.readouterr().out
    single = json.loads(out[:out.index('Plan only')])
    assert runner().main(base + ['--n', '6', '--call-mode', 'per_dimension']) == 0
    out = capsys.readouterr().out
    per = json.loads(out[:out.index('Plan only')])
    assert single['planned_calls'] == single['items'] and per['planned_calls'] == 5 * per['items']
    assert single['max_output_tokens'] == 2900 and per['max_output_tokens'] == 1200      # 5 x 500 + 400
    assert single['cost_upper_bound_cny'] > 0 and not (root / 'reports').exists()


def test_held_out_parts_run_once_each_and_combine(tmp_path, capsys):
    root = make_root(tmp_path)
    assert load_script('freeze_v3').main(['--root', str(root), '--judge', 'mock', '--call-mode', 'single']) == 0
    freeze = str(next((root / 'reports/protocol-freeze-v3').glob('*/freeze.json')))
    base = ['--root', str(root), '--dataset', 'medexpert', '--split', 'test', '--judge', 'mock', '--freeze', freeze,
            '--bootstrap', '20']
    sizes = [len(R.part_of(R.load_items(root, 'medexpert', 'test'), 'medexpert', 'test', k, 3)) for k in (1, 2, 3)]
    assert sum(sizes) == len(R.load_items(root, 'medexpert', 'test')) and all(sizes)
    for k in (1, 2):
        assert runner().main(base + ['--parts', '3', '--part', str(k)]) == 0
    with pytest.raises(SystemExit, match='only once'):
        runner().main(base + ['--parts', '3', '--part', '1'])
    with pytest.raises(SystemExit, match='re-partitioned'):
        runner().main(base + ['--parts', '2', '--part', '2'])
    runs = sorted((root / 'reports').glob('ordinal-v3-medexpert-test-part*'))
    agg = load_script('aggregate_ordinal_v3')
    with pytest.raises(SystemExit, match='exactly once'):
        agg.main(['--root', str(root), '--runs', *map(str, runs)])
    assert runner().main(base + ['--parts', '3', '--part', '3']) == 0
    runs = sorted((root / 'reports').glob('ordinal-v3-medexpert-test-part*'))
    preds = runs[0] / 'predictions.jsonl'
    original = preds.read_bytes()
    preds.write_bytes(original.replace(b'"Completed"', b'"Completed" ', 1))
    with pytest.raises(SystemExit, match='seal'):
        agg.main(['--root', str(root), '--runs', *map(str, runs)])
    preds.write_bytes(original)
    assert agg.main(['--root', str(root), '--runs', *map(str, runs)]) == 0
    m = json.loads(next((root / 'reports').glob('ordinal-v3-medexpert-test-combined-*/metrics.json')).read_text())
    assert m['planned_items'] == sum(sizes) and len(m['parts']) == 3


def test_parts_are_refused_on_dev(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(SystemExit, match='--parts is for validation/test'):
        runner().main(['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'dev', '--judge', 'mock',
                       '--parts', '2'])
