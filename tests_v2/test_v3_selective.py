"""Plan phase 1: selective evaluation with a finite-sample guarantee. Synthetic data, mock judges only."""
from __future__ import annotations

import json
import random

import pytest

from medical_jury.evaluation import selective as S
from medical_jury.judge_v2 import ScriptedBackend
from medical_jury.v3.used_groups import used_dev_groups_v3

from test_v3_rag_ablation import make_root
from v2_fixtures import load_script


def test_clopper_pearson_matches_known_values():
    assert S.clopper_pearson_upper(0, 100, 0.05) == pytest.approx(1 - 0.05 ** (1 / 100), abs=1e-6)
    assert S.clopper_pearson_upper(3, 132, 0.10) == pytest.approx(0.0498, abs=5e-4)
    assert S.clopper_pearson_upper(5, 5, 0.1) == 1.0 and S.clopper_pearson_upper(0, 0, 0.1) == 1.0
    assert S.clopper_pearson_upper(2, 50, 0.1) > 2 / 50


def test_error_definitions():
    assert S.consensus_error(True, [False, False]) == 1 and S.consensus_error(True, [True, False]) == 0
    assert S.pairwise_error(True, [True, False]) == 0.5


def test_vote_confidence_orders_unanimous_above_split_and_counts_failures():
    full = S.vote([{'verdict': 'met', 'confidence': 0.9}] * 3)
    split = S.vote([{'verdict': 'met', 'confidence': 0.9}, {'verdict': 'met', 'confidence': 0.9},
                    {'verdict': 'not_met', 'confidence': 0.95}])
    failed = S.vote([{'verdict': 'met', 'confidence': 0.9}, {'verdict': 'met', 'confidence': 0.9}, {'verdict': None}])
    assert full['confidence'] > split['confidence'] and full['verdict'] == split['verdict'] == 'met'
    assert failed['vote_share'] == pytest.approx(2 / 3) and S.vote([{'verdict': None}]) is None
    tie = S.vote([{'verdict': 'met', 'confidence': 0.6}, {'verdict': 'not_met', 'confidence': 0.9}])
    assert tie['verdict'] == 'not_met'


def test_calibration_is_fixed_sequence_and_skips_thin_thresholds():
    rows = [{'confidence': 0.99, 'error': 0}] * 5 + [{'confidence': 0.9, 'error': 0}] * 100 + \
           [{'confidence': 0.5, 'error': 1}] * 40 + [{'confidence': -1.0, 'error': 0}] * 10
    c = S.calibrate(rows, alpha=0.05, delta=0.1, min_selected=30)
    assert c['path'][0]['note'].startswith('skipped') and c['lambda'] == 0.9
    assert all(p['lambda'] >= 0 for p in c['path'])                 # failed (escalated) items are never selectable
    t = S.evaluate(rows, c['lambda'], 0.1)
    assert t['selected'] == 105 and t['errors'] == 0 and t['coverage'] == pytest.approx(105 / 155)
    assert S.calibrate(rows, alpha=0.001, delta=0.1)['lambda'] is None


def test_guarantee_holds_in_simulation():
    """Empirical check of the procedure: with an honest confidence signal, the test risk of the calibrated
    threshold exceeds alpha in at most ~delta of repetitions."""
    rnd = random.Random(0)
    alpha, delta, fails, runs = 0.10, 0.10, 0, 200

    def draw(n):
        out = []
        for _ in range(n):
            c = rnd.random()
            out.append({'confidence': round(c, 2), 'error': int(rnd.random() < 0.3 * (1 - c))})
        return out
    for _ in range(runs):
        lam = S.calibrate(draw(300), alpha, delta)['lambda']
        if lam is None:
            continue
        test = S.evaluate(draw(5000), lam, delta)
        fails += test['risk'] > alpha
    assert fails / runs <= delta + 0.05


def test_used_groups_include_v3_experiment_manifests(tmp_path):
    root = tmp_path
    (root / 'reports/rag-ablation-v3-x').mkdir(parents=True)
    (root / 'reports/rag-ablation-v3-x/sample-manifest.json').write_text(json.dumps({'group_ids': ['HBG-1', 'HBG-2']}))
    (root / 'reports/selective-v3-y').mkdir(parents=True)
    (root / 'reports/selective-v3-y/sample-manifest.json').write_text(json.dumps({'group_ids': ['HBG-3']}))
    groups, sources = used_dev_groups_v3(root)
    assert {'HBG-1', 'HBG-2', 'HBG-3'} <= groups and len(sources) == 3


def run(*argv):
    return load_script('run_selective_v3').main(list(argv))


def _stages(root, monkeypatch, varying: bool):
    if varying:                                          # a mock whose sampled calls differ from the T=0 call
        import run_review_v3
        real = run_review_v3.make_backend

        def fake(judge, root_, store, run_limit, max_tokens=800, responder=None, temperature=0.0):
            counter = {'n': 0}

            def respond(messages):
                counter['n'] += 1
                base = json.loads(responder(messages))
                if temperature > 0:
                    base['analysis'] = f'sampled variant {counter["n"]}'
                    base['checks'][0]['result'] = 'fail' if counter['n'] % 3 == 0 else base['checks'][0]['result']
                return json.dumps(base)
            return ScriptedBackend(respond, name='mock'), False, None
        monkeypatch.setattr(run_review_v3, 'make_backend', fake)
        assert real is not fake
    assert run('sample', '--root', str(root), '--groups', '16', '--pilot-groups', '4') == 0
    run_dir = next((root / 'reports').glob('selective-v3-*'))
    assert run('pilot', '--root', str(root), '--run-dir', str(run_dir)) == 0
    for k in (1, 2):
        assert run('run', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', str(k)) == 0
    return run_dir


def test_end_to_end_falls_back_to_verbalized_when_sampling_does_not_vary(tmp_path, capsys):
    root = make_root(tmp_path, n_groups=30)
    labels = root / 'data/private/physician-rater-index.jsonl'
    labels.rename(labels.with_suffix('.hidden'))      # sample/pilot/run must not need labels
    run_dir = _stages(root, None, varying=False)
    decision = json.loads((run_dir / 'pilot/decision.json').read_text())
    assert decision['sampling_effective'] is False
    assert json.loads((run_dir / 'run-layout.json').read_text())['k'] == 1
    labels.with_suffix('.hidden').rename(labels)
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(run_dir)) == 0
    out = json.loads(capsys.readouterr().out)
    assert out['primary']['estimator'] == 'verbalized'
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text())
    assert manifest['split_counts']['pilot'] >= 1 and set(manifest['split_of_group'].values()) == {'pilot', 'calibration', 'test'}


def test_end_to_end_with_sampling_uses_self_consistency(tmp_path, capsys, monkeypatch):
    root = make_root(tmp_path, n_groups=30)
    run_dir = _stages(root, monkeypatch, varying=True)
    assert json.loads((run_dir / 'pilot/decision.json').read_text())['sampling_effective'] is True
    assert json.loads((run_dir / 'run-layout.json').read_text())['k'] == 3
    with pytest.raises(SystemExit, match='already run'):
        run('run', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', '1')
    with pytest.raises(SystemExit, match='cannot be changed'):
        run('run', '--root', str(root), '--run-dir', str(run_dir), '--parts', '3', '--part', '1')
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(run_dir)) == 0
    metrics = json.loads((next(run_dir.glob('score-*')) / 'metrics.json').read_text())
    assert metrics['primary']['estimator'] == 'self_consistency' and set(metrics['estimators']) == {
        'verbalized', 'self_consistency'}
    assert set(metrics['estimators']['self_consistency']['by_alpha']) == {'0.05', '0.1'}


def test_score_refuses_missing_parts(tmp_path):
    root = make_root(tmp_path, n_groups=30)
    assert run('sample', '--root', str(root), '--groups', '16', '--pilot-groups', '4') == 0
    run_dir = next((root / 'reports').glob('selective-v3-*'))
    with pytest.raises(SystemExit, match='pilot first'):
        run('run', '--root', str(root), '--run-dir', str(run_dir))
    assert run('pilot', '--root', str(root), '--run-dir', str(run_dir)) == 0
    assert run('run', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', '1') == 0
    with pytest.raises(SystemExit, match='Part 2/2'):
        run('score', '--root', str(root), '--run-dir', str(run_dir))


def test_fixed_gate_protocol_end_to_end(tmp_path, capsys, monkeypatch):
    root = make_root(tmp_path, n_groups=40)
    _stages_gate = _gate_run(root, monkeypatch)
    manifest = json.loads((_stages_gate / 'sample-manifest.json').read_text())
    assert manifest['version'] == 'selective-v3-2' and manifest['fixed_k'] == 3
    assert set(manifest['split_of_group'].values()) == {'test'}
    assert not (_stages_gate / 'pilot').exists()
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(_stages_gate)) == 0
    metrics = json.loads((next(_stages_gate.glob('score-*')) / 'metrics.json').read_text())
    view = metrics['as_run']
    assert 'completed' not in metrics and set(view['by_theme']) <= {'communication', 'hedging'}
    p = view['primary']
    assert p['auto_decided'] + view['escalated']['items'] == view['items']
    assert p['success'] == (p['risk_ucb'] is not None and p['risk_ucb'] <= 0.05)
    items = [json.loads(l) for l in (next(_stages_gate.glob('score-*')) / 'items.jsonl').read_text().splitlines()]
    assert all(i['auto'] == (i['vote_share'] == 1.0) for i in items)


def _gate_run(root, monkeypatch):
    import run_review_v3

    def fake(judge, root_, store, run_limit, max_tokens=800, responder=None, temperature=0.0):
        counter = {'n': 0}

        def respond(messages):
            counter['n'] += 1
            base = json.loads(responder(messages))
            if temperature > 0 and counter['n'] % 4 == 0:
                base['checks'][0]['result'] = 'fail' if base['checks'][0]['result'] != 'fail' else 'pass'
            return json.dumps(base)
        return ScriptedBackend(respond, name='mock'), False, None
    monkeypatch.setattr(run_review_v3, 'make_backend', fake)
    assert run('sample', '--root', str(root), '--gate', '--groups', '8') == 0
    run_dir = next((root / 'reports').glob('selective-v3-*'))
    for k in (1, 2):
        assert run('run', '--root', str(root), '--run-dir', str(run_dir), '--parts', '2', '--part', str(k)) == 0
    assert json.loads((run_dir / 'run-layout.json').read_text())['k'] == 3
    assert not (root / 'reports/selective-v3-sentinels').exists()      # mock runs never claim the paid sentinel
    return run_dir


def test_completion_after_a_quota_stop_reports_both_views(tmp_path, capsys, monkeypatch):
    """Amendment v2.1: a 402 stops the run; only the interrupted units are re-sent, once; both views reported."""
    import run_review_v3
    from medical_jury.judge_v2 import BackendError
    state = {'calls': 0, 'quota': 20}

    def fake(judge, root_, store, run_limit, max_tokens=800, responder=None, temperature=0.0):
        def respond(messages):
            state['calls'] += 1
            if state['quota'] is not None and state['calls'] > state['quota']:
                raise BackendError('PROVIDER_HTTP_402')
            return responder(messages)
        return ScriptedBackend(respond, name='mock'), False, None
    monkeypatch.setattr(run_review_v3, 'make_backend', fake)
    root = make_root(tmp_path, n_groups=40)
    assert run('sample', '--root', str(root), '--gate', '--groups', '12') == 0
    run_dir = next((root / 'reports').glob('selective-v3-*'))
    assert run('run', '--root', str(root), '--run-dir', str(run_dir), '--threads', '1') == 0
    preds = [json.loads(l) for l in (run_dir / 'run-part1of1/predictions.jsonl').read_text().splitlines()]
    stopped = [p for p in preds if p['status'] != 'Completed']
    assert stopped and any(p.get('failure_code') == 'PROVIDER_HTTP_402' for p in stopped)
    state['quota'] = None                               # account topped up
    assert run('complete', '--root', str(root), '--run-dir', str(run_dir), '--threads', '1') == 0
    comp = [json.loads(l) for l in (run_dir / 'completion/predictions.jsonl').read_text().splitlines()]
    assert len(comp) == len(stopped) and all(c['status'] == 'Completed' for c in comp)
    with pytest.raises(SystemExit, match='already run'):
        run('complete', '--root', str(root), '--run-dir', str(run_dir))
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(run_dir)) == 0
    metrics = json.loads((next(run_dir.glob('score-*')) / 'metrics.json').read_text())
    assert metrics['as_run']['items_with_interrupted_samples'] > 0
    assert metrics['completed']['items_with_interrupted_samples'] == 0
    assert 'AFTER the as-run result' in metrics['completion_note']


def test_sample_order_ignores_stub_records():
    rs = load_script('run_selective_v3')
    stub = {'example_id': 'HBX-1#0', 'status': 'NotRun'}
    rec = {'example_id': 'HBX-1#2', 'sample': 2, 'status': 'Completed'}
    assert [rs._sample_index(r) for r in (stub, rec)] == [0, 2]
    assert rs._interrupted(stub) and rs._interrupted({'status': 'Failed', 'failure_code': 'PROVIDER_HTTP_402'})
    assert not rs._interrupted({'status': 'Failed', 'failure_code': 'OUTPUT_SCHEMA_INVALID'})
