"""Plan phase 2: router prompt v2 (decision tests) and the paired v1 vs v2 comparison. Mock judges only."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from medical_jury.evaluation import router_scoring as RSC
from medical_jury.judge_v2 import ScriptedBackend
from medical_jury.v3.router import (DECISION_TESTS_V2, ROUTER_PROMPT_VERSION, ROUTER_PROMPT_VERSION_2,
                                    SYSTEM_ROUTER, SituationRouter, build_router_messages, load_decision_tests,
                                    mock_router_responder)
from medical_jury.v3.schema import load_kb

from test_v3_rag_ablation import make_root
from v2_fixtures import load_script

REPO = Path(__file__).resolve().parents[1]
V1_PROMPT_SHA = '18ea42b02b493ede357450eeeee6bc4ecd3607bb1f793c3cbe420584c9c53112'


def _kb():
    return load_kb(REPO / 'kb' / 'v3')


def test_v1_prompt_is_unchanged_and_v1_messages_have_no_decision_tests():
    assert hashlib.sha256(SYSTEM_ROUTER.encode()).hexdigest() == V1_PROMPT_SHA
    kb = _kb()
    msgs = build_router_messages((('user', 'hi'),), kb.situations)
    payload = json.loads(msgs[1]['content'])
    assert msgs[0]['content'] == SYSTEM_ROUTER and all(set(g) == {'group', 'options'} for g in payload['groups'])
    tests = load_decision_tests(REPO / DECISION_TESTS_V2, kb.situations)
    with pytest.raises(ValueError, match='v2 feature'):
        build_router_messages((('user', 'hi'),), kb.situations, ROUTER_PROMPT_VERSION, tests)


def test_v2_messages_carry_decision_tests_only_for_listed_groups():
    kb = _kb()
    tests = load_decision_tests(REPO / DECISION_TESTS_V2, kb.situations)
    msgs = build_router_messages((('user', 'hi'),), kb.situations, ROUTER_PROMPT_VERSION_2, tests)
    payload = json.loads(msgs[1]['content'])
    with_tests = {g['group'] for g in payload['groups'] if 'decision_test' in g}
    assert with_tests == set(tests['tests']) and 'HB:communication' not in with_tests
    assert 'Decision principle' in msgs[0]['content']
    assert [list(g)[-1] for g in payload['groups']] == ['options'] * len(payload['groups'])
    with pytest.raises(ValueError, match='unknown router prompt version'):
        build_router_messages((('user', 'hi'),), kb.situations, 'v9')


def test_decision_tests_are_validated(tmp_path):
    kb = _kb()
    bad = tmp_path / 'bad.json'
    bad.write_text(json.dumps({'tests': {'HB:nope': 'x' * 40}}))
    with pytest.raises(ValueError, match='unknown groups'):
        load_decision_tests(bad, kb.situations)
    bad.write_text(json.dumps({'tests': {'HB:hedging': 'short'}}))
    with pytest.raises(ValueError, match='20-1500'):
        load_decision_tests(bad, kb.situations)


def test_router_records_version_and_tests_hash():
    kb = _kb()
    tests = load_decision_tests(REPO / DECISION_TESTS_V2, kb.situations)
    r = SituationRouter(ScriptedBackend(mock_router_responder, name='mock'), prompt_version=ROUTER_PROMPT_VERSION_2,
                        decision_tests=tests)
    rec = r.route((('user', 'I am a nurse'),), kb.situations, 'job')
    assert rec['status'] == 'Completed' and rec['prompt_version'] == ROUTER_PROMPT_VERSION_2
    assert rec['decision_tests_sha256'] == tests['sha256']


def test_mcnemar_and_decision_rule():
    assert RSC.mcnemar_exact(0, 0) == 1.0
    assert RSC.mcnemar_exact(2, 12) == pytest.approx(0.01294, abs=1e-4)
    assert RSC.mcnemar_exact(5, 5) == 1.0
    gold = ['A'] * 30 + ['B'] * 30
    rows = [{'gold': g, 'group': 'G', 'v1': g if i % 3 else 'X', 'v2': g} for i, g in enumerate(gold)]
    out = RSC.compare(rows, 'v1', 'v2', seed=1)
    assert out['adopt_v2'] and out['discordant'] == {'only_v1_right': 0, 'only_v2_right': 20}
    # v2 "wins" by always answering A: accuracy may rise on a skewed sample but macro recall falls -> not adopted
    skew = ['A'] * 50 + ['B'] * 10
    rows = [{'gold': g, 'group': 'G', 'v1': g if i % 4 else 'X', 'v2': 'A'} for i, g in enumerate(skew)]
    out = RSC.compare(rows, 'v1', 'v2', seed=1)
    assert out['accuracy']['v2'] > out['accuracy']['v1'] and not out['adopt_v2']


def _root(tmp_path, n_groups=30):
    root = make_root(tmp_path, n_groups=n_groups)
    shutil.copytree(REPO / 'kb' / 'router', root / 'kb' / 'router')
    return root


def run(*argv):
    return load_script('run_router_v2').main(list(argv))


def test_route_and_score_end_to_end_without_labels(tmp_path, capsys):
    root = _root(tmp_path)
    labels = root / 'data/private/physician-rater-index.jsonl'
    labels.rename(labels.with_suffix('.hidden'))      # no physician label is ever needed
    assert run('route', '--root', str(root), '--n-groups', '10', '--threads', '1') == 0
    run_dir = next((root / 'reports').glob('router-v3b-*'))
    routes = [json.loads(l) for l in (run_dir / 'route/routes.jsonl').read_text().splitlines()]
    assert len(routes) == 20 and {r['prompt_version'] for r in routes} == {ROUTER_PROMPT_VERSION, ROUTER_PROMPT_VERSION_2}
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text())
    assert manifest['outcome_blind'] is True and len(manifest['group_ids']) == 10
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(run_dir)) == 0
    metrics = json.loads((next(run_dir.glob('score-*')) / 'metrics.json').read_text())
    assert metrics['n'] == 10 and set(metrics['accuracy']) == {ROUTER_PROMPT_VERSION, ROUTER_PROMPT_VERSION_2}
    # the next sample excludes these groups
    assert run('route', '--root', str(root), '--n-groups', '10', '--threads', '1') == 0
    dirs = sorted((root / 'reports').glob('router-v3b-2*'))
    g = [set(json.loads((d / 'sample-manifest.json').read_text())['group_ids']) for d in dirs]
    assert len(g) == 2 and not (g[0] & g[1])


def test_complete_after_quota_stop_resends_only_interrupted(tmp_path, capsys, monkeypatch):
    import run_review_v3
    from medical_jury.judge_v2 import BackendError
    state = {'calls': 0, 'quota': 7}

    def fake(judge, root_, store, run_limit, max_tokens=800, responder=None, temperature=0.0):
        def respond(messages):
            state['calls'] += 1
            if state['quota'] is not None and state['calls'] > state['quota']:
                raise BackendError('PROVIDER_HTTP_402')
            return responder(messages)
        return ScriptedBackend(respond, name='mock'), False, None
    monkeypatch.setattr(run_review_v3, 'make_backend', fake)
    root = _root(tmp_path)
    assert run('route', '--root', str(root), '--n-groups', '8', '--threads', '1') == 0
    run_dir = next((root / 'reports').glob('router-v3b-*'))
    routes = [json.loads(l) for l in (run_dir / 'route/routes.jsonl').read_text().splitlines()]
    stopped = [r for r in routes if r['status'] != 'Completed']
    assert len(stopped) == 16 - 7
    state['quota'] = None
    assert run('complete', '--root', str(root), '--run-dir', str(run_dir), '--threads', '1') == 0
    comp = [json.loads(l) for l in (run_dir / 'completion/routes.jsonl').read_text().splitlines()]
    assert len(comp) == len(stopped) and all(c['status'] == 'Completed' for c in comp)
    with pytest.raises(SystemExit, match='already run'):
        run('complete', '--root', str(root), '--run-dir', str(run_dir))
    capsys.readouterr()
    assert run('score', '--root', str(root), '--run-dir', str(run_dir)) == 0
    metrics = json.loads((next(run_dir.glob('score-*')) / 'metrics.json').read_text())
    assert metrics['interrupted_left'] == {ROUTER_PROMPT_VERSION: 0, ROUTER_PROMPT_VERSION_2: 0}


def test_used_groups_cover_router_runs(tmp_path):
    from medical_jury.v3.used_groups import used_dev_groups_v3
    d = tmp_path / 'reports/router-v3b-x'
    d.mkdir(parents=True)
    (d / 'sample-manifest.json').write_text(json.dumps({'group_ids': ['HBG-9']}))
    assert 'HBG-9' in used_dev_groups_v3(tmp_path)[0]
