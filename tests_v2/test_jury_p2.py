import json
import random

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from medical_jury.domain import Rubric
from medical_jury.evaluation.aggregation import dawid_skene, majority, posterior
from medical_jury.judge_v2 import JudgeV2, OpenAICompatibleBackend, ScriptedBackend
from medical_jury.jury_config import build_backend, load_panel
from medical_jury.workflow_v2 import JuryEngine

from v2_fixtures import load_script, make_project


def fixed(verdict, conf=0.9):
    return lambda m: json.dumps({'analysis': 'fixture', 'verdict': verdict, 'confidence': conf, 'evidence': []})


# ---------------- aggregation ----------------

def test_majority_and_ties():
    assert majority({'a': True, 'b': True, 'c': False}) is True
    assert majority({'a': True, 'b': False}) is None
    assert majority({'a': None, 'b': False}) is False
    assert majority({}) is None


def test_dawid_skene_recovers_reliable_judge():
    rng = random.Random(0)
    truth = [rng.random() < 0.7 for _ in range(800)]
    items = []
    for t in truth:
        items.append({'good1': t if rng.random() < .92 else not t,
                      'good2': t if rng.random() < .9 else not t,
                      'noisy': rng.random() < .5,
                      'missing': None if rng.random() < .5 else (t if rng.random() < .85 else not t)})
    ds = dawid_skene(items)
    j = ds['judges']
    assert j['good1']['sensitivity'] > .85 and j['good1']['specificity'] > .8
    assert abs(j['noisy']['sensitivity'] - (1 - j['noisy']['specificity'])) < .1
    acc = sum((p >= .5) == t for p, t in zip(ds['posteriors'], truth)) / len(truth)
    assert acc > .9 and abs(ds['prior_met'] - .7) < .05


def test_posterior_ignores_unknown_and_missing_votes():
    params = {'a': {'sensitivity': .9, 'specificity': .9}}
    assert posterior({'a': True, 'z': False, 'b': None}, params, .5) == pytest.approx(.9)


# ---------------- backends / config ----------------

def test_openai_compatible_endpoint_policy(settings, store):
    OpenAICompatibleBackend('q', 'http://127.0.0.1:11434/v1', 'qwen', settings=settings, store=store)
    with pytest.raises(ValueError, match='allowlisted'):
        OpenAICompatibleBackend('x', 'https://evil.example/v1', 'm', settings=settings, store=store,
                                price_in=1, price_out=1, allowed_hosts=set())
    with pytest.raises(ValueError, match='allowlisted'):
        OpenAICompatibleBackend('x', 'http://api.ok.example/v1', 'm', settings=settings, store=store,
                                price_in=1, price_out=1, allowed_hosts={'api.ok.example'})
    with pytest.raises(ValueError, match='prices'):
        OpenAICompatibleBackend('x', 'https://api.ok.example/v1', 'm', settings=settings, store=store,
                                allowed_hosts={'api.ok.example'})


def test_local_backend_round_trip_without_ledger(case, rubric, settings, store):
    seen = {}

    def handler(request):
        seen['url'] = str(request.url)
        seen['body'] = json.loads(request.content)
        return httpx.Response(200, json={'model': 'qwen', 'choices': [{'finish_reason': 'stop', 'message': {
            'content': fixed('not_met', .7)(None)}}]})
    backend = OpenAICompatibleBackend('qwen-local', 'http://127.0.0.1:11434/v1', 'qwen2.5:14b', settings=settings,
                                      store=store, client=httpx.Client(transport=httpx.MockTransport(handler)))
    rec = JudgeV2(backend, store).judge(case, rubric, 'J')
    assert rec['status'] == 'Completed' and rec['verdict'] == 'not_met'
    assert seen['url'] == 'http://127.0.0.1:11434/v1/chat/completions' and 'tools' not in seen['body']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM costs').fetchone()[0] == 0


def test_paid_remote_backend_is_ledgered(case, rubric, settings, store):
    def handler(request):
        return httpx.Response(200, json={'usage': {'prompt_tokens': 1000, 'completion_tokens': 100},
                                         'choices': [{'finish_reason': 'stop', 'message': {'content': fixed('met')(None)}}]})
    backend = OpenAICompatibleBackend('third', 'https://api.ok.example/v1', 'm', settings=settings, store=store,
                                      run_limit_cny=5, price_in=2, price_out=8, allowed_hosts={'api.ok.example'},
                                      ceiling_cny=30, client=httpx.Client(transport=httpx.MockTransport(handler)))
    JudgeV2(backend, store).judge(case, rubric, 'J')
    with store.connect() as db:
        row = db.execute('SELECT state, accounted FROM costs').fetchone()
    assert row['state'] == 'settled' and row['accounted'] == pytest.approx((1000 * 2 + 100 * 8) / 1e6)


def test_panel_config_rejects_inline_keys(tmp_path, settings, store):
    (tmp_path / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'a', 'kind': 'mock', 'api_key': 'x'}]}))
    with pytest.raises(ValueError, match='keys'):
        load_panel(tmp_path)
    (tmp_path / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'a', 'kind': 'mock'}, {'name': 'a', 'kind': 'mock'}]}))
    with pytest.raises(ValueError, match='unique'):
        load_panel(tmp_path)
    (tmp_path / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'm', 'kind': 'mock'}]}))
    assert build_backend(load_panel(tmp_path)[0], settings, store, 1).name == 'm'


# ---------------- workflow ----------------

def engine(store, root, verdicts, **kw):
    judges = {n: JudgeV2(ScriptedBackend(fixed(v, c), n), store) for n, (v, c) in verdicts.items()}
    return JuryEngine(judges, store, root, checkpointer=kw.pop('checkpointer', InMemorySaver()), **kw)


def test_unanimous_confident_jury_needs_no_human(case, rubric, store, settings):
    e = engine(store, settings.root, {'a': ('met', .9), 'b': ('met', .9), 'c': ('met', .9)})
    out = e.start('J-auto', case.model_copy(update={'candidate_output': 'Use 15 mg/kg.'}), [rubric])
    assert out['status'] == 'Completed'
    row = out['report']['criteria'][0]
    assert row['final_status'] == 'jury_auto' and row['final_decision'] == 'met'
    assert (settings.root / 'reports/J-auto/jury-v2-report.json').exists()


def test_disagreement_interrupts_and_human_decision_is_stored_separately(case, rubric, store, settings):
    e = engine(store, settings.root, {'a': ('met', .9), 'b': ('not_met', .9), 'c': ('not_met', .6)})
    first = e.start('J-h', case, [rubric])
    assert first['status'] == 'AwaitingHumanReview'
    item = first['review']['items'][0]
    assert item['jury_decision'] == 'not_met' and 'JUDGES_DISAGREE' in item['reasons']
    with pytest.raises(ValueError):
        e.resume('J-h', {rubric.rubric_id: {'decision': 'met'}})  # reviewer id required


def test_resume_after_interrupt_with_sqlite_checkpoint(case, rubric, store, settings):
    with SqliteSaver.from_conn_string(str(settings.root / 'data/jury-v2-checkpoints.sqlite')) as saver:
        e = engine(store, settings.root, {'a': ('met', .9), 'b': ('not_met', .9), 'c': ('not_met', .6)},
                   checkpointer=saver)
        assert e.start('J-s', case, [rubric])['status'] == 'AwaitingHumanReview'
    with SqliteSaver.from_conn_string(str(settings.root / 'data/jury-v2-checkpoints.sqlite')) as saver:
        e2 = engine(store, settings.root, {'a': ('met', .9), 'b': ('not_met', .9), 'c': ('not_met', .6)},
                    checkpointer=saver)
        done = e2.resume('J-s', {rubric.rubric_id: {'decision': 'met', 'reviewer': 'dr-1', 'note': 'context'}})
    row = done['report']['criteria'][0]
    assert row['jury_decision'] == 'not_met' and row['final_decision'] == 'met'
    assert row['final_status'] == 'human_overrode' and row['human']['reviewer'] == 'dr-1'


def test_failed_judge_routes_to_review_and_batch_mode_does_not_block(case, rubric, store, settings):
    judges = {'a': JudgeV2(ScriptedBackend(fixed('met'), 'a'), store),
              'b': JudgeV2(ScriptedBackend(lambda m: '{bad', 'b'), store)}
    e = JuryEngine(judges, store, settings.root, require_human=False, checkpointer=InMemorySaver())
    out = e.start('J-f', case, [rubric])
    row = out['report']['criteria'][0]
    assert out['status'] == 'Completed' and 'JUDGE_FAILURE' in row['review_reasons']
    assert row['final_status'] == 'pending_human_review' and row['final_decision'] is None


def test_frozen_dawid_skene_overrides_naive_majority(case, rubric, store, settings):
    params = {'prior_met': .5, 'judges': {'a': {'sensitivity': .99, 'specificity': .99},
                                          'b': {'sensitivity': .55, 'specificity': .5},
                                          'c': {'sensitivity': .55, 'specificity': .5}}}
    e = engine(store, settings.root, {'a': ('met', .9), 'b': ('not_met', .9), 'c': ('not_met', .9)},
               ds_params=params, require_human=False)
    row = e.start('J-ds', case, [rubric])['report']['criteria'][0]
    assert row['method'] == 'dawid_skene_frozen' and row['jury_decision'] == 'met'


def test_cross_case_rubric_rejected(case, store, settings):
    e = engine(store, settings.root, {'a': ('met', .9)})
    other = Rubric(rubric_id='X', level='case', case_id='OTHER', source='s', version='1', text='t')
    with pytest.raises(ValueError):
        e.start('J-x', case, [other])


# ---------------- ablation script ----------------

def test_ablation_end_to_end(tmp_path):
    root = make_project(tmp_path)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': n, 'kind': 'mock'} for n in ('m1', 'm2', 'm3')]}))
    run = load_script('run_dev_v2')
    dirs = []
    for n in ('m1', 'm2', 'm3'):
        out = root / f'reports/r-{n}'
        assert run.main(['--root', str(root), '--backend', 'panel', '--judge', n, '--n-groups', '30', '--out', str(out)]) == 0
        dirs.append(str(out))
    ab = load_script('ablation_jury')
    assert ab.main(dirs + ['--root', str(root), '--bootstrap', '50']) == 0
    out = next((root / 'reports').glob('ablation-jury-*'))
    summary = json.loads((out / 'summary.json').read_text())
    assert set(summary['results']) == {'single:m1', 'single:m2', 'single:m3', 'majority', 'dawid_skene'}
    assert json.loads((out / 'ds-params.json').read_text())['fitted_on'].startswith('dev votes only')


def test_ablation_rejects_different_samples(tmp_path):
    root = make_project(tmp_path)
    (root / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': n, 'kind': 'mock'} for n in ('m1', 'm2')]}))
    run = load_script('run_dev_v2')
    run.main(['--root', str(root), '--backend', 'panel', '--judge', 'm1', '--n-groups', '20', '--out', str(root / 'reports/a')])
    run.main(['--root', str(root), '--backend', 'panel', '--judge', 'm2', '--n-groups', '25', '--out', str(root / 'reports/b')])
    with pytest.raises(SystemExit):
        load_script('ablation_jury').main([str(root / 'reports/a'), str(root / 'reports/b'), '--root', str(root)])
