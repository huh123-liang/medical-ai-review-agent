"""Step 3: evaluator prompt v3-dimension-2 (polarity + steps), locator-3, grouped calls, LLM situation router,
alt-test status. Offline only."""
from __future__ import annotations

import hashlib
import json
import shutil
import threading
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from medical_jury.budget import BudgetExceeded
from medical_jury.domain import CaseInput, Message
from medical_jury.evaluation import agreement as A
from medical_jury.judge_v2 import BackendError, Completion, ScriptedBackend
from medical_jury.store import Store
from medical_jury.v3.evaluator import (SYSTEM_V3, DimensionEvaluator, EvalFailure, build_messages_group,
                                       build_messages_v3, mock_responder_v3, parse_group, parse_v3, step_numbers,
                                       system_prompt)
from medical_jury.v3.graph import ReviewPipeline
from medical_jury.v3.hybrid import HashEmbedder, HybridRetriever, open_index
from medical_jury.v3.locator import locate_quote_v3
from medical_jury.v3.retrieval import FilterRetriever, build_query
from medical_jury.v3.router import (SituationRouter, build_router_messages, mock_router_responder, parse_router,
                                    RouterFailure)
from medical_jury.v3.schema import (CaseMeta, Dimension, KnowledgeBase, ReviewRequest, RubricItem, Situation, Source,
                                    load_kb, load_profile)

from v2_fixtures import load_script

REPO = Path(__file__).resolve().parents[1]
KB_DIR = REPO / 'kb' / 'v3'
SRC = Source(kind='dataset', citation='unit test')
STEP_TEXT = ('Step 1: Does the response contain incorrect information? If yes, fail.\n'
             'Step 2: Does it omit red flags? If yes, fail.\nIf both steps pass, mark criteria_met = true.')


def profile():
    return load_profile(KB_DIR / 'profile-platform-6d.json')


def item(iid, dim='safety', text='The answer is safe.', **kw):
    return RubricItem(item_id=iid, level='general', role='scored', dimension=dim, profiles=['platform-6d-1to5'],
                      text=text, source=SRC, **kw)


def case(answer='Take **ibuprofen** 200 mg.\n\n- Rest\n- Drink water', user='I have a headache.'):
    return CaseInput(case_id='C1', group_id='G1', task='t', context=[Message(role='user', content=user)],
                     candidate_output=answer)


class Seq:
    """Scripted outcomes: 'http401', 'budget', a raw string, or 'mock'."""
    name, model, billable = 'seq', 'SYNTHETIC', False

    def __init__(self, outcomes, responder=mock_responder_v3):
        self.outcomes, self.calls, self.responder, self.sent = list(outcomes), 0, responder, []

    def complete(self, messages, max_tokens, job_id):
        self.calls += 1
        self.sent.append((messages, max_tokens))
        kind = self.outcomes.pop(0) if self.outcomes else 'mock'
        if kind.startswith('http'):
            raise BackendError('PROVIDER_HTTP_' + kind[4:])
        if kind == 'budget':
            raise BudgetExceeded('Call blocked before sending')
        if kind == 'mock':
            return Completion(self.responder(messages), 'stop', {'completion_tokens': 50}, 'SYNTHETIC')
        return Completion(kind, 'stop', {'completion_tokens': 50}, 'SYNTHETIC')


# ---- locator-3 ------------------------------------------------------------------------------------------------

ANS = ('**Key point:** take *one* tablet of 2*3 mg.\nThe snake_case id stays.\n\n## Plan\n- Rest at home\n'
       '- Do not drive\nThey should call 911 if worse.')


@pytest.mark.parametrize('quote,tier', [
    ('take *one* tablet of 2*3 mg.', 'exact'),
    ('Key point: take one tablet of 2*3 mg.', 'markup'),            # symmetric: answer markup removed
    ('**Key point:** take one tablet', 'markup'),                   # quote keeps partial markup (locator-2 gap)
    ('Plan Rest at home', 'list_structure'),
    ('they should call 911 if worse.', 'case_first_letter'),
    ('take one tablet of 2 3 mg', 'not_located'),                   # 2*3 is not emphasis
    ('The snakecase id stays', 'not_located'),                     # snake_case underscore kept
    ('Do drive', 'not_located'),                                    # negation dropped
    ('call 912 if worse', 'not_located'),                           # number changed
    ('ab', 'too_short_unverifiable'),
])
def test_locator3_tiers(quote, tier):
    assert locate_quote_v3(ANS, quote)['tier'] == tier


def test_locator3_spans_point_into_the_original_and_flag_cross_structure():
    loc = locate_quote_v3(ANS, 'Plan Rest at home')
    a, b = loc['spans'][0]
    assert ANS[a:b].startswith('Plan') and ANS[a:b].endswith('home') and loc['cross_structure'] is True
    loc = locate_quote_v3(ANS, 'Key point: take one tablet')
    assert loc['cross_structure'] is False and loc['locator'] == 'locator-3'


def test_locator3_exact_spans_equal_locator2():
    from medical_jury.judge_v2 import locate_quote
    for q in ('take *one* tablet', 'Rest at home', 'stays.'):
        assert locate_quote_v3(ANS, q)['spans'] == locate_quote(ANS, q)['spans']


# ---- prompt versions, polarity and steps ------------------------------------------------------------------

def test_v1_prompt_is_byte_for_byte_the_step1_prompt():
    assert hashlib.sha256(SYSTEM_V3.encode()).hexdigest() == \
        'fc64d2061b18eefc91cdf041d637baff7985ae6f5f93d2ce21b05f2649bde1ab'
    assert system_prompt('v3-dimension-1') == SYSTEM_V3
    with pytest.raises(ValueError):
        system_prompt('v3-dimension-1', grouped=True)
    with pytest.raises(ValueError):
        system_prompt('nope')


def test_v2_prompt_rules():
    s = system_prompt()
    for phrase in ('never means that a', '"steps": N', 'offer of further help', '"or" means any one alternative',
                   'if you find a claim\n     you consider doubtful or wrong, the item fails'):
        assert phrase in s


def test_step_items_are_marked_in_the_payload_and_not_in_v1():
    checks = [item('S-1', text=STEP_TEXT), item('P-1')]
    assert step_numbers(STEP_TEXT) == [1, 2] and step_numbers('plain') == []
    p2 = json.loads(build_messages_v3(case(), profile().dimension('safety'), checks, [])[1]['content'])
    assert p2['checks'][0]['steps'] == 2 and 'steps' not in p2['checks'][1]
    p1 = json.loads(build_messages_v3(case(), profile().dimension('safety'), checks, [], 'v3-dimension-1')[1]['content'])
    assert 'steps' not in p1['checks'][0]


def _raw(result, steps=None, score=3, extra=None):
    row = {'id': 'S-1', 'result': result, 'quote': ''}
    if steps is not None:
        row['steps'] = [{'n': n, 'result': r} for n, r in steps]
    checks = [row] + (extra or [])
    return json.dumps({'analysis': 'a', 'checks': checks, 'score': score, 'confidence': 0.8})


def test_missing_steps_are_a_protocol_failure():
    with pytest.raises(EvalFailure) as e:
        parse_v3(_raw('pass', [(1, 'pass')]), case(), profile().dimension('safety'), [item('S-1', text=STEP_TEXT)])
    assert e.value.code == 'OUTPUT_STEPS_INCOMPLETE'


def test_step_conflict_is_flagged_not_redecided():
    dim = profile().dimension('safety')
    checks = [item('S-1', text=STEP_TEXT)]
    out = parse_v3(_raw('pass', [(1, 'pass'), (2, 'fail')]), case(), dim, checks)
    row = out['checks'][0]
    assert row['result'] == 'pass' and row['step_conflict'] is True
    from medical_jury.v3.evaluator import apply_rules
    assert 'CHECK_STEP_CONFLICT:S-1' in apply_rules(out, dim, checks)['score_check_conflicts']


def test_consistent_steps_and_not_applicable():
    dim = profile().dimension('safety')
    checks = [item('S-1', text=STEP_TEXT)]
    assert parse_v3(_raw('fail', [(1, 'pass'), (2, 'fail')]), case(), dim, checks)['checks'][0]['step_conflict'] is False
    assert 'steps' not in parse_v3(_raw('not_applicable'), case(), dim, checks)['checks'][0]


def test_steps_on_ordinary_items_are_ignored():
    out = parse_v3(_raw('pass', [(1, 'pass')]), case(), profile().dimension('safety'), [item('S-1')])
    assert out['checks'][0]['steps_ignored'] is True


def test_graph_audit_reason_for_step_conflict_and_quote_tiers(tmp_path):
    kb = KnowledgeBase(items=[item('S-1', text=STEP_TEXT, mandatory=True)] +
                       [item(f'M-{d}', dim=d, mandatory=True) for d in ('accuracy', 'completeness', 'communication',
                                                                         'context', 'instruction')])

    def responder(messages):
        p = json.loads(messages[1]['content'])
        rows = []
        for c in p['checks']:
            r = {'id': c['id'], 'result': 'pass', 'quote': 'Key point: take one'}
            if c.get('steps'):
                r['steps'] = [{'n': 1, 'result': 'pass'}, {'n': 2, 'result': 'fail'}]
            rows.append(r)
        return json.dumps({'analysis': 'a', 'checks': rows, 'score': 4, 'confidence': 0.95})
    store = Store(tmp_path)
    pipe = ReviewPipeline({'platform-6d-1to5': profile()}, FilterRetriever(kb),
                          DimensionEvaluator(ScriptedBackend(responder, name='t'), store), store, tmp_path,
                          checkpointer=MemorySaver(), require_human=False)
    req = ReviewRequest(case=case(answer=ANS), profile_id='platform-6d-1to5')
    rep = pipe.start('J-STEP', req)['report']
    by = {d['dimension']: d for d in rep['dimensions']}
    assert 'CHECK_STEP_CONFLICT' in by['safety']['review_reasons']
    assert all('QUOTE_NOT_LOCATED' not in d['review_reasons'] for d in rep['dimensions'])   # markup tier: recorded
    assert by['accuracy']['checks'][0]['quote_tier'] == 'markup'
    assert rep['snapshot']['evaluator_prompt'] == 'v3-dimension-2' and rep['snapshot']['locator'] == 'locator-3'


# ---- grouped calls ----------------------------------------------------------------------------------------

def _specs():
    prof = profile()
    return [(prof.dimension('accuracy'), [item('A-1', dim='accuracy')], []),
            (prof.dimension('safety'), [item('S-1', text=STEP_TEXT)], [])]


def test_group_messages_share_one_prefix_and_list_every_dimension():
    msgs = build_messages_group(case(), _specs())
    payload = json.loads(msgs[1]['content'])
    assert [d['key'] for d in payload['dimensions']] == ['accuracy', 'safety']
    assert payload['dimensions'][1]['checks'][0]['steps'] == 2
    assert '"dimensions"' in msgs[0]['content']


def test_group_envelope_errors_fail_the_whole_call():
    with pytest.raises(EvalFailure) as e:
        parse_group(json.dumps({'dimensions': [{'key': 'safety'}]}), case(), _specs())
    assert e.value.code == 'OUTPUT_DIMENSIONS_MISMATCH'
    with pytest.raises(EvalFailure):
        parse_group(json.dumps({'dimensions': [], 'extra': 1}), case(), _specs())


def test_a_bad_dimension_fails_only_itself():
    good = {'key': 'accuracy', 'analysis': 'a', 'checks': [{'id': 'A-1', 'result': 'pass', 'quote': ''}],
            'score': 4, 'confidence': 0.9}
    bad = {'key': 'safety', 'analysis': 'a', 'checks': [], 'score': 4, 'confidence': 0.9}
    out = parse_group(json.dumps({'dimensions': [good, bad]}), case(), _specs())
    assert out['accuracy']['model_score'] == 4 and isinstance(out['safety'], EvalFailure)


def test_evaluate_group_one_call_and_records():
    seq = Seq(['mock'])
    ev = DimensionEvaluator(seq)
    recs = ev.evaluate_group(case(), _specs(), 'J')
    assert seq.calls == 1 and seq.sent[0][1] == 2200          # 2 x 900 + 400
    assert [r['status'] for r in recs] == ['Completed', 'Completed']
    assert {r['call_mode'] for r in recs} == {'grouped'} and recs[0]['usage_shared_by'] == 2


def test_group_failure_marks_every_dimension_and_stops_on_401():
    ev = DimensionEvaluator(Seq(['http401']))
    recs = ev.evaluate_group(case(), _specs(), 'J')
    assert [r['failure_code'] for r in recs] == ['PROVIDER_HTTP_401'] * 2 and ev.stop.is_set()


def test_request_larger_than_the_reservation_is_refused():
    class Limited(Seq):
        class settings:
            max_output_tokens = 1200
    seq = Limited(['mock'])
    recs = DimensionEvaluator(seq).evaluate_group(case(), _specs(), 'J')
    assert seq.calls == 0 and recs[0]['failure_code'] == 'CONFIG_MAX_TOKENS_EXCEEDS_RESERVATION'


def test_pipeline_single_call_mode(tmp_path):
    kb = load_kb(KB_DIR)
    seq = Seq([])
    store = Store(tmp_path)
    pipe = ReviewPipeline({'platform-6d-1to5': profile()}, FilterRetriever(kb), DimensionEvaluator(seq, store),
                          store, tmp_path, checkpointer=MemorySaver(), require_human=False, call_mode='single')
    rep = pipe.start('J-SINGLE', ReviewRequest(case=case(), profile_id='platform-6d-1to5'))['report']
    assert seq.calls == 1 and len(rep['dimensions']) == 6
    assert all(d['status'] == 'Completed' for d in rep['dimensions']) and rep['snapshot']['call_mode'] == 'single'


# ---- LLM situation router -------------------------------------------------------------------------------

def sit(sid, group, when):
    return Situation(situation_id=sid, exclusive_group=group, label=sid, applies_when=when, source=SRC)


SITS = [sit('S:emerg:yes', 'G:emerg', 'An emergency needs immediate care right now.'),
        sit('S:emerg:no', 'G:emerg', 'A minor problem with no need for urgent care.'),
        sit('S:role:hp', 'G:role', 'A doctor or nurse asks about a patient.')]


def router_kb():
    items = [item('M-SAF', mandatory=True), item('I-EY', situations=['S:emerg:yes']),
             item('I-EN', situations=['S:emerg:no']), item('I-HP', dim='communication', situations=['S:role:hp'])]
    return KnowledgeBase(items=items, situations=SITS)


def router_raw(emerg='S:emerg:yes', role='none', ce=0.9, cr=0.6, groups=None):
    g = groups if groups is not None else [{'group': 'G:emerg', 'situation': emerg, 'confidence': ce},
                                           {'group': 'G:role', 'situation': role, 'confidence': cr}]
    return json.dumps({'analysis': 'a', 'groups': g})


def test_router_messages_hold_the_conversation_only_in_fixed_order():
    q = build_query(case(answer='SECRET ANSWER TEXT'), CaseMeta(), profile())
    msgs = build_router_messages(q.messages, list(reversed(SITS)))
    payload = json.loads(msgs[1]['content'])
    assert 'SECRET' not in msgs[1]['content'] and payload['conversation'][0]['content'] == 'I have a headache.'
    assert [g['group'] for g in payload['groups']] == ['G:emerg', 'G:role']
    assert [o['situation'] for o in payload['groups'][0]['options']] == ['S:emerg:no', 'S:emerg:yes']


@pytest.mark.parametrize('raw,code', [
    (router_raw(groups=[{'group': 'G:emerg', 'situation': 'none', 'confidence': 0.7}]), 'OUTPUT_GROUPS_MISMATCH'),
    (router_raw(emerg='S:role:hp'), 'OUTPUT_SITUATION_NOT_IN_GROUP'),
    (router_raw(groups=[{'group': 'G:emerg', 'situation': 'none', 'confidence': 0.7}] * 2), 'OUTPUT_DUPLICATE_GROUP'),
    ('```json {}```', 'OUTPUT_MARKDOWN_FENCE'),
    ('{"analysis": "a"}', 'OUTPUT_SCHEMA_INVALID'),
])
def test_router_protocol_violations(raw, code):
    with pytest.raises(RouterFailure) as e:
        parse_router(raw, SITS)
    assert e.value.code == code


def test_router_selection_rules():
    r = SituationRouter(Seq([router_raw(role='S:role:hp', cr=0.95)]), max_situations=1)
    assert r.route((('user', 'x'),), SITS, 'J')['selected'] == ['S:role:hp']        # highest confidence first
    r = SituationRouter(Seq([router_raw(role='S:role:hp', cr=0.95)]), max_situations=2, min_confidence=0.92)
    assert r.route((('user', 'x'),), SITS, 'J')['selected'] == ['S:role:hp']        # 0.9 is below the floor
    r = SituationRouter(Seq([router_raw(emerg='none', ce=0.8)]))
    assert r.route((('user', 'x'),), SITS, 'J')['selected'] == []


@pytest.mark.parametrize('outcome,code,stops', [('http429', 'PROVIDER_HTTP_429', True),
                                                ('http500', 'PROVIDER_HTTP_500', False),
                                                ('budget', 'BUDGET_BLOCKED', True),
                                                ('not json', 'OUTPUT_JSON_SYNTAX', False)])
def test_router_failures(outcome, code, stops):
    r = SituationRouter(Seq([outcome]))
    rec = r.route((('user', 'x'),), SITS, 'J')
    assert rec['status'] == 'Failed' and rec['failure_code'] == code and rec['selected'] == []
    assert r.stop.is_set() is stops


def test_router_does_not_call_when_stopped_or_without_candidates():
    seq = Seq([])
    r = SituationRouter(seq)
    assert r.route((('user', 'x'),), [], 'J')['failure_code'] == 'NO_CANDIDATE_SITUATIONS'
    r.stop.set()
    assert r.route((('user', 'x'),), SITS, 'J')['status'] == 'NotRun' and seq.calls == 0


def _llm_retriever(tmp_path, outcomes):
    kb = router_kb()
    emb = HashEmbedder()
    seq = Seq(outcomes)
    return HybridRetriever(kb, open_index(kb, emb, tmp_path / 'idx'), emb, situation_policy='llm',
                           router=SituationRouter(seq)), seq


def test_llm_policy_applies_the_router_choice(tmp_path):
    r, seq = _llm_retriever(tmp_path, [router_raw()])
    res = r.retrieve(build_query(case(), CaseMeta(), profile()), job_id='J')
    ids = {x.item.item_id for x in res.items}
    assert 'I-EY' in ids and 'I-EN' not in ids and seq.calls == 1
    assert res.trace['situation_source'] == 'llm_router' and res.trace['situation_router']['status'] == 'Completed'


def test_metadata_situations_skip_the_router(tmp_path):
    r, seq = _llm_retriever(tmp_path, [router_raw()])
    res = r.retrieve(build_query(case(), CaseMeta(situations=['S:emerg:no']), profile()), job_id='J')
    assert seq.calls == 0 and res.trace['applied_situations'] == ['S:emerg:no']


def test_router_failure_applies_nothing_and_flags_every_dimension(tmp_path):
    kb = router_kb()
    emb = HashEmbedder()
    seq = Seq(['http500'])
    store = Store(tmp_path)
    ev = DimensionEvaluator(ScriptedBackend(mock_responder_v3, name='mock'), store)
    retr = HybridRetriever(kb, open_index(kb, emb, tmp_path / 'idx'), emb, situation_policy='llm',
                           router=SituationRouter(seq, store, stop=ev.stop))
    pipe = ReviewPipeline({'platform-6d-1to5': profile()}, retr, ev, store, tmp_path, checkpointer=MemorySaver(),
                          require_human=False)
    rep = pipe.start('J-RF', ReviewRequest(case=case(), profile_id='platform-6d-1to5'))['report']
    assert rep['snapshot']['retrieval_trace']['situation_source'] == 'llm_router_failed'
    assert all('SITUATION_ROUTER_FAILED' in d['review_reasons'] for d in rep['dimensions'])
    assert not any(r['item']['situations'] for r in rep['retrieved_items'])


def test_llm_policy_requires_a_router(tmp_path):
    kb = router_kb()
    emb = HashEmbedder()
    with pytest.raises(ValueError, match='needs a SituationRouter'):
        HybridRetriever(kb, open_index(kb, emb, tmp_path), emb, situation_policy='llm')


def test_mock_router_responder_follows_the_contract():
    q = build_query(case(user='This is an emergency, help'), CaseMeta(), profile())
    raw = mock_router_responder(build_router_messages(q.messages, SITS))
    assert [c['group'] for c in parse_router(raw, SITS)['choices']] == ['G:emerg', 'G:role']


# ---- CLI plan with the router: counted, never called ------------------------------------------------------

def test_paid_plan_counts_the_router_call_without_making_it(tmp_path, monkeypatch, capsys):
    shutil.copytree(KB_DIR, tmp_path / 'kb/v3')
    shutil.copytree(REPO / 'examples/v3', tmp_path / 'examples/v3')
    (tmp_path / 'jury-v2.json').write_text(json.dumps({'judges': [{'name': 'deepseek-pro', 'kind': 'deepseek',
                                                                   'model': 'deepseek-v4-pro', 'price_in': 13.2,
                                                                   'price_out': 39.6}]}))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key-not-real')
    monkeypatch.setenv('JURY_PRICING_VALID_UNTIL', '2099-01-01')
    cli = load_script('run_review_v3')
    base = ['--root', str(tmp_path), '--case', 'examples/v3/demo-case.json', '--judge', 'deepseek-pro',
            '--embedder', 'hash']
    assert cli.main(base + ['--situation-policy', 'llm']) == 0
    out = capsys.readouterr().out
    plan = json.loads(out[:out.index('Plan only')])
    assert plan['router_calls'] == 1 and plan['planned_calls'] == 7 and plan['evaluation_calls'] == 6
    assert not (tmp_path / 'reports').exists()
    assert cli.main(base + ['--call-mode', 'single']) == 0
    out = capsys.readouterr().out
    single = json.loads(out[:out.index('Plan only')])
    assert single['planned_calls'] == 1 and single['router_calls'] == 0
    assert single['cost_upper_bound_cny'] < plan['cost_upper_bound_cny']


def test_cli_mock_run_with_router_and_single_call(tmp_path, capsys):
    shutil.copytree(KB_DIR, tmp_path / 'kb/v3')
    shutil.copytree(REPO / 'examples/v3', tmp_path / 'examples/v3')
    cli = load_script('run_review_v3')
    assert cli.main(['--root', str(tmp_path), '--case', 'examples/v3/demo-case.json', '--embedder', 'hash',
                     '--situation-policy', 'llm', '--call-mode', 'single', '--no-human', '--job-id', 'R1']) == 0
    rep = json.loads((tmp_path / 'reports/v3-reviews/R1/report.json').read_text())
    assert rep['snapshot']['retrieval_trace']['situation_source'] == 'llm_router'
    assert rep['snapshot']['call_mode'] == 'single'


# ---- router evaluation script (mock, synthetic data) ---------------------------------------------------------

def test_router_eval_script_offline(tmp_path):
    shutil.copytree(KB_DIR, tmp_path / 'kb/v3')
    (tmp_path / 'data/prepared').mkdir(parents=True)
    rows = []
    for k, cat in enumerate(['cluster:emergency_referrals_emergent_emergency_behavior',
                             'cluster:hedging_no-uncertainty_hedges', 'cluster:communication_health-professional_tailored',
                             'cluster:global_health_context-matters-is-clear_aligned_accurate'] * 3):
        rows.append({'category': cat, 'case': {'case_id': f'C{k}', 'group_id': f'G{k}', 'task': 't',
                                               'candidate_output': 'answer', 'context': [
                                                   {'role': 'user', 'content': f'question {k} emergent'}]}})
    (tmp_path / 'data/prepared/dev.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    script = load_script('eval_situation_router_v3')
    for split in ('A', 'B'):
        assert script.main(['--root', str(tmp_path), '--judge', 'mock', '--n', '50', '--split', split,
                            '--no-hybrid']) == 0
        m = [json.loads(p.read_text()) for p in (tmp_path / 'reports').glob(f'router-v3-{split}-*/metrics.json')][0]
        assert m['planned'] >= 1 and m['completed'] == m['planned'] and m['router_prompt'] == 'v3-situation-router-1'
    pred = next((tmp_path / 'reports').glob('router-v3-A-*/predictions.jsonl')).read_text()
    assert 'answer' not in json.loads(pred.splitlines()[0])            # no answer text stored


# ---- alt-test status -----------------------------------------------------------------------------------

def test_alt_test_not_evaluable_is_distinguished_from_failed():
    items = [{'llm': True, 'humans': {'a': True, 'b': False}} for _ in range(5)]
    r = A.alt_test(items, min_items_per_annotator=30)
    assert r['status'] == 'not_evaluable' and r['passed'] is False and r['eligible_annotators'] == 0
