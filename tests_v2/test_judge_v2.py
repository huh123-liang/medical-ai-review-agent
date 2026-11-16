import json

import httpx
import pytest

from medical_jury.budget import BudgetExceeded
from medical_jury.judge_v2 import (SYSTEM_V2, DeepSeekBackend, JudgeV2, LedgerBudget, ProtocolFailure,
                                   ScriptedBackend, build_messages_v2, met_probability, mock_responder, parse_v2)

GOOD = '{"analysis": "Asks about fever; answer gives 15 mg/kg.", "verdict": "met", "confidence": 0.8, "evidence": ["15 mg/kg every 6 hours"]}'


def test_messages_hide_labels_and_put_reasoning_first(case, rubric):
    msgs = build_messages_v2(case, rubric)
    payload = json.loads(msgs[1]['content'])
    assert set(payload) == {'conversation', 'answer_to_evaluate', 'criterion'}
    assert SYSTEM_V2.index('"analysis"') < SYSTEM_V2.index('"verdict"')
    assert 'label' not in msgs[1]['content'].lower()


def test_parse_good_output_and_locate_evidence(case):
    out = parse_v2(GOOD, case)
    assert out['verdict'] == 'met' and out['evidence_status'] == 'all_located_in_answer'


def test_unlocated_quote_is_recorded_not_fatal(case):
    raw = GOOD.replace('15 mg/kg every 6 hours', 'invented quote')
    assert parse_v2(raw, case)['evidence_status'] == 'some_unlocated_in_answer'


@pytest.mark.parametrize('raw,code', [
    ('{"verdict": "met", "analysis": "x", "confidence": 0.7, "evidence": []}', 'OUTPUT_VERDICT_BEFORE_ANALYSIS'),
    ('{"analysis": "x", "analysis": "y", "verdict": "met", "confidence": 0.7}', 'OUTPUT_JSON_DUPLICATE_KEYS'),
    ('```json\n{}\n```', 'OUTPUT_MARKDOWN_FENCE'),
    ('{"analysis": "x", "verdict": "yes", "confidence": 0.7}', 'OUTPUT_SCHEMA_INVALID'),
    ('{"analysis": "x", "verdict": "met", "confidence": 1.7}', 'OUTPUT_SCHEMA_INVALID'),
    ('{"analysis": "x", "verdict": "met", "confidence": 0.7, "extra": 1}', 'OUTPUT_SCHEMA_INVALID'),
    ('{"analysis": "x", "verdict": "met", "confidence": 0.7, "evidence": ["a","b","c"]}', 'OUTPUT_SCHEMA_INVALID'),
    ('{"analysis": "x", "verdict": "met"', 'OUTPUT_JSON_SYNTAX'),
    ('[1]', 'OUTPUT_NOT_OBJECT'),
])
def test_protocol_failures_are_not_repaired(case, raw, code):
    with pytest.raises(ProtocolFailure) as e:
        parse_v2(raw, case)
    assert e.value.code == code


def test_met_probability_mapping():
    assert met_probability('met', 0.8) == 0.8
    assert met_probability('not_met', 0.8) == pytest.approx(0.2)
    assert met_probability('met', 0.2) == 0.5  # self-contradicting confidence is clipped


def test_judge_records_failures_instead_of_dropping(case, rubric, store):
    ok = JudgeV2(ScriptedBackend(lambda m: GOOD), store).judge(case, rubric, 'J1')
    assert ok['status'] == 'Completed' and ok['p_met'] == 0.8
    truncated = JudgeV2(ScriptedBackend(lambda m: GOOD[:40], finish_reason='length'), store).judge(case, rubric, 'J1')
    assert truncated['status'] == 'Failed' and truncated['failure_code'] == 'INCOMPLETE_MODEL_OUTPUT'
    empty = JudgeV2(ScriptedBackend(lambda m: '  '), store).judge(case, rubric, 'J1')
    assert empty['failure_code'] == 'EMPTY_MODEL_OUTPUT'
    events = [e['event'] for e in store.events('J1')]
    assert events.count('judge_v2_result') == 3
    assert all('analysis' not in e['detail_json'] for e in store.events('J1'))


def test_mock_responder_is_deterministic_and_valid(case, rubric):
    msgs = build_messages_v2(case, rubric)
    assert mock_responder(msgs) == mock_responder(msgs)
    assert parse_v2(mock_responder(msgs), case)['analysis'].startswith('OFFLINE MOCK')


def _transport(body, status=200):
    seen = {}

    def handler(request):
        seen['json'] = json.loads(request.content)
        seen['url'] = str(request.url)
        return httpx.Response(status, json=body)
    return httpx.MockTransport(handler), seen


def test_deepseek_backend_settles_usage_and_sends_bounded_request(case, rubric, settings, store):
    body = {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 500, 'completion_tokens': 100},
            'choices': [{'finish_reason': 'stop', 'message': {'content': GOOD}}]}
    transport, seen = _transport(body)
    backend = DeepSeekBackend(settings, store, 5.0, client=httpx.Client(transport=transport), ceiling_cny=30)
    rec = JudgeV2(backend, store).judge(case, rubric, 'J2')
    assert rec['status'] == 'Completed'
    assert seen['url'] == settings.provider_url and seen['json']['temperature'] == 0
    assert seen['json']['max_tokens'] == 800
    with store.connect() as db:
        row = db.execute('SELECT state, accounted FROM costs').fetchone()
    assert row['state'] == 'settled' and row['accounted'] == pytest.approx((500 * 3 + 100 * 12) / 1e6)


def test_deepseek_backend_http_error_keeps_reservation(case, rubric, settings, store):
    transport, _ = _transport({'error': 'x'}, status=503)
    backend = DeepSeekBackend(settings, store, 5.0, client=httpx.Client(transport=transport), ceiling_cny=30)
    rec = JudgeV2(backend, store).judge(case, rubric, 'J3')
    assert rec['failure_code'] == 'PROVIDER_HTTP_503'
    with store.connect() as db:
        row = db.execute('SELECT state, accounted, reserved FROM costs').fetchone()
    assert row['state'] == 'unknown_or_failed' and row['accounted'] == row['reserved']


def test_ledger_budget_ceiling_and_run_limit(settings, store):
    b = LedgerBudget(settings, store, ceiling_cny=0.01)
    with pytest.raises(BudgetExceeded):
        b.reserve('J', 0.02, 10)
    with pytest.raises(BudgetExceeded):
        LedgerBudget(settings, store, ceiling_cny=30).reserve('J', 0.02, 0.01)
    assert LedgerBudget(settings, store, ceiling_cny=30).reserve('J', 0.02, 10)


def test_paid_backend_requires_reviewed_settings(settings, store):
    from dataclasses import replace
    with pytest.raises(ValueError):
        DeepSeekBackend(replace(settings, api_key=''), store, 1.0, ceiling_cny=10)
