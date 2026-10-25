from dataclasses import replace
import json

import httpx
import pytest

from medical_jury.budget import Budget
from medical_jury.provider import DeepSeekJudge, PROMPT_VERSION, ProviderError


def result_payload():
    return {'judgment':'met','reason':'Fixture, no clinical claim.','answer_evidence':[],
            'context_evidence':[],'review_required':False}


def call(settings,store,request_model,raw,*,finish='stop',public=False):
    # These tests explicitly preserve the historical v2 parser contract.
    s=replace(settings,api_key='do-not-leak-unit-test-credential',evidence_policy='strict-v2',judge_profile='v2-strict')
    case=request_model.case.model_copy(update={'source':'HealthBench meta-evaluation',
                   'source_version':'healthbench-meta-2025-05-07'}) if public else request_model.case
    def handler(request):
        data=json.loads(request.content)
        assert data['response_format']=={'type':'json_object'}
        assert 'all five keys' in data['messages'][0]['content']
        return httpx.Response(200,json={'usage':{'prompt_tokens':100,'completion_tokens':100},
                 'choices':[{'finish_reason':finish,'message':{'content':raw}}]})
    client=httpx.Client(transport=httpx.MockTransport(handler))
    judge=DeepSeekJudge(s,store,client)
    job=store.create_job(request_model)
    try:
        value=judge.judge(case,request_model.rubrics[0],job,1,True)
        return value,job
    finally:
        client.close()


@pytest.mark.parametrize('change',[{'review_required':'false'}, {'review_required':0},
        {'answer_evidence':'quote'}, {'context_evidence':None}, {'reason':23}, {'judgment':True}])
def test_strict_wire_types_not_coerced(settings,store,request_model,change):
    data=result_payload()|change
    with pytest.raises(ProviderError,match='OUTPUT_SCHEMA_INVALID'):
        call(settings,store,request_model,json.dumps(data))
    assert Budget(settings,store).total()['paid_attempts']==1


@pytest.mark.parametrize('field',['judgment','reason','answer_evidence','context_evidence','review_required'])
def test_all_wire_fields_are_required(settings,store,request_model,field):
    data=result_payload()
    data.pop(field)
    with pytest.raises(ProviderError,match='OUTPUT_SCHEMA_INVALID'):
        call(settings,store,request_model,json.dumps(data))


@pytest.mark.parametrize('raw,code',[
    ('```json\n{}\n```','OUTPUT_JSON_SYNTAX'),
    ('{"judgment":"met","judgment":"not_met"}','OUTPUT_JSON_DUPLICATE_KEYS'),
    ('[]','OUTPUT_SCHEMA_INVALID'),('','EMPTY_MODEL_OUTPUT')])
def test_failure_codes_are_distinct(settings,store,request_model,raw,code):
    with pytest.raises(ProviderError,match=code):
        call(settings,store,request_model,raw)
    files=list((settings.root/'data/private/provider-diagnostics').glob('*/*.json'))
    assert len(files)==1
    assert json.loads(files[0].read_text(encoding='utf-8'))['error_code']==code


def test_extra_keys_and_input_values_not_echoed(settings,store,request_model):
    data=result_payload()|{'do-not-leak-unit-test-credential':'not allowed'}
    with pytest.raises(ProviderError):
        call(settings,store,request_model,json.dumps(data))
    log=next((settings.root/'data/private/provider-diagnostics').glob('*/*.json')).read_text(encoding='utf-8')
    assert 'do-not-leak-unit-test-credential' not in log
    assert '<unknown_field>' in log
    assert 'excerpt' not in log


def test_public_failure_excerpt_is_redacted_and_bounded(settings,store,request_model):
    raw='invalid do-not-leak-unit-test-credential sk-'+'a'*32+' person@example.com '+('x'*3000)
    with pytest.raises(ProviderError,match='SYNTAX'):
        call(settings,store,request_model,raw,public=True)
    item=json.loads(next((settings.root/'data/private/provider-diagnostics').glob('*/*.json')).read_text(encoding='utf-8'))
    excerpt=item['redacted_output_excerpt']
    assert len(excerpt)<=1200
    assert 'do-not-leak' not in excerpt and 'a'*32 not in excerpt and 'person@example.com' not in excerpt
    assert item['output_chars']==len(raw) and len(item['output_sha256'])==64


def test_private_failure_has_no_output_excerpt(settings,store,request_model):
    with pytest.raises(ProviderError):
        call(settings,store,request_model,'invalid PRIVATE-CLINICAL-CONTENT')
    log=next((settings.root/'data/private/provider-diagnostics').glob('*/*.json')).read_text(encoding='utf-8')
    assert 'PRIVATE-CLINICAL-CONTENT' not in log and 'excerpt' not in log


def test_fabricated_quote_is_not_repaired_or_accepted(settings,store,request_model):
    data=result_payload()|{'answer_evidence':['not present in the fixture']}
    with pytest.raises(ProviderError,match='EVIDENCE_OR_PROTOCOL'):
        call(settings,store,request_model,json.dumps(data))
    assert Budget(settings,store).total()['paid_attempts']==1


def test_v2_valid_output_is_audited(settings,store,request_model):
    output,job=call(settings,store,request_model,json.dumps(result_payload()))
    assert output.judgment=='met'
    event=next(e for e in store.events(job) if e['event']=='provider_output_validated')
    assert PROMPT_VERSION in json.dumps(event)


def test_evidence_failure_identifies_field_index_and_hash_without_private_text(settings,store,request_model):
    data=result_payload()|{'reason':'x'*1500,'answer_evidence':['PRIVATE-MISMATCH-STRING']}
    with pytest.raises(ProviderError):
        call(settings,store,request_model,json.dumps(data))
    item=json.loads(next((settings.root/'data/private/provider-diagnostics').glob('*/*.json')).read_text(encoding='utf-8'))
    issue=item['details']['issues'][0]
    assert issue['field']=='answer_evidence' and issue['index']==0
    assert len(issue['quote_sha256'])==64
    assert 'PRIVATE-MISMATCH-STRING' not in json.dumps(item)


def test_public_evidence_fields_remain_diagnosable_when_reason_truncates_raw_excerpt(settings,store,request_model):
    data=result_payload()|{'reason':'x'*1500,'answer_evidence':['public fixture missing quote']}
    with pytest.raises(ProviderError):
        call(settings,store,request_model,json.dumps(data),public=True)
    item=json.loads(next((settings.root/'data/private/provider-diagnostics').glob('*/*.json')).read_text(encoding='utf-8'))
    assert item['excerpt_truncated']
    assert item['public_evidence_excerpts']['answer_evidence']==['public fixture missing quote']
