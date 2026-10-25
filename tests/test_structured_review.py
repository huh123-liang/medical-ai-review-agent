from dataclasses import replace
from datetime import date,timedelta
import hashlib
import json
from pathlib import Path
import httpx
import pytest
from pydantic import ValidationError

from medical_jury.config import ROOT
from medical_jury.domain import Message
from medical_jury.evidence_ids import catalog,EvidenceIdError
from medical_jury.fact_sources import SPECS,load_cards,retrieve_facts,FactSourceError,VERSION
from medical_jury.structured_review import messages,decode
from medical_jury.provider import parse_judgment,DeepSeekJudge,ProviderError,build_messages
from medical_jury.task_spec_v2 import TaskScopeError
from medical_jury.workflow import Engine


@pytest.fixture
def fact_root(tmp_path):
    folder=tmp_path/'data/facts/v1';folder.mkdir(parents=True)
    records=[]
    for spec in SPECS:
        raw=b'OFFLINE UNIT FIXTURE NOT AN OFFICIAL MEDICAL SOURCE'
        name=spec['id']+'.'+spec['format'];(folder/name).write_bytes(raw)
        records.append({'id':spec['id'],'url':spec['url'],'raw_file':name,
                        'sha256':hashlib.sha256(raw).hexdigest(),'retrieved_at':'unit-fixture'})
    (folder/'manifest.json').write_text(json.dumps({'version':VERSION,'reviewed_at':date.today().isoformat(),
        'review_until':(date.today()+timedelta(days=1)).isoformat(),'records':records}),encoding='utf-8')
    return tmp_path


def obj(**kwargs):
    return {'judgment':'met','reason':'unit observation','answer_ids':[], 'context_ids':[],
            'review_required':True,'task_kind':'other','fact_checks':[],**kwargs}


def test_facts_retrieved_only_by_topic_not_case_or_labels(fact_root,request_model):
    c=request_model.case.model_copy(update={'candidate_output':'aripiprazole dosing in hepatic impairment'})
    found=retrieve_facts(c,fact_root)
    assert [f['id'] for f in found['cards']]==['F-arip-hepatic-v1']
    assert retrieve_facts(c.model_copy(update={'case_id':'different','group_id':'new'}),fact_root)==found
    assert 'not_clinician_approved' in found['cards'][0]['review_status']


def test_no_relevant_fact_is_not_falsehood(fact_root,request_model):
    c=request_model.case.model_copy(update={'candidate_output':'A hypothetical plain text note.'})
    r=retrieve_facts(c,fact_root)
    assert r['cards']==[] and r['coverage']=='not_covered'


def test_formulation_exclusion(fact_root,request_model):
    c=request_model.case.model_copy(update={'candidate_output':'Abilify Maintena hepatic impairment'})
    assert not retrieve_facts(c,fact_root)['cards']


def test_source_hash_tampering_and_expiry_block(fact_root):
    folder=fact_root/'data/facts/v1'
    load_cards(fact_root)
    (folder/(SPECS[0]['id']+'.html')).write_bytes(b'tampered')
    with pytest.raises(FactSourceError):load_cards(fact_root)
    p=folder/'manifest.json';data=json.loads(p.read_text());data['review_until']='2000-01-01';p.write_text(json.dumps(data))
    with pytest.raises(FactSourceError):load_cards(fact_root)


def test_source_path_traversal_blocked(fact_root):
    p=fact_root/'data/facts/v1/manifest.json';data=json.loads(p.read_text());data['records'][0]['raw_file']='../../outside'
    p.write_text(json.dumps(data))
    with pytest.raises(FactSourceError):load_cards(fact_root)


def test_exact_id_output_remains_public_five_field_judgment(request_model):
    row=next(r for r in catalog(request_model.case) if r['kind']=='answer')
    trace={};review={}
    result=parse_judgment(json.dumps(obj(answer_ids=[row['id']])),request_model.case,True,
        profile='evidence-id-v1',rubric=request_model.rubrics[0],on_anchored=trace.update,on_structured=review.update)
    assert result.answer_evidence==[row['text']]
    assert set(result.model_dump())=={'judgment','reason','answer_evidence','context_evidence','review_required'}
    assert trace['spans'][0]['method']=='provided-source-id'
    assert review['medical_claims_fully_verified'] is False


def test_task_mismatch_is_failed_not_auto_regraded(request_model):
    c=request_model.case.model_copy(update={'context':[Message(role='user',content='Ask about further tests. rewrite')]})
    with pytest.raises(TaskScopeError):decode(obj(task_kind='clinical_request'),c,request_model.rubrics[0],'task-evidence-v2')
    result,_,_=decode(obj(task_kind='rewrite',judgment='not_met'),c,request_model.rubrics[0],'task-evidence-v2')
    assert result.judgment=='not_met'


def test_unverifiable_strict_facts_require_review(request_model):
    rubric=request_model.rubrics[0].model_copy(update={'text':'An extraordinarily high bar for every claim.'})
    with pytest.raises(FactSourceError):decode(obj(review_required=False),request_model.case,rubric,'evidence-id-v1')


def test_bound_fact_check_with_real_source_provenance(fact_root,request_model):
    c=request_model.case.model_copy(update={'candidate_output':'Aripiprazole hepatic impairment requires reduction for everyone.'})
    aid=next(r['id'] for r in catalog(c) if r['kind']=='answer')
    check={'claim_id':aid,'fact_id':'F-arip-hepatic-v1','relation':'conflicts','scope_note':'Oral label only; formulation/time need review.'}
    result,_,review=decode(obj(judgment='not_met',answer_ids=[aid],fact_checks=[check]),c,request_model.rubrics[0],'grounded-review-v1',fact_root)
    assert result.judgment=='not_met' and review['fact_checks'][0]['claim_id']==aid
    assert review['fact_evidence'][0]['source_sha256']
    assert not review['medical_claims_fully_verified']
    with pytest.raises(FactSourceError):decode(obj(answer_ids=[aid],fact_checks=[{**check,'fact_id':'F-imagined'}]),c,request_model.rubrics[0],'grounded-review-v1',fact_root)
    with pytest.raises(FactSourceError):decode(obj(fact_checks=[check]),c,request_model.rubrics[0],'grounded-review-v1',fact_root)


def test_wire_bounds_and_extra_fields_not_silently_repaired(request_model):
    with pytest.raises(ValidationError):decode(obj(extra='ignored?'),request_model.case,request_model.rubrics[0],'evidence-id-v1')
    with pytest.raises(ValidationError):decode(obj(answer_ids=['a']*4),request_model.case,request_model.rubrics[0],'evidence-id-v1')


def test_actual_http_adapter_and_audit_stay_offline(settings,store,request_model):
    seen=[]
    def handler(request):
        payload=json.loads(request.content);data=json.loads(payload['messages'][1]['content'])
        assert 'tools' not in payload and 'labels' not in data
        seen.append(data)
        raw=obj(answer_ids=[data['candidate_segments'][0]['id']])
        return httpx.Response(200,json={'model':'OFFLINE-MOCK-PRO-ENVELOPE','usage':{'prompt_tokens':100,'completion_tokens':80},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(raw)}}]})
    s=replace(settings,api_key='unit-placeholder-not-a-real-key',judge_profile='task-evidence-v2')
    def factory(_):return DeepSeekJudge(s,store,httpx.Client(transport=httpx.MockTransport(handler)))
    job=Engine(s,store,judge_factory=factory).run(request_model)
    assert job['status']=='Completed' and seen
    assert job['report']['structured_reviews'][0]['wire_protocol']=='structured-ids-v1'
    assert job['report']['weighted_score'] is None


def test_real_reviewed_snapshots_have_integrity():
    # Read-only integration check against already downloaded public sources. No network or API.
    if not (ROOT / 'data/facts/v1/manifest.json').is_file():
        pytest.skip('Optional author fact-source snapshots are not redistributed; no evidence fabricated.')
    cards=load_cards(ROOT)
    assert len(cards)==4 and all(c['source_sha256'] for c in cards)
