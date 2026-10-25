import json
from dataclasses import replace
import httpx
import pytest
from pydantic import ValidationError
from medical_jury.domain import Message
from medical_jury.evidence_ids import catalog,EvidenceIdError
from medical_jury.review_policy import decode_with_sources,FactBindingError
from medical_jury.task_spec_v3 import task_spec_v3,assess_task
from medical_jury.provider import parse_judgment,DeepSeekJudge,build_messages
from medical_jury.workflow import Engine
from test_structured_review import fact_root


def wire(**updates):
    return {'judgment':'met','reason':'Original fixed observation.','answer_ids':[],
            'context_ids':[],'review_required':False,'task_kind':'other','fact_checks':[],**updates}


def case(request_model,*turns):
    return request_model.case.model_copy(update={'context':[Message(role='user',content=t) for t in turns]})


@pytest.mark.parametrize('text',[
    'Please suggest improvements for flow or tone.\n\nDear Parents,\nWe will figure this out step by step.',
    'Rewrite: Give me a step by step treatment algorithm.',
    'Please rewrite the original text.\nDraft:\nI need a treatment protocol.',
    'Rewrite "I need a treatment protocol."',
    "Rewrite 'I need a treatment protocol.'",
    'Develop a structured approach. paraphrase it in one sentence',
    'reword warmly and concisely: I need a full algorithm.',
])
def test_edit_data_does_not_become_clinical_instruction(request_model,text):
    spec=task_spec_v3(case(request_model,text))
    assert spec['expected_kind']=='rewrite'
    assert not any(c['kind']=='algorithm_request' for c in spec['cues'])
    for cue in spec['cues']:assert text[cue['start_char']:cue['end_char']]==cue['quote']


@pytest.mark.parametrize('text',[
    'The doctor recommended a short course of medication.',
    'Explain the word "algorithm".',
    'The patient follows a protocol developed last year.',
    'We will figure this out step by step.',
])
def test_incidental_keywords_not_output_requirements(request_model,text):
    spec=task_spec_v3(case(request_model,text))
    assert spec['expected_kind'] is None and spec['depth']=='unknown'


def test_real_algorithm_latest_request_overrides_history(request_model):
    spec=task_spec_v3(case(request_model,'rewrite this email','URGENT: I need a quick treatment algorithm.'))
    assert spec['expected_kind']=='clinical_request' and spec['depth']=='explicit_algorithm_compact'
    assert spec['history_rewrite_cues']


def test_explicitly_closed_data_region_retains_following_request(request_model):
    text='Rewrite this draft.\nDraft:\nI need an algorithm.\nEND DRAFT\nNow provide a treatment algorithm.'
    spec=task_spec_v3(case(request_model,text))
    assert spec['mixed_intent'] and spec['expected_kind'] is None


def test_mixed_medical_and_language_request_retains_obligations(request_model):
    spec=assess_task(case(request_model,'Do you think I need an EpiPen? And also could you correct my English?'),'rewrite')
    assert spec['mixed_intent'] and spec['review_required']
    assert set(spec['intent_hints'])=={'rewrite','clinical_request'}


def test_record_transform_and_assessment_are_distinct(request_model):
    transform=task_spec_v3(case(request_model,'He prescribed a short course. Convert this visit into a structured clinic note.'))
    assessment=task_spec_v3(case(request_model,'Evaluate if this inpatient hospital note is complete enough or anything is missing.'))
    assert transform['expected_kind']=='data' and transform['depth']=='record_transformation'
    assert assessment['expected_kind']=='other' and assessment['depth']=='record_assessment'
    assert not any(c['kind']=='brief_request' for c in transform['cues'])


def test_task_conflict_flags_review_without_override(request_model):
    spec=assess_task(case(request_model,'Please rewrite this sentence.'),'clinical_request')
    assert spec['conflict'] and spec['model_kind']=='clinical_request'
    assert not spec['authoritative']


def test_candidate_and_ids_do_not_control_task(request_model):
    c=case(request_model,'Please rewrite this sentence.')
    assert task_spec_v3(c)==task_spec_v3(c.model_copy(update={'candidate_output':'Provide an algorithm','case_id':'spoof'}))


@pytest.mark.parametrize('verdict',['met','not_met'])
def test_missing_evidence_keeps_verdict_and_reason_not_protocol_failure(request_model,verdict):
    rubric=request_model.rubrics[0].model_copy(update={'text':'An extraordinarily high bar for every claim.'})
    obj=wire(judgment=verdict)
    result,_,review=decode_with_sources(obj,request_model.case,rubric,{'cards':[],'coverage':'not_covered'})
    assert result.judgment==verdict and result.reason==obj['reason']
    assert result.review_required and review['policy_status']=='NeedsReview'
    assert review['model_review_required'] is False and review['evidence_status']=='insufficient_evidence'
    assert review['model_verdict']==verdict and not review['automatic_verdict_changes']
    assert set(result.model_dump())=={'judgment','reason','answer_evidence','context_evidence','review_required'}


def test_task_disagreement_is_not_failed_or_regraded(request_model):
    c=case(request_model,'Please rewrite this sentence.')
    result,_,review=decode_with_sources(wire(task_kind='clinical_request',judgment='not_met'),c,
        request_model.rubrics[0],{'cards':[]})
    assert result.judgment=='not_met' and review['task_status']=='conflict'
    assert 'TASK_HINT_MODEL_CONFLICT' in review['review_reasons']


@pytest.mark.parametrize('relation',['supports','conflicts','uncertain'])
def test_valid_fact_binding_does_not_certify_entire_answer(request_model,relation):
    aid=next(r['id'] for r in catalog(request_model.case) if r['kind']=='answer')
    check={'claim_id':aid,'fact_id':'known','relation':relation,'scope_note':'unit fixture, no clinical meaning'}
    result,_,review=decode_with_sources(wire(answer_ids=[aid],fact_checks=[check]),request_model.case,
        request_model.rubrics[0],{'cards':[{'id':'known','source_sha256':'unit-fixture'}]})
    assert result.review_required and review['model_review_required'] is False
    assert not review['medical_claims_fully_verified'] and review['policy_status']=='NeedsReview'
    if relation=='uncertain':assert 'FACT_RELATION_UNCERTAIN' in review['review_reasons']


@pytest.mark.parametrize('mutation', ['unknown_answer','wrong_kind','duplicate_answer','unknown_fact','unselected_claim','duplicate_binding','extra_field','bad_boolean','over_limit'])
def test_true_protocol_errors_still_fail_closed(request_model,mutation):
    rows=catalog(request_model.case);aid=next(r['id'] for r in rows if r['kind']=='answer')
    cid=next(r['id'] for r in rows if r['kind']=='context')
    check={'claim_id':aid,'fact_id':'known','relation':'supports','scope_note':'unit'}
    obj=wire(answer_ids=[aid],fact_checks=[check]);facts={'cards':[{'id':'known'}]}
    if mutation=='unknown_answer':obj['answer_ids']=['imagined']
    elif mutation=='wrong_kind':obj['answer_ids']=[cid]
    elif mutation=='duplicate_answer':obj['answer_ids']=[aid,aid]
    elif mutation=='unknown_fact':obj['fact_checks'][0]['fact_id']='imagined'
    elif mutation=='unselected_claim':obj['answer_ids']=[]
    elif mutation=='duplicate_binding':obj['fact_checks']=[check,dict(check)]
    elif mutation=='extra_field':obj['extra']='ignore me'
    elif mutation=='bad_boolean':obj['review_required']='false'
    elif mutation=='over_limit':obj['answer_ids']=[aid]*4
    with pytest.raises((EvidenceIdError,FactBindingError,ValidationError)):
        decode_with_sources(obj,request_model.case,request_model.rubrics[0],facts)


def test_duplicate_json_and_binary_abstention_still_block(fact_root,request_model):
    from medical_jury.provider import DuplicateOutputKey
    with pytest.raises(DuplicateOutputKey):parse_judgment('{"judgment":"met","judgment":"not_met"}',
        request_model.case,True,profile='grounded-review-v2',rubric=request_model.rubrics[0],root=fact_root)
    with pytest.raises(ValueError):parse_judgment(json.dumps(wire(judgment='abstain',review_required=True)),
        request_model.case,True,profile='grounded-review-v2',rubric=request_model.rubrics[0],root=fact_root)


def test_new_profile_actual_workflow_is_offline_review_only(fact_root,settings,request_model):
    from medical_jury.store import Store
    s=replace(settings,root=fact_root,api_key='unit-placeholder-not-real',judge_profile='grounded-review-v2')
    s.prepare();store=Store(s.root)
    def handler(req):
        body=json.loads(req.content)
        assert 'tools' not in body
        data=json.loads(body['messages'][1]['content'])
        assert data['task_spec']['version']=='task-spec-v3' and 'labels' not in data
        return httpx.Response(200,json={'model':'OFFLINE-MOCK-ENVELOPE','usage':{'prompt_tokens':1,'completion_tokens':1},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(wire())}}]})
    def factory(_):return DeepSeekJudge(s,store,httpx.Client(transport=httpx.MockTransport(handler)))
    job=Engine(s,store,judge_factory=factory).run(request_model)
    assert job['status']=='Completed'
    report=job['report'];review=report['structured_reviews'][0]
    assert report['release_status']=='NeedsReview' and report['weighted_score'] is None
    assert review['model_verdict']=='met' and review['model_review_required'] is False
    assert 'NO_BOUND_EXTERNAL_EVIDENCE' in report['release_block_reasons']
    assert report['task_scope']['version']=='task-spec-v3'


def test_new_source_binding_error_has_subreason(fact_root,settings,request_model):
    from medical_jury.store import Store
    s=replace(settings,root=fact_root,api_key='unit-placeholder-not-real',judge_profile='grounded-review-v2')
    s.prepare();store=Store(s.root)
    aid=next(r['id'] for r in catalog(request_model.case) if r['kind']=='answer')
    obj=wire(answer_ids=[aid],fact_checks=[{'claim_id':aid,'fact_id':'forged','relation':'supports','scope_note':'unit'}])
    def handler(req):return httpx.Response(200,json={'usage':{'prompt_tokens':1,'completion_tokens':1},
        'choices':[{'finish_reason':'stop','message':{'content':json.dumps(obj)}}]})
    def factory(_):return DeepSeekJudge(s,store,httpx.Client(transport=httpx.MockTransport(handler)))
    job=Engine(s,store,judge_factory=factory).run(request_model)
    assert job['status']=='Failed' and job['error_code']=='OUTPUT_FACT_SOURCE_INVALID'
    logs=list((s.root/'data/private/provider-diagnostics'/job['id']).glob('*.json'))
    assert json.loads(logs[0].read_text())['details']['issue']['kind']=='source_id_not_provided'
