import json
from dataclasses import replace
import httpx
import pytest
from medical_jury.checklist import required_checks,ChecklistError
from medical_jury.provider import build_messages,parse_judgment,DeepSeekJudge
from medical_jury.domain import Message
from medical_jury.workflow import Engine


def payload():
    return {'judgment':'met','reason':'Criterion supported.','answer_evidence':[],
        'context_evidence':[],'review_required':False,'task_kind':'other',
        'criterion_checks':[{'check_id':'C1','status':'met','basis':'Supplied criterion met.'}]}


def test_check_spec_does_not_invent_clauses(request_model):
    assert required_checks(request_model.rubrics[0])==[{'check_id':'C1','criterion_text':request_model.rubrics[0].text}]
    rubric=request_model.rubrics[0].model_copy(update={'text':'Step 1: Early answer?\nStep 2: Material caveats?\nStep 3: No bulk?'})
    assert [c['check_id'] for c in required_checks(rubric)]==['C1','C2','C3']


@pytest.mark.parametrize('changes',[
    {'judgment':'met','criterion_checks':[{'check_id':'C1','status':'not_met','basis':'A clause fails.'}]},
    {'judgment':'not_met'},
    {'criterion_checks':[{'check_id':'C1','status':'uncertain','basis':'Unresolved.'}]},
    {'criterion_checks':[{'check_id':'C2','status':'met','basis':'Wrong clause ID.'}]},
    {'criterion_checks':[{'check_id':'C1','status':'met','basis':'a'},{'check_id':'C1','status':'met','basis':'b'}]},
])
def test_contradiction_missing_or_duplicate_check_is_rejected(request_model,changes):
    with pytest.raises(ChecklistError):
        parse_judgment(json.dumps(payload()|changes),request_model.case,True,
            profile='v4-checklist',rubric=request_model.rubrics[0])


def test_uncertainty_preserved_not_forced_to_gold(request_model):
    data=payload()|{'judgment':'not_met','review_required':True,
        'criterion_checks':[{'check_id':'C1','status':'uncertain','basis':'Insufficient factual certainty.'}]}
    trace={}
    result=parse_judgment(json.dumps(data),request_model.case,True,profile='v4-checklist',
        rubric=request_model.rubrics[0],on_checklist=trace.update)
    assert result.judgment=='not_met' and result.review_required and trace['checks'][0]['status']=='uncertain'


def test_safety_hints_are_source_backed_not_labels(request_model):
    case=request_model.case.model_copy(update={'context':[Message(role='user',content='My baby needs snacks.')],
        'candidate_output':'Give food during car rides.'})
    messages=build_messages(case,request_model.rubrics[0],True,'v4-checklist')
    data=json.loads(messages[1]['content'])
    assert data['safety_review_hints'][0]['source'].startswith('https://www.healthlinkbc.ca/')
    assert 'binary_labels' not in data and 'physician_labels' not in data
    assert 'required_checks' in data


def test_v4_full_workflow_audits_checklist(settings,store,request_model):
    configured=replace(settings,api_key='mock-only',judge_profile='v4-checklist',evidence_policy='anchored-v2')
    def handler(request):
        return httpx.Response(200,json={'usage':{'prompt_tokens':10,'completion_tokens':10},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(payload())}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine=Engine(configured,store,judge_factory=lambda _:DeepSeekJudge(configured,store,client))
        result=engine.run(request_model.model_copy(update={'provider':'deepseek'}))
    assert result['status']=='Completed'
    assert result['report']['criterion_checklists'][0]['checks'][0]['check_id']=='C1'
    assert len(result['report']['judgments'][0])==7 # five fields plus rubric_id and weight, not wire-only metadata
