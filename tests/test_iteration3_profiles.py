import json
import pytest
import httpx
from dataclasses import replace
from medical_jury.provider import SYSTEM, RUBRIC_FIRST, PROFILE_VERSIONS, build_messages, parse_judgment
from medical_jury.domain import CaseInput, Message, Rubric


def inputs():
    case=CaseInput(case_id='profile-test',group_id='group-test',task='Evaluate',
        context=[Message(role='user',content='What is this?')],candidate_output='A **literal finding** is present.',
        source='test',source_version='1',split='dev')
    rubric=Rubric(rubric_id='r-test',text='Accurate.',level='case',case_id=case.case_id,
        group_id=case.group_id,split='dev',source='test',version='1',approved=True)
    return case,rubric


def test_profiles_share_identical_label_free_payload():
    case,rubric=inputs()
    old=build_messages(case,rubric,True)
    new=build_messages(case,rubric,True,'v3-rubric-first')
    assert old[0]['content']==SYSTEM
    assert new[0]['content']==SYSTEM+RUBRIC_FIRST
    assert old[1]==new[1]
    assert set(json.loads(new[1]['content']))=={'task','context','candidate_output','rubric','rubric_id'}
    assert 'v3-rubric-first' in PROFILE_VERSIONS
    with pytest.raises(ValueError): build_messages(case,rubric,True,'unapproved')


def test_default_uses_repaired_evidence_but_not_unproven_prompt(settings):
    assert settings.evidence_policy=='anchored-v2'
    assert settings.judge_profile=='scoped-review-v1'


def test_anchor_keeps_decision_and_emits_original_text_and_offsets():
    case,_=inputs(); trace={}
    raw=json.dumps({'judgment':'not_met','reason':'Fixed rationale.','answer_evidence':['literal finding'],
                    'context_evidence':[],'review_required':False})
    result=parse_judgment(raw,case,True,'anchored-v1',trace.update)
    assert result.judgment=='not_met' and result.reason=='Fixed rationale.'
    assert trace['spans'][0]['source_sha256']
    span=trace['spans'][0]
    assert case.candidate_output[span['start_char']:span['end_char']]==result.answer_evidence[0]


def test_normalized_evidence_survives_full_workflow(settings,store,request_model):
    from medical_jury.provider import DeepSeekJudge
    from medical_jury.workflow import Engine
    case=request_model.case.model_copy(update={'candidate_output':'- **SPF 30:** Blocks about 97% of UVB rays.'})
    req=request_model.model_copy(update={'case':case,'provider':'deepseek'})
    configured=replace(settings,api_key='mock-only',evidence_policy='anchored-v1')
    raw={'judgment':'not_met','reason':'Mock fixed rationale.','answer_evidence':['SPF 30: Blocks about 97% of UVB rays'],
         'context_evidence':[],'review_required':False}
    def handler(request):
        return httpx.Response(200,json={'model':'mock-wire-model','usage':{'prompt_tokens':10,'completion_tokens':10},
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(raw)}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        engine=Engine(configured,store,judge_factory=lambda _:DeepSeekJudge(configured,store,client))
        result=engine.run(req)
    assert result['status']=='Completed'
    report=result['report']; item=report['judgments'][0]
    assert item['judgment']==raw['judgment'] and item['reason']==raw['reason']
    assert item['review_required'] is True
    assert report['evidence_policy']=='anchored-v1'
    assert report['evidence_resolution'][0]['normalized_count']==1
    assert item['answer_evidence'][0] in case.candidate_output


def test_paired_metrics_and_bootstrap_helpers():
    import runpy
    from pathlib import Path
    functions=runpy.run_path(str(Path(__file__).parents[1]/'scripts/analyze_paired_iteration3.py'))
    metrics=functions['metrics']; bootstrap=functions['paired_bootstrap']
    pairs=[(True,True),(False,False),(False,True),(True,False)]
    assert metrics(pairs)['agreement']==.5 and metrics(pairs)['cohen_kappa']==0
    assert metrics(pairs)['confusion']=={'TP':1,'TN':1,'FP':1,'FN':1}
    groups=[{'A':[(True,True)],'B':[(True,True)]},{'A':[(False,False)],'B':[(False,False)]}]
    ci=bootstrap(groups,n=30)
    assert ci['agreement']==[0,0] and ci['macro_f1']==[0,0]
