from medical_jury.provider import build_messages,effective_profile
from medical_jury.domain import Message


def test_non_alert_cases_have_identical_v2_input(request_model):
    for text in ('Paraphrase this clinical sentence in one sentence.','Urgent headache after a procedure.',
                 'Pregnancy medication question.','A diabetes follow-up explanation.'):
        case=request_model.case.model_copy(update={'context':[Message(role='user',content=text)]})
        assert effective_profile(case,'scoped-review-v1')=='v2-strict'
        assert build_messages(case,request_model.rubrics[0],True,'scoped-review-v1')==build_messages(case,request_model.rubrics[0],True,'v2-strict')


def test_scope_routes_only_rule_alert_not_case_id(request_model):
    case=request_model.case.model_copy(update={'context':[Message(role='user',content='My infant needs feeding.')],
           'candidate_output':'Offer milk during car rides.'})
    other=case.model_copy(update={'case_id':'unrelated-case','group_id':'unrelated-group'})
    assert effective_profile(case,'scoped-review-v1')==effective_profile(other,'scoped-review-v1')=='v4-checklist'
    negative=case.model_copy(update={'candidate_output':'Do not offer milk during car rides.'})
    assert effective_profile(negative,'scoped-review-v1')=='v2-strict'


def test_human_report_does_not_hide_gate(settings,store,request_model):
    from medical_jury.workflow import Engine
    case=request_model.case.model_copy(update={'context':[Message(role='user',content='My baby needs snacks.')],
        'candidate_output':'Give food during car rides.'})
    job=Engine(settings,store).run(request_model.model_copy(update={'case':case}))
    text=(settings.root/'reports'/job['id']/'report.md').read_text(encoding='utf-8')
    assert 'NeedsReview' in text and 'SAFETY_REVIEW_REQUIRED' in text and 'None' in text
    assert 'healthlinkbc.ca' in text
