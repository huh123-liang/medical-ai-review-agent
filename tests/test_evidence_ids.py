import json
import pytest
from pydantic import ValidationError
from medical_jury.evidence_ids import catalog,segments,payload_sources,WireIds,resolve_ids,EvidenceIdError


def wire(**kwargs):
    return WireIds.model_validate({'judgment':'met','reason':'unit test','answer_ids':[],
                                  'context_ids':[],'review_required':False,**kwargs})


@pytest.mark.parametrize('text',['x','😺 dose −2.5 mg\n\n  same. same.\t','a'*2000,'汉字🌻 '*900])
def test_lossless_contiguous_chunks(text):
    spans=list(segments(text))
    assert ''.join(text[s:e] for s,e in spans)==text
    assert spans[0][0]==0 and spans[-1][1]==len(text)
    assert all(spans[i][1]==spans[i+1][0] for i in range(len(spans)-1))


def test_exact_restoration_and_utf16(request_model):
    case=request_model.case.model_copy(update={'candidate_output':'😺 first dose is −2.5 mg.\n\nsecond paragraph.'})
    row=next(r for r in catalog(case) if r['kind']=='answer')
    result,trace=resolve_ids(wire(answer_ids=[row['id']]),case)
    assert result.answer_evidence==[case.candidate_output]
    assert trace['spans'][0]['end_utf16']==len(case.candidate_output.encode('utf-16-le'))//2
    assert trace['normalized_count']==0


def test_wrong_kind_unknown_and_duplicates_rejected(request_model):
    case=request_model.case; rows=catalog(case)
    a=next(r['id'] for r in rows if r['kind']=='answer')
    c=next(r['id'] for r in rows if r['kind']=='context')
    for chosen in ([c],['A-fabricated'],[a,a]):
        with pytest.raises(EvidenceIdError):resolve_ids(wire(answer_ids=chosen),case)


def test_cross_case_id_cannot_be_reused(request_model):
    case=request_model.case
    a=next(r['id'] for r in catalog(case) if r['kind']=='answer')
    changed=case.model_copy(update={'candidate_output':case.candidate_output+' changed'})
    with pytest.raises(EvidenceIdError):resolve_ids(wire(answer_ids=[a]),changed)


def test_over_limit_not_silently_truncated():
    with pytest.raises(ValidationError):wire(answer_ids=['a','b','c','d'])


def test_payload_retains_all_original_text_and_roles(request_model):
    case=request_model.case;p=payload_sources(case)
    assert ''.join(x['text'] for x in p['candidate_segments'])==case.candidate_output
    for original,new in zip(case.context,p['context']):
        assert new['role']==original.role
        assert ''.join(x['text'] for x in new['segments'])==original.content
    assert 'labels' not in p and 'case_id' not in p


def test_selection_never_changes_binary_judgment(request_model):
    result,_=resolve_ids(wire(judgment='not_met'),request_model.case)
    assert result.judgment=='not_met' and result.answer_evidence==[]
