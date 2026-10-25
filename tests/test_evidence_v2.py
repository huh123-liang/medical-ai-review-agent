import json
import pytest
from medical_jury.evidence import locate, EvidenceResolutionError
from medical_jury.provider import parse_judgment
from medical_jury.domain import CaseInput,Message


def test_duplicate_exact_positions_retained_not_chosen():
    text='Chief Complaint: one. Chief Complaint: two.'
    span=locate('Chief Complaint',[(0,text)],'answer_evidence',0,True)
    assert span['resolution']=='ambiguous' and span['candidate_count']==2
    assert 'start_char' not in span
    assert [s['start_char'] for s in span['candidates']]==[0,22]
    assert all(text[s['start_char']:s['end_char']]=='Chief Complaint' for s in span['candidates'])


def test_v1_still_rejects_duplicates():
    with pytest.raises(EvidenceResolutionError): locate('duplicate',[(0,'duplicate duplicate')],'answer_evidence',0)


def test_duplicates_across_messages_keep_source_identity():
    span=locate('same sentence.',[(0,'same sentence.'),(2,'same sentence.')],'context_evidence',0,True)
    assert [s['source_index'] for s in span['candidates']]==[0,2]


def test_ambiguous_normalization_with_different_source_text_is_rejected():
    with pytest.raises(EvidenceResolutionError,match='uniquely'):
        locate('This is a sufficiently long quoted sentence.',[(0,'This is a sufficiently **long** quoted sentence.'),
              (1,'This is a sufficiently long  quoted sentence.')],'context_evidence',0,True)


def test_identical_normalized_copies_have_all_locations():
    text='A **long quotation** appears. A **long quotation** appears.'
    result=locate('A long quotation appears.',[(0,text)],'answer_evidence',0,True)
    assert result['method']=='multiple-normalized' and len(result['candidates'])==2


def test_too_many_duplicate_locations_fail_closed():
    with pytest.raises(EvidenceResolutionError): locate('duplicate',[(0,'duplicate '*40)],'answer_evidence',0,True)


def test_fabricated_or_changed_numeric_quote_still_rejected():
    for quote in ('Give 5 mg daily.','Invented missing source.'):
        with pytest.raises(EvidenceResolutionError): locate(quote,[(0,'Give 15 mg daily.')],'answer_evidence',0,True)


def test_v2_preserves_decision_with_ambiguity_review():
    case=CaseInput(case_id='a',group_id='g',task='evaluate',context=[Message(role='user',content='input')],
                   candidate_output='Chief Complaint and Chief Complaint',source='fixture',source_version='1',split='dev')
    raw=json.dumps({'judgment':'met','reason':'unchanged','answer_evidence':['Chief Complaint'],
                    'context_evidence':[],'review_required':False})
    trace={}; result=parse_judgment(raw,case,True,'anchored-v2',trace.update)
    assert result.judgment=='met' and result.reason=='unchanged' and result.review_required
    assert trace['ambiguous_count']==1 and trace['normalized_count']==0
