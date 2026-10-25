from dataclasses import replace
import pytest

from medical_jury.domain import Judgment, Message, validate_judgment
from medical_jury.evidence import EvidenceResolutionError, locate, resolve_judgment


@pytest.mark.parametrize('source,quote',[
    ('- **SPF 30:** Blocks about 97% of UVB rays.','SPF 30: Blocks about 97% of UVB rays'),
    ('Une phrase n\u2019est pas nécessairement grave.',"Une phrase n'est pas nécessairement grave"),
    ('A sufficiently\u00a0long\n\nverbatim quotation','A sufficiently long verbatim quotation'),
    ('A cafe\u0301 with sufficient context','A café with sufficient context'),
    ('Use __original wording__ without changes.','Use original wording without changes.'),
])
def test_whitelisted_equivalences_anchor_to_exact_original(source,quote):
    span=locate(quote,[(0,source)],'answer_evidence',0)
    assert span['source_quote']==source[span['start_char']:span['end_char']]
    assert span['method']=='anchored-v1'
    assert len(span['source_sha256'])==64


@pytest.mark.parametrize('quote',[
    'Take 50 mg once daily for fever.',
    'Take 5 g once daily for fever.',
    'Do not take 5 mg once daily for fever.',
    'Take 5 mg twice daily for fever.',
    'take 5 mg once daily for fever.',
    'Take five mg once daily for fever.',
    'Take 5 mg ... for fever.',
])
def test_no_semantic_numeric_or_case_repair(quote):
    with pytest.raises(EvidenceResolutionError):
        locate(quote,[(0,'Take 5 mg once daily for fever.')],'answer_evidence',0)


@pytest.mark.parametrize('source,quote',[
    ('Left arm weakness without pain.','Right arm weakness without pain.'),
    ('Potassium <5.0 mmol/L.','Potassium >5.0 mmol/L.'),
    ('Value -2 and unit m².','Value 2 and unit m2.'),
    ('The number is 1000 per test.','The number is 1 000 per test.'),
])
def test_clinically_significant_symbols_unchanged(source,quote):
    with pytest.raises(EvidenceResolutionError):
        locate(quote,[(0,source)],'answer_evidence',0)


def test_repeated_exact_or_normalized_text_is_not_arbitrarily_located():
    with pytest.raises(EvidenceResolutionError) as error:
        locate('Repeated source quote',[(0,'Repeated source quote; Repeated source quote')],'answer_evidence',0)
    assert error.value.issue['kind']=='ambiguous_quote'
    with pytest.raises(EvidenceResolutionError):
        locate("A long quote isn't unique",[(0,'A long quote isn’t unique'),(1,'A long quote isn’t unique')],'context_evidence',0)


def test_browser_utf16_offsets_include_non_bmp_prefix():
    source='🩺 A sufficiently long original quote.'
    span=locate('A sufficiently long original quote',[(0,source)],'answer_evidence',0)
    assert span['start_char']==2 and span['start_utf16']==3


def test_cross_message_boundary_is_not_a_source_quote():
    with pytest.raises(EvidenceResolutionError):
        locate('first message\nsecond message',[(0,'first message'),(1,'second message')],'context_evidence',0)


def test_resolution_never_changes_decision_reason_or_case(request_model):
    case=request_model.case.model_copy(update={'candidate_output':'This is **an original quotation** for audit.'})
    before=case.model_dump()
    judgment=Judgment(judgment='not_met',reason='Original reason.',answer_evidence=['This is an original quotation for audit.'])
    resolved,trace=resolve_judgment(judgment,case)
    assert resolved.judgment==judgment.judgment and resolved.reason==judgment.reason
    assert resolved.review_required and trace['normalized_count']==1
    validate_judgment(resolved,case,True)
    assert case.model_dump()==before


def test_empty_evidence_remains_valid_for_omissions(request_model):
    judgment=Judgment(judgment='not_met',reason='Missing required information.')
    result,trace=resolve_judgment(judgment,request_model.case)
    assert result==judgment and trace['spans']==[]


@pytest.mark.parametrize('source,quote',[
    ('Use `**exact code text**` as data.','Use `exact code text` as data.'),
    ('15 mg once daily','5 mg once daily'),
    ('0.5 mg once daily','5 mg once daily'),
    ('-5 mg once daily','5 mg once daily'),
])
def test_no_code_rewrite_or_partial_number_extraction(source,quote):
    with pytest.raises(EvidenceResolutionError):
        locate(quote,[(0,source)],'answer_evidence',0)
