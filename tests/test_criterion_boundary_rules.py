import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from criterion_boundary_rules import audit_conjunction


def obligation(state,**extra):
    return {'id':'one','status':state,'applicability':'in_scope',**extra}


def test_missing_source_never_proves_falsehood():
    out=audit_conjunction([obligation('not_met',basis='no_source')])
    assert out['audit_suggestion']=='unresolved' and not out['clinical_release']


@pytest.mark.parametrize('basis',['could_add_more','length_only','disclaimer_present'])
def test_inadequate_failure_bases_not_accepted(basis):
    out=audit_conjunction([obligation('not_met',basis=basis,counterexample_ids=['x'])])
    assert out['audit_suggestion']=='unresolved'


def test_one_in_scope_failure_not_cancelled_by_other_passes():
    out=audit_conjunction([obligation('not_met',basis='counterexample',counterexample_ids=['x']),
                           {'id':'two','status':'met','applicability':'in_scope'}])
    assert out['audit_suggestion']=='not_met'
    assert not out['original_model_verdict_changed']


def test_unknown_clause_prevents_automatic_pass():
    out=audit_conjunction([obligation('met'),{'id':'two','status':'unresolved'}])
    assert out['audit_suggestion']=='unresolved'


def test_failure_outside_rubric_is_not_automatic_failure():
    out=audit_conjunction([obligation('not_met',applicability='out_of_scope',basis='counterexample',counterexample_ids=['x'])])
    assert out['audit_suggestion']=='unresolved'


def test_strict_full_answer_does_not_pass_from_one_supported_claim():
    out=audit_conjunction([obligation('met')],strict_factuality=True,full_claim_coverage=False)
    assert out['audit_suggestion']=='unresolved'


def test_appropriateness_not_forced_to_prove_all_medical_facts():
    out=audit_conjunction([obligation('met')],strict_factuality=False,full_claim_coverage=False)
    assert out['audit_suggestion']=='met' and not out['clinical_release']


@pytest.mark.parametrize('rows',[[],[obligation('invalid')],[obligation('met'),obligation('met')]])
def test_bad_audit_protocol_does_not_silently_pass(rows):
    with pytest.raises(ValueError):audit_conjunction(rows)
