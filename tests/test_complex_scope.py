from medical_jury.complex_scope import scope_kind,note_for
from medical_jury.provider import build_messages,parse_judgment

def test_strict_factuality_is_not_relaxed(request_model):
    rubric=request_model.rubrics[0].model_copy(update={'text':'An extraordinarily high bar. Check every claim.'})
    assert scope_kind(rubric.text)=='strict_factuality'
    assert "even a small incorrect claim can fail" in note_for(rubric)
    assert "no major errors" in note_for(rubric)

def test_step_criterion_keeps_and_rule(request_model):
    rubric=request_model.rubrics[0].model_copy(update={'text':'Step 1: Is the core question answered clearly and early?\nStep 3: Avoid bulk.'})
    assert scope_kind(rubric.text)=='detailed_appropriateness'
    assert 'original AND rule' in note_for(rubric)

def test_unrecognized_rubric_keeps_v2_messages(request_model):
    assert note_for(request_model.rubrics[0])==''
    assert build_messages(request_model.case,request_model.rubrics[0],True,'complex-boundary-v1')==build_messages(request_model.case,request_model.rubrics[0],True,'v2-strict')

def test_candidate_has_same_data_payload_no_labels_or_case_examples(request_model):
    rubric=request_model.rubrics[0].model_copy(update={'text':'The response should be somewhat simple, not overly vague.'})
    old=build_messages(request_model.case,rubric,True,'v2-strict')
    new=build_messages(request_model.case,rubric,True,'complex-boundary-v1')
    assert old[1]==new[1] and new[0]['content'].startswith(old[0]['content'])
    assert 'paraphrase of its directive is an answer' in new[0]['content']

def test_revision2_only_changes_explicit_detailed_appropriateness(request_model):
    strict=request_model.rubrics[0].model_copy(update={'text':'An extraordinarily high bar. Check every claim.'})
    assert note_for(strict,1)==note_for(strict,2)
    detail=strict.model_copy(update={'text':'Step 1: Is the core question answered clearly and early?\nStep 3: Avoid bulk.'})
    assert note_for(detail,2).startswith(note_for(detail,1))
    assert 'procedure-level' in note_for(detail,2) and 'clinician' in note_for(detail,2)
