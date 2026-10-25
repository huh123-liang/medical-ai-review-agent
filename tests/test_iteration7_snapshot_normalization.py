from medical_jury.domain import EvaluationRequest


def test_sparse_fixture_is_compared_to_normalized_sent_snapshot():
    sparse_case = {
        "case_id": "fixture", "group_id": "fixture", "task": "Check statement",
        "candidate_output": "A statement", "context": [{"role": "user", "content": "Check it"}],
        "source": "synthetic", "source_version": "v1", "split": "adhoc",
    }
    sparse_rubric = {
        "rubric_id": "r", "text": "Be accurate", "level": "case",
        "case_id": "fixture", "source": "synthetic", "version": "v1",
        "approved": True,
    }
    sent = EvaluationRequest(case=sparse_case, rubrics=[sparse_rubric])
    saved = sent.model_dump()
    rebuilt = EvaluationRequest(**saved)
    assert saved["case"] == rebuilt.case.model_dump()
    assert saved["rubrics"] == [r.model_dump() for r in rebuilt.rubrics]
    assert saved["case"]["candidate_output"] == sparse_case["candidate_output"]
    assert saved["rubrics"][0]["text"] == sparse_rubric["text"]
