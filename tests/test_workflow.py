import json

import pytest
from pydantic import ValidationError

from medical_jury.api import create_app
from medical_jury.budget import Budget
from medical_jury.domain import EvaluationRequest, Judgment, validate_judgment
from medical_jury.store import Store
from medical_jury.workflow import Engine


def test_real_langgraph_oracle_mock_report_has_no_clinical_claim(settings, store, request_model):
    result = Engine(settings, store).run(request_model)
    assert result["status"] == "Completed"
    report = result["report"]
    assert report["clinical_validation"] == "OFFLINE_MOCK_NOT_CLINICAL"
    assert report["criterion_count"] == 1
    assert len(report["snapshot_hash"]) == 64
    assert (settings.root / "reports" / result["id"] / "report.md").exists()
    assert Budget(settings, store).total()["paid_attempts"] == 0
    assert [r["event"] for r in store.events(result["id"])][-1] == "completed"


def test_physician_labels_cannot_enter_request(request_data):
    request_data["binary_labels"] = [True]
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(request_data)


def test_duplicate_rules_rejected_before_paid_calls(request_data):
    request_data["rubrics"].append(request_data["rubrics"][0].copy())
    with pytest.raises(ValidationError, match="Duplicate"):
        EvaluationRequest.model_validate(request_data)


def test_cross_case_rubric_rejected(request_data):
    request_data["rubrics"][0]["case_id"] = "ANOTHER-PATIENT"
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(request_data)


def test_unapproved_rubric_rejected(request_data):
    request_data["rubrics"][0]["approved"] = False
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(request_data)


def test_retrieval_cannot_receive_oracle_rules(request_data):
    request_data["mode"] = "retrieval"
    with pytest.raises(ValidationError):
        EvaluationRequest.model_validate(request_data)


def test_unsupported_evidence_is_rejected(request_model):
    with pytest.raises(ValueError, match="substring"):
        validate_judgment(Judgment(judgment="met", reason="invalid", answer_evidence=["fabricated lab value"]), request_model.case, True)


def test_binary_benchmark_does_not_silently_accept_abstention(request_model):
    with pytest.raises(ValueError, match="binary"):
        validate_judgment(Judgment(judgment="abstain", reason="missing context"), request_model.case, True)


def test_cancel_before_first_model_call(settings, store, request_model):
    job_id = store.create_job(request_model)
    store.cancel(job_id)
    result = Engine(settings, store).run(request_model, job_id)
    assert result["status"] == "Cancelled"
    assert Budget(settings, store).total()["paid_attempts"] == 0


def test_reading_store_does_not_interrupt_active_jobs(settings, store, request_model):
    job_id = store.create_job(request_model)
    assert Store(settings.root).get_job(job_id)["status"] == "Pending"
    store.recover_interrupted()
    assert store.get_job(job_id)["status"] == "Interrupted"


def test_missing_key_fails_closed_without_spend(settings, store, request_data):
    request_data["provider"] = "deepseek"
    result = Engine(settings, store).run(EvaluationRequest.model_validate(request_data))
    assert result["status"] == "Failed"
    assert Budget(settings, store).total()["accounted_cny"] == 0


def test_failed_evidence_is_not_saved_as_completed(settings, store, request_model):
    class BadJudge:
        def judge(self, *args):
            return Judgment(judgment="met", reason="fake", context_evidence=["not in record"])
    result = Engine(settings, store, judge_factory=lambda _: BadJudge()).run(request_model)
    assert result["status"] == "Failed"
    assert result["report"] is None


def test_local_api_requires_auth_and_runs_mock_only_when_requested(settings, store, request_data):
    from fastapi.testclient import TestClient
    import time
    app = create_app(settings, store)
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/budget").status_code == 401
        headers = {"Authorization": "Bearer " + store.admin_token()}
        response = client.post("/v1/evaluations", json=request_data, headers=headers)
        assert response.status_code == 202, response.text
        job_id = response.json()["job_id"]
        for _ in range(100):
            result = client.get("/v1/evaluations/" + job_id, headers=headers).json()
            if result["status"] in ("Completed", "Failed"):
                break
            time.sleep(0.01)
        assert result["status"] == "Completed"
        assert result["report"]["provider"] == "mock"
        assert client.get("/v1/budget", headers=headers).json()["paid_attempts"] == 0
        assert client.post("/v1/evaluations/" + job_id + "/retry", headers=headers).status_code == 409
        assert client.get("/v1/budget", headers=headers | {"Origin": "https://untrusted.example"}).status_code == 403
        assert client.post("/v1/evaluations", content="a" * 150000, headers=headers).status_code == 413
        invalid = dict(request_data, accidental_key="SENSITIVE-UNIT-TEST-VALUE")
        response = client.post("/v1/evaluations", json=invalid, headers=headers)
        assert response.status_code == 422
        assert "SENSITIVE-UNIT-TEST-VALUE" not in response.text
    app.state.jury_pool.shutdown(wait=True)


def test_api_idempotency_does_not_duplicate_jobs(settings, store, request_data):
    from fastapi.testclient import TestClient
    app = create_app(settings, store)
    headers = {"Authorization": "Bearer " + store.admin_token(), "Idempotency-Key": "unique-test-submission"}
    with TestClient(app) as client:
        first = client.post("/v1/evaluations", json=request_data, headers=headers)
        second = client.post("/v1/evaluations", json=request_data, headers=headers)
        assert first.json()["job_id"] == second.json()["job_id"]
        assert second.json()["created"] is False
        request_data["top_k"] = 2
        assert client.post("/v1/evaluations", json=request_data, headers=headers).status_code == 409
    app.state.jury_pool.shutdown(wait=True)
