from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import httpx
import pytest

from medical_jury.budget import Budget, BudgetExceeded
from medical_jury.domain import EvaluationRequest
from medical_jury.provider import DeepSeekJudge, ProviderError
from medical_jury.workflow import Engine


def paid_settings(settings, **kwargs):
    return replace(settings, api_key="unit-test-placeholder-not-a-real-key", **kwargs)


def test_concurrent_reservations_cannot_overspend(settings, store):
    budget = Budget(replace(settings, budget_cny=1), store)
    def attempt(i):
        try:
            budget.reserve("job-" + str(i), 0.4, 1)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert sum(pool.map(attempt, range(8))) == 2
    assert budget.total()["accounted_cny"] == pytest.approx(0.8)


def test_unknown_failed_call_keeps_reservation(settings, store):
    budget = Budget(settings, store)
    reservation = budget.reserve("x", 0.5, 1)
    budget.settle(reservation, None, False)
    assert budget.total()["accounted_cny"] == 0.5


def test_successful_usage_releases_only_unused_reservation(settings, store):
    budget = Budget(settings, store)
    reservation = budget.reserve("x", 0.5, 1)
    budget.settle(reservation, {"prompt_tokens": 100, "completion_tokens": 100}, True)
    assert budget.total()["accounted_cny"] == pytest.approx(0.0015)


def test_exceeded_reservation_halts_future_calls(settings, store):
    budget = Budget(settings, store)
    reservation = budget.reserve("x", 0.01, 1)
    with pytest.raises(BudgetExceeded):
        budget.settle(reservation, {"prompt_tokens": 100000, "completion_tokens": 100000}, True)
    with pytest.raises(BudgetExceeded):
        budget.reserve("y", 0.01, 1)


def test_pro_cannot_use_flash_rates(settings):
    with pytest.raises(ValueError, match="price ceiling"):
        paid_settings(settings, model="deepseek-v4-pro").validate_paid()


def test_expired_prices_block_paid_calls(settings):
    with pytest.raises(ValueError, match="expired"):
        paid_settings(settings, pricing_valid_until="2000-01-01").validate_paid()


def test_authentication_error_is_not_retried_or_leaked(settings, store, request_model):
    count = []
    def handler(request):
        count.append(1)
        return httpx.Response(401, json={"error": "sensitive provider text"})
    client = httpx.Client(transport=httpx.MockTransport(handler))
    judge = DeepSeekJudge(paid_settings(settings, max_attempts=2), store, client)
    job_id = store.create_job(request_model)
    with pytest.raises(ProviderError, match="HTTP_401") as caught:
        judge.judge(request_model.case, request_model.rubrics[0], job_id, 1, True)
    assert "sensitive" not in str(caught.value)
    assert len(count) == 1
    assert Budget(settings, store).total()["accounted_cny"] > 0


def test_timeout_retry_reserves_each_paid_attempt(settings, store, request_model):
    calls = []
    def handler(request):
        calls.append(1)
        raise httpx.ReadTimeout("unit timeout", request=request)
    judge = DeepSeekJudge(paid_settings(settings, max_attempts=2), store,
                         httpx.Client(transport=httpx.MockTransport(handler)))
    job_id = store.create_job(request_model)
    with pytest.raises(ProviderError):
        judge.judge(request_model.case, request_model.rubrics[0], job_id, 1, True)
    assert len(calls) == 2
    assert Budget(settings, store).total()["paid_attempts"] == 2


def test_provider_input_has_no_labels_or_tool_privileges(settings, store, request_model):
    seen = []
    def handler(request):
        body = json.loads(request.content)
        data = json.loads(body["messages"][1]["content"])
        assert set(data) == {"task", "context", "candidate_output", "rubric", "rubric_id"}
        assert "tools" not in body
        assert body["thinking"] == {"type": "disabled"}
        seen.append(1)
        return httpx.Response(200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 80},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"judgment": "met", "reason": "unit test", "answer_evidence":[], "context_evidence":[], "review_required":False})}}]})
    judge = DeepSeekJudge(paid_settings(settings), store, httpx.Client(transport=httpx.MockTransport(handler)))
    job_id = store.create_job(request_model)
    assert judge.judge(request_model.case, request_model.rubrics[0], job_id, 1, True).judgment == "met"
    assert seen == [1]


def test_token_truncation_is_not_a_completed_judgment(settings, store, request_model):
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={
        "usage": {"prompt_tokens": 100, "completion_tokens": 100},
        "choices": [{"finish_reason": "length", "message": {"content": '{"judgment":"met"}'}}]})))
    judge = DeepSeekJudge(paid_settings(settings), store, client)
    job_id = store.create_job(request_model)
    with pytest.raises(ProviderError, match="INCOMPLETE"):
        judge.judge(request_model.case, request_model.rubrics[0], job_id, 1, True)


def test_zero_budget_sends_no_request(settings, store, request_model):
    calls = []
    client = httpx.Client(transport=httpx.MockTransport(lambda _: calls.append(1)))
    judge = DeepSeekJudge(paid_settings(settings, budget_cny=0), store, client)
    job_id = store.create_job(request_model)
    with pytest.raises(BudgetExceeded):
        judge.judge(request_model.case, request_model.rubrics[0], job_id, 1, True)
    assert calls == []
