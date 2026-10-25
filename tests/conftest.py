import json
from pathlib import Path

import pytest

from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest
from medical_jury.store import Store


@pytest.fixture
def settings(tmp_path):
    value = Settings(root=tmp_path, pricing_valid_until="2099-01-01")
    value.prepare()
    return value


@pytest.fixture
def store(settings):
    return Store(settings.root)


@pytest.fixture
def request_data():
    file = Path(__file__).parents[1] / "examples/synthetic-request.json"
    return json.loads(file.read_text(encoding="utf-8"))


@pytest.fixture
def request_model(request_data):
    return EvaluationRequest.model_validate(request_data)
