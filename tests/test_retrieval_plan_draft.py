import json
from pathlib import Path
import pytest
from pydantic import ValidationError
from medical_jury.domain import Rubric


def test_proposed_contract_is_not_approved_current_runtime_rubric():
    path=Path(__file__).parents[1]/'examples/proposed-rubric-v2.json'
    draft=json.loads(path.read_text(encoding='utf-8'))
    assert draft['approval_status']=='draft' and draft['eligible_for_current_index'] is False
    with pytest.raises(ValidationError): Rubric.model_validate(draft)
