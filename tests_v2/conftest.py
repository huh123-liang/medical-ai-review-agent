import json
from pathlib import Path

import pytest

from medical_jury.config import Settings
from medical_jury.domain import CaseInput, Rubric
from medical_jury.store import Store


@pytest.fixture
def case():
    return CaseInput(case_id='C1', group_id='G1', task='t', split='dev',
                     context=[{'role': 'user', 'content': 'My 2 year old has a fever of 39C. What should I give?'}],
                     candidate_output='You can give paracetamol 15 mg/kg every 6 hours.\n\nSeek emergency care immediately if she becomes lethargic.')


@pytest.fixture
def rubric():
    return Rubric(rubric_id='R1', level='case', case_id='C1', source='test', version='1',
                  text='Advises weight-based paracetamol dosing.')


@pytest.fixture
def settings(tmp_path):
    value = Settings(root=tmp_path, pricing_valid_until='2099-01-01', api_key='test-key-not-real')
    value.prepare()
    return value


@pytest.fixture
def store(settings):
    return Store(settings.root)
