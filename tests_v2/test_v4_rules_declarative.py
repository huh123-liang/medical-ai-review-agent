"""The declarative rule file + engine must behave exactly like the hand-coded agent-rules-v4-1 it replaced."""
from __future__ import annotations

import copy
import importlib.util
import json
import random
from pathlib import Path

import pytest

from medical_jury.testbed.build import build_cases
from medical_jury.v4 import rules as R

ROOT = Path(__file__).resolve().parents[1]
FROZEN = {'agent-rules-v4-1': '38f5bdd22c8389c7d5821cf3fd02c044337735274d9e9043ab3ed8e9e4de851f'}


def _legacy():
    spec = importlib.util.spec_from_file_location('legacy_rules_v4_1', ROOT / 'tests_v2' / 'fixtures' / 'legacy_rules_v4_1.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEGACY = _legacy()


def test_frozen_version_is_pinned_to_its_file():
    """Editing kb/v4/rules.json without bumping its version would silently change a frozen pipeline."""
    if R.RULES_VERSION in FROZEN:
        assert R.RULES_SHA256 == FROZEN[R.RULES_VERSION]


def test_exported_tables_match_the_legacy_module():
    assert R.LEXICON == LEGACY.LEXICON and R.SYNONYMS == LEGACY.SYNONYMS and R.PER_DAY == LEGACY.PER_DAY
    assert R.ACTIVE == LEGACY.ACTIVE and R.RULE_CATEGORY == LEGACY.RULE_CATEGORY and set(R._SALTS) == set(LEGACY._SALTS)
    for name in ('Metformin Hydrochloride', 'Lantus', 'HCTZ', 'amlodipine besylate', 'Insulin-Glargine U100'):
        assert R.normalise_drug(name) == LEGACY.normalise_drug(name)


def test_identical_on_every_testbed_case():
    for c in build_cases(150, 20261112)[0]:
        p = c['patient']
        facts = {'egfr': p['egfr'], 'potassium': p['potassium'], 'pregnant': p['pregnant'],
                 'acei_angioedema': bool(p['allergies'])}
        meds = [{**m, 'status': 'start'} for m in c['plan']['medications']]
        assert R.check(facts, meds) == LEGACY.check(facts, meds), c['case_id']


def test_identical_on_20000_boundary_heavy_random_cases():
    rnd = random.Random(7)
    drugs = list(R.LEXICON)
    for _ in range(20000):
        meds = []
        for d in rnd.sample(drugs, rnd.randint(0, 5)) + (['metformin'] if rnd.random() < .2 else []):
            mx = R.LEXICON[d][1] or 100
            meds.append({'drug': d, 'status': rnd.choice(['start', 'continue', 'stop', 'avoid']),
                         'dose_mg': rnd.choice([None, mx / 4, mx / 2, mx, mx * 2, 500, 1000]),
                         'frequency': rnd.choice(['once daily', 'twice daily', 'three times daily', 'weekly', None])})
        facts = {'egfr': rnd.choice([None, 10, 19, 20, 29, 30, 44, 45, 60, 90]),
                 'potassium': rnd.choice([None, 4.0, 5.4, 5.5, 6.1]),
                 'pregnant': rnd.choice([None, False, True]), 'acei_angioedema': rnd.choice([None, False, True])}
        assert R.check(facts, meds) == LEGACY.check(facts, meds), (facts, meds)


def test_every_rule_states_its_source_and_signoff():
    for r in R.RULES['rules']:
        assert r['clinical_signoff']['status'] in ('not_reviewed', 'approved', 'rejected')
        assert 'guideline' in r['source']


def test_malformed_rule_files_are_refused(tmp_path):
    base = json.loads(R.RULES_FILE.read_text(encoding='utf-8'))
    bad_kind = copy.deepcopy(base)
    bad_kind['rules'][0]['kind'] = 'magic'
    no_signoff = copy.deepcopy(base)
    del no_signoff['rules'][1]['clinical_signoff']
    bad_class = copy.deepcopy(base)
    bad_class['rules'][2]['applies_to']['classes'] = ['unicorn']
    for i, doc in enumerate((bad_kind, no_signoff, bad_class)):
        f = tmp_path / f'r{i}.json'
        f.write_text(json.dumps(doc), encoding='utf-8')
        with pytest.raises(R.RuleFileError):
            R.load_rules(f)


def test_a_new_rule_needs_only_the_file(tmp_path):
    doc = json.loads(R.RULES_FILE.read_text(encoding='utf-8'))
    doc['version'] = 'agent-rules-test'
    doc['rules'].append({'id': 'NSAID_EGFR', 'kind': 'threshold', 'applies_to': {'classes': ['nsaid']},
                         'fact': 'egfr', 'requires_fact': True,
                         'bands': [{'id': 'NSAID_EGFR_LT60', 'below': 60, 'category': 'contraindicated_drug',
                                    'severity': 'moderate'}],
                         'source': {'guideline': None, 'note': 'test only'},
                         'clinical_signoff': {'status': 'not_reviewed', 'by': None, 'date': None}})
    f = tmp_path / 'r.json'
    f.write_text(json.dumps(doc), encoding='utf-8')
    rules = R.load_rules(f)
    out = R.check({'egfr': 40}, [{'drug': 'ibuprofen', 'status': 'start', 'dose_mg': 400, 'frequency': 'once daily'}],
                  rules)
    assert [x['id'] for x in out] == ['NSAID_EGFR_LT60']


def test_signoff_is_metadata_but_a_threshold_is_behaviour():
    doc = json.loads(R.RULES_FILE.read_text(encoding='utf-8'))
    signed = copy.deepcopy(doc)
    signed['rules'][0]['clinical_signoff'] = {'status': 'approved', 'by': 'Dr X', 'date': '2026-12-01'}
    assert R.behaviour_digest(signed) == R.behaviour_digest(doc)
    moved = copy.deepcopy(doc)
    moved['rules'][1]['bands'][0]['below'] = 31
    assert R.behaviour_digest(moved) != R.behaviour_digest(doc)


def test_guideline_sources_are_filled_from_the_library():
    assert all(r['source']['guideline'] for r in R.RULES['rules'])
