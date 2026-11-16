"""v4 error-detection pipeline: offline tests only (no network, no paid call)."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from medical_jury.evaluation import v4_scoring as SC  # noqa: E402
from medical_jury.judge_v2 import BackendError, ScriptedBackend  # noqa: E402
from medical_jury.testbed.build import build_cases  # noqa: E402
from medical_jury.testbed.errors import AGENT_RULES_WITHHELD, CATALOGUE, HELD_OUT_TYPES  # noqa: E402
from medical_jury.v4 import check as CK, criteria as CR, extract as EX, rules as RU  # noqa: E402
from medical_jury.v4.mocks import oracle_extraction, oracle_responder, plain_document  # noqa: E402
from medical_jury.v4.pipeline import ReviewerV4, rules_only, triage  # noqa: E402

SEED = 20261112


@pytest.fixture(scope='module')
def cases():
    return build_cases(150, SEED)[0]


@pytest.fixture(scope='module')
def library():
    return CR.load_library(ROOT)


def _truth_facts(p):
    return {'egfr': p['egfr'], 'potassium': p['potassium'], 'systolic': p['systolic'], 'diastolic': p['diastolic'],
            'pregnant': p['pregnant'], 'acei_angioedema': bool(p['allergies']), 'conditions': p['conditions']}


def _active(plan):
    return [{**m, 'status': 'start'} for m in plan['medications']]


# ---------------------------------------------------------------- agent rule layer

def test_agent_rules_silent_on_every_reference_plan(cases):
    for c in cases:
        if not c['gold']:
            assert RU.check(_truth_facts(c['patient']), _active(c['plan'])) == [], c['case_id']


def test_agent_rules_find_rule_visible_errors_except_withheld(cases):
    for c in cases:
        if not isinstance(c['gold'], dict) or not c['gold']['rule_covered']:
            continue
        hits = [f for f in RU.check(_truth_facts(c['patient']), _active(c['plan'])) if c['gold']['entity'] in f['drugs']]
        if c['gold']['error_id'] in AGENT_RULES_WITHHELD:
            assert not hits, c['case_id']
        else:
            assert hits and SC.found(c['gold'], {'findings': hits}), c['case_id']


def test_agent_rules_independent_of_testbed_rules():
    src = (ROOT / 'src/medical_jury/v4/rules.py').read_text(encoding='utf-8')
    assert 'testbed' not in src.split('"""', 2)[2]              # no import of the answer key
    assert 'TZD_HEART_FAILURE' not in src and 'NSAID_CKD' not in src


def test_missing_fact_gives_cannot_determine():
    out = RU.check({'egfr': None, 'potassium': 4.5}, [{'drug': 'metformin', 'dose_mg': 500, 'frequency': 'twice daily',
                                                      'status': 'start'}])
    assert [f['category'] for f in out] == ['cannot_determine']


def test_avoid_and_stop_are_not_prescriptions():
    meds = [{'drug': 'metformin', 'dose_mg': 1000, 'frequency': 'twice daily', 'status': 'avoid'},
            {'drug': 'spironolactone', 'dose_mg': 25, 'frequency': 'once daily', 'status': 'stop'}]
    assert RU.check({'egfr': 20, 'potassium': 6.1}, meds) == []


# ---------------------------------------------------------------- extraction

DESC = 'Age 60; blood pressure 150/90 mmHg; eGFR 40 mL/min/1.73m2, serum potassium 4.6 mmol/L.'


def _ex(meds, **facts):
    patient = {'egfr': {'value': 40, 'quote': 'eGFR 40'}, 'potassium': {'value': 4.6, 'quote': 'potassium 4.6'},
               'systolic': {'value': 150, 'quote': '150/90 mmHg'}, 'diastolic': {'value': 90, 'quote': '150/90 mmHg'},
               'pregnant': {'value': False, 'quote': None}, 'conditions': {'value': ['hypertension'], 'quote': None},
               **facts}
    return EX.parse_extraction(json.dumps({'patient': patient, 'medications': meds}))


def test_extraction_checks_pass_on_faithful_extraction():
    doc = 'Take amlodipine 5 mg once daily. Avoid ibuprofen.'
    ex = _ex([{'drug': 'Amlodipine', 'dose_mg': 5, 'frequency': 'once daily', 'status': 'start', 'quote': 'amlodipine 5 mg'},
              {'drug': 'ibuprofen', 'dose_mg': None, 'frequency': 'other', 'status': 'avoid', 'quote': 'Avoid ibuprofen'}])
    assert EX.check_extraction(ex, DESC, doc) == []
    assert EX.unknown_active_drugs(ex) == []


def test_extraction_checks_catch_omission_bad_quote_dose_and_unknown():
    doc = 'Take amlodipine 5 mg once daily and metformin 500 mg twice daily.'
    ex = _ex([{'drug': 'amlodipine', 'dose_mg': 50, 'frequency': 'once daily', 'status': 'start', 'quote': 'amlodipine 5 mg'},
              {'drug': 'zorbitol', 'dose_mg': 1, 'frequency': 'once daily', 'status': 'start', 'quote': 'not in text'}],
             egfr={'value': 41, 'quote': 'eGFR 40'})
    checks = {(p['check'], p['item']) for p in EX.check_extraction(ex, DESC, doc)}
    assert ('MENTIONED_DRUG_NOT_EXTRACTED', 'metformin') in checks
    assert ('DOSE_NOT_IN_TEXT', 'amlodipine') in checks
    assert ('MED_QUOTE_NOT_FOUND', 'zorbitol') in checks
    assert ('FACT_VALUE_NOT_IN_QUOTE', 'egfr') in checks
    assert EX.unknown_active_drugs(ex) == ['zorbitol']


def test_extraction_parse_rejects_bad_status():
    with pytest.raises(EX.ExtractFailure):
        EX.parse_extraction(json.dumps({'patient': {}, 'medications': [{'drug': 'x', 'status': 'maybe'}]}))


def test_insulin_hormone_phrases_are_not_drug_mentions():
    assert 'insulin' not in EX.mentioned_drugs('Pioglitazone is included to improve insulin sensitivity.')
    assert 'insulin' in EX.mentioned_drugs('Start insulin at bedtime.')


def test_gram_doses_and_synonyms():
    assert EX.doses_near('metformin 1 g twice daily', 'metformin') == {1000.0}
    assert RU.normalise_drug('Metformin Hydrochloride') == 'metformin'
    assert RU.normalise_drug('Lantus') == 'insulin glargine'


# ---------------------------------------------------------------- criteria (three levels)

def test_library_does_not_spell_out_held_out_types(library):
    text = json.dumps([library['general'], library['disease']['templates']]).lower()
    for word in ('sick-day', 'sick day', 'double', 'missed', 'sglt2', 'below 5', 'pioglitazone', 'nsaid', 'ibuprofen'):
        assert word not in text.replace('no rule for pioglitazone', '').replace('nsaids in ckd', ''), word
    rules = {r['id'] for r in library['disease']['rules']}
    assert not rules & {'TZD_HEART_FAILURE', 'NSAID_CKD'}
    for prompt in (CK.SYSTEM_CHECK, EX.SYSTEM_EXTRACT):
        for word in ('sick', 'double', 'pioglitazone', 'nsaid', 'sglt2'):
            assert word not in prompt.lower(), word


def test_criteria_follow_patient_facts(library):
    facts = {'systolic': 190, 'diastolic': 100, 'potassium': 6.2, 'conditions': ['type_2_diabetes', 'ascvd']}
    meds = [{'drug': 'insulin glargine', 'dose_mg': None, 'frequency': 'once daily', 'status': 'start'},
            {'drug': 'lisinopril', 'dose_mg': 10, 'frequency': 'once daily', 'status': 'start'}]
    items, coded = CR.applicable(library, facts, meds)
    ids = {i['id'] for i in items}
    assert {'T_URGENT_BP', 'T_URGENT_K', 'T_HBA1C', 'T_HYPO_EDU', 'T_FOLLOWUP'} <= ids
    assert 'T_K_CREAT' not in ids                       # potassium >= 6: urgent action, not a routine recheck
    assert any('190/100' in i['text'] for i in items)
    assert [f['id'] for f in coded] == ['T_STATIN_ASCVD']
    _, coded = CR.applicable(library, {'systolic': None, 'diastolic': None, 'potassium': None, 'conditions': []}, [])
    assert {f['category'] for f in coded} == {'cannot_determine'}


# ---------------------------------------------------------------- check parsing and triage

def test_parse_check_requires_every_criterion_and_closes_categories():
    items = [{'id': 'G01'}, {'id': 'T_FOLLOWUP'}]
    with pytest.raises(CK.CheckFailure):
        CK.parse_check(json.dumps({'criteria': [{'id': 'G01', 'status': 'met'}]}), items)
    out = CK.parse_check(json.dumps({'criteria': [{'id': 'G01', 'status': 'met'}, {'id': 'T_FOLLOWUP', 'status': 'not_met'},
                                                  {'id': 'INVENTED', 'status': 'not_met'}],
                                     'other_findings': [{'category': 'weird', 'severity': 'huge', 'entity': 'x'}]}), items)
    assert set(out['criteria']) == {'G01', 'T_FOLLOWUP'}
    assert out['other_findings'][0]['category'] == 'other' and out['other_findings'][0]['severity'] == 'moderate'


def test_code_checks_list_only_document_medicines_and_withheld_rules_stay_unchecked():
    meds = [{'drug': 'empagliflozin', 'status': 'start'}, {'drug': 'ibuprofen', 'status': 'start'},
            {'drug': 'pioglitazone', 'status': 'start'}, {'drug': 'metformin', 'status': 'avoid'}]
    cc = CK.code_checks_of([], meds)
    by = {c['medicine']: c['against'] for c in cc['checked']}
    assert set(by) == {'empagliflozin', 'ibuprofen', 'pioglitazone'}          # avoided metformin not listed
    assert 'eGFR' in by['empagliflozin'] and 'eGFR' not in by['ibuprofen']      # NSAID-kidney stays unchecked
    assert not any('heart' in a for v in by.values() for a in v)               # TZD-heart failure stays unchecked
    assert cc['flagged'] == []


def test_triage():
    sev = [{'severity': 'severe'}]
    assert triage(sev, [], [], [], [])['decision'] == 'escalate'
    assert triage([{'severity': 'minor'}], [], [], [], [])['decision'] == 'pass_with_findings'
    assert triage([], [], [], [], [])['decision'] == 'pass'
    assert triage([], [], ['zorbitol'], [], [])['reasons'] == ['unknown_drug']
    assert triage([], [], [], [{'check': 'X'}], [])['decision'] == 'escalate'


def test_failed_call_escalates_and_stops(library):
    def boom(messages):
        raise BackendError('HTTP_402')

    class Failing(ScriptedBackend):
        def complete(self, messages, max_tokens, job_id):
            boom(messages)
    rev = ReviewerV4(Failing(lambda m: ''), library)
    rec = rev.review(DESC, 'Take amlodipine 5 mg once daily.', 'J')
    assert rec['triage']['decision'] == 'escalate' and rec['calls'][0]['failure_code'] == 'HTTP_402'
    assert len(rec['calls']) == 1


def test_end_to_end_with_oracle_extraction(cases, library):
    docs = {(c['patient_summary'], plain_document(c)): c for c in cases}
    rev = ReviewerV4(ScriptedBackend(oracle_responder(docs)), library)
    test = [c for c in cases if c['split'] == 'test']
    recs = {c['case_id']: rev.review(c['patient_summary'], plain_document(c), c['case_id']) for c in test}
    s = SC.score_arm(test, recs, HELD_OUT_TYPES)
    assert s['primary']['clean_severe_false_positive']['k'] == 0
    assert s['primary']['severe_detection_seen_types']['rate'] == 1.0      # oracle extraction + keyword mock
    assert s['secondary']['extraction_exact']['medications']['rate'] == 1.0
    ro = SC.score_arm(test, {k: rules_only(v) for k, v in recs.items()}, HELD_OUT_TYPES)
    assert ro['secondary']['by_error_type']['O02_no_urgent_bp_action']['k'] == 0
    assert all(r['triage']['decision'] == 'pass' or r['findings'] or r['triage']['reasons']
               for r in recs.values())


# ---------------------------------------------------------------- scoring

def test_matching_is_category_and_entity_gated():
    g = {'error_id': 'C04_tenfold_dose', 'entity': 'amlodipine'}
    assert SC.matches(g, {'category': 'dose_error', 'entity': 'amlodipine'})
    assert not SC.matches(g, {'category': 'dose_error', 'entity': 'atorvastatin'})
    assert not SC.matches(g, {'category': 'contraindicated_drug', 'entity': 'amlodipine'})
    o2 = {'error_id': 'O02_no_urgent_bp_action', 'entity': 'urgent_bp'}
    assert not SC.matches(o2, {'category': 'missing_urgent_action', 'entity': 'urgent assessment for high potassium'})
    assert set(SC.MATCH) == set(CATALOGUE)


def test_wilson():
    assert SC.wilson(0, 0) is None
    lo, hi = SC.wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97


# ---------------------------------------------------------------- runner (mock judge, temp project)

def _project(tmp_path, cases):
    root = tmp_path / 'proj'
    shutil.copytree(ROOT / 'kb' / 'v4', root / 'kb' / 'v4')
    d = root / 'data' / 'testbed' / 'chronic-v2'
    d.mkdir(parents=True)
    with (d / 'cases.jsonl').open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps(c) + '\n')
    with (d / 'renders-mas_transcript-writer-2.jsonl').open('w', encoding='utf-8') as f:
        for c in cases:
            f.write(json.dumps({'case_id': c['case_id'], 'status': 'ok', 'text': plain_document(c),
                                'text_sha256': 'x'}) + '\n')
    return root


def test_runner_run_and_score_with_mock(tmp_path, cases, capsys):
    import run_review_v4 as RV
    sub = [c for c in cases if c['patient']['patient_id'] in ('TB-0060', 'TB-0061', 'TB-0062', 'TB-0063', 'TB-0064')]
    root = _project(tmp_path, sub)
    assert RV.main(['run', '--root', str(root), '--split', 'test', '--threads', '1']) == 0
    run_dir = next((root / 'reports').glob('review-v4-*'))
    assert RV.main(['run', '--root', str(root), '--run-dir', str(run_dir), '--arm', 'model_only',
                    '--split', 'test', '--threads', '1']) == 0
    units = RV.load_inputs(root, 'test', 'mas_transcript')
    assert units and all('gold' not in u and 'plan' not in u and 'patient' not in u for u in units)
    assert RV.main(['score', '--root', str(root), '--run-dir', str(run_dir)]) == 0
    metrics = json.loads(next(run_dir.glob('score-*/metrics.json')).read_text(encoding='utf-8'))
    assert set(metrics['arms']) == {'full-test-mas_transcript', 'model_only-test-mas_transcript',
                                    'rules_only-test-mas_transcript'}
    assert metrics['arms']['full-test-mas_transcript']['acceptance'] is not None
    rec_path = run_dir / 'full-test-mas_transcript' / 'records.jsonl'
    rec_path.write_text(rec_path.read_text(encoding='utf-8') + '\n{"tampered": 1}\n', encoding='utf-8')
    with pytest.raises(SystemExit):
        RV.main(['score', '--root', str(root), '--run-dir', str(run_dir)])


def test_primary_subset(tmp_path, cases):
    import run_review_v4 as RV
    root = _project(tmp_path, [c for c in cases if c['split'] == 'test'])
    units = RV.load_inputs(root, 'test', 'mas_transcript', subset='primary')
    by = {c['case_id']: c for c in cases}
    assert len(units) == sum(RV.primary_subset(c) for c in cases if c['split'] == 'test')
    assert all(by[u['case_id']]['gold'] is None or by[u['case_id']]['gold']['severity'] == 'severe' for u in units)
    assert not any(by[u['case_id']]['gold'] and by[u['case_id']]['gold']['error_id'] in HELD_OUT_TYPES for u in units)


def test_reconcile_downgrades_only_claims_that_contradict_code_checks():
    from medical_jury.v4.pipeline import reconcile
    meds = [{'drug': 'empagliflozin', 'status': 'start'}, {'drug': 'ibuprofen', 'status': 'start'},
            {'drug': 'amlodipine', 'status': 'start'}, {'drug': 'pioglitazone', 'status': 'start'}]
    flagged = [{'source': 'rule', 'id': 'DOSE_OVER_MAX', 'category': 'dose_error', 'drugs': ['amlodipine'], 'issue': ''}]
    cc = CK.code_checks_of(flagged, meds)

    def f(entity, issue, cat='contraindicated_drug', sev='severe', src='open_review'):
        return {'source': src, 'severity': sev, 'category': cat, 'entity': entity, 'issue': issue}
    out = reconcile([f('empagliflozin', 'contraindicated with eGFR 27'),          # code checked eGFR: downgrade
                     f('empagliflozin', 'contraindicated with potassium 5.7'),    # code checked potassium: downgrade
                     f('ibuprofen', 'NSAID contraindicated in CKD, eGFR 35'),    # kidney not checked for NSAID: keep
                     f('pioglitazone', 'contraindicated in heart failure'),      # not checked: keep
                     f('amlodipine', '50 mg above maximum', 'dose_error'),       # code flagged it: keep
                     f('empagliflozin', 'ketoacidosis risk with eGFR 25'),       # names an unchecked risk: keep
                     f('empagliflozin', 'eGFR 27', sev='moderate'),               # not severe: untouched
                     f('empagliflozin', 'eGFR 27', src='criterion')], cc)         # not open review: untouched
    assert [x['severity'] for x in out] == ['moderate', 'moderate', 'severe', 'severe', 'severe', 'severe',
                                            'moderate', 'severe']
    assert out[0]['severity_model'] == 'severe' and 'empagliflozin' in out[0]['reconciled']
    assert SC.found({'error_id': 'C08_nsaid_ckd', 'entity': 'ibuprofen'}, {'findings': out})


def test_empirical_bound_requires_same_prompts_and_sealed_pilot(tmp_path):
    import hashlib
    import run_review_v4 as RV
    d = tmp_path / 'pilot' / 'full-train-mas_transcript'
    d.mkdir(parents=True)
    rows = [{'case_id': f'C{i}', 'calls': [{'usage': {'prompt_tokens': 1000, 'completion_tokens': 500 + i}}]}
            for i in range(20)]
    (d / 'records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    (d / 'records-seal.json').write_text(json.dumps({'sha256': hashlib.sha256((d / 'records.jsonl').read_bytes())
                                                     .hexdigest()}), encoding='utf-8')
    prompts = {'check': {'version': 'x', 'sha256': 'y'}}
    (d / 'plan.json').write_text(json.dumps({'prompts': prompts, 'arm': 'full'}), encoding='utf-8')

    class S:
        input_rate, output_rate = 13.2, 39.6
    b = RV.empirical_upper_bound(d / 'records.jsonl', 100, 'full', prompts, S)
    worst = (1000 * 13.2 + 519 * 39.6) / 1e6
    assert abs(b['bound_cny'] - round(100 * worst * 1.3, 4)) < 1e-9
    with pytest.raises(SystemExit):
        RV.empirical_upper_bound(d / 'records.jsonl', 100, 'full', {'check': {'version': 'z'}}, S)
    assert RV.empirical_upper_bound(d / 'records.jsonl', 10, 'model_only', prompts, S)['bound_cny'] > 0
