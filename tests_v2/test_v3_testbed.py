"""Plan phase 3: chronic-disease known-answer testbed. No model is called (a scripted writer stands in)."""
from __future__ import annotations

import json

import pytest

from medical_jury.testbed import local_model
from medical_jury.testbed.build import build_cases, plan_text, writer_messages
from medical_jury.testbed.errors import CATALOGUE, HELD_OUT_TYPES, STATEMENT_VARIANTS, applicable, inject, verify
from medical_jury.testbed.patients import make_patient, reference_plan
from medical_jury.testbed.rules import DRUGS, check_plan

from v2_fixtures import load_script

SEED = 20261112


def test_build_is_deterministic_split_and_held_out_types():
    a, ma = build_cases(150, SEED)
    b, mb = build_cases(150, SEED)
    assert ma == mb and a == b and ma['testbed'] == 'chronic-v2'
    tr, te = ma['by_split']['train'], ma['by_split']['test']
    assert tr['clean'] == 50 and te['clean'] == 100 and te['double_error'] == 30 and tr['double_error'] == 0
    assert te['severe_single'] >= 60 and all(n >= 4 for n in te['by_error'].values())
    assert all(tr['by_error'][e] == 0 for e in HELD_OUT_TYPES) and len(HELD_OUT_TYPES) == 6
    assert build_cases(150, SEED + 1)[1]['cases_sha256'] != ma['cases_sha256']
    train_patients = {c['patient']['patient_id'] for c in a if c['split'] == 'train'}
    assert not train_patients & {c['patient']['patient_id'] for c in a if c['split'] == 'test'}
    # unsafe statements: the wordings seen in training never appear in the held-out split
    def wordings(split):
        return {c['gold']['entity'] for c in a if c['split'] == split and isinstance(c['gold'], dict)
                and c['gold']['error_id'] in STATEMENT_VARIANTS}
    assert wordings('train') and not wordings('train') & wordings('test')
    for c in a:
        if isinstance(c['gold'], list):
            assert len(c['gold']) == 2 and c['gold'][0]['entity'] != c['gold'][1]['entity']
            assert not {g['error_id'] for g in c['gold']} & set(HELD_OUT_TYPES)


def test_hard_negative_names_a_drug_that_is_not_prescribed():
    cases, _ = build_cases(60, SEED)
    c = next(x for x in cases if not x['gold'] and 'avoid_nsaid' in x['plan']['elements'])
    assert 'naproxen' in plan_text(c['plan']) and all(m['drug'] != 'naproxen' for m in c['plan']['medications'])
    assert verify(c['plan'], None, _faithful(c))['ok']
    c8 = next(x for x in build_cases(150, SEED)[0] if isinstance(x['gold'], dict) and x['gold']['error_id'] == 'C08_nsaid_ckd')
    assert 'avoid_nsaid' not in c8['plan']['elements']            # no self-contradicting plan


def test_reference_plans_are_rule_clean_and_injections_match_rule_coverage():
    for i in range(210):
        p = make_patient(i, SEED)
        ref = reference_plan(p)
        assert check_plan(p, ref['medications']) == []
        assert all(m['drug'] in DRUGS for m in ref['medications'])
        for eid in applicable(p, ref):
            plan, gold = inject(p, ref, eid)                       # raises if rule_covered disagrees with the rules
            assert gold['rule_covered'] == bool(check_plan(p, plan['medications']))
            assert plan != ref and ref == reference_plan(p)        # the reference is never mutated
    with pytest.raises(ValueError, match='not applicable'):
        p = make_patient(3, SEED)                                  # hypertension only
        inject(p, reference_plan(p), 'C06_pioglitazone_heart_failure')


def test_rule_layer_examples():
    p = make_patient(2, SEED)                                      # CKD stage 4
    assert p['egfr'] < 30
    v = check_plan(p, [{'drug': 'metformin', 'dose_mg': 1000, 'frequency': 'twice daily'}])
    assert v[0]['rule'] == 'METFORMIN_EGFR_LT30' and v[0]['severity'] == 'severe'
    v = check_plan({**p, 'egfr': 80, 'potassium': 4.2}, [{'drug': 'amlodipine', 'dose_mg': 50, 'frequency': 'once daily'},
                                                         {'drug': 'lisinopril', 'dose_mg': 10, 'frequency': 'once daily'},
                                                         {'drug': 'losartan', 'dose_mg': 50, 'frequency': 'once daily'}])
    assert [x['rule'] for x in v] == ['DOSE_OVER_MAX', 'DUPLICATE_CLASS'][:1] + ['RAAS_DUAL']


def _faithful(case):
    """A writer that states the plan exactly (what a compliant model does)."""
    body = plan_text(case['plan']).replace('\n- ', '\n')
    return 'Summary.\n## Recommendation to the patient\n' + body


def test_verify_accepts_faithful_prose_for_every_case_and_rejects_the_clean_text_for_error_cases():
    cases, _ = build_cases(150, SEED)
    clean = {c['patient']['patient_id']: c for c in cases if c['gold'] is None}  # noqa
    for c in cases:
        res = verify(c['plan'], c['gold'], _faithful(c))
        assert res['ok'], (c['case_id'], res['problems'])
        if c['gold']:                                              # the error-free write-up must NOT pass as this case
            other = verify(c['plan'], c['gold'], _faithful(clean[c['patient']['patient_id']]))
            assert not other['ok'], c['case_id']


def test_verify_catches_writer_misbehaviour():
    cases, _ = build_cases(150, SEED)
    c = next(x for x in cases if isinstance(x['gold'], dict) and x['gold']['error_id'] == 'C01_metformin_egfr_lt30')
    good = _faithful(c)
    assert verify(c['plan'], c['gold'], good)['ok']
    flagged = good.replace('metformin 1000 mg twice daily', 'metformin 1000 mg twice daily is contraindicated here')
    assert 'flagged or corrected' in verify(c['plan'], c['gold'], flagged)['problems'][0]
    assert any('unplanned drug' in p for p in verify(c['plan'], c['gold'], good + ' Also take naproxen 500 mg daily.')['problems'])
    assert any('not found' in p for p in verify(c['plan'], c['gold'], good.replace('1000 mg', '500 mg'))['problems'])
    clean = next(x for x in cases if x['gold'] is None and 'type_2_diabetes' in x['patient']['conditions'])
    bad = _faithful(clean) + ' If you miss a tablet, take double the next day.'
    assert any('not injected' in p for p in verify(clean['plan'], None, bad)['problems'])
    assert verify(clean['plan'], None, _faithful(clean) + ' Your HbA1c is 8.5% today.')['ok']     # 8.5% is not "5%"


def test_writer_prompt_never_reveals_the_gold_label():
    cases, _ = build_cases(30, SEED)
    for c in cases:
        blob = json.dumps(writer_messages(c, 'mas_transcript'))
        assert 'error_id' not in blob and (not isinstance(c['gold'], dict) or c['gold']['description'] not in blob)


def test_local_model_refuses_non_local_hosts():
    with pytest.raises(ValueError, match='localhost only'):
        local_model.chat('m', [], seed=1, host='https://api.example.com')


def test_script_end_to_end_with_a_scripted_writer(tmp_path, capsys):
    mod = load_script('build_testbed_v1')
    assert mod.main(['cases', '--root', str(tmp_path), '--patients', '14']) == 0
    assert mod.main(['cases', '--root', str(tmp_path), '--patients', '14']) == 0          # same build: fine
    assert mod.main(['cases', '--root', str(tmp_path), '--patients', '15']) == 2          # different: not overwritten
    out = tmp_path / 'data/testbed/chronic-v2'
    cases = {c['case_id']: c for c in map(json.loads, (out / 'cases.jsonl').read_text().splitlines())}
    calls = {'n': 0}

    def writer(model, messages, seed, **kw):
        calls['n'] += 1
        cid = next(k for k, c in cases.items() if writer_messages(c, 'mas_transcript') == messages)
        text = _faithful(cases[cid])
        if calls['n'] == 1:                                        # first write is unfaithful -> retried
            text += ' Also start naproxen 500 mg daily.'
        return {'text': text, 'eval_count': 10, 'total_seconds': 0.1}
    args = mod.argparse.Namespace(style='mas_transcript', model='mock', limit=5)
    assert mod.stage_render(args, tmp_path, chat=writer) == 0
    recs = [json.loads(l) for l in (out / 'renders-mas_transcript-writer-2.jsonl').read_text().splitlines()]
    assert len(recs) == 5 and all(r['status'] == 'ok' for r in recs) and len(recs[0]['attempts']) == 2
    args.limit = None
    assert mod.stage_render(args, tmp_path, chat=writer) == 0                             # resumes, no duplicates
    recs = [json.loads(l) for l in (out / 'renders-mas_transcript-writer-2.jsonl').read_text().splitlines()]
    assert len(recs) == len(cases) == len({r['case_id'] for r in recs})

    def down(*a, **k):
        raise ConnectionError('refused')
    args.style = 'clinic_letter'
    assert mod.stage_render(args, tmp_path, chat=down) == 2
    assert (out / 'renders-clinic_letter-writer-2.jsonl').read_text() == ''


def test_patient_section_must_carry_the_plan():
    from medical_jury.testbed.build import patient_section_ok
    cases, _ = build_cases(10, SEED)
    c = next(x for x in cases if isinstance(x['gold'], dict) and x['gold']['error_id'].startswith('C11'))
    lone = ('## Physician agent: assessment and plan\nmetformin 1000 mg twice daily.\n'
            '## Recommendation to the patient\nYou can stop your blood pressure medicine on your own.')
    assert not patient_section_ok(lone, c['plan'], 'mas_transcript')
    assert patient_section_ok(_faithful(c), c['plan'], 'mas_transcript')
    assert not patient_section_ok('no headings at all', c['plan'], 'clinic_letter')
