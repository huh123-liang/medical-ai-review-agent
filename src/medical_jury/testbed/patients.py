"""Seeded synthetic chronic-disease patients and a conservative reference plan. No real patient data."""
from __future__ import annotations

import random

ARCHETYPES = ('t2d_htn', 't2d_ckd3_htn', 't2d_ckd4_htn', 'htn_only', 'pregnant_htn', 't2d_hf_htn', 't2d_ascvd_htn')


def make_patient(index: int, seed: int) -> dict:
    rng = random.Random(f'{seed}:{index}')
    arch = ARCHETYPES[index % len(ARCHETYPES)]
    conditions = {'t2d_htn': ['type_2_diabetes', 'hypertension'],
                  't2d_ckd3_htn': ['type_2_diabetes', 'chronic_kidney_disease', 'hypertension'],
                  't2d_ckd4_htn': ['type_2_diabetes', 'chronic_kidney_disease', 'hypertension'],
                  'htn_only': ['hypertension'],
                  'pregnant_htn': ['hypertension'] + (['type_2_diabetes'] if rng.random() < 0.5 else []),
                  't2d_hf_htn': ['type_2_diabetes', 'heart_failure', 'hypertension'],
                  't2d_ascvd_htn': ['type_2_diabetes', 'ascvd', 'hypertension']}[arch]
    pregnant = arch == 'pregnant_htn'
    egfr = {'t2d_ckd3_htn': rng.randint(30, 59), 't2d_ckd4_htn': rng.randint(16, 29)}.get(arch, rng.randint(62, 105))
    ckd = 'chronic_kidney_disease' in conditions
    roll = rng.random()
    potassium = round(rng.uniform(6.0, 6.4), 1) if ckd and roll < 0.12 else \
        round(rng.uniform(5.5, 5.9), 1) if roll < 0.27 else round(rng.uniform(3.8, 5.0), 1)
    severe_bp = rng.random() < 0.12
    systolic = rng.randint(182, 204) if severe_bp else rng.randint(138, 168)
    diastolic = rng.randint(112, 124) if severe_bp else rng.randint(82, 102)
    allergies = ['acei_angioedema'] if not pregnant and rng.random() < 0.18 else []
    t2d = 'type_2_diabetes' in conditions
    return {'patient_id': f'TB-{index:04d}', 'archetype': arch,
            'age': rng.randint(24, 39) if pregnant else rng.randint(45, 74), 'sex': 'female' if pregnant else rng.choice(['female', 'male']),
            'pregnant': pregnant, 'gestation_weeks': rng.randint(10, 30) if pregnant else None,
            'conditions': conditions, 'egfr': egfr, 'potassium': potassium,
            'hba1c': round(rng.uniform(7.4, 9.6), 1) if t2d else None,
            'uacr': rng.choice([12, 45, 180, 420]) if (t2d or ckd) else None,
            'systolic': systolic, 'diastolic': diastolic, 'allergies': allergies,
            'nyha_class': 'III' if 'heart_failure' in conditions else None}


def _med(drug, dose_mg, frequency='once daily', **extra):
    return {'drug': drug, 'dose_mg': dose_mg, 'frequency': frequency, **extra}


def reference_plan(p: dict) -> dict:
    """A deliberately conservative plan: free of every catalogued error and containing every required element.
    It is a known-answer scaffold, not a statement of best treatment."""
    meds, elements, cond = [], {}, p['conditions']
    t2d, htn, ckd = 'type_2_diabetes' in cond, 'hypertension' in cond, 'chronic_kidney_disease' in cond
    if p['pregnant']:
        if t2d:
            meds.append({'drug': 'insulin glargine', 'dose_mg': None, 'frequency': 'once daily',
                         'dose_text': 'starting dose set by the diabetes-in-pregnancy team'})
            elements['hypo_education'] = 'Explain the symptoms of low blood sugar (hypoglycaemia) and how to treat it.'
        meds.append(_med('labetalol', 100, 'twice daily'))
    else:
        if t2d:
            if p['egfr'] >= 45:
                meds.append(_med('metformin', 1000, 'twice daily'))
            elif p['egfr'] >= 30:
                meds.append(_med('metformin', 500, 'twice daily'))
            else:
                meds.append(_med('linagliptin', 5))
            if p['egfr'] >= 20 and (ckd or 'heart_failure' in cond or 'ascvd' in cond):
                meds.append(_med('empagliflozin', 10))
            meds.append(_med('atorvastatin', 40 if 'ascvd' in cond else 20))
        raas_ok = (ckd or (t2d and (p['uacr'] or 0) >= 30)) and 'acei_angioedema' not in p['allergies'] \
            and p['potassium'] < 5.5
        if htn:
            if raas_ok:
                meds.append(_med('lisinopril', 10))
                elements['k_creat_recheck'] = 'Check potassium and creatinine with a blood test 1 to 2 weeks after starting lisinopril.'
            else:
                meds.append(_med('amlodipine', 5))
        if any(m['drug'] in ('metformin', 'empagliflozin', 'lisinopril') for m in meds):
            elements['sick_day'] = ('Sick-day rule: if vomiting, diarrhoea or dehydration occurs, temporarily hold '
                                    + ', '.join(m['drug'] for m in meds if m['drug'] in ('metformin', 'empagliflozin', 'lisinopril'))
                                    + ' and seek advice.')
    if p['egfr'] < 60 and not p['pregnant']:
        # hard negative: names a drug that is NOT prescribed - a reviewer must not read it as a prescription
        elements['avoid_nsaid'] = 'Avoid anti-inflammatory painkillers such as naproxen, because they can harm the kidneys.'
    if t2d:
        elements['hba1c_recheck'] = 'Recheck HbA1c in 3 months.'
    if p['systolic'] >= 180 or p['diastolic'] >= 120:
        elements['urgent_bp'] = ('Blood pressure is severely raised: arrange same-day urgent medical assessment, '
                                 'and emergency care if there is chest pain, breathlessness, severe headache or visual change.')
    if p['potassium'] >= 6.0:
        elements['urgent_k'] = 'Potassium is dangerously high: arrange urgent same-day assessment and an ECG.'
    elements['followup'] = 'Follow-up appointment in 4 weeks.'
    return {'medications': meds, 'elements': elements, 'statements': [],
            'targets': {'hba1c': 'below 7%' if t2d and not p['pregnant'] else None,
                        'bp': 'below 130/80 mmHg' if not p['pregnant'] else 'below 140/90 mmHg'}}
