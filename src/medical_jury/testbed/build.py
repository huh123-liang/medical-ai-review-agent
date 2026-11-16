"""Deterministic case construction and the prompts for the local writer model."""
from __future__ import annotations

import hashlib
import json
import random

from . import TESTBED_VERSION
from .errors import CATALOGUE, HELD_OUT_TYPES, applicable, inject
from .patients import make_patient, reference_plan
from .rules import RULES_VERSION, check_plan

WRITER_PROMPT_VERSION = 'writer-2'   # writer-1 (first 6-case trial) let a lone statement form the patient section
PATIENT_SECTION = {'mas_transcript': r'##\s*Recommendation to the patient', 'clinic_letter': r'Your medicines'}
STYLES = {
    'mas_transcript': (
        'You are the write-up component of a clinical multi-agent system. The system has ALREADY decided the final '
        'plan; you only write it up. Write the output document in English with exactly these sections:\n'
        '## Case summary\n## Physician agent: assessment and plan\n## Pharmacist agent: medication list\n'
        '## Recommendation to the patient\n'
        'The last section speaks to the patient directly and repeats, in plain language, every medication, every '
        'monitoring/safety item, every target and every statement of the plan, woven together in one natural text. '
        'Rules: state every medication with its dose in mg and its frequency exactly as given (generic names). '
        'Include every monitoring/safety item, every target and every statement of the plan. Do not add any other '
        'medication, test, warning or advice. Do not question, comment on or correct the plan. 250-400 words.'),
    'clinic_letter': (
        'You write a short clinic letter to a patient from a plan that has ALREADY been decided. English, plain '
        'language, with exactly these headings: Summary / Your medicines / What to watch for / Next steps.\n'
        'Rules: state every medicine with its dose in mg and how often, exactly as given (generic names). Include '
        'every monitoring/safety item, every target and every statement of the plan. Add nothing else and do not '
        'comment on or correct the plan. 180-300 words.'),
}


def patient_section_ok(text: str, plan: dict, style: str) -> bool:
    """The patient-facing part must carry the plan, not a single sentence: it has to name at least one medication
    (or, for a plan without medications, be at least two sentences long)."""
    import re
    m = re.search(PATIENT_SECTION[style], text, re.I)
    if not m:
        return False
    part = text[m.end():]
    drugs = [x['drug'] for x in plan['medications']]
    if drugs:
        return any(re.search(rf'{d}\b', part, re.I) for d in drugs)
    return len(re.findall(r'[.!?](\s|$)', part)) >= 2


def patient_summary(p: dict) -> str:
    cond = ', '.join(c.replace('_', ' ') for c in p['conditions'])
    parts = [f"{p['age']}-year-old {p['sex']}", f'conditions: {cond}']
    if p['pregnant']:
        parts.append(f"pregnant, {p['gestation_weeks']} weeks")
    if p['nyha_class']:
        parts.append(f"heart failure NYHA class {p['nyha_class']}")
    parts.append(f"blood pressure {p['systolic']}/{p['diastolic']} mmHg")
    parts.append(f"eGFR {p['egfr']} mL/min/1.73m2, serum potassium {p['potassium']} mmol/L")
    if p['hba1c'] is not None:
        parts.append(f"HbA1c {p['hba1c']}%")
    if p['uacr'] is not None:
        parts.append(f"urine albumin-creatinine ratio {p['uacr']} mg/g")
    parts.append('allergy: previous angioedema with an ACE inhibitor' if p['allergies'] else 'no known drug allergies')
    return '; '.join(parts) + '.'


def plan_text(plan: dict) -> str:
    lines = ['MEDICATIONS:']
    for m in plan['medications']:
        dose = f"{m['dose_mg']:g} mg" if m['dose_mg'] is not None else m['dose_text']
        lines.append(f"- {m['drug']} {dose} {m['frequency']}")
    lines.append('TARGETS:')
    lines += [f'- {k.upper() if k == "bp" else "HbA1c"} target {v}' for k, v in plan['targets'].items() if v]
    lines.append('MONITORING AND SAFETY ITEMS:')
    lines += [f'- {t}' for t in plan['elements'].values()] or ['- (none)']
    if plan['statements']:
        lines.append('STATEMENTS TO INCLUDE WORD FOR WORD:')
        lines += [f'- {t}' for t in plan['statements']]
    return '\n'.join(lines)


def writer_messages(case: dict, style: str) -> list[dict]:
    return [{'role': 'system', 'content': STYLES[style]},
            {'role': 'user', 'content': f"PATIENT: {case['patient_summary']}\n\nFINAL PLAN\n{plan_text(case['plan'])}"}]


TRAIN_PATIENTS = 50          # patients 0..49 are the training split (DSPy may see them); the rest are held out
DOUBLE_ERROR_CASES = 30      # held-out patients that also get one case with two errors


def build_cases(n_patients: int, seed: int, errors_per_patient: int = 2) -> tuple[list[dict], dict]:
    """One clean case per patient plus ``errors_per_patient`` single-error variants (least-used type first, counted
    per split). Training patients never receive a held-out error type; unsafe statements use wordings 0-1 in training
    and 2-3 in the held-out split. The first DOUBLE_ERROR_CASES held-out patients with two compatible seen types also
    get a two-error case. Fully determined by (n_patients, seed)."""
    rng = random.Random(seed)
    used = {'train': {e: 0 for e in CATALOGUE}, 'test': {e: 0 for e in CATALOGUE}}
    cases, doubles = [], 0
    for i in range(n_patients):
        split = 'train' if i < min(TRAIN_PATIENTS, n_patients // 3) else 'test'
        p = make_patient(i, seed)
        ref = reference_plan(p)
        if check_plan(p, ref['medications']):
            raise AssertionError(f'reference plan of {p["patient_id"]} violates the rule layer')
        base = {'patient': p, 'patient_summary': patient_summary(p), 'split': split}
        cases.append({'case_id': f'{p["patient_id"]}-clean', **base, 'plan': ref, 'gold': None})
        options = [e for e in applicable(p, ref) if split == 'test' or e not in HELD_OUT_TYPES]
        rng.shuffle(options)
        variant = rng.randrange(2) + (2 if split == 'test' else 0)
        for eid in sorted(options, key=lambda e: used[split][e])[:errors_per_patient]:
            plan, gold = inject(p, ref, eid, variant)
            used[split][eid] += 1
            cases.append({'case_id': f'{p["patient_id"]}-{eid.split("_")[0]}', **base, 'plan': plan, 'gold': gold})
        if split == 'test' and doubles < DOUBLE_ERROR_CASES:
            seen = [e for e in options if e not in HELD_OUT_TYPES]
            pair = next(((a, b) for a in seen for b in seen
                         if CATALOGUE[a][0] != CATALOGUE[b][0] and _compatible(p, ref, a, b, variant)), None)
            if pair:
                plan, g1 = inject(p, ref, pair[0], variant)
                plan, g2 = inject(p, plan, pair[1], variant, second=True)
                doubles += 1
                cases.append({'case_id': f'{p["patient_id"]}-D{pair[0][:3]}{pair[1][:3]}', **base, 'plan': plan,
                              'gold': [g1, g2]})
    body = json.dumps(cases, sort_keys=True, ensure_ascii=False).encode()

    def golds(c):
        return [] if not c['gold'] else c['gold'] if isinstance(c['gold'], list) else [c['gold']]
    manifest = {'testbed': TESTBED_VERSION, 'rules': RULES_VERSION, 'seed': seed, 'patients': n_patients,
                'errors_per_patient': errors_per_patient, 'cases': len(cases),
                'held_out_types': list(HELD_OUT_TYPES),
                'by_split': {sp: {'cases': sum(c['split'] == sp for c in cases),
                                  'clean': sum(c['split'] == sp and not c['gold'] for c in cases),
                                  'double_error': sum(c['split'] == sp and isinstance(c['gold'], list) for c in cases),
                                  'severe_single': sum(c['split'] == sp and isinstance(c['gold'], dict)
                                                       and c['gold']['severity'] == 'severe' for c in cases),
                                  'by_error': used[sp]} for sp in ('train', 'test')},
                'cases_sha256': hashlib.sha256(body).hexdigest()}
    return cases, manifest


def _compatible(p, ref, a, b, variant) -> bool:
    """Two errors can share a case if the second still applies after the first and they touch different things."""
    try:
        plan, g1 = inject(p, ref, a, variant)
        plan2, g2 = inject(p, plan, b, variant, second=True)
    except (ValueError, KeyError):
        return False
    for g in (g1, g2):       # a rule-visible error must still be rule-visible in the combined plan
        if g['rule_covered'] and not any(g['entity'] in v['drugs'] for v in check_plan(p, plan2['medications'])):
            return False
    from .errors import verify
    text = '## Recommendation to the patient\n' + plan_text(plan2).replace('\n- ', '\n')
    return g1['entity'] != g2['entity'] and verify(plan2, [g1, g2], text)['ok']
