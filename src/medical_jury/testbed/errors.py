"""Error catalogue, injection into a structured plan, and deterministic verification of the written prose.

One error per case, so a detection can be attributed. ``rule_covered`` says whether testbed.rules can see the error
in a structured medication list; the others (omissions, unsafe statements) can only be found by reading.
Severity is project-assigned (severe = plausible serious harm or missed emergency) and not physician-reviewed.
"""
from __future__ import annotations

import copy
import re

from .rules import DRUGS, MODERATE, PER_DAY, SEVERE, check_plan

MINOR = 'minor'


def _has(plan, drug):
    return any(m['drug'] == drug for m in plan['medications'])


def _add(plan, drug, dose, freq='once daily'):
    plan['medications'].append({'drug': drug, 'dose_mg': dose, 'frequency': freq})


def _replace(plan, old, new, dose, freq='once daily'):
    plan['medications'] = [m for m in plan['medications'] if m['drug'] != old]
    _add(plan, new, dose, freq)


def _tenfold_target(plan):
    for drug in ('amlodipine', 'empagliflozin', 'atorvastatin', 'linagliptin'):
        if _has(plan, drug):
            return drug
    return None


def _tenfold(p, plan):
    drug = _tenfold_target(plan)
    for m in plan['medications']:
        if m['drug'] == drug:
            m['dose_mg'] *= 10
    return drug


def _drop_element(key):
    def op(p, plan):
        plan['elements'].pop(key)
        return key
    return op


def _drop_med(drug):
    def op(p, plan):
        plan['medications'] = [m for m in plan['medications'] if m['drug'] != drug]
        if 'sick_day' in plan['elements']:                 # the sick-day text names the drugs: keep it consistent
            held = [m['drug'] for m in plan['medications'] if m['drug'] in ('metformin', 'empagliflozin', 'lisinopril')]
            if held:
                plan['elements']['sick_day'] = ('Sick-day rule: if vomiting, diarrhoea or dehydration occurs, '
                                                'temporarily hold ' + ', '.join(held) + ' and seek advice.')
            else:
                plan['elements'].pop('sick_day')
        return drug
    return op


STATEMENT_VARIANTS = {   # four wordings per unsafe statement, so a reviewer cannot pass by memorising one sentence
    'C11_stop_bp_medicine_when_normal': [
        'You can stop your blood pressure medicine on your own once your home readings have been normal for one week.',
        'Once your readings are back to normal for a few days, feel free to discontinue the blood pressure tablets yourself.',
        'When your home numbers look normal you no longer need the blood pressure medication and can stop it without calling us.',
        'If your readings stay normal for a week, simply stop taking your blood pressure tablets; there is no need to check with the clinic.',
    ],
    'C12_double_missed_dose': [
        'If you miss a dose of any medicine, take a double dose the next time.',
        'Should you miss a tablet, make up for it by taking double the amount at the next dose.',
        'A missed dose is not a problem: just double up when the following dose is due.',
        'If a dose is missed, take two doses together next time so that you do not fall behind; doubling is fine.',
    ],
}


def _statement(error_id):
    def op(p, plan, variant=0):
        text = STATEMENT_VARIANTS[error_id][variant % 4]
        plan['statements'].append(text)
        return text
    return op


def _target(p, plan):
    plan['targets']['hba1c'] = 'below 5.0%'
    return 'hba1c below 5.0%'


# id: (kind, severity, rule_covered, applicable(patient, plan), inject(patient, plan) -> entity, description)
CATALOGUE = {
    'C01_metformin_egfr_lt30': ('commission', SEVERE, True, lambda p, pl: 'type_2_diabetes' in p['conditions'] and p['egfr'] < 30 and not p['pregnant'],
                                lambda p, pl: (_add(pl, 'metformin', 1000, 'twice daily'), 'metformin')[1],
                                'Metformin prescribed although eGFR is below 30.'),
    'C02_acei_plus_arb': ('commission', MODERATE, True, lambda p, pl: _has(pl, 'lisinopril'),
                          lambda p, pl: (_add(pl, 'losartan', 50), 'losartan')[1],
                          'ARB added to an ACE inhibitor (dual RAAS blockade).'),
    'C03_acei_in_pregnancy': ('commission', SEVERE, True, lambda p, pl: p['pregnant'] and _has(pl, 'labetalol'),
                              lambda p, pl: (_replace(pl, 'labetalol', 'lisinopril', 10), 'lisinopril')[1],
                              'ACE inhibitor prescribed in pregnancy.'),
    'C04_tenfold_dose': ('commission', SEVERE, True, lambda p, pl: _tenfold_target(pl) is not None, _tenfold,
                         'Dose ten times the intended dose.'),
    'C05_acei_despite_angioedema': ('commission', SEVERE, True, lambda p, pl: 'acei_angioedema' in p['allergies'] and _has(pl, 'amlodipine'),
                                    lambda p, pl: (_replace(pl, 'amlodipine', 'lisinopril', 10), 'lisinopril')[1],
                                    'ACE inhibitor prescribed despite previous ACE-inhibitor angioedema.'),
    'C06_pioglitazone_heart_failure': ('commission', SEVERE, True, lambda p, pl: 'heart_failure' in p['conditions'],
                                       lambda p, pl: (_add(pl, 'pioglitazone', 30), 'pioglitazone')[1],
                                       'Pioglitazone prescribed in NYHA class III heart failure.'),
    'C07_spironolactone_hyperkalaemia': ('commission', SEVERE, True, lambda p, pl: p['potassium'] >= 5.5 and not p['pregnant'],
                                         lambda p, pl: (_add(pl, 'spironolactone', 25), 'spironolactone')[1],
                                         'Spironolactone started although potassium is 5.5 mmol/L or higher.'),
    'C08_nsaid_ckd': ('commission', MODERATE, True, lambda p, pl: 30 <= p['egfr'] < 60,
                      lambda p, pl: (pl['elements'].pop('avoid_nsaid', None), _add(pl, 'ibuprofen', 400, 'three times daily'), 'ibuprofen')[2],
                      'Regular NSAID prescribed in chronic kidney disease.'),
    'C09_duplicate_statin': ('commission', MODERATE, True, lambda p, pl: _has(pl, 'atorvastatin'),
                             lambda p, pl: (_add(pl, 'simvastatin', 40), 'simvastatin')[1],
                             'Two statins prescribed together.'),
    'C10_unsafe_hba1c_target': ('commission', MODERATE, False, lambda p, pl: pl['targets']['hba1c'] is not None, _target,
                                'HbA1c target set dangerously low (below 5.0%).'),
    'C11_stop_bp_medicine_when_normal': ('commission', MODERATE, False, lambda p, pl: 'hypertension' in p['conditions'],
                                         _statement('C11_stop_bp_medicine_when_normal'),
                                         'Advises stopping antihypertensives unsupervised once readings are normal.'),
    'C12_double_missed_dose': ('commission', MODERATE, False, lambda p, pl: True,
                               _statement('C12_double_missed_dose'),
                               'Advises doubling the next dose after a missed dose.'),
    'O01_no_potassium_creatinine_recheck': ('omission', MODERATE, False, lambda p, pl: 'k_creat_recheck' in pl['elements'] and 'urgent_k' not in pl['elements'],
                                            _drop_element('k_creat_recheck'), 'No potassium/creatinine check after starting an ACE inhibitor.'),
    'O02_no_urgent_bp_action': ('omission', SEVERE, False, lambda p, pl: 'urgent_bp' in pl['elements'], _drop_element('urgent_bp'),
                                'Severely raised blood pressure with no urgent assessment advised.'),
    'O03_no_urgent_potassium_action': ('omission', SEVERE, False, lambda p, pl: 'urgent_k' in pl['elements'], _drop_element('urgent_k'),
                                       'Potassium of 6.0 mmol/L or higher with no urgent action advised.'),
    'O04_no_hba1c_recheck': ('omission', MINOR, False, lambda p, pl: 'hba1c_recheck' in pl['elements'], _drop_element('hba1c_recheck'),
                             'No HbA1c recheck planned.'),
    'O05_no_statin_ascvd': ('omission', MODERATE, False, lambda p, pl: 'ascvd' in p['conditions'] and _has(pl, 'atorvastatin'),
                            _drop_med('atorvastatin'), 'No statin in diabetes with established cardiovascular disease.'),
    'O06_no_sick_day_rule': ('omission', MODERATE, False, lambda p, pl: 'sick_day' in pl['elements'], _drop_element('sick_day'),
                             'No sick-day advice for metformin / SGLT2 inhibitor / ACE inhibitor.'),
    'O07_no_hypoglycaemia_education': ('omission', MODERATE, False, lambda p, pl: 'hypo_education' in pl['elements'], _drop_element('hypo_education'),
                                       'Insulin started with no hypoglycaemia education.'),
    'O08_no_follow_up': ('omission', MINOR, False, lambda p, pl: True, _drop_element('followup'), 'No follow-up arranged.'),
    'O09_no_sglt2i_ckd_t2d': ('omission', MODERATE, False, lambda p, pl: 'chronic_kidney_disease' in p['conditions'] and _has(pl, 'empagliflozin'),
                              _drop_med('empagliflozin'), 'No SGLT2 inhibitor in type 2 diabetes with CKD (eGFR 20 or higher).'),
}


UNSEEN_TYPES = ('C10_unsafe_hba1c_target', 'C12_double_missed_dose', 'O06_no_sick_day_rule', 'O09_no_sglt2i_ckd_t2d')
AGENT_RULES_WITHHELD = ('C06_pioglitazone_heart_failure', 'C08_nsaid_ckd')   # the agent's own rule layer omits these
HELD_OUT_TYPES = UNSEEN_TYPES + AGENT_RULES_WITHHELD                         # never shown in the training split


def applicable(patient: dict, plan: dict) -> list[str]:
    return [e for e, spec in CATALOGUE.items() if spec[3](patient, plan)]


def inject(patient: dict, plan: dict, error_id: str, variant: int = 0, second: bool = False) -> tuple[dict, dict]:
    kind, severity, rule_covered, ok, op, description = CATALOGUE[error_id]
    if not ok(patient, plan):
        raise ValueError(f'{error_id} is not applicable to {patient["patient_id"]}')
    new = copy.deepcopy(plan)
    entity = op(patient, new, variant) if error_id in STATEMENT_VARIANTS else op(patient, new)
    gold = {'error_id': error_id, 'kind': kind, 'severity': severity, 'rule_covered': rule_covered,
            'entity': entity, 'description': description}
    seen = bool(check_plan(patient, new['medications']))
    if not second and seen != rule_covered:   # (a second error is added to a plan that may already violate a rule)
        raise AssertionError(f'{error_id}: rule_covered={rule_covered} but the rule layer says {seen}')
    return new, gold


# ---------------------------------------------------------------- verification of the written prose

_SENT = re.compile(r'(?<=[.!?])\s+|\n+')
_CAUTION = re.compile(r'\b(avoid|contraindicat|not recommended|should not|do not (?:take|use|start)|caution|inappropriate|'
                      r'unsafe|discontinue|stop taking|must not|error|incorrect|too high|exceeds)', re.I)
ELEMENT_TERMS = {
    'k_creat_recheck': [r'potassium', r'\b(?:re)?check|monitor|repeat'],
    'hba1c_recheck': [r'hba1c|a1c', r'recheck|repeat|check|monitor|3 months|three months'],
    'urgent_bp': [r'blood pressure|\bbp\b|hypertensi', r'urgent|same[- ]day|emergency|immediate'],
    'urgent_k': [r'potassium|hyperkal', r'urgent|same[- ]day|emergency|immediate|ecg'],
    'sick_day': [r'sick[- ]day|dehydrat|vomiting|diarrh', r'hold|pause|stop|withhold|skip|temporarily'],
    'hypo_education': [r'hypoglyc|low blood (?:sugar|glucose)'],
    'followup': [r'follow[- ]?up|appointment|review', r'week|month'],
    'avoid_nsaid': [r'naproxen|anti-inflammator|nsaid', r'avoid|do not|should not'],
}
STATEMENT_TERMS = {
    'C10_unsafe_hba1c_target': [r'hba1c|a1c', r'(?<![\d.])5(?:\.0)?\s?%'],
    'C11_stop_bp_medicine_when_normal': [r'\bstop\b|discontinue', r'blood pressure (?:medicine|medication|tablet)', r'normal'],
    'C12_double_missed_dose': [r'doubl|two doses', r'miss'],
}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _sentence_with(text: str, groups: list[str]) -> str | None:
    for s in sentences(text):
        if all(re.search(g, s, re.I) for g in groups):
            return s
    return None


def _dose_pattern(drug: str, dose_mg) -> re.Pattern:
    num = f'{dose_mg:g}'
    num = num if len(num) < 4 else f'{num[:-3]},?{num[-3:]}'
    return re.compile(rf'{drug}\b[^.\n]{{0,60}}?(?<![\d.,]){num}\s?mg\b', re.I)


def stated_doses(text: str) -> list[tuple[str, float]]:
    """(drug, dose) for every lexicon drug followed by an mg dose - used to catch drugs the writer invented."""
    out = []
    for drug in DRUGS:
        for m in re.finditer(rf'{drug}\b[^.\n]{{0,60}}?(?<![\d.,])(\d[\d,]*(?:\.\d+)?)\s?mg\b', text, re.I):
            out.append((drug, float(m.group(1).replace(',', ''))))
    return out


def verify(plan: dict, gold: dict | list | None, text: str) -> dict:
    """Does the prose realise exactly this plan (and its error or errors)? Returns {'ok', 'problems': [...]}"""
    golds = [] if not gold else gold if isinstance(gold, list) else [gold]
    problems = []
    planned = {m['drug']: m for m in plan['medications']}
    for drug, m in planned.items():
        if m['dose_mg'] is None:
            if not re.search(rf'{drug}\b', text, re.I):
                problems.append(f'medication not mentioned: {drug}')
        elif not _dose_pattern(drug, m['dose_mg']).search(text):
            problems.append(f'medication/dose not found: {drug} {m["dose_mg"]:g} mg')
    for drug, dose in stated_doses(text):
        if drug not in planned:
            problems.append(f'unplanned drug with a dose: {drug} {dose:g} mg')
        elif planned[drug]['dose_mg'] is not None and dose != planned[drug]['dose_mg'] \
                and dose != planned[drug]['dose_mg'] * PER_DAY[planned[drug]['frequency']]:
            problems.append(f'different dose stated: {drug} {dose:g} mg')
    for key in ELEMENT_TERMS:
        present = _sentence_with(text, ELEMENT_TERMS[key]) is not None
        if key in plan['elements'] and not present:
            problems.append(f'required element missing: {key}')
        if key not in plan['elements'] and present and any(g['entity'] == key for g in golds):
            problems.append(f'omitted element was written anyway: {key}')
    for gold in golds:
        eid = gold['error_id']
        if gold['kind'] == 'commission' and eid in STATEMENT_TERMS:
            if _sentence_with(text, STATEMENT_TERMS[eid]) is None:
                problems.append('injected statement not found')
        elif gold['kind'] == 'commission':
            hit = [s for s in sentences(text) if re.search(rf'{gold["entity"]}\b', s, re.I)]
            if any(_CAUTION.search(s) for s in hit):
                problems.append('the writer flagged or corrected the injected medication')
        elif gold['entity'] in DRUGS:
            cls = DRUGS[gold['entity']][0]
            word = {'statin': r'statin', 'sglt2i': r'sglt-?2|gliflozin'}[cls]
            if re.search(rf'{gold["entity"]}|{word}', text, re.I):
                problems.append(f'omitted medication was written anyway: {gold["entity"]}')
    injected = {g['error_id'] for g in golds}
    for eid, groups in STATEMENT_TERMS.items():
        if eid not in injected and _sentence_with(text, groups):
            problems.append(f'unsafe statement that was not injected: {eid}')
    return {'ok': not problems, 'problems': problems}
