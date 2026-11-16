"""Offline fixtures for plumbing tests and dry runs. NO clinical meaning.

``oracle_responder`` answers the extraction call from the testbed's structured truth (it cheats on purpose, so a
test can isolate the code steps); the check call is answered by keyword presence only."""
from __future__ import annotations

import json
import re

KEYWORDS = {'T_URGENT_BP': r'same-day|urgent', 'T_URGENT_K': r'urgent|same-day', 'T_K_CREAT': r'creatinine',
            'T_HBA1C': r'hba1c', 'T_HYPO_EDU': r'hypoglyc|low blood sugar', 'T_FOLLOWUP': r'follow-up|follow up'}


def _fact(v, quote):
    return {'value': v, 'quote': quote}


def oracle_extraction(case: dict) -> dict:
    p, text = case['patient'], case['patient_summary']

    def q(pattern):
        m = re.search(pattern, text)
        return m.group(0) if m else None
    meds = [{'drug': m['drug'], 'dose_mg': m['dose_mg'], 'frequency': m['frequency'], 'status': 'start',
             'quote': m['drug']} for m in case['plan']['medications']]
    if 'avoid_nsaid' in case['plan']['elements']:
        meds.append({'drug': 'naproxen', 'dose_mg': None, 'frequency': 'other', 'status': 'avoid', 'quote': 'naproxen'})
    return {'patient': {
        'egfr': _fact(p['egfr'], q(r'eGFR \d+')), 'potassium': _fact(p['potassium'], q(r'potassium [\d.]+')),
        'systolic': _fact(p['systolic'], q(r'\d+/\d+ mmHg')), 'diastolic': _fact(p['diastolic'], q(r'\d+/\d+ mmHg')),
        'hba1c': _fact(p['hba1c'], q(r'HbA1c [\d.]+')),
        'pregnant': _fact(p['pregnant'], q(r'pregnant, \d+ weeks')),
        'acei_angioedema': _fact(bool(p['allergies']), q(r'previous angioedema with an ACE inhibitor')),
        'conditions': _fact(list(p['conditions']), q(r'conditions: [^;]+'))},
        'medications': meds}


def keyword_check(payload: dict) -> dict:
    doc = payload['document'].lower()
    crit = []
    for c in payload['criteria']:
        pat = KEYWORDS.get(c['id'])
        ok = True if pat is None else bool(re.search(pat, doc))
        crit.append({'id': c['id'], 'status': 'met' if ok else 'not_met',
                     'reason': 'OFFLINE MOCK: keyword presence only.', 'quote': None})
    return {'criteria': crit, 'other_findings': []}


def oracle_responder(cases: dict[tuple[str, str], dict]):
    """cases: (case description, document) -> testbed case."""
    def respond(messages) -> str:
        try:
            payload = json.loads(messages[-1]['content'])
        except (TypeError, ValueError):             # an optimiser (reflection) prompt: propose nothing useful
            return json.dumps({'new_instruction': 'OFFLINE MOCK INSTRUCTION (no clinical meaning).'})
        if 'criteria' in payload:
            return json.dumps(keyword_check(payload))
        return json.dumps(oracle_extraction(cases[(payload['case_description'], payload['document'])]))
    return respond


def plain_document(case: dict) -> str:
    """A deterministic stand-in for the local-model write-up (tests only)."""
    from ..testbed.build import plan_text
    return plan_text(case['plan'])
