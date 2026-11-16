"""Step 4: the applicable criteria for one case, chosen by code from the extracted facts (no model routing).

Level 1 (general) items always apply. Level 2 templates become level 3 case items when their trigger holds; the case
item text carries the patient's own value. A trigger that needs a missing fact yields a cannot-determine finding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .rules import ACTIVE, LEXICON, MINOR, MODERATE, SEVERE

GENERAL = 'kb/v4/general-items.json'
DISEASE = 'kb/v4/disease-items.json'
FINDING_CATEGORIES = ('contraindicated_drug', 'dose_error', 'duplicate_therapy', 'drug_interaction',
                      'unsafe_instruction', 'unsafe_target', 'factual_error', 'internal_inconsistency',
                      'incomplete_instruction', 'missing_urgent_action', 'missing_monitoring', 'missing_treatment',
                      'missing_safety_advice', 'missing_follow_up', 'overclaiming', 'readability', 'other')
SEVERITIES = (SEVERE, MODERATE, MINOR)


def load_library(root: Path | str = '.') -> dict:
    root = Path(root)
    raw = {k: (root / p).read_bytes() for k, p in (('general', GENERAL), ('disease', DISEASE))}
    lib = {k: json.loads(v) for k, v in raw.items()}
    lib['sha256'] = {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}
    lib['templates'] = {t['id']: t for t in lib['disease']['templates']}
    return lib


def _active_classes(meds: list[dict]) -> set[str]:
    return {LEXICON[m['drug']][0] for m in meds if m['status'] in ACTIVE and m['drug'] in LEXICON}


def _cannot(template_id: str, missing: str) -> dict:
    return {'source': 'criteria', 'id': 'CANNOT_DETERMINE', 'category': 'cannot_determine', 'severity': MODERATE,
            'entity': template_id, 'issue': f'{template_id}: needed patient fact "{missing}" not found'}


def applicable(lib: dict, facts: dict, meds: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (items for the model to check, findings decided by code: code-checked templates and
    cannot-determine)."""
    items = [{'id': g['id'], 'level': 'general', 'text': g['text'], 'category': g['category'], 'entity': g['entity'],
              'severity_if_not_met': g['severity_if_not_met']} for g in lib['general']['items']]
    code_findings: list[dict] = []
    T = lib['templates']
    classes = _active_classes(meds)
    cond = set(facts.get('conditions') or [])

    def case_item(tid, prefix):
        t = T[tid]
        items.append({'id': tid, 'level': 'case', 'text': f'{prefix} {t["text"]}'.strip(), 'category': t['category'],
                      'entity': t['entity'], 'severity_if_not_met': t['severity_if_not_met']})
    sbp, dbp, k = facts.get('systolic'), facts.get('diastolic'), facts.get('potassium')
    if sbp is None or dbp is None:
        code_findings.append(_cannot('T_URGENT_BP', 'blood pressure'))
    elif sbp >= 180 or dbp >= 120:
        case_item('T_URGENT_BP', f'Blood pressure is {sbp:g}/{dbp:g} mmHg.')
    if k is None:
        code_findings.append(_cannot('T_URGENT_K', 'potassium'))
    elif k >= 6.0:
        case_item('T_URGENT_K', f'Potassium is {k:g} mmol/L.')
    if classes & {'acei', 'arb'} and k is not None and k < 6.0:
        case_item('T_K_CREAT', '')
    if 'type_2_diabetes' in cond:
        case_item('T_HBA1C', '')
        if 'ascvd' in cond and 'statin' not in classes:
            t = T['T_STATIN_ASCVD']
            code_findings.append({'source': 'criteria', 'id': 'T_STATIN_ASCVD', 'category': t['category'],
                                  'severity': t['severity_if_not_met'], 'entity': t['entity'],
                                  'issue': 'Type 2 diabetes with cardiovascular disease, but no statin is prescribed.'})
    if 'insulin' in classes:
        case_item('T_HYPO_EDU', '')
    case_item('T_FOLLOWUP', '')
    return items, code_findings
