"""Steps 5-6: one model call checks each applicable criterion and then reviews the document openly for anything else
that could cause harm. With an empty criteria list it is the 'model only' comparison arm."""
from __future__ import annotations

import json

from .criteria import FINDING_CATEGORIES, SEVERITIES

CHECK_PROMPT_VERSION = 'v4-check-3'
CRITERION_STATUSES = ('met', 'not_met', 'cannot_determine')
MAX_OTHER_FINDINGS = 5

SYSTEM_CHECK = (
    'You review chronic-disease management advice (type 2 diabetes, hypertension, chronic kidney disease) for ERRORS '
    'that could harm the patient. You are a screening aid; a clinician reviews anything you flag, so every false alarm '
    'costs clinician time. Report errors, not possible improvements.\n'
    'Input: the CASE DESCRIPTION, the DOCUMENT to review, a list of CRITERIA (it may be empty) and, when present, '
    'CODE_CHECKS.\n'
    'CODE_CHECKS: when present, it lists which medicine-against-value checks code has already run for this patient '
    '("checked") and what code flagged ("flagged"; empty means nothing). Do not report those checks again and do not '
    'contradict them. Anything not listed there has not been checked by code and is for you to find.\n'
    'Part 1. For every criterion, decide whether the DOCUMENT meets it: "met", "not_met", or "cannot_determine" (only '
    'when the document is ambiguous on exactly this point). A criterion that asks for something to be planned or '
    'arranged is met only if the document explicitly says so; a related target, value or general remark does not '
    'count. A medicine the document says to avoid is not prescribed. For "met" give nothing else. For "not_met" and '
    '"cannot_determine" give a reason of at most 25 words and, when possible, a quote copied exactly from the document.\n'
    'Part 2. Then read the whole document again and list at most ' + str(MAX_OTHER_FINDINGS) + ' OTHER clear errors '
    'that no criterion covers: a medicine that is wrong or contraindicated for this patient, a wrong dose, an unsafe '
    'instruction, an incorrect medical statement, or missing treatment or advice without which this patient could come '
    'to harm. Before reporting a contraindication or dose problem, compare the threshold with THIS patient\'s values in '
    'the case description; if the patient does not meet it, do not report it. Do NOT report: routine side-effect '
    'counselling, requests for more detail, a target or choice that is reasonable but not your preference, style, or '
    'any topic a criterion already covers. An empty list is the expected answer for a sound document.\n'
    'Severity: "severe" = following the document as written could plausibly cause serious harm within days to weeks '
    '(a medicine contraindicated at this patient\'s values, a dose clearly above the usual maximum, an instruction '
    'that is clearly dangerous); "moderate" = could cause harm or leaves out something this patient needs; "minor" = '
    'unlikely to cause harm. Do not use "severe" for anything debatable.\n'
    'Return one JSON object and nothing else: {"criteria": [{"id": "...", "status": "...", "reason": "..." (not for '
    'met), "quote": "..." or null (not for met)}], "other_findings": [{"category": one of ' + ', '.join(f'"{c}"' for c in FINDING_CATEGORIES)
    + ', "severity": "severe"|"moderate"|"minor", "entity": the medicine or topic concerned, "issue": at most 25 '
    'words, "quote": exact words from the document or null}]}.')


def check_messages(case_description: str, document: str, items: list[dict], system: str = SYSTEM_CHECK,
                   code_checks: dict | None = None) -> list[dict]:
    crit = [{'id': it['id'], 'criterion': it['text']} for it in items]
    payload = {'case_description': case_description, 'document': document, 'criteria': crit}
    if code_checks is not None:
        payload['code_checks'] = code_checks
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]


def code_checks_of(rule_findings: list[dict], medications: list[dict]) -> dict:
    """What the agent's rule layer checked and flagged, built by code from rules.py (no clinical text of its own).
    Only medicines the document starts or continues are listed, so nothing beyond the document is revealed."""
    from .rules import ACTIVE, LEXICON
    checked = []
    for m in medications:
        if m['status'] not in ACTIVE or m['drug'] not in LEXICON:
            continue
        cls, max_daily = LEXICON[m['drug']]
        against = ['potassium', 'same-class duplicate']
        if m['drug'] == 'metformin' or cls == 'sglt2i':
            against.append('eGFR')
        if max_daily is not None:
            against.append('usual maximum daily dose')
        if cls in ('acei', 'arb'):
            against += ['pregnancy', 'ACE inhibitor + ARB together']
        if cls == 'acei':
            against.append('previous angioedema')
        checked.append({'medicine': m['drug'], 'against': sorted(set(against))})
    flagged = [{'rule': f['id'], 'medicines': f.get('drugs', []), 'note': f.get('issue', '')}
               for f in rule_findings if f.get('source') == 'rule' and f['category'] != 'cannot_determine']
    return {'checked': checked, 'flagged': flagged}


class CheckFailure(ValueError):
    def __init__(self, code: str, detail: str = ''):
        super().__init__(f'{code}: {detail}')
        self.code, self.detail = code, detail


def parse_check(raw: str, items: list[dict]) -> dict:
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise CheckFailure('CHECK_NOT_JSON', str(e)[:200])
    if not isinstance(obj, dict) or not isinstance(obj.get('criteria', []), list) \
            or not isinstance(obj.get('other_findings', []), list):
        raise CheckFailure('CHECK_SCHEMA', 'need {"criteria": [...], "other_findings": [...]}')
    want = {it['id'] for it in items}
    got = {}
    for c in obj.get('criteria', []):
        if not isinstance(c, dict) or c.get('id') not in want:
            continue                                   # unknown ids are ignored, never invented into findings
        status = str(c.get('status', '')).lower()
        if status not in CRITERION_STATUSES:
            raise CheckFailure('CHECK_SCHEMA', f'bad status {status!r} for {c.get("id")}')
        got[c['id']] = {'status': status, 'reason': str(c.get('reason') or ''), 'quote': c.get('quote')}
    missing = sorted(want - set(got))
    if missing:
        raise CheckFailure('CHECK_MISSING_CRITERIA', ','.join(missing))
    other = []
    for f in obj.get('other_findings', [])[:MAX_OTHER_FINDINGS]:
        if not isinstance(f, dict):
            continue
        cat = str(f.get('category', 'other')).lower()
        sev = str(f.get('severity', '')).lower()
        other.append({'source': 'open_review', 'id': 'OPEN', 'category': cat if cat in FINDING_CATEGORIES else 'other',
                      'severity': sev if sev in SEVERITIES else 'moderate', 'entity': str(f.get('entity') or ''),
                      'issue': str(f.get('issue') or ''), 'quote': f.get('quote')})
    return {'criteria': got, 'other_findings': other}
