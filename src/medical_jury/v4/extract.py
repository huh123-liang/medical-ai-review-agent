"""Steps 0-2: one model call extracts patient facts (from the case description) and the recommendations (from the
document), each with a verbatim quote; code then checks the extraction against the source text.

A failed extraction check never silently passes a case: it is reported and sends the case to a human."""
from __future__ import annotations

import json
import re

from .rules import ACTIVE, LEXICON, normalise_drug

EXTRACT_PROMPT_VERSION = 'v4-extract-1'
STATUSES = ('start', 'continue', 'stop', 'avoid')
FREQUENCIES = ('once daily', 'twice daily', 'three times daily', 'four times daily', 'other')
CONDITIONS = ('type_2_diabetes', 'hypertension', 'chronic_kidney_disease', 'heart_failure', 'ascvd')
NUMERIC_FACTS = ('egfr', 'potassium', 'systolic', 'diastolic', 'hba1c')
BOOLEAN_FACTS = ('pregnant', 'acei_angioedema')

SYSTEM_EXTRACT = (
    'You extract information for a medication-safety check. You do not judge whether the advice is good.\n'
    'Return one JSON object with keys "patient" and "medications" and nothing else.\n'
    'PATIENT: take every fact from the CASE DESCRIPTION only. For each of egfr (mL/min/1.73m2), potassium (mmol/L), '
    'systolic and diastolic (mmHg), hba1c (%): {"value": number or null, "quote": exact words copied from the case '
    'description that contain the number, or null}. For pregnant and acei_angioedema (a past angioedema reaction to '
    'an ACE inhibitor): {"value": true or false, "quote": exact words or null}. For conditions: {"value": a list using '
    'only these names: ' + ', '.join(CONDITIONS) + ' ("ascvd" means established cardiovascular disease such as previous '
    'heart attack, stroke or peripheral arterial disease), "quote": exact words}. Use null when a number is not stated; '
    'never guess.\n'
    'MEDICATIONS: from the DOCUMENT, list every medicine the document names, one entry per medicine, including medicines '
    'it says to stop or to avoid. Each entry: {"drug": generic name in lower case, "dose_mg": number per dose in mg or '
    'null if no mg dose is given, "frequency": one of ' + ', '.join(f'"{f}"' for f in FREQUENCIES) + ', "status": one of '
    '"start" (new or prescribed now), "continue" (already taking and to keep taking), "stop" (to stop now), "avoid" '
    '(the document says not to take it), "quote": the exact words from the document that name the medicine}. '
    'A medicine mentioned only as something to avoid has status "avoid", never "start". Copy quotes character for '
    'character; do not paraphrase.')


def extract_messages(case_description: str, document: str, system: str = SYSTEM_EXTRACT) -> list[dict]:
    return [{'role': 'system', 'content': system},
            {'role': 'user', 'content': json.dumps({'case_description': case_description, 'document': document},
                                                   ensure_ascii=False)}]


class ExtractFailure(ValueError):
    def __init__(self, code: str, detail: str = ''):
        super().__init__(f'{code}: {detail}')
        self.code, self.detail = code, detail


def _num(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = re.search(r'-?\d+(?:\.\d+)?', str(v).replace(',', ''))
    return float(m.group()) if m else None


def parse_extraction(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise ExtractFailure('EXTRACT_NOT_JSON', str(e)[:200])
    if not isinstance(obj, dict) or not isinstance(obj.get('patient'), dict) or not isinstance(obj.get('medications'), list):
        raise ExtractFailure('EXTRACT_SCHEMA', 'need {"patient": {...}, "medications": [...]}')
    pat, facts, quotes = obj['patient'], {}, {}
    for k in NUMERIC_FACTS:
        item = pat.get(k) if isinstance(pat.get(k), dict) else {'value': pat.get(k), 'quote': None}
        facts[k], quotes[k] = _num(item.get('value')), item.get('quote')
    for k in BOOLEAN_FACTS:
        item = pat.get(k) if isinstance(pat.get(k), dict) else {'value': pat.get(k), 'quote': None}
        facts[k], quotes[k] = item.get('value') is True, item.get('quote')
    item = pat.get('conditions') if isinstance(pat.get('conditions'), dict) else {'value': pat.get('conditions'), 'quote': None}
    conds = item.get('value') if isinstance(item.get('value'), list) else []
    facts['conditions'] = sorted({c for c in conds if c in CONDITIONS})
    quotes['conditions'] = item.get('quote')
    meds = []
    for m in obj['medications']:
        if not isinstance(m, dict) or not m.get('drug'):
            raise ExtractFailure('EXTRACT_SCHEMA', f'bad medication entry {str(m)[:80]}')
        status = str(m.get('status', '')).lower()
        if status not in STATUSES:
            raise ExtractFailure('EXTRACT_SCHEMA', f'bad status {status!r}')
        freq = str(m.get('frequency') or 'other').lower()
        meds.append({'drug': normalise_drug(m['drug']), 'dose_mg': _num(m.get('dose_mg')),
                     'frequency': freq if freq in FREQUENCIES else 'other', 'status': status,
                     'quote': m.get('quote')})
    return {'facts': facts, 'fact_quotes': quotes, 'medications': meds}


def _norm(text: str) -> str:
    return ' '.join(str(text).lower().replace('’', "'").replace('–', '-').split())


def quote_found(quote, source: str) -> bool:
    return bool(quote) and _norm(quote) in _norm(source)


def mentioned_drugs(document: str) -> set[str]:
    """Lexicon names that appear in the document (word-bounded). 'insulin' alone is reported as 'insulin'."""
    low = _norm(document)
    # 'insulin sensitivity / resistance / levels ...' talks about the hormone, not the medicine
    low = re.sub(r'\binsulin (sensitivity|resistance|levels?|production|secretion)\b', r'hormone \1', low)
    found = {d for d in LEXICON if re.search(rf'(?<![a-z]){re.escape(d)}(?![a-z])', low)}
    if 'insulin glargine' in found:
        found.discard('insulin')
    return found


def doses_near(document: str, drug: str, window: int = 120) -> set[float]:
    """mg amounts written within ``window`` characters after a mention of the drug (g converted to mg)."""
    low, out = _norm(document), set()
    for m in re.finditer(rf'(?<![a-z]){re.escape(drug)}(?![a-z])', low):
        seg = low[m.end(): m.end() + window]
        for n, unit in re.findall(r'(\d[\d,]*(?:\.\d+)?)\s*(mg|g|milligrams?|grams?)\b', seg):
            v = float(n.replace(',', ''))
            out.add(v * 1000 if unit in ('g', 'gram', 'grams') else v)
    return out


def check_extraction(ex: dict, case_description: str, document: str) -> list[dict]:
    """Code checks of step 2. Returns problems; an empty list means the extraction passed."""
    problems = []
    for k in NUMERIC_FACTS:
        v, q = ex['facts'][k], ex['fact_quotes'][k]
        if v is None:
            continue
        if not quote_found(q, case_description):
            problems.append({'check': 'FACT_QUOTE_NOT_FOUND', 'item': k})
        elif not any(abs(float(x.replace(',', '')) - v) < 1e-9 for x in re.findall(r'\d[\d,]*(?:\.\d+)?', str(q))):
            problems.append({'check': 'FACT_VALUE_NOT_IN_QUOTE', 'item': k})
    for k in BOOLEAN_FACTS:
        if ex['facts'][k] and not quote_found(ex['fact_quotes'][k], case_description):
            problems.append({'check': 'FACT_QUOTE_NOT_FOUND', 'item': k})
    for m in ex['medications']:
        if not quote_found(m['quote'], document):
            problems.append({'check': 'MED_QUOTE_NOT_FOUND', 'item': m['drug']})
    names = {m['drug'] for m in ex['medications']}
    for d in sorted(mentioned_drugs(document)):
        if d not in names and not (d == 'insulin' and any('insulin' in n for n in names)):
            problems.append({'check': 'MENTIONED_DRUG_NOT_EXTRACTED', 'item': d})
    for m in ex['medications']:
        if m['status'] in ACTIVE and m['dose_mg'] is not None and m['drug'] in LEXICON:
            near = doses_near(document, m['drug'])
            if near and m['dose_mg'] not in near:
                problems.append({'check': 'DOSE_NOT_IN_TEXT', 'item': m['drug'],
                                 'detail': f"{m['dose_mg']:g} vs {sorted(near)}"})
    return problems


def unknown_active_drugs(ex: dict) -> list[str]:
    return sorted({m['drug'] for m in ex['medications'] if m['status'] in ACTIVE and m['drug'] not in LEXICON})
