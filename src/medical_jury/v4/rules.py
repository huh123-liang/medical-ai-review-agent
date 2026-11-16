"""The AGENT's deterministic rule layer (level 2, rules): a small engine over the declarative file kb/v4/rules.json.

Independent of testbed/rules.py on purpose: separate lexicon, separate code, and two rules deliberately absent
(TZD in heart failure, NSAID in CKD). Rules run only on medications the document starts or continues; a rule that
needs a patient fact that was not found returns a CANNOT_DETERMINE finding, which sends the case to a human.

The file carries, per rule, its guideline source and its clinical sign-off (all 'not_reviewed' today). Changing
anything that can change a finding needs a new "version", or tests_v2 fails on the pinned behaviour digest of the
frozen version; sources and sign-off are metadata outside that digest."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

# The repository's kb/ folder; MEDICAL_JURY_V4_RULES points elsewhere (e.g. an installed package without the repo).
RULES_FILE = Path(os.environ.get('MEDICAL_JURY_V4_RULES')
                  or Path(__file__).resolve().parents[3] / 'kb' / 'v4' / 'rules.json')
SEVERE, MODERATE, MINOR = 'severe', 'moderate', 'minor'
SEVERITIES = (SEVERE, MODERATE, MINOR)
KINDS = ('dose_over_max', 'threshold', 'flag', 'all_classes_present', 'duplicate_class')


class RuleFileError(ValueError):
    pass


METADATA = ('source', 'clinical_signoff', 'description')


def behaviour_digest(doc: dict) -> str:
    """SHA-256 of everything that can change a finding. Sources, clinical sign-off and descriptions are metadata:
    a clinician signing a rule off does not need a new version; changing a threshold does."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in METADATA and k not in ('sha256', 'file_sha256')}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(strip(doc), sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def load_rules(path: Path = RULES_FILE) -> dict:
    if not Path(path).is_file():
        raise RuleFileError(f'rule file not found: {path} (set MEDICAL_JURY_V4_RULES to its location)')
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    doc['file_sha256'] = hashlib.sha256(raw).hexdigest()
    doc['sha256'] = behaviour_digest(doc)
    ids, classes = set(), {e['class'] for e in doc['lexicon'].values()}
    for r in doc['rules']:
        if r['kind'] not in KINDS:
            raise RuleFileError(f"{r['id']}: unknown kind {r['kind']!r}")
        outputs = r.get('bands', [r])
        for o in outputs:
            if o['severity'] not in SEVERITIES:
                raise RuleFileError(f"{o['id']}: bad severity {o['severity']!r}")
            if o['id'] in ids:
                raise RuleFileError(f"duplicate rule id {o['id']}")
            ids.add(o['id'])
        for c in r.get('applies_to', {}).get('classes', []) + r.get('except_classes', []):
            if c not in classes:
                raise RuleFileError(f"{r['id']}: class {c!r} is not in the lexicon")
        for d in r.get('applies_to', {}).get('drugs', []):
            if d not in doc['lexicon']:
                raise RuleFileError(f"{r['id']}: drug {d!r} is not in the lexicon")
        if 'clinical_signoff' not in r or 'source' not in r:
            raise RuleFileError(f"{r['id']}: every rule needs source and clinical_signoff")
    return doc


RULES = load_rules()
RULES_VERSION = RULES['version']
RULES_SHA256 = RULES['sha256']
PER_DAY = RULES['per_day']
ACTIVE = tuple(RULES['active_statuses'])
# generic name -> (class, usual maximum total daily dose in mg or None = not dose-checked here)
LEXICON: dict[str, tuple[str, float | None]] = {d: (e['class'], e['max_daily_mg']) for d, e in RULES['lexicon'].items()}
RAAS = ('acei', 'arb')
RULE_CATEGORY = {RULES['cannot_determine']['id']: RULES['cannot_determine']['category'],
                 **{o['id']: o['category'] for r in RULES['rules'] for o in r.get('bands', [r])}}
SYNONYMS = RULES['synonyms']
_SALTS = tuple(' ' + s for s in RULES['salts'])


def normalise_drug(name: str) -> str:
    n = ' '.join(str(name).lower().replace('-', ' ').split())
    for s in _SALTS:
        if n.endswith(s):
            n = n[: -len(s)]
    return SYNONYMS.get(n, n)


def daily_mg(med: dict, per_day: dict = PER_DAY) -> float | None:
    if med.get('dose_mg') is None or med.get('frequency') not in per_day:
        return None
    return float(med['dose_mg']) * per_day[med['frequency']]


def _finding(rule_id, category, severity, drugs, note=''):
    return {'source': 'rule', 'id': rule_id, 'category': category, 'severity': severity,
            'entity': ', '.join(sorted(set(drugs))), 'drugs': sorted(set(drugs)), 'issue': note or rule_id}


def _band_holds(band: dict, value, first_med) -> bool:
    if 'below' in band and not value < band['below']:
        return False
    if 'at_least' in band and not value >= band['at_least']:
        return False
    if 'daily_mg_above' in band and not (daily_mg(first_med) or 0) > band['daily_mg_above']:
        return False
    return True


def check(facts: dict, medications: list[dict], rules: dict | None = None) -> list[dict]:
    """facts: extracted patient facts (value or None). medications: extracted, lexicon-known names only."""
    rules = rules or RULES
    lexicon, cd = rules['lexicon'], rules['cannot_determine']
    active = [m for m in medications if m.get('status') in rules['active_statuses'] and m['drug'] in lexicon]
    by_class: dict[str, list[str]] = {}
    for m in active:
        by_class.setdefault(lexicon[m['drug']]['class'], []).append(m['drug'])
    out = []
    for r in rules['rules']:
        kind, applies = r['kind'], r.get('applies_to', {})
        if 'drugs' in applies:
            meds = [m for m in active if m['drug'] in applies['drugs']]
            targets = [d for d in applies['drugs'] if any(m['drug'] == d for m in meds)]
        else:
            targets = [d for c in applies.get('classes', []) for d in by_class.get(c, [])]
            meds = [m for m in active if m['drug'] in targets]
        if kind == 'dose_over_max':
            for m in active:
                d, mx = daily_mg(m, rules['per_day']), lexicon[m['drug']]['max_daily_mg']
                if d is not None and mx is not None and d > mx:
                    hi = r['severity_if_ratio_at_least']
                    out.append(_finding(r['id'], r['category'], hi['severity'] if d >= hi['ratio'] * mx else r['severity'],
                                        [m['drug']], f'{d:g} mg/day > {mx:g}'))
        elif kind == 'threshold':
            if not targets:
                continue
            value = facts.get(r['fact'])
            if value is None:
                if r['requires_fact']:
                    out.append(_finding(cd['id'], cd['category'], cd['severity'], targets,
                                        f"{r['id']}: needed patient fact not found"))
                continue
            band = next((b for b in r['bands'] if _band_holds(b, value, meds[0] if meds else {})), None)
            if band is not None:
                out.append(_finding(band['id'], band['category'], band['severity'], targets))
        elif kind == 'flag':
            if targets and facts.get(r['fact']):
                out.append(_finding(r['id'], r['category'], r['severity'], targets))
        elif kind == 'all_classes_present':
            if all(by_class.get(c) for c in applies['classes']):
                out.append(_finding(r['id'], r['category'], r['severity'], targets))
        elif kind == 'duplicate_class':
            for cls, drugs in by_class.items():
                if len(set(drugs)) > 1 and cls not in r['except_classes']:
                    out.append(_finding(r['id'], r['category'], r['severity'], drugs, cls))
    order = {s: i for i, s in enumerate(rules['severity_order'])}
    return sorted(out, key=lambda f: (order[f['severity']], f['id']))
