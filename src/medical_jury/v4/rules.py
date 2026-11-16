"""The AGENT's deterministic rule layer (level 2, rules). Independent of testbed/rules.py on purpose:
separate lexicon, separate code, and two rules deliberately absent (TZD in heart failure, NSAID in CKD).
Rules run only on medications the document starts or continues; a rule that needs a patient fact that was not
found returns a CANNOT_DETERMINE finding, which sends the case to a human."""
from __future__ import annotations

RULES_VERSION = 'agent-rules-v4-1'
SEVERE, MODERATE, MINOR = 'severe', 'moderate', 'minor'
PER_DAY = {'once daily': 1, 'twice daily': 2, 'three times daily': 3, 'four times daily': 4}
ACTIVE = ('start', 'continue')

# generic name -> (class, usual maximum total daily dose in mg or None = not dose-checked here)
LEXICON: dict[str, tuple[str, float | None]] = {
    'metformin': ('biguanide', 2550),
    'empagliflozin': ('sglt2i', 25), 'dapagliflozin': ('sglt2i', 10), 'canagliflozin': ('sglt2i', 300),
    'linagliptin': ('dpp4i', 5), 'sitagliptin': ('dpp4i', 100),
    'pioglitazone': ('tzd', 45),
    'glimepiride': ('sulfonylurea', 8), 'gliclazide': ('sulfonylurea', None), 'glipizide': ('sulfonylurea', None),
    'insulin glargine': ('insulin', None), 'insulin': ('insulin', None),
    'semaglutide': ('glp1ra', None), 'liraglutide': ('glp1ra', None),
    'lisinopril': ('acei', 80), 'ramipril': ('acei', 20), 'enalapril': ('acei', 40), 'perindopril': ('acei', None),
    'losartan': ('arb', 100), 'valsartan': ('arb', 320), 'irbesartan': ('arb', 300), 'telmisartan': ('arb', 80),
    'amlodipine': ('ccb', 10), 'nifedipine': ('ccb', None),
    'labetalol': ('beta_blocker', 2400), 'bisoprolol': ('beta_blocker', None), 'metoprolol': ('beta_blocker', None),
    'spironolactone': ('mra', 100), 'eplerenone': ('mra', None),
    'hydrochlorothiazide': ('thiazide', 50), 'chlorthalidone': ('thiazide', None), 'indapamide': ('thiazide', None),
    'furosemide': ('loop', None),
    'atorvastatin': ('statin', 80), 'simvastatin': ('statin', 40), 'rosuvastatin': ('statin', 40),
    'pravastatin': ('statin', 80),
    'ibuprofen': ('nsaid', 3200), 'naproxen': ('nsaid', 1250), 'diclofenac': ('nsaid', None),
    'aspirin': ('antiplatelet', None), 'clopidogrel': ('antiplatelet', None),
}
RAAS = ('acei', 'arb')
RULE_CATEGORY = {'METFORMIN_EGFR_LT30': 'contraindicated_drug', 'METFORMIN_EGFR_30_44_DOSE': 'dose_error',
                 'SGLT2I_EGFR_LT20': 'contraindicated_drug', 'RAAS_DUAL': 'duplicate_therapy',
                 'RAAS_PREGNANCY': 'contraindicated_drug', 'ACEI_ANGIOEDEMA': 'contraindicated_drug',
                 'MRA_HYPERKALAEMIA': 'contraindicated_drug', 'RAAS_HYPERKALAEMIA': 'drug_interaction',
                 'DOSE_OVER_MAX': 'dose_error', 'DUPLICATE_CLASS': 'duplicate_therapy',
                 'CANNOT_DETERMINE': 'cannot_determine'}
SYNONYMS = {'glargine': 'insulin glargine', 'lantus': 'insulin glargine', 'basal insulin': 'insulin glargine',
            'insulin glargine u100': 'insulin glargine', 'hctz': 'hydrochlorothiazide', 'acetylsalicylic acid': 'aspirin'}
_SALTS = (' hydrochloride', ' hcl', ' sodium', ' potassium', ' calcium', ' besylate', ' maleate', ' succinate',
          ' tartrate', ' propionyl')


def normalise_drug(name: str) -> str:
    n = ' '.join(str(name).lower().replace('-', ' ').split())
    for s in _SALTS:
        if n.endswith(s):
            n = n[: -len(s)]
    return SYNONYMS.get(n, n)


def daily_mg(med: dict) -> float | None:
    if med.get('dose_mg') is None or med.get('frequency') not in PER_DAY:
        return None
    return float(med['dose_mg']) * PER_DAY[med['frequency']]


def _finding(rule, severity, drugs, note=''):
    return {'source': 'rule', 'id': rule, 'category': RULE_CATEGORY[rule], 'severity': severity,
            'entity': ', '.join(sorted(set(drugs))), 'drugs': sorted(set(drugs)), 'issue': note or rule}


def check(facts: dict, medications: list[dict]) -> list[dict]:
    """facts: extracted patient facts (value or None). medications: extracted, lexicon-known names only."""
    active = [m for m in medications if m.get('status') in ACTIVE and m['drug'] in LEXICON]
    out, by_class = [], {}
    for m in active:
        cls, max_daily = LEXICON[m['drug']]
        by_class.setdefault(cls, []).append(m['drug'])
        d = daily_mg(m)
        if d is not None and max_daily is not None and d > max_daily:
            out.append(_finding('DOSE_OVER_MAX', SEVERE if d >= 2 * max_daily else MODERATE, [m['drug']],
                                f'{d:g} mg/day > {max_daily:g}'))
    egfr, k, pregnant = facts.get('egfr'), facts.get('potassium'), facts.get('pregnant')
    raas = by_class.get('acei', []) + by_class.get('arb', [])

    def need(value, rule, drugs):
        if value is None:
            out.append(_finding('CANNOT_DETERMINE', MODERATE, drugs, f'{rule}: needed patient fact not found'))
            return False
        return True
    met = [m for m in active if m['drug'] == 'metformin']
    if met and need(egfr, 'METFORMIN_EGFR', ['metformin']):
        if egfr < 30:
            out.append(_finding('METFORMIN_EGFR_LT30', SEVERE, ['metformin']))
        elif egfr < 45 and (daily_mg(met[0]) or 0) > 1000:
            out.append(_finding('METFORMIN_EGFR_30_44_DOSE', MODERATE, ['metformin']))
    if by_class.get('sglt2i') and need(egfr, 'SGLT2I_EGFR', by_class['sglt2i']) and egfr < 20:
        out.append(_finding('SGLT2I_EGFR_LT20', MODERATE, by_class['sglt2i']))
    if by_class.get('acei') and by_class.get('arb'):
        out.append(_finding('RAAS_DUAL', MODERATE, raas))
    if raas and pregnant:
        out.append(_finding('RAAS_PREGNANCY', SEVERE, raas))
    if by_class.get('acei') and facts.get('acei_angioedema'):
        out.append(_finding('ACEI_ANGIOEDEMA', SEVERE, by_class['acei']))
    if by_class.get('mra') and need(k, 'MRA_POTASSIUM', by_class['mra']) and k >= 5.5:
        out.append(_finding('MRA_HYPERKALAEMIA', SEVERE, by_class['mra']))
    if raas and need(k, 'RAAS_POTASSIUM', raas) and k >= 5.5:
        out.append(_finding('RAAS_HYPERKALAEMIA', MODERATE, raas))
    for cls, drugs in by_class.items():
        if len(set(drugs)) > 1 and cls not in ('insulin',):
            out.append(_finding('DUPLICATE_CLASS', MODERATE, drugs, cls))
    order = {SEVERE: 0, MODERATE: 1, MINOR: 2}
    return sorted(out, key=lambda f: (order[f['severity']], f['id']))
