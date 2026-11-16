"""Deterministic medication rules over a structured plan. Pure functions.

Used (a) to prove every reference plan is free of the catalogued errors and (b) later as the review agent's
deterministic layer. Deliberately small: each rule cites a widely used source; none was reviewed by a physician.
"""
from __future__ import annotations

RULES_VERSION = 'testbed-rules-1'
SEVERE, MODERATE, MINOR = 'severe', 'moderate', 'minor'
PER_DAY = {'once daily': 1, 'twice daily': 2, 'three times daily': 3}

# drug -> (class, usual maximum total daily dose in mg; None = not dosed in mg here)
DRUGS: dict[str, tuple[str, float | None]] = {
    'metformin': ('biguanide', 2550), 'empagliflozin': ('sglt2i', 25), 'dapagliflozin': ('sglt2i', 10),
    'linagliptin': ('dpp4i', 5), 'sitagliptin': ('dpp4i', 100), 'pioglitazone': ('tzd', 45),
    'glimepiride': ('sulfonylurea', 8), 'insulin glargine': ('insulin', None),
    'lisinopril': ('acei', 80), 'ramipril': ('acei', 20), 'losartan': ('arb', 100), 'valsartan': ('arb', 320),
    'amlodipine': ('ccb', 10), 'labetalol': ('beta_blocker', 2400), 'spironolactone': ('mra', 100),
    'hydrochlorothiazide': ('thiazide', 50), 'atorvastatin': ('statin', 80), 'simvastatin': ('statin', 40),
    'ibuprofen': ('nsaid', 3200), 'naproxen': ('nsaid', 1250),
}
SOURCES = {
    'METFORMIN_EGFR_LT30': 'Metformin US prescribing information (contraindicated eGFR < 30); KDIGO 2022 Diabetes in CKD',
    'METFORMIN_EGFR_30_44_DOSE': 'KDIGO 2022 Diabetes in CKD (reduce dose when eGFR 30-44)',
    'RAAS_DUAL': 'KDIGO 2021 Blood Pressure in CKD; ONTARGET (avoid ACE inhibitor + ARB combination)',
    'RAAS_PREGNANCY': 'ACE inhibitor / ARB prescribing information (fetal toxicity); ACOG',
    'DOSE_OVER_MAX': 'Usual maximum daily dose from prescribing information',
    'ACEI_ANGIOEDEMA': 'ACE inhibitor prescribing information (history of angioedema)',
    'TZD_HEART_FAILURE': 'Pioglitazone boxed warning (NYHA class III-IV heart failure)',
    'MRA_HYPERKALAEMIA': 'Spironolactone prescribing information (hyperkalaemia)',
    'RAAS_START_HYPERKALAEMIA': 'KDIGO 2021 Blood Pressure in CKD (potassium before starting RAAS inhibitors)',
    'NSAID_CKD': 'KDIGO 2012/2024 CKD (avoid NSAIDs when eGFR < 30; caution < 60)',
    'DUPLICATE_CLASS': 'Therapeutic duplication (two drugs of one class)',
    'SGLT2I_EGFR_LT20': 'KDIGO 2022 Diabetes in CKD (initiate when eGFR >= 20)',
}


def daily_mg(med: dict) -> float | None:
    if med.get('dose_mg') is None:
        return None
    return med['dose_mg'] * PER_DAY[med['frequency']]


def check_plan(patient: dict, medications: list[dict]) -> list[dict]:
    """Return rule violations [{'rule','severity','drugs'}] for a medication list, most severe first."""
    out = []
    by_class: dict[str, list[str]] = {}
    for m in medications:
        cls, max_daily = DRUGS[m['drug']]
        by_class.setdefault(cls, []).append(m['drug'])
        d = daily_mg(m)
        if d is not None and max_daily is not None and d > max_daily:
            out.append({'rule': 'DOSE_OVER_MAX', 'severity': SEVERE if d >= 2 * max_daily else MODERATE,
                        'drugs': [m['drug']]})
    egfr, k = patient['egfr'], patient['potassium']
    met = [m for m in medications if m['drug'] == 'metformin']
    if met and egfr < 30:
        out.append({'rule': 'METFORMIN_EGFR_LT30', 'severity': SEVERE, 'drugs': ['metformin']})
    elif met and egfr < 45 and (daily_mg(met[0]) or 0) > 1000:
        out.append({'rule': 'METFORMIN_EGFR_30_44_DOSE', 'severity': MODERATE, 'drugs': ['metformin']})
    raas = by_class.get('acei', []) + by_class.get('arb', [])
    if by_class.get('acei') and by_class.get('arb'):
        out.append({'rule': 'RAAS_DUAL', 'severity': MODERATE, 'drugs': raas})
    if patient.get('pregnant') and raas:
        out.append({'rule': 'RAAS_PREGNANCY', 'severity': SEVERE, 'drugs': raas})
    if 'acei_angioedema' in patient.get('allergies', []) and by_class.get('acei'):
        out.append({'rule': 'ACEI_ANGIOEDEMA', 'severity': SEVERE, 'drugs': by_class['acei']})
    if 'heart_failure' in patient['conditions'] and by_class.get('tzd'):
        out.append({'rule': 'TZD_HEART_FAILURE', 'severity': SEVERE, 'drugs': by_class['tzd']})
    if k >= 5.5 and by_class.get('mra'):
        out.append({'rule': 'MRA_HYPERKALAEMIA', 'severity': SEVERE, 'drugs': by_class['mra']})
    if k >= 5.5 and raas:
        out.append({'rule': 'RAAS_START_HYPERKALAEMIA', 'severity': MODERATE, 'drugs': raas})
    if by_class.get('nsaid') and egfr < 60:
        out.append({'rule': 'NSAID_CKD', 'severity': SEVERE if egfr < 30 else MODERATE, 'drugs': by_class['nsaid']})
    if by_class.get('sglt2i') and egfr < 20:
        out.append({'rule': 'SGLT2I_EGFR_LT20', 'severity': MODERATE, 'drugs': by_class['sglt2i']})
    for cls, drugs in by_class.items():
        if len(drugs) > 1:
            out.append({'rule': 'DUPLICATE_CLASS', 'severity': MODERATE, 'drugs': drugs})
    order = {SEVERE: 0, MODERATE: 1, MINOR: 2}
    return sorted(out, key=lambda v: (order[v['severity']], v['rule']))
