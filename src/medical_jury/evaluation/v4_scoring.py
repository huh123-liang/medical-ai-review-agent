"""Scoring for the v4 error-detection agent on the known-answer testbed (PROTOCOL-V3-TESTBED part 2).

Matching is deterministic: a gold error counts as found when some finding has a category in the error's accepted set
AND its entity/issue text names the error's medicine or topic. No model judges the match. Pure functions, no I/O."""
from __future__ import annotations

import math
import random
import re
from collections import defaultdict

# error id -> (accepted finding categories, entity pattern searched in the finding's entity + issue text)
MATCH = {
    'C01_metformin_egfr_lt30': ({'contraindicated_drug', 'dose_error'}, r'metformin'),
    'C02_acei_plus_arb': ({'duplicate_therapy', 'contraindicated_drug', 'drug_interaction'}, r'losartan|lisinopril|arb|dual'),
    'C03_acei_in_pregnancy': ({'contraindicated_drug'}, r'lisinopril|ace inhibitor|acei'),
    'C04_tenfold_dose': ({'dose_error'}, None),        # entity = the drug whose dose was multiplied
    'C05_acei_despite_angioedema': ({'contraindicated_drug'}, r'lisinopril|ace inhibitor|acei|angioedema'),
    'C06_pioglitazone_heart_failure': ({'contraindicated_drug'}, r'pioglitazone'),
    'C07_spironolactone_hyperkalaemia': ({'contraindicated_drug', 'drug_interaction'}, r'spironolactone'),
    'C08_nsaid_ckd': ({'contraindicated_drug'}, r'ibuprofen|nsaid|anti-inflammatory'),
    'C09_duplicate_statin': ({'duplicate_therapy'}, r'simvastatin|atorvastatin|statin'),
    'C10_unsafe_hba1c_target': ({'unsafe_target', 'factual_error', 'unsafe_instruction'}, r'hba1c|target|glyc'),
    'C11_stop_bp_medicine_when_normal': ({'unsafe_instruction', 'factual_error'}, r'stop|blood pressure|antihypertensive'),
    'C12_double_missed_dose': ({'unsafe_instruction', 'factual_error', 'dose_error'}, r'doubl|missed|two doses'),
    'O01_no_potassium_creatinine_recheck': ({'missing_monitoring'}, r'potassium|creatinine|kidney function|renal'),
    'O02_no_urgent_bp_action': ({'missing_urgent_action'}, r'blood pressure|\bbp\b|hypertensi'),
    'O03_no_urgent_potassium_action': ({'missing_urgent_action'}, r'potassium|hyperkal'),
    'O04_no_hba1c_recheck': ({'missing_monitoring'}, r'hba1c'),
    'O05_no_statin_ascvd': ({'missing_treatment'}, r'statin'),
    'O06_no_sick_day_rule': ({'missing_safety_advice'}, r'sick|dehydrat|vomit|illness'),
    'O07_no_hypoglycaemia_education': ({'missing_safety_advice'}, r'hypoglyc|low blood sugar'),
    'O08_no_follow_up': ({'missing_follow_up'}, r'follow|review|appointment'),
    'O09_no_sglt2i_ckd_t2d': ({'missing_treatment'}, r'sglt2|empagliflozin|flozin'),
}


def golds(case: dict) -> list[dict]:
    g = case.get('gold')
    return [] if not g else g if isinstance(g, list) else [g]


def matches(gold: dict, finding: dict) -> bool:
    cats, pattern = MATCH[gold['error_id']]
    if finding.get('category') not in cats:
        return False
    text = f"{finding.get('entity', '')} {finding.get('issue', '')}".lower()
    pattern = pattern or re.escape(str(gold['entity']).lower())
    return re.search(pattern, text) is not None


def found(gold: dict, record: dict) -> bool:
    return any(matches(gold, f) for f in record.get('findings', []))


def wilson(k: int, n: int, z: float = 1.959964) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def rate(k: int, n: int) -> dict:
    return {'k': k, 'n': n, 'rate': round(k / n, 4) if n else None, 'ci95': wilson(k, n)}


def extraction_exact(case: dict, record: dict) -> dict | None:
    """Exact agreement of the extracted active medication list and key facts with the testbed truth."""
    ex = record.get('extraction')
    if ex is None:
        return None
    truth = sorted((m['drug'], m['dose_mg'], m['frequency']) for m in case['plan']['medications'])
    got = sorted((m['drug'], m['dose_mg'], m['frequency']) for m in ex['medications'] if m['status'] in ('start', 'continue'))
    p, f = case['patient'], ex['facts']
    facts_ok = (f['egfr'] == p['egfr'] and f['potassium'] == p['potassium'] and f['systolic'] == p['systolic']
                and f['diastolic'] == p['diastolic'] and f['pregnant'] == p['pregnant']
                and f['acei_angioedema'] == bool(p['allergies']) and set(f['conditions']) == set(p['conditions']))
    return {'medications': truth == got, 'facts': facts_ok}


def score_arm(cases: list[dict], records: dict[str, dict], unseen: tuple[str, ...]) -> dict:
    """cases: testbed cases (one style); records: case_id -> arm record. Missing records count as not found and, for
    clean cases, are excluded from the false-positive denominator (reported as missing)."""
    sev_seen = [0, 0]
    unseen_ct = [0, 0]
    clean_fp, clean_n, clean_any = 0, 0, 0
    by_type = defaultdict(lambda: [0, 0])
    by_vis = defaultdict(lambda: [0, 0])
    sev_escalated = [0, 0]
    double = [0, 0]
    escal = defaultdict(int)
    ext = {'medications': [0, 0], 'facts': [0, 0]}
    missing = 0
    for c in cases:
        r = records.get(c['case_id'])
        if r is None:
            missing += 1
        gs = golds(c)
        if r is not None:
            escal[r['triage']['decision']] += 1
            e = extraction_exact(c, r)
            if e is not None:
                for k in ext:
                    ext[k][0] += e[k]
                    ext[k][1] += 1
        if not gs:
            if r is not None:
                clean_n += 1
                clean_fp += any(f['severity'] == 'severe' for f in r['findings'])
                clean_any += bool(r['findings'])
            continue
        hits = [r is not None and found(g, r) for g in gs]
        if len(gs) == 2:
            double[0] += all(hits)
            double[1] += 1
        for g, h in zip(gs, hits):
            by_type[g['error_id']][0] += h
            by_type[g['error_id']][1] += 1
            if g['error_id'] in unseen:
                unseen_ct[0] += h
                unseen_ct[1] += 1
                continue
            by_vis['rule_visible' if g['rule_covered'] else 'reading_only'][0] += h
            by_vis['rule_visible' if g['rule_covered'] else 'reading_only'][1] += 1
            if g['severity'] == 'severe' and len(gs) == 1:
                sev_seen[0] += h
                sev_seen[1] += 1
                sev_escalated[0] += bool(h and r['triage']['decision'] == 'escalate')
                sev_escalated[1] += 1
    n_rec = sum(escal.values())
    return {
        'primary': {'severe_detection_seen_types': rate(*sev_seen),
                    'clean_severe_false_positive': rate(clean_fp, clean_n)},
        'secondary': {'unseen_type_detection': rate(*unseen_ct),
                      'clean_any_finding': rate(clean_any, clean_n),
                      'double_error_both_found': rate(*double),
                      'severe_found_and_escalated': rate(*sev_escalated),
                      'by_visibility': {k: rate(*v) for k, v in sorted(by_vis.items())},
                      'by_error_type': {k: rate(*v) for k, v in sorted(by_type.items())},
                      'triage': {k: rate(v, n_rec) for k, v in sorted(escal.items())},
                      'extraction_exact': {k: rate(*v) for k, v in ext.items()}},
        'missing_records': missing,
    }


def accept(primary: dict, min_detection: float = 0.90, max_fp: float = 0.10) -> dict:
    d, f = primary['severe_detection_seen_types'], primary['clean_severe_false_positive']
    ok = d['rate'] is not None and f['rate'] is not None and d['rate'] >= min_detection and f['rate'] <= max_fp
    return {'meets_thresholds': ok, 'rule': f'severe detection >= {min_detection} and clean severe FP <= {max_fp} '
                                            '(point estimates; CIs reported alongside)'}


def paired_diff(cases: list[dict], a: dict[str, dict], b: dict[str, dict], unseen: tuple[str, ...], seed: int,
                reps: int = 4000) -> dict:
    """Paired difference in severe-detection (seen types, single-error cases), b minus a, bootstrap over cases."""
    rows = [(found(g, a[c['case_id']]), found(g, b[c['case_id']])) for c in cases
            for g in golds(c) if len(golds(c)) == 1 and g['severity'] == 'severe' and g['error_id'] not in unseen
            and c['case_id'] in a and c['case_id'] in b]
    if not rows:
        return {'n': 0}
    rng = random.Random(seed)
    n = len(rows)
    d = [int(y) - int(x) for x, y in rows]
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return {'n': n, 'difference': round(sum(d) / n, 4),
            'ci95': [round(stats[int(0.025 * reps)], 4), round(stats[int(0.975 * reps) - 1], 4)]}
