"""Scorer for external single-annotator datasets (e.g. MedExpert).

Most items carry ONE clinician label, so the per-item 'judge vs physician-physician' delta of the
primary protocol is only computable on the small double-annotated subset. This scorer reports:
  * judge vs clinician on all items (kappa, AC1, Macro-F1, sensitivity for the defect),
  * judge vs clinician AND clinician vs clinician on the double-annotated subset (same items),
  * constant baselines and a permutation test,
per criterion, never pooled across criteria.
"""
from __future__ import annotations

from collections import defaultdict

from . import agreement as A


def _k(pairs):
    return A.cohen_kappa([p for p, _ in pairs], [g for _, g in pairs]) if pairs else None


def score_external(predictions: list[dict], labels: dict[str, dict], n_boot: int = 2000, seed: int = 0) -> dict:
    by_criterion: dict[str, dict] = {}
    criteria = sorted({labels[p['example_id']]['criterion'] for p in predictions})
    for code in criteria:
        preds = [p for p in predictions if labels[p['example_id']]['criterion'] == code]
        done = [p for p in preds if p.get('status') == 'Completed' and p.get('verdict') in ('met', 'not_met')]
        rows = [{'pred': p['verdict'] == 'met', **labels[p['example_id']]} for p in done]
        jh = [(r['pred'], l) for r in rows for l in r['labels']]
        dbl = [r for r in rows if len(r['labels']) >= 2]
        jh_dbl = [(r['pred'], l) for r in dbl for l in r['labels']]
        hh_dbl = A.pairwise_human_pairs([r['labels'] for r in dbl])
        groups = defaultdict(list)
        for r in rows:
            groups[r['group_id']].append(r)
        summary = A.binary_summary(jh)
        c = summary['confusion']
        by_criterion[code] = {
            'planned': len(preds), 'completed': len(done), 'failure_share': 1 - len(done) / len(preds) if preds else None,
            'judge_vs_clinician': summary,
            'defect_sensitivity': (c['TP'] / (c['TP'] + c['FN'])) if (c['TP'] + c['FN']) else None,
            'defect_precision': (c['TP'] / (c['TP'] + c['FP'])) if (c['TP'] + c['FP']) else None,
            'kappa_group_bootstrap_95ci': A.group_bootstrap_ci(
                groups, lambda rs: _k([(r['pred'], l) for r in rs for l in r['labels']]), n_boot, seed),
            'double_annotated_subset': {
                'items': len(dbl), 'judge_vs_clinician_kappa': _k(jh_dbl),
                'clinician_vs_clinician_kappa': _k(hh_dbl),
                'clinician_vs_clinician_ac1': A.gwet_ac1([p for p, _ in hh_dbl], [g for _, g in hh_dbl], [True, False]) if hh_dbl else None,
                'warning': 'Very small subset: wide uncertainty; the clinician-clinician reference itself is unstable.'},
            'constant_baselines': A.constant_baselines(jh) if jh else None,
            'permutation_test': A.permutation_test_kappa([(r['pred'], r['labels']) for r in rows], 2000, seed) if jh else None,
        }
    return {'by_criterion': by_criterion,
            'interpretation': ('met = defect present. Single-annotator labels: compare judge kappa with the '
                               'clinician-clinician kappa on the double-annotated subset, not with 1.0.'),
            'limitations': ['External public dataset; criteria are an adapter paraphrase of the severity scale.',
                            'Clinician inter-annotator agreement on this dataset is low; labels are noisy references.',
                            'Not a clinical validation.']}
