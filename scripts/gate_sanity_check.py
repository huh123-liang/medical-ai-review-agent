"""Validate the validator: would a REAL physician pass PROTOCOL-V2's acceptance gates?

On items with >= 3 physicians, hold one physician out as the 'judge' and score them against the
remaining physicians with the exact scorer used for the model. Gates that a physician routinely fails
are miscalibrated (too strict or underpowered) and must be fixed BEFORE freezing. No model calls.

    python scripts/gate_sanity_check.py
Writes reports/gate-sanity/summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.evaluation.data import load_rater_index  # noqa: E402
from medical_jury.evaluation.scoring import score_predictions  # noqa: E402


def physician_as_judge(refs: list[dict], hold: int, n_boot: int, seed: int) -> dict:
    preds, others = [], {}
    for r in refs:
        labels, ids = r['labels'], r['physician_ids']
        verdict = labels[hold]
        preds.append({'example_id': r['example_id'], 'status': 'Completed',
                      'verdict': 'met' if verdict else 'not_met', 'p_met': 0.9 if verdict else 0.1})
        others[r['example_id']] = {**r, 'labels': labels[:hold] + labels[hold + 1:],
                                   'physician_ids': ids[:hold] + ids[hold + 1:]}
    s = score_predictions(preds, others, n_boot=n_boot, seed=seed)
    m = s['conditional_completed_metrics']
    return {'held_out_position': hold, 'n_items': len(refs),
            'kappa_judge_vs_physicians': m['judge_vs_physician']['cohen_kappa'],
            'kappa_physicians_same_items': m['physician_vs_physician_same_items']['cohen_kappa'],
            'delta_kappa': m['delta_kappa'], 'delta_ci': [m['delta_kappa_group_bootstrap_95ci']['lower'],
                                                          m['delta_kappa_group_bootstrap_95ci']['upper']],
            'checks': {k: v for k, v in s['acceptance_checks'].items() if k != 'alt_test_passed'}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=11)
    args = p.parse_args(argv)
    result = {}
    for split in ('dev', 'validation'):
        refs = [r for r in load_rater_index(args.root, with_labels=True, splits={split}) if len(r['labels']) >= 3]
        result[split] = [physician_as_judge(refs, h, args.bootstrap, args.seed) for h in (0, 1, 2)]
    result['reading'] = ('A gate is usable only if held-out physicians usually pass it. Delta-kappa CI half-width '
                         'shrinks roughly with 1/sqrt(n): size the frozen natural sample so a physician-level judge '
                         'passes the non-inferiority margin with high probability.')
    out = args.root / 'reports/gate-sanity'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for split in ('dev', 'validation'):
        for row in result[split]:
            print(split, row['n_items'], 'hold', row['held_out_position'], 'delta', round(row['delta_kappa'], 3),
                  'CI', [round(x, 3) for x in row['delta_ci']], row['checks'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
