"""Combine the parts of a held-out ordinal run into the final pre-registered metrics.

    python scripts/aggregate_ordinal_v3.py --runs reports/ordinal-v3-medexpert-test-part1of3-* \
        reports/ordinal-v3-medexpert-test-part2of3-* reports/ordinal-v3-medexpert-test-part3of3-*

Checks before any label is read: every part has a seal whose sha256 matches its predictions file; all parts
share the same dataset, split, freeze and part count; parts 1..K each appear exactly once; together they cover
the split's items exactly. Writes reports/ordinal-v3-<dataset>-<split>-combined-<UTC>/metrics.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.domain import digest  # noqa: E402
from medical_jury.evaluation import ordinal_run as R  # noqa: E402
from medical_jury.evaluation.ordinal import score_ordinal  # noqa: E402
from medical_jury.evaluation.runner import status_counts  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    root = args.root
    plans, records = [], []
    for run in args.runs:
        run = run if run.is_absolute() else root / run
        plan = json.loads((run / 'plan.json').read_text(encoding='utf-8'))
        seal = json.loads((run / 'seal.json').read_text(encoding='utf-8'))
        data = (run / 'predictions.jsonl').read_bytes()
        if hashlib.sha256(data).hexdigest() != seal['predictions_sha256']:
            raise SystemExit(f'{run.name}: predictions do not match their seal.')
        plans.append(plan)
        records += [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
    keys = {(pl['dataset'], pl['split'], json.dumps(pl['freeze'], sort_keys=True), (pl['part'] or {}).get('parts', 1))
            for pl in plans}
    if len(keys) != 1:
        raise SystemExit(f'Runs differ in dataset/split/freeze/part count: {sorted(keys)}')
    dataset, split, _, parts = next(iter(keys))
    if plans[0]['freeze'] is None:
        raise SystemExit('Only frozen held-out runs can be aggregated.')
    got = sorted((pl['part'] or {'part': 1})['part'] for pl in plans)
    if got != list(range(1, parts + 1)):
        raise SystemExit(f'Parts present {got}, expected each of 1..{parts} exactly once.')
    expected = sorted(i['item_id'] for i in R.load_items(root, dataset, split))
    seen = [r['item_id'] for r in records if r.get('item_id')]
    if sorted(seen) != expected:
        raise SystemExit('The parts do not cover the split exactly (missing or duplicate items).')
    profile = R.profile_for(root, dataset)
    joined = R.join_labels(records, root / 'data' / 'private' / 'ordinal-v3' / f'{dataset}-labels.jsonl')
    metrics = score_ordinal(joined, profile, n_boot=args.bootstrap, primary_dimensions=R.DATASETS[dataset]['primary'])
    metrics.update({'label': f'HELD-OUT {dataset}/{split}, {parts} parts combined', 'freeze': plans[0]['freeze'],
                    'parts': [{'job_id': pl['job_id'], 'part': pl['part']} for pl in plans],
                    'item_ids_sha256': digest(expected), 'status_counts': status_counts(records)})
    out = root / 'reports' / f"ordinal-v3-{dataset}-{split}-combined-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True)
    with (out / 'metrics.json').open('x', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({'out': str(out), 'mean_primary_qwk': metrics['mean_primary_qwk'],
                      'mean_primary_qwk_ci95': metrics['mean_primary_qwk_group_bootstrap_95ci'],
                      'status_counts': metrics['status_counts']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
