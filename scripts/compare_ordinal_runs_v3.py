"""Paired comparison of two DEV ordinal runs on the same items (e.g. two prompt versions), with the decision
rule pre-registered in docs/project/PROTOCOL-V3-ORDINAL.md section 9.

    python scripts/compare_ordinal_runs_v3.py --a reports/ordinal-v3-...-<old> --b reports/ordinal-v3-...-<new>

Checks first: both runs are dev runs (no freeze), sealed predictions match their seal, same dataset, judge and
call mode, identical item sets. Writes reports/ordinal-v3-compare-<UTC>-<rand>/comparison.json (never overwrites).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.evaluation import ordinal_run as R  # noqa: E402
from medical_jury.evaluation.ordinal import COVERAGE_MAX_FAILURE_SHARE, compare_runs  # noqa: E402


def load_run(run: Path) -> tuple[dict, list[dict]]:
    plan = json.loads((run / 'plan.json').read_text(encoding='utf-8'))
    seal = json.loads((run / 'seal.json').read_text(encoding='utf-8'))
    if hashlib.sha256((run / 'predictions.jsonl').read_bytes()).hexdigest() != seal['predictions_sha256']:
        raise SystemExit(f'{run.name}: predictions do not match their seal.')
    if plan.get('freeze') is not None or plan['split'] != 'dev':
        raise SystemExit(f'{run.name}: only dev runs can be compared for development decisions.')
    joined = [json.loads(line) for line in (run / 'joined.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    return plan, joined


def decide(cmp: dict, fail_share: dict) -> dict:
    """Section 9 rule: adopt B only if both runs meet coverage and B's mean primary QWK is higher."""
    if any(v > COVERAGE_MAX_FAILURE_SHARE for v in fail_share.values()):
        return {'adopt_b': False, 'reason': f'coverage rule failed {fail_share}: no decision, report only'}
    d = cmp['delta_b_minus_a']
    if d is None:
        return {'adopt_b': False, 'reason': 'mean primary QWK undefined in one run: keep A'}
    return {'adopt_b': d > 0, 'reason': f'delta {d:+.4f} ' + ('> 0: adopt B' if d > 0 else '<= 0: keep A')}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--a', type=Path, required=True, help='baseline run directory')
    p.add_argument('--b', type=Path, required=True, help='candidate run directory')
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    root = args.root
    runs = {k: (v if v.is_absolute() else root / v) for k, v in (('a', args.a), ('b', args.b))}
    (pa, ja), (pb, jb) = load_run(runs['a']), load_run(runs['b'])
    for key in ('dataset', 'judge', 'call_mode', 'item_ids_sha256'):
        if pa[key] != pb[key]:
            raise SystemExit(f'Runs differ in {key}: {pa[key]!r} vs {pb[key]!r}.')
    dataset = pa['dataset']
    cmp = compare_runs(ja, jb, R.profile_for(root, dataset), R.DATASETS[dataset]['primary'], n_boot=args.bootstrap)
    fail_share = {k: sum(r['status'] != 'Completed' for r in j) / len(j) for k, j in (('a', ja), ('b', jb))}
    result = {'label': 'DEV comparison (development evidence only)', 'dataset': dataset,
              'a': {'job_id': pa['job_id'], 'prompt_version': pa['prompt_version'], 'run': runs['a'].name},
              'b': {'job_id': pb['job_id'], 'prompt_version': pb['prompt_version'], 'run': runs['b'].name},
              'failure_share': fail_share, 'decision': decide(cmp, fail_share), **cmp}
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / 'reports' / f'ordinal-v3-compare-{stamp}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    with (out / 'comparison.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({'out': str(out), 'a': result['a'], 'b': result['b'],
                      'mean_primary_qwk_a': cmp['mean_primary_qwk_a'], 'mean_primary_qwk_b': cmp['mean_primary_qwk_b'],
                      'delta_b_minus_a': cmp['delta_b_minus_a'], 'delta_ci95': cmp['delta_group_bootstrap_95ci'],
                      'decision': result['decision']}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
