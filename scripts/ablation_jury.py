"""DEV ablation: single judges vs majority vote vs Dawid-Skene, on the SAME frozen sample.

    python scripts/ablation_jury.py reports/dev-v2-deepseek-pro-<t> reports/dev-v2-qwen-local-<t> reports/dev-v2-<third>-<t>
All runs must share one sample-manifest hash (same --seed/--n-groups). Dawid-Skene is fit on
the judges' votes only (no labels); its parameters are saved for freezing into the jury.
Outputs reports/ablation-jury-<stamp>/{summary.json, ds-params.json}.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.evaluation.aggregation import dawid_skene, majority  # noqa: E402
from medical_jury.evaluation.data import references_by_id  # noqa: E402
from medical_jury.evaluation.scoring import score_predictions  # noqa: E402


def load_run(path: Path):
    plan = json.loads((path / 'plan.json').read_text(encoding='utf-8'))
    manifest = json.loads((path / 'sample-manifest.json').read_text(encoding='utf-8'))
    preds = [json.loads(l) for l in (path / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    return plan, manifest, preds


def compact(score: dict) -> dict:
    m = score['conditional_completed_metrics']
    return {'failure_share': score['failure_share'], 'kappa': m['judge_vs_physician']['cohen_kappa'],
            'ac1': m['judge_vs_physician']['gwet_ac1'], 'agreement': m['judge_vs_physician']['agreement'],
            'delta_kappa': m['delta_kappa'], 'delta_ci': [m['delta_kappa_group_bootstrap_95ci']['lower'],
                                                          m['delta_kappa_group_bootstrap_95ci']['upper']],
            'accepted_if_this_were_validation': score['accepted'], 'checks': score['acceptance_checks']}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args(argv)
    runs = [load_run(r if r.is_absolute() else args.root / r) for r in args.runs]
    hashes = {m['manifest_sha256'] for _, m, _ in runs}
    if len(hashes) != 1:
        raise SystemExit('Runs use different samples; ablation requires one identical frozen sample.')
    if runs[0][1]['split'] != 'dev':
        raise SystemExit('Ablation/aggregator fitting is dev-only.')
    names = [plan.get('judge') or plan['backend'] for plan, _, _ in runs]
    if len(set(names)) != len(names):
        raise SystemExit('Each run must come from a different judge.')
    refs = references_by_id(args.root, {'dev'})
    ids = runs[0][1]['example_ids']
    by_judge = {n: {r['example_id']: r for r in preds} for n, (_, _, preds) in zip(names, runs)}
    votes = []
    for eid in ids:
        votes.append({n: (by_judge[n][eid]['verdict'] == 'met') if by_judge[n].get(eid, {}).get('status') == 'Completed'
                      else None for n in names})
    ds = dawid_skene(votes)
    results = {}
    for n in names:
        results[f'single:{n}'] = compact(score_predictions([by_judge[n].get(e, {'example_id': e, 'status': 'NotRun'})
                                                             for e in ids], refs, n_boot=args.bootstrap))
    maj = []
    ds_preds = []
    for eid, v, post in zip(ids, votes, ds['posteriors']):
        decision = majority(v)
        maj.append({'example_id': eid, 'status': 'Completed' if decision is not None else 'Undecided',
                    'verdict': 'met' if decision else 'not_met'})
        valid = any(x is not None for x in v.values())
        ds_preds.append({'example_id': eid, 'status': 'Completed' if valid else 'Failed',
                         'verdict': 'met' if post >= 0.5 else 'not_met', 'p_met': post})
    results['majority'] = compact(score_predictions(maj, refs, n_boot=args.bootstrap))
    results['dawid_skene'] = compact(score_predictions(ds_preds, refs, n_boot=args.bootstrap))
    pairwise_agreement = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = [(v[a], v[b]) for v in votes if v[a] is not None and v[b] is not None]
            pairwise_agreement[f'{a}~{b}'] = (sum(x == y for x, y in both) / len(both)) if both else None
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = args.root / 'reports' / f'ablation-jury-{stamp}'
    out.mkdir(parents=True)
    params = {k: ds[k] for k in ('prior_met', 'judges', 'iterations', 'log_likelihood')}
    params.update(judges_order=names, fitted_on='dev votes only (no labels)', sample_manifest_sha256=hashes.pop())
    (out / 'ds-params.json').write_text(json.dumps(params, indent=2), encoding='utf-8')
    summary = {'judges': names, 'n_items': len(ids), 'results': results,
               'judge_pairwise_vote_agreement': pairwise_agreement, 'dawid_skene_params': params,
               'rule': 'Keep the multi-judge jury only if it beats the best single judge on dev; otherwise ship the single judge.',
               'status': 'DEV_ABLATION_NOT_VALIDATION'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: {'kappa': v['kappa'], 'delta_kappa': v['delta_kappa']} for k, v in results.items()}, indent=2))
    print(f'-> {out.relative_to(args.root)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
