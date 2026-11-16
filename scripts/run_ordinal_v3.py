"""Score an ordinal dataset split with the v3 agent, seal the predictions, then compute agreement.

Plan first (nothing is sent, nothing is written):
    python scripts/run_ordinal_v3.py --dataset bionlp-trec --split dev --n 40 --judge deepseek-pro
Run after the owner approved the printed plan:
    python scripts/run_ordinal_v3.py --dataset bionlp-trec --split dev --n 40 --judge deepseek-pro \
        --authorized --run-limit 15
Validation / external test (once only, frozen system):
    python scripts/run_ordinal_v3.py --dataset bionlp-trec --split validation --judge deepseek-pro \
        --freeze reports/protocol-freeze-v3/<id>/freeze.json --authorized --run-limit <x>

Writes reports/ordinal-v3-<dataset>-<split>-<UTC>-<rand>/:
    plan.json, predictions.jsonl (live, no labels), seal.json (sha256 of predictions, written BEFORE labels are
    read), metrics.json, joined.jsonl (prediction + rater scores per item, for error analysis).
Never overwrites; validation/test leave a sentinel so they cannot be run twice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from medical_jury.budget import Budget  # noqa: E402
from medical_jury.domain import CaseInput, digest  # noqa: E402
from medical_jury.evaluation import ordinal_run as R  # noqa: E402
from medical_jury.evaluation.ordinal import score_ordinal  # noqa: E402
from medical_jury.evaluation.runner import (LiveWriter, StopController, budget_gate, ledger_status,  # noqa: E402
                                            run_planned, status_counts)
from medical_jury.jury_config import load_panel  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v3.evaluator import (EVALUATOR_PROMPT_VERSION, PROMPT_VERSIONS,  # noqa: E402
                                       build_messages_group, build_messages_v3)

LICENCE = ('Dataset licences unstated by the authors: local research use only (owner-approved 2026-10-25); '
           'conversation + answer + scoring profile are sent to the judge provider; no labels, no rater ids.')


def judge_spec(root: Path, judge: str) -> dict:
    if judge == 'mock':
        return {'name': 'mock', 'kind': 'mock', 'model': 'OFFLINE-SCRIPTED-NOT-CLINICAL'}
    spec = next((j for j in load_panel(root) if j['name'] == judge), None)
    if spec is None:
        raise SystemExit(f'Judge {judge!r} not in jury-v2.json.')
    return spec


def messages_for(item: dict, profile, call_mode: str, prompt_version: str) -> list[list[dict]]:
    case = CaseInput.model_validate(item['request']['case'])
    if call_mode == 'single':
        return [build_messages_group(case, [(d, [], []) for d in profile.dimensions], prompt_version)]
    return [build_messages_v3(case, d, [], [], prompt_version) for d in profile.dimensions]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--dataset', choices=sorted(R.DATASETS), required=True)
    p.add_argument('--split', required=True)
    p.add_argument('--n', type=int, default=None, help='dev only: whole question groups until >= n items')
    p.add_argument('--seed', default='ordinal-v3-dev-subset-1')
    p.add_argument('--complement', action='store_true',
                   help='dev only: run the dev items NOT in the --n/--seed subset (the unseen remainder, dev-B)')
    p.add_argument('--judge', default='mock')
    p.add_argument('--call-mode', choices=R.CALL_MODES, default='single')
    p.add_argument('--prompt-version', choices=list(PROMPT_VERSIONS), default=EVALUATOR_PROMPT_VERSION)
    p.add_argument('--part', type=int, default=1, help='held-out only: which batch of --parts to run')
    p.add_argument('--parts', type=int, default=1, help='held-out only: split the run into this many batches')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--freeze', type=Path, default=None)
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    root = args.root
    from run_review_v3 import make_backend          # same budgeted backend factory as the review CLI

    held_out = args.split != 'dev'
    if held_out and args.n:
        raise SystemExit('--n (subsets) is allowed on dev only; validation/test run in full.')
    if args.complement and not args.n:
        raise SystemExit('--complement needs the --n (and --seed) that define the subset to exclude.')
    if not held_out and args.parts != 1:
        raise SystemExit('--parts is for validation/test runs; use --n on dev.')
    if held_out and not args.freeze:
        raise SystemExit(f'{args.split} runs need --freeze <freeze.json> (protocol section 6).')
    spec = judge_spec(root, args.judge)
    profile = R.profile_for(root, args.dataset)
    current = R.system_fields(root, spec, args.call_mode, args.prompt_version)
    freeze_info = None
    if args.freeze:
        freeze = json.loads((args.freeze if args.freeze.is_absolute() else root / args.freeze).read_text(encoding='utf-8'))
        bad = R.freeze_mismatches(freeze, current)
        if bad:
            raise SystemExit(f'System differs from the freeze in: {", ".join(bad)}. No call made.')
        freeze_info = {'freeze_sha256': freeze.get('freeze_sha256'), 'path': str(args.freeze)}

    items = R.subset(R.load_items(root, args.dataset, args.split), args.n, args.seed, args.complement)
    items = R.part_of(items, args.dataset, args.split, args.part, args.parts)
    store = Store(root)
    max_tokens = R.max_tokens_for(args.call_mode, profile)
    backend, paid, settings = make_backend(args.judge, root, store, args.run_limit, max_tokens)
    upper = 0.0
    calls = sum(len(messages_for(i, profile, args.call_mode, args.prompt_version)) for i in items)
    if paid:
        b = Budget(replace(settings, max_output_tokens=max_tokens), None)
        upper = round(sum(b.estimate(m) for i in items
                          for m in messages_for(i, profile, args.call_mode, args.prompt_version)), 4)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = f'ORD3-{args.dataset}-{args.split}-{stamp}-{uuid4().hex[:6]}'
    plan = {'job_id': job, 'dataset': args.dataset, 'split': args.split, 'items': len(items),
            'groups': len({i['group_id'] for i in items}), 'item_ids_sha256': digest(sorted(i['item_id'] for i in items)),
            'subset': {'n': args.n, 'seed': args.seed, 'complement': args.complement} if args.n else None,
            'planned_calls': calls,
            'part': {'part': args.part, 'parts': args.parts} if args.parts > 1 else None,
            'judge': current['judge'], 'call_mode': args.call_mode, 'prompt_version': args.prompt_version,
            'max_output_tokens': max_tokens, 'threads': args.threads, 'cost_upper_bound_cny': upper,
            'run_limit_cny': args.run_limit, 'ledger': ledger_status(root) if paid else None,
            'freeze': freeze_info, 'label': 'HELD-OUT (frozen)' if held_out else 'DEV (development evidence only)',
            'data_flow': LICENCE,
            'stop_conditions': 'HTTP 401/402/403/429 or budget -> immediate stop; 3 consecutive failures; '
                               '>5 failures in the first 50 items; no automatic retry; rest recorded NotRun.'}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    if held_out:                                     # once only: claim the slot before the first call
        sdir = root / 'reports' / 'ordinal-v3-sentinels'
        sdir.mkdir(parents=True, exist_ok=True)
        layout = sdir / f'{args.dataset}-{args.split}-parts.json'
        try:
            with layout.open('x', encoding='utf-8') as f:
                json.dump({'parts': args.parts, 'freeze': freeze_info}, f)
        except FileExistsError:
            fixed = json.loads(layout.read_text(encoding='utf-8'))
            if fixed['parts'] != args.parts or fixed['freeze'] != freeze_info:
                raise SystemExit(f'{args.dataset}/{args.split} is already being run as {fixed["parts"]} part(s) '
                                 'with another freeze/partition; it cannot be re-partitioned.')
        name = f'{args.dataset}-{args.split}' + (f'-part{args.part}of{args.parts}' if args.parts > 1 else '')
        sentinel = sdir / f'{name}.json'
        try:
            with sentinel.open('x', encoding='utf-8') as f:
                json.dump({'job_id': job, 'freeze': freeze_info, 'started_at': stamp}, f)
        except FileExistsError:
            raise SystemExit(f'{name} was already run ({sentinel}). It may run only once.')
    part_tag = f'-part{args.part}of{args.parts}' if args.parts > 1 else ('-complement' if args.complement else '')
    out = root / 'reports' / f'ordinal-v3-{args.dataset}-{args.split}{part_tag}-{stamp}-{job[-6:]}'
    out.mkdir(parents=True)
    with (out / 'plan.json').open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    controller = StopController()
    evaluator = R.make_evaluator(backend, store, controller.event, args.prompt_version)
    writer = LiveWriter(out / 'predictions.jsonl')
    try:
        records, interrupted = run_planned(
            items, lambda it: R.score_item(evaluator, it, profile, args.call_mode, job),
            id_of=lambda it: it['item_id'], writer=writer, controller=controller, threads=args.threads)
    finally:
        writer.close()
        if hasattr(backend, 'close'):
            backend.close()
    for rec, it in zip(records, items):                       # NotRun stubs lack these keys
        rec.setdefault('item_id', it['item_id'])
        rec.setdefault('group_id', it['group_id'])
    sealed = hashlib.sha256((out / 'predictions.jsonl').read_bytes()).hexdigest()
    with (out / 'seal.json').open('x', encoding='utf-8') as f:     # sealed BEFORE any label is read
        json.dump({'predictions_sha256': sealed, 'records': len(records), 'status_counts': status_counts(records),
                   'stop': controller.summary(), 'interrupted': interrupted,
                   'sealed_at': datetime.now(timezone.utc).isoformat()}, f, indent=2)
    labels_path = root / 'data' / 'private' / 'ordinal-v3' / f'{args.dataset}-labels.jsonl'
    joined = R.join_labels(records, labels_path)
    metrics = score_ordinal(joined, profile, n_boot=args.bootstrap, primary_dimensions=R.DATASETS[args.dataset]['primary'])
    if args.parts > 1:
        plan['label'] += f' PART {args.part}/{args.parts}: conditional; combine all parts with aggregate_ordinal_v3.py'
    metrics.update({'job_id': job, 'label': plan['label'], 'predictions_sha256': sealed, 'freeze': freeze_info,
                    'part': plan['part'],
                    'status_counts': status_counts(records), 'stop': controller.summary()})
    with (out / 'metrics.json').open('x', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with (out / 'joined.jsonl').open('x', encoding='utf-8') as f:
        for j in joined:
            f.write(json.dumps(j, ensure_ascii=False) + '\n')
    summary = {d: {k: v['primary'].get(k) for k in ('n_pairs', 'qwk', 'exact_agreement', 'mae')}
               for d, v in metrics['dimensions'].items()}
    print(json.dumps({'out': str(out), 'label': plan['label'], 'status_counts': metrics['status_counts'],
                      'mean_primary_qwk': metrics['mean_primary_qwk'],
                      'mean_primary_qwk_ci95': metrics['mean_primary_qwk_group_bootstrap_95ci'],
                      'per_dimension': summary}, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
