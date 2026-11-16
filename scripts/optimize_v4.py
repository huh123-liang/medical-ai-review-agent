"""DSPy (GEPA) optimisation of the v4 CHECK instruction on the TRAIN split only (PLAN-V4 section 5).

    # needs a sealed full-arm run on the whole train split (scripts/run_review_v4.py run --arm full --split train)
    python scripts/optimize_v4.py --train-run-dir reports/review-v4-<...> --judge deepseek-pro --run-limit 12 \\
        --prices-checked 2026-11-20 [--authorized]

Writes reports/review-v4-dspy-<stamp>/optimised-check.json. The test split is never read. The result is used only
through `run_review_v4.py run --check-prompt-file ...`, and only if its guard checks are clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from medical_jury.budget import Budget  # noqa: E402
from medical_jury.evaluation.runner import budget_gate, ledger_status  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v4.pipeline import CHECK_MAX_TOKENS  # noqa: E402

import run_review_v4 as RV  # noqa: E402

VERSION = 'review-v4-dspy-1'
VAL_PATIENT_FROM = 35            # train-split patients 0-34 -> GEPA minibatches, 35-49 -> GEPA validation
VAL_CASES = 20


def split_examples(cases: list[dict], records: dict, documents: dict) -> tuple[list, list]:
    from medical_jury.v4.optimize import make_example
    usable = [c for c in cases if c['split'] == 'train' and c['case_id'] in records
              and records[c['case_id']].get('items') is not None and c['case_id'] in documents]
    if any(c['split'] != 'train' for c in usable):
        raise SystemExit('only the train split may be used for optimisation')
    num = lambda c: int(c['patient']['patient_id'].split('-')[1])      # noqa: E731
    train = [c for c in usable if num(c) < VAL_PATIENT_FROM]
    val = [c for c in usable if num(c) >= VAL_PATIENT_FROM][:VAL_CASES]
    ex = lambda c: make_example(c, records[c['case_id']], documents[c['case_id']])  # noqa: E731
    return [ex(c) for c in train], [ex(c) for c in val]


def main(argv=None) -> int:
    from run_review_v3 import make_backend
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--train-run-dir', type=Path, required=True)
    p.add_argument('--style', default=RV.PRIMARY_STYLE)
    p.add_argument('--judge', default='mock')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--max-task-calls', type=int, default=180)
    p.add_argument('--max-reflection-calls', type=int, default=20)
    p.add_argument('--prices-checked', type=date.fromisoformat)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--authorized', action='store_true')
    args = p.parse_args(argv)
    root = args.root
    run_dir = args.train_run_dir if args.train_run_dir.is_absolute() else root / args.train_run_dir
    path = run_dir / f'full-train-{args.style}' / 'records.jsonl'
    RV._verify_seal(path)
    records = {r['case_id']: r for r in RV._read_jsonl(path) if r.get('status') == 'Completed'}
    cases = [c for c in RV._read_jsonl(RV.testbed_dir(root) / 'cases.jsonl') if c['split'] == 'train']
    documents = {u['case_id']: u['document'] for u in RV.load_inputs(root, 'train', args.style)}
    train, val = split_examples(cases, records, documents)
    store = Store(root)
    backend, paid, settings = make_backend(args.judge, root, store, args.run_limit, CHECK_MAX_TOKENS,
                                           responder=_offline(root))
    if paid and args.prices_checked is None:
        raise SystemExit('Paid run: recheck the provider price list and pass --prices-checked YYYY-MM-DD.')
    upper = 0.0
    if paid:
        est = Budget(replace(settings, max_output_tokens=CHECK_MAX_TOKENS), None)
        worst = max(est.estimate([{'role': 'system', 'content': 'x' * 8000}, {'role': 'user', 'content': e.payload}])
                    for e in train + val)
        reflection = est.estimate([{'role': 'user', 'content': 'x' * (4 * max(len(e.payload) for e in train) + 12000)}])
        upper = round(args.max_task_calls * worst + args.max_reflection_calls * reflection, 4)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    job = f'REVIEWV4-DSPY-{stamp}-{uuid4().hex[:6]}'
    plan = {'job_id': job, 'version': VERSION, 'train_run_dir': run_dir.name, 'style': args.style,
            'train_examples': len(train), 'val_examples': len(val),
            'train_records_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'caps': {'task_calls': args.max_task_calls, 'reflection_calls': args.max_reflection_calls},
            'judge': args.judge, 'prices_checked': args.prices_checked.isoformat() if args.prices_checked else None,
            'cost_upper_bound_cny': upper, 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None,
            'data_flow': 'Synthetic TRAIN-split case descriptions and write-ups only; the test split is never read.',
            'label': 'DEVELOPMENT (instruction optimisation, train split)'}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    if paid:
        RV._claim(root / 'reports' / RV.SENTINELS / f'dspy-{run_dir.name}.json', {'job_id': job, 'started_at': stamp})
    out = root / 'reports' / f'review-v4-dspy-{stamp}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    RV._write_json(out / 'plan.json', plan)
    from medical_jury.v4.optimize import optimise_check
    result = optimise_check(train, val, backend, backend, job, args.max_task_calls, args.max_reflection_calls,
                            args.seed)
    result.update(job_id=job, version=VERSION, offline_mock=not paid,
                  usable=not result['guard_problems'] and result['changed'])
    RV._write_json(out / 'optimised-check.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('instruction', 'train_cases', 'val_cases')},
                     indent=2))
    print(f'Written: {out}')
    return 0


def _offline(root):
    return RV._offline_responder(root)


if __name__ == '__main__':
    raise SystemExit(main())
