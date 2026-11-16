"""Measure the LLM situation router against HealthBench situations (and against hybrid retrieval).

Gold = the HealthBench theme x situation of each conversation (category metadata; physician met/not-met
labels are NOT read). Groups come from the same dev-A / dev-B split as the retrieval evaluation, so the
comparison with similarity retrieval (dev-B top-1 0.117, majority 0.112) is on the same footing.

    python scripts/eval_situation_router_v3.py --split A --n 60                  # plan only, nothing sent
    python scripts/eval_situation_router_v3.py --split A --n 60 --judge deepseek-pro --authorized --run-limit 1
    python scripts/eval_situation_router_v3.py --judge mock --n 20               # offline smoke (mock router)

Protocol: tune/inspect on dev-A only; run dev-B once with the frozen router prompt and report it.
Writes reports/router-v3-<split>-<UTC>-<rand>/ (plan.json, predictions.jsonl, metrics.json); never overwrites.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from medical_jury.budget import Budget  # noqa: E402
from medical_jury.domain import digest  # noqa: E402
from medical_jury.evaluation.runner import budget_gate, ledger_status  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v3.retrieval import build_query  # noqa: E402
from medical_jury.v3.router import (ROUTER_MAX_OUTPUT_TOKENS, ROUTER_PROMPT_VERSION, SituationRouter,  # noqa: E402
                                    build_router_messages, mock_router_responder)
from medical_jury.v3.schema import CaseMeta, load_kb, load_profile  # noqa: E402


def sample(queries: list[dict], split: str, n: int, seed: int) -> list[dict]:
    pool = sorted((q for q in queries if q['split'] == split), key=lambda q: q['group_id'])
    rnd = random.Random(f'router-v3:{split}:{seed}')
    return rnd.sample(pool, min(n, len(pool)))


def _ci(values: list[float], reps: int = 2000, seed: int = 7) -> list[float] | None:
    if not values:
        return None
    rnd, n = random.Random(seed), len(values)
    stats = sorted(sum(values[rnd.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return [round(stats[int(0.025 * reps)], 4), round(stats[int(0.975 * reps) - 1], 4)]


def metrics(rows: list[dict], majority: str) -> dict:
    done = [r for r in rows if r['status'] == 'Completed']
    n = len(done)
    if not n:
        return {'completed': 0, 'planned': len(rows)}
    group_of = lambda s: s.rsplit(':', 1)[0]  # noqa: E731
    top1 = sum(r['selected'][:1] == [r['gold']] for r in done)
    any_pick = sum(r['gold'] in r['all_picks'] for r in done)
    in_group = [r for r in done if r['gold_group_choice'] is not None]
    return {'planned': len(rows), 'completed': n, 'failed_or_not_run': len(rows) - n,
            'top1_acc': round(top1 / n, 4),
            'top1_acc_ci95': _ci([float(r['selected'][:1] == [r['gold']]) for r in done]),
            'within_gold_group_acc_ci95': _ci([float(r['gold_group_choice'] == r['gold']) for r in in_group]),
            'theme_top1_acc': round(sum(bool(r['selected']) and group_of(r['selected'][0]) == group_of(r['gold'])
                                        for r in done) / n, 4),
            'gold_in_any_pick': round(any_pick / n, 4),
            'within_gold_group_acc': round(sum(r['gold_group_choice'] == r['gold'] for r in in_group) / len(in_group), 4)
            if in_group else None,
            'gold_group_answered_none': round(sum(r['gold_group_choice'] == 'none' for r in in_group) / len(in_group), 4)
            if in_group else None,
            'mean_non_none_picks': round(sum(len(r['all_picks']) for r in done) / n, 3),
            'empty_selection_rate': round(sum(not r['selected'] for r in done) / n, 4),
            'majority_top1_acc_same_rows': round(sum(r['gold'] == majority for r in done) / n, 4),
            'hybrid_top1_acc_same_rows': (round(sum(r['hybrid_top1'] == r['gold'] for r in done) / n, 4)
                                          if all('hybrid_top1' in r for r in done) else None)}


def main(argv=None) -> int:
    import eval_retrieval_v3 as ev
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--split', choices=['A', 'B'], default='A')
    p.add_argument('--n', type=int, default=60)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--judge', default='deepseek-pro')
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--no-hybrid', action='store_true', help='Skip the hybrid comparison (needs the local model).')
    args = p.parse_args(argv)
    root = args.root
    kb, profile = load_kb(root / 'kb/v3'), load_profile(root / 'kb/v3/profile-platform-6d.json')
    queries, skipped = ev.load_queries(root / 'data/prepared/dev.jsonl', None)
    for q in queries:
        q['split'] = ev.split_of(q['group_id'])
    rows = sample(queries, args.split, args.n, args.seed)
    if not rows:
        raise SystemExit(f'No dev-{args.split} groups to evaluate.')
    common = Counter(q['gold'] for q in queries if q['split'] == 'A').most_common(1)
    majority = common[0][0] if common else ''
    store = Store(root)
    from run_review_v3 import make_backend
    backend, paid, settings = make_backend(args.judge, root, store, args.run_limit, ROUTER_MAX_OUTPUT_TOKENS,
                                           mock_router_responder)
    built = []
    for q in rows:
        query = build_query(q['case'], CaseMeta(), profile)
        cands = [s for s in kb.situations]
        built.append((q, query, cands, build_router_messages(query.messages, cands)))
    upper = 0.0
    if paid:
        b = Budget(replace(settings, max_output_tokens=ROUTER_MAX_OUTPUT_TOKENS), None)
        upper = round(sum(b.estimate(m) for *_, m in built), 4)
    job = f'ROUTER-V3-{args.split}-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
    plan = {'job_id': job, 'split': args.split, 'n': len(rows), 'seed': args.seed, 'judge': args.judge,
            'router_prompt': ROUTER_PROMPT_VERSION, 'planned_calls': len(rows), 'max_output_tokens': ROUTER_MAX_OUTPUT_TOKENS,
            'cost_upper_bound_cny': upper, 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None, 'groups_sha256': digest(sorted(q['group_id'] for q in rows)),
            'data_flow': 'conversation + 17 situation descriptions to the judge provider; no answers, no labels.',
            'stop_conditions': 'HTTP 401/402/403/429, budget block or settlement excess stop the run; '
                               'other failures are recorded and counted as not completed.'}
    print(json.dumps(plan, indent=2))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    hybrid = None
    if not args.no_hybrid and (root / '.cache/fastembed').exists():
        from run_review_v3 import make_retriever
        hybrid = make_retriever(kb, root, 'hybrid', 'fastembed')
    router = SituationRouter(backend, store, max_situations=1)
    out = root / 'reports' / f"router-v3-{args.split}-{job.split('-', 3)[3]}"
    out.mkdir(parents=True)
    with (out / 'plan.json').open('x', encoding='utf-8') as f:
        json.dump(plan, f, indent=2)
    results = []
    try:
        with (out / 'predictions.jsonl').open('x', encoding='utf-8') as f:
            for q, query, cands, _ in built:
                rec = router.route(query.messages, cands, job)
                gold_group = q['gold'].rsplit(':', 1)[0]
                choices = rec.get('choices') or []
                row = {'group_id': q['group_id'], 'gold': q['gold'], 'status': rec['status'],
                       'failure_code': rec.get('failure_code'), 'selected': rec.get('selected', []),
                       'all_picks': [c['situation'] for c in choices if c['situation'] != 'none'],
                       'gold_group_choice': next((c['situation'] for c in choices if c['group'] == gold_group), None),
                       'confidences': {c['group']: c['confidence'] for c in choices},
                       'usage': rec.get('usage'), 'messages_sha256': rec['messages_sha256']}
                if hybrid is not None:
                    ranked = hybrid.rank(query)
                    row['hybrid_top1'] = (hybrid.select(ranked) or [''])[0]
                results.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                if router.stop.is_set():
                    break
    finally:
        if hasattr(backend, 'close'):
            backend.close()
    for q, *_ in built[len(results):]:
        results.append({'group_id': q['group_id'], 'gold': q['gold'], 'status': 'NotRun', 'selected': [],
                        'all_picks': [], 'gold_group_choice': None})
    m = {'job_id': job, 'router_prompt': ROUTER_PROMPT_VERSION, 'majority_situation': majority,
         'stopped_early': router.stop.is_set(), **metrics(results, majority)}
    with (out / 'metrics.json').open('x', encoding='utf-8') as f:
        json.dump(m, f, indent=2)
    print(json.dumps({'out': str(out), **m}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
