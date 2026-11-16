"""Plan phase 2: situation router v1 vs v2 on fresh HealthBench dev groups (PROTOCOL-V3-ROUTER-V2).

    # 1. sample fresh dev groups (label-blind) and route every conversation with BOTH prompt versions (paired)
    python scripts/run_router_v2.py route --judge deepseek-pro --run-limit 20
    # 1b. only if a technical stop (402/429/budget ...) interrupted stage 1: re-send the interrupted calls once
    python scripts/run_router_v2.py complete --run-dir reports/router-v3b-<...> --judge deepseek-pro --run-limit 10
    # 2. after sealing: read the gold situation (category metadata) and apply the pre-registered rule (no call)
    python scripts/run_router_v2.py score --run-dir reports/router-v3b-<...>

Every stage is plan-first: without --authorized a paid stage prints calls, cost upper bound and ledger and writes
nothing. The paid route stage claims a once-only sentinel, so the sample cannot be redrawn after seeing results.
Only the conversation is sent (never the answer, never physician labels; the router never sees the category).
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
from medical_jury.domain import digest  # noqa: E402
from medical_jury.evaluation import router_scoring as RSC  # noqa: E402
from medical_jury.evaluation.data import load_prepared  # noqa: E402
from medical_jury.evaluation.runner import (IMMEDIATE_STOP, LiveWriter, StopController, budget_gate,  # noqa: E402
                                            ledger_status, run_planned, status_counts)
from medical_jury.evaluation.sampling import natural_sample  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v3 import ablation as AB  # noqa: E402
from medical_jury.v3.router import (DECISION_TESTS_V2, ROUTER_MAX_OUTPUT_TOKENS, ROUTER_PROMPT_VERSION,  # noqa: E402
                                    ROUTER_PROMPT_VERSION_2, SituationRouter, build_router_messages,
                                    load_decision_tests, mock_router_responder)
from medical_jury.v3.schema import load_kb  # noqa: E402
from medical_jury.v3.used_groups import used_dev_groups_v3  # noqa: E402

VERSION = 'router-v2-compare-1'
KB = 'kb/v3'
V1, V2 = ROUTER_PROMPT_VERSION, ROUTER_PROMPT_VERSION_2
SHORT = {V1: 'v1', V2: 'v2'}
DEFAULT_GROUPS = 150
DEFAULT_SEED = 20261029
SENTINELS = 'router-v3b-sentinels'
DATA_FLOW = ('HealthBench (MIT) public dev conversations only (no answer, no criterion, no labels) are sent to the '
             'judge provider, twice per conversation (router v1 and v2). The gold situation (category metadata) is '
             'read only in the score stage, after the routes are sealed.')
STOP_RULES = ('HTTP 401/402/403/429 or budget -> immediate stop; 3 consecutive failures; >5 failures in the first '
              '50 calls; no automatic retry; rest recorded NotRun (may be completed once with the complete stage).')


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _write_json(path: Path, obj) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _seal(path: Path, records, controller, interrupted) -> None:
    _write_json(path.with_name(path.stem + '-seal.json'),
                {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'records': len(records),
                 'status_counts': status_counts(records), 'stop': controller.summary(), 'interrupted': interrupted,
                 'sealed_at': datetime.now(timezone.utc).isoformat()})


def _verify_seal(path: Path) -> None:
    seal = json.loads(path.with_name(path.stem + '-seal.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != seal['sha256']:
        raise SystemExit(f'{path} does not match its seal.')


def _claim(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json(path, payload)
    except FileExistsError:
        raise SystemExit(f'{path} exists: this stage was already started and may run only once.') from None


def _conversation(row: dict) -> tuple[tuple[str, str], ...]:
    return tuple((m['role'], m['content']) for m in row['case']['context'])


def _interrupted(rec: dict) -> bool:
    return rec.get('status') == 'NotRun' or (rec.get('status') == 'Failed' and rec.get('failure_code') in IMMEDIATE_STOP)


def _run_dir(args, root: Path) -> Path:
    if args.run_dir is None:
        raise SystemExit(f'{args.stage} needs --run-dir (the folder written by the route stage).')
    return args.run_dir if args.run_dir.is_absolute() else root / args.run_dir


def _units(root: Path, rows: list[dict], kb, tests: dict) -> list[dict]:
    """(item, version) units, interleaved so that an interruption leaves both versions equally complete."""
    units = []
    for row in rows:
        conv = _conversation(row)
        for v in (V1, V2):
            msgs = build_router_messages(conv, kb.situations, v, tests if v == V2 else None)
            units.append({'unit_id': f"{row['example_id']}@{SHORT[v]}", 'item_id': row['example_id'],
                          'group_id': row['case']['group_id'], 'version': v, 'conversation': conv,
                          'messages': msgs, 'messages_sha256': digest(msgs)})
    return units


def _execute(args, root: Path, units: list[dict], kb, tests, label: str, out: Path, extra_plan: dict,
             before_calls=None) -> int:
    from run_review_v3 import make_backend          # same budgeted DeepSeek backend as the review CLI
    store = Store(root)
    backend, paid, settings = make_backend(args.judge, root, store, args.run_limit, ROUTER_MAX_OUTPUT_TOKENS,
                                           responder=mock_router_responder)
    upper = 0.0
    if paid:
        est = Budget(replace(settings, max_output_tokens=ROUTER_MAX_OUTPUT_TOKENS), None)
        upper = round(sum(est.estimate(u['messages']) for u in units), 4)
    stamp = _stamp()
    job = f'ROUTERV2-{label}-{stamp}-{uuid4().hex[:6]}'
    plan = {'job_id': job, 'stage': label, 'version': VERSION, **extra_plan, 'planned_calls': len(units),
            'calls_by_prompt': {v: sum(u['version'] == v for u in units) for v in (V1, V2)},
            'decision_tests_sha256': tests['sha256'], 'kb_sha256': kb.sha256(), 'judge': args.judge,
            'max_output_tokens': ROUTER_MAX_OUTPUT_TOKENS, 'units_sha256': digest(sorted(u['unit_id'] for u in units)),
            'threads': args.threads, 'cost_upper_bound_cny': upper, 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None, 'label': 'DEV comparison (fresh, unseen dev groups)',
            'data_flow': DATA_FLOW, 'stop_conditions': STOP_RULES}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    if before_calls:
        before_calls(paid, job, stamp)
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f'{out} was already run; it may run only once.') from None
    _write_json(out / 'plan.json', plan)
    controller = StopController()
    routers = {V1: SituationRouter(backend, store, stop=controller.event, prompt_version=V1),
               V2: SituationRouter(backend, store, stop=controller.event, prompt_version=V2, decision_tests=tests)}

    def work(u):
        rec = routers[u['version']].route(u['conversation'], kb.situations, job)
        if rec.get('messages_sha256') != u['messages_sha256']:
            raise RuntimeError('messages changed between plan and call')
        return {'unit_id': u['unit_id'], 'item_id': u['item_id'], 'group_id': u['group_id'], **rec}
    writer = LiveWriter(out / 'routes.jsonl')
    try:
        records, interrupted = run_planned(units, work, id_of=lambda u: u['unit_id'], writer=writer,
                                           controller=controller, threads=args.threads)
    finally:
        writer.close()
        if hasattr(backend, 'close'):
            backend.close()
    _seal(out / 'routes.jsonl', records, controller, interrupted)
    print(json.dumps({'out': str(out), 'status_counts': status_counts(records), 'interrupted': interrupted},
                     indent=2))
    return 0


def _selected_rows(root: Path, run_dir: Path) -> list[dict]:
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
    rows = {r['example_id']: r for r in load_prepared(root, 'dev')}
    return [rows[e] for e in manifest['example_ids']]


# ---------------------------------------------------------------- stage: route (paid, once)

def stage_route(args, root: Path) -> int:
    kb = load_kb(root / KB)
    tests = load_decision_tests(root / DECISION_TESTS_V2, kb.situations)
    excluded, used_runs = used_dev_groups_v3(root)
    selected, manifest = natural_sample(load_prepared(root, 'dev'), args.n_groups, args.seed, 'dev',
                                        exclude_groups=excluded)
    for row in selected:                              # every gold situation must resolve before any call
        AB.resolve(row['category'], kb)
    manifest.update(version=VERSION, excluded_from_runs=used_runs, prompts=[V1, V2],
                    decision_tests_sha256=tests['sha256'])
    units = _units(root, selected, kb, tests)
    stamp = _stamp()
    out = root / 'reports' / f'router-v3b-{stamp}-{uuid4().hex[:6]}'

    def claim(paid, job, s):
        if paid:                                      # one paid sample only: no redraw after seeing results
            _claim(root / 'reports' / SENTINELS / f'route-{VERSION}.json', {'job_id': job, 'started_at': s})
        out.mkdir(parents=True)
        _write_json(out / 'sample-manifest.json', manifest)
    extra = {'items': len(selected), 'excluded_groups': len(excluded),
             'sample_manifest_sha256': manifest['manifest_sha256'], 'seed': args.seed}
    return _execute(args, root, units, kb, tests, 'route', out / 'route', extra, claim)


# ---------------------------------------------------------------- stage: complete (paid, once, only if interrupted)

def stage_complete(args, root: Path) -> int:
    run_dir = _run_dir(args, root)
    kb = load_kb(root / KB)
    tests = load_decision_tests(root / DECISION_TESTS_V2, kb.situations)
    path = run_dir / 'route' / 'routes.jsonl'
    _verify_seal(path)
    done = {r['example_id']: r for r in _read_jsonl(path)}          # run_planned keys records by unit id
    units = _units(root, _selected_rows(root, run_dir), kb, tests)
    todo = [u for u in units if _interrupted(done[u['unit_id']])]
    if not todo:
        print('Nothing to complete: no call was stopped by a technical interruption.')
        return 0

    def claim(paid, job, s):
        if paid:
            _claim(root / 'reports' / SENTINELS / f'complete-{run_dir.name}.json',
                   {'job_id': job, 'units': len(todo), 'started_at': s})
    return _execute(args, root, todo, kb, tests, 'complete', run_dir / 'completion',
                    {'run_dir': run_dir.name, 'items': len({u['item_id'] for u in todo})}, claim)


# ---------------------------------------------------------------- stage: score (free; gold read only here)

def load_routes(run_dir: Path) -> dict[str, dict]:
    path = run_dir / 'route' / 'routes.jsonl'
    _verify_seal(path)
    routes = {r['example_id']: r for r in _read_jsonl(path)}
    comp = run_dir / 'completion' / 'routes.jsonl'
    if comp.exists():
        _verify_seal(comp)
        for r in _read_jsonl(comp):
            if _interrupted(routes[r['example_id']]):
                routes[r['example_id']] = {**r, 'completed_later': True}
    return routes


def stage_score(args, root: Path) -> int:
    run_dir = _run_dir(args, root)
    kb = load_kb(root / KB)
    routes = load_routes(run_dir)
    selected = _selected_rows(root, run_dir)
    rows, conf = [], {V1: [], V2: []}
    for row in selected:                              # gold situation read only now, after sealing
        target = AB.resolve(row['category'], kb)
        r = {'item_id': row['example_id'], 'group': target.group, 'gold': target.gold_situation}
        for v in (V1, V2):
            rec = routes.get(f"{row['example_id']}@{SHORT[v]}")
            r[v] = AB.rag_selection(rec, target)
            r[f'{v}:status'] = (rec or {}).get('status', 'NotRun')
            choice = next((c for c in (rec or {}).get('choices', []) if c['group'] == target.group), None)
            if choice is not None:
                conf[v].append((choice['confidence'], r[v] == r['gold']))
        rows.append(r)
    status = {v: {s: sum(r[f'{v}:status'] == s for r in rows) for s in {r[f'{v}:status'] for r in rows}}
              for v in (V1, V2)}
    result = RSC.compare(rows, V1, V2, args.seed)
    metrics = {'version': VERSION, 'run_dir': run_dir.name, **result, 'status': status,
               'interrupted_left': {v: sum(_interrupted({'status': r[f'{v}:status']}) for r in rows) for v in (V1, V2)},
               'macro_recall_min_n': RSC.MIN_SITUATION_N,
               'recall_by_situation': {v: RSC.recall_by_situation(rows, v) for v in (V1, V2)},
               'confusion': {v: RSC.confusion(rows, v) for v in (V1, V2)},
               'confidence_auroc': {v: RSC.auroc([c for c, _ in conf[v]], [ok for _, ok in conf[v]])
                                    for v in (V1, V2)},
               'note': 'Gold = HealthBench category metadata; failures and "none" count as wrong.'}
    out = run_dir / f'score-{_stamp()}'
    out.mkdir()
    _write_json(out / 'metrics.json', metrics)
    with (out / 'items.jsonl').open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(json.dumps({k: metrics[k] for k in ('n', 'accuracy', 'difference', 'difference_ci95', 'discordant',
                                              'mcnemar_exact_p', 'macro_recall', 'by_group', 'adopt_v2', 'rule',
                                              'status', 'confidence_auroc')}, indent=2, ensure_ascii=False))
    print(f'Written: {out}')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('route', 'complete', 'score'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--judge', default='mock')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--n-groups', type=int, default=DEFAULT_GROUPS)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--authorized', action='store_true')
    args = p.parse_args(argv)
    return {'route': stage_route, 'complete': stage_complete, 'score': stage_score}[args.stage](args, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
