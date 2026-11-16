"""Step 6: single agent vs router + evaluator (RAG) vs oracle rubric on HealthBench meta (PROTOCOL-V3-RAG-ABLATION).

Three stages, each plan-first (prints calls, cost upper bound, ledger; writes nothing without --authorized):

    # 1. sample fresh dev groups (label-blind) and route every conversation (1 router call per example)
    python scripts/run_rag_ablation_v3.py route --judge deepseek-pro --run-limit 30
    # 2. evaluator calls for all three arms, de-duplicated (identical requests are sent once); optional batches
    python scripts/run_rag_ablation_v3.py evaluate --run-dir reports/rag-ablation-v3-<...> --judge deepseek-pro \
        --parts 3 --part 1 --run-limit 30
    # 3. after every batch is sealed: join physician labels and score (no model call)
    python scripts/run_rag_ablation_v3.py score --run-dir reports/rag-ablation-v3-<...>

The paid route stage claims a once-only sentinel, so the sample cannot be redrawn after seeing results.
Physician labels are read only in the score stage, after all predictions are sealed.
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
from medical_jury.datasets.ordinal_v3 import assign_splits  # noqa: E402
from medical_jury.domain import CaseInput, digest  # noqa: E402
from medical_jury.evaluation.ablation_scoring import score_ablation  # noqa: E402
from medical_jury.evaluation.data import load_prepared, references_by_id  # noqa: E402
from medical_jury.evaluation.runner import (LiveWriter, StopController, budget_gate, ledger_status,  # noqa: E402
                                            run_planned, status_counts)
from medical_jury.evaluation.sampling import natural_sample, topic_of  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v3 import ablation as AB  # noqa: E402
from medical_jury.v3.evaluator import (DimensionEvaluator, build_messages_v3,  # noqa: E402
                                       mock_responder_v3)
from medical_jury.v3.router import (ROUTER_MAX_OUTPUT_TOKENS, ROUTER_PROMPT_VERSION, SituationRouter,  # noqa: E402
                                    build_router_messages, mock_router_responder)
from medical_jury.v3.schema import load_kb  # noqa: E402
from medical_jury.v3.used_groups import used_dev_groups_v3  # noqa: E402

KB = 'kb/v3'
PROMPT_VERSION = 'v3-dimension-2'
EVAL_MAX_TOKENS = 800
DEFAULT_GROUPS = 150
DEFAULT_SEED = 20261026
DATA_FLOW = ('HealthBench (MIT) public dev conversations: conversation (router) or conversation + completion + '
             'criterion (evaluator) are sent to the judge provider; physician labels are never sent and are read '
             'only in the score stage.')
STOP_RULES = ('HTTP 401/402/403/429 or budget -> immediate stop; 3 consecutive failures; >5 failures in the first '
              '50 calls; no automatic retry; rest recorded NotRun.')


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _write_json(path: Path, obj) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _seal(path: Path, records: list[dict], controller: StopController, interrupted: bool) -> str:
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(path.with_name(path.stem + '-seal.json'),
                {'sha256': sha, 'records': len(records), 'status_counts': status_counts(records),
                 'stop': controller.summary(), 'interrupted': interrupted, 'sealed_at': datetime.now(timezone.utc).isoformat()})
    return sha


def _verify_seal(path: Path) -> dict:
    seal = json.loads(path.with_name(path.stem + '-seal.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != seal['sha256']:
        raise SystemExit(f'{path} does not match its seal.')
    return seal


def _conversation(row: dict) -> tuple[tuple[str, str], ...]:
    return tuple((m['role'], m['content']) for m in row['case']['context'])


def _backend(args, root, store, max_tokens, responder):
    from run_review_v3 import make_backend          # same budgeted DeepSeek backend as the review CLI
    return make_backend(args.judge, root, store, args.run_limit, max_tokens, responder=responder)


def _upper(settings, paid, max_tokens, requests) -> float:
    if not paid:
        return 0.0
    b = Budget(replace(settings, max_output_tokens=max_tokens), None)
    return round(sum(b.estimate(m) for m in requests), 4)


def _claim(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_json(path, payload)
    except FileExistsError:
        raise SystemExit(f'{path} exists: this stage was already started and may run only once.') from None


# ---------------------------------------------------------------- stage 1: sample + route

def stage_route(args, root: Path) -> int:
    kb = load_kb(root / KB)
    excluded, used_runs = used_dev_groups_v3(root)
    selected, manifest = natural_sample(load_prepared(root, 'dev'), args.n_groups, args.seed, 'dev',
                                        exclude_groups=excluded)
    for row in selected:                              # every target must resolve before any call
        AB.resolve(row['category'], kb)
    manifest['excluded_from_runs'] = used_runs
    store = Store(root)
    backend, paid, settings = _backend(args, root, store, ROUTER_MAX_OUTPUT_TOKENS, mock_router_responder)
    requests = [build_router_messages(_conversation(r), kb.situations) for r in selected]
    stamp = _stamp()
    job = f'RAGAB3-route-{stamp}-{uuid4().hex[:6]}'
    plan = {'job_id': job, 'stage': 'route', 'ablation': AB.ABLATION_VERSION, 'items': len(selected),
            'groups': len({r['case']['group_id'] for r in selected}), 'excluded_groups': len(excluded),
            'sample_manifest_sha256': manifest['manifest_sha256'], 'planned_calls': len(requests),
            'judge': args.judge, 'router_prompt': ROUTER_PROMPT_VERSION, 'max_output_tokens': ROUTER_MAX_OUTPUT_TOKENS,
            'kb_sha256': kb.sha256(), 'threads': args.threads,
            'cost_upper_bound_cny': _upper(settings, paid, ROUTER_MAX_OUTPUT_TOKENS, requests),
            'run_limit_cny': args.run_limit, 'ledger': ledger_status(root) if paid else None,
            'label': 'DEV ablation (fresh, unseen dev groups; run once)', 'data_flow': DATA_FLOW,
            'stop_conditions': STOP_RULES}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    if paid:                                          # one paid sample only: no redraw after seeing results
        _claim(root / 'reports' / 'rag-ablation-v3-sentinels' / 'route.json', {'job_id': job, 'started_at': stamp})
    out = root / 'reports' / f'rag-ablation-v3-{stamp}-{job[-6:]}'
    out.mkdir(parents=True)
    _write_json(out / 'sample-manifest.json', manifest)
    _write_json(out / 'plan-route.json', plan)
    controller = StopController()
    router = SituationRouter(backend, store, stop=controller.event)
    by_id = {r['example_id']: r for r in selected}

    def work(row):
        rec = router.route(_conversation(row), kb.situations, job)
        return {'example_id': row['example_id'], 'group_id': row['case']['group_id'], **rec}
    writer = LiveWriter(out / 'routes.jsonl')
    try:
        records, interrupted = run_planned(selected, work, id_of=lambda r: r['example_id'], writer=writer,
                                           controller=controller, threads=args.threads)
    finally:
        writer.close()
        if hasattr(backend, 'close'):
            backend.close()
    for rec, row in zip(records, selected):
        rec.setdefault('group_id', by_id[row['example_id']]['case']['group_id'])
    _seal(out / 'routes.jsonl', records, controller, interrupted)
    print(json.dumps({'run_dir': str(out), 'status_counts': status_counts(records)}, indent=2))
    return 0


# ---------------------------------------------------------------- stage 2: evaluator calls

def build_requests(root: Path, run_dir: Path, kb):
    """Items with their per-arm request ids, and the de-duplicated requests (no labels involved)."""
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
    _verify_seal(run_dir / 'routes.jsonl')
    routes = {r['example_id']: r for r in _read_jsonl(run_dir / 'routes.jsonl')}
    rows = {r['example_id']: r for r in load_prepared(root, 'dev')}
    items, requests = [], {}
    for eid in manifest['example_ids']:
        row = rows[eid]
        case = CaseInput.model_validate(row['case'])
        target = AB.resolve(row['category'], kb)
        route = routes.get(eid)
        chosen = AB.rag_selection(route, target)
        entry = {'example_id': eid, 'group_id': case.group_id, 'category': row['category'],
                 'topic': topic_of(row['category']), 'arms': {},
                 'rag': {'router_status': (route or {}).get('status', 'NotRun'), 'selected': chosen,
                         'gold': target.gold_situation}}
        for arm in AB.ARMS:
            check = AB.check_for_arm(arm, kb, target, route)
            if check is None:
                entry['arms'][arm] = None
                continue
            messages = build_messages_v3(case, AB.CRITERION_DIMENSION, [check], [], PROMPT_VERSION)
            key = digest(messages)
            entry['arms'][arm] = key
            requests.setdefault(key, {'request_id': key, 'example_id': eid, 'group_id': case.group_id,
                                      'case': case, 'check': check, 'messages': messages, 'arms': []})
            requests[key]['arms'].append(arm)
        entry['rag']['assessed'] = entry['arms']['rag'] is not None
        items.append(entry)
    return items, list(requests.values())


def _part(requests: list[dict], part: int, parts: int) -> list[dict]:
    if parts < 1 or not 1 <= part <= parts:
        raise SystemExit('need 1 <= --part <= --parts')
    if parts == 1:
        return requests
    names = {f'part{k}': 1 / parts for k in range(1, parts)}
    names[f'part{parts}'] = 1 - sum(names.values())
    assigned = assign_splits([{'group_id': r['group_id']} for r in requests], names, f'rag-ablation-v3-parts:{parts}')
    return [r for r in requests if assigned[r['group_id']] == f'part{part}']


def stage_evaluate(args, root: Path) -> int:
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    kb = load_kb(root / KB)
    items, requests = build_requests(root, run_dir, kb)
    batch = _part(requests, args.part, args.parts)
    store = Store(root)
    backend, paid, settings = _backend(args, root, store, EVAL_MAX_TOKENS, mock_responder_v3)
    stamp = _stamp()
    job = f'RAGAB3-eval-{stamp}-{uuid4().hex[:6]}'
    arm_calls = {a: sum(a in r['arms'] for r in batch) for a in AB.ARMS}
    plan = {'job_id': job, 'stage': 'evaluate', 'run_dir': run_dir.name, 'part': {'part': args.part, 'parts': args.parts},
            'items_in_run': len(items), 'unique_requests_in_run': len(requests), 'planned_calls': len(batch),
            'requests_by_arm_in_part': arm_calls, 'rag_not_assessed_items': sum(not i['rag']['assessed'] for i in items),
            'judge': args.judge, 'prompt_version': PROMPT_VERSION, 'max_output_tokens': EVAL_MAX_TOKENS,
            'kb_sha256': kb.sha256(), 'request_ids_sha256': digest(sorted(r['request_id'] for r in batch)),
            'threads': args.threads, 'cost_upper_bound_cny': _upper(settings, paid, EVAL_MAX_TOKENS,
                                                                    [r['messages'] for r in batch]),
            'run_limit_cny': args.run_limit, 'ledger': ledger_status(root) if paid else None,
            'data_flow': DATA_FLOW, 'stop_conditions': STOP_RULES}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    layout = run_dir / 'evaluate-layout.json'
    try:
        _write_json(layout, {'parts': args.parts, 'requests_sha256': digest(sorted(r['request_id'] for r in requests))})
    except FileExistsError:
        fixed = json.loads(layout.read_text(encoding='utf-8'))
        if fixed['parts'] != args.parts:
            raise SystemExit(f'This run is already split into {fixed["parts"]} part(s); it cannot be re-partitioned.')
    out = run_dir / f'evaluate-part{args.part}of{args.parts}'
    try:
        out.mkdir()
    except FileExistsError:
        raise SystemExit(f'{out.name} was already run; each part may run only once.') from None
    _write_json(out / 'plan.json', plan)
    controller = StopController()
    evaluator = DimensionEvaluator(backend, store, max_tokens=EVAL_MAX_TOKENS, stop=controller.event,
                                   prompt_version=PROMPT_VERSION)

    def work(req):
        rec = evaluator.evaluate(req['case'], AB.CRITERION_DIMENSION, [req['check']], [], job)
        if rec.get('messages_sha256') != req['request_id']:
            raise RuntimeError('request id does not match the messages sent')
        keep = ('status', 'failure_code', 'checks', 'model_score', 'confidence', 'usage', 'finish_reason',
                'latency_s', 'raw_sha256', 'raw_chars', 'analysis', 'score_check_conflicts')
        # run_planned stores the unit id (= request id) in 'example_id'; the HealthBench example id is kept here
        return {'request_id': req['request_id'], 'item_example_id': req['example_id'], 'group_id': req['group_id'],
                'arms': req['arms'], **{k: rec.get(k) for k in keep}}
    writer = LiveWriter(out / 'predictions.jsonl')
    try:
        records, interrupted = run_planned(batch, work, id_of=lambda r: r['request_id'], writer=writer,
                                           controller=controller, threads=args.threads)
    finally:
        writer.close()
        if hasattr(backend, 'close'):
            backend.close()
    for rec, req in zip(records, batch):
        rec['request_id'] = req['request_id']
    _seal(out / 'predictions.jsonl', records, controller, interrupted)
    print(json.dumps({'out': str(out), 'status_counts': status_counts(records)}, indent=2))
    return 0


# ---------------------------------------------------------------- stage 3: labels + metrics

def stage_score(args, root: Path) -> int:
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    kb = load_kb(root / KB)
    items, requests = build_requests(root, run_dir, kb)
    layout = json.loads((run_dir / 'evaluate-layout.json').read_text(encoding='utf-8'))
    if layout['requests_sha256'] != digest(sorted(r['request_id'] for r in requests)):
        raise SystemExit('The requests no longer match the ones the evaluate stage was planned with.')
    parts = layout['parts']
    results = {}
    for k in range(1, parts + 1):
        path = run_dir / f'evaluate-part{k}of{parts}' / 'predictions.jsonl'
        if not path.exists():
            raise SystemExit(f'Part {k}/{parts} has not been run.')
        _verify_seal(path)
        for rec in _read_jsonl(path):
            if rec['request_id'] in results:
                raise SystemExit(f'Request {rec["request_id"][:12]} appears twice.')
            results[rec['request_id']] = rec
    if set(results) != {r['request_id'] for r in requests}:
        raise SystemExit('The parts do not cover the planned requests exactly.')
    refs = references_by_id(root, {'dev'})          # labels are read only here, after every part is sealed
    rows = []
    for item in items:
        ref = refs[item['example_id']]
        verdicts = {}
        for arm, key in item['arms'].items():
            verdicts[arm] = 'met' if key is None else AB.verdict(results[key])   # not assessed -> met
        rows.append({'example_id': item['example_id'], 'group_id': item['group_id'], 'topic': item['topic'],
                     'verdicts': verdicts, 'rag': item['rag'], 'labels': list(ref['labels']),
                     'physician_ids': list(ref.get('physician_ids', []))})
    metrics = score_ablation(rows, n_boot=args.bootstrap)
    usage = [r.get('usage') or {} for r in results.values()]
    metrics.update({'run_dir': run_dir.name, 'ablation': AB.ABLATION_VERSION, 'prompt_version': PROMPT_VERSION,
                    'router_prompt': ROUTER_PROMPT_VERSION, 'kb_sha256': kb.sha256(), 'parts': parts,
                    'evaluator_calls': len(results), 'evaluator_status_counts': status_counts(results.values()),
                    'evaluator_tokens': {'prompt': sum(u.get('prompt_tokens', 0) for u in usage),
                                         'completion': sum(u.get('completion_tokens', 0) for u in usage),
                                         'max_completion': max((u.get('completion_tokens', 0) for u in usage), default=0)},
                    'label': 'DEV ablation on fresh dev groups (development evidence; not a clinical claim)'})
    out = run_dir / f'score-{_stamp()}'
    out.mkdir()
    _write_json(out / 'metrics.json', metrics)
    with (out / 'items.jsonl').open('x', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ('example_id', 'group_id', 'topic', 'verdicts', 'rag', 'labels')},
                               ensure_ascii=False) + '\n')
    print(json.dumps({'out': str(out), 'primary': metrics['primary'], 'primary_reading': metrics['primary_reading'],
                      'comparisons': {k: {x: v[x] for x in ('items', 'kappa_a', 'kappa_b', 'delta_kappa', 'delta_kappa_ci95')}
                                      for k, v in metrics['comparisons'].items()},
                      'router': metrics['router']}, indent=2, ensure_ascii=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('route', 'evaluate', 'score'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run-dir', type=Path, default=None)
    p.add_argument('--n-groups', type=int, default=DEFAULT_GROUPS)
    p.add_argument('--seed', type=int, default=DEFAULT_SEED)
    p.add_argument('--judge', default='mock')
    p.add_argument('--part', type=int, default=1)
    p.add_argument('--parts', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--bootstrap', type=int, default=2000)
    args = p.parse_args(argv)
    if args.stage != 'route' and args.run_dir is None:
        raise SystemExit(f'{args.stage} needs --run-dir (the folder written by the route stage).')
    return {'route': stage_route, 'evaluate': stage_evaluate, 'score': stage_score}[args.stage](args, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
