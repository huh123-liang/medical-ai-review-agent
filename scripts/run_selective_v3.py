"""Plan phase 1: selective evaluation with a guarantee on fresh HealthBench dev groups (PROTOCOL-V3-SELECTIVE).

Stages (plan-first; nothing is sent or written without --authorized for a paid judge):

    python scripts/run_selective_v3.py sample                       # free: draw fresh groups, split cal/test/pilot
    python scripts/run_selective_v3.py pilot --run-dir reports/selective-v3-<...> --judge deepseek-pro --run-limit 5
    python scripts/run_selective_v3.py run   --run-dir reports/selective-v3-<...> --judge deepseek-pro \
        --parts 4 --part 1 --run-limit 50
    python scripts/run_selective_v3.py score --run-dir reports/selective-v3-<...>   # labels read only here

Every item is judged K times against its ORACLE criterion (the KB criterion of the gold situation; same request as
the step-6 oracle arm): sample 0 at temperature 0 (the deployed setting), samples 1..K-1 at the sampling temperature.
The pilot (separate groups, no labels) checks that sampling changes the outputs at all; the pre-registered fallback
if it does not is the verbalized-confidence estimator only.
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
from medical_jury.evaluation import selective as S  # noqa: E402
from medical_jury.evaluation.data import load_prepared, references_by_id  # noqa: E402
from medical_jury.evaluation.runner import (IMMEDIATE_STOP, LiveWriter, StopController, budget_gate,  # noqa: E402
                                            ledger_status, run_planned, status_counts)
from medical_jury.evaluation.sampling import natural_sample, topic_of  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.v3 import ablation as AB  # noqa: E402
from medical_jury.v3.evaluator import DimensionEvaluator, build_messages_v3, mock_responder_v3  # noqa: E402
from medical_jury.v3.schema import load_kb  # noqa: E402
from medical_jury.v3.used_groups import used_dev_groups_v3  # noqa: E402

KB = 'kb/v3'
PROMPT_VERSION = 'v3-dimension-2'
EVAL_MAX_TOKENS = 800
VERSION = 'selective-v3-1'
DEFAULTS = {'groups': 400, 'pilot_groups': 10, 'k': 3, 'temperature': 0.7, 'seed': 20261027,
            'alphas': (0.05, 0.10), 'primary_alpha': 0.05, 'delta': 0.10, 'min_selected': 30}
# Protocol v2 (2026-10-27, PROTOCOL-V3-SELECTIVE section 10): a FIXED gate chosen on the v1 development data.
GATE = {'version': 'selective-v3-2', 'themes': ['communication', 'hedging'], 'k': 3, 'temperature': 0.7,
        'auto_if': 'theme in themes AND all K judgments completed AND all K verdicts agree', 'alpha': 0.05,
        'delta': 0.10, 'error': 'consensus', 'seed': 20261028, 'groups': 250}
DATA_FLOW = ('HealthBench (MIT) public dev: conversation + completion + criterion are sent to the judge provider; '
             'physician labels are never sent and are read only in the score stage.')
STOP_RULES = ('HTTP 401/402/403/429 or budget -> immediate stop; 3 consecutive failures; >5 failures in the first '
              '50 calls; no automatic retry; rest recorded NotRun.')


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _write_json(path: Path, obj):
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _seal(path: Path, records, controller, interrupted):
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(path.with_name(path.stem + '-seal.json'),
                {'sha256': sha, 'records': len(records), 'status_counts': status_counts(records),
                 'stop': controller.summary(), 'interrupted': interrupted,
                 'sealed_at': datetime.now(timezone.utc).isoformat()})


def _verify_seal(path: Path):
    seal = json.loads(path.with_name(path.stem + '-seal.json').read_text(encoding='utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest() != seal['sha256']:
        raise SystemExit(f'{path} does not match its seal.')
    return seal


def _sample_index(rec: dict) -> int:
    """Sample number from the unit id ('<item>#<s>'); NotRun stubs written by run_planned carry no 'sample'."""
    return int(str(rec.get('example_id', '')).rsplit('#', 1)[-1]) if '#' in str(rec.get('example_id', '')) \
        else int(rec.get('sample', 0))


def _interrupted(rec: dict) -> bool:
    """Not answered because of a technical stop (provider quota/auth/rate limit, budget, run stop) - not a model
    output failure. Only these may be completed later (amendment v2.1)."""
    return rec.get('status') == 'NotRun' or (rec.get('status') == 'Failed' and rec.get('failure_code') in IMMEDIATE_STOP)


def _run_dir(args, root):
    if args.run_dir is None:
        raise SystemExit(f'{args.stage} needs --run-dir (the folder written by the sample stage).')
    return args.run_dir if args.run_dir.is_absolute() else root / args.run_dir


# ------------------------------------------------------------------ stage: sample (free, label-blind)

def stage_sample(args, root: Path) -> int:
    kb = load_kb(root / KB)
    excluded, sources = used_dev_groups_v3(root)
    rows = load_prepared(root, 'dev')
    if args.gate:                                  # v2: only the gate's themes (chosen by category, label-blind)
        rows = [r for r in rows if topic_of(r['category']) in GATE['themes']]
        args.pilot_groups, args.seed = 0, GATE['seed']
        if args.groups == DEFAULTS['groups']:
            args.groups = GATE['groups']
    total = args.groups + args.pilot_groups
    selected, manifest = natural_sample(rows, total, args.seed, 'dev', exclude_groups=excluded)
    for row in selected:
        AB.resolve(row['category'], kb)
    pool = [{'group_id': r['case']['group_id']} for r in selected]
    if args.gate:
        split = {g['group_id']: 'test' for g in pool}
    else:
        pilot_share = args.pilot_groups / total
        split = assign_splits(pool, {'pilot': pilot_share, 'calibration': (1 - pilot_share) / 2,
                                     'test': 1 - pilot_share - (1 - pilot_share) / 2}, f'{VERSION}:split:{args.seed}')
    manifest.update({'excluded_from': sources, 'split_of_group': split,
                     'version': GATE['version'] if args.gate else VERSION,
                     'split_counts': {s: sum(v == s for v in split.values()) for s in ('pilot', 'calibration', 'test')}})
    if args.gate:
        all_groups = {r['case']['group_id'] for r in load_prepared(root, 'dev')}
        gate_groups = {r['case']['group_id'] for r in rows}
        manifest.update({'policy': GATE, 'fixed_k': GATE['k'],
                         'gate_share_of_dev_groups': len(gate_groups) / len(all_groups)})
    stamp = _stamp()
    out = root / 'reports' / f'selective-v3-{stamp}-{uuid4().hex[:6]}'
    out.mkdir(parents=True)
    _write_json(out / 'sample-manifest.json', manifest)
    print(json.dumps({'run_dir': str(out), 'version': manifest['version'], 'items': len(selected),
                      'excluded_groups': len(excluded), 'split_counts': manifest['split_counts'],
                      'themes': GATE['themes'] if args.gate else 'all',
                      'manifest_sha256': manifest['manifest_sha256']}, indent=2))
    return 0


def _units(root: Path, run_dir: Path, kb, splits: set[str], k: int):
    """(item, sample index) work units for the requested splits; labels are never touched."""
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
    rows = {r['example_id']: r for r in load_prepared(root, 'dev')}
    units = []
    for eid in manifest['example_ids']:
        row = rows[eid]
        split = manifest['split_of_group'][row['case']['group_id']]
        if split not in splits:
            continue
        case = CaseInput.model_validate(row['case'])
        target = AB.resolve(row['category'], kb)
        check = AB.check_for_arm('oracle', kb, target)
        messages = build_messages_v3(case, AB.CRITERION_DIMENSION, [check], [], PROMPT_VERSION)
        for s in range(k):
            units.append({'unit_id': f'{eid}#{s}', 'item_id': eid, 'group_id': case.group_id, 'split': split,
                          'sample': s, 'case': case, 'check': check, 'messages': messages,
                          'messages_sha256': digest(messages)})
    return manifest, units


def _part(units, part, parts):
    if parts < 1 or not 1 <= part <= parts:
        raise SystemExit('need 1 <= --part <= --parts')
    if parts == 1:
        return units
    names = {f'part{i}': 1 / parts for i in range(1, parts)}
    names[f'part{parts}'] = 1 - sum(names.values())
    assigned = assign_splits([{'group_id': u['group_id']} for u in units], names, f'{VERSION}:parts:{parts}')
    return [u for u in units if assigned[u['group_id']] == f'part{part}']


def _execute(args, root, run_dir, units, label, out, before_calls=None):
    from run_review_v3 import make_backend
    store = Store(root)
    b0, paid, settings = make_backend(args.judge, root, store, args.run_limit, EVAL_MAX_TOKENS, mock_responder_v3, 0.0)
    bt, _, _ = make_backend(args.judge, root, store, args.run_limit, EVAL_MAX_TOKENS, mock_responder_v3,
                            args.temperature)
    upper = 0.0
    if paid:
        est = Budget(replace(settings, max_output_tokens=EVAL_MAX_TOKENS), None)
        upper = round(sum(est.estimate(u['messages']) for u in units), 4)
    stamp = _stamp()
    job = f'SELV3-{label}-{stamp}-{uuid4().hex[:6]}'
    run_version = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8')).get('version', VERSION)
    plan = {'job_id': job, 'stage': label, 'version': run_version, 'run_dir': run_dir.name, 'items': len({u['item_id'] for u in units}),
            'planned_calls': len(units), 'samples_per_item': args.k, 'temperature_sample0': 0.0,
            'temperature_others': args.temperature, 'judge': args.judge, 'prompt_version': PROMPT_VERSION,
            'criterion': 'oracle (KB criterion of the gold situation)', 'max_output_tokens': EVAL_MAX_TOKENS,
            'units_sha256': digest(sorted(u['unit_id'] for u in units)), 'threads': args.threads,
            'cost_upper_bound_cny': upper, 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None, 'data_flow': DATA_FLOW, 'stop_conditions': STOP_RULES}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0, None
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2, None
    if before_calls:
        before_calls(paid, job, stamp)
    try:
        out.mkdir()
    except FileExistsError:
        raise SystemExit(f'{out.name} was already run; it may run only once.') from None
    _write_json(out / 'plan.json', plan)
    controller = StopController()
    ev0 = DimensionEvaluator(b0, store, max_tokens=EVAL_MAX_TOKENS, stop=controller.event, prompt_version=PROMPT_VERSION)
    evt = DimensionEvaluator(bt, store, max_tokens=EVAL_MAX_TOKENS, stop=controller.event, prompt_version=PROMPT_VERSION)

    def work(u):
        rec = (ev0 if u['sample'] == 0 else evt).evaluate(u['case'], AB.CRITERION_DIMENSION, [u['check']], [], job)
        if rec.get('messages_sha256') != u['messages_sha256']:
            raise RuntimeError('messages changed between plan and call')
        return {'unit_id': u['unit_id'], 'item_id': u['item_id'], 'group_id': u['group_id'], 'split': u['split'],
                'sample': u['sample'], 'temperature': 0.0 if u['sample'] == 0 else args.temperature,
                'verdict': AB.verdict(rec), **{k: rec.get(k) for k in (
                    'status', 'failure_code', 'confidence', 'usage', 'finish_reason', 'raw_sha256', 'raw_chars',
                    'analysis')}}
    writer = LiveWriter(out / 'predictions.jsonl')
    try:
        records, interrupted = run_planned(units, work, id_of=lambda u: u['unit_id'], writer=writer,
                                           controller=controller, threads=args.threads)
    finally:
        writer.close()
        for b in (b0, bt):
            if hasattr(b, 'close'):
                b.close()
    _seal(out / 'predictions.jsonl', records, controller, interrupted)
    print(json.dumps({'out': str(out), 'status_counts': status_counts(records)}, indent=2))
    return 0, records


# ------------------------------------------------------------------ stage: pilot (label-free check of sampling)

def stage_pilot(args, root: Path) -> int:
    run_dir = _run_dir(args, root)
    _, units = _units(root, run_dir, load_kb(root / KB), {'pilot'}, args.k)

    def claim(paid, job, stamp):              # one paid selective sample only: no redraw after seeing anything
        if paid:
            path = root / 'reports' / 'selective-v3-sentinels' / 'pilot.json'
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _write_json(path, {'job_id': job, 'run_dir': run_dir.name, 'started_at': stamp})
            except FileExistsError:
                raise SystemExit(f'{path} exists: the paid selective pilot may run only once.') from None
    code, records = _execute(args, root, run_dir, units, 'pilot', run_dir / 'pilot', claim)
    if records is None:
        return code
    by_item = {}
    for r in records:
        by_item.setdefault(r.get('item_id'), []).append(r)
    varied = sum(len({r.get('raw_sha256') for r in rs if r.get('status') == 'Completed'}) > 1 for rs in by_item.values())
    verdict_varied = sum(len({r.get('verdict') for r in rs if r.get('verdict')}) > 1 for rs in by_item.values())
    decision = {'items': len(by_item), 'items_with_varied_outputs': varied, 'items_with_varied_verdicts': verdict_varied,
                'sampling_effective': varied >= 0.2 * len(by_item),
                'rule': 'pre-registered: sampling counts as effective if outputs differ across samples for >= 20% of '
                        'pilot items; otherwise the primary estimator falls back to verbalized confidence (K=1).'}
    _write_json(run_dir / 'pilot' / 'decision.json', decision)
    print(json.dumps(decision, indent=2))
    return 0


# ------------------------------------------------------------------ stage: run (calibration + test items)

def stage_run(args, root: Path) -> int:
    run_dir = _run_dir(args, root)
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('fixed_k'):                    # v2: K and temperature fixed by the protocol (no pilot)
        k = manifest['fixed_k']
        args.temperature = manifest['policy']['temperature']
    else:
        decision_path = run_dir / 'pilot' / 'decision.json'
        if not decision_path.exists():
            raise SystemExit('Run the pilot first (its decision fixes K).')
        decision = json.loads(decision_path.read_text(encoding='utf-8'))
        k = args.k if decision['sampling_effective'] else 1
        if k != args.k:
            print('Pilot found sampling ineffective: K reduced to 1 (pre-registered fallback).')
    args.k = k
    _, units = _units(root, run_dir, load_kb(root / KB), {'calibration', 'test'}, k)
    layout = run_dir / 'run-layout.json'
    if layout.exists():
        fixed = json.loads(layout.read_text(encoding='utf-8'))
        if fixed['parts'] != args.parts or fixed['k'] != k:
            raise SystemExit(f'This run is fixed at {fixed["parts"]} part(s) and K={fixed["k"]}; it cannot be changed.')

    def fix_layout(paid, job, stamp):
        if not layout.exists():
            if paid and manifest.get('policy'):    # one paid run of each fixed-gate protocol version
                path = root / 'reports' / 'selective-v3-sentinels' / f"run-{manifest['version']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    _write_json(path, {'job_id': job, 'run_dir': run_dir.name, 'started_at': stamp})
                except FileExistsError:
                    raise SystemExit(f'{path} exists: {manifest["version"]} may run only once.') from None
            _write_json(layout, {'parts': args.parts, 'k': k, 'units_sha256': digest(sorted(u['unit_id'] for u in units))})
    batch = _part(units, args.part, args.parts)
    code, _ = _execute(args, root, run_dir, batch, f'run-part{args.part}of{args.parts}',
                       run_dir / f'run-part{args.part}of{args.parts}', fix_layout)
    return code


# ------------------------------------------------------------------ stage: complete (amendment v2.1)

def _load_parts(run_dir: Path, units):
    layout = json.loads((run_dir / 'run-layout.json').read_text(encoding='utf-8'))
    parts = layout['parts']
    results = {}
    for i in range(1, parts + 1):
        path = run_dir / f'run-part{i}of{parts}' / 'predictions.jsonl'
        if not path.exists():
            raise SystemExit(f'Part {i}/{parts} has not been run.')
        _verify_seal(path)
        for rec in _read_jsonl(path):
            results[rec['example_id']] = rec          # run_planned keys records by unit id
    if set(results) != {u['unit_id'] for u in units}:
        raise SystemExit('The parts do not cover the planned units exactly.')
    return layout, results


def stage_complete(args, root: Path) -> int:
    """Re-send ONLY the units that a technical stop left unanswered, once (PROTOCOL-V3-SELECTIVE 10.9)."""
    run_dir = _run_dir(args, root)
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('policy'):
        raise SystemExit('Completion is defined only for the fixed-gate protocol (v2).')
    layout = json.loads((run_dir / 'run-layout.json').read_text(encoding='utf-8'))
    k = layout['k']
    args.k, args.temperature = k, manifest['policy']['temperature']
    _, units = _units(root, run_dir, load_kb(root / KB), {'calibration', 'test'}, k)
    _, results = _load_parts(run_dir, units)
    todo = [u for u in units if _interrupted(results[u['unit_id']])]
    if not todo:
        print('Nothing to complete: no unit was stopped by a technical interruption.')
        return 0

    def claim(paid, job, stamp):
        if paid:
            path = root / 'reports' / 'selective-v3-sentinels' / f"completion-{manifest['version']}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                _write_json(path, {'job_id': job, 'run_dir': run_dir.name, 'units': len(todo), 'started_at': stamp})
            except FileExistsError:
                raise SystemExit(f'{path} exists: the completion may run only once.') from None
    code, _ = _execute(args, root, run_dir, todo, 'completion', run_dir / 'completion', claim)
    return code


# ------------------------------------------------------------------ stage: score (labels read only here)

def stage_score(args, root: Path) -> int:
    run_dir = _run_dir(args, root)
    layout = json.loads((run_dir / 'run-layout.json').read_text(encoding='utf-8'))
    k = layout['k']
    _, units = _units(root, run_dir, load_kb(root / KB), {'calibration', 'test'}, k)
    if digest(sorted(u['unit_id'] for u in units)) != layout['units_sha256']:
        raise SystemExit('Planned units changed since the run started.')
    _, results = _load_parts(run_dir, units)
    refs = references_by_id(root, {'dev'})
    manifest = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('policy'):
        completed = None
        comp = run_dir / 'completion' / 'predictions.jsonl'
        if comp.exists():
            _verify_seal(comp)
            completed = dict(results)
            for rec in _read_jsonl(comp):
                if rec['example_id'] not in results or not _interrupted(results[rec['example_id']]):
                    raise SystemExit('The completion contains a unit that was not interrupted.')
                completed[rec['example_id']] = rec
        return _score_gate(args, run_dir, manifest, units, results, refs, k, completed)
    items = {}
    for u in units:
        items.setdefault(u['item_id'], {'item_id': u['item_id'], 'group_id': u['group_id'], 'split': u['split'],
                                        'samples': []})['samples'].append(results[u['unit_id']])
    rows = {'verbalized': [], 'self_consistency': []}
    item_out = []
    for it in items.values():
        ref = refs[it['item_id']]
        labels = list(ref['labels'])
        samples = sorted(it['samples'], key=_sample_index)
        s0 = samples[0]
        sc = S.vote([{'verdict': r.get('verdict'), 'confidence': float(r.get('confidence') or 0.0)} for r in samples])
        est = {'verbalized': None if s0.get('verdict') is None else
               {'verdict': s0['verdict'], 'confidence': float(s0.get('confidence') or 0.0)},
               'self_consistency': sc}
        for name, e in est.items():
            if e is None:                               # failed call: never auto-decided (escalated)
                rows[name].append({'split': it['split'], 'confidence': -1.0, 'error': 0, 'pairwise': 0.0,
                                   'failed': True, 'group_id': it['group_id']})
                continue
            met = e['verdict'] == 'met'
            rows[name].append({'split': it['split'], 'confidence': e['confidence'],
                               'error': S.consensus_error(met, labels), 'pairwise': S.pairwise_error(met, labels),
                               'failed': False, 'group_id': it['group_id']})
        item_out.append({'item_id': it['item_id'], 'split': it['split'], 'topic': topic_of(ref.get('category', '')),
                         'labels': labels, 'verbalized': est['verbalized'], 'self_consistency': sc})
    metrics = {'version': VERSION, 'k': k, 'delta': args.delta, 'alphas': list(args.alphas),
               'primary': {'estimator': 'self_consistency' if k > 1 else 'verbalized', 'alpha': args.primary_alpha,
                           'error': 'consensus'}, 'estimators': {}}
    for name, rr in rows.items():
        if name == 'self_consistency' and k == 1:
            continue
        cal = [r for r in rr if r['split'] == 'calibration']
        test = [r for r in rr if r['split'] == 'test']
        done = [r for r in rr if not r['failed']]
        block = {'items': {'calibration': len(cal), 'test': len(test)}, 'failed': sum(r['failed'] for r in rr),
                 'auroc_no_consensus_error': S.auroc([r['confidence'] for r in done], [r['error'] == 0 for r in done]),
                 'risk_coverage_test': S.risk_coverage(test), 'by_alpha': {}}
        for a in args.alphas:
            c = S.calibrate(cal, a, args.delta, args.min_selected)
            t = S.evaluate(test, c['lambda'], args.delta)
            sel = [r for r in test if c['lambda'] is not None and r['confidence'] >= c['lambda']]
            t['pairwise_disagreement_selected'] = (sum(r['pairwise'] for r in sel) / len(sel)) if sel else None
            t['guarantee_held_on_test'] = None if t['risk'] is None else t['risk'] <= a
            block['by_alpha'][str(a)] = {'calibration': c, 'test': t}
        block['test_overall'] = {'risk_all_items': sum(r['error'] for r in test) / len(test) if test else None,
                                 'pairwise_all_items': sum(r['pairwise'] for r in test) / len(test) if test else None}
        metrics['estimators'][name] = block
    usage = [r.get('usage') or {} for r in results.values()]
    metrics['tokens'] = {'prompt': sum(u.get('prompt_tokens', 0) for u in usage),
                         'completion': sum(u.get('completion_tokens', 0) for u in usage)}
    metrics['status_counts'] = status_counts(results.values())
    out = run_dir / f'score-{_stamp()}'
    out.mkdir()
    _write_json(out / 'metrics.json', metrics)
    with (out / 'items.jsonl').open('x', encoding='utf-8') as f:
        for it in item_out:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    p = metrics['estimators'][metrics['primary']['estimator']]['by_alpha'][str(args.primary_alpha)]
    print(json.dumps({'out': str(out), 'primary': metrics['primary'], 'lambda': p['calibration']['lambda'],
                      'calibration_coverage': p['calibration']['calibration_coverage'], 'test': p['test']}, indent=2))
    return 0


def _gate_view(manifest, units, results, refs, k):
    policy = manifest['policy']
    items = {}
    for u in units:
        items.setdefault(u['item_id'], []).append(results[u['unit_id']])
    rows = []
    for item_id, samples in items.items():
        ref = refs[item_id]
        labels = list(ref['labels'])
        samples = sorted(samples, key=_sample_index)
        sc = S.vote([{'verdict': r.get('verdict'), 'confidence': float(r.get('confidence') or 0.0)} for r in samples])
        auto = bool(sc) and sc['completed'] == k and sc['vote_share'] == 1.0
        s0 = samples[0].get('verdict')
        rows.append({'item_id': item_id, 'theme': topic_of(ref.get('category', '')), 'labels': labels, 'auto': auto,
                     'verdict': sc['verdict'] if sc else None, 'vote_share': sc['vote_share'] if sc else None,
                     'completed_samples': sc['completed'] if sc else 0, 'verdict_sample0': s0,
                     'error': S.consensus_error(sc['verdict'] == 'met', labels) if sc else 0,
                     'pairwise': S.pairwise_error(sc['verdict'] == 'met', labels) if sc else None,
                     'error_sample0': S.consensus_error(s0 == 'met', labels) if s0 else 0,
                     'split_labels': len(set(labels)) > 1,
                     'interrupted_samples': sum(_interrupted(r) for r in samples)})
    auto = [r for r in rows if r['auto']]
    k_err, n_auto = sum(r['error'] for r in auto), len(auto)
    ucb = S.clopper_pearson_upper(k_err, n_auto, policy['delta']) if n_auto else None
    s0rows = [r for r in rows if r['verdict_sample0']]
    k0 = sum(r['error_sample0'] for r in s0rows)
    by_theme = {}
    for t in sorted({r['theme'] for r in rows}):
        sub = [r for r in rows if r['theme'] == t]
        sub_auto = [r for r in sub if r['auto']]
        by_theme[t] = {'items': len(sub), 'auto': len(sub_auto), 'errors_auto': sum(r['error'] for r in sub_auto),
                       'errors_sample0': sum(r['error_sample0'] for r in sub)}
    view = {'items': len(rows), 'items_with_interrupted_samples': sum(r['interrupted_samples'] > 0 for r in rows),
            'primary': {'auto_decided': n_auto, 'coverage_within_gate_themes': n_auto / len(rows) if rows else 0.0,
                        'estimated_coverage_all_dev_groups': (n_auto / len(rows)) * manifest['gate_share_of_dev_groups']
                        if rows else 0.0,
                        'errors': k_err, 'risk': (k_err / n_auto) if n_auto else None, 'risk_ucb': ucb,
                        'success': ucb is not None and ucb <= policy['alpha'],
                        'pairwise_disagreement_auto': (sum(r['pairwise'] for r in auto) / n_auto) if n_auto else None,
                        'auto_items_without_physician_consensus': sum(r['split_labels'] for r in auto)},
            'secondary_theme_only_sample0': {'items': len(s0rows), 'errors': k0,
                                             'risk': k0 / len(s0rows) if s0rows else None,
                                             'risk_ucb': S.clopper_pearson_upper(k0, len(s0rows), policy['delta'])
                                             if s0rows else None},
            'escalated': {'items': len(rows) - n_auto,
                          'errors_if_auto_decided': sum(r['error'] for r in rows if not r['auto'] and r['verdict'])},
            'by_theme': by_theme, 'status_counts': status_counts(results.values())}
    return view, rows


def _score_gate(args, run_dir, manifest, units, results, refs, k, completed=None) -> int:
    """Fixed-gate protocol (v2): no calibration; one binomial test of the pre-registered policy. With a v2.1
    completion, both the as-run view and the completed view are reported."""
    as_run, rows = _gate_view(manifest, units, results, refs, k)
    metrics = {'version': manifest['version'], 'policy': manifest['policy'], 'k': k, 'as_run': as_run}
    if completed is not None:
        metrics['completed'], rows = _gate_view(manifest, units, completed, refs, k)
        metrics['completion_note'] = ('Amendment v2.1: units stopped by the provider quota error were re-sent once; '
                                      'the decision to complete was taken AFTER the as-run result had been seen.')
    usage = [r.get('usage') or {} for r in (completed or results).values()]
    metrics['tokens'] = {'prompt': sum(u.get('prompt_tokens', 0) for u in usage),
                         'completion': sum(u.get('completion_tokens', 0) for u in usage)}
    out = run_dir / f'score-{_stamp()}'
    out.mkdir()
    _write_json(out / 'metrics.json', metrics)
    with (out / 'items.jsonl').open('x', encoding='utf-8') as f:
        for it in rows:
            f.write(json.dumps(it, ensure_ascii=False) + '\n')
    summary = {'out': str(out), 'version': manifest['version'],
               'as_run': {'primary': as_run['primary'], 'secondary_theme_only_sample0': as_run['secondary_theme_only_sample0']}}
    if completed is not None:
        summary['completed'] = {'primary': metrics['completed']['primary'],
                                'secondary_theme_only_sample0': metrics['completed']['secondary_theme_only_sample0']}
    print(json.dumps(summary, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('sample', 'pilot', 'run', 'complete', 'score'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run-dir', type=Path, default=None)
    p.add_argument('--groups', type=int, default=DEFAULTS['groups'])
    p.add_argument('--pilot-groups', type=int, default=DEFAULTS['pilot_groups'])
    p.add_argument('--seed', type=int, default=DEFAULTS['seed'])
    p.add_argument('--k', type=int, default=DEFAULTS['k'])
    p.add_argument('--temperature', type=float, default=DEFAULTS['temperature'])
    p.add_argument('--judge', default='mock')
    p.add_argument('--part', type=int, default=1)
    p.add_argument('--parts', type=int, default=1)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--delta', type=float, default=DEFAULTS['delta'])
    p.add_argument('--min-selected', type=int, default=DEFAULTS['min_selected'])
    p.add_argument('--gate', action='store_true',
                   help='sample stage only: protocol v2 fixed gate (themes communication + hedging, K=3, no pilot)')
    args = p.parse_args(argv)
    args.alphas, args.primary_alpha = DEFAULTS['alphas'], DEFAULTS['primary_alpha']
    if args.k < 1 or args.k > 9:
        raise SystemExit('--k must be 1..9')
    return {'sample': stage_sample, 'pilot': stage_pilot, 'run': stage_run, 'complete': stage_complete,
            'score': stage_score}[args.stage](args, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
