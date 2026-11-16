"""v4 error-detection agent on the known-answer testbed (PLAN-V4-ERROR-DETECTION, PROTOCOL-V3-TESTBED part 2).

    # pilot on the TRAIN split (development only; small, repeatable)
    python scripts/run_review_v4.py run --arm full --split train --style mas_transcript --max-cases 10 \\
        --judge deepseek-pro --run-limit 2 --prices-checked 2026-11-20
    # acceptance run on the TEST split: plan first, then --authorized; once per (arm, style, prompts)
    python scripts/run_review_v4.py run --arm full --split test --style mas_transcript --judge deepseek-pro \\
        --run-limit 40 --prices-checked 2026-11-20 [--authorized]
    # add another arm to the same run folder
    python scripts/run_review_v4.py run --run-dir reports/review-v4-<...> --arm model_only ...
    # score (free): gold is read only here, after the records are sealed
    python scripts/run_review_v4.py score --run-dir reports/review-v4-<...>

Only the synthetic case description and the written-up advice are sent (no structured truth, no error labels).
Without --authorized a paid stage prints the plan and writes nothing.
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
from medical_jury.domain import digest  # noqa: E402
from medical_jury.evaluation import v4_scoring as SC  # noqa: E402
from medical_jury.evaluation.runner import (LiveWriter, StopController, budget_gate, ledger_status,  # noqa: E402
                                            run_planned, status_counts)
from medical_jury.store import Store  # noqa: E402
from medical_jury.testbed import TESTBED_VERSION  # noqa: E402
from medical_jury.testbed.build import WRITER_PROMPT_VERSION  # noqa: E402
from medical_jury.testbed.errors import HELD_OUT_TYPES  # noqa: E402
from medical_jury.v4.check import check_messages  # noqa: E402
from medical_jury.v4.criteria import load_library  # noqa: E402
from medical_jury.v4.extract import extract_messages  # noqa: E402
from medical_jury.v4.mocks import oracle_responder  # noqa: E402
from medical_jury.v4.pipeline import CHECK_MAX_TOKENS, ReviewerV4, rules_only  # noqa: E402

VERSION = 'review-v4-eval-1'
SENTINELS = 'review-v4-sentinels'
PRIMARY_STYLE = 'mas_transcript'
DATA_FLOW = ('Synthetic testbed only (no real patient data): the case-description text and the local-model write-up '
             'are sent to the judge provider. Structured patient data, the reference plan and the error labels are '
             'never sent; labels are read only in the score stage, after the records are sealed.')
STOP_RULES = ('HTTP 401/402/403/429 or budget -> immediate stop; 3 consecutive failures; >5 failures in the first '
              '50 cases; no automatic retry; the rest is recorded NotRun.')


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def _write_json(path: Path, obj) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def testbed_dir(root: Path) -> Path:
    return root / 'data' / 'testbed' / TESTBED_VERSION


def primary_subset(case: dict) -> bool:
    """Cases that enter the primary endpoints: error-free, or one severe error of a type seen in training."""
    g = case['gold']
    return g is None or (isinstance(g, dict) and g['severity'] == 'severe' and g['error_id'] not in HELD_OUT_TYPES)


def load_inputs(root: Path, split: str, style: str, max_cases: int | None = None, subset: str = 'all',
                offset: int = 0) -> list[dict]:
    """Label-free units: case id, case-description text and the verified write-up. Gold is dropped here (it is
    consulted only to choose the pre-registered 'primary' subset; the model never sees it)."""
    cases = {c['case_id']: c for c in _read_jsonl(testbed_dir(root) / 'cases.jsonl')}
    path = testbed_dir(root) / f'renders-{style}-{WRITER_PROMPT_VERSION}.jsonl'
    if not path.exists():
        raise SystemExit(f'{path} not found: render the {style} write-ups first.')
    renders = {r['case_id']: r for r in _read_jsonl(path) if r.get('status') == 'ok'}
    units = [{'unit_id': f'{cid}@{style}', 'case_id': cid, 'style': style,
              'case_description': c['patient_summary'], 'document': renders[cid]['text'],
              'text_sha256': renders[cid]['text_sha256']}
             for cid, c in sorted(cases.items()) if c['split'] == split and cid in renders
             and (subset == 'all' or primary_subset(c))]
    units = units[offset:]
    return units[:max_cases] if max_cases else units


def plan_upper_bound(settings, units: list[dict], library: dict, arm: str) -> float:
    """Conservative: every check call is priced with ALL general items and ALL templates, at the check token cap."""
    est = Budget(replace(settings, max_output_tokens=CHECK_MAX_TOKENS), None)
    every = [{'id': i['id'], 'text': i['text']} for i in library['general']['items']] + \
            [{'id': t['id'], 'text': f"Potassium is 6.4 mmol/L. {t['text']}"} for t in library['disease']['templates']]
    total = 0.0
    for u in units:
        if arm == 'full':
            total += est.estimate(extract_messages(u['case_description'], u['document']))
        total += est.estimate(check_messages(u['case_description'], u['document'], every if arm == 'full' else [],
                                             code_checks={'checked': [{'medicine': 'x' * 15, 'against': ['potassium'] * 6}] * 8,
                                                          'flagged': []} if arm == 'full' else None))
    return round(total, 4)


EMPIRICAL_MARGIN = 1.3


def empirical_upper_bound(pilot_records: Path, n_units: int, arm: str, prompts: dict, settings) -> dict:
    """Cost bound from a sealed pilot run with the SAME prompts: n x the most expensive pilot case x 1.3, priced at
    the ledger's conservative rates. Used for the plan gate only with explicit owner approval (--pilot-run-dir); the
    ledger still enforces --run-limit on actual usage during the run."""
    _verify_seal(pilot_records)
    plan = json.loads((pilot_records.parent / 'plan.json').read_text(encoding='utf-8'))
    # a 'full' pilot also bounds 'model_only': that arm makes only the check call, with a shorter input
    if plan['prompts']['check'] != prompts['check'] or (plan['arm'] != arm and not (plan['arm'] == 'full'
                                                                                     and arm == 'model_only')):
        raise SystemExit('pilot run used different prompts or arm; its costs cannot bound this run')
    if arm == 'full' and plan['prompts'] != prompts:
        raise SystemExit('pilot run used different prompts; its costs cannot bound this run')
    per_case = []
    for r in _read_jsonl(pilot_records):
        c = sum((call.get('usage') or {}).get('prompt_tokens', 0) * settings.input_rate / 1e6 +
                (call.get('usage') or {}).get('completion_tokens', 0) * settings.output_rate / 1e6 for call in r['calls'])
        per_case.append(c)
    if len(per_case) < 20:
        raise SystemExit('pilot too small to bound costs (need at least 20 cases)')
    worst = max(per_case)
    return {'pilot': str(pilot_records.parent.parent.name), 'pilot_cases': len(per_case),
            'pilot_mean_case_cny': round(sum(per_case) / len(per_case), 5), 'pilot_max_case_cny': round(worst, 5),
            'margin': EMPIRICAL_MARGIN, 'bound_cny': round(n_units * worst * EMPIRICAL_MARGIN, 4),
            'expected_cny': round(n_units * sum(per_case) / len(per_case), 4)}


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
        raise SystemExit(f'{path} exists: this acceptance run was already started and may run only once.') from None


def _record_status(rec: dict) -> dict:
    """run_planned / StopController need one status per unit: the first failed or not-run call decides it."""
    bad = next((c for c in rec['calls'] if c['status'] != 'Completed'), None)
    if bad is None:
        return {'status': 'Completed'}
    return {'status': bad['status'], 'failure_code': bad.get('failure_code')}


def stage_run(args, root: Path) -> int:
    from run_review_v3 import make_backend          # same budgeted DeepSeek backend as the other CLIs
    library = load_library(root)
    units = load_inputs(root, args.split, args.style, args.max_cases, args.subset, args.offset)
    if not units:
        raise SystemExit('No rendered cases for this split and style.')
    store = Store(root)
    backend, paid, settings = make_backend(args.judge, root, store, args.run_limit, CHECK_MAX_TOKENS,
                                           responder=_offline_responder(root))
    if paid and args.prices_checked is None:
        raise SystemExit('Paid run: recheck the provider price list and pass --prices-checked YYYY-MM-DD.')
    if paid and args.split == 'test' and (args.max_cases or args.offset):
        raise SystemExit('The acceptance (test) run is all-or-nothing; --max-cases is for the train split.')
    reviewer = ReviewerV4(backend, library, store)
    if args.check_prompt_file:
        opt = json.loads(Path(args.check_prompt_file).read_text(encoding='utf-8'))
        if not opt.get('usable') or opt.get('offline_mock'):
            raise SystemExit(f'{args.check_prompt_file} is not usable (guard problems, unchanged, or offline mock).')
        if digest(opt['instruction']) != opt['sha256']:
            raise SystemExit('optimised instruction does not match its recorded hash')
        reviewer.check_system = opt['instruction']
        reviewer.check_version = f"v4-check-1+gepa-{opt['sha256'][:8]}"
    upper = plan_upper_bound(settings, units, library, args.arm) if paid else 0.0
    worst_case = upper
    stamp = _stamp()
    run_dir = (args.run_dir if args.run_dir.is_absolute() else root / args.run_dir) if args.run_dir else \
        root / 'reports' / f'review-v4-{stamp}-{uuid4().hex[:6]}'
    out = run_dir / f'{args.arm}-{args.split}-{args.style}'   # one folder per (arm, split, style)
    prompts = {'extract': reviewer._header('full')['extract_prompt'], 'check': reviewer._header('full')['check_prompt']}
    empirical = None
    if paid and args.pilot_run_dir:
        pr = args.pilot_run_dir if args.pilot_run_dir.is_absolute() else root / args.pilot_run_dir
        found = sorted(pr.glob('full-*/records.jsonl')) or sorted(pr.glob('*/records.jsonl'))
        if not found:
            raise SystemExit(f'no sealed records under {pr}')
        empirical = empirical_upper_bound(found[0], len(units), args.arm, prompts, settings)
        upper = empirical['bound_cny']
    job = f'REVIEWV4-{args.arm}-{args.split}-{stamp}-{uuid4().hex[:6]}'
    plan = {'job_id': job, 'version': VERSION, 'arm': args.arm, 'split': args.split, 'style': args.style,
            'subset': args.subset, 'offset': args.offset,
            'testbed': TESTBED_VERSION, 'writer_prompt': WRITER_PROMPT_VERSION, 'cases': len(units),
            'units_sha256': digest(sorted((u['unit_id'], u['text_sha256']) for u in units)),
            'planned_calls': len(units) * (2 if args.arm == 'full' else 1), 'prompts': prompts,
            'library_sha256': library['sha256'], 'judge': args.judge,
            'price_cny_per_mtok': {'in': settings.input_rate, 'out': settings.output_rate} if paid else None,
            'prices_checked': args.prices_checked, 'max_output_tokens': CHECK_MAX_TOKENS, 'threads': args.threads,
            'cost_upper_bound_cny': upper, 'cost_upper_bound_worst_case_cny': worst_case,
            'cost_bound_basis': 'empirical (pilot x 1.3, owner-approved)' if empirical else 'worst case (token caps)',
            'empirical_bound': empirical, 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None,
            'label': 'ACCEPTANCE (test split, once)' if args.split == 'test' else 'DEVELOPMENT (train split)',
            'data_flow': DATA_FLOW, 'stop_conditions': STOP_RULES, 'out': str(out)}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    if paid and not args.authorized:
        print('Plan only; no call made and nothing written. Re-run with --authorized after review.')
        return 0
    refusal = budget_gate(plan, paid, False)
    if refusal:
        print(refusal)
        return 2
    if paid and args.split == 'test':
        key = digest([args.arm, args.style, args.subset, prompts, library['sha256']])[:12]
        _claim(root / 'reports' / SENTINELS / f'test-{args.arm}-{args.style}-{key}.json',
               {'job_id': job, 'started_at': stamp, 'out': str(out)})
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise SystemExit(f'{out} was already run.') from None
    _write_json(out / 'plan.json', plan)
    controller = StopController()
    reviewer.stop = controller.event
    review = reviewer.review if args.arm == 'full' else reviewer.review_model_only

    def work(u):
        rec = review(u['case_description'], u['document'], f"{job}:{u['unit_id']}")
        return {'unit_id': u['unit_id'], 'case_id': u['case_id'], 'style': u['style'], **_record_status(rec), **rec}
    writer = LiveWriter(out / 'records.jsonl')
    try:
        records, interrupted = run_planned(units, work, id_of=lambda u: u['unit_id'], writer=writer,
                                           controller=controller, threads=args.threads)
    finally:
        writer.close()
        if hasattr(backend, 'close'):
            backend.close()
    _seal(out / 'records.jsonl', records, controller, interrupted)
    print(json.dumps({'out': str(out), 'status_counts': status_counts(records), 'interrupted': interrupted,
                      'ledger_after': ledger_status(root) if paid else None}, indent=2))
    return 0


def _offline_responder(root: Path):
    """Mock judge: answers extraction from the structured truth (plumbing only, NOT a result)."""
    try:
        cases = _read_jsonl(testbed_dir(root) / 'cases.jsonl')
    except FileNotFoundError:
        return lambda messages: '{}'
    lookup = {}
    for style in ('mas_transcript', 'clinic_letter'):
        path = testbed_dir(root) / f'renders-{style}-{WRITER_PROMPT_VERSION}.jsonl'
        if path.exists():
            by_id = {c['case_id']: c for c in cases}
            for r in _read_jsonl(path):
                if r.get('status') == 'ok':
                    lookup[(by_id[r['case_id']]['patient_summary'], r['text'])] = by_id[r['case_id']]
    return oracle_responder(lookup)


def load_records(run_dir: Path) -> dict[str, dict[str, dict]]:
    """{'<arm>-<split>-<style>': {case_id: record}}, seal-verified."""
    out = {}
    for path in sorted(run_dir.glob('*/records.jsonl')):
        _verify_seal(path)
        out[path.parent.name] = {r['case_id']: r for r in _read_jsonl(path) if r.get('status') != 'NotRun'
                                 and 'triage' in r}
    return out


def stage_score(args, root: Path) -> int:
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    arms = load_records(run_dir)
    if not arms:
        raise SystemExit(f'No sealed records under {run_dir}.')
    cases = _read_jsonl(testbed_dir(root) / 'cases.jsonl')       # gold read only now
    for name, recs in list(arms.items()):
        if name.startswith('full-'):
            arms['rules_only-' + name[len('full-'):]] = {k: rules_only(v) for k, v in recs.items()}
    metrics = {'version': VERSION, 'run_dir': run_dir.name, 'held_out_types': list(HELD_OUT_TYPES), 'arms': {},
               'paired': {}, 'note': 'Known-answer synthetic testbed; severities and rules are project-written, not '
                                     'physician-reviewed. Primary: test split, mas_transcript style.'}
    for name, recs in sorted(arms.items()):
        arm, split, style = name.split('-', 2)
        subset = [c for c in cases if c['split'] == split and c['case_id'] in recs]
        s = SC.score_arm(subset, recs, HELD_OUT_TYPES)
        s['acceptance'] = SC.accept(s['primary']) if split == 'test' and style == PRIMARY_STYLE else None
        metrics['arms'][name] = s
    for a in ('rules_only', 'model_only'):
        for name in arms:
            if name.startswith('full-') and f'{a}-{name[5:]}' in arms:
                split = name.split('-', 2)[1]
                subset = [c for c in cases if c['split'] == split]
                metrics['paired'][f'full_minus_{a}:{name[5:]}'] = SC.paired_diff(
                    subset, arms[f'{a}-{name[5:]}'], arms[name], HELD_OUT_TYPES, args.seed)
    out = run_dir / f'score-{_stamp()}'
    out.mkdir()
    _write_json(out / 'metrics.json', metrics)
    print(json.dumps({n: {'primary': m['primary'], 'acceptance': m['acceptance'],
                          'unseen': m['secondary']['unseen_type_detection'],
                          'triage': {k: v['rate'] for k, v in m['secondary']['triage'].items()}}
                      for n, m in metrics['arms'].items()}, indent=2, ensure_ascii=False))
    print(json.dumps(metrics['paired'], indent=2))
    print(f'Written: {out}')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=('run', 'score'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--arm', choices=('full', 'model_only'), default='full')
    p.add_argument('--split', choices=('train', 'test'), default='train')
    p.add_argument('--style', choices=('mas_transcript', 'clinic_letter'), default=PRIMARY_STYLE)
    p.add_argument('--max-cases', type=int)
    p.add_argument('--subset', choices=('all', 'primary'), default='all')
    p.add_argument('--offset', type=int, default=0, help='skip the first N units (train-split pilots only)')
    p.add_argument('--pilot-run-dir', type=Path, help='owner-approved: bound cost from this sealed same-prompt pilot')
    p.add_argument('--check-prompt-file', type=Path, help='optimised-check.json from scripts/optimize_v4.py')
    p.add_argument('--judge', default='mock')
    p.add_argument('--run-limit', type=float, default=1.0)
    p.add_argument('--prices-checked', type=date.fromisoformat, default=None)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=20261115)
    p.add_argument('--authorized', action='store_true')
    args = p.parse_args(argv)
    if args.prices_checked is not None:
        args.prices_checked = args.prices_checked.isoformat()
    if args.stage == 'score' and args.run_dir is None:
        raise SystemExit('score needs --run-dir.')
    return {'run': stage_run, 'score': stage_score}[args.stage](args, args.root)


if __name__ == '__main__':
    raise SystemExit(main())
