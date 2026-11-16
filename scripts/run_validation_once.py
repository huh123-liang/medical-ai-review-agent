"""One-shot frozen run on validation (or, with --final, test). PROTOCOL-V2 §3-6.

Plan:  python scripts/run_validation_once.py reports/protocol-freeze/<id>/freeze.json --kind natural
Run:   python scripts/run_validation_once.py reports/protocol-freeze/<id>/freeze.json --kind natural --authorized
Then:  ... --kind panel --authorized          (alt-test sample)
Final: python scripts/run_test_once.py reports/protocol-freeze/<id>/freeze.json --kind natural --authorized

Refuses if any frozen artifact or code changed, or if this split/kind already started once
(a lock file is created BEFORE the first call; crashes do not unlock it). Labels are read only
after predictions are sealed. Failures/undecided items stay in the denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from medical_jury.budget import Budget  # noqa: E402
from medical_jury.config import Settings  # noqa: E402
from medical_jury.domain import CaseInput, Rubric, digest  # noqa: E402
from medical_jury.evaluation import freeze as F  # noqa: E402
from medical_jury.evaluation.data import load_prepared, references_by_id, verify_rater_index, write_jsonl  # noqa: E402
from medical_jury.evaluation.runner import (IMMEDIATE_STOP, LiveWriter, StopController, budget_gate,  # noqa: E402
                                           ledger_status, run_planned, status_counts)
from medical_jury.evaluation.sampling import representativeness  # noqa: E402
from medical_jury.evaluation.scoring import score_predictions  # noqa: E402
from medical_jury.judge_v2 import MAX_OUTPUT_TOKENS_V2, JudgeV2, build_messages_v2  # noqa: E402
from medical_jury.jury_config import build_backend  # noqa: E402
from medical_jury.store import Store  # noqa: E402
from medical_jury.workflow_v2 import JuryEngine  # noqa: E402


def billable(spec):
    return spec['kind'] == 'deepseek' or float(spec.get('price_in', 0)) > 0 or float(spec.get('price_out', 0)) > 0


def build_judges(root, freeze, settings, store, run_limit, stop):
    judges = {}
    for spec in freeze['jury']:
        backend = build_backend(spec, settings, store, run_limit)
        program = freeze['programs'].get(spec['name'])
        if program:
            from medical_jury.dspy_judge import DspyJudge, build_program
            prog = build_program()
            prog.load(str(root / program))
            judges[spec['name']] = DspyJudge(prog, backend, store, name=spec['name'], stop=stop)
        else:
            judges[spec['name']] = JudgeV2(backend, store, stop=stop)
    return judges


def main(argv=None, split: str = 'validation') -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('freeze', type=Path)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--kind', choices=['natural', 'panel'], required=True)
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--final', action='store_true', help='Required for the test split.')
    p.add_argument('--run-limit', type=float, default=150.0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--allow-partial', action='store_true',
                   help='Start even if the upper bound exceeds the remaining cumulative budget (owner approval needed).')
    args = p.parse_args(argv)
    root = args.root
    freeze_path = args.freeze if args.freeze.is_absolute() else root / args.freeze
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))
    freeze_id = freeze['freeze_sha256'][:12]
    problems = F.verify(root, freeze)
    if problems:
        raise SystemExit('Frozen protocol violated; refusing to run:\n- ' + '\n- '.join(problems))
    runs_dir = root / 'reports/frozen-runs'
    if split == 'test':
        if not args.final:
            raise SystemExit('The test split needs --final.')
        needed = [runs_dir / f'{freeze_id}-validation-{k}' / 'summary.json' for k in ('natural', 'panel')]
        if not all(n.exists() for n in needed):
            raise SystemExit('Run both validation kinds for this freeze before the test split.')
    manifest = freeze['manifests'][split][args.kind]
    ids = manifest['example_ids']
    rows = {r['example_id']: r for r in load_prepared(root, split) if r['example_id'] in set(ids)}
    if len(rows) != len(ids):
        raise SystemExit('Prepared data does not contain every frozen example id.')
    paid = any(billable(s) for s in freeze['jury'])
    settings = Settings.load(root) if paid else Settings(root=root)
    upper = 0.0
    for spec in freeze['jury']:
        if not billable(spec):
            continue
        rates = {'input_rate': float(spec['price_in']), 'output_rate': float(spec['price_out'])} \
            if spec.get('price_in') is not None else {}
        b = Budget(replace(settings, max_output_tokens=MAX_OUTPUT_TOKENS_V2, **rates), None)
        # DSPy prompts carry demos; x3 is a deliberately loose upper bound for them.
        factor = 3 if spec['name'] in freeze['programs'] else 1
        upper += factor * sum(b.estimate(build_messages_v2(CaseInput(**rows[i]['case']), Rubric(**rows[i]['rubric'])))
                              for i in ids)
    plan = {'freeze_id': freeze_id, 'split': split, 'kind': args.kind, 'items': len(ids),
            'judges': [s['name'] for s in freeze['jury']], 'calls': len(ids) * len(freeze['jury']),
            'cost_upper_bound_cny': round(upper, 2), 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None,
            'scoring_index': verify_rater_index(root, ids=ids, split=split),
            'stop_conditions': StopController().summary()['rules']}
    print(json.dumps(plan, indent=2))
    if paid and not args.authorized:
        print('Plan only. Nothing was locked or called. Re-run with --authorized.')
        return 0
    refusal = budget_gate(plan, paid, args.allow_partial)
    if refusal:
        raise SystemExit(refusal)
    if not plan['scoring_index']['ok']:
        raise SystemExit('Scoring index not verified; no calls made: ' + '; '.join(plan['scoring_index']['problems']))
    out = runs_dir / f'{freeze_id}-{split}-{args.kind}'
    lock = runs_dir / f'{freeze_id}-{split}-{args.kind}.lock'
    runs_dir.mkdir(parents=True, exist_ok=True)
    try:
        with lock.open('x', encoding='utf-8') as f:
            f.write(json.dumps({**plan, 'started_at': datetime.now(timezone.utc).isoformat()}))
    except FileExistsError:
        raise SystemExit(f'{split}/{args.kind} already started once for freeze {freeze_id}; one-shot rule.') from None
    out.mkdir(parents=True, exist_ok=False)
    store = Store(root)
    run_id = f'FROZEN-{freeze_id}-{split}-{args.kind}'
    store.audit(run_id, 'frozen_run_started', plan)
    controller = StopController()
    judges = build_judges(root, freeze, settings, store, args.run_limit, controller.event)
    ds = None
    if 'ds_params' in freeze['artifacts']:
        ds = json.loads((root / freeze['artifacts']['ds_params']).read_text(encoding='utf-8'))
    engine = JuryEngine(judges, store, root, ds_params=ds, require_human=False, persist_reports=False)

    def one(example_id):
        """Jury result for one item. Judge-level failure codes are PROPAGATED (not just 'Undecided'), so
        401/402/403/429/budget/call-cap codes reach the StopController and stop the run at once."""
        row = rows[example_id]
        res = engine.start(run_id, CaseInput(**row['case']), [Rubric(**row['rubric'])])
        c = res['report']['criteria'][0]
        codes = c.get('judge_failure_codes', {})
        common = {'example_id': example_id, 'votes': c['votes'], 'failed_judges': c['failed_judges'],
                  'judge_failure_codes': codes}
        if c['jury_decision'] is None:
            if codes and all(str(v).startswith('STOPPED_') or v == 'NotRun' for v in codes.values()):
                return {**common, 'status': 'NotRun', 'failure_code': controller.not_run_code()}
            if codes:
                worst = next((v for v in codes.values() if v in IMMEDIATE_STOP), None) or sorted(map(str, codes.values()))[0]
                return {**common, 'status': 'Failed', 'failure_code': worst}
            return {**common, 'status': 'Undecided', 'failure_code': 'JURY_TIE'}
        rec = {**common, 'status': 'Completed', 'verdict': c['jury_decision'], 'p_met': c['p_met'],
               'vote_share': c.get('vote_share'), 'judge_p_met': c.get('judge_p_met'),
               'ds_posterior': c.get('ds_posterior'), 'review_reasons': c['review_reasons'], 'method': c['method']}
        immediate = next((v for v in codes.values() if v in IMMEDIATE_STOP), None)
        if immediate:  # a decided item can still carry a stop-worthy failure from one jury member
            controller.trigger(IMMEDIATE_STOP[immediate], {'failure_code': immediate, 'example_id': example_id})
        return rec

    writer = LiveWriter(out / 'predictions.partial.jsonl')
    try:
        predictions, interrupted = run_planned(ids, one, id_of=lambda eid: eid, writer=writer,
                                               controller=controller, threads=args.threads)
    finally:
        writer.close()
        for j in judges.values():
            backend = getattr(j, 'backend', None)
            if hasattr(backend, 'close'):
                backend.close()
    write_jsonl(out / 'predictions.jsonl', predictions)
    sealed = digest(predictions)
    store.audit(run_id, 'predictions_sealed', {'sha256': sealed, 'n': len(predictions)})
    check = verify_rater_index(root, ids=ids, split=split)
    if not check['ok'] or check['index_sha256'] != plan['scoring_index']['index_sha256']:
        raise SystemExit('Scoring index changed or failed verification after the run; predictions are sealed '
                         'and kept, but not scored: ' + '; '.join(check['problems']))
    refs = references_by_id(root, {split})
    acc = freeze['acceptance']
    threshold = freeze['decision']['threshold']
    scored = score_predictions(predictions, refs, n_boot=2000, seed=20261025, threshold=threshold,
                               alt_epsilon=acc['alt_epsilon'], alt_min_items=acc['alt_min_items_per_annotator'])
    summary = {'status': f'FROZEN_{split.upper()}_{args.kind.upper()}', 'plan': plan, 'predictions_sha256': sealed,
               'scores': scored,
               'representativeness': representativeness(ids, {k: v['labels'] for k, v in refs.items()}, list(refs)),
               'review_queue_share': sum(bool(x.get('review_reasons')) for x in predictions) / len(predictions),
               'status_counts': status_counts(predictions), **controller.summary(), 'interrupted': interrupted,
               'scoring_index_sha256': plan['scoring_index']['index_sha256'],
               'interpretation': ('natural sample -> primary metric A (delta kappa non-inferiority); '
                                  'panel sample -> primary metric B (alt-test). Both must pass. '
                                  'Not a clinical validation.')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    m = scored['conditional_completed_metrics']
    print(json.dumps({'out': str(out.relative_to(root)), 'failure_share': scored['failure_share'],
                      'judge_kappa': m['judge_vs_physician']['cohen_kappa'],
                      'physician_kappa_same_items': m['physician_vs_physician_same_items']['cohen_kappa'],
                      'delta_kappa': m['delta_kappa'], 'delta_ci': m['delta_kappa_group_bootstrap_95ci'],
                      'alt_test': {k: scored['alt_test'][k] for k in ('eligible_annotators', 'winning_rate', 'passed', 'status')},
                      'acceptance_checks': scored['acceptance_checks']}, indent=2))
    return 130 if interrupted else 0


if __name__ == '__main__':
    raise SystemExit(main())
