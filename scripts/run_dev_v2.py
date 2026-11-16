"""Jury v2 DEV run: natural sample -> judge v2 -> independent scoring, calibration, metamorphic checks.

Offline by default (hash-based mock; numbers are pipeline checks only):
    python scripts/run_dev_v2.py --n-groups 200
Paid DeepSeek run: first WITHOUT --authorized to print the plan and cost upper bound, then
    python scripts/run_dev_v2.py --backend panel --judge deepseek-pro --n-groups 200 --run-limit 10 --authorized
Groups already sent to a model by earlier non-mock dev runs are excluded by default (--include-used-groups
to keep them; --rerun-of <dir> to repeat an earlier sample deliberately).
Only the dev split is used. Labels are read by the scorer after all predictions are written.
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
from medical_jury.evaluation import calibration, metamorphic  # noqa: E402
from medical_jury.evaluation.data import load_prepared, references_by_id, verify_rater_index, write_jsonl  # noqa: E402
from medical_jury.evaluation.runner import (LiveWriter, StopController, budget_gate, ledger_status,  # noqa: E402
                                           run_planned, status_counts)
from medical_jury.evaluation.sampling import natural_sample, representativeness, topic_of  # noqa: E402
from medical_jury.evaluation.scoring import score_predictions  # noqa: E402
from medical_jury.judge_v2 import (MAX_OUTPUT_TOKENS_V2, PROMPT_VERSION_V2, DeepSeekBackend, JudgeV2,  # noqa: E402
                                   ScriptedBackend, build_messages_v2, mock_responder)
from medical_jury.jury_config import build_backend, load_panel  # noqa: E402
from medical_jury.store import Store  # noqa: E402

STOP_AFTER_CALLS, STOP_MAX_FAILURES = 50, 5


def _previous_runs(root, manifest_sha) -> list[dict]:
    """Earlier non-mock dev runs on the SAME sample (plan.json present; 'started' = predictions exist)."""
    found = []
    for plan_file in sorted((root / 'reports').glob('dev-v2-*/plan.json')):
        try:
            prev = json.loads(plan_file.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if prev.get('sample_manifest_sha256') != manifest_sha or prev.get('backend') == 'mock':
            continue
        run_dir = plan_file.parent
        summary = run_dir / 'summary.json'
        stop_reason = None
        if summary.exists():
            try:
                stop_reason = json.loads(summary.read_text(encoding='utf-8')).get('stop_reason')
            except (OSError, ValueError):
                stop_reason = 'unreadable_summary'
        found.append({'run': str(run_dir.relative_to(root)), 'prompt_version': prev.get('prompt_version'),
                      'started': (run_dir / 'predictions.partial.jsonl').exists(), 'stop_reason': stop_reason})
    return found


def _median(v: list) -> float:
    """Conventional median (mean of the two middle values for an even count)."""
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def _length_stats(predictions) -> dict:
    """Output-length evidence for the 800-token question (raw chars and provider completion tokens).
    ``truncated_finish_length`` counts only finish_reason == 'length' (real token truncation); other
    non-stop endings are listed separately in ``finish_reasons`` and never called truncation."""
    tokens = [r['usage'].get('completion_tokens') for r in predictions
              if isinstance(r.get('usage'), dict) and isinstance(r['usage'].get('completion_tokens'), int)]
    chars = [r['raw_chars'] for r in predictions if isinstance(r.get('raw_chars'), int)]

    def q(values):
        if not values:
            return None
        v = sorted(values)
        return {'n': len(v), 'min': v[0], 'median': _median(v), 'max': v[-1]}
    reasons: dict[str, int] = {}
    for r in predictions:
        if r.get('finish_reason') is not None:
            reasons[str(r['finish_reason'])] = reasons.get(str(r['finish_reason']), 0) + 1
    return {'completion_tokens': q(tokens), 'raw_chars': q(chars), 'median_definition': 'conventional',
            'finish_reasons': dict(sorted(reasons.items())),
            'truncated_finish_length': reasons.get('length', 0),
            'incomplete_non_stop': sum(r.get('failure_code') == 'INCOMPLETE_MODEL_OUTPUT' for r in predictions),
            'max_output_tokens': MAX_OUTPUT_TOKENS_V2,
            'max_share_of_limit': (max(tokens) / MAX_OUTPUT_TOKENS_V2) if tokens else None}


def _used_dev_groups(root) -> tuple[list[str], list[str]]:
    """Groups already sent to a model in earlier NON-mock dev-v2 runs that started (predictions exist).
    Fresh dev runs exclude them by default, so seen material is never silently re-used."""
    groups, runs = set(), []
    for plan_file in sorted((root / 'reports').glob('dev-v2-*/plan.json')):
        run_dir = plan_file.parent
        try:
            prev = json.loads(plan_file.read_text(encoding='utf-8'))
            man = json.loads((run_dir / 'sample-manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if prev.get('backend') == 'mock' or not (run_dir / 'predictions.partial.jsonl').exists():
            continue
        groups.update(man.get('group_ids', []))
        runs.append(str(run_dir.relative_to(root)))
    return sorted(groups), runs


def _complete_as_not_run(items, id_of, writer, controller) -> list[dict]:
    """Planned-order records for a batch that never returned: keep whatever was already written to its
    partial file and write every missing item as NotRun (no model call). Never duplicates a line."""
    done = {}
    if writer.path.exists():
        for line in writer.path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec.get('example_id')] = rec
    out = []
    for item in items:
        eid = id_of(item)
        if eid not in done:
            done[eid] = {'example_id': eid, 'status': 'NotRun', 'failure_code': controller.not_run_code()}
            writer.write(done[eid])
        out.append(done[eid])
    return out


def _check_rerun_of(root, rerun_of, started_before) -> str | None:
    """--rerun-of must name an earlier STARTED run on this exact sample; anything else is refused."""
    if rerun_of is None:
        return None
    target = (rerun_of if rerun_of.is_absolute() else root / rerun_of).resolve()
    allowed = {(root / r['run']).resolve() for r in started_before}
    if not started_before:
        return '--rerun-of given, but no earlier started run used this sample; nothing to rerun. No calls made.'
    if target not in allowed:
        return (f'--rerun-of {rerun_of} is not an earlier started run on this sample '
                f'(candidates: {", ".join(r["run"] for r in started_before)}). No calls made.')
    return None


def _spec(args):
    return next((j for j in load_panel(args.root) if j['name'] == args.judge), {}) if args.judge else {}


def _billable(args):
    spec = _spec(args)
    return spec.get('kind') == 'deepseek' or float(spec.get('price_in', 0)) > 0 or float(spec.get('price_out', 0)) > 0


def _model_name(args, settings):
    if args.backend == 'mock':
        return 'mock'
    if args.backend == 'deepseek':
        return settings.model
    return _spec(args).get('model', '?')


def make_backend(args, settings, store):
    if args.backend == 'mock':
        return ScriptedBackend(mock_responder, name='mock')
    if args.backend == 'deepseek':
        return DeepSeekBackend(settings, store, args.run_limit)
    spec = next((j for j in load_panel(args.root) if j['name'] == args.judge), None)
    if spec is None:
        raise SystemExit(f'Judge {args.judge!r} not found in jury-v2.json.')
    return build_backend(spec, settings, store, args.run_limit)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--n-groups', type=int, default=200)
    p.add_argument('--seed', type=int, default=2026102501)
    p.add_argument('--backend', choices=['mock', 'deepseek', 'panel'], default='mock')
    p.add_argument('--judge', default=None, help='Judge name in jury-v2.json (with --backend panel).')
    p.add_argument('--authorized', action='store_true', help='Required for any paid call.')
    p.add_argument('--run-limit', type=float, default=50.0, help='Per-run CNY limit (runaway guard).')
    p.add_argument('--metamorphic-per-kind', type=int, default=0)
    p.add_argument('--repeats', type=int, default=0, help='Repeat the first 20 items k times (self-consistency).')
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--rerun-of', type=Path, default=None,
                   help='Required to run a sample that an earlier non-mock run already used (deliberate rerun).')
    p.add_argument('--allow-partial', action='store_true',
                   help='Start even if the cost upper bound exceeds the remaining cumulative budget (owner approval needed).')
    p.add_argument('--include-used-groups', action='store_true',
                   help='Do NOT exclude groups already sent to a model by earlier non-mock dev runs (default: exclude).')
    args = p.parse_args(argv)

    root = args.root
    settings = Settings.load(root) if args.backend != 'mock' else Settings(root=root)
    if args.backend == 'panel' and not args.judge:
        raise SystemExit('--backend panel needs --judge NAME.')
    paid = args.backend == 'deepseek' or (args.backend == 'panel' and _billable(args))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    label = args.judge if args.backend == 'panel' else args.backend
    out = args.out or root / 'reports' / f'dev-v2-{label}-{stamp}'
    rows = load_prepared(root, 'dev')
    used_groups, used_runs = ([], []) if (args.backend == 'mock' or args.include_used_groups or args.rerun_of) \
        else _used_dev_groups(root)
    selected, manifest = natural_sample(rows, args.n_groups, args.seed, 'dev', exclude_groups=used_groups)
    suite = metamorphic.build_suite(rows, args.seed, args.metamorphic_per_kind) if args.metamorphic_per_kind else []

    spec = _spec(args) if args.backend == 'panel' else {}
    rates = ({'input_rate': float(spec['price_in']), 'output_rate': float(spec['price_out'])}
             if spec.get('price_in') is not None and spec.get('price_out') is not None else {})
    budget = Budget(replace(settings, max_output_tokens=MAX_OUTPUT_TOKENS_V2, **rates), None)
    planned_messages = [build_messages_v2(CaseInput(**r['case']), Rubric(**r['rubric'])) for r in selected]
    planned_messages += [build_messages_v2(c.original, c.rubric) for c in suite]
    planned_messages += [build_messages_v2(c.perturbed, c.rubric) for c in suite]
    repeat_items = selected[:20] if args.repeats else []
    planned_messages += [build_messages_v2(CaseInput(**r['case']), Rubric(**r['rubric']))
                         for r in repeat_items for _ in range(args.repeats)]
    upper = round(sum(budget.estimate(m) for m in planned_messages), 4)
    plan = {'backend': args.backend, 'judge': label, 'model': _model_name(args, settings),
            'prompt_version': PROMPT_VERSION_V2, 'planned_calls': len(planned_messages),
            'cost_upper_bound_cny': upper if paid else 0.0,
            'run_limit_cny': args.run_limit, 'sample_manifest_sha256': manifest['manifest_sha256'],
            'topic_distribution_sample': manifest['topic_distribution_sample'],
            'metamorphic_cases': len(suite), 'repeat_calls': len(repeat_items) * args.repeats,
            'ledger': ledger_status(root) if paid else None,
            'scoring_index': verify_rater_index(root, ids=[r['example_id'] for r in selected], split='dev'),
            'previous_runs_on_this_sample': _previous_runs(root, manifest['manifest_sha256']),
            'rerun_of': str(args.rerun_of) if args.rerun_of else None,
            'excluded_used_groups': {'n_groups': len(used_groups), 'from_runs': used_runs},
            'writes': ['reports/<this run>/ only (plan, manifest, predictions[, metamorphic, repeats], summary); '
                       'audit + cost rows in data/jury.sqlite only when --authorized'],
            'stop_conditions': StopController(window=STOP_AFTER_CALLS, max_failures=STOP_MAX_FAILURES).summary()['rules'],
            'on_stop': 'completed items kept; remaining items recorded as NotRun with the stop reason; no retry'}
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
    (out / 'sample-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if paid and not args.authorized:
        print('Plan written. No paid call made. Re-run with --authorized after reviewing the cost bound.')
        return 0
    refusal = budget_gate(plan, paid, args.allow_partial)
    if refusal:
        print(refusal)
        return 2
    if not plan['scoring_index']['ok']:
        print('Scoring index not verified; no calls made: ' + '; '.join(plan['scoring_index']['problems']))
        return 2
    started_before = [r for r in plan['previous_runs_on_this_sample'] if r['started']]
    bad_rerun = _check_rerun_of(root, args.rerun_of, started_before) if args.backend != 'mock' else None
    if bad_rerun:
        print(bad_rerun)
        return 2
    if args.backend != 'mock' and started_before and not args.rerun_of:
        print('This exact sample was already run by: ' + ', '.join(r['run'] for r in started_before) +
              '. A rerun (e.g. after a prompt change) must be deliberate: pass --rerun-of <that run dir>. '
              'No calls made.')
        return 2

    store = Store(root)
    job_id = f'DEV-V2-{stamp}'
    store.audit(job_id, 'dev_v2_started', plan)
    backend = make_backend(args, settings, store)
    controller = StopController(window=STOP_AFTER_CALLS, max_failures=STOP_MAX_FAILURES)
    judge = JudgeV2(backend, store, stop=controller.event)
    writer = LiveWriter(out / 'predictions.partial.jsonl')

    def work(row):
        rec = judge.judge(CaseInput(**row['case']), Rubric(**row['rubric']), job_id)
        rec.update(group_id=row['case']['group_id'], topic=topic_of(row['category']))
        return rec

    mm_results, runs, interrupted = [], [], False
    mm_items = [(i, side, case) for i, case in enumerate(suite) for side in ('original', 'perturbed')]
    rep_items = [(row, k) for row in repeat_items for k in range(args.repeats)]
    mm_records, rep_records = [], []
    mm_writer = LiveWriter(out / 'metamorphic.partial.jsonl') if mm_items else None
    rep_writer = LiveWriter(out / 'repeats.partial.jsonl') if rep_items else None

    def mm_work(item):
        i, side, case = item
        rec = judge.judge(case.original if side == 'original' else case.perturbed, case.rubric, job_id)
        rec.update(kind=case.kind, relation=case.relation, side=side, source_example_id=case.source_example_id)
        return rec

    def mm_id(x):
        return f'MM-{x[0]}-{x[1]}'

    def rep_id(x):
        return f"REP-{x[0]['example_id']}-{x[1]}"

    def rep_work(item):
        row, k = item
        rec = judge.judge(CaseInput(**row['case']), Rubric(**row['rubric']), job_id)
        rec.update(repeat_index=k, source_example_id=row['example_id'])
        return rec

    predictions = mm_records_done = rep_records_done = None
    try:
        try:
            predictions, interrupted = run_planned(selected, work, id_of=lambda r: r['example_id'], writer=writer,
                                                   controller=controller, threads=1)
            # F09: metamorphic and repeat calls use the SAME runner: written line by line, one shared
            # StopController, nothing retried. Batches run even after a stop/interrupt: run_planned then
            # makes NO calls and writes every planned item as NotRun, so the audit trail stays complete.
            if mm_items:
                mm_records_done, stopped = run_planned(mm_items, mm_work, id_of=mm_id, writer=mm_writer,
                                                       controller=controller, threads=1)
                interrupted = interrupted or stopped
            if rep_items:
                rep_records_done, stopped = run_planned(rep_items, rep_work, id_of=rep_id, writer=rep_writer,
                                                        controller=controller, threads=1)
                interrupted = interrupted or stopped
        except KeyboardInterrupt:  # Ctrl+C BETWEEN batches (inside a batch run_planned handles it)
            interrupted = True
            controller.interrupt()
        # Any batch that did not finish is completed here as NotRun, item by item, without calls.
        if predictions is None:
            predictions = _complete_as_not_run(selected, lambda r: r['example_id'], writer, controller)
        if mm_items and mm_records_done is None:
            mm_records_done = _complete_as_not_run(mm_items, mm_id, mm_writer, controller)
        if rep_items and rep_records_done is None:
            rep_records_done = _complete_as_not_run(rep_items, rep_id, rep_writer, controller)
    finally:
        writer.close()
        for w in (mm_writer, rep_writer):
            if w:
                w.close()
        if hasattr(backend, 'close'):
            backend.close()
    mm_records, rep_records = mm_records_done or [], rep_records_done or []
    by_id = {r['example_id']: r for r in mm_records}
    for i, case in enumerate(suite):
        o, q = by_id.get(f'MM-{i}-original', {}), by_id.get(f'MM-{i}-perturbed', {})
        mm_results.append({'kind': case.kind, 'relation': case.relation, 'source_example_id': case.source_example_id,
                           'original_verdict': o.get('verdict'), 'perturbed_verdict': q.get('verdict'),
                           'statuses': [o.get('status', 'NotRun'), q.get('status', 'NotRun')]})
    rep_by_id = {r['example_id']: r for r in rep_records}
    for row in repeat_items:
        runs.append([rep_by_id.get(f"REP-{row['example_id']}-{k}", {}).get('verdict') for k in range(args.repeats)])
    if mm_records:
        write_jsonl(out / 'metamorphic.jsonl', mm_records)
    if rep_records:
        write_jsonl(out / 'repeats.jsonl', rep_records)
    if controller.stop_reason:
        store.audit(job_id, 'run_stopped', controller.summary())
    write_jsonl(out / 'predictions.jsonl', predictions)
    sealed = digest(predictions)
    store.audit(job_id, 'predictions_sealed', {'sha256': sealed, 'n': len(predictions)})

    # ---- independent scoring (labels read only now) ----
    check = verify_rater_index(root, ids=[r['example_id'] for r in selected], split='dev')
    if not check['ok'] or check['index_sha256'] != plan['scoring_index']['index_sha256']:
        raise SystemExit('Scoring index changed or failed verification after the run; predictions are sealed '
                         'and kept, but not scored: ' + '; '.join(check['problems']))
    refs = references_by_id(root, {'dev'})
    cal_rows = [{'split': 'dev', 'group_id': r['group_id'], 'p_met': r['p_met'], 'verdict': r['verdict'],
                 'confidence': r['confidence'], 'labels': refs[r['example_id']]['labels']}
                for r in predictions if r['status'] == 'Completed']
    threshold = calibration.choose_threshold(cal_rows) if cal_rows else None
    summary = {
        'status': 'OFFLINE_MOCK_PIPELINE_CHECK_NOT_AGREEMENT' if args.backend == 'mock' else 'DEV_RESULT_NOT_VALIDATION',
        'plan': plan, 'predictions_sha256': sealed, **controller.summary(),
        'interrupted': interrupted, 'status_counts': status_counts(predictions),
        'scoring_index_sha256': plan['scoring_index']['index_sha256'],
        'output_length': _length_stats(predictions),
        'confidence_out_of_contract': sum(bool(r.get('confidence_out_of_contract')) for r in predictions),
        'evidence_status_counts': {k: sum(r.get('evidence_status') == k for r in predictions) for k in
                                   sorted({r.get('evidence_status') for r in predictions if r.get('evidence_status')})},
        'scored_with_verdict': score_predictions(predictions, refs, n_boot=1000, seed=args.seed),
        'threshold_fit_on_dev': threshold,
        'reliability': calibration.reliability(cal_rows),
        'risk_coverage': calibration.risk_coverage(cal_rows),
        'representativeness': representativeness([r['example_id'] for r in selected],
                                                 {k: v['labels'] for k, v in refs.items()},
                                                 [r['example_id'] for r in rows]),
        'metamorphic': metamorphic.evaluate_relations(mm_results) if mm_results else None,
        'metamorphic_status_counts': status_counts(mm_records) if mm_records else None,
        'repeat_status_counts': status_counts(rep_records) if rep_records else None,
        'self_consistency': metamorphic.self_consistency(runs) if runs else None,
        'failure_codes': {c: sum(1 for r in predictions if r.get('failure_code') == c)
                          for c in sorted({r.get('failure_code') for r in predictions if r.get('failure_code')})},
    }
    if threshold and threshold['chosen_threshold'] is not None:
        summary['scored_with_dev_threshold_in_sample'] = score_predictions(
            predictions, refs, n_boot=1000, seed=args.seed, threshold=threshold['chosen_threshold'])
        summary['scored_with_dev_threshold_in_sample']['warning'] = 'Threshold fitted on these same items: optimistic.'
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    s = summary['scored_with_verdict']['conditional_completed_metrics']
    print(json.dumps({'status': summary['status'], 'out': str(out.relative_to(root)),
                      'judge_kappa': s['judge_vs_physician']['cohen_kappa'],
                      'physician_kappa_same_items': s['physician_vs_physician_same_items']['cohen_kappa'],
                      'delta_kappa': s['delta_kappa'], 'failure_share': summary['scored_with_verdict']['failure_share'],
                      'sample_vs_population_met_rate': summary['representativeness']}, indent=2))
    return 130 if interrupted else 0


if __name__ == '__main__':
    raise SystemExit(main())
