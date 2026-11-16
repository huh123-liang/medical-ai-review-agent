"""External validation on MedExpert (clinician-annotated omissions + factuality errors).

The agent is NOT tuned on this data. Recommended: run it with the system frozen on HealthBench:
    python scripts/run_external_medexpert.py --freeze reports/protocol-freeze/<id>/freeze.json
    python scripts/run_external_medexpert.py --freeze reports/protocol-freeze/<id>/freeze.json --authorized
Before a freeze exists you may run the current hand-written v2 judge (labelled PRELIMINARY):
    python scripts/run_external_medexpert.py --judge deepseek-pro --authorized
Offline pipeline check (no calls):  python scripts/run_external_medexpert.py --judge mock

Side effects (by design):
* plan mode (paid judge without --authorized): reads the raw file and the ledger (read-only) and
  writes ONLY reports/external-plans/<stamp>/plan.json. No data/ writes, no lock, no labels touched.
* run mode: writes ONLY under reports/external-runs/ (lock, predictions.partial.jsonl, predictions.jsonl,
  manifest.json, summary.json) plus audit/cost rows in data/jury.sqlite. Nothing is written to data/
  external or data/private: judge inputs are built in memory from label-free rows (severity fields are
  dropped on load), and labels are derived from the raw file only AFTER predictions are sealed.
One run per system (lock file). 540 pairs x 2 criteria = 1080 calls.
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
from medical_jury.datasets import medexpert as MX  # noqa: E402
from medical_jury.domain import CaseInput, Rubric, digest  # noqa: E402
from medical_jury.evaluation import freeze as F  # noqa: E402
from medical_jury.evaluation.data import write_jsonl  # noqa: E402
from medical_jury.evaluation.external import score_external  # noqa: E402
from medical_jury.evaluation.runner import (LiveWriter, StopController, budget_gate, ledger_status,  # noqa: E402
                                           run_planned, status_counts)
from medical_jury.judge_v2 import (MAX_OUTPUT_TOKENS_V2, PROMPT_VERSION_V2, JudgeV2, ScriptedBackend,  # noqa: E402
                                   build_messages_v2, mock_responder)
from medical_jury.jury_config import build_backend, load_panel  # noqa: E402
from medical_jury.store import Store  # noqa: E402

RAW = 'data/external/medexpert/medexpert-all.jsonl'


def billable(spec):
    return spec['kind'] == 'deepseek' or float(spec.get('price_in', 0)) > 0 or float(spec.get('price_out', 0)) > 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--raw', default=RAW)
    p.add_argument('--freeze', type=Path, default=None)
    p.add_argument('--judge', default=None, help="Judge in jury-v2.json, or 'mock' (ignored with --freeze).")
    p.add_argument('--authorized', action='store_true')
    p.add_argument('--run-limit', type=float, default=160.0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--limit', type=int, default=None, help='Smoke-test subset (mock only).')
    p.add_argument('--allow-partial', action='store_true',
                   help='Start even if the upper bound exceeds the remaining cumulative budget (owner approval needed).')
    args = p.parse_args(argv)
    root = args.root
    raw_path = root / args.raw
    if not raw_path.exists():
        raise SystemExit(f'{raw_path} missing. Download sonal-ssj/MedExpert (medexpert-all) as JSONL there.')
    raw_sha = MX.file_sha256(raw_path)
    # Label fields are dropped on load: severity columns are never held in memory before sealing (F07).
    public, manifest = MX.build_inputs(MX.load_raw(raw_path, keep=MX.INPUT_FIELDS))
    manifest['raw_sha256'] = raw_sha

    # ---- which system is being tested ----
    freeze, program, threshold = None, None, None
    if args.freeze:
        fpath = args.freeze if args.freeze.is_absolute() else root / args.freeze
        freeze = json.loads(fpath.read_text(encoding='utf-8'))
        problems = F.verify(root, freeze)
        if problems:
            raise SystemExit('Frozen system changed; refusing:\n- ' + '\n- '.join(problems))
        if len(freeze['jury']) != 1:
            raise SystemExit('External runner supports single-judge freezes (option 3).')
        spec = freeze['jury'][0]
        program = freeze['programs'].get(spec['name'])
        threshold = freeze['decision']['threshold']
        system_id = 'freeze-' + freeze['freeze_sha256'][:12]
    else:
        if not args.judge:
            raise SystemExit('Give --freeze (recommended) or --judge NAME.')
        spec = {'name': 'mock', 'kind': 'mock'} if args.judge == 'mock' else \
            next((j for j in load_panel(root) if j['name'] == args.judge), None)
        if spec is None:
            raise SystemExit(f'Judge {args.judge!r} not in jury-v2.json.')
        system_id = 'prelim-' + digest({'spec': spec, 'prompt': PROMPT_VERSION_V2})[:12]
    if args.limit and spec['kind'] != 'mock':
        raise SystemExit('--limit is for offline smoke tests only; a paid external run must cover all items.')
    items = public[:args.limit] if args.limit else public
    paid = billable(spec)
    settings = Settings.load(root) if paid else Settings(root=root)
    upper = 0.0
    if paid:
        rates = {'input_rate': float(spec['price_in']), 'output_rate': float(spec['price_out'])} \
            if spec.get('price_in') is not None else {}
        b = Budget(replace(settings, max_output_tokens=MAX_OUTPUT_TOKENS_V2, **rates), None)
        upper = (3 if program else 1) * sum(b.estimate(build_messages_v2(CaseInput(**i['case']), Rubric(**i['rubric'])))
                                            for i in items)
    plan = {'dataset': 'MedExpert', 'system': system_id, 'judge': spec['name'], 'program': program,
            'threshold': threshold, 'items': len(items), 'calls': len(items),
            'cost_upper_bound_cny': round(upper, 2), 'run_limit_cny': args.run_limit,
            'ledger': ledger_status(root) if paid else None,
            'stop_conditions': StopController().summary()['rules'],
            'status': 'FROZEN_EXTERNAL' if freeze else 'PRELIMINARY_NOT_FROZEN',
            'data_flow': 'Question + chatbot answer + criterion text are sent to the judge provider; '
                         'no labels or annotator ids are sent. Check the MedExpert terms before paid use.',
            'manifest': manifest}
    print(json.dumps({k: v for k, v in plan.items() if k != 'manifest'}, indent=2, ensure_ascii=False))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if paid and not args.authorized:
        plan_dir = root / 'reports/external-plans' / f'medexpert-{system_id}-{stamp}'
        plan_dir.mkdir(parents=True, exist_ok=False)
        (plan_dir / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f'Plan only -> {plan_dir.relative_to(root)}. Nothing locked, called or written under data/.')
        return 0
    refusal = budget_gate(plan, paid, args.allow_partial)
    if refusal:
        raise SystemExit(refusal)
    runs = root / 'reports/external-runs'
    runs.mkdir(parents=True, exist_ok=True)
    tag = f'medexpert-{system_id}' + (f'-limit{args.limit}' if args.limit else '')
    try:
        with (runs / f'{tag}.lock').open('x', encoding='utf-8') as f:
            f.write(json.dumps({'started_at': datetime.now(timezone.utc).isoformat(), 'calls': len(items)}))
    except FileExistsError:
        raise SystemExit(f'{tag} already ran once; external validation is one-shot per system.') from None
    out = runs / tag
    out.mkdir(parents=True, exist_ok=False)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    store = Store(root)
    run_id = f'EXT-{tag}'
    store.audit(run_id, 'external_run_started', {k: v for k, v in plan.items() if k != 'manifest'})
    backend = ScriptedBackend(mock_responder, name='mock') if spec['kind'] == 'mock' else \
        build_backend(spec, settings, store, args.run_limit)
    controller = StopController()
    if program:
        from medical_jury.dspy_judge import DspyJudge, build_program
        prog = build_program()
        prog.load(str(root / program))
        judge = DspyJudge(prog, backend, store, name=spec['name'], stop=controller.event)
    else:
        judge = JudgeV2(backend, store, stop=controller.event)

    def work(item):
        rec = judge.judge(CaseInput(**item['case']), Rubric(**item['rubric']), run_id)
        if threshold is not None and rec.get('status') == 'Completed':
            rec['verdict'] = 'met' if rec['p_met'] >= threshold else 'not_met'
            rec['decision_source'] = 'frozen_threshold'
        return rec

    writer = LiveWriter(out / 'predictions.partial.jsonl')
    try:
        predictions, interrupted = run_planned(items, work, id_of=lambda i: i['example_id'], writer=writer,
                                               controller=controller, threads=args.threads)
    finally:
        writer.close()
        if hasattr(backend, 'close'):
            backend.close()
    write_jsonl(out / 'predictions.jsonl', predictions)
    sealed = digest(predictions)
    store.audit(run_id, 'predictions_sealed', {'sha256': sealed, 'n': len(predictions)})

    # ---- labels: re-derived from the unchanged raw file only now ----
    if MX.file_sha256(raw_path) != raw_sha:
        raise SystemExit('Raw MedExpert file changed during the run; predictions are kept but not scored.')
    private = MX.build_references(MX.load_raw(raw_path))
    labels = {r['example_id']: r for r in private}
    scores = score_external(predictions, labels)
    summary = {'plan': plan, 'predictions_sha256': sealed, 'scores': scores,
               'status_counts': status_counts(predictions), **controller.summary(),
               'interrupted': interrupted,
               'mock_warning': 'Mock judge: pipeline check only.' if spec['kind'] == 'mock' else None}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    brief = {code: {'kappa': v['judge_vs_clinician']['cohen_kappa'], 'ac1': v['judge_vs_clinician']['gwet_ac1'],
                    'macro_f1': v['judge_vs_clinician']['macro_f1'], 'defect_sensitivity': v['defect_sensitivity'],
                    'failure_share': v['failure_share'],
                    'clinician_reference_kappa_double_subset': v['double_annotated_subset']['clinician_vs_clinician_kappa'],
                    'judge_kappa_double_subset': v['double_annotated_subset']['judge_vs_clinician_kappa']}
             for code, v in scores['by_criterion'].items()}
    print(json.dumps({'out': str(out.relative_to(root)), 'status': plan['status'],
                      'status_counts': summary['status_counts'], 'by_criterion': brief}, indent=2))
    return 130 if interrupted else 0


if __name__ == '__main__':
    raise SystemExit(main())
