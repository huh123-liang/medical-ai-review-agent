"""Optimize the DSPy judge on DEV data only (MIPROv2 by default).

Plan only (no calls):  python scripts/optimize_dspy.py --judge deepseek-pro
Run:                   python scripts/optimize_dspy.py --judge deepseek-pro --authorized --run-limit 80
Offline smoke test:    python scripts/optimize_dspy.py --judge mock --optimizer bootstrap --n-train 20 --n-val 10

* Train/val are disjoint conversation groups drawn outcome-blind from dev.
* Physician labels are used ONLY as the metric target on dev (allowed by PROTOCOL-V2 §4).
  Demos are bootstrapped traces that matched all physicians; no labeled demos are fabricated.
* Every LM call (task and proposer) goes through the ledgered backend; --max-calls is a hard stop.
Outputs reports/dspy-opt-<stamp>/{program.json, threshold.json, summary.json}; freeze program + threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import dspy  # noqa: E402

from medical_jury.config import Settings  # noqa: E402
from medical_jury.domain import CaseInput, Rubric  # noqa: E402
from medical_jury.dspy_judge import (DspyJudge, LedgerEngine, LedgerEngineError, build_program,  # noqa: E402
                                     program_sha256, to_inputs)
from medical_jury.evaluation import calibration  # noqa: E402
from medical_jury.evaluation.data import load_prepared, references_by_id  # noqa: E402
from medical_jury.evaluation.runner import ledger_status  # noqa: E402
from medical_jury.evaluation.sampling import natural_sample  # noqa: E402
from medical_jury.evaluation.scoring import score_predictions  # noqa: E402
from medical_jury.judge_v2 import MAX_OUTPUT_TOKENS_V2, ScriptedBackend  # noqa: E402
from medical_jury.jury_config import build_backend, load_panel  # noqa: E402
from medical_jury.store import Store  # noqa: E402


def dspy_mock_responder(messages) -> str:
    """Offline fixture that speaks DSPy's JSON adapter format; hash-derived, no clinical meaning."""
    from medical_jury.domain import digest
    h = int(digest(messages)[:8], 16)
    return json.dumps({'reasoning': 'OFFLINE MOCK placeholder.', 'verdict': 'met' if h % 4 else 'not_met',
                       'confidence': 0.5 + (h % 50) / 100})


class CallCap:
    """Hard, thread-safe cap on LM calls (F06). Both complete() and stream() go through the same
    counter; the check and the increment happen under one lock, so concurrent optimizer threads
    can never exceed ``max_calls``. Once reached, every later call fails with CALL_CAP_REACHED."""

    def __init__(self, engine: LedgerEngine, max_calls: int):
        import threading
        if max_calls < 1:
            raise ValueError('max_calls must be positive.')
        self.engine, self.max_calls = engine, max_calls
        self.issued = 0
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def _admit(self):
        with self._lock:
            if self.issued >= self.max_calls or self.engine.budget_error:
                self.engine.budget_error = self.engine.budget_error or 'CALL_CAP_REACHED'
                raise LedgerEngineError(self.engine.budget_error)
            self.issued += 1

    def complete(self, request):
        self._admit()
        return self.engine.complete(request)

    def stream(self, request):
        from dspy.lm15 import response_to_events
        self._admit()
        return response_to_events(self.engine.complete(request))


def metric(example, pred, trace=None):
    """Mean agreement of the verdict with each physician label. For demo bootstrapping (trace set)
    only traces that agree with ALL physicians are accepted."""
    verdict = getattr(pred, 'verdict', None)
    if verdict not in ('met', 'not_met'):
        return 0.0 if trace is None else False
    labels = example.physician_labels
    score = sum((verdict == 'met') == l for l in labels) / len(labels)
    return score if trace is None else score == 1.0


def examples(rows, refs):
    out = []
    for r in rows:
        case, rubric = CaseInput(**r['case']), Rubric(**r['rubric'])
        out.append(dspy.Example(**to_inputs(case, rubric), physician_labels=refs[r['example_id']]['labels'],
                                example_id=r['example_id']).with_inputs('conversation', 'answer', 'criterion'))
    return out


def run_optimization(program, lm, trainset, valset, dev_val_score, args):
    """Baseline dev-val score -> optimizer -> optimized dev-val score, all through ONE capped LM."""
    with dspy.context(lm=lm, adapter=dspy.JSONAdapter()):
        baseline = dev_val_score(program, 'baseline')
        if baseline.get('budget_or_cap_stop'):
            raise LedgerEngineError(baseline['budget_or_cap_stop'])
        if args.optimizer == 'mipro':
            optimizer = dspy.MIPROv2(metric=metric, auto=args.auto, num_threads=args.threads,
                                     max_bootstrapped_demos=3, max_labeled_demos=0, seed=args.seed)
            # data_aware_proposer would summarise raw training examples, which carry physician_labels.
            compiled = optimizer.compile(program, trainset=trainset, valset=valset,
                                         requires_permission_to_run=False, data_aware_proposer=False)
        else:
            optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=3, max_labeled_demos=0)
            compiled = optimizer.compile(program, trainset=trainset)
        optimized = dev_val_score(compiled, 'optimized')
    if optimized.get('budget_or_cap_stop'):
        raise LedgerEngineError(optimized['budget_or_cap_stop'])
    return compiled, baseline, optimized


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--judge', required=True, help="Judge name in jury-v2.json, or 'mock'.")
    p.add_argument('--optimizer', choices=['mipro', 'bootstrap'], default='mipro')
    p.add_argument('--auto', choices=['light', 'medium', 'heavy'], default='light')
    p.add_argument('--n-train', type=int, default=150)
    p.add_argument('--n-val', type=int, default=100)
    p.add_argument('--seed', type=int, default=2026102502)
    p.add_argument('--max-calls', type=int, default=1500)
    p.add_argument('--run-limit', type=float, default=80.0)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--authorized', action='store_true')
    args = p.parse_args(argv)
    root = args.root
    rows = load_prepared(root, 'dev')
    sample, manifest = natural_sample(rows, args.n_train + args.n_val, args.seed, 'dev')
    train_rows, val_rows = sample[:args.n_train], sample[args.n_train:]
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = root / 'reports' / f'dspy-opt-{args.judge}-{stamp}'
    out.mkdir(parents=True)
    paid = args.judge != 'mock'
    plan = {'judge': args.judge, 'optimizer': args.optimizer, 'auto': args.auto, 'n_train': len(train_rows),
            'n_val': len(val_rows), 'max_calls': args.max_calls, 'run_limit_cny': args.run_limit,
            'sample_manifest_sha256': manifest['manifest_sha256'], 'split': 'dev',
            'ledger': ledger_status(root) if paid else None,
            'spend_bound': ('min(--run-limit, cumulative ceiling remaining); every call is reserved in the ledger '
                            'before sending and --max-calls is a hard, thread-safe cap'),
            'note': 'MIPROv2 light typically needs several hundred to ~2000 calls; the ledger and --max-calls bound it.'}
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2))
    store = Store(root)
    if args.judge == 'mock':
        backend = ScriptedBackend(dspy_mock_responder, name='mock')
    else:
        if not args.authorized:
            print('Plan written; no calls made. Re-run with --authorized.')
            return 0
        if args.run_limit > plan['ledger']['remaining_cny']:
            print(f"--run-limit {args.run_limit} exceeds the remaining cumulative budget "
                  f"{plan['ledger']['remaining_cny']}; no calls made. Lower --run-limit.")
            return 2
        spec = next((j for j in load_panel(root) if j['name'] == args.judge), None)
        if spec is None:
            raise SystemExit(f'Judge {args.judge!r} not in jury-v2.json.')
        backend = build_backend(spec, Settings.load(root), store, args.run_limit)
    job_id = f'DSPY-OPT-{stamp}'
    engine = CallCap(LedgerEngine(backend, job_id, MAX_OUTPUT_TOKENS_V2), args.max_calls)
    lm = dspy.LM(f'ledger/{backend.model}', engine=engine, cache=False, num_retries=0,
                 temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS_V2)
    lm.ledger_engine = engine  # budget_error is forwarded through CallCap
    refs = references_by_id(root, {'dev'})
    trainset, valset = examples(train_rows, refs), examples(val_rows, refs)
    program = build_program()

    last_preds: dict[str, list] = {}

    def dev_val_score(prog, tag):
        judge = DspyJudge(prog, backend, store, name=args.judge, lm=lm)  # same capped engine: counted in --max-calls
        preds = []
        for r in val_rows:
            rec = judge.judge(CaseInput(**r['case']), Rubric(**r['rubric']), f'{job_id}-{tag}')
            preds.append({**rec, 'example_id': r['example_id']})
        s = score_predictions(preds, {r['example_id']: refs[r['example_id']] for r in val_rows}, n_boot=500)
        m = s['conditional_completed_metrics']
        last_preds[tag] = preds
        stop = next((x['failure_code'] for x in preds if x.get('failure_code') in
                     ('BUDGET_BLOCKED', 'BUDGET_SETTLEMENT_EXCEEDED', 'CALL_CAP_REACHED', 'STOPPED_BY_RUN_STOP')), None)
        return {'budget_or_cap_stop': stop, 'verdicts': ''.join('M' if x.get('verdict') == 'met' else ('.' if x.get('verdict') else 'x') for x in preds),
                'failure_share': s['failure_share'], 'kappa': m['judge_vs_physician']['cohen_kappa'],
                'ac1': m['judge_vs_physician']['gwet_ac1'], 'delta_kappa': m['delta_kappa']}

    try:
        compiled, baseline, optimized = run_optimization(program, lm, trainset, valset, dev_val_score, args)
    except (LedgerEngineError, RuntimeError) as error:
        aborted = {'status': 'ABORTED', 'reason': str(error)[:200], 'calls_used': engine.issued,
                   'budget_error': engine.engine.budget_error, 'plan': plan,
                   'note': 'All sent calls are in the cost ledger. No program was saved.'}
        (out / 'summary.json').write_text(json.dumps(aborted, indent=2), encoding='utf-8')
        print(json.dumps(aborted, indent=2))
        return 3
    leak = [k for k in ('physician_labels', 'example_id') if k in json.dumps(compiled.dump_state(), default=str)]
    if leak:
        raise SystemExit(f'Compiled program contains label/id fields {leak}; refusing to save it.')
    compiled.save(str(out / 'program.json'))
    cal_rows = [{'split': 'dev', 'group_id': refs[x['example_id']]['group_id'], 'p_met': x['p_met'],
                 'verdict': x['verdict'], 'confidence': x['confidence'], 'labels': refs[x['example_id']]['labels']}
                for x in last_preds['optimized'] if x.get('status') == 'Completed']
    threshold = calibration.choose_threshold(cal_rows) if cal_rows else None
    (out / 'threshold.json').write_text(json.dumps(threshold, indent=2), encoding='utf-8')
    summary = {'status': 'DEV_OPTIMIZATION_NOT_VALIDATION' if args.judge != 'mock' else 'OFFLINE_MOCK_PIPELINE_CHECK',
               'plan': plan, 'program_sha256': program_sha256(compiled), 'calls_used': engine.issued,
               'dev_val_baseline': baseline, 'dev_val_optimized': optimized,
               'threshold_fit_on_dev_val': threshold,
               'reliability_dev_val': calibration.reliability(cal_rows) if cal_rows else None,
               'risk_coverage_dev_val': calibration.risk_coverage(cal_rows) if cal_rows else None,
               'warning': ('dev-val was used to select the program, so these numbers are optimistic; '
                           'only the frozen validation run counts.')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({**{k: summary[k] for k in ('status', 'program_sha256', 'calls_used', 'dev_val_baseline', 'dev_val_optimized')},
                      'chosen_threshold': threshold['chosen_threshold'] if threshold else None}, indent=2))
    print(f'-> {out.relative_to(root)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
