"""Finish unattempted slots in a stopped pilot, without retrying failed slots.

Uses the original batch's cumulative cost origin and preserves every first-pass result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from medical_jury.benchmark import examples, score_run
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest, canonical, digest
from medical_jury.locking import RunnerLock
from medical_jury.store import Store
from medical_jury.workflow import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--max-total-cost', type=float, default=0.50)
    args = parser.parse_args()
    settings = Settings.load()
    settings.validate_paid()
    if not 0 < args.max_total_cost <= 0.50:
        raise ValueError('This continuation is restricted to the authorized CNY0.50 pilot.')
    parent_bytes = args.summary.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    original = json.loads(parent_bytes.decode('utf-8'))
    if original['provider'] != 'deepseek' or original['model'] != settings.model or original['split'] != 'dev':
        raise ValueError('Continuation must preserve development split, provider and model.')
    rows = examples(settings.root, 'dev', original['planned_examples'])
    predictions = original['predictions']
    if [r['example_id'] for r in rows] != [r['example_id'] for r in predictions]:
        raise ValueError('Original pilot selection no longer matches prepared inputs.')
    cost_origin = original['budget']['accounted_cny'] - original['batch_accounted_cny']
    with RunnerLock(settings.root):
        store = Store(settings.root)
        budget = Budget(settings, store)
        engine = Engine(settings, store)
        for index, (row, pred) in enumerate(zip(rows, predictions), 1):
            if pred['status'] != 'NotRun':
                continue
            remaining = args.max_total_cost - (budget.total()['accounted_cny'] - cost_origin)
            if remaining <= 0:
                break
            request = EvaluationRequest(case=row['case'], rubrics=[row['rubric']], provider='deepseek',
                        mode='oracle', benchmark_binary=True, max_run_cost_cny=min(0.25, remaining))
            result = engine.run(request)
            report = result.get('report')
            pred.update(job_id=result['id'], status=result['status'], judgment=(
                report['judgments'][0]['judgment'] if result['status'] == 'Completed' and report else None))
            print(f'{index}/{len(rows)} {result["id"]} {result["status"]}', flush=True)
        evaluation = score_run(settings.root, predictions, 'deepseek')
        output = dict(original, predictions=predictions, evaluation=evaluation,
                      budget=budget.total(), executed_jobs=sum('job_id' in p for p in predictions),
                      batch_accounted_cny=round(budget.total()['accounted_cny']-cost_origin, 6),
                      parent_summary_sha256=parent_hash,
                      parent_hash_method='sha256_file_bytes',
                      protocol_note='Only NotRun slots continued. Failed slots were not retried or excluded. Same prompt/model.')
        folder = settings.root / 'reports' / ('pilot-first-pass-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
        print(canonical({'report':str(folder), 'evaluation':evaluation, 'batch_accounted_cny':output['batch_accounted_cny']}))


if __name__ == '__main__':
    main()
