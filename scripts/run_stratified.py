"""Prepare, estimate, then explicitly execute one frozen, versioned dev experiment."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from medical_jury.benchmark import score_run
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest
from medical_jury.locking import RunnerLock
from medical_jury.provider import PROMPT_VERSION, build_messages
from medical_jury.store import Store
from medical_jury.stratified import TOPICS, load_plan, prepare_plan
from medical_jury.workflow import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','estimate','run'])
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--max-batch-cost',type=float,default=5.0)
    args = parser.parse_args()
    settings = Settings.load()
    settings.prepare()
    if args.action=='prepare':
        plan=prepare_plan(settings.root,args.manifest)
        print(json.dumps({k:v for k,v in plan.items() if k!='entries'},ensure_ascii=False,indent=2))
        return
    plan, rows = load_plan(settings.root,args.manifest)
    if not 0 < args.max_batch_cost <= 5:
        raise ValueError('This experiment has a CNY5 secondary cap within the project CNY30 cap.')
    settings.validate_paid()
    if settings.model != 'deepseek-flash' or settings.max_attempts != 1:
        raise ValueError('This experiment requires Flash and one first-pass attempt per example.')
    store=Store(settings.root)
    budget=Budget(settings,store)
    estimates=[budget.estimate(build_messages(EvaluationRequest(case=r['case'],rubrics=[r['rubric']]).case,
                                            EvaluationRequest(case=r['case'],rubrics=[r['rubric']]).rubrics[0],True)) for r in rows]
    preflight={'prompt_version':PROMPT_VERSION,'planned':len(rows),'sum_full_reservations_cny':round(sum(estimates),6),
               'maximum_single_reservation_cny':max(estimates),'batch_cap_cny':args.max_batch_cost,
               'project_budget_before':budget.total()}
    print(json.dumps(preflight,ensure_ascii=False,indent=2),flush=True)
    if sum(estimates)>min(args.max_batch_cost,budget.total()['remaining_cny']) or max(estimates)>0.50:
        raise ValueError('Full conservative reservation estimate exceeds authorized bounds; no request sent.')
    if args.action=='estimate':
        return
    # Freeze the manifest, prompt and batch budget before the first paid request.
    with RunnerLock(settings.root):
        folder=settings.root/'reports'/('stratified-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
        folder.mkdir(parents=True)
        start_cost=budget.total()['accounted_cny']
        predictions=[{'example_id':r['example_id'],'status':'NotRun','judgment':None,
                      'topic':entry['topic'],'heuristic_scenario':entry['heuristic_scenario']}
                     for r,entry in zip(rows,plan['entries'])]
        summary={'provider':'deepseek','model':settings.model,'prompt_version':PROMPT_VERSION,
                 'manifest_hash':plan['manifest_hash'],'protocol':'FIRST_PASS_ORACLE_NO_OUTPUT_REPAIR',
                 'preflight':preflight,'planned_examples':len(rows),'predictions':predictions}
        (folder/'manifest.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
        engine=Engine(settings,store)
        consecutive_transport=0
        for i,(row,pred) in enumerate(zip(rows,predictions),1):
            remaining=args.max_batch_cost-(budget.total()['accounted_cny']-start_cost)
            if remaining<=0:
                break
            request=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',mode='oracle',
                          max_run_cost_cny=min(0.50,remaining),benchmark_binary=True)
            result=engine.run(request)
            pred.update(job_id=result['id'],status=result['status'],error_code=result['error_code'])
            if result['status']=='Completed':
                pred['judgment']=result['report']['judgments'][0]['judgment']
            summary['budget']=budget.total()
            summary['batch_accounted_cny']=round(budget.total()['accounted_cny']-start_cost,6)
            # Persist after every item; interrupted experiments retain the intended denominator.
            (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            print(f'{i}/{len(rows)} {pred["topic"]} {result["status"]} {result["error_code"] or ""} total_CNY={budget.total()["accounted_cny"]:.6f}',flush=True)
            code=result['error_code'] or ''
            consecutive_transport=consecutive_transport+1 if code.startswith('PROVIDER_TRANSPORT') else 0
            if code in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'} or consecutive_transport>=3:
                break
        summary['evaluation']=score_run(settings.root,predictions,'deepseek',bootstrap=1000)
        summary['by_topic']={topic:score_run(settings.root,[p for p in predictions if p['topic']==topic],'deepseek') for topic in TOPICS}
        summary['status_counts']=dict(Counter(p['status'] for p in predictions))
        summary['error_counts']=dict(Counter(p.get('error_code') for p in predictions if p.get('error_code')))
        summary['budget']=budget.total()
        summary['batch_accounted_cny']=round(budget.total()['accounted_cny']-start_cost,6)
        summary['limitations']=plan['limitations']+['Different samples and prompt from pilot: no causal improvement claim.',
                           'Strict schema is prompt plus local enforcement, not a provider-enforced json_schema guarantee.']
        (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'report':str(folder),'evaluation':summary['evaluation'],'status_counts':summary['status_counts'],
                          'error_counts':summary['error_counts'],'batch_accounted_cny':summary['batch_accounted_cny'],
                          'budget':summary['budget']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
