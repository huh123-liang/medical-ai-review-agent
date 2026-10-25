"""Bounded targeted wiring smoke, not a new accuracy benchmark or paired improvement claim."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest,digest
from medical_jury.locking import RunnerLock
from medical_jury.provider import build_messages,effective_profile
from medical_jury.store import Store
from medical_jury.stratified import load_plan
from medical_jury.workflow import Engine

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-4'
INDICES=(11,27,34,61,64,67)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['estimate','run']);args=p.parse_args()
    settings=replace(Settings.load(),judge_profile='scoped-review-v1',evidence_policy='anchored-v2')
    settings.validate_paid();assert settings.max_attempts==1 and settings.model=='deepseek-flash'
    store=Store(ROOT);budget=Budget(settings,store)
    _,rows=load_plan(ROOT,ROOT/'reports/iteration-4/paired/manifest.json')
    routes=[]
    for index,row in enumerate(rows,1):
        req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']])
        selected=effective_profile(req.case,settings.judge_profile)
        messages=build_messages(req.case,req.rubrics[0],True,settings.judge_profile)
        assert messages==build_messages(req.case,req.rubrics[0],True,selected)
        if selected=='v2-strict':assert messages==build_messages(req.case,req.rubrics[0],True,'v2-strict')
        routes.append({'index':index,'effective_profile':selected,'messages_hash':digest(messages)})
    estimates=[budget.estimate(build_messages(EvaluationRequest(case=rows[i-1]['case'],rubrics=[rows[i-1]['rubric']]).case,
        EvaluationRequest(case=rows[i-1]['case'],rubrics=[rows[i-1]['rubric']]).rubrics[0],True,settings.judge_profile)) for i in INDICES]
    assert sum(estimates)<=min(.5,budget.total()['remaining_cny'])
    pre={'protocol':'TARGETED_ROUTING_WIRING_SMOKE_NOT_ACCURACY_VALIDATION','indices':INDICES,'planned_calls':6,
        'estimated_full_reservations_cny':round(sum(estimates),6),'batch_cap_cny':.5,'budget_before':budget.total(),
        'routing_all_70_inputs':routes,'unchanged_v2_input_count':sum(r['effective_profile']=='v2-strict' for r in routes),
        'selection':'Post-hoc regression/safety cases; not random or independent validation.',
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/medical_jury').glob('*.py'))}}
    print(json.dumps({k:v for k,v in pre.items() if k not in {'routing_all_70_inputs','code_sha256'}},indent=2),flush=True)
    if args.action=='estimate':return
    with RunnerLock(ROOT):
        folder=OUT/'scoped-smoke';folder.mkdir(exist_ok=False)
        archive=folder/'frozen-code';archive.mkdir()
        for p in (ROOT/'src/medical_jury').glob('*.py'):(archive/p.name).write_bytes(p.read_bytes())
        (folder/'preflight.json').write_text(json.dumps(pre,indent=2),encoding='utf-8')
        start=budget.total()['accounted_cny'];engine=Engine(settings,store);results=[]
        for index in INDICES:
            row=rows[index-1];remaining=.5-(budget.total()['accounted_cny']-start)
            req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',mode='oracle',max_run_cost_cny=min(.25,remaining))
            job=engine.run(req)
            result={'index':index,'job_id':job['id'],'status':job['status'],'error_code':job['error_code']}
            if job['status']=='Completed':result.update(judgment=job['report']['judgments'][0]['judgment'],
                release_status=job['report']['release_status'],effective_profile=job['snapshot']['effective_judge_profile'])
            results.append(result)
            summary={'preflight':pre,'results':results,'state':'Running','budget':budget.total(),
                     'batch_accounted_cny':round(budget.total()['accounted_cny']-start,6)}
            (folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            print(json.dumps(result),flush=True)
            if result['error_code'] in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'}:break
        summary['state']='Finished';(folder/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'budget':summary['budget'],'batch_accounted_cny':summary['batch_accounted_cny']}),flush=True)

if __name__=='__main__':main()
