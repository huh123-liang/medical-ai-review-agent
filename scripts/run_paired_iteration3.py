"""Frozen first-pass A/B on the identical 70 development examples. No gold in requests."""
import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest, digest
from medical_jury.locking import RunnerLock
from medical_jury.provider import build_messages, PROFILE_VERSIONS
from medical_jury.store import Store, timestamp
from medical_jury.stratified import load_plan
from medical_jury.workflow import Engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-3'
BASE=ROOT/'reports/stratified-20261024-194713'
PROFILES={'A':'v2-strict','B':'v3-rubric-first'}
CAP=8.0


def dump(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['estimate','run'])
    args=parser.parse_args()
    for gate in (1,2):
        state=json.loads((OUT/f'gate-{gate}.json').read_text(encoding='utf-8'))
        assert state.get('status',state.get('verdict','')).startswith('PASS'), 'Previous architecture gate has not passed.'
    settings=Settings.load(); settings.prepare(); settings.validate_paid()
    assert settings.model=='deepseek-flash' and settings.max_attempts==1 and settings.max_output_tokens==1200
    store=Store(ROOT); budget=Budget(settings,store)
    plan,rows=load_plan(ROOT,BASE/'manifest.json')
    assert len(rows)==70 and len({r['case']['group_id'] for r in rows})==70
    rng=random.Random(20261024)
    schedule=[]; estimates=[]
    for index,row in enumerate(rows,1):
        order=list(PROFILES); rng.shuffle(order)
        req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',mode='oracle',benchmark_binary=True,max_run_cost_cny=0.5)
        for arm in order:
            messages=build_messages(req.case,req.rubrics[0],True,PROFILES[arm])
            estimate=budget.estimate(messages); estimates.append(estimate)
            schedule.append({'index':index,'arm':arm,'example_id':row['example_id'],
                             'input_hash':digest(row),'messages_hash':digest(messages),'reservation_estimate_cny':estimate})
    preflight={'created_at':timestamp(),'planned_calls':len(schedule),'examples':70,'seed':20261024,
        'model':settings.model,'evidence_policy':'anchored-v1','profiles':PROFILES,
        'prompt_versions':{a:PROFILE_VERSIONS[p] for a,p in PROFILES.items()},
        'sum_full_reservations_cny':round(sum(estimates),6),'maximum_single_reservation_cny':max(estimates),
        'batch_cap_cny':CAP,'budget_before':budget.total(),'manifest_hash':plan['manifest_hash'],
        'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       sorted((ROOT/'src/medical_jury').glob('*.py'))},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'schedule':schedule,'protocol':'FRESH_INTERLEAVED_PAIRED_FIRST_PASS_ORACLE_NO_RETRIES',
        'limitations':['Development set already inspected; not independent validation.',
                       'One run per arm; residual provider nondeterminism remains.',
                       'External review notes and physician labels excluded from judge messages.']}
    assert sum(estimates)<=min(CAP,budget.total()['remaining_cny']) and max(estimates)<=0.5
    print(json.dumps({k:v for k,v in preflight.items() if k not in {'schedule','code_sha256'}},indent=2),flush=True)
    if args.action=='estimate': return
    with RunnerLock(ROOT):
        folder=OUT/'paired'; folder.mkdir(exist_ok=False)
        dump(folder/'preflight.json',preflight); dump(folder/'manifest.json',plan)
        dump(OUT/'gate-3.json',{'status':'PASS_TO_PAID_PAIRED_DEV_EXPERIMENT','objective_alignment':True,
            'architecture_alignment':True,'platform_unchanged':True,'schema_and_workflow_tests_passed':91,
            'labels_in_judge_messages':False,'paired_examples':70,'calls':140,'budget_cap_cny':CAP,
            'frozen_preflight':str((folder/'preflight.json').relative_to(ROOT)),
            'next':'Run once, report all failures and regressions without cherry-picking.'})
        start=budget.total()['accounted_cny']
        summary={'preflight':{k:v for k,v in preflight.items() if k!='schedule'},'predictions':[
            {**s,'status':'NotRun','judgment':None,'job_id':None,'error_code':None} for s in schedule],
            'started_at':timestamp(),'state':'Running'}
        dump(folder/'summary.json',summary)
        engines={a:Engine(replace(settings,judge_profile=p,evidence_policy='anchored-v1'),store) for a,p in PROFILES.items()}
        consecutive_transport=0
        for n,pred in enumerate(summary['predictions'],1):
            remaining=CAP-(budget.total()['accounted_cny']-start)
            if pred['reservation_estimate_cny']>remaining:
                summary['halt_reason']='BATCH_BUDGET'; break
            row=rows[pred['index']-1]
            request=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',mode='oracle',
                                      benchmark_binary=True,max_run_cost_cny=min(0.5,remaining))
            # Verify the exact same payload before each call, not just at preflight.
            assert pred['input_hash']==digest(row)
            assert pred['messages_hash']==digest(build_messages(request.case,request.rubrics[0],True,PROFILES[pred['arm']]))
            result=engines[pred['arm']].run(request)
            pred.update(job_id=result['id'],status=result['status'],error_code=result['error_code'])
            if result['status']=='Completed':
                pred['judgment']=result['report']['judgments'][0]['judgment']
            summary.update(budget=budget.total(),batch_accounted_cny=round(budget.total()['accounted_cny']-start,6))
            dump(folder/'summary.json',summary)
            print(f'{n}/140 item={pred["index"]} arm={pred["arm"]} {pred["status"]} {pred["error_code"] or ""} total_CNY={budget.total()["accounted_cny"]:.6f}',flush=True)
            code=pred['error_code'] or ''
            consecutive_transport=consecutive_transport+1 if code.startswith('PROVIDER_TRANSPORT') else 0
            if code in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'} or consecutive_transport>=3:
                summary['halt_reason']=code; break
        summary.update(state='Finished',finished_at=timestamp(),budget=budget.total(),
                       batch_accounted_cny=round(budget.total()['accounted_cny']-start,6))
        dump(folder/'summary.json',summary)
        print(json.dumps({'folder':str(folder),'budget':budget.total(),'batch_accounted_cny':summary['batch_accounted_cny']}),flush=True)

if __name__=='__main__': main()
