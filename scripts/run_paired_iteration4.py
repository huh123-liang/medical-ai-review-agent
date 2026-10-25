"""One frozen development A/B; both arms share evidence v2 and deterministic safety gate."""
import argparse
import hashlib
import json
import random
from dataclasses import replace
from pathlib import Path
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest,digest
from medical_jury.locking import RunnerLock
from medical_jury.provider import build_messages,PROFILE_VERSIONS
from medical_jury.store import Store,timestamp
from medical_jury.stratified import load_plan
from medical_jury.workflow import Engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-4'; BASE=ROOT/'reports/iteration-3/paired'
PROFILES={'A':'v2-strict','B':'v4-checklist'}
CAP=8.0

def dump(path,data):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['estimate','run']);args=parser.parse_args()
    for n in (1,2):
        assert json.loads((OUT/f'gate-{n}.json').read_text())['status'].startswith('PASS')
    settings=Settings.load();settings.prepare();settings.validate_paid()
    assert settings.model=='deepseek-flash' and settings.max_attempts==1 and settings.max_output_tokens==1200
    store=Store(ROOT);budget=Budget(settings,store);plan,rows=load_plan(ROOT,BASE/'manifest.json')
    assert len(rows)==70 and all(r['case']['split']=='dev' for r in rows)
    rng=random.Random(20261025);schedule=[]
    for i,row in enumerate(rows,1):
        arms=list(PROFILES);rng.shuffle(arms)
        req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']])
        for arm in arms:
            messages=build_messages(req.case,req.rubrics[0],True,PROFILES[arm])
            schedule.append({'index':i,'arm':arm,'example_id':row['example_id'],
                'input_hash':digest(row),'messages_hash':digest(messages),'estimated_cny':budget.estimate(messages)})
    total=sum(p['estimated_cny'] for p in schedule)
    assert total<=min(CAP,budget.total()['remaining_cny']) and max(p['estimated_cny'] for p in schedule)<=0.5
    pre={'created_at':timestamp(),'protocol':'FROZEN_FIRST_PASS_PAIRED_DEV_V2_V4_WITH_SHARED_SAFETY_GATE',
        'profiles':PROFILES,'prompt_versions':{a:PROFILE_VERSIONS[p] for a,p in PROFILES.items()},
        'evidence_policy':'anchored-v2','model':settings.model,'thinking':'disabled','temperature':0,
        'max_output_tokens':1200,'first_pass_only':True,'seed':20261025,'planned_examples':70,'planned_calls':140,
        'manifest_hash':plan['manifest_hash'],'estimated_full_reservations_cny':round(total,6),'batch_cap_cny':CAP,
        'budget_before':budget.total(),'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT/'src/medical_jury').glob('*.py'))},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'schedule':schedule,
        'interpretation':'B combines clause checking and a narrow source-backed safety hint; not a causal ablation of these components.',
        'success_policy':'Do not promote B solely on aggregate mean: inspect complex-task performance, safety case, coverage and regressions.',
        'limitations':['Inspected development examples; no independent validation or clinical clearance.',
            'Same physician labels retained; disputed references not relabelled.','Model alias may vary; metadata and temporal ordering recorded.']}
    print(json.dumps({k:v for k,v in pre.items() if k not in {'schedule','code_sha256'}},indent=2),flush=True)
    if args.action=='estimate': return
    with RunnerLock(ROOT):
        folder=OUT/'paired';folder.mkdir(exist_ok=False);archive=folder/'frozen-code';archive.mkdir()
        for p in (ROOT/'src/medical_jury').glob('*.py'): (archive/p.name).write_bytes(p.read_bytes())
        dump(folder/'preflight.json',pre);dump(folder/'manifest.json',plan)
        start=budget.total()['accounted_cny'];summary={'state':'Running','started_at':timestamp(),'preflight':{k:v for k,v in pre.items() if k!='schedule'},
            'predictions':[{**p,'status':'NotRun','judgment':None,'job_id':None,'error_code':None} for p in schedule]}
        dump(OUT/'experiment-gate.json',{'status':'PASS_TO_BOUNDED_PAID_DEV_TEST','objective_alignment':True,
            'architecture_alignment':'One Agent; deterministic screen/checklist validation; no second model or tool loop.',
            'platform_unchanged':True,'labels_in_model_input':False,'estimated_full_reservations_cny':round(total,6),
            'batch_cap_cny':CAP,'cumulative_cap_cny':30,'next':'Run exactly once, preserve failures; review before promotion.'})
        dump(folder/'summary.json',summary)
        engines={a:Engine(replace(settings,judge_profile=p,evidence_policy='anchored-v2'),store) for a,p in PROFILES.items()}
        network_failures=0
        for n,p in enumerate(summary['predictions'],1):
            remaining=CAP-(budget.total()['accounted_cny']-start)
            if p['estimated_cny']>remaining: summary['halt_reason']='BATCH_BUDGET';break
            row=rows[p['index']-1];req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',
                mode='oracle',benchmark_binary=True,max_run_cost_cny=min(.5,remaining))
            assert digest(row)==p['input_hash'] and digest(build_messages(req.case,req.rubrics[0],True,PROFILES[p['arm']]))==p['messages_hash']
            result=engines[p['arm']].run(req)
            p.update(job_id=result['id'],status=result['status'],error_code=result['error_code'])
            if result['status']=='Completed': p['judgment']=result['report']['judgments'][0]['judgment']
            summary.update(budget=budget.total(),batch_accounted_cny=round(budget.total()['accounted_cny']-start,6))
            dump(folder/'summary.json',summary)
            print(f'{n}/140 item={p["index"]} arm={p["arm"]} {p["status"]} {p["error_code"] or ""} total_CNY={budget.total()["accounted_cny"]:.6f}',flush=True)
            code=p['error_code'] or '';network_failures=network_failures+1 if code=='PROVIDER_TRANSPORT_ERROR' else 0
            if code in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'} or network_failures>=3:
                summary['halt_reason']=code;break
        summary.update(state='Finished',finished_at=timestamp(),budget=budget.total(),batch_accounted_cny=round(budget.total()['accounted_cny']-start,6))
        dump(folder/'summary.json',summary)
        print(json.dumps({'folder':str(folder),'budget':budget.total(),'batch_accounted_cny':summary['batch_accounted_cny']}),flush=True)

if __name__=='__main__':main()
