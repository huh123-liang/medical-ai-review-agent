"""Freeze outcome-blind new development groups, then run one first-pass paired test."""
import argparse
import hashlib
import json
import random
from collections import Counter,defaultdict
from dataclasses import replace
from pathlib import Path
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest,digest
from medical_jury.locking import RunnerLock
from medical_jury.provider import build_messages
from medical_jury.store import Store,timestamp
from medical_jury.stratified import load_plan
from medical_jury.workflow import Engine

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-5'
PROFILES={'A':'v2-strict','B':'complex-boundary-v1'}
START_CNY=2.427675;ITERATION_CAP=10.0
CATS=tuple('cluster:complex_responses_'+s for s in ('detailed_accuracy_hedging','detailed_appropriate','simple_accuracy_hedging','simple_appropriate'))

def dump(path,obj):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)

def write_plan(chosen,extra,path):
    dev=ROOT/'data/prepared/dev.jsonl';labels=ROOT/'data/private/physician-labels.jsonl'
    result={'split':'dev','sampling_version':'iteration5-v1','planned_examples':len(chosen),
        'dev_file_sha256':hashlib.sha256(dev.read_bytes()).hexdigest(),'labels_file_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),
        'entries':[{'example_id':r['example_id'],'group_id':r['case']['group_id'],'input_hash':digest(r),
                    'category':r['category'],'topic':'complex_responses'} for r in chosen],**extra}
    result['manifest_hash']=digest(result)
    with path.open('x',encoding='utf-8') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in {'entries','excluded_groups'}},indent=2))

def prepare(phase):
    if phase=='calibration':
        _,rows=load_plan(ROOT,ROOT/'reports/iteration-4/paired/manifest.json')
        selected=[r for r in rows if r['category'] in CATS];assert len(selected)==10
        write_plan(selected,{'previously_seen':True,'purpose':'Old regression/calibration, not new cases'},OUT/'calibration-manifest.json')
        return
    gate=json.loads((OUT/'gate-2.json').read_text(encoding='utf-8'));assert gate['status'].startswith('PASS')
    store=Store(ROOT)
    with store.connect() as db:
        jobs=list(db.execute('SELECT request_json FROM jobs'))
        excluded={json.loads(r[0])['case']['group_id'] for r in jobs}
        excluded.update(r[0] for r in db.execute('SELECT DISTINCT group_id FROM rubrics'))
    # Never load physician values for sampling; only preserve their file hash.
    rows=[json.loads(line) for line in (ROOT/'data/prepared/dev.jsonl').read_text(encoding='utf-8').splitlines()]
    pools=defaultdict(list)
    for row in rows:
        if row['category'] in CATS and row['case']['group_id'] not in excluded:pools[row['category']].append(row)
    rng=random.Random(520261024)
    for cat in CATS:pools[cat].sort(key=lambda r:r['example_id']);rng.shuffle(pools[cat])
    selected=[];used=set(excluded)
    for _ in range(20):
        for cat in CATS:
            row=next((r for r in pools[cat] if r['case']['group_id'] not in used),None)
            if row is None:raise ValueError('Insufficient unseen independent groups; do not silently backfill.')
            selected.append(row);used.add(row['case']['group_id'])
    assert len(selected)==80 and len({r['case']['group_id'] for r in selected})==80
    write_plan(selected,{'previously_seen':False,'seed':520261024,'category_counts':dict(Counter(r['category'] for r in selected)),
        'outcome_blind_sampling':True,'excluded_groups':sorted(excluded),'excluded_group_count':len(excluded),
        'exclusion_basis':'All historical jobs including mock/failed and all indexed rubric groups.',
        'novelty_limit':'New to recorded project runs/index; public data may be in pretraining; semantic near-duplicates not fully ruled out.'},OUT/'new-manifest.json')

def execute(phase,run):
    settings=Settings.load();settings.prepare();settings.validate_paid()
    assert settings.model=='deepseek-flash' and settings.max_attempts==1 and settings.max_output_tokens==1200
    assert json.loads((OUT/'gate-1.json').read_text())['status'].startswith('PASS')
    if phase=='new':assert json.loads((OUT/'gate-2.json').read_text())['status'].startswith('PASS')
    plan,rows=load_plan(ROOT,OUT/f'{phase}-manifest.json');store=Store(ROOT);budget=Budget(settings,store)
    cap=2.0 if phase=='calibration' else 8.0;rng=random.Random(2026102405 if phase=='calibration' else 2026102406)
    schedule=[]
    for i,row in enumerate(rows,1):
        req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']]);order=list(PROFILES);rng.shuffle(order)
        for arm in order:
            msgs=build_messages(req.case,req.rubrics[0],True,PROFILES[arm])
            schedule.append({'index':i,'arm':arm,'example_id':row['example_id'],'group_id':row['case']['group_id'],
                'category':row['category'],'input_hash':digest(row),'messages_hash':digest(msgs),'estimate_cny':budget.estimate(msgs)})
    estimate=sum(p['estimate_cny'] for p in schedule)
    assert estimate<=min(cap,ITERATION_CAP-(budget.total()['accounted_cny']-START_CNY),budget.total()['remaining_cny'])
    assert max(p['estimate_cny'] for p in schedule)<=.5
    pre={'protocol':'FIRST_PASS_ORACLE_PAIRED_COMPLEX_BOUNDARY','phase':phase,'created_at':timestamp(),
        'profiles':PROFILES,'manifest_hash':plan['manifest_hash'],'planned_examples':len(rows),'planned_calls':len(schedule),
        'model':settings.model,'evidence_policy':'anchored-v2','max_output_tokens':1200,'temperature':0,
        'sum_full_reservations_cny':round(estimate,6),'phase_cap_cny':cap,'iteration_cap_cny':ITERATION_CAP,
        'iteration_start_accounted_cny':START_CNY,'budget_before':budget.total(),
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/medical_jury').glob('*.py'))},
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'schedule':schedule,
        'promotion_policy':'No automatic default change. Check coverage, per-category regressions, strict-fact scope and paired uncertainty; new dev is not hidden validation.',
        'boundary_doc_sha256':hashlib.sha256((ROOT/'docs/ITERATION-5-BOUNDARIES-AND-PROTOCOL.md').read_bytes()).hexdigest()}
    print(json.dumps({k:v for k,v in pre.items() if k not in {'schedule','code_sha256'}},indent=2),flush=True)
    if not run:return
    with RunnerLock(ROOT):
        folder=OUT/phase;folder.mkdir(exist_ok=False);archive=folder/'frozen-code';archive.mkdir()
        for p in (ROOT/'src/medical_jury').glob('*.py'):(archive/p.name).write_bytes(p.read_bytes())
        dump(folder/'preflight.json',pre);dump(folder/'manifest.json',plan)
        start=budget.total()['accounted_cny'];summary={'state':'Running','preflight':{k:v for k,v in pre.items() if k!='schedule'},
            'predictions':[{**p,'status':'NotRun','judgment':None,'job_id':None,'error_code':None} for p in schedule]}
        dump(folder/'summary.json',summary)
        engines={a:Engine(replace(settings,judge_profile=profile,evidence_policy='anchored-v2'),store) for a,profile in PROFILES.items()}
        network_failures=0
        for n,p in enumerate(summary['predictions'],1):
            remaining=min(cap-(budget.total()['accounted_cny']-start),ITERATION_CAP-(budget.total()['accounted_cny']-START_CNY))
            if p['estimate_cny']>remaining:summary['halt_reason']='PHASE_OR_ITERATION_BUDGET';break
            row=rows[p['index']-1];req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',mode='oracle',max_run_cost_cny=min(.5,remaining))
            assert digest(row)==p['input_hash'] and digest(build_messages(req.case,req.rubrics[0],True,PROFILES[p['arm']]))==p['messages_hash']
            job=engines[p['arm']].run(req);p.update(job_id=job['id'],status=job['status'],error_code=job['error_code'])
            if job['status']=='Completed':p['judgment']=job['report']['judgments'][0]['judgment']
            summary.update(budget=budget.total(),phase_accounted_cny=round(budget.total()['accounted_cny']-start,6))
            dump(folder/'summary.json',summary)
            print(f'{phase} {n}/{len(schedule)} item={p["index"]} arm={p["arm"]} {p["status"]} {p["error_code"] or ""} total_CNY={budget.total()["accounted_cny"]:.6f}',flush=True)
            code=p['error_code'] or '';network_failures=network_failures+1 if code=='PROVIDER_TRANSPORT_ERROR' else 0
            if code in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'} or network_failures>=3:summary['halt_reason']=code;break
        summary.update(state='Finished',finished_at=timestamp(),budget=budget.total(),phase_accounted_cny=round(budget.total()['accounted_cny']-start,6))
        dump(folder/'summary.json',summary)
        print(json.dumps({'phase':phase,'budget':budget.total(),'cost':summary['phase_accounted_cny']}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','estimate','run']);p.add_argument('phase',choices=['calibration','new']);args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.action=='prepare':prepare(args.phase)
    else:execute(args.phase,args.action=='run')

if __name__=='__main__':main()
