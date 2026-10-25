"""Immutable, first-pass Pro paired runs with global ledger and phase checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import EvaluationRequest, digest
from medical_jury.locking import RunnerLock
from medical_jury.provider import build_messages
from medical_jury.store import Store, timestamp
from medical_jury.stratified import load_plan
from medical_jury.workflow import Engine

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-6'
PROFILES={'A':'complex-boundary-v2','B':'task-scope-v1'}
CATS=tuple('cluster:complex_responses_'+s for s in ('detailed_accuracy_hedging','detailed_appropriate','simple_accuracy_hedging','simple_appropriate'))


def dump(path, obj, *, exclusive=False):
    if exclusive:
        with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    else:
        temp=path.with_suffix('.tmp')
        temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
        temp.replace(path)


def code_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src/medical_jury').glob('*.py'))}


def prepare(phase):
    assert json.loads((OUT/'gate-1.json').read_text())['status'].startswith('PASS')
    if phase=='fixed':
        original,rows=load_plan(ROOT,ROOT/'reports/iteration-4/paired/manifest.json')
        assert len(rows)==70
        plan={**original,'parent_manifest_hash':original['manifest_hash'],
              'purpose':'Entire existing fixed70 regression; previously seen development, not independent validation.'}
        plan.pop('manifest_hash')
        plan['manifest_hash']=digest(plan)
        dump(OUT/'fixed-manifest.json',plan,exclusive=True)
        return
    assert json.loads((OUT/'gate-2.json').read_text())['status'].startswith('PASS')
    frozen=json.loads((OUT/'fixed/preflight.json').read_text())
    assert code_hashes()==frozen['code_sha256'], 'No candidate changes after fixed regression.'
    with Store(ROOT).connect() as db:
        excluded={json.loads(r[0])['case']['group_id'] for r in db.execute('SELECT request_json FROM jobs')}
        excluded.update(r[0] for r in db.execute('SELECT DISTINCT group_id FROM rubrics'))
    # No physician values are loaded; only their hash binds later offline evaluation.
    rows=[json.loads(x) for x in (ROOT/'data/prepared/dev.jsonl').read_text(encoding='utf-8').splitlines()]
    pools=defaultdict(list)
    for r in rows:
        if r['category'] in CATS and r['case']['group_id'] not in excluded:pools[r['category']].append(r)
    rng=random.Random(620261024)
    for pool in pools.values():pool.sort(key=lambda r:r['example_id']);rng.shuffle(pool)
    selected=[];used=set(excluded)
    for _ in range(10):
        for cat in CATS:
            row=next((r for r in pools[cat] if r['case']['group_id'] not in used),None)
            if row is None:raise ValueError('Insufficient unseen groups; do not silently backfill.')
            used.add(row['case']['group_id']);selected.append(row)
    dev=ROOT/'data/prepared/dev.jsonl';labels=ROOT/'data/private/physician-labels.jsonl'
    plan={'split':'dev','sampling_version':'iteration6-v1','planned_examples':40,
          'dev_file_sha256':hashlib.sha256(dev.read_bytes()).hexdigest(),
          'labels_file_sha256':hashlib.sha256(labels.read_bytes()).hexdigest(),
          'seed':620261024,'outcome_blind_sampling':True,'excluded_groups':sorted(excluded),
          'exclusion_basis':'All previous jobs including mock/failed and all indexed rubric groups.',
          'category_counts':dict(Counter(r['category'] for r in selected)),
          'entries':[{'example_id':r['example_id'],'group_id':r['case']['group_id'],
                      'input_hash':digest(r),'category':r['category'],'topic':'complex_responses'} for r in selected],
          'limitations':'New to this project, not hidden validation or proven absent from model pretraining.'}
    plan['manifest_hash']=digest(plan)
    dump(OUT/'new-manifest.json',plan,exclusive=True)


def make_preflight(settings, phase, store):
    assert settings.model=='deepseek-v4-pro' and settings.max_attempts==1 and settings.max_output_tokens==1200
    settings.validate_paid()
    plan,rows=load_plan(ROOT,OUT/f'{phase}-manifest.json')
    budget=Budget(settings,store);schedule=[];rng=random.Random(620261024 if phase=='fixed' else 620261025)
    for index,row in enumerate(rows,1):
        req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']])
        arms=list(PROFILES);rng.shuffle(arms)
        for arm in arms:
            msgs=build_messages(req.case,req.rubrics[0],True,PROFILES[arm])
            schedule.append({'index':index,'arm':arm,'example_id':row['example_id'],
                             'group_id':row['case']['group_id'],'input_hash':digest(row),
                             'messages_hash':digest(msgs),'category':row['category'],
                             'estimate_cny':budget.estimate(msgs)})
    estimate=round(sum(p['estimate_cny'] for p in schedule),6)
    phase_cap=min(26.0 if phase=='fixed' else 20.0,budget.total()['remaining_cny'])
    return {'protocol':'PRO_SAME_MODEL_FIRST_PASS_PAIRED_NO_RETRY','phase':phase,
            'created_at':timestamp(),'profiles':PROFILES,'model':settings.model,
            'rates':{'input_ceiling':settings.input_rate,'output_ceiling':settings.output_rate},
            'temperature':0,'thinking':'disabled','max_output_tokens':1200,
            'manifest_hash':plan['manifest_hash'],'planned_examples':len(rows),
            'planned_calls':len(schedule),'sum_full_reservations_cny':estimate,
            'phase_cap_cny':phase_cap,'budget_before':budget.total(),
            'code_sha256':code_hashes(),'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'schedule':schedule},plan,rows


def execute(phase, real):
    settings=Settings.load();settings.prepare();store=Store(ROOT)
    pre,plan,rows=make_preflight(settings,phase,store)
    print(json.dumps({k:v for k,v in pre.items() if k not in {'schedule','code_sha256'}},indent=2),flush=True)
    assert pre['sum_full_reservations_cny'] <= pre['phase_cap_cny'], 'Phase cannot fit full conservative reservations; stop, do not shrink silently.'
    assert max(p['estimate_cny'] for p in pre['schedule']) <= 1
    if not real:return
    assert json.loads((OUT/('gate-1.json' if phase=='fixed' else 'gate-2.json')).read_text())['status'].startswith('PASS')
    if phase=='new':
        assert code_hashes()==json.loads((OUT/'fixed/preflight.json').read_text())['code_sha256']
    with RunnerLock(ROOT):
        folder=OUT/phase;folder.mkdir(exist_ok=False)
        frozen=folder/'frozen-code';frozen.mkdir()
        for p in (ROOT/'src/medical_jury').glob('*.py'):(frozen/p.name).write_bytes(p.read_bytes())
        dump(folder/'preflight.json',pre);dump(folder/'manifest.json',plan)
        budget=Budget(settings,store);start=budget.total()['accounted_cny']
        summary={'state':'Running','phase':phase,'model':settings.model,
                 'predictions':[{**p,'status':'NotRun','judgment':None,'job_id':None,'error_code':None} for p in pre['schedule']]}
        dump(folder/'summary.json',summary)
        network=0
        def call(p):
            row=rows[p['index']-1]
            current=replace(settings,judge_profile=PROFILES[p['arm']],evidence_policy='anchored-v2')
            req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',mode='oracle',max_run_cost_cny=1)
            assert digest(row)==p['input_hash'] and digest(build_messages(req.case,req.rubrics[0],True,current.judge_profile))==p['messages_hash']
            return Engine(current,store).run(req)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for offset in range(0,len(summary['predictions']),2):
                batch=summary['predictions'][offset:offset+2]
                # Reserve enough room for the entire next pair, never submit a full queue.
                if budget.total()['accounted_cny']-start+sum(p['estimate_cny'] for p in batch)>pre['phase_cap_cny']:
                    summary['halt_reason']='PHASE_BUDGET';break
                futures={pool.submit(call,p):p for p in batch}
                for f in as_completed(futures):
                    p=futures[f];job=f.result()
                    p.update(status=job['status'],job_id=job['id'],error_code=job['error_code'])
                    if job['status']=='Completed':p['judgment']=job['report']['judgments'][0]['judgment']
                    network=network+1 if p['error_code']=='PROVIDER_TRANSPORT_ERROR' else 0
                    dump(folder/'summary.json',summary)
                    print(f'{phase} item={p["index"]} arm={p["arm"]} {p["status"]} {p["error_code"] or ""} total={budget.total()["accounted_cny"]:.6f}',flush=True)
                    if p['error_code'] in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'} or network>=3:
                        summary['halt_reason']=p['error_code']
                if summary.get('halt_reason'):break
        summary.update(state='Finished',finished_at=timestamp(),budget=budget.total(),
                       phase_accounted_cny=round(budget.total()['accounted_cny']-start,6))
        dump(folder/'summary.json',summary)
        print(json.dumps({'state':summary['state'],'budget':summary['budget'],'halt':summary.get('halt_reason')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','estimate','run']);p.add_argument('phase',choices=['fixed','new']);args=p.parse_args()
    if args.action=='prepare':prepare(args.phase)
    else:execute(args.phase,args.action=='run')


if __name__=='__main__':main()
