"""Predeclared, bounded Pro runs. Labels never enter model or sample selection."""
from __future__ import annotations
import argparse
import hashlib
import json
import random
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
from medical_jury.config import Settings
from medical_jury.budget import Budget
from medical_jury.domain import EvaluationRequest,digest
from medical_jury.provider import build_messages
from medical_jury.stratified import load_plan
from medical_jury.store import Store,timestamp
from medical_jury.locking import RunnerLock
from medical_jury.workflow import Engine

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-7'
PROFILES={'A':'evidence-id-v1','B':'grounded-review-v1'}
CATS=tuple('cluster:complex_responses_'+s for s in ('detailed_accuracy_hedging','detailed_appropriate','simple_accuracy_hedging','simple_appropriate'))


def dump(path,obj,exclusive=False):
    if exclusive:
        with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    else:
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze():
    data={'created_at':timestamp(),'core':{p.name:sha(p) for p in sorted((ROOT/'src/medical_jury').glob('*.py'))},
          'runner_sha256':sha(Path(__file__)),
          'analyzer_sha256':sha(ROOT/'scripts/analyze_iteration7.py'),
          'analysis_helpers':{n:sha(ROOT/'scripts'/n) for n in ('analyze_iteration5.py','analyze_paired_iteration3.py')},
          'protocol_sha256':sha(ROOT/'docs/ITERATION-7-PROTOCOL.md'),
          'fact_files':{p.name:sha(p) for p in sorted((ROOT/'data/facts/v1').iterdir()) if p.is_file()}}
    dump(OUT/'freeze.json',data,True)


def check_freeze():
    f=json.loads((OUT/'freeze.json').read_text())
    assert {p.name:sha(p) for p in (ROOT/'src/medical_jury').glob('*.py')}==f['core']
    assert sha(Path(__file__))==f['runner_sha256']
    assert sha(ROOT/'scripts/analyze_iteration7.py')==f['analyzer_sha256']
    assert all(sha(ROOT/'scripts'/n)==h for n,h in f['analysis_helpers'].items())
    assert sha(ROOT/'docs/ITERATION-7-PROTOCOL.md')==f['protocol_sha256']
    assert all(sha(ROOT/'data/facts/v1'/n)==h for n,h in f['fact_files'].items())
    return f


def fixtures():
    statements=[
      ('oral aripiprazole hepatic function',
       'For oral ABILIFY tablets, hepatic function alone with Child-Pugh 5-15 does not require a dose adjustment.',
       'The oral ABILIFY label requires dose reduction solely because a patient has mild Child-Pugh A hepatic impairment.'),
      ('fentanyl transdermal metabolism',
       'Fentanyl is primarily metabolized via CYP3A4; urinary metabolites do not establish primary renal metabolism.',
       'Transdermal fentanyl is primarily metabolized by the kidneys rather than CYP3A4.'),
      ('lithium with ACE inhibitors',
       'Renin-angiotensin antagonists may increase lithium concentrations and require closer lithium monitoring.',
       'ACE inhibitors are free of lithium interactions and require no additional lithium monitoring.'),
      ('Lyme tick bite prophylaxis with doxycycline',
       'Single-dose doxycycline prophylaxis can include children of any age if tick-bite and patient-safety criteria are met.',
       'CDC Lyme single-dose doxycycline prophylaxis has a mandatory minimum age of eight years.')]
    rows=[];expected={}
    for topic,positive,negative in statements:
        for text,truth in ((positive,True),(negative,False)):
            i=len(rows)+1;eid=f'I7-SOURCE-FIXTURE-{i}'
            rows.append({'example_id':eid,'category':'source-consistency-fixture',
                         'case':{'case_id':eid,'group_id':eid,'task':'Assess the fixed educational statement.',
                                 'context':[{'role':'user','content':f'Check this educational claim about {topic} for the stated label/CDC scope.'}],
                                 'candidate_output':text,'source':'SYNTHETIC_PUBLIC_SOURCE_FIXTURE_NOT_PHYSICIAN_DATA',
                                 'source_version':'iteration7-v1','split':'adhoc'},
                         'rubric':{'rubric_id':eid+'-R','text':'The medical statement must be factually accurate for the stated product, population and indication.',
                                   'level':'case','case_id':eid,'source':'Researcher-authored synthetic source consistency test, not physician rubric',
                                   'version':'iteration7-v1','approved':True}})
            expected[eid]=truth
    dump(OUT/'source-fixture-inputs.json',rows,True)
    dump(ROOT/'data/private/iteration7-fixture-expected.json',{'warning':'Researcher source-consistency expectations, NOT physician labels.','expected':expected},True)
    plan={'split':'adhoc','planned_examples':8,'source_file_sha256':sha(OUT/'source-fixture-inputs.json'),
          'entries':[{'example_id':r['example_id'],'input_hash':digest(r)} for r in rows]}
    plan['manifest_hash']=digest(plan);dump(OUT/'smoke-manifest.json',plan,True)


def prepare(phase):
    check_freeze()
    if phase=='smoke':fixtures();return
    if phase.startswith('fixed'):
        original,rows=load_plan(ROOT,ROOT/'reports/iteration-6/fixed/manifest.json')
        sl=slice(0,35) if phase=='fixed-first' else slice(35,70)
        plan={**original,'entries':original['entries'][sl],'planned_examples':35,
              'parent_fixed70_manifest':original['manifest_hash'],'phase':phase}
        plan.pop('manifest_hash');plan['manifest_hash']=digest(plan)
        dump(OUT/f'{phase}-manifest.json',plan,True);return
    assert json.loads((OUT/'gate-4.json').read_text())['status'].startswith('PASS')
    with Store(ROOT).connect() as db:
        excluded={json.loads(r[0])['case']['group_id'] for r in db.execute('SELECT request_json FROM jobs')}
        excluded.update(r[0] for r in db.execute('SELECT DISTINCT group_id FROM rubrics'))
    rows=[json.loads(x) for x in (ROOT/'data/prepared/dev.jsonl').read_text(encoding='utf-8').splitlines()]
    pools=defaultdict(list)
    for r in rows:
        if r['category'] in CATS and r['case']['group_id'] not in excluded:pools[r['category']].append(r)
    rng=random.Random(720261025)
    for pool in pools.values():pool.sort(key=lambda r:r['example_id']);rng.shuffle(pool)
    selected=[];used=set(excluded)
    for _ in range(8):
        for cat in CATS:
            row=next((r for r in pools[cat] if r['case']['group_id'] not in used),None)
            if row is None:raise ValueError('Not enough unseen independent groups.')
            selected.append(row);used.add(row['case']['group_id'])
    plan={'split':'dev','planned_examples':32,'sampling_version':'iteration7-v1','seed':720261025,
          'dev_file_sha256':sha(ROOT/'data/prepared/dev.jsonl'),'labels_file_sha256':sha(ROOT/'data/private/physician-labels.jsonl'),
          'outcome_blind_sampling':True,'excluded_groups':sorted(excluded),
          'category_counts':dict(Counter(r['category'] for r in selected)),
          'entries':[{'example_id':r['example_id'],'group_id':r['case']['group_id'],'input_hash':digest(r),
                      'category':r['category'],'topic':'complex_responses'} for r in selected],
          'limitations':'New-to-project public development groups; not hidden validation or absence from model pretraining.'}
    plan['manifest_hash']=digest(plan);dump(OUT/'new-manifest.json',plan,True)


def load_rows(phase):
    if phase!='smoke':return load_plan(ROOT,OUT/f'{phase}-manifest.json')
    plan=json.loads((OUT/'smoke-manifest.json').read_text());copy=dict(plan);assert digest({k:v for k,v in copy.items() if k!='manifest_hash'})==plan['manifest_hash']
    assert sha(OUT/'source-fixture-inputs.json')==plan['source_file_sha256']
    rows=json.loads((OUT/'source-fixture-inputs.json').read_text())
    assert all(digest(r)==e['input_hash'] for r,e in zip(rows,plan['entries']))
    return plan,rows


def execute(phase,real):
    frozen=check_freeze();settings=Settings.load();settings.validate_paid()
    assert settings.model=='deepseek-v4-pro' and settings.max_attempts==1 and settings.max_output_tokens==1200
    if phase=='fixed-first':
        assert json.loads((OUT/'gate-smoke.json').read_text())['status'].startswith('PASS')
    if phase=='fixed-second':assert json.loads((OUT/'fixed-first/summary.json').read_text())['state']=='Finished'
    plan,rows=load_rows(phase);store=Store(ROOT);budget=Budget(settings,store)
    schedule=[];rng=random.Random(720261024+len(rows))
    for i,row in enumerate(rows,1):
        req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']]);arms=list(PROFILES);rng.shuffle(arms)
        for a in arms:
            msgs=build_messages(req.case,req.rubrics[0],True,PROFILES[a],ROOT)
            schedule.append({'index':i,'arm':a,'example_id':row['example_id'],'category':row['category'],
                             'input_hash':digest(row),'messages_hash':digest(msgs),'estimate_cny':budget.estimate(msgs)})
    estimate=round(sum(p['estimate_cny'] for p in schedule),6)
    assert estimate<=budget.total()['remaining_cny'], 'Complete next batch cannot fit conservative reservations; stop without silently shrinking it.'
    pre={'created_at':timestamp(),'phase':phase,'protocol':'SINGLE_PRO_FIRST_PASS_SOURCE_IDS_TASK_FACTS',
         'profiles':PROFILES,'model':settings.model,'temperature':0,'thinking':'disabled','max_output_tokens':1200,
         'manifest_hash':plan['manifest_hash'],'planned_examples':len(rows),'planned_calls':len(schedule),
         'sum_full_reservations_cny':estimate,'budget_before':budget.total(),'freeze':frozen,'schedule':schedule}
    print(json.dumps({k:v for k,v in pre.items() if k not in {'schedule','freeze'}},indent=2),flush=True)
    if not real:return
    with RunnerLock(ROOT):
        folder=OUT/phase;folder.mkdir(exist_ok=False);archive=folder/'frozen-code';archive.mkdir()
        for p in (ROOT/'src/medical_jury').glob('*.py'):(archive/p.name).write_bytes(p.read_bytes())
        dump(folder/'preflight.json',pre);dump(folder/'manifest.json',plan)
        summary={'state':'Running','phase':phase,'predictions':[{**p,'status':'NotRun','judgment':None,'job_id':None,'error_code':None} for p in schedule]}
        dump(folder/'summary.json',summary);start=budget.total()['accounted_cny'];network=0
        def call(p):
            row=rows[p['index']-1];s=replace(settings,judge_profile=PROFILES[p['arm']],evidence_policy='anchored-v2')
            req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']],provider='deepseek',max_run_cost_cny=1)
            assert digest(row)==p['input_hash'] and digest(build_messages(req.case,req.rubrics[0],True,s.judge_profile,ROOT))==p['messages_hash']
            return Engine(s,store).run(req)
        with ThreadPoolExecutor(max_workers=2) as pool:
            for offset in range(0,len(schedule),2):
                batch=summary['predictions'][offset:offset+2]
                futures={pool.submit(call,p):p for p in batch}
                for future in as_completed(futures):
                    p=futures[future];job=future.result()
                    p.update(status=job['status'],job_id=job['id'],error_code=job['error_code'])
                    if job['status']=='Completed':p['judgment']=job['report']['judgments'][0]['judgment']
                    network=network+1 if p['error_code']=='PROVIDER_TRANSPORT_ERROR' else 0
                    dump(folder/'summary.json',summary)
                    print(f'{phase} item={p["index"]} arm={p["arm"]} {p["status"]} {p["error_code"] or ""}',flush=True)
                    if network>=3 or p['error_code'] in {'PROVIDER_HTTP_401','PROVIDER_HTTP_402','PROVIDER_HTTP_403','BUDGET_BLOCKED'}:summary['halt_reason']=p['error_code']
                if summary.get('halt_reason'):break
        summary.update(state='Finished',finished_at=timestamp(),budget=budget.total(),phase_accounted_cny=round(budget.total()['accounted_cny']-start,6))
        dump(folder/'summary.json',summary)
        print(json.dumps({'phase':phase,'budget':summary['budget'],'halt':summary.get('halt_reason')}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['freeze','prepare','estimate','run']);p.add_argument('phase',nargs='?',choices=['smoke','fixed-first','fixed-second','new']);args=p.parse_args()
    if args.action=='freeze':freeze()
    elif args.action=='prepare':prepare(args.phase)
    else:execute(args.phase,args.action=='run')


if __name__=='__main__':main()
