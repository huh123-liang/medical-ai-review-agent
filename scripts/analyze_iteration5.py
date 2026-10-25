"""Frozen phase analysis: all failures, each physician and conversation-level uncertainty."""
import argparse
import hashlib
import json
import random
import runpy
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from medical_jury.benchmark import score_run
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-5'
METRICS=runpy.run_path(str(ROOT/'scripts/analyze_paired_iteration3.py'))['metrics']

def compare(items):
    result={};pairs={a:[] for a in ('A','B')};n_gold=sum(len(i['labels']) for i in items)
    for arm in pairs:
        usable=[i for i in items if i['arms'][arm]['status']=='Completed']
        arm_pairs=[(i['arms'][arm]['judgment']=='met',g) for i in usable for g in i['labels']]
        result[arm]={'completed':len(usable),'planned':len(items),'coverage':len(usable)/len(items),
            'delivered_label_matches':sum(x==y for x,y in arm_pairs),'planned_label_comparisons':n_gold,
            'delivered_physician_match_rate':sum(x==y for x,y in arm_pairs)/n_gold}
    common=[i for i in items if all(i['arms'][a]['status']=='Completed' for a in pairs)]
    for arm in pairs:pairs[arm]=[(i['arms'][arm]['judgment']=='met',g) for i in common for g in i['labels']]
    result['common_valid_examples']=len(common)
    result['common_valid_metrics']={a:METRICS(p) for a,p in pairs.items()}
    return result

def bootstrap(items,iterations=3000):
    strata=defaultdict(list)
    for i in items:strata[i['category']].append(i)
    rng=random.Random(520261024);delivered=[];agreement=[]
    for _ in range(iterations):
        sample=[i for group in strata.values() for i in rng.choices(group,k=len(group))]
        scores=compare(sample)
        delivered.append(scores['B']['delivered_physician_match_rate']-scores['A']['delivered_physician_match_rate'])
        a=scores['common_valid_metrics']['A']['agreement'];b=scores['common_valid_metrics']['B']['agreement']
        if a is not None and b is not None:agreement.append(b-a)
    def interval(values):
        values.sort();return [values[int((len(values)-1)*q)] for q in (.025,.975)] if values else None
    return {'delivered_match_rate_delta_95ci':interval(delivered),'common_agreement_delta_95ci':interval(agreement),
            'resampling':'Paired by conversation, within each fixed criterion-category stratum.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['calibration','new']);args=p.parse_args();folder=OUT/args.phase
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'));assert summary['state']=='Finished'
    pre=json.loads((folder/'preflight.json').read_text(encoding='utf-8'))
    for name,sha in pre['code_sha256'].items():assert hashlib.sha256((folder/'frozen-code'/name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((ROOT/'scripts/run_iteration5.py').read_bytes()).hexdigest()==pre['runner_sha256']
    plan,rows=load_plan(ROOT,folder/'manifest.json')
    refs={r['example_id']:r for r in map(json.loads,(ROOT/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    assert args.phase!='new' or (len(rows)==80 and not set(plan['excluded_groups'])&{r['case']['group_id'] for r in rows})
    arms={a:{p['index']:p for p in summary['predictions'] if p['arm']==a} for a in ('A','B')}
    db=sqlite3.connect(f'file:{(ROOT/"data/jury.sqlite").as_posix()}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    items=[];cost=0;errors=[]
    for index,row in enumerate(rows,1):
        item={'index':index,'example_id':row['example_id'],'group_id':row['case']['group_id'],
            'category':row['category'],'labels':refs[row['example_id']]['binary_labels'],'arms':{}}
        for arm in arms:
            prediction=dict(arms[arm][index])
            if prediction['job_id']:
                job=dict(db.execute('SELECT * FROM jobs WHERE id=?',(prediction['job_id'],)).fetchone())
                snap=json.loads(job['snapshot_json'])
                assert snap['case']==row['case'] and snap['rubrics']==[row['rubric']] and snap['judge_profile']==pre['profiles'][arm]
                fees=list(db.execute('SELECT * FROM costs WHERE job_id=?',(prediction['job_id'],)))
                assert len(fees)==1;cost+=sum(f['accounted'] for f in fees)
                if prediction['status']=='Completed':
                    report=json.loads(job['report_json']);judgment=report['judgments'][0]
                    assert prediction['judgment']==judgment['judgment']
                    prediction.update(reason=judgment['reason'],release_status=report['release_status'],
                        review_required=report['review_required'],research_score=report['research_weighted_score'])
                else:
                    logs=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'data/private/provider-diagnostics'/prediction['job_id']).glob('*.json')]
                    prediction['diagnostics']=[{'code':d['error_code'],'details':d.get('details'),
                        'quotes':d.get('public_evidence_excerpts')} for d in logs]
                    errors.append({'index':index,'arm':arm,'error':prediction['error_code'],'diagnostics':prediction['diagnostics']})
            item['arms'][arm]=prediction
        items.append(item)
    db.close();assert abs(cost-summary['phase_accounted_cny'])<1e-6 and summary['budget']['accounted_cny']<=30
    scores=compare(items)
    for arm in ('A','B'):
        m=scores['common_valid_metrics'][arm];c=m['confusion'];n=sum(c.values())
        if n:
            assert abs((c['TP']+c['TN'])/n-m['agreement'])<1e-12
            expected=((c['TP']+c['FP'])*(c['TP']+c['FN'])+(c['TN']+c['FN'])*(c['TN']+c['FP']))/(n*n)
            if expected<1:assert abs((m['agreement']-expected)/(1-expected)-m['cohen_kappa'])<1e-12
            else:assert m['cohen_kappa'] is None
    bycat={cat:compare([i for i in items if i['category']==cat]) for cat in dict.fromkeys(i['category'] for i in items)}
    improved=[];regressed=[]
    for item in items:
        if len(set(item['labels']))!=1:continue
        good={a:item['arms'][a]['status']=='Completed' and (item['arms'][a]['judgment']=='met')==item['labels'][0] for a in arms}
        if good['B'] and not good['A']:improved.append(item['index'])
        if good['A'] and not good['B']:regressed.append(item['index'])
    result={'phase':args.phase,'source_manifest':plan['manifest_hash'],'new_to_project':args.phase=='new',
        'planned':len(rows),'distinct_groups':len({r['case']['group_id'] for r in rows}),'scores':scores,
        'by_category':bycat,'status_counts':{a:dict(Counter(p['status'] for p in arm.values())) for a,arm in arms.items()},
        'primary_evaluation':{a:score_run(ROOT,list(arm.values()),'deepseek') for a,arm in arms.items()},
        'uncertainty':bootstrap(items),'unanimous_improved_indices':improved,'unanimous_regressed_indices':regressed,
        'errors':errors,'phase_accounted_cny':round(cost,6),'budget_at_phase_end':summary['budget'],'items':items,
        'annotation_summary':{'true_labels':sum(sum(i['labels']) for i in items),
            'false_labels':sum(len(i['labels'])-sum(i['labels']) for i in items),
            'unanimous_examples':sum(len(set(i['labels']))==1 for i in items),
            'fixed_binary_annotation_match_ceiling':sum(max(sum(i['labels']),len(i['labels'])-sum(i['labels'])) for i in items)/sum(len(i['labels']) for i in items)},
        'limitations':['All original physician labels retained; no clinical adjudication.',
            'Public development sample, not hidden validation or evidence of absence from model pretraining.',
            'Missing predictions count as undelivered matches, never fabricated not_met labels.',
            'Common-valid metrics are conditional when either arm fails; do not compare with earlier different cohorts.']}
    (folder/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=[f'# 第5轮 / {args.phase} 对照结果','',
        'A=原v2；B=complex-boundary-v1。给定原准则的oracle测试，两组各首次请求一次，引用v2和安全门相同；不改医生标签，不重试。',
        '本样本是旧回归集。' if args.phase=='calibration' else '新80条来自4子类各20条、80个不同开发对话组；按元数据随机抽样、不按标签挑题，排除以前所有job与已索引group。不是隐藏验证集。','',
        '| 指标 | A | B |','| --- | --- | --- |',
        f'| 完整通过 | {scores["A"]["completed"]}/{len(items)} | {scores["B"]["completed"]}/{len(items)} |',
        f'| 全计划可交付标签匹配率 | {scores["A"]["delivered_physician_match_rate"]:.2%} | {scores["B"]["delivered_physician_match_rate"]:.2%} |',
        f'| 共同有效逐医生一致率 | {scores["common_valid_metrics"]["A"]["agreement"]:.2%} | {scores["common_valid_metrics"]["B"]["agreement"]:.2%} |',
        f'| 共同有效Cohen Kappa | {scores["common_valid_metrics"]["A"]["cohen_kappa"]:.4f} | {scores["common_valid_metrics"]["B"]["cohen_kappa"]:.4f} |','',
        f'共同有效{scores["common_valid_examples"]}条；原始医生比较{scores["A"]["planned_label_comparisons"]}条。有失败时整批二分类主指标按旧协议保留为空；“可交付匹配率”以所有原标签为分母，失败只表示未交付匹配，不等于预测not_met。',
        f'配对分层对话bootstrap95%区间：{result["uncertainty"]}。',
        f'医生一致项改善题号：{improved}；退步题号：{regressed}（仅本manifest内题号，不是旧70条题号）。','',
        '| 子类 | 计划 | A可交付匹配率 | B可交付匹配率 |','| --- | --- | --- | --- |']
    for cat,value in bycat.items():lines.append(f'| {cat} | {value["A"]["planned"]} | {value["A"]["delivered_physician_match_rate"]:.1%} | {value["B"]["delivered_physician_match_rate"]:.1%} |')
    lines+=['',f'费用：本阶段{cost:.6f}元；截至阶段结束累计{summary["budget"]["accounted_cny"]:.6f}元。保守本地记账，不是供应商发票。',
        '完整理由、失败与来源job见analysis.json。临床事实争议不凭标签匹配自动裁决。','']
    (folder/'结果解读.md').write_text('\n'.join(lines),encoding='utf-8')
    if args.phase=='calibration':
        (OUT/'gate-2.json').write_text(json.dumps({'status':'PASS_TO_FROZEN_NEW_DEV_DIAGNOSIS_NOT_PROMOTION',
            'objective_alignment':'Test clarified task/rubric scope on unseen project groups, not optimize labels.',
            'architecture_alignment':'Same single-Agent five-field contract, no extra model, fact-tool or vector-index change.',
            'platform_alignment':'No platform writes or changes to doctor scores.',
            'calibration':scores,'default_promoted':False,'next':'Freeze candidate and outcome-blind new80 manifest; no further prompt tuning on new outcomes.'},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'items','by_category','errors'}},indent=2))

if __name__=='__main__':main()
