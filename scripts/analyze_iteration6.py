"""Offline paired meta-evaluation. Physician labels are read only after inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

from analyze_iteration5 import compare, bootstrap
from medical_jury.benchmark import score_run
from medical_jury.domain import digest
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-6'


def load_items(folder):
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert summary['state']=='Finished'
    pre=json.loads((folder/'preflight.json').read_text(encoding='utf-8'))
    plan,rows=load_plan(ROOT,folder/'manifest.json')
    assert pre['manifest_hash']==plan['manifest_hash']
    for name,sha in pre['code_sha256'].items():
        assert hashlib.sha256((folder/'frozen-code'/name).read_bytes()).hexdigest()==sha
    assert pre['runner_sha256']==hashlib.sha256((ROOT/'scripts/run_iteration6.py').read_bytes()).hexdigest()
    labels={r['example_id']:r['binary_labels'] for r in map(json.loads,(ROOT/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    arms={a:{p['index']:p for p in summary['predictions'] if p['arm']==a} for a in ('A','B')}
    items=[];cost=0;reported=Counter()
    db=sqlite3.connect(str(ROOT/'data/jury.sqlite'));db.row_factory=sqlite3.Row
    for index,row in enumerate(rows,1):
        entry={'index':index,'example_id':row['example_id'],'group_id':row['case']['group_id'],
               'category':row['category'],'topic':plan['entries'][index-1].get('topic','complex_responses'),
               'labels':labels[row['example_id']],'arms':{}}
        for arm in ('A','B'):
            p=dict(arms[arm][index])
            if p['job_id']:
                job=dict(db.execute('SELECT * FROM jobs WHERE id=?',(p['job_id'],)).fetchone())
                snap=json.loads(job['snapshot_json'])
                assert snap['case']==row['case'] and snap['rubrics']==[row['rubric']]
                assert snap['model']=='deepseek-v4-pro' and snap['judge_profile']==pre['profiles'][arm]
                fees=list(db.execute('SELECT * FROM costs WHERE job_id=?',(p['job_id'],)))
                assert len(fees)==1
                cost+=sum(f['accounted'] for f in fees)
                for evt in db.execute("SELECT detail_json FROM audit WHERE job_id=? AND event='provider_model_metadata'",(p['job_id'],)):
                    reported[json.loads(evt[0])['reported_model']]+=1
                if p['status']=='Completed':
                    report=json.loads(job['report_json']);judgment=report['judgments'][0]
                    assert p['judgment']==judgment['judgment']
                    p.update(reason=judgment['reason'],review_required=judgment['review_required'],
                             release_status=report['release_status'],release_block_reasons=report['release_block_reasons'],
                             answer_evidence=judgment['answer_evidence'],context_evidence=judgment['context_evidence'])
                else:
                    logs=[json.loads(f.read_text(encoding='utf-8')) for f in (ROOT/'data/private/provider-diagnostics'/p['job_id']).glob('*.json')]
                    p['diagnostics']=[{'code':d['error_code'],'details':d.get('details')} for d in logs]
            entry['arms'][arm]=p
        items.append(entry)
    db.close()
    assert len({i['group_id'] for i in items})==len(items)
    assert abs(cost-summary['phase_accounted_cny'])<1e-6
    assert summary['budget']['accounted_cny']<=30
    if pre['phase']=='new':
        assert len(items)==40 and not set(plan['excluded_groups'])&{i['group_id'] for i in items}
        assert pre['code_sha256']==json.loads((OUT/'fixed/preflight.json').read_text())['code_sha256']
    return items,summary,pre,round(cost,6),dict(reported)


def verify_metrics(scores):
    for arm in ('A','B'):
        m=scores['common_valid_metrics'][arm];c=m['confusion'];n=sum(c.values())
        if not n:continue
        accuracy=(c['TP']+c['TN'])/n
        assert abs(accuracy-m['agreement'])<1e-12
        expected=((c['TP']+c['FP'])*(c['TP']+c['FN'])+(c['TN']+c['FN'])*(c['TN']+c['FP']))/(n*n)
        if expected<1:assert abs((accuracy-expected)/(1-expected)-m['cohen_kappa'])<1e-12


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['fixed','new']);args=p.parse_args()
    folder=OUT/args.phase
    items,summary,pre,cost,reported=load_items(folder)
    scores=compare(items);verify_metrics(scores)
    bycat={cat:compare([i for i in items if i['category']==cat]) for cat in dict.fromkeys(i['category'] for i in items)}
    complex_items=[i for i in items if 'complex_responses' in i['category']]
    complex_scores=compare(complex_items)
    improved=[];regressed=[];mismatches=[]
    for i in items:
        good={a:i['arms'][a]['status']=='Completed' and len(set(i['labels']))==1 and (i['arms'][a]['judgment']=='met')==i['labels'][0] for a in ('A','B')}
        if good['B'] and not good['A']:improved.append(i['index'])
        if good['A'] and not good['B']:regressed.append(i['index'])
        if any(i['arms'][a]['status']!='Completed' or any((i['arms'][a]['judgment']=='met')!=g for g in i['labels']) for a in ('A','B')):
            mismatches.append(i['index'])
    uncertainty_items = [{**i,'category':i['topic']} for i in items] if args.phase=='fixed' else items
    uncertainty = bootstrap(uncertainty_items)
    uncertainty['resampling'] = ('Paired conversation resampling within seven fixed topic strata.'
                                if args.phase=='fixed' else 'Paired conversation resampling within four fixed criterion-category strata.')
    result={'phase':args.phase,'model':pre['model'],'profiles':pre['profiles'],'planned':len(items),
            'scores':scores,'complex_scores':complex_scores,'by_category':bycat,
            'uncertainty':uncertainty,'complex_uncertainty':bootstrap(complex_items),
            'bootstrap_strata':'seven topics for fixed70; four criterion categories for new40',
            'status_counts':{a:dict(Counter(i['arms'][a]['status'] for i in items)) for a in ('A','B')},
            'primary_evaluation':{a:score_run(ROOT,[i['arms'][a] for i in items],'deepseek') for a in ('A','B')},
            'provider_reported_models':reported,'unanimous_improved_indices':improved,
            'unanimous_regressed_indices':regressed,'mismatch_indices':mismatches,
            'phase_accounted_cny':cost,'budget':summary['budget'],'halt_reason':summary.get('halt_reason'),
            'annotation_summary':{'comparisons':sum(len(i['labels']) for i in items),
                                  'met':sum(sum(i['labels']) for i in items),
                                  'disputed_examples':sum(len(set(i['labels']))>1 for i in items)},
            'items':items,
            'limitations':['Public developer meta-evaluation, not clinical validation or hidden test.',
                           'Original binary physician labels retained, not 1-5 clinical scores.',
                           'Common-valid metrics are conditional on both arms succeeding.',
                           'Historical Flash comparisons confound model and prompt; within-phase Pro comparison isolates prompt inputs.']}
    (folder/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def pct(x):return f'{x:.2%}' if x is not None else 'N/A'
    def number(x):return f'{x:.4f}' if x is not None else 'N/A'
    lines=[f'# 第6轮 / {args.phase}：V4 Pro同模型对照','',
           'A=complex-boundary-v2；B=task-scope-v1；两组均V4 Pro，原准则及医生标签不变、首次一次、无重试。','',
           '| 指标 | A旧提示词 | B任务范围修复 |','| --- | --- | --- |',
           f'| 完整通过 | {scores["A"]["completed"]}/{len(items)} | {scores["B"]["completed"]}/{len(items)} |',
           f'| 全计划可交付匹配率 | {pct(scores["A"]["delivered_physician_match_rate"])} | {pct(scores["B"]["delivered_physician_match_rate"])} |',
           f'| 共同有效逐医生一致率 | {pct(scores["common_valid_metrics"]["A"]["agreement"])} | {pct(scores["common_valid_metrics"]["B"]["agreement"])} |',
           f'| Cohen Kappa | {number(scores["common_valid_metrics"]["A"]["cohen_kappa"])} | {number(scores["common_valid_metrics"]["B"]["cohen_kappa"])} |',
           f'| macro-F1 | {number(scores["common_valid_metrics"]["A"]["macro_f1"])} | {number(scores["common_valid_metrics"]["B"]["macro_f1"])} |','',
           f'共同有效{scores["common_valid_examples"]}条。失败保留为未交付，不虚构not_met；有失败时整批正式二分类主指标空缺。',
           f'同题配对95%区间：{result["uncertainty"]}。','',
           '| 子类 | A一致率 | B一致率 | 共同条数 |','| --- | --- | --- | --- |']
    for cat,v in bycat.items():
        lines.append(f'| {cat} | {pct(v["common_valid_metrics"]["A"]["agreement"])} | {pct(v["common_valid_metrics"]["B"]["agreement"])} | {v["common_valid_examples"]} |')
    lines+=['',f'一致标签病例改善：{improved}；退步：{regressed}。样本题号只在本manifest内有效。',
            f'本阶段保守结算{cost:.6f}元，累计{summary["budget"]["accounted_cny"]:.6f}元，剩余{summary["budget"]["remaining_cny"]:.6f}元。',
            f'实际provider返回模型：{reported}。','',
            '模型变更不等于Prompt修复；不能把旧混合70条和新complex40条横比后宣称提升/退步。',
            '仍是给定正确准则的oracle任务，不代表向量检索已验证，也不是临床替代医生。','']
    (folder/'结果解读.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'items','by_category','primary_evaluation'}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
