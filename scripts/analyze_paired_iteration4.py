"""Offline accounting, strict paired comparisons and a non-clinical iteration gate."""
import hashlib
import json
import runpy
import sqlite3
from collections import Counter
from pathlib import Path
from medical_jury.benchmark import score_run,binary_metrics
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/iteration-4'; FOLDER=OUT/'paired'

def main():
    summary=json.loads((FOLDER/'summary.json').read_text(encoding='utf-8'))
    assert summary['state']=='Finished','Do not publish interim comparisons.'
    pre=json.loads((FOLDER/'preflight.json').read_text(encoding='utf-8'))
    for path,sha in pre['code_sha256'].items():
        assert hashlib.sha256((FOLDER/'frozen-code'/Path(path).name).read_bytes()).hexdigest()==sha
    assert hashlib.sha256((ROOT/'scripts/run_paired_iteration4.py').read_bytes()).hexdigest()==pre['runner_sha256']
    _,rows=load_plan(ROOT,FOLDER/'manifest.json')
    references={r['example_id']:r for r in map(json.loads,(ROOT/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    old=json.loads((ROOT/'reports/iteration-3/paired-analysis.json').read_text(encoding='utf-8'))
    helpers=runpy.run_path(str(ROOT/'scripts/analyze_paired_iteration3.py'))
    metrics=helpers['metrics'];bootstrap=helpers['paired_bootstrap']
    arms={a:{p['index']:p for p in summary['predictions'] if p['arm']==a} for a in ('A','B')}
    assert all(len(p)==70 for p in arms.values())
    conn=sqlite3.connect(f'file:{(ROOT/"data/jury.sqlite").as_posix()}?mode=ro',uri=True);conn.row_factory=sqlite3.Row
    cost=0;details=[];groups=[];improved=[];regressed=[];provider_models=Counter();errors=[]
    for index,row in enumerate(rows,1):
        labels=references[row['example_id']]['binary_labels']
        assert labels==old['items'][index-1]['labels'],'Original physician labels changed.'
        item={'index':index,'topic':old['items'][index-1]['topic'],'physician_labels':labels,'arms':{}}
        for arm in arms:
            pred=arms[arm][index]; detail=dict(pred)
            if pred['job_id']:
                job=dict(conn.execute('SELECT * FROM jobs WHERE id=?',(pred['job_id'],)).fetchone())
                assert job['status']==pred['status']
                snap=json.loads(job['snapshot_json']);assert snap['case']==row['case'] and snap['rubrics']==[row['rubric']]
                assert snap['judge_profile']==pre['profiles'][arm] and snap['evidence_policy']=='anchored-v2'
                fees=list(conn.execute('SELECT * FROM costs WHERE job_id=?',(pred['job_id'],)))
                assert len(fees)==1;cost+=sum(f['accounted'] for f in fees)
                events=list(conn.execute('SELECT * FROM audit WHERE job_id=?',(pred['job_id'],)))
                for event in events:
                    if event['event']=='provider_model_metadata':provider_models[json.loads(event['detail_json'])['reported_model']]+=1
                if pred['status']=='Completed':
                    report=json.loads(job['report_json']);j=report['judgments'][0]
                    assert j['judgment']==pred['judgment']
                    detail.update(reason=j['reason'],review_required=report['review_required'],
                        release_status=report['release_status'],weighted_score=report['weighted_score'],
                        research_weighted_score=report['research_weighted_score'],
                        safety_flags=len(report['safety_screen']['findings']),
                        ambiguous_quotes=sum(t.get('ambiguous_count',0) for t in report['evidence_resolution']),
                        checklists=report['criterion_checklists'])
                    for trace in report['evidence_resolution']:
                        for span in trace['spans']:
                            for candidate in span.get('candidates',[span]):
                                source=row['case']['candidate_output'] if candidate['field']=='answer_evidence' else row['case']['context'][candidate['source_index']]['content']
                                assert source[candidate['start_char']:candidate['end_char']]==candidate['source_quote']
                                assert hashlib.sha256(source.encode()).hexdigest()==candidate['source_sha256']
                    if arm=='B':assert len(report['criterion_checklists'])==1
                else:
                    logs=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'data/private/provider-diagnostics'/pred['job_id']).glob('*.json')]
                    detail['failure_details']=[{'error_code':d['error_code'],'details':d.get('details'),
                         'public_evidence_excerpts':d.get('public_evidence_excerpts')} for d in logs]
                    errors.append({'index':index,'arm':arm,'job_id':pred['job_id'],'details':detail['failure_details']})
            item['arms'][arm]=detail
        if all(item['arms'][a]['status']=='Completed' for a in arms):
            group={a:[(item['arms'][a]['judgment']=='met',g) for g in labels] for a in arms};groups.append(group)
            if len(set(labels))==1:
                ac=group['A'][0][0]==labels[0];bc=group['B'][0][0]==labels[0]
                if not ac and bc:improved.append(index)
                if ac and not bc:regressed.append(index)
        details.append(item)
    conn.close();assert abs(cost-summary['batch_accounted_cny'])<1e-6 and cost<=8 and summary['budget']['accounted_cny']<=30
    common={a:metrics([p for g in groups for p in g[a]]) for a in arms}
    # Independently recompute agreement and kappa from direct 2x2 counts.
    for arm,m in common.items():
        c=m['confusion'];n=sum(c.values());po=(c['TP']+c['TN'])/n
        pe=((c['TP']+c['FP'])*(c['TP']+c['FN'])+(c['TN']+c['FN'])*(c['TN']+c['FP']))/(n*n)
        assert abs(po-m['agreement'])<1e-12 and abs((po-pe)/(1-pe)-m['cohen_kappa'])<1e-12
    by_topic={}
    for topic in dict.fromkeys(i['topic'] for i in details):
        eligible=[i for i in details if i['topic']==topic and all(i['arms'][a]['status']=='Completed' for a in arms)]
        by_topic[topic]={'common_examples':len(eligible),**{a:metrics([(i['arms'][a]['judgment']=='met',g) for i in eligible for g in i['physician_labels']]) for a in arms}}
    result={'protocol':pre['protocol'],'planned_per_arm':70,'common_valid_examples':len(groups),
        'status_counts':{a:dict(Counter(p['status'] for p in arm.values())) for a,arm in arms.items()},
        'arm_evaluation':{a:score_run(ROOT,list(arm.values()),'deepseek') for a,arm in arms.items()},
        'common_valid_metrics':common,'paired_delta_95ci':bootstrap(groups),
        'by_topic_common_valid':by_topic,'unanimous_improved':improved,'unanimous_regressed':regressed,
        'failure_details':errors,'provider_reported_models':dict(provider_models),'items':details,
        'batch_accounted_cny':round(cost,6),'budget':summary['budget'],
        'default_prompt_promotion':False,
        'conclusion':'Global v4 not promoted. Scoped default retains identical v2 input except source-backed safety alerts; v4 remains experimental outside that narrow route.',
        'limitations':pre['limitations']+['Safety hints and clause checking jointly changed; not component-specific causal proof.',
            'Model-vs-doctor metrics are not safety-gate sensitivity or clinical correctness.','Incomplete-arm primary scores withheld; common-valid comparison is conditional.']}
    (OUT/'paired-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第4轮：引用歧义、安全门控与复杂回答检查','','## 完成内容及边界','',
        '- 引用v2保留所有候选位置并标复核，不擅选第一处；140份历史输出离线回放均可定位，旧判定/理由不变。',
        '- 新增一条来源可审阅的窄范围安全规则；历史第27条现在会阻止可用分数放行，但原模型met和医生标签保留。不代表全面医学安全检测。',
        '- v4在同一次Agent调用中输出任务类型和逐条准则检查，本地拦截缺项、重复、逻辑矛盾；最终公开判定仍为原五字段。没有新增第二Agent。',
        '- Rubrics适用性与检索测试另有实施方案；本轮没有改现有向量索引或原平台。','',
        '## 同70条开发集的固定A/B','',
        '两组共用引用v2、安全筛查、Flash、temperature=0和1200输出上限；A为v2，B为v4检查表+窄范围安全提示。各70次首次请求，不重试，原标签保留。B同时改变两部分，不能将效果单独归因于其中之一。','',
        '| 指标 | A 原v2 | B v4检查表 |','| --- | --- | --- |',
        f'| 完整通过 | {result["status_counts"]["A"].get("Completed",0)}/70 | {result["status_counts"]["B"].get("Completed",0)}/70 |',
        f'| 共同有效：逐医生一致率 | {common["A"]["agreement"]:.2%} | {common["B"]["agreement"]:.2%} |',
        f'| 共同有效：Cohen Kappa | {common["A"]["cohen_kappa"]:.4f} | {common["B"]["cohen_kappa"]:.4f} |',
        f'| 共同有效：macro-F1 | {common["A"]["macro_f1"]:.4f} | {common["B"]["macro_f1"]:.4f} |','',
        f'共同有效{len(groups)}/70条、{common["A"]["n_physician_comparisons"]}次医生比较。失败未删除，不完整组不发布整批主要分数。配对95%区间：{result["paired_delta_95ci"]}。',
        f'医生一致题改善：{improved}；退步：{regressed}。这些只是开发样本观察，不能当临床或未见样本验证。','',
        '## 按任务检查，避免平均分掩盖退步','','| 任务 | 共同样本 | A一致率 | B一致率 |','| --- | --- | --- | --- |']
    for topic,t in by_topic.items():lines.append(f'| {topic} | {t["common_examples"]} | {t["A"]["agreement"]:.1%} | {t["B"]["agreement"]:.1%} |')
    lines+=['','## 重点案例：原判定、来源风险与可用性分开','','| 题号 | 医生原标签 | A判定 | B判定 | A/B放行状态 |','| --- | --- | --- | --- | --- |']
    for i in (7,27,34,40,61,64,66,67):
        item=details[i-1];a=item['arms']['A'];b=item['arms']['B']
        lines.append(f'| {i} | {item["physician_labels"]} | {a["judgment"]} | {b["judgment"]} | {a.get("release_status",a["status"])} / {b.get("release_status",b["status"])} |')
    lines+=['','不得将NeedsReview解释为模型判断已经正确；它表示不得直接使用该汇总分。完整理由、检查表和错误见 paired-analysis.json 与每条job报告。','',
        '## 费用、关卡与下一步','',f'固定A/B保守记账{cost:.6f}元；截至该实验结束累计{summary["budget"]["accounted_cny"]:.6f}元；当时剩余{summary["budget"]["remaining_cny"]:.6f}元。此处不含后续6条冒烟；最新合计见completion-summary.json。供应商账单为准。',
        'gate-1验证引用，gate-2验证安全门控，experiment-gate验证付费条件，gate-3复核评分实验，gate-4核对检索计划。',
        '默认引用升级anchored-v2。全局v4不推广；后续保守scoped-review-v1路由在没有确定性安全告警时保留完全相同的v2输入，仅安全规则触发时调用v4一次。69/70原开发输入与v2一致；复杂回答仍走v2，不宣称已解决遗留语义错判。',
        '下一步先看新增失败和未见开发样本，再实施已制定的适用性/检索方案。不要现在接MVP2或扩展多Agent。','']
    (OUT/'第四轮结果与架构核对.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'gate-3.json').write_text(json.dumps({'status':'COMPLETED_DEV_EXPERIMENT_WITH_LIMITATIONS',
        'objective_alignment':'Structured single-Agent checks and explicit review improve auditability; no replacement-of-doctor claim.',
        'architecture_alignment':'Same workflow; versioned schema and snapshots, deterministic evidence/safety/check consistency, no extra Agent.',
        'platform_alignment':'Original platform unchanged; keep automatic results separate from physician labels and locked scores.',
        'default_prompt_promotion':False,'global_v4_promotion':False,'scoped_default':'scoped-review-v1',
        'evidence_policy':'anchored-v2','clinical_validation':False,
        'budget':summary['budget'],'next':'Deliver applicability/retrieval plan only, without changing current index.'},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'items','by_topic_common_valid','failure_details'}},indent=2))

if __name__=='__main__':main()
