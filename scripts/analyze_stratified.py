"""Independent accounting and diagnostics of the frozen 70-item experiment; no API calls."""
from __future__ import annotations

import argparse
import difflib
import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from medical_jury.benchmark import binary_metrics
from medical_jury.stratified import TOPICS, load_plan


TOPIC_ZH={'emergency_referrals':'就医紧急程度','communication':'医患／专业沟通','context_seeking':'缺失信息追问',
          'health_data_tasks':'医疗数据任务','hedging':'不确定性表达','global_health':'地域与资源差异','complex_responses':'复杂回答与详略控制'}


def normalized(text):
    text=unicodedata.normalize('NFKC',text).translate(str.maketrans({'\u2019':"'",'\u2018':"'",'\u201c':'"','\u201d':'"'}))
    return ' '.join(text.split())


def metrics_from_counts(tp,tn,fp,fn):
    n=tp+tn+fp+fn
    if not n:
        return None
    po=(tp+tn)/n
    pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/(n*n)
    positive_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    negative_f1=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else None
    return {'agreement':po,'cohen_kappa':(po-pe)/(1-pe) if pe<1 else None,
            'macro_f1':(positive_f1+negative_f1)/2 if positive_f1 is not None and negative_f1 is not None else None,
            'comparisons':n,'matches':tp+tn,'confusion':{'TP':tp,'TN':tn,'FP':fp,'FN':fn}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    folder=args.report.resolve()
    root=Path(__file__).resolve().parents[1]
    summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    plan,rows=load_plan(root,folder/'manifest.json')
    refs={r['example_id']:r for r in map(json.loads,(root/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    conn=sqlite3.connect(f'file:{(root/"data/jury.sqlite").as_posix()}?mode=ro',uri=True)
    conn.row_factory=sqlite3.Row
    topic_counts={t:Counter() for t in TOPICS}
    pairs=[]
    details=[]
    failures=[]
    total_cost=0.0
    tokens=Counter()
    total_gold=0
    ceiling_correct=0
    unanimous_attempted=unanimous_completed=unanimous_matches=0
    for i,(row,pred,entry) in enumerate(zip(rows,summary['predictions'],plan['entries']),1):
        assert row['example_id']==pred['example_id']
        gold=refs[row['example_id']]['binary_labels']
        total_gold+=len(gold)
        ceiling_correct+=max(sum(gold),len(gold)-sum(gold))
        unanimous=len(set(gold))==1
        unanimous_attempted+=unanimous
        topic=entry['topic']
        topic_counts[topic][pred['status']]+=1
        job=dict(conn.execute('SELECT status,error_code,snapshot_json,report_json FROM jobs WHERE id=?',(pred['job_id'],)).fetchone())
        assert job['status']==pred['status']
        snapshot=json.loads(job['snapshot_json'])
        assert snapshot['prompt_version']==summary['prompt_version']
        assert snapshot['retrieval']['mode']=='oracle'
        reason=None
        if pred['status']=='Completed':
            report=json.loads(job['report_json'])
            judgment=report['judgments'][0]
            assert judgment['judgment']==pred['judgment']
            context='\n'.join(m['content'] for m in row['case']['context'])
            assert all(q and q in row['case']['candidate_output'] for q in judgment['answer_evidence'])
            assert all(q and q in context for q in judgment['context_evidence'])
            reason=judgment['reason']
            guess=pred['judgment']=='met'
            pairs.extend((guess,g) for g in gold)
            for g in gold:
                topic_counts[topic]['comparisons']+=1
                topic_counts[topic]['matches']+=guess==g
            if unanimous:
                unanimous_completed+=1
                unanimous_matches+=guess==gold[0]
        else:
            diagnostics=[json.loads(p.read_text(encoding='utf-8')) for p in
                         (root/'data/private/provider-diagnostics'/pred['job_id']).glob('*.json')]
            issue={'index':i,'topic':topic,'job_id':pred['job_id'],'error_code':job['error_code'],
                   'has_diagnostic':bool(diagnostics),'mismatched_quotes':[]}
            for diagnostic in diagnostics:
                excerpt=diagnostic.get('redacted_output_excerpt')
                if not excerpt or diagnostic.get('excerpt_truncated'):
                    continue
                try:
                    output=json.loads(excerpt)
                except ValueError:
                    continue
                reason=output.get('reason')
                for field,text in [('answer_evidence',row['case']['candidate_output']),
                                   ('context_evidence','\n'.join(m['content'] for m in row['case']['context']))]:
                    for quote in output.get(field,[]):
                        if quote not in text:
                            nearest=max(text.splitlines(),key=lambda line:difflib.SequenceMatcher(None,quote,line).ratio(),default='')
                            issue['mismatched_quotes'].append({'field':field,'quote':quote,
                                'typography_whitespace_equivalent':normalized(quote) in normalized(text),
                                'nearest_source_line':nearest[:500]})
            failures.append(issue)
        fees=list(conn.execute('SELECT accounted,usage_json,state FROM costs WHERE job_id=?',(pred['job_id'],)))
        assert len(fees)==1, 'First-pass protocol unexpectedly retried a call.'
        for fee in fees:
            total_cost+=fee['accounted']
            assert fee['state']=='settled'
            usage=json.loads(fee['usage_json'])
            tokens['prompt_tokens']+=usage['prompt_tokens']
            tokens['completion_tokens']+=usage['completion_tokens']
        details.append({'index':i,'topic':topic,'category':row['category'],'heuristic_scenario':entry['heuristic_scenario'],
                        'status':pred['status'],'judgment':pred['judgment'],'physician_labels':gold,
                        'physicians_agree':unanimous,'job_id':pred['job_id'],'reason':reason})
    conn.close()
    counts=Counter()
    for guess,gold in pairs:
        counts[('TP' if gold else 'FP') if guess else ('FN' if gold else 'TN')]+=1
    independent=metrics_from_counts(*(counts[k] for k in ('TP','TN','FP','FN')))
    implementation=binary_metrics(pairs)
    for metric in ('agreement','macro_f1','cohen_kappa'):
        assert abs(independent[metric]-implementation[metric])<1e-12
    assert abs(total_cost-summary['batch_accounted_cny'])<1e-6
    completed=sum(d['status']=='Completed' for d in details)
    assert len(details)==70 and len({r['case']['group_id'] for r in rows})==70
    assert not set(plan['excluded_prior_groups']) & {r['case']['group_id'] for r in rows}
    assert completed/70==summary['evaluation']['coverage']
    assert summary['budget']['accounted_cny']<=30 and total_cost<=5
    output={'review_status':'Share with caveats','planned':70,'completed':completed,'failed':70-completed,
            'distinct_conversations':70,'prompt_version':summary['prompt_version'],
            'json_and_schema_passed':completed+sum(f['error_code']=='OUTPUT_EVIDENCE_OR_PROTOCOL_INVALID' for f in failures),
            'primary_metrics':summary['evaluation']['metrics'],
            'successful_subset_diagnostic':independent,'budget':summary['budget'],
            'batch_accounted_cny':round(total_cost,6),'tokens':dict(tokens),
            'annotation_ceiling_on_planned_sample':ceiling_correct/total_gold,
            'unanimous_labels':{'planned':unanimous_attempted,'completed':unanimous_completed,'matching':unanimous_matches},
            'by_topic':{k:dict(v) for k,v in topic_counts.items()},'failure_details':failures,'items':details}
    (folder/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第二轮分层真实测试与工程更新','',
        '日期：2026-10-24。独立Agent项目；原MVP2平台未修改。','',
        '## 结论','',f'- 70条首次请求、70个不同对话；{completed}条完整通过，{70-completed}条失败，覆盖率{completed/70:.1%}。',
        f'- JSON及严格字段结构校验通过{output["json_and_schema_passed"]}/70；引用校验不能放宽成“看起来差不多”。',
        f'- 本轮保守记账{total_cost:.6f}元；累计{summary["budget"]["accounted_cny"]:.6f}元；30元总预算剩余{summary["budget"]["remaining_cny"]:.6f}元。不是供应商账单。',
        '- 有失败项，按预定协议不发布整批正式agreement、Kappa或macro-F1。','',
        '## 本轮已经实现的工程改进','',
        '- Prompt升级medical-jury-criterion-v2-strict，给出完整JSON模板及字段类型、必填、长度与引用规则。',
        '- 本地strict schema要求5个字段全部存在；拒绝类型自动转换、多余字段、重复键、Markdown包裹和非JSON输出。',
        '- 错误拆为JSON语法、字段结构、引用/协议、空输出、截断、传输与响应封装错误。',
        '- 失败日志保存错误类型、位置/字段、输出hash及长度；仅公开基准可保存脱敏限长片段，私人病例不保存输出原文。',
        '- 不记录请求头、密钥或原始HTTP错误正文；格式失败不自动重试，不覆盖首轮失败记录。',
        '- 本地62项回归测试通过（后续新增测试以最新pytest为准）。','',
        '## 分层方案与覆盖','',
        '固定seed=20261024，7个官方任务大类各10条；每类预先配置4条医生一致满足、4条一致不满足、2条分歧项。抽样使用标签，但Agent输入不含标签。排除首轮5个对话，同对话只取一条。',
        '按上下文关键词检查临床主题多样性：慢性/代谢/肾脏9条、心理健康9条、妇女儿童9条、用药11条、心肺8条、感染/皮肤11条、其他13条。这些是覆盖标签，不是医生确诊或官方疾病标注。',
        '有意平衡的开发集诊断实验，不代表真实临床发生率，也不是隐藏测试集验收。','',
        '| 医疗任务类别 | 完整通过/计划 | 失败 | 成功子集逐医生一致（仅诊断） |',
        '| --- | --- | --- | --- |']
    for topic in TOPICS:
        c=topic_counts[topic]
        lines.append(f'| {TOPIC_ZH[topic]} | {c["Completed"]}/10 | {c["Failed"]} | {c["matches"]}/{c["comparisons"]} |')
    lines+=['','## 如何理解质量分数','',
        f'在成功的{completed}条内，与公开医生的{independent["comparisons"]}条判断逐一比较，{independent["matches"]}次一致（{independent["agreement"]:.1%}）。仅成功子集诊断：macro-F1={independent["macro_f1"]:.4f}，二分类Cohen Kappa={independent["cohen_kappa"]:.4f}。这些不是整批正式分数；失败样本可能非随机。',
        f'预先选样中有14条医生分歧项，部分题目有3位医生标注；按本轮实际{total_gold}条医生标注逐一比较，固定二分类预测能达到的一致率上限约为{ceiling_correct/total_gold:.1%}，不是100%。',
        f'医生一致的56条中，完成{unanimous_completed}条，其中{unanimous_matches}条与一致标签相符；有明确错判，不能称已达到医生水平。',
        '未报告1–5分MAE、Spearman或加权Kappa；本数据没有相应的1–5分标注。','',
        '## 失败诊断','']
    for failure in failures:
        lines+=[f'### 第{failure["index"]}条 / {TOPIC_ZH[failure["topic"]]}','',
                f'错误码：{failure["error_code"]}；失败日志存在：{failure["has_diagnostic"]}。']
        for mismatch in failure['mismatched_quotes']:
            lines += [f'- {mismatch["field"]}：`{mismatch["quote"]}`',
                      f'  - 仅Unicode标点/空格规范化后可匹配：{mismatch["typography_whitespace_equivalent"]}。',
                      f'  - 最相近原文行（不是自动修复依据）：`{mismatch["nearest_source_line"]}`']
        lines += ['']
    lines+=['## 边界与下一步','',
        '1. 本轮仍为给定准则的oracle裁判测试，不是RAG检索成功率。',
        '2. 与首轮的样本分布不同，不能将完成率变化直接归因于Prompt升级；需同题配对对照才能主张提升。',
        '3. JSON格式暂未失败不等于永久不会失败；仍需保留严格验证及费用上限。',
        '4. 引用失败后可研究“可追溯、唯一定位”的排版规范化，但不能模糊匹配接受内容，也不能改变原判断刷分。',
        '5. 后续在开发集上归类医生一致组中的错误，再设计受控的few-shot或检索实验；先不要接入旧平台或扩展多Agent。','']
    (folder/'第二轮测试解读.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in output.items() if k not in {'items','failure_details'}},ensure_ascii=False,indent=2))
    print(json.dumps({'failure_details':failures},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
