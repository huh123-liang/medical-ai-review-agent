"""Offline audit supplement; never submits a model request or changes predictions."""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path
from analyze_iteration5 import compare
from analyze_iteration6 import verify_metrics
from run_iteration7 import ROOT, OUT, load_rows, check_freeze, sha, dump


def collect(phase):
    result=json.loads((OUT/phase/'analysis.json').read_text(encoding='utf-8'))
    parts=['fixed-first','fixed-second'] if phase=='fixed' else [phase]
    rows=[row for part in parts for row in load_rows(part)[1]]
    assert [r['example_id'] for r in rows]==[i['example_id'] for i in result['items']]
    expected_labels=sum(len(i['labels']) for i in result['items'])
    assert all(result['scores'][a]['planned_label_comparisons']==expected_labels for a in ('A','B'))
    disagreements=[]
    for row,item in zip(rows,result['items']):
        failed=any(p['status']!='Completed' for p in item['arms'].values())
        wrong=any(p['status']=='Completed' and any((p['judgment']=='met')!=g for g in item['labels']) for p in item['arms'].values())
        if not (wrong or failed):continue
        disagreements.append({**item,'rubric_text':row['rubric']['text'],
                              'context':row['case']['context'],
                              'candidate_output':row['case']['candidate_output']})
    dump(OUT/phase/'disagreement-records.json',disagreements)
    lines=[f'# 第7轮 {phase}：原标签分歧/失败记录','',
           '完整保留原医生标签、准则、上下文和回答。分歧不自动等于模型错误；需要结合医学来源与准则边界复核。未更改任何预测或标签。','']
    for i in disagreements:
        lines += [f'## {i["index"]} / {i["category"]}',
                  f'原医生标签：{i["labels"]}',f'准则：{i["rubric_text"]}','', '### 最新用户请求']
        users=[m['content'] for m in i['context'] if m['role']=='user']
        lines += [users[-1] if users else '(无)', '', '### 两组记录']
        for a,p in i['arms'].items():
            lines += [f'- {a}: {p["status"]} / {p.get("judgment")} / {p.get("error_code")}',
                      f'  task_kind={p.get("task_kind")}; reason={p.get("reason")}',
                      f'  selected sources={p.get("fact_cards_provided",[])}; checks={p.get("fact_checks",[])}']
        lines += ['', '### 候选回答',i['candidate_output'],'']
    (OUT/phase/'分歧与失败原文记录.md').write_text('\n'.join(lines),encoding='utf-8')
    audit={'phase':phase,'planned':len(rows),'labels':expected_labels,
           'disagreement_or_failure_examples':len(disagreements),
           'completed_task_kinds':{a:dict(Counter(i['arms'][a].get('task_kind') for i in result['items'] if i['arms'][a]['status']=='Completed')) for a in ('A','B')},
           'review_required':{a:sum(i['arms'][a].get('review_required',False) for i in result['items']) for a in ('A','B')},
           'binding_relations':{a:dict(Counter(c['relation'] for i in result['items'] for c in i['arms'][a].get('fact_checks',[]))) for a in ('A','B')}}
    if phase=='fixed':
        old=json.loads((ROOT/'reports/iteration-6/fixed/analysis.json').read_text(encoding='utf-8'))
        byid={i['example_id']:i for i in old['items']}
        pairs=[]
        for i in result['items']:
            before=byid[i['example_id']]
            assert before['labels']==i['labels']
            if before['arms']['A']['status']=='Completed' and i['arms']['A']['status']=='Completed':
                pairs.append({**i,'arms':{'A':before['arms']['A'],'B':i['arms']['A']}})
        historical=compare(pairs);verify_metrics(historical)
        audit['historical_iteration6A_vs_iteration7A_common_only']=historical
        audit['historical_caveat']='Same cases and retained physician labels, but noncontemporaneous runs; not an isolated causal estimate of source-ID effects.'
    dump(OUT/phase/'audit-supplement.json',audit)
    return audit


def main():
    check_freeze()
    amend=json.loads((OUT/'analysis-amendment-1.json').read_text())
    assert sha(ROOT/amend['original_analyzer'])==amend['original_sha256']
    assert sha(ROOT/amend['active_analyzer'])==amend['active_sha256']
    summaries=[collect(p) for p in ('fixed','new') if (OUT/p/'analysis.json').exists()]
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
