"""Offline meta-evaluation and source-consistency accounting; never calls a model."""
from __future__ import annotations
import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from analyze_iteration5 import compare,bootstrap
from analyze_iteration6 import verify_metrics
from medical_jury.benchmark import score_run
from medical_jury.domain import digest
from run_iteration7 import OUT,ROOT,PROFILES,load_rows,check_freeze,sha,dump


def read_phase(phase):
    folder=OUT/phase;summary=json.loads((folder/'summary.json').read_text())
    assert summary['state']=='Finished'
    pre=json.loads((folder/'preflight.json').read_text());check_freeze()
    assert pre['freeze']==json.loads((OUT/'freeze.json').read_text())
    assert all(sha(folder/'frozen-code'/name)==h for name,h in pre['freeze']['core'].items())
    plan,rows=load_rows(phase);assert plan['manifest_hash']==pre['manifest_hash']
    db=sqlite3.connect(str(ROOT/'data/jury.sqlite'));db.row_factory=sqlite3.Row
    predictions={a:{p['index']:p for p in summary['predictions'] if p['arm']==a} for a in PROFILES}
    items=[];cost=0;models=Counter()
    for index,row in enumerate(rows,1):
        item={'phase':phase,'index':index,'example_id':row['example_id'],'group_id':row['case']['group_id'],
              'category':row['category'],'topic':plan['entries'][index-1].get('topic','complex_responses'),'arms':{}}
        for a in PROFILES:
            p=dict(predictions[a][index]);assert p['input_hash']==digest(row)
            if p['job_id']:
                job=dict(db.execute('SELECT * FROM jobs WHERE id=?',(p['job_id'],)).fetchone())
                snap=json.loads(job['snapshot_json'])
                assert snap['case']==row['case'] and snap['rubrics']==[row['rubric']]
                assert snap['judge_profile']==PROFILES[a] and snap['model']=='deepseek-v4-pro'
                assert snap['message_hashes']==[p['messages_hash']]
                costs=list(db.execute('SELECT * FROM costs WHERE job_id=?',(p['job_id'],)))
                assert len(costs)<=1;cost+=sum(c['accounted'] for c in costs)
                p['fact_cards_provided']=[c['id'] for c in snap['fact_review']['cards']]
                for e in db.execute("SELECT detail_json FROM audit WHERE job_id=? AND event='provider_model_metadata'",(p['job_id'],)):
                    models[json.loads(e[0])['reported_model']]+=1
                if p['status']=='Completed':
                    report=json.loads(job['report_json']);judgment=report['judgments'][0]
                    assert p['judgment']==judgment['judgment']
                    review=report['structured_reviews'][0]
                    p.update(reason=judgment['reason'],review_required=judgment['review_required'],
                             answer_evidence=judgment['answer_evidence'],task_kind=review['task_kind'],
                             task_spec=review['task_spec'],fact_checks=review['fact_checks'],
                             release_status=report['release_status'],weighted_score=report['weighted_score'])
                else:
                    files=(ROOT/'data/private/provider-diagnostics'/p['job_id']).glob('*.json')
                    p['diagnostics']=[{'code':(d:=json.loads(f.read_text(encoding='utf-8')))['error_code'],
                                       'details':d.get('details')} for f in files]
            item['arms'][a]=p
        items.append(item)
    db.close()
    assert abs(cost-summary['phase_accounted_cny'])<1e-6
    return items,summary,round(cost,6),dict(models)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','fixed','new']);args=p.parse_args()
    if args.phase=='smoke':
        items,summary,cost,models=read_phase('smoke')
        expected=json.loads((ROOT/'data/private/iteration7-fixture-expected.json').read_text())['expected']
        for i in items:i['labels']=[expected[i['example_id']]]
        scores=compare(items)
        result={'phase':'smoke','status':'SYNTHETIC_SOURCE_CONSISTENCY_NOT_PHYSICIAN_VALIDATION',
                'scores':scores,'phase_accounted_cny':cost,'budget':summary['budget'],'models':models,'items':items}
        dump(OUT/'smoke/analysis.json',result)
        # This gate checks a functioning output adapter, not physician-level quality.
        passed=all(scores[a]['completed']>=7 for a in PROFILES)
        gate={'status':'PASS_TO_FIXED_RESEARCH_REGRESSION' if passed else 'BLOCKED_FIX_WIRE_BEFORE_BENCHMARK',
              'goal_alignment':'Source-ID, task and source-binding sanity only; fixture expectations are not doctor labels.',
              'architecture_alignment':'One bounded Pro call, no autonomous tools, compatible canonical Judgment.',
              'platform_alignment':'Independent project only. No clinical release.',
              'scores':scores,'budget':summary['budget'],'promoted':False}
        dump(OUT/'gate-smoke.json',gate)
        print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False,indent=2));return
    phases=['fixed-first','fixed-second'] if args.phase=='fixed' else ['new']
    parts=[read_phase(s) for s in phases];items=[];cost=0;models=Counter()
    for values,summary,fee,reported in parts:
        cost+=fee;models.update(reported)
        for item in values:
            item['batch_index']=item['index'];item['index']=len(items)+1;items.append(item)
    refs={r['example_id']:r for r in map(json.loads,(ROOT/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    for i in items:i['labels']=refs[i['example_id']]['binary_labels']
    assert len({i['group_id'] for i in items})==len(items)
    assert len(items)==(70 if args.phase=='fixed' else 32)
    scores=compare(items);verify_metrics(scores)
    bycat={c:compare([i for i in items if i['category']==c]) for c in dict.fromkeys(i['category'] for i in items)}
    strata=[{**i,'category':i['topic']} for i in items] if args.phase=='fixed' else items
    uncertainty=bootstrap(strata)
    uncertainty['resampling']='Paired by conversation, within seven topic strata for fixed70 or four criterion-category strata for new32.'
    complex_rows=[i for i in items if 'complex_responses' in i['category']]
    improved=[];regressed=[]
    for i in items:
        if len(set(i['labels']))!=1:continue
        good={a:i['arms'][a]['status']=='Completed' and (i['arms'][a]['judgment']=='met')==i['labels'][0] for a in PROFILES}
        if good['B'] and not good['A']:improved.append(i['index'])
        if good['A'] and not good['B']:regressed.append(i['index'])
    source_checks={a:sum(len(i['arms'][a].get('fact_checks',[])) for i in items) for a in PROFILES}
    result={'phase':args.phase,'model':'deepseek-v4-pro','profiles':PROFILES,'planned':len(items),
            'scores':scores,'complex_scores':compare(complex_rows),'by_category':bycat,'uncertainty':uncertainty,
            'primary_evaluation':{a:score_run(ROOT,[i['arms'][a] for i in items],'deepseek') for a in PROFILES},
            'errors':{a:dict(Counter(i['arms'][a]['error_code'] for i in items if i['arms'][a]['status']!='Completed')) for a in PROFILES},
            'provider_reported_models':dict(models),'fact_checks_bound':source_checks,
            'fact_hit_examples':{a:sum(bool(i['arms'][a].get('fact_cards_provided')) for i in items) for a in PROFILES},
            'unanimous_improved_indices':improved,'unanimous_regressed_indices':regressed,
            'phase_accounted_cny':round(cost,6),'budget':parts[-1][1]['budget'],'items':items,
            'limitations':['Public development meta-evaluation, not hidden or clinical validation.',
                           'Both arms use evidence IDs; A/B changes task and fact components together, not separate causal effects.',
                           'Fact card hit and ID binding do not prove claim correctness or entire-answer coverage.',
                           'With failures, common-valid metrics are conditional and full-batch primary metrics withheld.']}
    folder=OUT/args.phase;folder.mkdir(exist_ok=True)
    dump(folder/'analysis.json',result)
    def pct(x):return f'{x:.2%}' if x is not None else 'N/A'
    lines=[f'# 第7轮 / {args.phase}：V4 Pro ID协议同题对照','',
           'A=证据ID+原complex范围说明；B=证据ID+TaskSpec v2+本地4卡来源。均单次Pro、原医生标签、无重试。','',
           '| 指标 | A | B |','| --- | --- | --- |',
           f'| 完整通过 | {scores["A"]["completed"]}/{len(items)} | {scores["B"]["completed"]}/{len(items)} |',
           f'| 全计划可交付标签匹配率 | {pct(scores["A"]["delivered_physician_match_rate"])} | {pct(scores["B"]["delivered_physician_match_rate"])} |',
           f'| 共同有效逐医生一致率 | {pct(scores["common_valid_metrics"]["A"]["agreement"])} | {pct(scores["common_valid_metrics"]["B"]["agreement"])} |','',
           f'共同有效{scores["common_valid_examples"]}条；配对95%区间：{uncertainty}。',
           '有失败时整批主二分类指标保留为空。不能仅报成功子集，也不把跨轮不同样本横比当版本提升。','',
           '| 子类 | A一致率 | B一致率 | 共同条数 |','| --- | --- | --- | --- |']
    for c,v in bycat.items():lines.append(f'| {c} | {pct(v["common_valid_metrics"]["A"]["agreement"])} | {pct(v["common_valid_metrics"]["B"]["agreement"])} | {v["common_valid_examples"]} |')
    lines+=['',f'一致标签改善题号{improved}；退步题号{regressed}。题号仅本manifest内有效。',
            f'提供事实卡的题数{result["fact_hit_examples"]}，有效claim-ID→fact-ID绑定数{source_checks}；这不是医学正确率。',
            f'本阶段保守结算{cost:.6f}元，累计{result["budget"]["accounted_cny"]:.6f}元。',
            '医疗事实覆盖有限；原100条Rubrics索引未改。来源fixture与医生标签实验严格分开。','']
    (folder/'结果解读.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'items','by_category','primary_evaluation'}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
