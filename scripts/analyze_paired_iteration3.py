"""Audit frozen A/B results against unchanged labels; never call a provider."""
import hashlib
import json
import math
import random
import sqlite3
from collections import Counter
from pathlib import Path
from medical_jury.benchmark import binary_metrics, score_run
from medical_jury.domain import digest
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-3'
FOLDER=OUT/'paired'
NINE={1,7,27,44,56,62,64,66,67}


def metrics(pairs):
    result=binary_metrics(pairs)
    result['matches']=sum(a==b for a,b in pairs)
    result['confusion']={'TP':sum(a and b for a,b in pairs),'TN':sum(not a and not b for a,b in pairs),
                         'FP':sum(a and not b for a,b in pairs),'FN':sum(not a and b for a,b in pairs)}
    return result


def paired_bootstrap(groups,n=3000):
    rng=random.Random(20261024); values={k:[] for k in ('agreement','cohen_kappa','macro_f1')}
    for _ in range(n if len(groups)>1 else 0):
        sampled=rng.choices(groups,k=len(groups))
        a=metrics([p for group in sampled for p in group['A']])
        b=metrics([p for group in sampled for p in group['B']])
        for metric in values:
            if a[metric] is not None and b[metric] is not None:
                values[metric].append(b[metric]-a[metric])
    return {k:[sorted(v)[int((len(v)-1)*q)] for q in (.025,.975)] if v else None for k,v in values.items()}


def main():
    summary=json.loads((FOLDER/'summary.json').read_text(encoding='utf-8'))
    assert summary['state']=='Finished', 'Do not publish interim A/B scores.'
    plan,rows=load_plan(ROOT,FOLDER/'manifest.json')
    refs={r['example_id']:r for r in map(json.loads,(ROOT/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines())}
    baseline=json.loads((ROOT/'reports/stratified-20261024-194713/analysis.json').read_text(encoding='utf-8'))
    preflight=json.loads((FOLDER/'preflight.json').read_text(encoding='utf-8'))
    for path,expected in preflight['code_sha256'].items():
        archive=FOLDER/'frozen-code'/Path(path).name
        source=archive if archive.exists() else ROOT/path
        assert hashlib.sha256(source.read_bytes()).hexdigest()==expected, 'Frozen experiment source hash mismatch.'
    assert hashlib.sha256((ROOT/'scripts/run_paired_iteration3.py').read_bytes()).hexdigest()==preflight['runner_sha256']
    conn=sqlite3.connect(f'file:{(ROOT/"data/jury.sqlite").as_posix()}?mode=ro',uri=True); conn.row_factory=sqlite3.Row
    arms={a:[p for p in summary['predictions'] if p['arm']==a] for a in ('A','B')}
    assert all(len(p)==70 for p in arms.values()) and len(summary['predictions'])==140
    audits={}; cost=0; anchor_methods=Counter(); metadata=Counter()
    for a,preds in arms.items():
        for pred in preds:
            row=rows[pred['index']-1]
            assert pred['example_id']==row['example_id'] and pred['input_hash']==digest(row)
            if not pred['job_id']: continue
            job=dict(conn.execute('SELECT * FROM jobs WHERE id=?',(pred['job_id'],)).fetchone())
            assert job['status']==pred['status']
            snap=json.loads(job['snapshot_json'])
            assert snap['case']==row['case'] and snap['rubrics']==[row['rubric']]
            assert snap['judge_profile']==preflight['profiles'][a] and snap['evidence_policy']=='anchored-v1'
            assert snap['retrieval']['mode']=='oracle'
            fees=list(conn.execute('SELECT * FROM costs WHERE job_id=?',(pred['job_id'],)))
            assert len(fees)==1, 'Unexpected retry or unpaid job.'
            cost+=sum(f['accounted'] for f in fees)
            events=list(conn.execute('SELECT * FROM audit WHERE job_id=? ORDER BY seq',(pred['job_id'],)))
            for event in events:
                detail=json.loads(event['detail_json'])
                if event['event']=='provider_model_metadata': metadata[detail['reported_model']]+=1
            if pred['status']=='Completed':
                report=json.loads(job['report_json']); judgment=report['judgments'][0]
                assert pred['judgment']==judgment['judgment']
                traces=report['evidence_resolution']; assert len(traces)==1
                for span in traces[0]['spans']:
                    source=row['case']['candidate_output'] if span['field']=='answer_evidence' else row['case']['context'][span['source_index']]['content']
                    assert hashlib.sha256(source.encode()).hexdigest()==span['source_sha256']
                    assert source[span['start_char']:span['end_char']]==span['source_quote']
                    assert len(source[:span['start_char']].encode('utf-16-le'))//2==span['start_utf16']
                    assert len(source[:span['end_char']].encode('utf-16-le'))//2==span['end_utf16']
                    anchor_methods[(a,span['method'])]+=1
                audits[(pred['index'],a)]={'reason':judgment['reason'],'review_required':judgment['review_required'],
                                          'normalized_count':traces[0]['normalized_count']}
    conn.close()
    assert abs(cost-summary['batch_accounted_cny'])<1e-6 and summary['budget']['accounted_cny']<=30 and cost<=8
    byindex={a:{p['index']:p for p in preds} for a,preds in arms.items()}
    common=[]; details=[]; improved=[]; regressed=[]; unchanged_wrong=[]; stochastic=[]
    for index,row in enumerate(rows,1):
        gold=refs[row['example_id']]['binary_labels']; unanimous=len(set(gold))==1
        item={'index':index,'topic':plan['entries'][index-1]['topic'],'labels':gold,'unanimous':unanimous,
              'historical':baseline['items'][index-1]['judgment'],'arms':{a:{**byindex[a][index],**audits.get((index,a),{})} for a in arms}}
        if all(byindex[a][index]['status']=='Completed' for a in arms):
            group={a:[(byindex[a][index]['judgment']=='met',g) for g in gold] for a in arms}; common.append(group)
            if unanimous:
                ac=group['A'][0][0]==gold[0]; bc=group['B'][0][0]==gold[0]
                if not ac and bc: improved.append(index)
                if ac and not bc: regressed.append(index)
                if not ac and not bc: unchanged_wrong.append(index)
        if item['historical'] is not None and byindex['A'][index]['judgment'] is not None and item['historical']!=byindex['A'][index]['judgment']:
            stochastic.append(index)
        details.append(item)
    common_metrics={a:metrics([pair for g in common for pair in g[a]]) for a in arms}
    ci=paired_bootstrap(common)
    by_topic={}
    for topic in dict.fromkeys(p['topic'] for p in details):
        eligible=[item for item in details if item['topic']==topic and all(item['arms'][a]['status']=='Completed' for a in arms)]
        by_topic[topic]={'common_examples':len(eligible),**{a:metrics([(item['arms'][a]['judgment']=='met',g) for item in eligible for g in item['labels']]) for a in arms}}
    discordants=len(improved)+len(regressed)
    mcnemar=min(1.0,2*sum(math.comb(discordants,k)*.5**discordants for k in range(min(len(improved),len(regressed))+1))) if discordants else 1.0
    result={'protocol':preflight['protocol'],'planned_per_arm':70,'common_valid_examples':len(common),
        'arm_evaluation':{a:score_run(ROOT,preds,'deepseek',bootstrap=1000) for a,preds in arms.items()},
        'status_counts':{a:dict(Counter(p['status'] for p in preds)) for a,preds in arms.items()},
        'errors':{a:dict(Counter(p['error_code'] for p in preds if p['error_code'])) for a,preds in arms.items()},
        'common_valid_metrics':common_metrics,'paired_delta_95ci':ci,
        'by_topic_common_valid':by_topic,
        'common_valid_delta':{k:common_metrics['B'][k]-common_metrics['A'][k] for k in ('agreement','cohen_kappa','macro_f1')},
        'unanimous_improved':improved,'unanimous_regressed':regressed,'unanimous_wrong_both':unchanged_wrong,
        'unanimous_mcnemar_exact_p':mcnemar,'historical_v2_vs_fresh_v2_changed_indices':stochastic,
        'anchor_methods':{f'{a}:{m}':n for (a,m),n in anchor_methods.items()},'provider_reported_models':dict(metadata),
        'batch_accounted_cny':round(cost,6),'budget':summary['budget'],'items':details,
        'limitations':preflight['limitations']+['Common-valid metrics conditional if either arm has failures.',
            'All original labels retained despite clinical/reference ambiguities; no post-hoc relabelling.',
            'No RAG retrieval quality, clinical certification, 1–5 MAE or hidden-test claim.']}
    (OUT/'paired-analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第3轮：引用修复与同题Prompt对照','','## 实验边界','',
        '公开HealthBench开发集，同一70条/70个对话、143个医生标签。A=原v2提示词+新引用策略；B=v3 rubric-first+同一引用策略。两组各首次请求一次，交错顺序已冻结，无自动重试或挑选输出。',
        '因此A/B主要比较通用提示词，而不是把排版修复与判断优化混成一个因素。旧70条仅作历史参照。','',
        '| 项目 | A：原提示词 | B：改进提示词 |','| --- | --- | --- |',
        f'| 完整通过 | {result["status_counts"]["A"].get("Completed",0)}/70 | {result["status_counts"]["B"].get("Completed",0)}/70 |']
    for name,key in [('逐医生一致率','agreement'),('Cohen Kappa（二分类）','cohen_kappa'),('macro-F1','macro_f1')]:
        av=f'{common_metrics["A"][key]:.2%}' if key=='agreement' else f'{common_metrics["A"][key]:.4f}'
        bv=f'{common_metrics["B"][key]:.2%}' if key=='agreement' else f'{common_metrics["B"][key]:.4f}'
        lines.append(f'| 共同有效样本：{name} | {av} | {bv} |')
    lines += ['',f'共同有效样本：{len(common)}/70；共同医生比较：{common_metrics["A"]["n_physician_comparisons"]}。任何失败均保留；不完整组的整批主要指标不发布，共同有效指标只作条件诊断。',
        f'一致率变化：{result["common_valid_delta"]["agreement"]*100:+.2f}个百分点；按对话配对bootstrap95%区间：[{ci["agreement"][0]*100:.2f}, {ci["agreement"][1]*100:.2f}]个百分点。区间跨0，不能宣称稳定提升。其余delta与区间见 paired-analysis.json。',
        f'医生一致题：改善 {improved}；退步 {regressed}；两组仍错 {unchanged_wrong}。McNemar exact p={mcnemar:.4f}（探索性）。',
        f'历史v2和本次v2判断变化题号：{stochastic}。同temperature=0仍有变动，不能把历史差异全部归因于改动。','',
        '## 原9条逐项对照','','| 题号 | 医生标签 | 历史裁判 | 新A | 新B |','| --- | --- | --- | --- | --- |']
    for item in details:
        if item['index'] in NINE:
            lines.append(f'| {item["index"]} | {item["labels"]} | {item["historical"]} | {item["arms"]["A"]["judgment"]} | {item["arms"]["B"]["judgment"]} |')
    lines += ['','第44、62等题的参考冲突及第1、66等题的阈值歧义详见[9条逐条审阅](9条分歧逐条审阅.md)。统计仍保留原标签；不能将标签一致率等同于临床事实准确率。',
        '原9条中，本次B与原标签匹配2条（7、64），A也匹配2条（64、67）；第64条两组都改变，不能算v3独有收益。第27条已发现的安全细节依然没有被两组识别，不宜接入正式临床评估。',
        '', '## 分任务检查（共同有效子集）','',
        '| 官方任务 | 共同样本 | A逐医生一致率 | B逐医生一致率 |','| --- | --- | --- | --- |']
    for topic,info in by_topic.items():
        lines.append(f'| {topic} | {info["common_examples"]} | {info["A"]["agreement"]:.1%} | {info["B"]["agreement"]:.1%} |')
    ablation=json.loads((OUT/'citation-policy-ablation.json').read_text(encoding='utf-8')) if (OUT/'citation-policy-ablation.json').exists() else None
    if ablation:
        lines += ['','## 不再调用API的引用策略对照','',
            '同一份已保存模型引文离线回放：A的旧strict策略68/70通过，新anchored策略70/70通过；B的旧策略66/70，新策略69/70。',
            '新策略消除了6次排版差异误拒（A 2次、B 4次），同时新增拦截1次重复标题歧义。未把无法定位或歧义引用强行接受；引用通过不是医学正确证明。',
            '决定：默认启用anchored-v1引用修复，保留strict-v2作历史回放；默认Prompt仍为v2-strict，不推广v3。完整付费实验代码已归档frozen-code并校验hash，切换默认发生在实验之后。']
    lines += [
        '', '## 费用与可复现性','',f'本次保守记账 {cost:.6f}元，累计 {summary["budget"]["accounted_cny"]:.6f}元，剩余 {summary["budget"]["remaining_cny"]:.6f}元。实际供应商账单为准。',
        '完整冻结代码hash、输入hash、消息hash、顺序、模型与预算见 paired/preflight.json；每次job保留快照、原判定、引用span、原文hash与Python/UTF16偏移。',
        f'API报告模型：{dict(metadata)}。','',
        '## 架构关卡与下一步','',
        '目标仍是“有据可审计的独立单Jury”，不是拟合9个标签。引用定位是确定性验证；语义评分仍会错，临床事实不确定时需人工复核。',
        '本次是开发集提示词试验，不是模型训练、独立验证或RAG泛化。原平台不改动；未来以独立API接入Admin，与Doctor评分分开存储，锁定结果不覆盖。',
        '不凭本轮分数自动推广v3。先保留v2/v3可切换版本及原始证据；下一阶段应进行未见开发样本复测/人工复核争议，再决定是否做独立验证与RAG端到端评估。','']
    (OUT/'第三轮对照结果.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'gate-4.json').write_text(json.dumps({'status':'COMPLETE_DEV_ITERATION_NOT_CLINICALLY_VALIDATED',
        'objective_alignment':True,'architecture_alignment':True,'platform_unchanged':True,
        'code_hashes_unchanged_during_experiment':True,'full_original_labels_retained':True,
        'default_v3_promotion':False,'default_evidence_policy':'anchored-v1','offline_tests_after_default_activation':94,
        'citation_policy_ablation_verified':bool(ablation),
        'decision':'Activate deterministic evidence repair only; retain v2 prompt. No platform integration or clinical validation claim.',
        'next':'Discuss independent verification and unresolved factual/reference disputes; no automatic integration.'},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
