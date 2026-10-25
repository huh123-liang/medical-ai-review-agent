"""Read-only, independent arithmetic over a completed pilot; no provider calls."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path

from medical_jury.benchmark import examples


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / 'reports/pilot-first-pass-20261024-193334'
    summary_file = folder / 'summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8'))
    public = {r['example_id']: r for r in examples(root, 'dev', 10)}
    labels_file = root / 'data/private/physician-labels.jsonl'
    labels = {r['example_id']: r for r in map(json.loads, labels_file.read_text(encoding='utf-8').splitlines())}
    db = sqlite3.connect(f'file:{(root / "data/jury.sqlite").as_posix()}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    details, comparisons, matches, prompt_tokens, output_tokens, cost = [], 0, 0, 0, 0, 0.0
    confusion = Counter()
    for i, prediction in enumerate(summary['predictions'], 1):
        row = public[prediction['example_id']]
        reference = labels[prediction['example_id']]['binary_labels']
        job = dict(db.execute('SELECT status,error_code,created_at,updated_at,snapshot_json FROM jobs WHERE id=?',
                              (prediction['job_id'],)).fetchone())
        assert job['status'] == prediction['status']
        snapshot = json.loads(job.pop('snapshot_json'))
        assert snapshot['prompt_version'] == 'medical-jury-criterion-v1'
        assert snapshot['mode'] == 'oracle' if 'mode' in snapshot else snapshot['retrieval']['mode'] == 'oracle'
        evidence_valid = None
        reason = None
        if prediction['status'] == 'Completed':
            report = json.loads((root / 'reports' / prediction['job_id'] / 'report.json').read_text(encoding='utf-8'))
            judged = report['judgments'][0]
            assert judged['judgment'] == prediction['judgment']
            context = '\n'.join(m['content'] for m in row['case']['context'])
            evidence_valid = (all(q and q in row['case']['candidate_output'] for q in judged['answer_evidence']) and
                              all(q and q in context for q in judged['context_evidence']))
            assert evidence_valid
            reason = judged['reason']
            decision = prediction['judgment'] == 'met'
            for gold in reference:
                comparisons += 1
                matches += decision == gold
                confusion[('TP' if gold else 'FP') if decision else ('FN' if gold else 'TN')] += 1
        for fee in db.execute('SELECT accounted,state,usage_json FROM costs WHERE job_id=?', (prediction['job_id'],)):
            cost += fee['accounted']
            usage = json.loads(fee['usage_json']) if fee['usage_json'] else {}
            prompt_tokens += usage.get('prompt_tokens', 0)
            output_tokens += usage.get('completion_tokens', 0)
            assert fee['state'] == 'settled'
        details.append({'index':i, 'job_id':prediction['job_id'], 'example_id':prediction['example_id'],
                        'group_id':row['case']['group_id'], 'category':row['category'],
                        'rubric_text_sha256':hashlib.sha256(row['rubric']['text'].encode()).hexdigest(),
                        'status':prediction['status'], 'judgment':prediction['judgment'],
                        'physician_labels':reference, 'physician_disagreement':len(set(reference))>1,
                        'reason':reason, 'exact_evidence_valid':evidence_valid,
                        'error_code':job['error_code'],
                        'duration_seconds':round((datetime.fromisoformat(job['updated_at'])-
                                                 datetime.fromisoformat(job['created_at'])).total_seconds(),3)})
    db.close()
    completed = sum(r['status']=='Completed' for r in details)
    assert len(details) == 10
    assert abs(cost-summary['batch_accounted_cny']) < 0.000001
    assert cost <= 0.50
    assert completed/10 == summary['evaluation']['coverage']
    assert summary['evaluation']['metrics'] is None
    result = {'review_status':'Share with caveats', 'mode':'oracle_given_rubric_NOT_RAG_VALIDATION',
              'summary_sha256':hashlib.sha256(summary_file.read_bytes()).hexdigest(),
              'planned':10,'attempted':10,'completed':completed,'failed':10-completed,
              'distinct_conversation_groups':len({r['group_id'] for r in details}),
              'distinct_rubric_texts':len({r['rubric_text_sha256'] for r in details}),
              'category_counts':dict(Counter(r['category'] for r in details)),
              'physician_disputed_items':sum(r['physician_disagreement'] for r in details),
              'primary_metrics':None, 'primary_metrics_reason':'One failed prediction; no complete-batch agreement/Kappa/F1 published.',
              'successful_only_diagnostic':{'warning':'CONDITIONAL ON 9 SUCCESSFUL ITEMS; NOT THE TEN-ITEM PRIMARY SCORE.',
                    'physician_comparisons':comparisons,'matches':matches,
                    'conditional_agreement':matches/comparisons, 'confusion':dict(confusion)},
              'conservative_accounted_cny':round(cost,6), 'batch_cap_cny':0.50,
              'prompt_tokens':prompt_tokens,'completion_tokens':output_tokens,'items':details}
    (folder/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# DeepSeek Flash首轮真实测试解读', '', '日期：2026-10-24。原平台未修改。本解读脚本不调用模型；首次请求总计10次，无重试。', '',
        '## 结论', '',
        f'- 原定10条全部尝试，{completed}条完成、{10-completed}条失败；完成率{completed/10:.0%}。',
        f'- 输入{prompt_tokens:,} tokens、输出{output_tokens:,} tokens，保守记账{cost:.6f}元，小于0.50元。失败输出的用量也计入；不是供应商账单金额。',
        '- 保留失败及最初中止报告；只续跑未尝试的第9、10条，没有替换题目、重试直到成功或更换模型。',
        '- 完整批次的agreement、Kappa、macro-F1按预先协议不发布。不能将完成率当准确率。', '',
        '## 逐项结果', '', '| 序号 | Agent判定 | 两位公开医生标签 | 运行结果 |', '| --- | --- | --- | --- |']
    for row in details:
        doctor = ' / '.join('满足' if v else '不满足' for v in row['physician_labels'])
        decision = {'met':'满足','not_met':'不满足',None:'无有效判定'}[row['judgment']]
        lines.append(f'| {row["index"]} | {decision} | {doctor} | {row["status"]} |')
    lines += ['', '## 如何理解评分表现', '',
        f'仅在成功的9条中，和每条的两位医生逐一比较，共{matches}/{comparisons}次一致（{matches/comparisons:.1%}）。这是成功子集的诊断数，不是整批正式成绩，也不是医学准确率。',
        '成功项中6条医生意见一致，Agent均与其一致；另3条医生各持不同意见，因此不宜直接把分歧都称为Agent错误。',
        '第8条医生也存在分歧，但模型格式失败，不能猜测其判断。', '',
        '## 发现的问题', '',
        '1. 输出可靠性：第8条为INVALID_JUDGMENT_JSON；服务器返回HTTP 200、finish_reason=stop及usage，但结果未通过JSON/严格schema校验。不能说是网络中断或因预算不足。原始失败输出没有留存，无法进一步判定是JSON语法、字段名、类型还是多余字段；这是诊断日志的缺口。',
        '2. 采样代表性：按文件顺序选了10条，实际只有5个对话组、2种不同准则文本，6条急诊转诊紧急场景、4条条件性紧急场景。不是10名独立患者，更不能代表糖尿病、高血压等慢性病评审能力。',
        '3. 人类分歧：10条中4条的两位医生标签不同；后续必须区分医生一致组和分歧组，不能把多数票/平票强行当无争议真值。',
        '4. 检索未验证：这次oracle实验直接提供准则，用来隔离裁判能力，不是自动选rubric的RAG能力测试。',
        '5. 引用与临床真实性不同：9条成功输出的引用均能精确定位，但不等于模型的医学解释已被独立核实。',
        '6. 凭据配置：发现Key被误放在启动脚本的目录切换行，已迁移至.env.local并恢复脚本；建议用户在供应商后台轮换Key，不在聊天或Git中传播。', '',
        '## 下一步优先级（尚未执行）', '',
        '1. 输出校验：增加可脱敏、限长的schema错误诊断；固定结构示例/字段约束；若增加一次修复调用，应新建版本、保留首轮失败率，并纳入预算。',
        '2. 验证选样：固定种子，按准则/类别、正负标签及对话分层；确认不将医生标签提供给Agent，不在测试集调Prompt。',
        '3. 使用分层的50–100条开发集做第二轮，先以预调用费用上界规划预算，再单独确认；不要由这10条推断长期性能。',
        '4. 稳定裁判基线后，再测准则检索、多Agent和平台集成。', '',
        '详细机器结果：analysis.json；原始合并结果：summary.json；每条完整报告在reports/JURY-*。',
        '可在所审范围内带限制分享（Share with caveats），不支持“已达到医生水平”或“可替代医生”的结论。', '']
    (folder/'测试结果解读.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
