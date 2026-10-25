"""Offline scorer, allowed to read frozen physician labels ONLY after prediction seal exists."""
from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter

from medical_jury.benchmark import binary_metrics as project_metrics
from medical_jury.store import timestamp

from common import (ARM_A, ARM_B, PRIVATE, REPORT, ROOT, canonical, check_freeze, digest, protected,
                    read, sha, topic, write)
from metrics import binary_metrics, score


def fmt(x):
    return '未测/不适用' if x is None else f'{x:.4f}'


def main():
    if (REPORT / 'analysis.json').exists():
        raise FileExistsError('Analysis already frozen; do not overwrite post-unblinding results.')
    freeze, manifest = check_freeze()
    assert sha(REPORT / 'protocol.json') == freeze['protocol_sha256']
    seal = read(REPORT / 'prediction-seal.json')
    assert sha(REPORT / 'predictions-frozen.json') == seal['predictions_sha256']
    state = read(REPORT / 'predictions-frozen.json')
    assert state['state'] == 'Finished' and state['labels_deserialized_by_predictor'] is False
    # First scorer label read happens here, AFTER every prediction/failure/not-run record is sealed.
    label_read_at = timestamp()
    references = read(PRIVATE / 'references.json')['labels']
    rows = read(REPORT / 'inputs.json')
    predictions = {(r['example_id'], r['arm']): r for r in state['predictions']}
    items = []
    with sqlite3.connect(f'file:{(ROOT / "data/jury.sqlite").as_posix()}?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        old = read(REPORT / 'old-ledger-state.json')
        jobs = {r['id']: dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')}
        costs = {r['id']: dict(r) for r in db.execute('SELECT * FROM costs ORDER BY id')}
        assert digest([jobs[k] for k in old['job_ids']]) == old['jobs_sha256']
        assert digest([costs[k] for k in old['cost_ids']]) == old['costs_sha256']
        new_ids = {r['job_id'] for r in state['predictions'] if r.get('job_id')}
        assert set(jobs) - set(old['job_ids']) == new_ids
        fees = [v for k, v in costs.items() if k not in set(old['cost_ids'])]
        assert all(r['job_id'] in new_ids and r['state'] not in {'reserved', 'reservation_exceeded'} for r in fees)
        actual = sum(r['accounted'] for r in fees)
        assert abs(actual - state['round_accounted_cny']) < 1e-6
        for index, row in enumerate(rows, 1):
            ref = references[row['example_id']]
            assert ref['group_id'] == row['case']['group_id']
            assert len(ref['binary_labels']) == len(ref['anonymized_physician_ids'])
            item = {'index': index, 'example_id': row['example_id'], 'group_id': row['case']['group_id'],
                    'topic': topic(row), 'category': row['category'], 'labels': ref['binary_labels'],
                    'anonymized_physician_ids': ref['anonymized_physician_ids'], 'arms': {}}
            for arm in [ARM_A, ARM_B]:
                p = predictions[(row['example_id'], arm)]
                assert p['input_sha256'] == digest(row)
                if p.get('job_id'):
                    job = jobs[p['job_id']]
                    assert job['status'] == p['status']
                    if p['status'] == 'Completed':
                        report = json.loads(job['report_json'])
                        decision = report['judgments'][0]['judgment'] if arm == ARM_A else report['criterion_decision']
                        assert decision == p['decision']
                item['arms'][arm] = p
            items.append(item)
    result = {'source': 'HealthBench original public physician annotations', 'split': 'validation',
              'reference_label_read_at': label_read_at, 'prediction_sealed_at': seal['sealed_at'],
              'unit': 'One criterion-specific decision per conversation, compared to each physician rating',
              'n_cases': len(items), 'n_groups': len({i['group_id'] for i in items}),
              'n_original_physician_comparisons': sum(len(i['labels']) for i in items),
              'unanimous_cases': sum(len(set(i['labels'])) == 1 for i in items),
              'disputed_cases': sum(len(set(i['labels'])) > 1 for i in items),
              'arms': {arm: score(items, arm) for arm in [ARM_A, ARM_B]},
              'cost': {'round_accounted_cny': state['round_accounted_cny'], 'cumulative': state['budget']},
              'halt_reason': state.get('halt_reason'), 'candidate_stop_rule': state.get('candidate_stop_rule'),
              'limitations': read(REPORT / 'protocol.json')['limits']}
    total = sum(len(i['labels']) for i in items)
    result['reference_descriptives'] = {
        'physician_positive_labels': sum(sum(i['labels']) for i in items),
        'always_met': binary_metrics([(True, label) for i in items for label in i['labels']]),
        'always_not_met': binary_metrics([(False, label) for i in items for label in i['labels']]),
        'max_possible_binary_pairwise_agreement_given_disputed_labels':
             sum(max(sum(i['labels']), len(i['labels']) - sum(i['labels'])) for i in items) / total}
    # Independently recompute every substantive metric using the pre-existing implementation.
    for arm in [ARM_A, ARM_B]:
        pairs = [(i['arms'][arm]['decision'] == 'met', label) for i in items
                 if i['arms'][arm]['status'] == 'Completed' and i['arms'][arm]['decision'] in {'met', 'not_met'}
                 for label in i['labels']]
        old_metrics = project_metrics(pairs)
        new_metrics = result['arms'][arm]['conditional_decided_metrics']
        for key in ['agreement', 'macro_f1', 'cohen_kappa']:
            assert (old_metrics[key] is None and new_metrics[key] is None) or abs(old_metrics[key] - new_metrics[key]) < 1e-12
    result['metrics_independently_recomputed'] = True
    result['by_topic'] = {t: {a: score([i for i in items if i['topic'] == t], a, bootstrap=0)
                             for a in [ARM_A, ARM_B]} for t in dict.fromkeys(i['topic'] for i in items)}
    write(REPORT / 'analysis.json', result, exclusive=True)
    write(REPORT / 'per-case-comparison.json', items, exclusive=True)
    with (REPORT / 'per-case-comparison.csv').open('x', encoding='utf-8-sig', newline='') as f:
        fields = ['index', 'example_id', 'topic', 'physician_labels', 'A_status', 'A_decision', 'B_status', 'B_decision']
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in items:
            w.writerow({'index': i['index'], 'example_id': i['example_id'], 'topic': i['topic'],
                        'physician_labels': canonical(i['labels']), 'A_status': i['arms'][ARM_A]['status'],
                        'A_decision': i['arms'][ARM_A]['decision'], 'B_status': i['arms'][ARM_B]['status'],
                        'B_decision': i['arms'][ARM_B]['decision']})
    lines = ['# 原医生标签与两臂逐条对照', '', '医生标签是外部参考意见，不宣称绝对医学真值。7条原争议例未纳入。', '',
             '| # | 主题 | 原医生逐人标签 | A 当前基线 | B 三态候选 |', '|---|---|---|---|---|']
    for i in items:
        vector = ', '.join('met' if g else 'not_met' for g in i['labels'])
        cells = [f"{i['arms'][a]['status']} / {i['arms'][a]['decision'] or i['arms'][a].get('error_code') or '未运行'}" for a in [ARM_A, ARM_B]]
        lines.append(f"| {i['index']} | {i['topic']} | {vector} | {cells[0]} | {cells[1]} |")
    lines += ['', '## 分歧、未决与失败的可追溯说明', '',
              '以下理由来自模型原输出，不是研究者新裁定。没有新增医学来源核查，不能认定所有分歧都是模型错或医生错。原文完整保存在inputs.json；逐条结构化结果在per-case-comparison.json。', '']
    for i, row in zip(items, rows):
        if all(i['arms'][a]['status'] == 'Completed' and i['arms'][a]['decision'] in {'met', 'not_met'}
               and all((i['arms'][a]['decision'] == 'met') == g for g in i['labels']) for a in [ARM_A, ARM_B]):
            continue
        lines += [f"### {i['index']} · {i['example_id']}", '', '**原准则：** ' + row['rubric']['text'], '',
                  '**原回答开头（截取仅供查找，不是全部判分依据）：** ' + row['case']['candidate_output'][:350], '']
        for a in [ARM_A, ARM_B]:
            p = i['arms'][a]
            lines += [f"- {a}: {p['status']} / {p['decision']}；{p.get('error_code') or ''}",
                      '  原理由：' + (p.get('reason') or '无有效输出；保持失败/未运行。')]
        lines += ['', '研究状态：仅记录与原标签的关系，未完成独立临床分歧裁定。', '']
    (REPORT / '逐条对照与分歧.md').write_text('\n'.join(lines), encoding='utf-8')
    old_unresolved = read(ROOT / 'reports/iteration-12-shadow-adapter-r3/seven-unresolved-status.json')
    assert all(r['new_clinical_judgment'] is None for r in old_unresolved) and len(old_unresolved) == 7
    check_freeze()
    verification = {'status': 'PUBLIC_PHYSICIAN_SUBSET_REAL_MODEL_EVALUATED_WITH_LIMITS',
        'predictions_sealed_before_scoring': seal['sealed_at'] <= label_read_at,
        'physician_labels_passed_to_model': False, 'labels_original_and_unchanged': True,
        'old_jobs_and_costs_unchanged': True, 'new_costs_reconciled_to_experiment_jobs': True,
        'new_paid_attempts': len(fees), 'new_jobs': len(new_ids), 'round_cost_cny': round(actual, 6),
        'cumulative_cost_cny': state['budget']['accounted_cny'],
        'protected_file_count': len(freeze['protected_files']), 'protected_files_unchanged': protected() == freeze['protected_files'],
        'original_platform_accessed': False, 'runtime_core_or_default_prompt_changed': False,
        'rubrics_or_answers_or_labels_changed': False, 'seven_prior_cases_still_unresolved': True,
        'new_gold_labels_invented': 0, 'metrics_recomputed_independently': True,
        'medical_entailment_verified': False, 'clinical_release': False, 'gate': 'STOP_AND_REQUEST_USER_APPROVAL'}
    write(REPORT / 'verification.json', verification, exclusive=True)
    a, b = result['arms'][ARM_A], result['arms'][ARM_B]
    report = f'''# 第15轮：公开医生标签盲评结果与架构反思

## 真实完成的实验

固定HealthBench官方meta-evaluation的{len(items)}个项目未使用validation对话组，每主题2组，一组一条原准则。回答、准则和原始逐医生标签全部冻结；预测器未反序列化医生标签，预测封存后独立评分器才读取标签。不是新增患者、不是完整HealthBench复现、不是RAG检索测试。

原医生比较数{total}，全体一致样本{result['unanimous_cases']}，医生有分歧样本{result['disputed_cases']}。保留每名医生的原判断，不压成一个人为“正确答案”。

| 项目 | A 当前scoped-review-v1 | B 新B1三态研究候选 |
|---|---|---|
| 完成且二元可比较 | {a['decided_cases']}/{len(items)} | {b['decided_cases']}/{len(items)} |
| 未定 | {a['undetermined_cases']} | {b['undetermined_cases']} |
| 失败或未运行 | {a['failed_or_not_run_cases']} | {b['failed_or_not_run_cases']} |
| 整批主指标 | {'可报告该小样本指标' if a['primary_metrics'] else '覆盖不完整，保留为空'} | {'可报告该小样本指标' if b['primary_metrics'] else '覆盖不完整，保留为空'} |
| 已决子集逐医生一致率 | {fmt(a['conditional_decided_metrics']['agreement'])} | {fmt(b['conditional_decided_metrics']['agreement'])} |
| 已决子集Macro-F1 | {fmt(a['conditional_decided_metrics']['macro_f1'])} | {fmt(b['conditional_decided_metrics']['macro_f1'])} |
| 已决子集Cohen Kappa | {fmt(a['conditional_decided_metrics']['cohen_kappa'])} | {fmt(b['conditional_decided_metrics']['cohen_kappa'])} |

若覆盖不完整，上表已决指标只作条件性诊断，不能说是整批准确率。具体分母、混淆矩阵、组重抽样区间、逐主题与always-met基线见analysis.json。两臂协议和token上限不同，不能把分差直接归因于某句Prompt改善。

## 主要风险与解释边界

新候选仍缺已确认的原子义务清单与完整医学事实来源；我们只把原准则整体绑定为O1研究草案，不伪造医生认可的拆解。因此unable_to_determine可能反映证据/适用性能力不足，不是通过，也不是协议失败。精确定位/结构通过不能证明语义医学核查正确。

即使逐医生一致率较高，也只是这14组给定准则上的标签一致性；不能推广到所有疾病、真实医院数据或临床安全。若医生彼此分歧，任意二元裁判也不可能同时匹配每个人。公开数据预训练污染与语义近重复仍是限制。未根据结果重写Prompt、改标签或再次挑题重跑。

## 费用与核验

新增付费尝试{len(fees)}次，本轮保守记账{actual:.6f}元，累计{state['budget']['accounted_cny']:.6f}元；低于30元。旧任务和费用逐行摘要未变，所有新费用归属于本轮；受保护文件{len(freeze['protected_files'])}个未变。原平台未访问，原7例仍未决。

## 目标与架构核对

1. 目标符合：从纯工程检查进入有外部医生参考标签的真实预测检验，且不预先把标签当提示。
2. 架构符合：单Jury、冻结输入/准则、确定性校验和独立评分。新候选只是实验适配器，未接旧API或平台。
3. 证据边界符合但能力未齐：Oracle评分没有证明RAG检索，候选研究草案没有变成权威临床规范；复杂输出失败/未定均保留。
4. 不能凭本轮宣称替代医生或总体临床准确率。小样本、标签分歧、协议差异及缺事实来源都影响结论。

## 下一步建议（等用户审批）

先逐条复核本轮失败、未定和分歧的来源：区分坐标/协议、任务边界、准则解释和真实事实支持缺口；不把模型理由当事实证明。候选是否应使用完整准则直判、何时拆解义务、哪些来源可用，需要独立设计与开发集验证。

本轮validation一旦用于改方法，就不再是下一轮未见验证材料。保持当前原标签与预测，不重跑刷分；下一候选用新的开发组，冻结后再用新的独立验证组。是否扩充MORQA多维评价、真实来源库或接平台，另行审批。停止在本轮。
'''
    (REPORT / '结果与架构反思.md').write_text(report, encoding='utf-8')
    write(REPORT / 'analysis-seal.json', {'sealed_at': timestamp(), 'analysis_sha256': sha(REPORT / 'analysis.json'),
          'prediction_seal_sha256': sha(REPORT / 'prediction-seal.json'), 'reference_sha256': manifest['sealed_reference_sha256']}, exclusive=True)
    print(json.dumps({'status': verification['status'], 'arms': result['arms'], 'reference': result['reference_descriptives'],
          'new_cost': actual, 'cumulative': state['budget']['accounted_cny'], 'verification': verification},
          ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
