"""Verify documentation and rerun EXISTING offline tests; no model generation.

Creates only a new report namespace and isolated pytest data via the unchanged
iteration-12 harness. This is not a medical/semantic acceptance implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
if ROOT != Path(r'E:\AI agnet') or PACKAGE.name != 'review-methods-v1':
    raise RuntimeError('Run only the delivered package under E:/AI agnet/docs/review-methods-v1.')
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_iteration12_shadow import ledger, protected as old_protected, tests, shadow


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(out, name, value):
    path = (out / name).resolve()
    if not path.is_relative_to(out.resolve()):
        raise ValueError('Output outside new report namespace.')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def protected():
    result = old_protected()
    for folder in ['src', 'tests', 'docs', 'scripts', 'experiments', 'examples']:
        for p in sorted((ROOT / folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts and not p.is_relative_to(PACKAGE):
                result[str(p.relative_to(ROOT))] = sha(p)
    for folder in ['iteration-10-review-pack', 'iteration-12-shadow-adapter-r3']:
        for p in sorted((ROOT / 'reports' / folder).rglob('*')):
            if p.is_file():
                result[str(p.relative_to(ROOT))] = sha(p)
    for p in ROOT.iterdir():
        if p.is_file():
            result[str(p.relative_to(ROOT))] = sha(p)
    return result


def package_hashes():
    return {p.relative_to(PACKAGE).as_posix(): sha(p) for p in sorted(PACKAGE.iterdir()) if p.is_file()}


def document_checks():
    passed = []

    def check(name, condition):
        if not condition:
            raise AssertionError('Document check failed: ' + name)
        passed.append(name)

    expected = {'README.md', 'paper-gap-matrix.md', 'scoring-spec-v1.1.md',
                'item-evidence-template-v1.1.md', 'blank-review-record.json',
                'offline-acceptance-plan.md', 'acceptance-cases.json', 'verify_design.py'}
    check('exact_delivery_members', {p.name for p in PACKAGE.iterdir() if p.is_file()} == expected)
    texts = {p.name: p.read_text(encoding='utf-8') for p in PACKAGE.glob('*.md')}
    for name, text in texts.items():
        check('readable_utf8_' + name, bool(text.strip()) and '\ufffd' not in text)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if target.startswith('https://'):
                continue
            resolved = (PACKAGE / target.split('#', 1)[0]).resolve()
            check('local_link_' + name + '_' + target, resolved.is_relative_to(ROOT) and resolved.is_file())

    matrix = texts['paper-gap-matrix.md']
    for n in range(1, 8):
        check(f'paper_P{n:02d}_documented', f'| P{n:02d} ' in matrix)
    for n in range(1, 10):
        check(f'gap_G{n:02d}_documented', f'| G{n:02d} ' in matrix)
    plan = read(PACKAGE / 'acceptance-cases.json')
    check('plan_not_reported_as_executed', plan['status'] == 'planned_not_executed_by_document_check')
    check('no_gold_labels_in_plan', plan['clinical_gold_labels_added'] == 0)
    rows = plan['cases']
    check('thirty_unique_planned_cases', [r['id'] for r in rows] == [f'A{i:02d}' for i in range(1, 31)])
    check('all_cases_still_planned', all(r['execution_status'] == 'planned' for r in rows))
    check('every_gap_has_planned_check', {r['gap'] for r in rows} == {f'G{i:02d}' for i in range(1, 10)})
    check('every_paper_has_trace', {p for r in rows for p in r['papers']} == {f'P{i:02d}' for i in range(1, 8)})
    check('every_case_has_scope_and_oracle', all(r['scenario'] and r['expected'] and r['oracle'] for r in rows))
    check('case_ids_trace_to_matrix', all(r['id'] in matrix for r in rows))
    template = read(PACKAGE / 'blank-review-record.json')
    check('blank_record_not_prediction', template['record_status'] == 'blank_template')
    check('no_prefilled_clinical_decisions', all(template['decisions'][k] is None for k in
          ['original_model_verdict', 'criterion_decision', 'official_score']))
    check('no_prefilled_evidence', not any(template[k] for k in
          ['claims', 'source_catalog', 'claim_source_checks', 'obligation_findings']))
    check('no_fabricated_bindings', all(v is None for v in template['input_bindings'].values()))
    check('no_clinical_release_or_export', template['decisions']['clinical_release'] is False
          and template['decisions']['benchmark_export_eligible'] is False)
    check('no_independent_review_claim', template['provenance']['independent_review_received'] is False)
    check('no_historical_verdict_or_label_changes', template['audit']['original_label_overwritten'] is False
          and template['audit']['historical_verdict_changed'] is False)
    cases = read(ROOT / 'reports/iteration-12-shadow-adapter-r3/seven-unresolved-status.json')
    check('seven_distinct_unresolved_cases', len(cases) == len({r['review_id'] for r in cases}) == 7)
    check('seven_no_new_judgment_or_fields', all(r['status'] == 'unresolved_no_independent_review'
          and r['new_clinical_judgment'] is None and r['new_item_fields_fabricated'] is False for r in cases))
    return {'status': 'DOCUMENT_STRUCTURE_CHECKED_NOT_SEMANTIC_VALIDATION',
            'checks_passed': len(passed), 'checks': passed, 'planned_acceptance_cases': len(rows),
            'planned_acceptance_cases_executed': 0, 'new_model_predictions': 0,
            'new_independent_clinical_reviews': 0}


def verify(out):
    freeze = read(out / 'scope-freeze.json')
    assert package_hashes() == freeze['delivery_files']
    assert protected() == freeze['protected_before']
    assert ledger() == freeze['ledger_before']
    manifest = read(out / 'delivery-manifest.json')
    assert all(sha(out / name) == h for name, h in manifest['files'].items())
    return read(out / 'verification.json')


def main(out, verify_only):
    if verify_only:
        print(json.dumps(verify(out), ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError('Report exists; use --verify or a new versioned --output-name.')
    store = shadow.STORE_BASE / out.name
    if store.exists():
        raise FileExistsError('Test namespace exists; never reuse or delete prior data.')
    prior = read(ROOT / 'reports/iteration-12-shadow-adapter-r3/verification.json')
    before = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'ledger_before': ledger(),
              'protected_before': protected(), 'delivery_files': package_hashes(),
              'paid_calls_authorized': False, 'original_platform_changes_authorized': False}
    if before['ledger_before'] != prior['financial_state']:
        raise RuntimeError('Financial state drifted since prior verification; inspect before running.')
    out.mkdir()
    write(out, 'scope-freeze.json', before)
    try:
        docs = document_checks()
        write(out, 'document-checks.json', docs)
        regression = tests(out, store)
        assert protected() == before['protected_before']
        assert ledger() == before['ledger_before']
        assert package_hashes() == before['delivery_files']
        result = {
            'status': 'METHODS_AND_ACCEPTANCE_DESIGN_VERIFIED_NOT_MEDICAL_VALIDATION',
            'document_checks': docs, 'existing_offline_regression': regression,
            'protected_file_count': len(before['protected_before']), 'protected_files_unchanged': True,
            'new_paid_calls': 0, 'new_model_predictions': 0, 'new_clinical_gold_labels': 0,
            'new_clinical_agreement_metrics': None, 'unresolved_cases': 7,
            'runtime_core_changed': False, 'default_prompt_changed': False,
            'new_candidate_prompt_created': False, 'existing_api_changed': False,
            'runtime_shadow_hook_added': False, 'old_platform_accessed_by_runner': False,
            'financial_state': ledger(), 'gate': 'STOP_AND_REQUEST_USER_APPROVAL'
        }
        write(out, 'verification.json', result)
        counts = regression['summary']
        reflection = f'''# 第13轮：目标与架构核对、下一步审批

## 本轮实际完成

建立7篇论文与9类缺口对应表；增补判分规范v1.1和空白逐项证据模板；设计30个后续验收场景及Oracle/检索、无示例/少量示例的分离对照。没有编写新的运行Prompt或模型调用代码。

文档结构检查 {docs['checks_passed']} 项通过；这只是目录、引用、编号、空白模板和计划状态检查。30个新场景仍是planned，执行数0，不冒充新增语义测试成绩。

重跑已有离线测试：{counts['tests']}项，失败{counts['failures']}、错误{counts['errors']}、跳过{counts['skipped']}；分组见verification.json及JUnit。这不是医学准确率或新Prompt改善证据。

核验{len(before['protected_before'])}个受保护文件前后不变；正式任务、费用行及账本指纹不变。新增付费调用0，累计{result['financial_state']['accounted_cny']:.6f}元。7例继续未决，没有新医生标签、一致率或Kappa。

## 反思：是否在正确道路上

1. **目标符合，但只完成设计层。** 围绕单一医疗审查Agent的任务、准则和证据，不扩大为更多Agent或追求简历数值。尚未证明判得更准。
2. **架构符合。** 保持冻结输入→单Jury→确定性校验/计分→审计；新规范在旁路，旧AND契约、API及默认关闭的影子模块未改。
3. **避免过严与误放行的方向符合。** 明确证据不足不等于错误、未知不能自动通过、负向met不等于好、OR不能伪装AND。规则尚未转成模型行为保证。
4. **证据独立性保留。** 区分定位、支持、适用性和覆盖。哈希/协议通过不代表医学事实正确；7例没有医生时继续未决。
5. **与原平台兼容方向符合。** 本轮未访问MVP2；未来人工评分、Jury判断、研究复核独立保存，经版本化接口接入，不覆盖锁定评分。未进行真实集成测试。
6. **本轮验收通过的范围有限。** 证明文档自洽、既有回归保持、文件与账本未变；不能证明已建成审核示例记忆、通用RAG、语义核验器、DSPy/TextGrad或本地模型部署。

## 下一部分建议：需用户审批，本轮不执行

只准备非默认候选Prompt和小型开发材料：冻结输出契约，写无示例B1及少量示例B2的提示模板；示例仅取来源可核查的开发材料或明确非医疗结构样例，标注审核范围。不得使用7例或验证/隐藏标签作示范答案。

再把A01–A17等结构/任务边界中可确定的部分做离线最小对照，明确哪些只有待审预期、哪些能由程序验证；继续不付费、不改核心或平台。既有真实输出只用于兼容与缺项盘点，不能假装新Prompt回放。

若需要证明新Prompt有效，之后另行申请小额真实V4 Pro生成方案，先约定样本分组、独立标签、token/费用与停止条件。没有独立依据时只报告结构与支持缺口，不报医学准确率提升。

停止在本部分，等待用户批准下一部分。
'''
        (out / '目标架构反思与下一步.md').write_text(reflection, encoding='utf-8')
        write(out, 'delivery-manifest.json', {'files': {p.relative_to(out).as_posix(): sha(p)
              for p in sorted(out.rglob('*')) if p.is_file()}, 'manifest_excludes_itself': True})
        verify(out)
        print(json.dumps({'status': result['status'], 'document_checks': docs['checks_passed'],
                          'planned_new_cases_executed': 0, 'existing_tests': regression,
                          'protected_files': result['protected_file_count'], 'new_paid_calls': 0,
                          'accounted_cny': result['financial_state']['accounted_cny'],
                          'output': str(out)}, ensure_ascii=False, indent=2))
    except Exception as exc:
        write(out, 'design-verification-failure.json', {'status': 'FAILED_PRESERVED',
              'error_type': type(exc).__name__, 'paid_calls_authorized': False,
              'next_step': 'Inspect preserved evidence; use a new report namespace after repair.'})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output-name', default='iteration-13-methods-and-acceptance')
    args = parser.parse_args()
    if not re.fullmatch(r'iteration-13-methods-and-acceptance(?:-r[1-9][0-9]*)?', args.output_name):
        parser.error('Use a new bounded iteration-13 report name.')

    def blocked(*args, **kwargs):
        raise AssertionError('No network in this design verification.')

    with patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked), \
         patch.object(socket, 'create_connection', blocked), patch.object(socket, 'getaddrinfo', blocked):
        main(ROOT / 'reports' / args.output_name, args.verify)
