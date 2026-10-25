"""Offline contract experiment; never modifies runtime or grades unresolved cases.

Default creates a fresh report directory and refuses overwrites. --verify reads a
completed report only. No provider import, network, secrets display, or DB writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

from prepare_iteration10_review_materials import ledger, protected_hashes

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/item-evidence-v1'
PRIOR = ROOT / 'reports/iteration-10-review-pack'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(EXP))
contract = load('isolated_item_contract', EXP / 'contract.py')


def protected():
    files = protected_hashes()
    for folder in [PRIOR, ROOT / 'reports/iteration-9-source-audit']:
        files.update({str(p.relative_to(ROOT)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()})
    for name in ['REVIEW-SCORING-SPEC-V1.md', 'REVIEW-EVIDENCE-TEMPLATE-V1.md']:
        p = ROOT / 'docs' / name
        files[str(p.relative_to(ROOT))] = sha(p)
    return files


def code_hashes():
    paths = [p for p in EXP.iterdir() if p.is_file()]
    paths += [Path(__file__), ROOT / 'docs/ITEM-EVIDENCE-DESIGN-V1.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def write_text(out, name, text):
    p = out / name
    assert p.resolve().is_relative_to(out.resolve())
    p.parent.mkdir(exist_ok=True, parents=True)
    p.write_text(text, encoding='utf-8')


def write(out, name, data):
    write_text(out, name, json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def old_case_inventory(out):
    """Read seven already exported case packets. Neither labels nor research advice are targets."""
    mappings = read(PRIOR / 'coordinator-only/case-map.json')
    originals = {c['audit_id']: c for c in read(ROOT / 'reports/iteration-8-offline/cases.json')}
    case_rows, legacy_rows = [], []
    for mapping in mappings:
        rid = mapping['review_id']
        p = PRIOR / f'reviewer-stage-1/cases/{rid}.json'
        exported = read(p)
        original = originals[mapping['audit_id']]
        assert exported['context'] == original['case']['context']
        assert exported['candidate_output'] == original['case']['candidate_output']
        assert exported['rubric_text'] == original['rubric']['text']
        assert set(exported) == {'review_id', 'context', 'candidate_output', 'rubric_text'}
        case_rows.append({'review_id': rid, 'status': 'unresolved_no_independent_review',
                          'input_file_sha256': sha(p), 'original_text_unchanged': True,
                          'new_item_evidence': None, 'approved_obligation_manifest': False,
                          'physician_labels_used_as_test_targets': False,
                          'research_advice_used_as_gold': False, 'new_prediction': False})
        for arm, saved in original['arms'].items():
            info = saved['raw_output']
            if info['kind'] == 'exact_original_output':
                raw = info['raw']
                assert hashlib.sha256(raw.encode()).hexdigest() == info['original_output_sha256']
                old = contract.strict_loads(raw)
                material_kind = 'exact_historical_raw_json'
                payload_hash = hashlib.sha256(raw.encode()).hexdigest()
            elif saved['report'] is not None:
                # Explicitly a normalized stored-report projection, not invented original wire.
                old = saved['report']['judgments'][0]
                material_kind = 'saved_report_projection_not_original_wire'
                payload_hash = contract.digest(old)
            else:
                legacy_rows.append({'review_id': rid, 'arm': arm, 'material_kind': 'unavailable',
                                    'inspection': 'skipped_no_complete_material', 'new_item_evidence': None})
                continue
            frozen = contract.digest(old)
            view = contract.inspect_legacy(old)
            assert contract.digest(old) == frozen
            assert view['original_verdict'] == old['judgment'] and view['original_reason'] == old['reason']
            assert view['item_evidence'] is None and not view['grading_performed']
            legacy_rows.append({'review_id': rid, 'arm': arm, 'material_kind': material_kind,
                                'payload_sha256': payload_hash, 'inspection': view['compatibility'],
                                'original_verdict_and_reason_preserved': True, 'new_item_evidence': None,
                                'new_coverage': 'unknown', 'new_prediction': False,
                                'historical_status_changed': False})
    assert len(case_rows) == 7 and len(legacy_rows) == 14
    assert all(c['status'] == 'unresolved_no_independent_review' for c in case_rows)
    write(out, 'seven-unresolved-status.json', case_rows)
    write(out, 'legacy-compatibility-inventory.json', legacy_rows)
    return {'unresolved_cases_preserved': len(case_rows), 'legacy_materials': len(legacy_rows),
            'legacy_material_kinds': dict(Counter(r['material_kind'] for r in legacy_rows)),
            'legacy_new_fields_fabricated': 0, 'new_clinical_labels': 0}


def run_tests(out):
    args = ['-c', str(EXP / 'pytest.ini'), str(EXP / 'test_contract.py'), '-p', 'no:cacheprovider',
            '--tb=short', '--junitxml=' + str(out / 'offline-contract-tests.xml')]
    child = """import os,socket,pytest
from unittest.mock import patch
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
def blocked(*args,**kwargs):raise AssertionError('No network during contract experiment.')
with patch.object(socket.socket,'connect',blocked),patch.object(socket.socket,'connect_ex',blocked),patch.object(socket,'create_connection',blocked),patch.object(socket,'getaddrinfo',blocked):
    raise SystemExit(pytest.main(ARGS))
""".replace('ARGS', repr(args))
    env = os.environ.copy()
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    execution = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', child], cwd=ROOT,
                               env=env, text=True, encoding='utf-8', capture_output=True, timeout=120)
    write_text(out, 'offline-test-console.txt', execution.stdout + execution.stderr)
    if execution.returncode:
        write(out, 'failure.json', {'stage': 'offline_tests', 'returncode': execution.returncode,
                                   'paid_calls': 0, 'old_results_overwritten': False})
        raise RuntimeError('Offline contract tests failed; preserve this directory and use a new report name after repair.')
    suites = ET.parse(out / 'offline-contract-tests.xml').getroot().findall('testsuite')
    counts = {key: sum(int(s.get(key, '0')) for s in suites) for key in ['tests', 'failures', 'errors', 'skipped']}
    assert counts['tests'] > 0 and counts['failures'] == counts['errors'] == counts['skipped'] == 0
    assert len(ET.parse(out / 'offline-contract-tests.xml').getroot().findall('.//testcase')) == counts['tests']
    return counts


def verify(out):
    freeze = read(out / 'scope-freeze.json')
    manifest = read(out / 'delivery-manifest.json')
    assert protected() == freeze['protected_before'], 'Existing protected files changed.'
    assert ledger() == freeze['ledger_before'], 'Financial state changed.'
    assert code_hashes() == freeze['experiment_code'], 'Experiment changed after its tests.'
    assert all(sha(out / name) == expected for name, expected in manifest['files'].items())
    results = read(out / 'verification.json')
    assert all(r['status'] == 'unresolved_no_independent_review' for r in read(out / 'seven-unresolved-status.json'))
    return results


def main(out, verify_only=False):
    if verify_only:
        print(json.dumps(verify(out), ensure_ascii=False, indent=2))
        return
    assert not out.exists(), 'Do not overwrite past outputs. Use --verify or a new --output-name.'
    before = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'ledger_before': ledger(),
              'protected_before': protected(), 'experiment_code': code_hashes(),
              'paid_calls_authorized': False, 'runtime_integration_authorized': False,
              'old_platform_changes_authorized': False,
              'dependencies': {n: version(n) for n in ['jsonschema', 'pytest', 'pydantic']}}
    assert before['ledger_before'] == read(PRIOR / 'coordinator-only/verification.json')['financial_state']
    out.mkdir()
    write(out, 'scope-freeze.json', before)
    counts = run_tests(out)
    inventory = old_case_inventory(out)
    fixture_module = load('synthetic_contract_fixtures', EXP / 'test_contract.py')
    inputs, output = fixture_module.fixture_pair()
    write(out, 'candidate-output.schema.json', contract.SCHEMA)
    write(out, 'synthetic-only-example.json', {'kind': 'NON_MEDICAL_SYNTHETIC_EXAMPLE_NOT_MODEL_OUTPUT',
                                             'input': inputs, 'output': output,
                                             'inspection': contract.inspect(json.dumps(output, ensure_ascii=False), inputs)})
    assert protected() == before['protected_before']
    assert ledger() == before['ledger_before']
    assert code_hashes() == before['experiment_code']
    results = {'status': 'OFFLINE_CONTRACT_CHECKED_NOT_MEDICAL_GRADING_VALIDATION',
               'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'offline_tests': counts,
               **inventory, 'protected_files_unchanged': True, 'protected_file_count': len(before['protected_before']),
               'runtime_core_modified': False, 'runtime_or_default_integration': False,
               'paid_calls': 0, 'new_model_predictions': 0, 'financial_state': ledger(),
               'original_platform_accessed_by_runner': False, 'new_clinical_agreement_metrics': None,
               'clinical_release': False, 'gate': 'STOP_FOR_USER_APPROVAL',
               'limits': ['Synthetic contract tests are not medical accuracy or entailment tests.',
                          'Seven seen diagnostic cases remain unresolved; no gold labels added.',
                          'Old raw/projection inventory is not a new-format model replay.',
                          'No new task recognizer, authoritative rubric decomposition or medical fact verifier was built.']}
    write(out, 'verification.json', results)
    write(out, 'development-log.json', {
        'preliminary_run': '70 passed, 2 setup errors with automatically generated long parameter IDs; no medical verdicts involved.',
        'repair': 'Use explicit short IDs for strict-JSON parameterized tests; no expected verdicts or assertions were relaxed.',
        'completed_run': counts, 'test_targets': 'Protocol definitions and synthetic text, never research advice or physician labels.'})
    summary = f"""# 逐项证据设计与离线测试结果

## 完成内容

- 候选契约已形成，详见 `E:\\AI agnet\\docs\\ITEM-EVIDENCE-DESIGN-V1.md`。Schema与非医疗示例在本目录；实验代码独立，不被运行核心导入。
- 严格JSON、冻结绑定、原文定位、逐项义务、来源配对、声明逻辑及独立复核状态，共 **{counts['tests']} 项合成测试通过**；失败/错误/跳过均0。网络在执行期间封锁。
- 七例继续 unresolved；原文、阶段1/2材料、来源、原标签与旧核心均未修改。
- 盘点14份旧材料：6份完整原始JSON、8份保存报告投影；未生成逐项字段，未追溯改失败，不是新模型回放或新医学判分。
- 无新付费调用，账本仍 {results['financial_state']['accounted_cny']:.6f} 元。累计预算上限30元。

## 反思：是否符合目标与架构

**符合工程方向，尚未证明医学判断改善。** 仍是单Agent判断、程序核验和审计；不加Agent、不增工具权限、不修改原平台。模型原判与证据状态、结构完整性和复核状态独立，避免把缺证据当失败、把合法引用当医学正确。

关键局限：本轮没有新的任务识别器、权威Rubric拆解或医学事实验证器；这些仍需独立验证。结构覆盖只针对已提供义务ID，不能发现草案未列出的真实义务；模型自报的主张清单和来源关系也不能自证完整或正确。候选永远NeedsReview，不是经过校准的临床自动放行系统。

7例已见且因分歧被选，不能当新的独立测试集；没有医生就不制造专家共识。未来平台仍须单独保存人工分、Agent研究判定和审计/复核信息。本轮只做接口设计，没有改旧平台或业务数据库。

## 下一步（待审批，本轮不执行）

1. 审阅契约的复杂度、长度上限和草案义务清单的确认方式。对7例继续保留未决；没有专家时只推进不依赖临床金标准的结构检查。
2. 若批准，做**非默认、可开关的影子适配层**：保持旧wire/API不变，候选逐项记录单独保存，先用Mock与历史兼容性测试，不参与正式计分，不新增付费调用、不接原平台。
3. 影子层通过后，再申请小额真实模型生成实验，单独列出样本、token/预算上限、失败计费和停止条件。不得宣称离线71项通过等于医生一致率改善；付费测试需再次授权。

现在停止，等待用户审批。
"""
    write_text(out, '结果与架构反思.md', summary)
    paths = {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    write(out, 'delivery-manifest.json', {'files': paths, 'manifest_excludes_itself': True})
    verify(out)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output-name', default='iteration-11-item-evidence')
    args = parser.parse_args()
    if not re.fullmatch(r'iteration-11-item-evidence(?:-r[1-9][0-9]*)?', args.output_name):
        parser.error('Output name must be a versioned iteration-11 directory under reports.')
    out = ROOT / 'reports' / args.output_name
    def no_network(*args, **kwargs):
        raise AssertionError('No network during offline contract experiment.')
    with patch.object(socket.socket, 'connect', no_network), patch.object(socket.socket, 'connect_ex', no_network), \
         patch.object(socket, 'create_connection', no_network), patch.object(socket, 'getaddrinfo', no_network):
        main(out, args.verify)
