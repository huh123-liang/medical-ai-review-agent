"""Freeze and verify candidate prompts and NONMEDICAL offline tests; never call a model."""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if ROOT != Path(r'E:\AI agnet') or HERE.name != 'prompt-candidate-v1':
    raise RuntimeError('Run this isolated experiment only at its delivered project path.')
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_iteration12_shadow import ledger, tests as existing_tests, shadow
spec = importlib.util.spec_from_file_location('iteration13_frozen_design', ROOT / 'docs/review-methods-v1/verify_design.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
from candidate import SCHEMA, VERSION, build_messages, canonical, check_candidate, digest
from fixtures import audit_examples, make_pair, reviewed_examples


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(out, name, value):
    p = (out / name).resolve()
    if not p.is_relative_to(out.resolve()):
        raise ValueError('Output outside new report.')
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def protected():
    paths = previous.protected()
    paths = {k: v for k, v in paths.items() if not (ROOT / k).is_relative_to(HERE)}
    for folder in [ROOT / 'docs/review-methods-v1', ROOT / 'reports/iteration-13-methods-and-acceptance']:
        paths.update({str(p.relative_to(ROOT)): sha(p) for p in sorted(folder.rglob('*'))
                      if p.is_file() and '__pycache__' not in p.parts})
    return paths


def delivery_hashes():
    return {p.name: sha(p) for p in sorted(HERE.iterdir()) if p.is_file()}


def new_tests(out, store):
    base = store / 'candidate-pytest-data'
    if base.exists():
        raise FileExistsError('Never reuse a pytest data directory.')
    args = ['-c', str(HERE / 'pytest.ini'), str(HERE / 'test_candidate.py'), '-p', 'no:cacheprovider',
            '--tb=short', '--basetemp=' + str(base), '--junitxml=' + str(out / 'candidate-tests.xml')]
    child = '''import os,socket,pytest
from unittest.mock import patch
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
def blocked(*a,**k):raise AssertionError('No network in candidate tests.')
with patch.object(socket.socket,'connect',blocked),patch.object(socket.socket,'connect_ex',blocked),patch.object(socket,'create_connection',blocked),patch.object(socket,'getaddrinfo',blocked):
    for host in ('127.0.0.1','198.51.100.1'):
        s=socket.socket()
        try:
            try:s.connect((host,443))
            except AssertionError:pass
            else:raise AssertionError('Connection escaped offline guard.')
        finally:s.close()
    try:socket.getaddrinfo('example.invalid',443)
    except AssertionError:pass
    else:raise AssertionError('DNS escaped guard.')
    print('CANDIDATE_NETWORK_GUARD_VERIFIED')
    raise SystemExit(pytest.main(ARGS))
'''.replace('ARGS', repr(args))
    env = os.environ.copy()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1')
    run = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', child], cwd=HERE,
                         env=env, capture_output=True, text=True, encoding='utf-8', timeout=90)
    (out / 'candidate-test-console.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
    if run.returncode:
        raise RuntimeError('Candidate tests failed. Preserve report and repair in a versioned follow-up.')
    assert 'CANDIDATE_NETWORK_GUARD_VERIFIED' in run.stdout
    root = ET.parse(out / 'candidate-tests.xml').getroot()
    counts = {k: sum(int(s.get(k, '0')) for s in root.findall('testsuite'))
              for k in ['tests', 'failures', 'errors', 'skipped']}
    assert counts['tests'] > 0 and counts['failures'] == counts['errors'] == counts['skipped'] == 0
    functions = {n.name for n in ast.parse((HERE / 'test_candidate.py').read_text(encoding='utf-8')).body
                 if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
    coverage = read(HERE / 'coverage-map.json')
    assert len(coverage['rows']) == 30 and [r['id'] for r in coverage['rows']] == [f'A{i:02d}' for i in range(1, 31)]
    assert {name for r in coverage['rows'] for name in r['tests']} <= functions
    passed = {r.get('name', '').split('[', 1)[0] for r in root.findall('.//testcase')
              if r.find('failure') is None and r.find('error') is None and r.find('skipped') is None}
    assert {name for r in coverage['rows'] for name in r['tests']} <= passed
    write(out, 'coverage-map-results.json', {'meaning': 'Partial program checks only; all limitations preserved.',
          'all_thirty_semantic_cases_passed': False, 'source_map_sha256': sha(HERE / 'coverage-map.json'),
          'rows': coverage['rows']})
    return counts


def prepare_materials(out):
    cards = reviewed_examples()
    audit = audit_examples()
    i, _ = make_pair()
    bundles = [build_messages(i, arm=arm, target_group='synthetic-target-dev',
                             examples=cards if arm == 'B2_three' else [], enabled=True)
               for arm in ['B1_zero', 'B2_three']]
    assert bundles[0]['input_sha256'] == bundles[1]['input_sha256']
    assert bundles[0]['messages'][0] == bundles[1]['messages'][0]
    pa, pb = [json.loads(b['messages'][1]['content']) for b in bundles]
    assert pa.pop('demonstrations') == [] and len(pb.pop('demonstrations')) == 3 and pa == pb
    write(out, 'candidate-schema.json', SCHEMA)
    write(out, 'reviewed-nonmedical-examples.json', cards)
    write(out, 'nonmedical-example-audit.json', audit)
    write(out, 'frozen-nonmedical-target.json', i)
    for b in bundles:
        write(out, b['arm'] + '.messages.json', b)
        text = '\n\n'.join('## ' + m['role'].upper() + '\n\n' + m['content'] for m in b['messages'])
        (out / (b['arm'] + '.prompt.md')).write_text('# NONDEFAULT CANDIDATE — NOT EXECUTED\n\n' + text, encoding='utf-8')
    checks = []
    for kind in ['pass', 'fail', 'unknown']:
        inputs, authored = make_pair(kind)
        raw = canonical(authored)
        result = check_candidate(raw, inputs)
        assert result['protocol_status'] == 'valid' and not result['clinical_release'] and result['score'] is None
        checks.append({'origin': 'hand_authored_nonmedical_fixture_not_model_prediction',
                       'fixture': kind, 'input': inputs, 'supplied_output': authored, 'inspection': result})
    write(out, 'hand-authored-output-inspections.json', checks)
    return {'two_arms_same_target_and_system': True, 'sole_message_difference': 'demonstrations',
            'examples': 3, 'nonmedical_examples_only': True, 'physician_reviewed_examples': 0,
            'model_calls': 0, 'new_model_outputs': 0, 'hand_authored_outputs_checked': 3,
            'prompt_bytes': {b['arm']: b['byte_length'] for b in bundles},
            'token_counts': None, 'token_limit_validated': False,
            'medical_performance_comparison': 'not_measured', 'protocol_version': VERSION}


def verify(out):
    f = read(out / 'scope-freeze.json')
    assert delivery_hashes() == f['delivery_files']
    assert protected() == f['protected_before']
    assert ledger() == f['ledger_before']
    for name, expected in read(out / 'delivery-manifest.json')['files'].items():
        assert sha(out / name) == expected
    return read(out / 'verification.json')


def main(out, verify_only):
    if verify_only:
        result = verify(out)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if out.exists():
        raise FileExistsError('Preserve existing reports. Use --verify or new --output-name.')
    store = shadow.STORE_BASE / out.name
    if store.exists():
        raise FileExistsError('Test namespace exists; never delete or reuse old data.')
    prior = read(ROOT / 'reports/iteration-13-methods-and-acceptance/verification.json')
    assert ledger() == prior['financial_state'], 'Financial state changed; inspect before proceeding.'
    # Compare actual frozen historical files, not just newly captured current bytes.
    old_freeze = read(ROOT / 'reports/iteration-13-methods-and-acceptance/scope-freeze.json')
    assert all(sha(ROOT / p) == h for p, h in old_freeze['protected_before'].items())
    freeze = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'protected_before': protected(),
              'delivery_files': delivery_hashes(), 'ledger_before': ledger(),
              'paid_calls_authorized': False, 'runtime_changes_authorized': False}
    out.mkdir()
    write(out, 'scope-freeze.json', freeze)
    try:
        materials = prepare_materials(out)
        write(out, 'prompt-comparison-design.json', materials)
        old = existing_tests(out, store)
        new = new_tests(out, store)
        unresolved = read(ROOT / 'reports/iteration-12-shadow-adapter-r3/seven-unresolved-status.json')
        assert len(unresolved) == 7 and all(r['new_clinical_judgment'] is None and
                 r['status'] == 'unresolved_no_independent_review' for r in unresolved)
        assert protected() == freeze['protected_before'] and ledger() == freeze['ledger_before']
        assert delivery_hashes() == freeze['delivery_files']
        result = {'status': 'CANDIDATE_PROMPTS_OFFLINE_CHECKED_NOT_MODEL_OR_MEDICAL_VALIDATION',
                  'existing_offline_regression': old, 'new_candidate_engineering_tests': new,
                  'total_tests': old['summary']['tests'] + new['tests'], 'prompt_materials': materials,
                  'protected_file_count': len(freeze['protected_before']), 'protected_files_unchanged': True,
                  'new_paid_calls': 0, 'new_model_predictions': 0, 'unresolved_cases': 7,
                  'new_clinical_gold_labels': 0, 'new_clinical_agreement_metrics': None,
                  'runtime_core_changed': False, 'default_prompt_changed': False,
                  'old_platform_accessed_by_runner': False, 'legacy_api_or_schema_changed': False,
                  'new_protocol_wired_to_legacy': False, 'financial_state': ledger(),
                  'gate': 'STOP_AND_REQUEST_USER_APPROVAL'}
        write(out, 'verification.json', result)
        reflection = f'''# 第14轮：完成内容、反思与下一步审批

## 实际完成

已准备B1无示例、B2三示例候选Prompt；system、schema与冻结目标相同，唯一消息差异为demonstrations。3例均为手工非医疗文本，经显式字符串/协议断言核对，不是医生审核的临床例子。输出样例也是手工构造，不是DeepSeek生成。

已有离线回归{old['summary']['tests']}项通过，新增消息组装/隔离/候选结构测试{new['tests']}项通过，合计{result['total_tests']}项；不作为医学正确率或两版Prompt效果。

{len(freeze['protected_before'])}个受保护文件前后不变；费用账本不变，新增付费调用0，累计{result['financial_state']['accounted_cny']:.6f}元。7条案例继续未决，未读其医生标签作预期、未产生新医学标签。

## 反思：目标和架构

1. **方向符合。** 仍围绕单一Jury的任务边界、证据与可追溯输出，不增加Agent、自动工具循环、平台写入或知识库规模。
2. **发现的真实冲突已隔离。** 旧候选只支持二元原判，新规范需要unable_to_determine；本轮使用独立三态研究协议，不强迫未知变失败，不修改旧API/协议，也不进行静默转换。代价是暂时不能直接接旧影子入口，需要之后版本化适配审批。
3. **工程验证符合，但不代表模型遵守。** 阻网、白名单、精确坐标和声明层AND可由程序检查；真正任务识别、引用语义支持、临床适用性、完整断言和注入抵抗仍未验证。A01–A30只部分覆盖，详见coverage-map-results.json，未将旧计划改为全部通过。
4. **示例边界符合。** 只用确定性非医疗例，三例固定顺序、dev分组及精确内容hash隔离；没有可声称临床有效的记忆库或ICL提升。语义近重复还需后续治理，不能靠字符串hash解决全部泄漏。
5. **资源风险仍待处理。** Prompt字节数：{materials['prompt_bytes']}；这些不是tokens。包含schema/全量快照/三例的消息可能较长，未验证V4 Pro输出token充分性、结构成功率、延迟或真实成本。
6. **平台边界符合。** 原平台未访问；默认Prompt、核心和旧影子模块未接线。未来应保持医生评分、Agent结论与复核记录分离，经版本接口受控接入。本轮不是集成验证。

## 下一步建议，待用户批准

先完成一份小额真实生成的预注册方案，不立即调用：明确任务范围和分组、来源/示例审核、三态协议与旧接口映射策略、目标模型上下文/输出tokens、每次预约与总轮费用、结构失败/截断/预算停止条件、完整分母及独立评测依据。

建议先评B1是否可稳定输出三态和精确定位，再决定是否评B2；二者始终同模型/同输入/同准则/同来源。首次真实生成只检验格式、证据定位和成本，没有独立医生/公共标签时不声称医学改进。调用与任何运行接线均另批。

本轮停止，等待用户批准下一部分。
'''
        (out / '目标架构反思与下一步.md').write_text(reflection, encoding='utf-8')
        write(out, 'delivery-manifest.json', {'files': {p.relative_to(out).as_posix(): sha(p)
              for p in sorted(out.rglob('*')) if p.is_file()}, 'manifest_excludes_itself': True})
        verify(out)
        print(json.dumps({'status': result['status'], 'old_tests': old, 'new_tests': new,
                          'total': result['total_tests'], 'prompt_bytes': materials['prompt_bytes'],
                          'protected_files': result['protected_file_count'], 'new_paid_calls': 0,
                          'accounted_cny': result['financial_state']['accounted_cny'], 'output': str(out)},
                         ensure_ascii=False, indent=2))
    except Exception as exc:
        write(out, 'failure-preserved.json', {'status': 'FAILED_NOT_DISCARDED',
              'error_type': type(exc).__name__, 'paid_calls_authorized': False,
              'next': 'Inspect saved console/JUnit; repair separately and use a new report namespace.'})
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--output-name', default='iteration-14-candidate-prompts')
    args = p.parse_args()
    if not re.fullmatch(r'iteration-14-candidate-prompts(?:-r[1-9][0-9]*)?', args.output_name):
        p.error('Use a bounded new iteration-14 report name.')

    def blocked(*a, **k):
        raise AssertionError('No network in candidate verification.')

    with patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked), \
         patch.object(socket, 'create_connection', blocked), patch.object(socket, 'getaddrinfo', blocked):
        main(ROOT / 'reports' / args.output_name, args.verify)
