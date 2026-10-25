"""Offline-only shadow verification. New report/store namespaces; no primary writes."""
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
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from prepare_iteration10_review_materials import ledger
from verify_iteration11_item_evidence import protected as protected_previous

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'experiments/item-evidence-shadow-v1'
PREVIOUS = ROOT / 'reports/iteration-11-item-evidence'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shadow = load('offline_sidecar_verifier', EXP / 'shadow.py')


def protected():
    result = protected_previous()
    for p in sorted(PREVIOUS.rglob('*')):
        if p.is_file():
            result[str(p.relative_to(ROOT))] = sha(p)
    result.update(read(PREVIOUS / 'scope-freeze.json')['experiment_code'])
    # Compare current bytes, not just carry a prior expected digest.
    return {name: sha(ROOT / name) for name in result}


def current_code():
    paths = [p for p in EXP.iterdir() if p.is_file()]
    paths += [Path(__file__), ROOT / 'docs/SHADOW-ADAPTER-V1.md']
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def write(out, name, value):
    p = out / name
    assert p.resolve().is_relative_to(out.resolve())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def tests(out, store_root):
    base = store_root / 'pytest-data'
    assert not base.exists(), 'Test basetemp must be new, never a user or prior data directory.'
    store_root.mkdir(parents=True, exist_ok=True)
    args = ['-c', str(EXP / 'pytest.ini'), str(ROOT / 'tests'),
            str(ROOT / 'experiments/item-evidence-v1/test_contract.py'), str(EXP / 'test_shadow.py'),
            '-p', 'no:cacheprovider', '--tb=short', '--basetemp=' + str(base),
            '--junitxml=' + str(out / 'offline-tests.xml')]
    child = """import os,socket,threading,pytest
from unittest.mock import patch
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
def blocked(*args,**kwargs):raise AssertionError('No network in shadow verification.')
real_connect,real_pair=socket.socket.connect,socket.socketpair
local=threading.local()
def guarded_connect(sock,address):
    if getattr(local,'inside_pair',False) and isinstance(address,tuple) and address[0] in ('127.0.0.1','::1'):
        return real_connect(sock,address)
    return blocked()
def guarded_pair(*args,**kwargs):
    previous=getattr(local,'inside_pair',False)
    local.inside_pair=True
    try:return real_pair(*args,**kwargs)
    finally:local.inside_pair=previous
with patch.object(socket.socket,'connect',guarded_connect),patch.object(socket.socket,'connect_ex',blocked),patch.object(socket,'create_connection',blocked),patch.object(socket,'getaddrinfo',blocked),patch.object(socket,'socketpair',guarded_pair):
    a,b=socket.socketpair()
    try:
        a.send(b'x')
        assert b.recv(1)==b'x'
    finally:a.close();b.close()
    for host in ('127.0.0.1','198.51.100.1'):
        s=socket.socket()
        try:
            try:s.connect((host,443))
            except AssertionError:pass
            else:raise AssertionError('Manual connection escaped guard.')
        finally:s.close()
    try:socket.getaddrinfo('example.invalid',443)
    except AssertionError:pass
    else:raise AssertionError('DNS escaped guard.')
    print('OFFLINE_GUARD_VERIFIED_INTERNAL_SOCKETPAIR_ONLY')
    raise SystemExit(pytest.main(ARGS))
""".replace('ARGS', repr(args))
    env = os.environ.copy()
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    run = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', child], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding='utf-8', timeout=150)
    (out / 'offline-test-console.txt').write_text(run.stdout + run.stderr, encoding='utf-8')
    if run.returncode:
        write(out, 'failure.json', {'stage': 'offline_tests', 'returncode': run.returncode,
                                   'ledger_after': ledger(), 'old_files_unchanged':
                                   protected() == read(out / 'scope-freeze.json')['protected_before']})
        raise RuntimeError('Tests failed; preserve output and use a new versioned report after repair.')
    assert 'OFFLINE_GUARD_VERIFIED_INTERNAL_SOCKETPAIR_ONLY' in run.stdout
    root = ET.parse(out / 'offline-tests.xml').getroot()
    counts = {k: sum(int(s.get(k, '0')) for s in root.findall('testsuite'))
              for k in ['tests', 'failures', 'errors', 'skipped']}
    breakdown = Counter()
    contract_functions = {node.name for node in ast.parse(
        (ROOT / 'experiments/item-evidence-v1/test_contract.py').read_text(encoding='utf-8')).body
        if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
    for node in root.findall('.//testcase'):
        name = node.get('classname', '').split('.')[-1]
        function = node.get('name', '').split('[', 1)[0]
        is_contract = name in {'', 'test_contract'} and function in contract_functions
        category = 'item_contract' if is_contract else 'shadow_adapter' if name == 'test_shadow' else 'existing_core'
        breakdown[category] += 1
    assert counts['tests'] == sum(breakdown.values()) and counts['tests'] > 0
    assert counts['failures'] == counts['errors'] == counts['skipped'] == 0
    assert breakdown['item_contract'] == 71 and breakdown['shadow_adapter'] >= 36
    return {'summary': counts, 'breakdown': dict(breakdown)}


def replay_supplied_material(out, store_root):
    store = shadow.ShadowStore(store_root / 'demo-records')
    receipts = []
    disabled = shadow.ShadowAdapter(store=store)
    toy = {'judgment': 'not_met', 'reason': 'Synthetic primary kept unchanged.',
           'answer_evidence': [], 'context_evidence': [], 'review_required': False}
    assert disabled.passthrough(toy) is toy and not store.root.exists()
    fixtures = load('shadow_toy_fixtures', ROOT / 'experiments/item-evidence-v1/test_contract.py')
    inputs, candidate = fixtures.fixture_pair()
    adapter = shadow.ShadowAdapter(enabled=True, store=store)
    for name in ['valid', 'missing-source', 'invalid-json']:
        supplied = deepcopy(candidate)
        if name == 'missing-source':
            supplied['checks'] = []
        raw = '{not-json' if name == 'invalid-json' else json.dumps(supplied, ensure_ascii=False)
        frozen = shadow.canonical(toy)
        assert adapter.passthrough(toy, request_id='toy-' + name, snapshot=inputs, candidate_json=raw,
                                   origin='synthetic_fixture', case_reference='SYNTHETIC-colour') is toy
        assert shadow.canonical(toy) == frozen
        assert adapter.last_receipt['status'] == 'recorded'
        receipts.append({'example': 'nonmedical-' + name, 'receipt': adapter.last_receipt})
    mapping = read(ROOT / 'reports/iteration-10-review-pack/coordinator-only/case-map.json')
    cases = {c['audit_id']: c for c in read(ROOT / 'reports/iteration-8-offline/cases.json')}
    unresolved = []
    for row in mapping:
        rid, case = row['review_id'], cases[row['audit_id']]
        blind = read(ROOT / f'reports/iteration-10-review-pack/reviewer-stage-1/cases/{rid}.json')
        assert blind['context'] == case['case']['context'] and blind['candidate_output'] == case['case']['candidate_output']
        assert blind['rubric_text'] == case['rubric']['text']
        for arm, saved in case['arms'].items():
            raw_info = saved['raw_output']
            if raw_info['kind'] == 'exact_original_output':
                raw = raw_info['raw']
                assert shadow.text_sha(raw) == raw_info['original_output_sha256']
                primary = json.loads(raw)
                origin = 'historical_original'
            else:
                assert saved['report'] is not None
                primary = saved['report']['judgments'][0]
                origin = 'historical_projection'
            before = shadow.canonical(primary)
            assert adapter.passthrough(primary, request_id=f'legacy-{rid}-{arm}', origin=origin, case_reference=rid) is primary
            assert shadow.canonical(primary) == before
            assert adapter.last_receipt['status'] == 'recorded'
            receipts.append({'case_reference': rid, 'arm': arm, 'origin': origin, 'receipt': adapter.last_receipt})
        unresolved.append({'review_id': rid, 'status': 'unresolved_no_independent_review',
                           'new_clinical_judgment': None, 'new_item_fields_fabricated': False})
    records = store.records()
    assert len(records) == 17
    assert all(not r['clinical_release'] and not r['benchmark_export_eligible'] and r['official_score'] is None for r in records)
    legacy = [r for r in records if r['origin'] != 'synthetic_fixture']
    assert len(legacy) == 14 and all(r['state'] == 'legacy_requires_new_instrumentation' for r in legacy)
    assert all(r['inspection'] is None and r['candidate_payload'] is None for r in legacy)
    assert all(r['case_status'] == 'unresolved_no_independent_review' for r in legacy)
    # Repeat one exact legacy observation; no duplicate row or overwrite.
    rid = mapping[-1]['review_id']
    receipt = adapter.observe(primary, request_id=f'legacy-{rid}-{arm}', origin=origin, case_reference=rid)
    assert receipt['status'] == 'already_recorded' and len(store.records()) == 17
    write(out, 'sidecar-receipts.json', receipts)
    write(out, 'seven-unresolved-status.json', unresolved)
    summary = {'demo_record_count': len(records), 'nonmedical_supplied_examples': 3,
               'legacy_materials': 14, 'legacy_origins': dict(Counter(r['origin'] for r in legacy)),
               'legacy_missing_item_output_not_fabricated': True, 'seven_cases_still_unresolved': True,
               'state_counts': dict(Counter(r['state'] for r in records)), 'idempotency_rechecked': True,
               'store_path': str(store.root / 'shadow.sqlite'), 'store_sha256': sha(store.root / 'shadow.sqlite'),
               'new_model_outputs_generated': 0, 'clinical_releases': 0}
    write(out, 'sidecar-store-summary.json', summary)
    return summary


def verify(out):
    freeze, manifest = read(out / 'scope-freeze.json'), read(out / 'delivery-manifest.json')
    assert protected() == freeze['protected_before']
    assert current_code() == freeze['new_code']
    assert ledger() == freeze['ledger_before']
    assert all(sha(out / name) == expected for name, expected in manifest['files'].items())
    store = read(out / 'sidecar-store-summary.json')
    assert sha(Path(store['store_path'])) == store['store_sha256']
    return read(out / 'verification.json')


def main(out, verify_only):
    if verify_only:
        print(json.dumps(verify(out), ensure_ascii=False, indent=2))
        return
    assert not out.exists(), 'Existing report preserved; use --verify or a new --output-name.'
    # Check the prior completed experiment has not already drifted before our baseline.
    prior_freeze = read(PREVIOUS / 'scope-freeze.json')
    assert ledger() == prior_freeze['ledger_before']
    assert all(sha(ROOT / p) == h for p, h in prior_freeze['experiment_code'].items())
    frozen = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'ledger_before': ledger(),
              'protected_before': protected(), 'new_code': current_code(),
              'paid_calls_authorized': False, 'original_platform_changes_authorized': False,
              'runtime_hook_enabled': False, 'default_switch': False}
    out.mkdir()
    write(out, 'scope-freeze.json', frozen)
    store_root = shadow.STORE_BASE / out.name
    assert not store_root.exists(), 'Use a fresh store namespace; never overwrite existing records.'
    test_results = tests(out, store_root)
    sidecar = replay_supplied_material(out, store_root)
    assert protected() == frozen['protected_before'] and current_code() == frozen['new_code']
    assert ledger() == frozen['ledger_before']
    result = {'status': 'OFFLINE_SHADOW_ADAPTER_VERIFIED_NOT_MEDICAL_VALIDATION',
              'tests': test_results, 'sidecar': sidecar,
              'default_enabled': False, 'runtime_hook_added': False, 'existing_api_changed': False,
              'existing_primary_files_unchanged': True, 'protected_file_count': len(frozen['protected_before']),
              'old_platform_accessed_by_runner': False, 'new_paid_calls': 0,
              'financial_state': ledger(), 'unresolved_cases': 7, 'new_clinical_gold_labels': 0,
              'new_clinical_agreement_metrics': None, 'gate': 'STOP_AND_REQUEST_USER_APPROVAL'}
    write(out, 'verification.json', result)
    write(out, 'development-notes.json', {
        'preflight': ['An initial test-launch command had an extra closing parenthesis; corrected without running tests.',
                      'A first pytest setup lacked its dedicated parent directory; created only the new shadow namespace.',
                      'Then 36 shadow tests passed.',
                      'First combined run: 350 passed, 2 API tests blocked by Windows event-loop internal socketpair.',
                      'Failure report preserved in iteration-12-shadow-adapter. Only harness guard repaired: permits internal socketpair, blocks manual loopback, outbound and DNS.',
                      'r2 had 352 passing tests, but report grouping failed because pytest emitted a blank classname for the out-of-root contract tests. Grouping now checks exact function names against the unchanged test AST; tests and assertions unchanged.',
                      'Final combined run is the recorded result.'],
        'clinical_targets_added': False, 'old_labels_or_research_suggestions_used_as_expected_answers': False})
    counts = test_results['breakdown']
    report = f"""# 非默认影子适配层：交付、验证与反思

## 已完成

- 影子模块默认关闭；关闭时无读写，显式开启后只检查调用方提供的离线输出。没有模型调用入口或线上接线。
- 旧主结果按原对象返回，不新增字段、不改原判/分数/API；影子检查、存储和调用失败只给独立诊断。
- 逐项候选与检查记录保存到专用本地SQLite，主结果仅保存哈希；不会写正式jury.sqlite。追加历史、重复请求、冲突、格式失败、密钥形态正文抑制和目录边界均有离线测试。
- 代码和开关说明见 `E:\\AI agnet\\docs\\SHADOW-ADAPTER-V1.md`。默认启动与原平台未改。

## 验证结果

- 原核心回归 {counts['existing_core']} 项，上一轮逐项契约 {counts['item_contract']} 项，影子层 {counts['shadow_adapter']} 项；合计 **{test_results['summary']['tests']} 项通过**，失败/错误/跳过均0。仅允许Windows事件循环内部socketpair；已验证普通本机连接、外连与DNS仍被阻断。
- 3份非医疗合成输出仅验证结构；14份历史材料（6份原始JSON、8份保存报告投影）保持原结果，缺少逐项字段不补造。
- 7条案例继续 unresolved；没有新增医学判定、医生共识或一致率/Kappa。
- 影子演示库共17条记录，重复读取/重复提交未新增或覆盖旧行；所有记录均不能临床放行或进入正式计分。
- {len(frozen['protected_before'])}个既有受保护文件前后相同，正式任务数、费用行与费用指纹不变。本轮付费调用0，累计 {result['financial_state']['accounted_cny']:.6f} 元（上限30元）。

## 完成后的目标与架构核对

1. **目标：符合。** 解决“候选证据如何不干扰原系统地记录与核对”，没有用7例追逐分数或把审计意见当答案。
2. **架构：符合。** 单Agent主线未改；旁路是确定性校验和存储模块，不是第二Agent，不增加工具权限。
3. **证据边界：保留。** 引用、哈希、义务覆盖只能证明结构与来源追溯；主张完整性、任务识别及医学支持关系仍未得到独立验证。
4. **回归与隔离：本轮检查通过。** 旧核心、API定义、配置、材料、标签与账本未改；错误不会改主结果。没有进行真实模型/线上负载或生产故障验证。
5. **是否能进入临床或直接接平台：不能。** 模块仅离线可用，当前未挂到生产Workflow；没有专家评审，也没有新模型真实生成质量或成本数据。

## 下一步建议（等待用户审批，本轮不实施）

1. 先冻结一个**小额真实生成测试方案**：明确非默认候选Prompt、任务/义务草案、来源范围、样本按对话组隔离、输出token上限、单次/整轮费用及停止条件。7例仍只作诊断，不作金标准。
2. 经单独批准后才做少量真实DeepSeek生成，验证能否稳定产出新格式及真实token/延迟。没有医生时，只能报告结构成功率与证据缺口，不能宣称医学准确率提升。
3. 依据上述结果再决定是否接到独立Agent的受控影子钩子；正式计分、原平台和多Agent集成均需另行审批。

本轮停止，等待审批。
"""
    (out / '结果与架构反思.md').write_text(report, encoding='utf-8')
    write(out, 'delivery-manifest.json', {'files': {p.relative_to(out).as_posix(): sha(p)
                                                  for p in sorted(out.rglob('*')) if p.is_file()},
                                         'manifest_excludes_itself': True})
    verify(out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--output-name', default='iteration-12-shadow-adapter-r3')
    args = parser.parse_args()
    if not re.fullmatch(r'iteration-12-shadow-adapter(?:-r[1-9][0-9]*)?', args.output_name):
        parser.error('Use a new iteration-12 report name under this project.')
    def blocked(*args, **kwargs):
        raise AssertionError('No network during offline shadow verification.')
    with patch.object(socket.socket, 'connect', blocked), patch.object(socket.socket, 'connect_ex', blocked), \
         patch.object(socket, 'create_connection', blocked), patch.object(socket, 'getaddrinfo', blocked):
        main(ROOT / 'reports' / args.output_name, args.verify)
