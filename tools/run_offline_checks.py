"""Run isolated source/contract suites with all external sockets blocked.

Creates a new ignored reports namespace and never uses model credentials.
The shadow tests require temporary storage beneath their allowlisted data root.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ("core", "tests", 245),
    ("item_contract", "experiments/item-evidence-v1/test_contract.py", 71),
    ("shadow", "experiments/item-evidence-shadow-v1/test_shadow.py", 36),
    ("prompt_candidate", "experiments/prompt-candidate-v1/test_candidate.py", 64),
    ("blind_protocol", "experiments/physician-blind-v1-r2/test_protocol.py", 21),
]

CHILD = r'''
import os, socket, sys, threading
from unittest.mock import patch
sys.path.insert(0, SOURCE_ROOT)
import medical_jury
from pathlib import Path
assert Path(medical_jury.__file__).resolve().is_relative_to(Path(SOURCE_ROOT).resolve()), 'Wrong source checkout'
import pytest
def blocked(*args, **kwargs):
    raise AssertionError('External and manual network access disabled during offline checks')
real_connect, real_pair = socket.socket.connect, socket.socketpair
local = threading.local()
def guarded_connect(sock, address):
    if getattr(local, 'inside_pair', False) and isinstance(address, tuple) and address[0] in ('127.0.0.1', '::1'):
        return real_connect(sock, address)
    return blocked()
def guarded_pair(*args, **kwargs):
    previous = getattr(local, 'inside_pair', False)
    local.inside_pair = True
    try:
        return real_pair(*args, **kwargs)
    finally:
        local.inside_pair = previous
with patch.object(socket.socket, 'connect', guarded_connect), patch.object(socket.socket, 'connect_ex', blocked), patch.object(socket, 'create_connection', blocked), patch.object(socket, 'getaddrinfo', blocked), patch.object(socket, 'socketpair', guarded_pair):
    a, b = socket.socketpair()
    try:
        a.send(b'x')
        assert b.recv(1) == b'x'
    finally:
        a.close(); b.close()
    for host in ('127.0.0.1', '198.51.100.1'):
        s = socket.socket()
        try:
            try: s.connect((host, 443))
            except AssertionError: pass
            else: raise AssertionError('Manual connection escaped guard')
        finally: s.close()
    try: socket.getaddrinfo('example.invalid', 443)
    except AssertionError: pass
    else: raise AssertionError('DNS escaped guard')
    print('OFFLINE_GUARD_VERIFIED_INTERNAL_SOCKETPAIR_ONLY')
    raise SystemExit(pytest.main(TEST_ARGS))
'''


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:8]
    out = ROOT / 'reports' / ('public-offline-' + stamp)
    temp = ROOT / 'data/shadow-item-evidence-v1' / ('public-check-' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    temp.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1', PYTHONDONTWRITEBYTECODE='1',
               ANONYMIZED_TELEMETRY='False', LANGCHAIN_TRACING_V2='false',
               PYTHONPATH=str(ROOT / 'src'))
    # No inherited model/agent credentials are needed, even for mocked tests.
    for key in list(env):
        if key.endswith(('_API_KEY', '_API_TOKEN')) or key in {'DEEPSEEK_API_KEY', 'LANGCHAIN_API_KEY'}:
            env.pop(key)
    results = []
    for name, target, expected in SUITES:
        xml = out / (name + '.xml')
        args = [str(ROOT / target), '-q', '-p', 'no:cacheprovider', '--tb=short',
                '--basetemp=' + str(temp / name), '--junitxml=' + str(xml)]
        if name != 'core':
            args = ['-c', str(ROOT / 'experiments/item-evidence-shadow-v1/pytest.ini')] + args
        child = CHILD.replace('SOURCE_ROOT', repr(str(ROOT / 'src'))).replace('TEST_ARGS', repr(args))
        print('Running offline suite:', name, flush=True)
        run = subprocess.run([sys.executable, '-X', 'utf8', '-B', '-c', child], cwd=ROOT,
                             env=env, capture_output=True, text=True, encoding='utf-8', timeout=300)
        (out / (name + '.log')).write_text(run.stdout + run.stderr, encoding='utf-8')
        counts = {'tests': 0, 'failures': 0, 'errors': 0, 'skipped': 0}
        if xml.exists():
            tree = ET.parse(xml).getroot()
            for suite in tree.iter('testsuite'):
                for key in counts:
                    counts[key] += int(suite.get(key, '0'))
        allowed_skips = (1 if name == 'core' and not (ROOT / 'data/facts/v1/manifest.json').is_file() else 0)
        skipped_names = [case.get('name') for case in tree.iter('testcase') if case.find('skipped') is not None] if xml.exists() else []
        skips_match = (skipped_names == ['test_real_reviewed_snapshots_have_integrity'] if allowed_skips else not skipped_names)
        passed = (run.returncode == 0 and counts['tests'] == expected
                  and counts['failures'] == counts['errors'] == 0
                  and counts['skipped'] == allowed_skips and skips_match
                  and 'OFFLINE_GUARD_VERIFIED_INTERNAL_SOCKETPAIR_ONLY' in run.stdout)
        results.append({'suite': name, 'expected_tests': expected, 'returncode': run.returncode,
                        'counts': counts, 'skipped_tests': skipped_names, 'passed': passed})
        print(json.dumps(results[-1]), flush=True)
    summary = {'created_at_utc': stamp, 'paid_model_calls': 0, 'network_guard': 'verified',
               'source_checkout': str(ROOT), 'suites': results,
               'total_tests': sum(row['counts']['tests'] for row in results),
               'total_skipped': sum(row['counts']['skipped'] for row in results),
               'total_passed': sum(row['counts']['tests'] - row['counts']['failures'] - row['counts']['errors'] - row['counts']['skipped'] for row in results),
               'all_passed': all(row['passed'] for row in results),
               'note': 'Engineering contract tests, not medical validation cases.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print('Report:', out)
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
