"""Offline engineering checks; no clinical expected answers or provider imports."""
from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import pytest

from shadow import (ROOT, STORE_BASE, ShadowAdapter, ShadowError, ShadowStore,
                    canonical, json_sha, validate_snapshot, contract_module)

sys.path.insert(0, str(ROOT / 'experiments/item-evidence-v1'))
from test_contract import fixture_pair


@pytest.fixture(autouse=True)
def network_off(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('No network in shadow tests.')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket.socket, 'connect_ex', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


@pytest.fixture
def store(tmp_path):
    # The runner pins basetemp to STORE_BASE, never to the real application DB.
    assert tmp_path.resolve().is_relative_to(STORE_BASE.resolve())
    return ShadowStore(tmp_path / 'sidecar')


def primary():
    return {'judgment': 'not_met', 'reason': 'Original toy result.', 'answer_evidence': ['blue'],
            'context_evidence': [], 'review_required': False}


def args():
    i, o = fixture_pair()
    return {'request_id': 'toy-1', 'snapshot': i, 'candidate_json': json.dumps(o, ensure_ascii=False),
            'origin': 'synthetic_fixture', 'case_reference': 'SYNTHETIC-colour'}


def test_default_off_no_write_no_inspector_no_payload_access(store, monkeypatch):
    class Explosive:
        def __iter__(self):
            raise AssertionError('Should not inspect disabled payload.')
    a = ShadowAdapter(store=store)
    p = Explosive()
    assert a.passthrough(p, snapshot=Explosive()) is p
    assert a.last_receipt['status'] == 'disabled'
    assert not store.root.exists()


def test_disabled_ignores_environment_flag(store, monkeypatch):
    monkeypatch.setenv('ITEM_EVIDENCE_SHADOW_ENABLED', 'true')
    a = ShadowAdapter(store=store)
    assert a.observe(primary(), **args())['status'] == 'disabled'
    assert not store.root.exists()


@pytest.mark.parametrize('value', ['true', 1, None])
def test_explicit_boolean_required(value):
    with pytest.raises(ShadowError, match='ENABLED_MUST_BE_EXPLICIT_BOOLEAN'):
        ShadowAdapter(enabled=value)


def test_enabled_records_separately_without_mutating_any_primary_fields(store):
    p, a = primary(), ShadowAdapter(enabled=True, store=store)
    before = deepcopy(p)
    result = a.passthrough(p, **args())
    assert result is p and p == before
    assert a.last_receipt['status'] == 'recorded'
    r = store.records()[0]
    assert r['candidate_payload']['model_verdict'] == 'met'  # Deliberately conflicts with toy primary.
    assert r['primary_result_body'] is None and r['primary_result_sha256'] == json_sha(p)
    assert r['inspection']['model_verdict'] == 'met' and p['judgment'] == 'not_met'
    assert r['official_score'] is None and not r['clinical_release'] and not r['benchmark_export_eligible']


def test_switch_off_does_not_delete_or_rewrite_previous_history(store):
    ShadowAdapter(enabled=True, store=store).observe(primary(), **args())
    before = store.records()
    disabled = ShadowAdapter(store=store)
    assert disabled.observe(primary(), **args())['status'] == 'disabled'
    assert store.records() == before


def test_same_request_is_idempotent_and_different_content_conflicts(store):
    a, p = ShadowAdapter(enabled=True, store=store), primary()
    assert a.observe(p, **args())['status'] == 'recorded'
    first = store.records()
    assert a.observe(p, **args())['status'] == 'already_recorded'
    p['reason'] = 'Different result, same request id.'
    receipt = a.observe(p, **args())
    assert receipt['diagnostic_code'] == 'IDEMPOTENCY_CONFLICT'
    assert store.records() == first


def test_snapshot_full_hash_freezes_non_authoritative_task_hint(store):
    a = ShadowAdapter(enabled=True, store=store)
    kw = args()
    assert a.observe(primary(), **kw)['status'] == 'recorded'
    kw['snapshot']['task_hint'] = 'clinical_request'
    # Old contract's case binding did not include hint; sidecar full snapshot hash does.
    receipt = a.observe(primary(), **kw)
    assert receipt['diagnostic_code'] == 'IDEMPOTENCY_CONFLICT'
    assert len(store.records()) == 1


@pytest.mark.parametrize('candidate,expected', [('not json', 'INVALID_JSON'),
                                              ('{"x":1,"x":2}', 'DUPLICATE_JSON_KEY')])
def test_malformed_candidate_has_safe_failure_record_primary_unchanged(store, candidate, expected):
    a, p, kw = ShadowAdapter(enabled=True, store=store), primary(), args()
    kw['candidate_json'] = candidate
    before = deepcopy(p)
    assert a.passthrough(p, **kw) is p and p == before
    r = store.records()[0]
    assert r['state'] == 'candidate_protocol_invalid'
    assert r['candidate_payload'] is None
    assert expected in r['diagnostic_codes']
    assert r['inspection']['score'] is None


def test_unknown_evidence_id_is_rejected_without_becoming_primary_failure(store):
    kw = args()
    o = json.loads(kw['candidate_json'])
    o['claims'][0]['location']['doc_id'] = 'S999'
    kw['candidate_json'] = json.dumps(o)
    a, p = ShadowAdapter(enabled=True, store=store), primary()
    assert a.passthrough(p, **kw) is p
    assert store.records()[0]['diagnostic_codes'] == ['UNKNOWN_DOCUMENT_ID']


def test_missing_external_evidence_stays_valid_needs_review(store):
    kw = args()
    o = json.loads(kw['candidate_json'])
    o['checks'] = []
    kw['candidate_json'] = json.dumps(o)
    a = ShadowAdapter(enabled=True, store=store)
    a.observe(primary(), **kw)
    r = store.records()[0]
    assert r['state'] == 'candidate_protocol_valid_needs_review'
    assert r['inspection']['model_verdict'] == 'met'
    assert 'NO_BOUND_EXTERNAL_SOURCE_CHECKS' in r['inspection']['consistency_flags']


def test_no_candidate_cannot_be_invented_from_legacy_reason(store):
    a = ShadowAdapter(enabled=True, store=store)
    a.observe(primary(), request_id='legacy', origin='historical_projection', case_reference='REVIEW-01')
    r = store.records()[0]
    assert r['state'] == 'legacy_requires_new_instrumentation'
    assert r['candidate_payload'] is None and r['snapshot'] is None and r['inspection'] is None
    assert r['case_status'] == 'unresolved_no_independent_review'


def test_candidate_without_snapshot_is_not_graded(store):
    kw = args()
    kw['snapshot'] = None
    ShadowAdapter(enabled=True, store=store).observe(primary(), **kw)
    r = store.records()[0]
    assert r['state'] == 'not_inspected' and r['inspection'] is None


@pytest.mark.parametrize('mutation,code', [('label', 'SNAPSHOT_FIELD_ALLOWLIST'),
                                         ('document', 'SNAPSHOT_INTEGRITY_MISMATCH')])
def test_bad_snapshot_rejected_and_primary_preserved(store, mutation, code):
    kw = args()
    if mutation == 'label':
        kw['snapshot']['physician_labels'] = [True]
    else:
        kw['snapshot']['documents']['A']['text'] = 'tampered'
    a, p = ShadowAdapter(enabled=True, store=store), primary()
    assert a.passthrough(p, **kw) is p
    assert a.last_receipt['diagnostic_code'] == code
    assert not store.root.exists()


def test_source_catalog_cannot_supply_extra_model_metadata(store):
    kw = args()
    kw['snapshot']['documents']['S1']['credential'] = 'must not be copied'
    receipt = ShadowAdapter(enabled=True, store=store).observe(primary(), **kw)
    assert receipt['diagnostic_code'] == 'SNAPSHOT_INTEGRITY_MISMATCH'


def test_secretlike_candidate_is_hashed_not_written_verbatim(store):
    kw = args()
    secret = 'sk-' + 'x' * 32
    o = json.loads(kw['candidate_json'])
    o['model_reason'] = secret
    kw['candidate_json'] = json.dumps(o)
    a = ShadowAdapter(enabled=True, store=store)
    receipt = a.observe(primary(), **kw)
    assert secret not in json.dumps(receipt) and secret not in canonical(store.records())
    assert store.records()[0]['state'] == 'payload_withheld'
    assert store.records()[0]['candidate_payload'] is None
    assert secret.encode() not in (store.root / 'shadow.sqlite').read_bytes()


def test_oversized_candidate_is_not_silently_truncated_or_stored(store):
    kw = args()
    kw['candidate_json'] = 'x' * 96001
    ShadowAdapter(enabled=True, store=store).observe(primary(), **kw)
    r = store.records()[0]
    assert r['candidate_payload'] is None and r['candidate_bytes'] == 96001
    assert r['diagnostic_codes'] == ['CANDIDATE_BYTE_LIMIT']


@pytest.mark.parametrize('failure', [OSError('private detail'), RuntimeError('secret detail'),
                                   sqlite3.OperationalError('locked detail')])
def test_store_failure_never_escapes_or_echoes_error_body(failure):
    class BrokenStore:
        def write(self, *args):
            raise failure
    a, p = ShadowAdapter(enabled=True, store=BrokenStore()), primary()
    assert a.passthrough(p, **args()) is p
    assert a.last_receipt['status'] == 'shadow_failed'
    assert 'detail' not in json.dumps(a.last_receipt)


def test_reject_writes_outside_dedicated_store_namespace():
    a = ShadowAdapter(enabled=True, store=ShadowStore(ROOT / 'data'))
    p = primary()
    assert a.passthrough(p, **args()) is p
    assert a.last_receipt['diagnostic_code'] == 'STORE_OUTSIDE_SHADOW_NAMESPACE'


def test_db_rows_are_append_only(store):
    ShadowAdapter(enabled=True, store=store).observe(primary(), **args())
    db = sqlite3.connect(store.root / 'shadow.sqlite')
    try:
        with pytest.raises(sqlite3.IntegrityError, match='append only'):
            db.execute('DELETE FROM observations')
        with pytest.raises(sqlite3.IntegrityError, match='append only'):
            db.execute("UPDATE observations SET fingerprint='fake'")
    finally:
        db.close()
    assert len(store.records()) == 1


def test_store_schema_version_mismatch_is_not_automatically_migrated(store):
    store.root.mkdir(parents=True)
    db = sqlite3.connect(store.root / 'shadow.sqlite')
    db.execute('PRAGMA user_version=99')
    db.close()
    a = ShadowAdapter(enabled=True, store=store)
    a.observe(primary(), **args())
    assert a.last_receipt['diagnostic_code'] == 'STORE_VERSION_UNSUPPORTED'


def test_record_tamper_detected_on_read(store):
    ShadowAdapter(enabled=True, store=store).observe(primary(), **args())
    db = sqlite3.connect(store.root / 'shadow.sqlite')
    db.execute('DROP TRIGGER observations_no_update')
    db.execute("UPDATE observations SET record_sha256='fake'")
    db.commit()
    db.close()
    with pytest.raises(ShadowError, match='STORED_RECORD_INTEGRITY_ERROR'):
        store.records()


def test_parallel_idempotency_does_not_duplicate_records(store):
    a = ShadowAdapter(enabled=True, store=store)
    # Initialize DDL once before racing independent transactions.
    a.observe(primary(), **args())
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda _: a.observe(primary(), **args()), range(6)))
    assert all(r['status'] == 'already_recorded' for r in results)
    assert len(store.records()) == 1


def test_reading_empty_store_does_not_create_it(store):
    assert store.records() == [] and not store.root.exists()


def test_cli_disabled_does_not_read_even_nonexistent_files():
    script = Path(__file__).with_name('offline_cli.py')
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', str(script), '--primary-file', 'does-not-exist'],
                            capture_output=True, text=True, encoding='utf-8', env=env, timeout=10)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {'status': 'disabled', 'files_read': 0, 'records_written': 0}


def test_cli_enabled_reads_only_supplied_files_and_records_separately(store):
    kw = args()
    directory = store.root.parent
    directory.mkdir(parents=True, exist_ok=True)
    paths = {'primary': primary(), 'snapshot': kw['snapshot'], 'candidate': json.loads(kw['candidate_json'])}
    for name, value in paths.items():
        (directory / (name + '.json')).write_text(json.dumps(value), encoding='utf-8')
    command = [sys.executable, '-B', str(Path(__file__).with_name('offline_cli.py')), '--enable-shadow',
               '--request-id', 'cli-toy', '--store-root', str(store.root)]
    for name in paths:
        command.extend(['--' + name + '-file', str(directory / (name + '.json'))])
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=10)
    assert result.returncode == 0 and json.loads(result.stdout)['status'] == 'recorded'
    assert len(store.records()) == 1


def test_primary_non_json_is_passed_through_without_hiding_failure(store):
    p, a = object(), ShadowAdapter(enabled=True, store=store)
    assert a.passthrough(p, **args()) is p
    assert a.last_receipt['diagnostic_code'] == 'NON_JSON_VALUE'


def test_bad_shadow_call_cannot_break_primary_passthrough(store):
    a, p = ShadowAdapter(enabled=True, store=store), primary()
    assert a.passthrough(p, unexpected_option='bad-call') is p
    assert a.last_receipt['diagnostic_code'] == 'SHADOW_INVOCATION_ERROR'


@pytest.mark.parametrize('metadata,code', [({'request_id': '../escape'}, 'INVALID_REQUEST_ID'),
                                         ({'case_reference': 'unknown'}, 'INVALID_CASE_REFERENCE'),
                                         ({'origin': 'paid_live'}, 'UNSUPPORTED_ORIGIN')])
def test_metadata_allowlist_rejected_without_touching_primary(store, metadata, code):
    kw = args()
    kw.update(metadata)
    a, p = ShadowAdapter(enabled=True, store=store), primary()
    assert a.passthrough(p, **kw) is p
    assert a.last_receipt['diagnostic_code'] == code
