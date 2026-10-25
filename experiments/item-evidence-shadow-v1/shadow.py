"""Default-off offline sidecar. Not imported by production and never calls a model.

Only the caller can supply observations. Primary objects pass through unchanged;
shadow failures produce diagnostic receipts, never change grading or API results.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORE_BASE = ROOT / 'data/shadow-item-evidence-v1'
CONTRACT_PATH = ROOT / 'experiments/item-evidence-v1/contract.py'
CONTRACT_SHA256 = '2d3b840ac1e0aafe54e46f06d265990dda496e60d18855f20c04b9a21c0292e8'
VERSION = 'offline-shadow-sidecar-v1'
MAX_SNAPSHOT_BYTES = 2_000_000
MAX_PRIMARY_BYTES = 2_000_000
MAX_ROWS = 5000
ORIGINS = {'synthetic_fixture', 'historical_original', 'historical_projection'}
REF_PATTERN = re.compile(r'^(?:REVIEW-0[1-7]|SYNTHETIC-[A-Za-z0-9-]{1,48})$')
ID_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$')
SECRET_PATTERN = re.compile(r'(?i)sk-[a-z0-9_-]{16,}|bearer\s+[a-z0-9._-]{16,}|-----BEGIN (?:RSA |EC )?PRIVATE KEY-----')
SNAPSHOT_KEYS = {'binding', 'documents', 'manifest', 'task_hint', 'context', 'candidate_output', 'rubric_text'}


class ShadowError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def json_sha(value):
    return text_sha(canonical(value))


def frozen_json(value, bound):
    try:
        encoded = canonical(value)
    except (TypeError, ValueError, RecursionError):
        raise ShadowError('NON_JSON_VALUE') from None
    if len(encoded.encode('utf-8')) > bound:
        raise ShadowError('JSON_BOUND_EXCEEDED')
    return json.loads(encoded), text_sha(encoded)


def contract_module():
    # Pinned pre-existing checker: do not mutate the tested iteration-11 module.
    if hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest() != CONTRACT_SHA256:
        raise ShadowError('CONTRACT_VERSION_CHANGED')
    spec = importlib.util.spec_from_file_location('frozen_shadow_contract_v1', CONTRACT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_snapshot(value, contract):
    frozen, snapshot_hash = frozen_json(value, MAX_SNAPSHOT_BYTES)
    if not isinstance(frozen, dict) or set(frozen) != SNAPSHOT_KEYS:
        raise ShadowError('SNAPSHOT_FIELD_ALLOWLIST')
    if not isinstance(frozen['documents'], dict) or not isinstance(frozen['manifest'], dict):
        raise ShadowError('SNAPSHOT_SHAPE')
    sources = [(k, d) for k, d in frozen['documents'].items() if re.fullmatch(r'S[1-9][0-9]{0,2}', k)]
    sources.sort(key=lambda item: int(item[0][1:]))
    if [k for k, _ in sources] != [f'S{i}' for i in range(1, len(sources) + 1)]:
        raise ShadowError('NONCONTIGUOUS_SOURCE_CATALOG')
    try:
        rebuilt = contract.prepare_input(
            frozen['context'], frozen['candidate_output'], frozen['rubric_text'],
            frozen['manifest']['required'], manifest_status=frozen['manifest']['status'],
            source_texts=[d['text'] for _, d in sources], task_hint=frozen['task_hint'])
    except (contract.ContractError, KeyError, TypeError, ValueError, AttributeError):
        raise ShadowError('SNAPSHOT_RECONSTRUCTION_FAILED') from None
    if rebuilt != frozen:
        raise ShadowError('SNAPSHOT_INTEGRITY_MISMATCH')
    # This full hash also freezes task_hint, not just the four legacy binding hashes.
    return frozen, snapshot_hash


class ShadowStore:
    """Local append-only records in a dedicated namespace; no primary database access.

    No file is created by construction. Writes use bounded SQLite transactions.
    This is not encryption or independent protection against a machine owner.
    """
    def __init__(self, root=None):
        self.root = Path(root) if root is not None else STORE_BASE / 'manual-offline'

    def _path(self):
        base, root = STORE_BASE.resolve(), self.root.resolve()
        if base != ROOT.resolve() / 'data/shadow-item-evidence-v1':
            raise ShadowError('SHADOW_NAMESPACE_REDIRECTED')
        if not root.is_relative_to(base):
            raise ShadowError('STORE_OUTSIDE_SHADOW_NAMESPACE')
        path = root / 'shadow.sqlite'
        if path.resolve() != path or not path.resolve().is_relative_to(base):
            raise ShadowError('STORE_PATH_REDIRECTED')
        return path

    def _connect(self):
        path = self._path()
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=0.25)
        try:
            db.execute('PRAGMA busy_timeout=250')
            version = db.execute('PRAGMA user_version').fetchone()[0]
            if version not in (0, 1):
                raise ShadowError('STORE_VERSION_UNSUPPORTED')
            db.executescript('''
                CREATE TABLE IF NOT EXISTS observations (
                    request_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    record_sha256 TEXT NOT NULL
                );
                CREATE TRIGGER IF NOT EXISTS observations_no_update BEFORE UPDATE ON observations
                  BEGIN SELECT RAISE(ABORT, 'append only'); END;
                CREATE TRIGGER IF NOT EXISTS observations_no_delete BEFORE DELETE ON observations
                  BEGIN SELECT RAISE(ABORT, 'append only'); END;
                PRAGMA user_version=1;
            ''')
            return db
        except Exception:
            db.close()
            raise

    def write(self, request_id, fingerprint, record):
        payload = canonical(record)
        payload_sha = text_sha(payload)
        db = self._connect()
        try:
            with db:
                db.execute('BEGIN IMMEDIATE')
                existing = db.execute('SELECT fingerprint,record_json,record_sha256 FROM observations WHERE request_id=?',
                                      (request_id,)).fetchone()
                if existing:
                    if text_sha(existing[1]) != existing[2]:
                        raise ShadowError('STORED_RECORD_INTEGRITY_ERROR')
                    if existing[0] != fingerprint:
                        raise ShadowError('IDEMPOTENCY_CONFLICT')
                    return {'status': 'already_recorded', 'record_sha256': existing[2]}
                if db.execute('SELECT COUNT(*) FROM observations').fetchone()[0] >= MAX_ROWS:
                    raise ShadowError('STORE_RECORD_LIMIT')
                db.execute('INSERT INTO observations VALUES (?,?,?,?)', (request_id, fingerprint, payload, payload_sha))
            return {'status': 'recorded', 'record_sha256': payload_sha}
        finally:
            db.close()

    def records(self):
        path = self._path()
        if not path.exists():
            return []
        db = sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True, timeout=0.25)
        try:
            result = []
            for payload, expected in db.execute('SELECT record_json,record_sha256 FROM observations ORDER BY request_id'):
                if text_sha(payload) != expected:
                    raise ShadowError('STORED_RECORD_INTEGRITY_ERROR')
                result.append(json.loads(payload))
            return result
        finally:
            db.close()


class ShadowAdapter:
    """Explicit opt-in side observer. No environment variable auto-enables it.

    observe() returns a shadow-only receipt; passthrough() returns exactly its
    primary_result object. last_receipt is thread-local and has no clinical score.
    The application has no hook to this class in this release.
    """
    def __init__(self, *, enabled=False, store=None):
        if type(enabled) is not bool:
            raise ShadowError('ENABLED_MUST_BE_EXPLICIT_BOOLEAN')
        self.enabled = enabled
        self.store = store if store is not None else ShadowStore()
        self._local = threading.local()

    @property
    def last_receipt(self):
        return getattr(self._local, 'receipt', None)

    def passthrough(self, primary_result, **kwargs):
        try:
            self.observe(primary_result, **kwargs)
        except Exception:
            self._local.receipt = {'status': 'shadow_failed', 'diagnostic_code': 'SHADOW_INVOCATION_ERROR',
                                   'record_written': False, 'primary_changed': False}
        return primary_result

    def observe(self, primary_result, *, request_id=None, snapshot=None, candidate_json=None,
                origin='synthetic_fixture', case_reference=None):
        if not self.enabled:
            receipt = {'status': 'disabled', 'record_written': False, 'primary_changed': False}
            self._local.receipt = receipt
            return receipt
        try:
            receipt = self._capture(primary_result, request_id=request_id, snapshot=snapshot,
                                    candidate_json=candidate_json, origin=origin, case_reference=case_reference)
        except ShadowError as exc:
            receipt = {'status': 'shadow_failed', 'diagnostic_code': exc.code,
                       'record_written': False, 'primary_changed': False}
        except sqlite3.Error:
            receipt = {'status': 'shadow_failed', 'diagnostic_code': 'SHADOW_STORAGE_ERROR',
                       'record_written': False, 'primary_changed': False}
        except Exception:
            # No arbitrary exception text, raw inputs, filesystem paths or secrets in receipts.
            receipt = {'status': 'shadow_failed', 'diagnostic_code': 'SHADOW_INTERNAL_ERROR',
                       'record_written': False, 'primary_changed': False}
        self._local.receipt = receipt
        return receipt

    def _capture(self, primary_result, *, request_id, snapshot, candidate_json, origin, case_reference):
        if not isinstance(request_id, str) or not ID_PATTERN.fullmatch(request_id):
            raise ShadowError('INVALID_REQUEST_ID')
        if origin not in ORIGINS:
            raise ShadowError('UNSUPPORTED_ORIGIN')
        if case_reference is not None and (not isinstance(case_reference, str) or not REF_PATTERN.fullmatch(case_reference)):
            raise ShadowError('INVALID_CASE_REFERENCE')
        if SECRET_PATTERN.search(request_id):
            raise ShadowError('SECRETLIKE_METADATA')
        _, primary_hash = frozen_json(primary_result, MAX_PRIMARY_BYTES)
        contract = contract_module()
        state = 'legacy_requires_new_instrumentation'
        snapshot_hash, candidate_hash, candidate_bytes = None, None, None
        saved_snapshot, saved_candidate, inspection = None, None, None
        diagnostics = []
        if candidate_json is not None:
            if not isinstance(candidate_json, str):
                raise ShadowError('CANDIDATE_MUST_BE_RAW_JSON_STRING')
            candidate_hash = text_sha(candidate_json)
            candidate_bytes = len(candidate_json.encode('utf-8'))
            if candidate_bytes > contract.MAX_BYTES:
                state, diagnostics = 'candidate_rejected', ['CANDIDATE_BYTE_LIMIT']
            elif SECRET_PATTERN.search(candidate_json):
                state, diagnostics = 'payload_withheld', ['SECRETLIKE_CANDIDATE_WITHHELD']
            elif snapshot is None:
                state, diagnostics = 'not_inspected', ['NO_FROZEN_SNAPSHOT']
            else:
                saved_snapshot, snapshot_hash = validate_snapshot(snapshot, contract)
                if SECRET_PATTERN.search(canonical(saved_snapshot)):
                    saved_snapshot = None
                    state, diagnostics = 'payload_withheld', ['SECRETLIKE_SNAPSHOT_WITHHELD']
                else:
                    inspection = contract.inspect(candidate_json, saved_snapshot)
                    if inspection['protocol_status'] == 'valid':
                        saved_candidate = contract.strict_loads(candidate_json)
                        state = 'candidate_protocol_valid_needs_review'
                    else:
                        state = 'candidate_protocol_invalid'
                        diagnostics = [e['code'] for e in inspection['protocol_errors']]
                        # Invalid outputs are hashed, not echoed or silently repaired.
                        inspection = {k: v for k, v in inspection.items() if k != 'model_reason'}
                        if inspection.get('model_verdict') not in {'met', 'not_met'}:
                            inspection['model_verdict'] = None
        else:
            diagnostics = ['NO_ITEM_OUTPUT_PROVIDED_NO_FIELDS_FABRICATED']
        fingerprint = json_sha({'primary': primary_hash, 'snapshot': snapshot_hash, 'candidate': candidate_hash,
                                'origin': origin, 'case_reference': case_reference, 'version': VERSION})
        record = {'schema_version': VERSION, 'request_id': request_id,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(),
                  'origin': origin, 'origin_attestation': 'caller_declared_offline_material',
                  'case_reference': case_reference,
                  'case_status': 'unresolved_no_independent_review' if case_reference and case_reference.startswith('REVIEW-')
                                 else 'synthetic_or_unadjudicated',
                  'primary_result_sha256': primary_hash, 'primary_result_body': None,
                  'snapshot_sha256': snapshot_hash, 'snapshot': saved_snapshot,
                  'candidate_sha256': candidate_hash, 'candidate_bytes': candidate_bytes,
                  'candidate_payload': saved_candidate, 'state': state, 'inspection': inspection,
                  'diagnostic_codes': diagnostics, 'clinical_release': False,
                  'benchmark_export_eligible': False, 'official_score': None,
                  'paid_call_performed': False, 'new_medical_gold_label': False,
                  'legacy_result_modified': False, 'fingerprint': fingerprint}
        stored = self.store.write(request_id, fingerprint, record)
        return {**stored, 'record_written': stored['status'] == 'recorded', 'shadow_state': state,
                'primary_changed': False, 'diagnostic_codes': diagnostics}
