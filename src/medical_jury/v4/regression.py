"""Regression checks for v4 (offline, no paid call).

* mock_digests: every testbed case through the LangGraph chain with the oracle mock backend -> one digest per case.
  Catches any change in the code path (extraction checks, rules, criteria, merge, reconcile, triage, persist).
* replay_record: a sealed record from a REAL run is replayed through everything after the model calls (rules, criteria
  selection, merge, reconcile, triage) using the model outputs the record kept (extraction, criterion verdicts, open
  findings with their pre-reconcile severity). Catches changes that would alter real decisions, for free.
  The raw provider text is not stored (privacy), so the two model calls themselves are not replayed; quote checks use
  the values recorded at run time.
"""
from __future__ import annotations

from pathlib import Path

from ..domain import digest
from . import PIPELINE_VERSION
from .check import CHECK_PROMPT_VERSION, SYSTEM_CHECK, code_checks_of
from .criteria import applicable
from .extract import EXTRACT_PROMPT_VERSION, SYSTEM_EXTRACT
from .pipeline import merge_findings, triage
from .rules import RULES_SHA256, RULES_VERSION, check as rule_check


def versions(library: dict) -> dict:
    return {'pipeline': PIPELINE_VERSION, 'rules_version': RULES_VERSION, 'rules_sha256': RULES_SHA256,
            'extract_prompt': {'version': EXTRACT_PROMPT_VERSION, 'sha256': digest(SYSTEM_EXTRACT)},
            'check_prompt': {'version': CHECK_PROMPT_VERSION, 'sha256': digest(SYSTEM_CHECK)},
            'library_sha256': library['sha256']}


def timeless(rec: dict) -> dict:
    """Wall-clock latency differs between any two runs; everything else must match."""
    return {**rec, 'calls': [{k: v for k, v in c.items() if k != 'latency_s'} for c in rec.get('calls', [])]}


def mock_digests(root: Path, library: dict, cases: list[dict]) -> dict[str, str]:
    from ..judge_v2 import ScriptedBackend
    from .graph import ReviewGraphV4
    from .mocks import oracle_responder, plain_document
    backend = ScriptedBackend(oracle_responder({(c['patient_summary'], plain_document(c)): c for c in cases}))
    graph = ReviewGraphV4(backend, library, require_human=False, call_deadline_s=None)
    out = {}
    for c in cases:
        rec = graph.start(c['patient_summary'], plain_document(c), f"reg:{c['case_id']}")['record']
        out[c['case_id']] = digest(timeless(rec))
    return out


def _opened_from_record(findings: list[dict]) -> list[dict]:
    """The check call's open findings as they were BEFORE reconciliation (severity as the model gave it)."""
    out = []
    for f in findings:
        if f.get('source') != 'open_review':
            continue
        g = {k: v for k, v in f.items() if k not in ('reconciled', 'severity_model')}
        if 'severity_model' in f:
            g['severity'] = f['severity_model']
        out.append(g)
    return out


def replay_record(rec: dict, library: dict) -> dict:
    """{'status': identical | changed | needs_model_call | not_replayable, ...}"""
    ex = rec.get('extraction')
    if rec.get('arm') != 'full' or ex is None or rec.get('status') == 'Cancelled':
        return {'status': 'not_replayable'}
    facts, meds = ex['facts'], ex['medications']
    rule_out = rule_check(facts, meds)
    items, code_findings = applicable(library, facts, meds)
    if [{'id': i['id'], 'level': i['level'], 'text': i['text']} for i in items] != rec['items']:
        return {'status': 'needs_model_call', 'why': 'the applicable criteria changed; new ones were never answered'}
    check_call = next((c for c in rec['calls'] if c['step'] == 'check'), None)
    ok = check_call is not None and check_call.get('status') == 'Completed'
    chk = {'criteria': rec['criteria_results']} if ok else None
    findings, uncertain = merge_findings(rule_out, code_findings, items, chk,
                                         _opened_from_record(rec['findings']) if ok else [],
                                         code_checks_of(rule_out, meds))
    tri = triage(findings, uncertain, ex['unknown_active_drugs'], ex['problems'],
                 [] if ok else [check_call['failure_code'] if check_call else 'NO_CHECK_CALL'])
    if findings == rec['findings'] and uncertain == rec['uncertain'] and tri == rec['triage']:
        return {'status': 'identical'}
    return {'status': 'changed', 'triage_before': rec['triage'], 'triage_after': tri,
            'findings_before': len(rec['findings']), 'findings_after': len(findings),
            'severe_before': sum(f['severity'] == 'severe' for f in rec['findings']),
            'severe_after': sum(f['severity'] == 'severe' for f in findings)}


def same_versions(rec: dict, library: dict) -> bool:
    v = versions(library)
    return (rec.get('pipeline') == v['pipeline'] and rec.get('rules_version') == v['rules_version']
            and rec.get('extract_prompt') == v['extract_prompt'] and rec.get('check_prompt') == v['check_prompt']
            and rec.get('library_sha256') == v['library_sha256'])
