"""v4 regression gate and tracing: offline tests only."""
from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

import regression_gate as GATE  # noqa: E402
from medical_jury.judge_v2 import ScriptedBackend  # noqa: E402
from medical_jury.testbed.build import build_cases  # noqa: E402
from medical_jury.v4 import criteria as CR, rules as R  # noqa: E402
from medical_jury.v4.graph import JsonlTracer, ReviewGraphV4  # noqa: E402
from medical_jury.v4.mocks import oracle_responder, plain_document  # noqa: E402
from medical_jury.v4.pipeline import ReviewerV4  # noqa: E402
from medical_jury.v4.regression import replay_record  # noqa: E402


@pytest.fixture(scope='module')
def cases():
    return build_cases(150, 20261112)[0]


@pytest.fixture(scope='module')
def library():
    return CR.load_library(ROOT)


def _stricter_metformin(monkeypatch):
    rules = copy.deepcopy(R.RULES)
    met = next(r for r in rules['rules'] if r['id'] == 'METFORMIN_EGFR')
    met['bands'][0]['below'] = 50                                  # a silent clinical change
    monkeypatch.setattr(R, 'RULES', rules)


def test_current_code_matches_the_stored_baseline():
    code, summary = GATE.mock_gate(GATE.BASELINE, accept=False)
    assert code == 0, summary


def test_a_silent_rule_change_is_refused(tmp_path, monkeypatch):
    base = tmp_path / 'b.json'
    shutil.copy(GATE.BASELINE, base)
    _stricter_metformin(monkeypatch)
    code, summary = GATE.mock_gate(base, accept=True)               # --accept cannot launder a silent change
    assert code == 1 and summary['changed'] > 0 and 'no version changed' in summary['result']
    assert json.loads(base.read_text()) == json.loads(GATE.BASELINE.read_text())


def test_a_versioned_change_needs_explicit_acceptance(tmp_path, monkeypatch):
    base = tmp_path / 'b.json'
    shutil.copy(GATE.BASELINE, base)
    _stricter_metformin(monkeypatch)
    import medical_jury.v4.regression as REG
    monkeypatch.setattr(REG, 'RULES_VERSION', 'agent-rules-test')
    code, summary = GATE.mock_gate(base, accept=False)
    assert code == 1 and 'rules_version' in summary['versions_changed']
    assert GATE.mock_gate(base, accept=True)[0] == 0


def _records(cases, library, n=40):
    backend = ScriptedBackend(oracle_responder({(c['patient_summary'], plain_document(c)): c for c in cases}))
    rev = ReviewerV4(backend, library)
    return [{**rev.review(c['patient_summary'], plain_document(c), c['case_id']), 'case_id': c['case_id']}
            for c in cases[:n]]


def test_replay_reproduces_records_and_detects_changes(cases, library, monkeypatch):
    recs = _records(cases, library)
    assert {replay_record(r, library)['status'] for r in recs} == {'identical'}
    tampered = copy.deepcopy(recs[0])
    tampered['triage'] = {'decision': 'pass', 'reasons': []} if recs[0]['triage']['decision'] != 'pass' else \
        {'decision': 'escalate', 'reasons': ['severe_finding']}
    assert replay_record(tampered, library)['status'] == 'changed'
    moved = copy.deepcopy(recs[0])
    moved['items'][0]['text'] += ' (reworded)'
    assert replay_record(moved, library)['status'] == 'needs_model_call'
    _stricter_metformin(monkeypatch)
    assert 'changed' in {replay_record(r, library)['status'] for r in recs}


def test_trace_has_one_span_per_node_and_no_case_text(tmp_path, cases, library):
    backend = ScriptedBackend(oracle_responder({(c['patient_summary'], plain_document(c)): c for c in cases}))
    rev = ReviewerV4(backend, library)
    c = next(c for c in cases
             if rev.review(c['patient_summary'], plain_document(c), 'p')['triage']['decision'] == 'escalate')
    trace = tmp_path / 'trace.jsonl'
    graph = ReviewGraphV4(backend, library, tracer=JsonlTracer(trace))
    graph.start(c['patient_summary'], plain_document(c), 'T1')
    graph.resume('T1', {'decision': 'accept', 'reviewer': 'A'})
    spans = [json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
    names = [s['name'] for s in spans]
    for n in ('extract', 'verify_extraction', 'rule_layer', 'select_criteria', 'check', 'aggregate', 'triage',
              'human_review', 'persist'):
        assert f'v4.{n}' in names
    assert {s['trace_id'] for s in spans} == {'T1'}
    assert [s['status'] for s in spans if s['name'] == 'v4.human_review'] == ['interrupted', 'ok']
    text = trace.read_text(encoding='utf-8')
    for fragment in (c['patient_summary'][:40], plain_document(c)[:40]):
        assert fragment not in text                                # no case or document text in traces
    llm = [s for s in spans if s['attributes']['node.kind'] == 'llm']
    assert len(llm) == 2 and all(s['attributes']['llm.status'] == 'Completed' for s in llm)
