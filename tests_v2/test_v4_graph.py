"""v4 LangGraph orchestration: offline tests only (no network, no paid call).

The graph must persist exactly the record ReviewerV4 produces, so frozen v4 results stay valid under either driver."""
from __future__ import annotations

from pathlib import Path

import pytest

from medical_jury.judge_v2 import BackendError, ScriptedBackend
from medical_jury.testbed.build import build_cases
from medical_jury.v4 import criteria as CR
from medical_jury.v4.graph import GRAPH_VERSION, ReviewGraphV4
from medical_jury.v4.mocks import oracle_responder, plain_document
from medical_jury.v4.pipeline import ReviewerV4

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261112


@pytest.fixture(scope='module')
def cases():
    return [c for c in build_cases(150, SEED)[0] if c['split'] == 'test']


@pytest.fixture(scope='module')
def library():
    return CR.load_library(ROOT)


def _timeless(rec):
    """Wall-clock latency differs between any two runs; everything else must match exactly."""
    return {**rec, 'calls': [{k: v for k, v in c.items() if k != 'latency_s'} for c in rec['calls']]}


def _backend(cases):
    return ScriptedBackend(oracle_responder({(c['patient_summary'], plain_document(c)): c for c in cases}))


def test_topology_matches_the_documented_graph(library):
    g = ReviewGraphV4(ScriptedBackend(lambda m: ''), library, require_human=False)
    edges = {(e.source, e.target, e.conditional) for e in g.graph.get_graph().edges}
    assert edges == {
        ('__start__', 'screen_input', False),
        ('screen_input', 'extract', True), ('screen_input', 'check_only', True),
        ('extract', 'verify_extraction', True), ('extract', 'rule_layer', True),
        ('extract', 'select_criteria', True), ('extract', 'triage', True),
        ('verify_extraction', 'check', False), ('rule_layer', 'check', False), ('select_criteria', 'check', False),
        ('check', 'aggregate', False), ('aggregate', 'verify_severe', False), ('verify_severe', 'triage', False),
        ('check_only', 'triage', False),
        ('triage', 'human_review', True), ('triage', 'persist', True),
        ('human_review', 'persist', False), ('persist', '__end__', False)}


@pytest.mark.parametrize('arm', ['full', 'model_only'])
def test_graph_record_equals_reviewer_record(cases, library, arm):
    rev = ReviewerV4(_backend(cases), library)
    graph = ReviewGraphV4(_backend(cases), library, require_human=False)
    for c in cases:
        doc = plain_document(c)
        want = rev.review(c['patient_summary'], doc, c['case_id']) if arm == 'full' else \
            rev.review_model_only(c['patient_summary'], doc, c['case_id'])
        got = graph.start(c['patient_summary'], doc, c['case_id'], arm=arm)
        assert got['status'] == 'Completed' and got['graph'] == GRAPH_VERSION
        assert _timeless(got['record']) == _timeless(want), c['case_id']


def test_failed_extraction_skips_the_parallel_branches(library):
    class Failing(ScriptedBackend):
        def complete(self, messages, max_tokens, job_id):
            raise BackendError('HTTP_402')
    graph = ReviewGraphV4(Failing(lambda m: ''), library, require_human=False)
    want = ReviewerV4(Failing(lambda m: ''), library).review('Age 60; eGFR 40.', 'Take amlodipine 5 mg.', 'J')
    got = graph.start('Age 60; eGFR 40.', 'Take amlodipine 5 mg.', 'J')['record']
    assert _timeless(got) == _timeless(want) and len(got['calls']) == 1 and got['triage']['decision'] == 'escalate'
    assert 'items' not in got and got['extraction'] is None


def test_escalated_case_waits_for_a_human_and_keeps_the_agent_triage(cases, library):
    rev = ReviewerV4(_backend(cases), library)
    case = next(c for c in cases if rev.review(c['patient_summary'], plain_document(c), 'probe')['triage']['decision']
                == 'escalate')
    graph = ReviewGraphV4(_backend(cases), library, require_human=True)
    out = graph.start(case['patient_summary'], plain_document(case), 'J1')
    assert out['status'] == 'AwaitingHumanReview' and out['review']['triage']['decision'] == 'escalate'
    with pytest.raises(ValueError):
        graph.resume('J1', {'decision': 'override'})            # an override must name the final triage
    done = graph.resume('J1', {'decision': 'override', 'final_triage': 'pass_with_findings', 'reviewer': 'R'})
    rec = done['record']
    assert done['status'] == 'Completed' and rec['triage']['decision'] == 'escalate'
    assert rec['human_review'] == {'decision': 'override', 'final_triage': 'pass_with_findings', 'reviewer': 'R',
                                   'note': ''}


def test_passed_case_never_interrupts(cases, library):
    rev = ReviewerV4(_backend(cases), library)
    case = next(c for c in cases if rev.review(c['patient_summary'], plain_document(c), 'probe')['triage']['decision']
                != 'escalate')
    out = ReviewGraphV4(_backend(cases), library, require_human=True).start(case['patient_summary'],
                                                                           plain_document(case), 'J2')
    assert out['status'] == 'Completed' and 'human_review' not in out['record']


def test_unknown_arm_is_refused(library):
    with pytest.raises(ValueError):
        ReviewGraphV4(ScriptedBackend(lambda m: ''), library, require_human=False).start('x', 'y', 'J3', arm='both')


# ---------------------------------------------------------------- interruption: cancel and revise

def _escalated(cases, library):
    rev = ReviewerV4(_backend(cases), library)
    return next(c for c in cases
                if rev.review(c['patient_summary'], plain_document(c), 'probe')['triage']['decision'] == 'escalate')


def test_cancel_before_start_sends_nothing(cases, library):
    backend = _backend(cases)
    graph = ReviewGraphV4(backend, library)
    graph.cancel('C0', 'duplicate request', by='ops')
    out = graph.start(cases[0]['patient_summary'], plain_document(cases[0]), 'C0')
    assert out['status'] == 'Cancelled' and backend.calls == 0 and graph.status('C0') == 'Cancelled'


def test_cancel_while_running_stops_new_model_calls(cases, library):
    """The cancel arrives while the extraction call is in flight: the check call is never sent."""
    holder = {}
    inner = _backend(cases)

    class CancelDuringExtract(ScriptedBackend):
        def complete(self, messages, max_tokens, job_id):
            if self.calls == 0:
                holder['graph'].cancel('C1', 'clinician withdrew the case')
            self.calls += 1
            return inner.complete(messages, max_tokens, job_id)
    backend = CancelDuringExtract(lambda m: '')
    holder['graph'] = graph = ReviewGraphV4(backend, library)
    c = _escalated(cases, library)
    out = graph.start(c['patient_summary'], plain_document(c), 'C1')
    rec = out['record']
    assert out['status'] == 'Cancelled' and backend.calls == 1
    assert [x['status'] for x in rec['calls']] == ['Completed']      # nothing after the cancel, not even code nodes
    assert rec['triage'] is None and 'findings' not in rec and 'human_review' not in rec
    assert rec['cancellation']['reason'] == 'clinician withdrew the case'


def test_cancel_while_waiting_for_a_human_closes_the_wait(cases, library):
    graph = ReviewGraphV4(_backend(cases), library)
    c = _escalated(cases, library)
    assert graph.start(c['patient_summary'], plain_document(c), 'C2')['status'] == 'AwaitingHumanReview'
    graph.cancel('C2', 'patient transferred')
    assert graph.status('C2') == 'Cancelled'
    rec = graph.graph.get_state(graph._config('C2')).values['record']
    assert rec['triage']['decision'] == 'escalate'               # the agent's own output is kept, marked cancelled
    assert rec['cancellation']['state_when_cancelled'] == 'AwaitingHumanReview'
    with pytest.raises(ValueError):
        graph.resume('C2', {'decision': 'accept'})
    with pytest.raises(ValueError):
        graph.cancel('C2', 'again')


def test_finished_or_reused_jobs_are_refused(cases, library):
    graph = ReviewGraphV4(_backend(cases), library, require_human=False)
    c = cases[0]
    graph.start(c['patient_summary'], plain_document(c), 'C3')
    with pytest.raises(ValueError):
        graph.cancel('C3', 'too late')
    with pytest.raises(ValueError):
        graph.start(c['patient_summary'], plain_document(c), 'C3')


def test_cancel_affects_only_its_own_job(cases, library):
    graph = ReviewGraphV4(_backend(cases), library, require_human=False)
    graph.cancel('other', 'x')
    out = graph.start(cases[0]['patient_summary'], plain_document(cases[0]), 'C4')
    assert out['status'] == 'Completed'


def test_revise_keeps_the_superseded_decision(cases, library):
    graph = ReviewGraphV4(_backend(cases), library)
    c = _escalated(cases, library)
    graph.start(c['patient_summary'], plain_document(c), 'R1')
    first = graph.resume('R1', {'decision': 'accept', 'reviewer': 'A',
                                'supersedes': {'forged': True}})['record']['human_review']
    assert 'supersedes' not in first                              # resume() cannot forge a revision chain
    second = graph.revise('R1', {'decision': 'override', 'final_triage': 'pass', 'reviewer': 'B',
                                 'note': 'mis-click'})['record']
    assert second['triage']['decision'] == 'escalate'             # agent triage never changes
    assert second['human_review']['revision'] == 1 and second['human_review']['supersedes'] == first
    third = graph.revise('R1', {'decision': 'defer', 'reviewer': 'C'})['record']['human_review']
    assert third['revision'] == 2 and third['supersedes']['reviewer'] == 'B'
    decisions = [h.values['record']['human_review']['reviewer'] for h in graph.graph.get_state_history(graph._config('R1'))
                 if h.values.get('record')]
    assert sorted(decisions) == ['A', 'B', 'C']                   # every branch is still in the history


def test_revise_needs_a_human_decision(cases, library):
    graph = ReviewGraphV4(_backend(cases), library, require_human=False)
    graph.start(cases[0]['patient_summary'], plain_document(cases[0]), 'R2')
    with pytest.raises(ValueError):
        graph.revise('R2', {'decision': 'accept'})


def test_concurrent_resumes_record_exactly_one_decision(cases, library):
    import threading
    graph = ReviewGraphV4(_backend(cases), library)
    c = _escalated(cases, library)
    graph.start(c['patient_summary'], plain_document(c), 'R3')
    results, errors = [], []

    def go(who):
        try:
            results.append(graph.resume('R3', {'decision': 'accept', 'reviewer': who}))
        except ValueError as e:
            errors.append(e)
    ts = [threading.Thread(target=go, args=(w,)) for w in ('A', 'B', 'C', 'D')]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(results) == 1 and len(errors) == 3


# ---------------------------------------------------------------- call deadline

class _Hanging(ScriptedBackend):
    """Never answers until released (a stalled connection)."""
    def __init__(self):
        super().__init__(lambda m: '')
        self.release = __import__('threading').Event()

    def complete(self, messages, max_tokens, job_id):
        self.calls += 1
        self.release.wait(10)
        raise BackendError('PROVIDER_TIMEOUT')


def test_a_stalled_call_fails_at_its_deadline(library):
    import time
    backend = _Hanging()
    graph = ReviewGraphV4(backend, library, require_human=False, call_deadline_s=0.3)
    t0 = time.monotonic()
    rec = graph.start('Age 60; eGFR 40.', 'Take amlodipine 5 mg.', 'D1')['record']
    assert time.monotonic() - t0 < 3
    assert rec['calls'][0]['failure_code'] == 'CALL_DEADLINE_EXCEEDED' and rec['triage']['decision'] == 'escalate'
    backend.release.set()


def test_cancel_stops_waiting_for_an_in_flight_call(library):
    import threading
    import time
    backend = _Hanging()
    graph = ReviewGraphV4(backend, library, require_human=False, call_deadline_s=60)
    threading.Timer(0.3, lambda: graph.cancel('D2', 'withdrawn')).start()
    t0 = time.monotonic()
    out = graph.start('Age 60; eGFR 40.', 'Take amlodipine 5 mg.', 'D2')
    assert time.monotonic() - t0 < 3 and out['status'] == 'Cancelled'
    assert out['record']['calls'][0]['failure_code'] == 'CANCELLED_IN_FLIGHT' and backend.calls == 1
    backend.release.set()


def test_deadline_wrapper_keeps_backend_identity(cases, library):
    rec = ReviewGraphV4(_backend(cases), library, require_human=False).start(
        cases[0]['patient_summary'], plain_document(cases[0]), 'D3')['record']
    assert {c['judge'] for c in rec['calls']} == {'scripted'}
    assert {c['model'] for c in rec['calls']} == {'OFFLINE-SCRIPTED-NOT-CLINICAL'}


# ---------------------------------------------------------------- durable state across restarts

def _durable(tmp_path, backend, library, **kw):
    from medical_jury.v4.graph import open_durable
    d = open_durable(tmp_path / 'v4.sqlite')
    return ReviewGraphV4(backend, library, **d, **kw), d['checkpointer'].conn


def test_paused_review_survives_a_restart(tmp_path, cases, library):
    c = _escalated(cases, library)
    g1, conn = _durable(tmp_path, _backend(cases), library)
    assert g1.start(c['patient_summary'], plain_document(c), 'P1')['status'] == 'AwaitingHumanReview'
    conn.close()
    del g1
    backend = _backend(cases)
    g2, conn = _durable(tmp_path, backend, library)
    assert g2.status('P1') == 'AwaitingHumanReview'
    out = g2.resume('P1', {'decision': 'accept', 'reviewer': 'A'})
    assert out['status'] == 'Completed' and backend.calls == 0       # no model call repeated after the restart
    conn.close()


def test_cancellation_survives_a_restart(tmp_path, cases, library):
    g1, conn = _durable(tmp_path, _backend(cases), library)
    g1.cancel('P2', 'withdrawn')
    conn.close()
    backend = _backend(cases)
    g2, conn = _durable(tmp_path, backend, library)
    assert g2.start(cases[0]['patient_summary'], plain_document(cases[0]), 'P2')['status'] == 'Cancelled'
    assert backend.calls == 0
    conn.close()


class _CrashOnCheck(ScriptedBackend):
    def __init__(self, inner):
        super().__init__(lambda m: '')
        self.inner = inner

    def complete(self, messages, max_tokens, job_id):
        self.calls += 1
        if job_id.endswith(':check'):
            raise RuntimeError('process killed')               # not a BackendError: the run itself dies
        return self.inner.complete(messages, max_tokens, job_id)


def test_crashed_run_recovers_without_repeating_finished_calls(tmp_path, cases, library):
    c = cases[0]
    want = ReviewerV4(_backend(cases), library).review(c['patient_summary'], plain_document(c), 'P3')
    g1, conn = _durable(tmp_path, _CrashOnCheck(_backend(cases)), library, require_human=False)
    with pytest.raises(RuntimeError):
        g1.start(c['patient_summary'], plain_document(c), 'P3')
    assert g1.status('P3') == 'Running'
    conn.close()
    backend = _backend(cases)
    g2, conn = _durable(tmp_path, backend, library, require_human=False)
    got = g2.recover('P3')['record']
    assert backend.calls == 1                                      # only the check call; extraction came from the checkpoint
    assert _timeless(got) == _timeless(want)
    with pytest.raises(ValueError):
        g2.recover('P3')
    conn.close()


def test_cancelled_stale_run_ends_without_new_calls(tmp_path, cases, library):
    c = cases[0]
    g1, conn = _durable(tmp_path, _CrashOnCheck(_backend(cases)), library, require_human=False)
    with pytest.raises(RuntimeError):
        g1.start(c['patient_summary'], plain_document(c), 'P4')
    conn.close()
    backend = _backend(cases)
    g2, conn = _durable(tmp_path, backend, library, require_human=False)
    g2.cancel('P4', 'abandon after crash')
    out = g2.recover('P4')
    assert out['status'] == 'Cancelled' and backend.calls == 0
    conn.close()
