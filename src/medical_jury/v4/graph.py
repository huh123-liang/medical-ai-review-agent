"""The v4 review chain as a LangGraph state graph (orchestration only).

    START -> route_arm --full--------> extract (LLM call 1)
                                         |-- failed -------------------------------------------> triage
                                         `-- parallel: verify_extraction | rule_layer | select_criteria
                                                       -> join -> check (LLM call 2) -> aggregate -> triage
          route_arm --model_only--> check_only (LLM, comparison arm) ----------------------------> triage
    triage --escalate (and a human is required)--> human_review (interrupt; resumable from the checkpoint)
    triage --otherwise--> persist;  human_review -> persist -> END

Every node calls the same functions as ReviewerV4 (pipeline.py), so the record it persists is identical to
ReviewerV4.review / review_model_only for the same backend replies; the frozen pipeline and its results are unchanged.
A human decision is stored NEXT TO the agent's triage, never over it.

Interruption (append-only, nothing is deleted):
  * cancel(job_id): per job (other jobs in the same process are not affected). Before the run starts -> start() refuses;
    while it runs -> no further model call is sent (the next one is recorded NotRun/CANCELLED), the run ends in persist
    with status 'Cancelled' and no findings; while it waits for a human -> the wait is closed and the run is persisted
    as 'Cancelled'. A finished run cannot be cancelled (use revise for a human decision).
  * recover(job_id): continue a run that stopped mid-way (crash, restart) from its last checkpoint; completed model
    calls are not sent again. With open_durable(path), checkpoints and cancellations live in one SQLite file.
  * revise(job_id, decision): forks from the checkpoint where the run waited for the human and resumes it with the new
    decision; the new decision names the one it supersedes, and the old branch stays in the checkpoint history.
Nothing the agent does needs undoing: it has no tools and writes nothing outside its own record and audit log; money
already spent cannot be refunded, so cost is bounded before a run (budget gate) instead.
"""
from __future__ import annotations

import json
import operator
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..domain import digest
from ..judge_v2 import BackendError
from .check import check_messages, code_checks_of, parse_check
from .criteria import applicable
from .guard import screen_input as run_input_screen
from .verify import POLICIES, VERIFY_MAX_TOKENS, VERIFY_PROMPT_VERSION, SYSTEM_VERIFY, parse_verify, verify_messages
from .extract import check_extraction, extract_messages, parse_extraction, quote_found, unknown_active_drugs
from .pipeline import CHECK_MAX_TOKENS, EXTRACT_MAX_TOKENS, ReviewerV4, merge_findings, triage
from .rules import check as rule_check

GRAPH_VERSION = 'review-v4-graph-3'
DEV_PIPELINE = 'review-v4.2-dev'                    # marks any record made with a development feature switched on
FEATURES = ('input_screen', 'severe_verifier')
ARMS = ('full', 'model_only')
HUMAN_DECISIONS = {'accept', 'override', 'defer'}
TRIAGE_DECISIONS = {'escalate', 'pass_with_findings', 'pass'}
PARALLEL = ('verify_extraction', 'rule_layer', 'select_criteria')


def validate_human(answer) -> dict:
    decision = answer.get('decision') if isinstance(answer, dict) else None
    if decision not in HUMAN_DECISIONS:
        raise ValueError(f'human decision must be one of {sorted(HUMAN_DECISIONS)}')
    if decision == 'override' and answer.get('final_triage') not in TRIAGE_DECISIONS:
        raise ValueError('an override needs final_triage')
    return {'decision': decision, 'final_triage': answer.get('final_triage') if decision == 'override' else None,
            'reviewer': str(answer.get('reviewer') or ''), 'note': str(answer.get('note') or '')}


DEFAULT_CALL_DEADLINE_S = 150.0      # wall-clock cap per model call (httpx's own timeout is per phase, 90 s)


class DeadlineBackend:
    """Wraps a backend: stop WAITING for a call when it passes its wall-clock deadline or when its job is cancelled.

    The request itself keeps running in a daemon thread until the inner backend returns or its own HTTP timeout fires,
    so the budget ledger still settles the real usage; only the graph stops waiting for it."""

    def __init__(self, inner, deadline_s: float, is_cancelled, poll_s: float = 0.1):
        self.inner, self.deadline_s, self.is_cancelled, self.poll_s = inner, deadline_s, is_cancelled, poll_s
        self.name, self.model = getattr(inner, 'name', '?'), getattr(inner, 'model', '?')
        self.billable = getattr(inner, 'billable', True)

    def __getattr__(self, item):                         # close(), settings, ... of the inner backend
        return getattr(self.inner, item)

    def complete(self, messages, max_tokens, job_id):
        box, done = {}, threading.Event()

        def work():
            try:
                box['value'] = self.inner.complete(messages, max_tokens, job_id)
            except BaseException as error:               # re-raised in the caller's thread below
                box['error'] = error
            finally:
                done.set()
        threading.Thread(target=work, name=f'v4-call-{job_id}', daemon=True).start()
        job = job_id.rsplit(':', 1)[0]
        waited = 0.0
        while not done.wait(self.poll_s):
            waited += self.poll_s
            if self.is_cancelled(job):
                raise BackendError('CANCELLED_IN_FLIGHT')
            if waited >= self.deadline_s:
                raise BackendError('CALL_DEADLINE_EXCEEDED')
        if 'error' in box:
            raise box['error']
        return box['value']


NODE_KIND = {'extract': 'llm', 'check': 'llm', 'check_only': 'llm', 'verify_severe': 'llm', 'human_review': 'human'}


class JsonlTracer:
    """One JSON line per node execution, OpenTelemetry-style field names. Never contains case or document text:
    only ids, versions, timings, call status / usage and SHA-256 digests."""

    def __init__(self, path):
        self.path, self._lock = Path(path), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, span: dict) -> None:
        line = json.dumps(span, ensure_ascii=False, sort_keys=True)
        with self._lock, self.path.open('a', encoding='utf-8') as f:
            f.write(line + '\n')


def _span_attributes(name: str, out) -> dict:
    kind = NODE_KIND.get(name, 'code')
    if kind == 'llm' and isinstance(out, dict) and not out.get('calls'):
        kind = 'noop'                                     # e.g. verify_severe switched off: no call was made
    a = {'node.kind': kind, 'output.sha256': digest(out) if isinstance(out, dict) else None}
    if not isinstance(out, dict):
        return a
    for call in out.get('calls', []):
        a.update({'llm.step': call.get('step'), 'llm.status': call.get('status'),
                  'llm.failure_code': call.get('failure_code'), 'llm.model': call.get('model'),
                  'llm.latency_s': call.get('latency_s'), 'llm.usage': call.get('usage'),
                  'llm.raw_sha256': call.get('raw_sha256')})
    if 'findings' in out:
        a.update({'findings.count': len(out['findings']),
                  'findings.severe': sum(f.get('severity') == 'severe' for f in out['findings'])})
    if out.get('triage_result') is not None and 'triage_result' in out:
        a.update({'triage.decision': out['triage_result']['decision'],
                  'triage.reasons': out['triage_result']['reasons']})
    if out.get('cancellation'):
        a['cancelled'] = True
    return a


class MemoryCancellations:
    """Cancellation register for one process (tests, notebooks)."""

    def __init__(self):
        self._lock, self._d = threading.Lock(), {}

    def get(self, job_id):
        with self._lock:
            return self._d.get(job_id)

    def put(self, job_id, record: dict) -> dict:
        """First cancellation wins; returns the stored one."""
        with self._lock:
            return self._d.setdefault(job_id, record)


class SqliteCancellations:
    """Cancellation register shared by every process that opens the same file; survives restarts."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn, self._lock = conn, threading.Lock()
        with self._lock, self.conn:
            self.conn.execute('CREATE TABLE IF NOT EXISTS v4_cancellations (job_id TEXT PRIMARY KEY, record TEXT NOT NULL)')

    def get(self, job_id):
        with self._lock:
            row = self.conn.execute('SELECT record FROM v4_cancellations WHERE job_id = ?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, job_id, record: dict) -> dict:
        with self._lock, self.conn:
            self.conn.execute('INSERT OR IGNORE INTO v4_cancellations VALUES (?, ?)', (job_id, json.dumps(record)))
        return self.get(job_id)


def open_durable(path) -> dict:
    """Checkpoints and cancellations in one SQLite file: paused, crashed or cancelled jobs survive a restart.
    Pass the result as ReviewGraphV4(..., **open_durable(path)); close it with ['checkpointer'].conn.close()."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute('PRAGMA journal_mode=WAL')
    return {'checkpointer': SqliteSaver(conn), 'cancellations': SqliteCancellations(conn)}


def _first(a, b):
    """Reducer: the first cancellation recorded wins (parallel branches may both see it)."""
    return a or b


class ReviewStateV4(TypedDict, total=False):
    job_id: str
    arm: str
    case_description: str
    document: str
    calls: Annotated[list[dict], operator.add]     # reducer: each model call appends its audit record
    extraction: dict | None
    problems: list[dict]
    unknown: list[str]
    rule_out: list[dict]
    items: list[dict]
    code_findings: list[dict]
    check_result: dict | None
    failed: list[str]
    findings: list[dict]
    uncertain: list[dict]
    triage_result: dict
    human_decision: dict
    cancellation: Annotated[dict | None, _first]
    input_screen: dict
    record: dict


class ReviewGraphV4:
    def __init__(self, backend, library: dict, store=None, *, require_human: bool = True, checkpointer=None,
                 stop: threading.Event | None = None, call_deadline_s: float | None = DEFAULT_CALL_DEADLINE_S,
                 cancellations=None, tracer=None, features=(), verifier_backend=None,
                 verifier_policy: str = 'flag'):
        unknown = set(features) - set(FEATURES)
        if unknown:
            raise ValueError(f'unknown features {sorted(unknown)}; known: {FEATURES}')
        if 'severe_verifier' in features and verifier_backend is None:
            raise ValueError('severe_verifier needs a verifier_backend (ideally another model family)')
        if verifier_policy not in POLICIES:
            raise ValueError(f'verifier_policy must be one of {POLICIES}')
        self.features, self.verifier_policy = frozenset(features), verifier_policy
        if call_deadline_s is not None:
            backend = DeadlineBackend(backend, call_deadline_s, lambda job: self._cancel_of(job) is not None)
        self.reviewer = ReviewerV4(backend, library, store, **({'stop': stop} if stop is not None else {}))
        self.verifier = None
        if verifier_backend is not None:
            if call_deadline_s is not None:
                verifier_backend = DeadlineBackend(verifier_backend, call_deadline_s,
                                                   lambda job: self._cancel_of(job) is not None)
            self.verifier = ReviewerV4(verifier_backend, library, store, stop=self.reviewer.stop)
        self.library, self.store, self.require_human = library, store, require_human
        self._cancel_lock, self._job_locks = threading.Lock(), {}
        self.cancellations = cancellations if cancellations is not None else MemoryCancellations()
        self.tracer = tracer
        g = StateGraph(ReviewStateV4)
        for name in ('screen_input', 'extract', *PARALLEL, 'check', 'check_only', 'aggregate', 'verify_severe', 'triage',
                     'human_review', 'persist'):
            g.add_node(name, self._traced(name, getattr(self, name)))
        g.add_edge(START, 'screen_input')
        g.add_conditional_edges('screen_input', self.route_arm, {'full': 'extract', 'model_only': 'check_only'})
        g.add_conditional_edges('extract', self.route_after_extract, [*PARALLEL, 'triage'])
        g.add_edge(list(PARALLEL), 'check')                 # join: check waits for all three code branches
        g.add_edge('check', 'aggregate')
        g.add_edge('aggregate', 'verify_severe')
        g.add_edge('verify_severe', 'triage')
        g.add_edge('check_only', 'triage')
        g.add_conditional_edges('triage', self.route_after_triage, ['human_review', 'persist'])
        g.add_edge('human_review', 'persist')
        g.add_edge('persist', END)
        if checkpointer is None:                          # cancel / revise / restart need a checkpoint per job
            checkpointer = InMemorySaver()
        self.graph = g.compile(checkpointer=checkpointer)

    def _traced(self, name, fn):
        if self.tracer is None:
            return fn

        def node(state):
            start, t0, status, err, out = datetime.now(timezone.utc).isoformat(), time.monotonic(), 'ok', None, None
            try:
                out = fn(state)
                return out
            except GraphInterrupt:
                status = 'interrupted'                      # waiting for a human: not an error
                raise
            except Exception as e:
                status, err = 'error', type(e).__name__
                raise
            finally:
                self.tracer.emit({'trace_id': state.get('job_id'), 'span_id': uuid.uuid4().hex[:16],
                                  'name': f'v4.{name}', 'start_time': start,
                                  'duration_ms': round((time.monotonic() - t0) * 1000, 2), 'status': status,
                                  'error.type': err, 'graph.version': GRAPH_VERSION, 'arm': state.get('arm'),
                                  'attributes': _span_attributes(name, out)})
        node.__name__ = name
        return node

    # ---- routing -------------------------------------------------------------------------------
    @staticmethod
    def route_arm(state) -> str:
        if state.get('arm', 'full') not in ARMS:
            raise ValueError(f"unknown arm {state.get('arm')!r}")
        return state.get('arm', 'full')

    @staticmethod
    def route_after_extract(state):
        if state.get('cancellation') or state.get('extraction') is None:
            return 'triage'
        return list(PARALLEL)

    def route_after_triage(self, state) -> str:
        if state.get('cancellation') or state.get('triage_result') is None:
            return 'persist'
        return 'human_review' if self.require_human and state['triage_result']['decision'] == 'escalate' else 'persist'

    # ---- cancellation helpers --------------------------------------------------------------------
    def _job_lock(self, job_id) -> threading.Lock:
        """One lock per job: resume / cancel / revise of the same job never interleave."""
        with self._cancel_lock:
            return self._job_locks.setdefault(job_id, threading.Lock())

    def _cancel_of(self, job_id):
        return self.cancellations.get(job_id)

    def _model_call(self, step, messages, max_tokens, job_id, parse, reviewer=None):
        """Stop issuing new operations: a cancelled job sends no further model call."""
        c = self._cancel_of(job_id)
        if c is not None:
            backend = self.reviewer.backend
            return ({'step': step, 'messages_sha256': digest(messages), 'judge': getattr(backend, 'name', '?'),
                     'model': getattr(backend, 'model', '?'), 'status': 'NotRun', 'failure_code': 'CANCELLED'},
                    None, c)
        call, parsed = (reviewer or self.reviewer)._call(step, messages, max_tokens, job_id, parse)
        return call, parsed, self._cancel_of(job_id)       # a cancel that arrived during the call also counts

    # ---- nodes: development features (no-ops unless switched on) --------------------------------
    def screen_input(self, state):
        if 'input_screen' not in self.features:
            return {}
        return {'input_screen': run_input_screen(state['case_description'], state['document'])}

    def verify_severe(self, state):
        if 'severe_verifier' not in self.features or self._cancel_of(state['job_id']) is not None:
            return {}
        findings, calls = [dict(f) for f in state['findings']], []
        for i, f in enumerate(findings):
            if f.get('source') != 'open_review' or f.get('severity') != 'severe':
                continue
            call, res, c = self._model_call(f'verify_{i}', verify_messages(state['case_description'], state['document'],
                                                                           f), VERIFY_MAX_TOKENS, state['job_id'],
                                            parse_verify, reviewer=self.verifier)
            calls.append(call)
            if c is not None:
                return {'calls': calls, 'findings': findings, 'cancellation': c}
            f['verification'] = {'prompt': VERIFY_PROMPT_VERSION, 'model': call.get('model'),
                                 'agree': None if res is None else res['agree'],
                                 'reason': '' if res is None else res['reason'],
                                 'failure_code': call.get('failure_code')}
            if res is not None and not res['agree'] and self.verifier_policy == 'downgrade':
                f.update(severity='moderate', severity_before_verifier='severe')
        return {'calls': calls, 'findings': findings}

    # ---- nodes: model calls ----------------------------------------------------------------------
    def extract(self, state):
        call, ex, c = self._model_call('extract', extract_messages(state['case_description'], state['document'],
                                                                  self.reviewer.extract_system),
                                       EXTRACT_MAX_TOKENS, state['job_id'], parse_extraction)
        out = {'calls': [call], 'extraction': ex}
        if ex is None:
            out.update(findings=[], uncertain=[], problems=[], unknown=[], failed=[call['failure_code']])
        if c is not None:
            out['cancellation'] = c
        return out

    def check(self, state):
        ex, items = state['extraction'], state['items']
        call, chk, c = self._model_call('check', check_messages(state['case_description'], state['document'], items,
                                                                self.reviewer.check_system,
                                                                code_checks_of(state['rule_out'], ex['medications'])),
                                        CHECK_MAX_TOKENS, state['job_id'], lambda raw: parse_check(raw, items))
        out = {'calls': [call], 'check_result': chk, 'failed': [] if chk is not None else [call['failure_code']]}
        if c is not None:
            out['cancellation'] = c
        return out

    def check_only(self, state):
        call, chk, c = self._model_call('check_only', check_messages(state['case_description'], state['document'], [],
                                                                     self.reviewer.check_system),
                                        CHECK_MAX_TOKENS, state['job_id'], lambda raw: parse_check(raw, []))
        findings = [] if chk is None else \
            [{**f, 'quote_verified': quote_found(f['quote'], state['document']) if f['quote'] else None}
             for f in chk['other_findings']]
        out = {'calls': [call], 'check_result': chk, 'findings': findings, 'uncertain': [], 'problems': [],
               'unknown': [], 'failed': [] if chk is not None else [call['failure_code']]}
        if c is not None:
            out['cancellation'] = c
        return out

    # ---- nodes: deterministic code ---------------------------------------------------------------
    def verify_extraction(self, state):
        ex = state['extraction']
        return {'problems': check_extraction(ex, state['case_description'], state['document']),
                'unknown': unknown_active_drugs(ex)}

    def rule_layer(self, state):
        ex = state['extraction']
        return {'rule_out': rule_check(ex['facts'], ex['medications'])}

    def select_criteria(self, state):
        ex = state['extraction']
        items, code_findings = applicable(self.library, ex['facts'], ex['medications'])
        return {'items': items, 'code_findings': code_findings}

    def aggregate(self, state):
        chk = state['check_result']
        opened = [] if chk is None else \
            [{**f, 'quote_verified': quote_found(f['quote'], state['document']) if f['quote'] else None}
             for f in chk['other_findings']]
        findings, uncertain = merge_findings(state['rule_out'], state['code_findings'], state['items'], chk, opened,
                                             code_checks_of(state['rule_out'], state['extraction']['medications']))
        return {'findings': findings, 'uncertain': uncertain}

    def triage(self, state):
        c = state.get('cancellation') or self._cancel_of(state['job_id'])
        if c is not None:                                 # never present a partial review as a triage decision
            return {'triage_result': None, 'cancellation': c}
        t = triage(state.get('findings', []), state.get('uncertain', []), state.get('unknown', []),
                   state.get('problems', []), state.get('failed', []))
        if (state.get('input_screen') or {}).get('flagged'):    # the model may have been steered: a human reads it
            t = {'decision': 'escalate', 'reasons': t['reasons'] + ['input_flagged']}
        if self.verifier_policy == 'flag' and \
                any((f.get('verification') or {}).get('agree') is False for f in state.get('findings', [])):
            t = {'decision': 'escalate', 'reasons': t['reasons'] + ['verifier_disagreed']}
        return {'triage_result': t}

    # ---- node: human in the loop -----------------------------------------------------------------
    def human_review(self, state):
        answer = interrupt({'job_id': state['job_id'], 'triage': state['triage_result'],
                            'findings': [{k: f.get(k) for k in ('source', 'id', 'category', 'severity', 'entity', 'issue')}
                                         for f in state.get('findings', [])],
                            'uncertain': [{k: f.get(k) for k in ('source', 'id', 'entity', 'issue')}
                                          for f in state.get('uncertain', [])],
                            'instructions': "Reply {'decision': 'accept'|'override'|'defer', 'final_triage': "
                                            "'escalate'|'pass_with_findings'|'pass' (override only), 'reviewer': str, "
                                            "'note': str}"})
        human = validate_human(answer)                    # only the known fields; a revision chain cannot be forged
        if self.store is not None:
            self.store.audit(state['job_id'], 'v4_human_review', human)
        return {'human_decision': human}

    def persist(self, state):
        arm = state.get('arm', 'full')
        rec = {**self.reviewer._header(arm), 'calls': state['calls']}
        if state.get('cancellation'):
            rec.update(status='Cancelled', cancellation=state['cancellation'], triage=state.get('triage_result'))
            if self.store is not None:
                self.store.audit(state['job_id'], 'v4_cancelled', state['cancellation'])
            return {'record': rec}
        if arm == 'model_only':
            rec.update(findings=state['findings'], uncertain=[], triage=state['triage_result'])
        elif state.get('extraction') is None:
            rec.update(extraction=None, findings=[], uncertain=[], triage=state['triage_result'])
        else:
            ex, chk = state['extraction'], state['check_result']
            rec.update(extraction={**ex, 'problems': state['problems'], 'unknown_active_drugs': state['unknown']},
                       items=[{'id': i['id'], 'level': i['level'], 'text': i['text']} for i in state['items']],
                       criteria_results=chk['criteria'] if chk else None,
                       findings=state['findings'], uncertain=state['uncertain'], triage=state['triage_result'])
        if self.features or self.library['disease'].get('version', '').endswith('-dev'):
            rec['dev'] = {'pipeline': DEV_PIPELINE, 'features': sorted(self.features),
                          'disease_library': self.library['disease'].get('version'),
                          'verifier': {'prompt': VERIFY_PROMPT_VERSION, 'sha256': digest(SYSTEM_VERIFY),
                                       'policy': self.verifier_policy} if 'severe_verifier' in self.features else None,
                          'note': 'development configuration: not evaluated, not comparable with v4.1 results'}
            if 'input_screen' in state:
                rec['input_screen'] = state['input_screen']
        if 'human_decision' in state:
            rec['human_review'] = state['human_decision']     # next to the agent's triage, never over it
        return {'record': rec}

    # ---- driver ----------------------------------------------------------------------------------
    def start(self, case_description: str, document: str, job_id: str, arm: str = 'full') -> dict:
        if self._started(job_id):
            raise ValueError(f'job {job_id!r} was already started; a job id is used once')
        if self._cancel_of(job_id) is not None:
            return {'status': 'Cancelled', 'graph': GRAPH_VERSION, 'cancellation': self._cancel_of(job_id)}
        payload = {'job_id': job_id, 'arm': arm, 'case_description': case_description, 'document': document}
        return self._result(self.graph.invoke(payload, self._config(job_id)))

    def resume(self, job_id: str, decision: dict) -> dict:
        clean = validate_human(decision)                   # refuse bad input before it reaches the checkpoint
        with self._job_lock(job_id):
            if self.status(job_id) != 'AwaitingHumanReview':
                raise ValueError(f'job {job_id!r} is not waiting for a human ({self.status(job_id)})')
            return self._result(self.graph.invoke(Command(resume=clean), self._config(job_id)))

    def status(self, job_id: str) -> str:
        snap = self.graph.get_state(self._config(job_id))
        if not snap.values:
            return 'Cancelled' if self._cancel_of(job_id) is not None else 'NotStarted'
        if snap.values.get('record'):
            return snap.values['record'].get('status', 'Completed')
        if snap.next == ('human_review',):
            return 'AwaitingHumanReview'
        return 'Running'

    def cancel(self, job_id: str, reason: str, by: str = '') -> dict:
        """Stop issuing new operations for this job, then close it. Returns the cancellation record."""
        with self._job_lock(job_id):
            return self._cancel_locked(job_id, reason, by)

    def _cancel_locked(self, job_id, reason, by):
        state = self.status(job_id)
        if state in ('Completed', 'Cancelled') and self._started(job_id):
            raise ValueError(f'job {job_id!r} is already {state.lower()}; a finished run is not cancelled')
        c = {'reason': str(reason), 'by': str(by), 'at': datetime.now(timezone.utc).isoformat(), 'state_when_cancelled': state}
        c = self.cancellations.put(job_id, c)
        if self.store is not None:
            self.store.audit(job_id, 'v4_cancel_requested', c)
        if state == 'AwaitingHumanReview':                 # close the wait, then persist as cancelled
            self.graph.update_state(self._config(job_id), {'cancellation': c}, as_node='human_review')
            self.graph.invoke(None, self._config(job_id))
        return c

    def revise(self, job_id: str, decision: dict) -> dict:
        """Replace the human decision of a finished run by forking from the paused checkpoint (old one kept)."""
        clean = validate_human(decision)
        with self._job_lock(job_id):
            return self._revise_locked(job_id, clean)

    def _revise_locked(self, job_id, clean):
        snap = self.graph.get_state(self._config(job_id))
        rec = (snap.values or {}).get('record') or {}
        if rec.get('status') == 'Cancelled' or 'human_review' not in rec:
            raise ValueError(f'job {job_id!r} has no human decision to revise')
        paused = next((h for h in self.graph.get_state_history(self._config(job_id)) if h.next == ('human_review',)),
                      None)
        if paused is None:
            raise ValueError(f'no paused checkpoint for job {job_id!r}')
        prev = rec['human_review']
        human = {**clean, 'supersedes': prev, 'revision': prev.get('revision', 0) + 1}
        if self.store is not None:
            self.store.audit(job_id, 'v4_human_revised', human)
        # Fork: a new checkpoint after the paused one, as if human_review had returned the new decision.
        # (Resuming the old interrupt again would replay the stored first answer.)
        fork = self.graph.update_state(paused.config, {'human_decision': human}, as_node='human_review')
        return self._result(self.graph.invoke(None, fork))

    def recover(self, job_id: str) -> dict:
        """Continue a run that stopped mid-way (crash, restart) from its last checkpoint. Model calls that already
        completed are not sent again; a cancelled job ends as Cancelled without any further model call."""
        with self._job_lock(job_id):
            state = self.status(job_id)
            if state != 'Running':
                raise ValueError(f'job {job_id!r} has nothing to recover ({state})')
            if self.store is not None:
                self.store.audit(job_id, 'v4_recovered', {'from_checkpoint': self.graph.get_state(
                    self._config(job_id)).config['configurable'].get('checkpoint_id')})
            return self._result(self.graph.invoke(None, self._config(job_id)))

    def _started(self, job_id) -> bool:
        return bool(self.graph.get_state(self._config(job_id)).values)

    @staticmethod
    def _config(job_id):
        return {'configurable': {'thread_id': job_id}}

    @staticmethod
    def _result(out: dict) -> dict:
        if out.get('__interrupt__'):
            return {'status': 'AwaitingHumanReview', 'graph': GRAPH_VERSION, 'review': out['__interrupt__'][0].value}
        rec = out['record']
        return {'status': rec.get('status', 'Completed'), 'graph': GRAPH_VERSION, 'record': rec}
