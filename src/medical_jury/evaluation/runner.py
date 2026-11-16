"""Crash-safe execution of a planned item list (used by dev, frozen and external runners).

Guarantees:
* every finished record is appended to ``<out>/predictions.partial.jsonl`` immediately (flush + fsync),
  so a crash or Ctrl+C never loses predictions whose calls were already paid for;
* the returned list has EXACTLY one record per planned item, in planned order: items never started
  are recorded as NotRun with the stop reason, so denominators stay complete;
* an unexpected exception in one item becomes a Failed record for that item, not a lost batch;
* a single ``StopController`` decides when to stop and records WHY; once it has stopped, no further
  item starts a call. Nothing is ever retried automatically.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

# Failure codes that stop the run at once (no automatic retry of any kind).
IMMEDIATE_STOP = {
    'PROVIDER_HTTP_401': 'auth_or_quota',   # authentication
    'PROVIDER_HTTP_402': 'auth_or_quota',   # insufficient balance
    'PROVIDER_HTTP_403': 'auth_or_quota',   # permission
    'PROVIDER_HTTP_429': 'rate_limit',      # rate limited: stop, do not back off and retry
    'BUDGET_BLOCKED': 'budget',
    'BUDGET_SETTLEMENT_EXCEEDED': 'budget',
    'CALL_CAP_REACHED': 'call_cap',
}
# When several reasons fire (e.g. concurrent items), the most serious one is reported first.
REASON_PRIORITY = ['budget', 'auth_or_quota', 'rate_limit', 'call_cap', 'interrupted',
                   'consecutive_failures', 'failure_rate_rule']


class StopController:
    """Shared stop decision for one run.

    Rules (all configurable, all recorded):
      * immediate stop on the codes in IMMEDIATE_STOP;
      * stop after ``consecutive_limit`` consecutive non-Completed items (format errors, truncation,
        empty output, transport/5xx...); one Completed item resets the streak;
      * stop if more than ``max_failures`` failures occur within the first ``window`` items;
      * ``interrupt()`` for Ctrl+C.
    ``event`` is shared with the judges so a judge refuses to call once the run has stopped.
    """

    def __init__(self, event: threading.Event | None = None, *, consecutive_limit: int = 3,
                 window: int = 50, max_failures: int = 5):
        self.event = event or threading.Event()
        self.consecutive_limit, self.window, self.max_failures = consecutive_limit, window, max_failures
        self._lock = threading.Lock()
        self.reasons: dict[str, dict] = {}
        self.done = self.failed = self.streak = 0
        self.streak_codes: list[str] = []

    @property
    def stopped(self) -> bool:
        return self.event.is_set()

    @property
    def stop_reason(self) -> str | None:
        for reason in REASON_PRIORITY:
            if reason in self.reasons:
                return reason
        return 'unrecorded_stop' if self.event.is_set() else None

    def trigger(self, reason: str, detail: dict | None = None) -> None:
        with self._lock:
            self.reasons.setdefault(reason, detail or {})
        self.event.set()

    def interrupt(self) -> None:
        self.trigger('interrupted', {})

    def observe(self, rec: dict) -> None:
        if rec.get('status') == 'NotRun':
            return
        code = rec.get('failure_code')
        with self._lock:
            self.done += 1
            if rec.get('status') == 'Completed':
                self.streak, self.streak_codes = 0, []
                return
            self.failed += 1
            self.streak += 1
            self.streak_codes.append(code or rec.get('status', '?'))
            streak, codes, done, failed = self.streak, list(self.streak_codes), self.done, self.failed
        if code in IMMEDIATE_STOP:
            self.trigger(IMMEDIATE_STOP[code], {'failure_code': code, 'after_items': done})
        if streak >= self.consecutive_limit:
            self.trigger('consecutive_failures', {'codes': codes, 'after_items': done})
        if done <= self.window and failed > self.max_failures:
            self.trigger('failure_rate_rule', {'failures': failed, 'after_items': done})

    def not_run_code(self) -> str:
        reason = self.stop_reason
        return f'STOPPED_{reason.upper()}' if reason else 'STOPPED_BEFORE_START'

    def summary(self) -> dict:
        return {'stop_reason': self.stop_reason, 'stop_reasons': dict(self.reasons),
                'budget_stopped': 'budget' in self.reasons, 'items_observed': self.done,
                'items_failed': self.failed,
                'rules': {'immediate_stop_codes': IMMEDIATE_STOP, 'consecutive_limit': self.consecutive_limit,
                          'failure_rate_rule': f'>{self.max_failures} failures within the first {self.window} items',
                          'automatic_retry': False}}


class LiveWriter:
    """Append-only JSONL writer; refuses to reuse an existing file (no silent overwrite)."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open('x', encoding='utf-8')
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n'
        with self._lock:
            self._fh.write(line)
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


def run_planned(items: list, work: Callable[[object], dict], *, id_of: Callable[[object], str],
                writer: LiveWriter, controller: StopController, threads: int = 1) -> tuple[list[dict], bool]:
    """Run ``work`` over ``items``. Returns (records_in_planned_order, interrupted)."""
    results: dict[str, dict] = {}
    interrupted = False

    def guarded(item):
        eid = id_of(item)
        if controller.stopped:
            rec = {'example_id': eid, 'status': 'NotRun', 'failure_code': controller.not_run_code()}
        else:
            try:
                rec = work(item)
            except KeyboardInterrupt:
                controller.interrupt()  # set the stop flag BEFORE the worker can pick up another item
                raise
            except Exception as error:  # one bad item must not lose the batch
                rec = {'example_id': eid, 'status': 'Failed', 'failure_code': 'UNHANDLED_' + type(error).__name__}
            rec['example_id'] = eid
            if rec.get('status') == 'NotRun' and controller.stopped:
                rec['failure_code'] = controller.not_run_code()  # judge refused because the run had stopped
        writer.write(rec)
        controller.observe(rec)
        return rec

    pool = ThreadPoolExecutor(max_workers=max(1, threads))
    futures = {pool.submit(guarded, item): id_of(item) for item in items}
    try:
        for future in as_completed(futures):
            rec = future.result()
            results[rec['example_id']] = rec
    except KeyboardInterrupt:
        interrupted = True
        controller.interrupt()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for future, eid in futures.items():
            if eid not in results and future.done() and not future.cancelled() and future.exception() is None:
                results[eid] = future.result()
    ordered = []
    for item in items:
        eid = id_of(item)
        if eid in results:
            ordered.append(results[eid])
        else:
            rec = {'example_id': eid, 'status': 'NotRun', 'failure_code': controller.not_run_code()}
            writer.write(rec)
            ordered.append(rec)
    return ordered, interrupted


def status_counts(records: Iterable[dict]) -> dict:
    counts: dict[str, int] = {}
    for r in records:
        key = r.get('status', '?') + (':' + r['failure_code'] if r.get('failure_code') else '')
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def ledger_status(root: Path, ceiling_cny: float | None = None) -> dict:
    """Read-only view of the shared cost ledger (no writes, no table creation)."""
    import math
    import sqlite3
    db = root / 'data/jury.sqlite'
    accounted, rows = 0.0, 0
    if db.exists():
        with sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True) as con:
            if con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='costs'").fetchone():
                accounted = float(con.execute('SELECT COALESCE(SUM(accounted),0) FROM costs').fetchone()[0])
                rows = int(con.execute('SELECT COUNT(*) FROM costs').fetchone()[0])
    if ceiling_cny is None:
        from ..judge_v2 import v2_ceiling_from_env
        ceiling_cny = v2_ceiling_from_env(root)
    remaining = max(0.0, ceiling_cny - accounted)
    return {'ledger_accounted_cny': round(accounted, 6), 'ledger_rows': rows, 'ceiling_cny': ceiling_cny,
            'remaining_cny': round(remaining, 6) if math.isfinite(remaining) else None,
            'note': 'Local conservative ledger (shared with legacy runs), not a provider invoice.'}


def budget_gate(plan: dict, paid: bool, allow_partial: bool) -> str | None:
    """Return a refusal message, or None if the run may start."""
    if not paid:
        return None
    if plan['cost_upper_bound_cny'] > plan['run_limit_cny']:
        return f"Upper bound {plan['cost_upper_bound_cny']} exceeds --run-limit {plan['run_limit_cny']}; no calls made."
    remaining = plan['ledger']['remaining_cny']
    if plan['cost_upper_bound_cny'] > remaining and not allow_partial:
        return (f"Upper bound {plan['cost_upper_bound_cny']} exceeds the remaining cumulative budget {remaining}; "
                'no calls made. (--allow-partial would start anyway and stop at the ceiling, recording the rest '
                'as NotRun; use it only with explicit owner approval.)')
    return None
