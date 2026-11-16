"""Structured data-use approval record for paid runs on non-public data (decision 2026-10-26).

Sending MIMIC or hospital data to an external model provider is allowed by the owner when an approval exists.
The run must say exactly which approval it relies on, so that the record answers later questions on its own:

    --data-approval-ref "approver=<name or body>;date=YYYY-MM-DD;provider=<deepseek|...>;scope=<datasets, use>"

All four fields are required; unknown fields are refused; ``provider`` must match the provider the run actually
sends data to. The parsed record is written into the run plan (and so into the audit trail).
PhysioNet guidance (2025-09-24) asks that an external service be verified for zero data retention, no training
use and no human review; the scope text is the place to state what was verified.
"""
from __future__ import annotations

from datetime import date

FIELDS = ('approver', 'date', 'provider', 'scope')
PUBLIC_SOURCES = frozenset({'synthetic-demo', 'synthetic', 'healthbench', 'medexpert', 'public'})
EXAMPLE = 'approver=<name>;date=2026-10-26;provider=deepseek;scope=MIMIC-IV chronic-disease cases, research use'


class ApprovalError(ValueError):
    pass


def is_public(data_source: str) -> bool:
    return data_source.strip().lower() in PUBLIC_SOURCES


def parse_approval(ref: str | None) -> dict:
    if not ref or not ref.strip():
        raise ApprovalError(f'No data-use approval given. Use --data-approval-ref "{EXAMPLE}".')
    record: dict[str, str] = {}
    for part in ref.split(';'):
        if not part.strip():
            continue
        if '=' not in part:
            raise ApprovalError(f'Approval field {part.strip()!r} is not key=value. Expected: "{EXAMPLE}".')
        key, value = (x.strip() for x in part.split('=', 1))
        key = key.lower()
        if key not in FIELDS:
            raise ApprovalError(f'Unknown approval field {key!r}; allowed: {", ".join(FIELDS)}.')
        if key in record:
            raise ApprovalError(f'Approval field {key!r} given twice.')
        record[key] = value
    missing = [k for k in FIELDS if not record.get(k)]
    if missing:
        raise ApprovalError(f'Approval record is missing {", ".join(missing)}. Expected: "{EXAMPLE}".')
    try:
        when = date.fromisoformat(record['date'])
    except ValueError:
        raise ApprovalError('Approval date must be YYYY-MM-DD.') from None
    if when > date.today():
        raise ApprovalError('Approval date lies in the future.')
    if len(record['scope']) < 8:
        raise ApprovalError('Approval scope is too short to say which data and use it covers.')
    record['provider'] = record['provider'].lower()
    return {k: record[k] for k in FIELDS}


def check_run(data_source: str, ref: str | None, provider: str, paid: bool) -> dict | None:
    """None for public/synthetic data or unpaid (offline) runs; otherwise the validated approval record.
    Raises ApprovalError when a paid run on non-public data lacks a matching approval."""
    if not paid or is_public(data_source):
        return parse_approval(ref) if (ref and paid) else None
    record = parse_approval(ref)
    if record['provider'] != provider.lower():
        raise ApprovalError(f'The approval covers provider {record["provider"]!r}, but this run sends data to '
                            f'{provider.lower()!r}.')
    return record
