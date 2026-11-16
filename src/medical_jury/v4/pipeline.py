"""Steps 0-8 wired together. Two model calls per case (extract, check); the rest is code.

Arms (PLAN-V4 section 7): 'full' (review), 'model_only' (review_model_only: one check call, no criteria list, no
extraction), and 'rules_only' (derived from a full record without further calls)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from ..budget import BudgetExceeded
from ..domain import digest
from ..evaluation.runner import IMMEDIATE_STOP
from ..judge_v2 import BackendError, raw_trace
from . import PIPELINE_VERSION
from .check import CHECK_PROMPT_VERSION, SYSTEM_CHECK, CheckFailure, check_messages, code_checks_of, parse_check
from .criteria import applicable
from .extract import (EXTRACT_PROMPT_VERSION, SYSTEM_EXTRACT, ExtractFailure, check_extraction, extract_messages,
                      parse_extraction, quote_found, unknown_active_drugs)
from .rules import RULES_VERSION, SEVERE, check as rule_check

EXTRACT_MAX_TOKENS = 1200

# Step 7 reconciliation: a SEVERE open-review claim that a medicine is contraindicated / overdosed because of a value
# that code has already checked for that medicine (and code found nothing) is downgraded to moderate and labelled.
# It stays in the findings (so it is still visible and still counts for detection); it only stops forcing escalation.
# A claim that also names any risk code did not check for that medicine is left untouched.
_VALUE_TERMS = {'eGFR': r'egfr|kidney|renal|ckd|lactic', 'potassium': r'potassium|hyperkal',
                'usual maximum daily dose': r'\bdose|maximum|\bmg\b', 'pregnancy': r'pregnan',
                'previous angioedema': r'angioedema', 'same-class duplicate': r'duplicat|two .*same class'}
_OTHER_RISK = r'heart|cardiac|failure|oedema|edema|fluid|bleed|ulcer|liver|hepat|hypoglyc|fracture|pancreat|' \
              r'ketoacid|infection|dehydrat|interact|blood pressure|hypotens|allerg'
RECONCILE_CATEGORIES = ('contraindicated_drug', 'dose_error', 'drug_interaction')


def reconcile(findings: list[dict], code_checks: dict) -> list[dict]:
    import re
    from .rules import LEXICON
    checked = {c['medicine']: set(c['against']) for c in code_checks.get('checked', [])}
    flagged = {d for f in code_checks.get('flagged', []) for d in f.get('medicines', [])}
    out = []
    for f in findings:
        if f.get('source') != 'open_review' or f.get('severity') != SEVERE or f.get('category') not in RECONCILE_CATEGORIES:
            out.append(f)
            continue
        text = f"{f.get('entity', '')} {f.get('issue', '')}".lower()
        drugs = [d for d in LEXICON if re.search(rf'(?<![a-z]){re.escape(d)}(?![a-z])', text)]
        drugs = [d for d in drugs if not (d == 'insulin' and 'insulin glargine' in drugs)]
        values = {v for v, pat in _VALUE_TERMS.items() if re.search(pat, text)}
        if (len(drugs) == 1 and drugs[0] in checked and drugs[0] not in flagged and values
                and values <= checked[drugs[0]] and not re.search(_OTHER_RISK, text)):
            out.append({**f, 'severity': 'moderate', 'severity_model': SEVERE,
                        'reconciled': f"conflicts with code check of {drugs[0]} against {', '.join(sorted(values))}"})
        else:
            out.append(f)
    return out
CHECK_MAX_TOKENS = 2000


def merge_findings(rule_out: list[dict], code_findings: list[dict], items: list[dict], chk: dict | None,
                   opened: list[dict], code_checks: dict) -> tuple[list[dict], list[dict]]:
    """Step 7 as one pure function (used by ReviewerV4, the LangGraph aggregate node and the record replay):
    code findings, then criteria answered not_met / cannot_determine, then reconciled open-review findings.
    `opened` = the check call's other findings with quote_verified already set."""
    findings = [f for f in rule_out + code_findings if f['category'] != 'cannot_determine']
    uncertain = [f for f in rule_out + code_findings if f['category'] == 'cannot_determine']
    if chk is not None:
        for it in items:
            res = chk['criteria'][it['id']]
            if res['status'] == 'not_met':
                findings.append({'source': 'criterion', 'id': it['id'], 'category': it['category'],
                                 'severity': it['severity_if_not_met'], 'entity': it['entity'],
                                 'issue': f"{it['text']} - {res['reason']}", 'quote': res['quote']})
            elif res['status'] == 'cannot_determine':
                uncertain.append({'source': 'criterion', 'id': it['id'], 'category': 'cannot_determine',
                                  'severity': 'moderate', 'entity': it['entity'], 'issue': res['reason']})
        findings.extend(reconcile(opened, code_checks))
    return findings, uncertain


def triage(findings: list[dict], uncertain: list[dict], unknown: list[str], extraction_problems: list[dict],
           failed: list[str]) -> dict:
    reasons = []
    if failed:
        reasons.append('call_failed')
    if any(f['severity'] == SEVERE for f in findings):
        reasons.append('severe_finding')
    if uncertain:
        reasons.append('cannot_determine')
    if unknown:
        reasons.append('unknown_drug')
    if extraction_problems:
        reasons.append('extraction_check_failed')
    decision = 'escalate' if reasons else 'pass_with_findings' if findings else 'pass'
    return {'decision': decision, 'reasons': reasons}


@dataclass
class ReviewerV4:
    backend: object
    library: dict
    store: object | None = None
    stop: threading.Event = field(default_factory=threading.Event)
    extract_system: str = SYSTEM_EXTRACT
    check_system: str = SYSTEM_CHECK
    extract_version: str = EXTRACT_PROMPT_VERSION
    check_version: str = CHECK_PROMPT_VERSION

    def _call(self, step, messages, max_tokens, job_id, parse):
        rec = {'step': step, 'messages_sha256': digest(messages), 'judge': getattr(self.backend, 'name', '?'),
               'model': getattr(self.backend, 'model', '?')}
        if self.stop.is_set():
            return {**rec, 'status': 'NotRun', 'failure_code': 'STOPPED_BY_RUN_STOP'}, None
        started = time.monotonic()
        try:
            completion = self.backend.complete(messages, max_tokens, f'{job_id}:{step}')
        except BackendError as error:
            if error.code in IMMEDIATE_STOP:
                self.stop.set()
            return self._audit(job_id, {**rec, 'status': 'Failed', 'failure_code': error.code}), None
        except BudgetExceeded as error:
            self.stop.set()
            code = 'BUDGET_SETTLEMENT_EXCEEDED' if 'Actual usage' in str(error) else 'BUDGET_BLOCKED'
            return self._audit(job_id, {**rec, 'status': 'Failed', 'failure_code': code}), None
        rec.update(latency_s=round(time.monotonic() - started, 3), usage=completion.usage,
                   finish_reason=completion.finish_reason, **raw_trace(completion.content))
        if completion.finish_reason != 'stop' or not isinstance(completion.content, str):
            return self._audit(job_id, {**rec, 'status': 'Failed', 'failure_code': 'INCOMPLETE_MODEL_OUTPUT'}), None
        try:
            parsed = parse(completion.content)
        except (ExtractFailure, CheckFailure) as error:
            return self._audit(job_id, {**rec, 'status': 'Failed', 'failure_code': error.code,
                                        'failure_detail': error.detail}), None
        return self._audit(job_id, {**rec, 'status': 'Completed'}), parsed

    def _audit(self, job_id, rec):
        if self.store is not None:
            self.store.audit(job_id, f'v4_{rec["step"]}', rec)
        return rec

    def _header(self, arm):
        return {'pipeline': PIPELINE_VERSION, 'arm': arm, 'rules_version': RULES_VERSION,
                'extract_prompt': {'version': self.extract_version, 'sha256': digest(self.extract_system)},
                'check_prompt': {'version': self.check_version, 'sha256': digest(self.check_system)},
                'library_sha256': self.library['sha256']}

    def review(self, case_description: str, document: str, job_id: str) -> dict:
        out = {**self._header('full'), 'calls': []}
        call, ex = self._call('extract', extract_messages(case_description, document, self.extract_system),
                              EXTRACT_MAX_TOKENS, job_id, parse_extraction)
        out['calls'].append(call)
        if ex is None:
            out.update(extraction=None, findings=[], uncertain=[],
                       triage=triage([], [], [], [], [call['failure_code']]))
            return out
        problems = check_extraction(ex, case_description, document)
        unknown = unknown_active_drugs(ex)
        rule_out = rule_check(ex['facts'], ex['medications'])
        items, code_findings = applicable(self.library, ex['facts'], ex['medications'])
        call, chk = self._call('check', check_messages(case_description, document, items, self.check_system,
                                                       code_checks_of(rule_out, ex['medications'])),
                               CHECK_MAX_TOKENS, job_id, lambda raw: parse_check(raw, items))
        out['calls'].append(call)
        failed = [] if chk is not None else [call['failure_code']]
        opened = [] if chk is None else \
            [{**f, 'quote_verified': quote_found(f['quote'], document) if f['quote'] else None}
             for f in chk['other_findings']]
        findings, uncertain = merge_findings(rule_out, code_findings, items, chk, opened,
                                             code_checks_of(rule_out, ex['medications']))
        out.update(extraction={**ex, 'problems': problems, 'unknown_active_drugs': unknown},
                   items=[{'id': i['id'], 'level': i['level'], 'text': i['text']} for i in items],
                   criteria_results=chk['criteria'] if chk else None,
                   findings=findings, uncertain=uncertain, triage=triage(findings, uncertain, unknown, problems, failed))
        return out

    def review_model_only(self, case_description: str, document: str, job_id: str) -> dict:
        out = {**self._header('model_only'), 'calls': []}
        call, chk = self._call('check_only', check_messages(case_description, document, [], self.check_system),
                               CHECK_MAX_TOKENS, job_id, lambda raw: parse_check(raw, []))
        out['calls'].append(call)
        findings = [] if chk is None else \
            [{**f, 'quote_verified': quote_found(f['quote'], document) if f['quote'] else None}
             for f in chk['other_findings']]
        out.update(findings=findings, uncertain=[],
                   triage=triage(findings, [], [], [], [] if chk is not None else [call['failure_code']]))
        return out


def rules_only(full: dict) -> dict:
    """The 'rules only' arm from a full record, with no further calls: rule layer + code-checked templates."""
    out = {k: full[k] for k in ('pipeline', 'rules_version', 'extract_prompt', 'library_sha256') if k in full}
    out['arm'] = 'rules_only'
    ex = full.get('extraction')
    if ex is None:
        return {**out, 'findings': [], 'uncertain': [], 'triage': triage([], [], [], [], ['extract_failed'])}
    coded = [f for f in full['findings'] if f['source'] in ('rule', 'criteria')]
    uncertain = [f for f in full['uncertain'] if f['source'] in ('rule', 'criteria')]
    return {**out, 'findings': coded, 'uncertain': uncertain,
            'triage': triage(coded, uncertain, ex['unknown_active_drugs'], ex['problems'], [])}
