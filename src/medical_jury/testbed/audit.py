"""Stricter, post-hoc audit of written-up cases (testbed-verify-2). Added 2026-11-16 after the P1/P3 review showed
that verify() (testbed-rules-1) accepted write-ups that weakened an urgent instruction or invented a dose.

It does NOT change any existing render, case, label or score; it only reports which write-ups do not faithfully carry
the reference plan, so results can also be reported on the subset that passes both checks."""
from __future__ import annotations

import re

from .errors import sentences

AUDIT_VERSION = 'testbed-verify-2'
_TIMING = r'same[- ]day|\btoday\b|right away|immediate(?:ly)?\b|\bnow\b|as soon as possible|without delay|this afternoon'
_ACTION = r'assess|\bseen\b|\bsee\b|review|doctor|clinic|hospital|emergency|a&e|urgent care|medical attention|evaluat'
_CONDITION = r'\bif\b|\bunless\b|\bshould\b|\bwhen you\b|\bin case\b|remain|persist|stay[s]? (?:high|raised)|still'
_TOPIC = {'urgent_bp': r'blood pressure|\bbp\b|hypertensi', 'urgent_k': r'potassium|hyperkal'}


def _clauses(sentence: str) -> list[str]:
    return [c for c in re.split(r'[,;:—–]| - | and (?=go|get|seek|attend|call|see|arrange|you)', sentence) if c.strip()]


def urgent_ok(text: str, element: str) -> bool:
    """True if a sentence on the topic says, with no condition before it, to be assessed the same day (e.g. 'arrange
    same-day urgent medical assessment'); 'if you get chest pain, go to A&E right away' is not enough."""
    ss = sentences(text)
    for i, s in enumerate(ss):
        context = s if i == 0 else ss[i - 1] + ' ' + s          # 'Given the severity, ...' refers to the sentence before
        if not re.search(_TOPIC[element], context, re.I):
            continue
        for m in re.finditer(_TIMING, s, re.I):
            if re.search(_ACTION, s[max(0, m.start() - 80):m.end() + 80], re.I) \
                    and not re.search(_CONDITION, s[:m.start()], re.I):
                return True
    return False


def invented_doses(plan: dict, text: str) -> list[str]:
    """Medicines whose dose the plan leaves to someone else (dose_mg None) but the write-up gives an mg dose."""
    out = []
    for m in plan['medications']:
        if m['dose_mg'] is None and re.search(rf'{re.escape(m["drug"])}\b[^.\n]{{0,40}}?\d[\d,.]*\s?mg\b', text, re.I):
            out.append(m['drug'])
    return out


def audit(plan: dict, text: str) -> list[str]:
    problems = []
    for element in ('urgent_bp', 'urgent_k'):
        if element in plan['elements'] and not urgent_ok(text, element):
            problems.append(f'{element}: no unconditional same-day assessment')
    problems += [f'invented mg dose: {d}' for d in invented_doses(plan, text)]
    return problems
