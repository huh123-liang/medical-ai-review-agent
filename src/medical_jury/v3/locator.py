"""locator-3: where is a model quote in the answer? Only ``exact`` counts as located.

Diagnostic tiers (never upgraded to ``exact``; spans always refer to the ORIGINAL answer):
* ``markup``            - equal after removing Markdown emphasis/code markers from BOTH quote and answer
                          (symmetric, unlike locator-2). A single ``*`` or ``_`` is removed only when it is not
                          between two letters/digits, so ``2*3`` and ``snake_case`` are kept.
* ``whitespace``        - additionally equal after collapsing runs of whitespace.
* ``list_structure``    - additionally equal after removing list markers / heading hashes at line starts.
* ``case_first_letter`` - equal at one of the tiers above except for the case of the quote's first letter.
``cross_structure`` is set when a non-exact match spans a line break of the answer (e.g. a heading glued
to a list item): the characters are there in order, but they are not one sentence of the answer.

Changes of words, numbers, negations, punctuation or omissions inside the quote are never matched.
"""
from __future__ import annotations

import re

LOCATOR_VERSION = 'locator-3'
MIN_QUOTE_WORD_CHARS = 3
TIERS = ('exact', 'markup', 'whitespace', 'list_structure', 'case_first_letter', 'not_located',
         'too_short_unverifiable')
_LIST_MARK = re.compile(r'(?m)^[ \t]*(?:[-*+•]|\d{1,3}[.)]|#{1,6})[ \t]+')


def _all_spans(text: str, quote: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    while quote:
        k = text.find(quote, start)
        if k < 0:
            break
        spans.append((k, k + len(quote)))
        start = k + 1
    return spans


def _strip_markup(text: str, idx: list[int]) -> tuple[str, list[int]]:
    out, m, i, n = [], [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '`':
            i += 1
            continue
        if ch in '*_':
            j = i
            while j < n and text[j] == ch:
                j += 1
            run = j - i
            prev = text[i - 1] if i > 0 else ' '
            nxt = text[j] if j < n else ' '
            if run >= 2 or not (prev.isalnum() and nxt.isalnum()):
                i = j                       # emphasis delimiter run: drop
                continue
        out.append(ch)
        m.append(idx[i])
        i += 1
    return ''.join(out), m


def _collapse_ws(text: str, idx: list[int]) -> tuple[str, list[int]]:
    out, m, prev_space = [], [], False
    for ch, k in zip(text, idx):
        if ch.isspace():
            if prev_space:
                continue
            out.append(' ')
            m.append(k)
            prev_space = True
        else:
            out.append(ch)
            m.append(k)
            prev_space = False
    return ''.join(out), m


def _strip_lists(text: str, idx: list[int]) -> tuple[str, list[int]]:
    keep = [True] * len(text)
    for mt in _LIST_MARK.finditer(text):
        for k in range(mt.start(), mt.end()):
            keep[k] = False
    return ''.join(c for c, k in zip(text, keep) if k), [i for i, k in zip(idx, keep) if k]


def _layers(text: str):
    """Yield (tier, normalised text, index map back to the original) from weakest change to strongest."""
    idx = list(range(len(text)))
    t1, m1 = _strip_markup(text, idx)
    yield 'markup', t1, m1
    t2, m2 = _collapse_ws(t1, m1)
    yield 'whitespace', t2, m2
    t3, m3 = _strip_lists(text, idx)
    t3, m3 = _strip_markup(t3, m3)
    t3, m3 = _collapse_ws(t3, m3)
    yield 'list_structure', t3, m3


def _normalise_quote(q: str, tier: str) -> str:
    """The same transformations as the answer layer, in the same order (symmetric)."""
    if tier == 'list_structure':
        q, _ = _strip_lists(q, list(range(len(q))))
    t, _ = _strip_markup(q, list(range(len(q))))
    if tier in ('whitespace', 'list_structure'):
        t, _ = _collapse_ws(t, list(range(len(t))))
        t = t.strip()
    return t


def _swap_first(q: str) -> str | None:
    for k, ch in enumerate(q):
        if ch.isalpha():
            sw = ch.swapcase()
            return None if sw == ch else q[:k] + sw + q[k + 1:]
    return None


def locate_quote_v3(answer: str, quote: str) -> dict:
    entry = {'chars': len(quote), 'has_ellipsis': '...' in quote or '…' in quote, 'cross_structure': False,
             'locator': LOCATOR_VERSION}
    if sum(ch.isalnum() for ch in quote) < MIN_QUOTE_WORD_CHARS:
        return {**entry, 'tier': 'too_short_unverifiable', 'spans': []}
    spans = _all_spans(answer, quote)
    if spans:
        return {**entry, 'tier': 'exact', 'spans': [list(s) for s in spans]}
    layers = list(_layers(answer))
    for candidates, case_tier in (([quote], None), ([_swap_first(quote)], 'case_first_letter')):
        if candidates[0] is None:
            continue
        for tier, text, idx in layers:
            q = _normalise_quote(candidates[0], tier)
            if sum(ch.isalnum() for ch in q) < MIN_QUOTE_WORD_CHARS:
                continue
            found = _all_spans(text, q)
            if found:
                orig = [[idx[a], idx[b - 1] + 1] for a, b in found]
                cross = any('\n' in answer[a:b] for a, b in orig)
                return {**entry, 'tier': case_tier or tier, 'spans': orig, 'cross_structure': cross}
    return {**entry, 'tier': 'not_located', 'spans': []}
