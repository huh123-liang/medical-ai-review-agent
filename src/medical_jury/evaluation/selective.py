"""Selective evaluation with a finite-sample guarantee (plan phase 1, PROTOCOL-V3-SELECTIVE).

The judge decides an item automatically only when its confidence reaches a threshold lambda; otherwise the item
is escalated to a physician. lambda is calibrated on labelled calibration items so that, with probability at least
1 - delta, the error rate among automatically decided items is at most alpha (Learn-then-Test fixed-sequence
procedure with an exact Clopper-Pearson upper bound; Angelopoulos/Bates et al.; applied to LLM judges in Jung et al.,
"Trust or Escalate", ICLR 2025). The bound is then checked once on a separate test half.

Error definitions (one item, judge verdict v, physician labels L):
* ``consensus`` (primary): an error only if all physicians agree and v differs from them. Items where physicians
  disagree have no consensus to violate and count as non-errors (they are reported separately).
* ``pairwise`` (secondary): mean disagreement with each physician (bounded below by physician disagreement).
"""
from __future__ import annotations

import math
from collections import Counter


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space (stable for large n)."""
    if p <= 0:
        return 1.0
    if p >= 1:
        return 0.0 if k < n else 1.0
    lp, lq = math.log(p), math.log1p(-p)
    base = math.lgamma(n + 1)
    return min(1.0, sum(math.exp(base - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
                        for i in range(0, k + 1)))


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    """Exact one-sided (1 - delta) upper confidence bound for a binomial proportion (k errors in n)."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    lo, hi = k / n, 1.0
    for _ in range(80):                       # bisection: largest p with P(X <= k | p) >= delta
        mid = (lo + hi) / 2
        if binom_cdf(k, n, mid) > delta:
            lo = mid
        else:
            hi = mid
    return hi


def consensus_error(verdict_met: bool, labels: list[bool]) -> int:
    return int(len(set(labels)) == 1 and verdict_met != labels[0])


def pairwise_error(verdict_met: bool, labels: list[bool]) -> float:
    return sum(verdict_met != l for l in labels) / len(labels)


def vote(samples: list[dict]) -> dict | None:
    """Self-consistency confidence from K sampled judgments [{'verdict': 'met'|'not_met'|None, 'confidence': float}].
    Confidence = share of completed samples agreeing with the majority, tie-broken by their mean verbalized
    confidence (scaled into the gap between vote levels). None when no sample completed."""
    done = [s for s in samples if s.get('verdict') in ('met', 'not_met')]
    if not done:
        return None
    counts = Counter(s['verdict'] for s in done)
    top = max(counts.values())
    leaders = sorted(v for v, c in counts.items() if c == top)
    if len(leaders) > 1:                       # exact tie: decide by mean verbalized confidence, else 'not_met'
        mean = {v: sum(s['confidence'] for s in done if s['verdict'] == v) / top for v in leaders}
        verdict = max(leaders, key=lambda v: (mean[v], v == 'not_met'))
    else:
        verdict = leaders[0]
    agree = [s for s in done if s['verdict'] == verdict]
    share = len(agree) / len(samples)          # failed samples count against confidence
    verbal = sum(s['confidence'] for s in agree) / len(agree)
    return {'verdict': verdict, 'vote_share': share, 'verbalized_mean': verbal,
            'confidence': share + verbal / (10 * max(1, len(samples))), 'completed': len(done), 'k': len(samples)}


def calibrate(rows: list[dict], alpha: float, delta: float, min_selected: int = 30) -> dict:
    """rows: [{'confidence': float, 'error': 0/1}]. Fixed-sequence testing from the strictest threshold down; the
    returned lambda is the smallest one whose whole prefix of tested thresholds kept UCB <= alpha.
    Thresholds selecting fewer than ``min_selected`` items are skipped, not tested: whether a threshold is tested
    depends only on confidences (never on labels), so skipping keeps the fixed-sequence guarantee."""
    grid = sorted({r['confidence'] for r in rows if r['confidence'] >= 0}, reverse=True)   # < 0 = failed, escalated
    path, chosen = [], None
    for lam in grid:
        sel = [r for r in rows if r['confidence'] >= lam]
        k, n = sum(r['error'] for r in sel), len(sel)
        if n < min_selected:
            path.append({'lambda': lam, 'selected': n, 'errors': None, 'ucb': None, 'accepted': None,
                         'note': 'skipped: too few items'})
            continue
        ucb = clopper_pearson_upper(k, n, delta)
        ok = ucb <= alpha
        path.append({'lambda': lam, 'selected': n, 'errors': k, 'ucb': ucb, 'accepted': ok})
        if not ok:
            break
        chosen = lam
    coverage = (sum(r['confidence'] >= chosen for r in rows) / len(rows)) if (chosen is not None and rows) else 0.0
    return {'alpha': alpha, 'delta': delta, 'lambda': chosen, 'calibration_coverage': coverage,
            'calibration_items': len(rows), 'path': path}


def evaluate(rows: list[dict], lam: float | None, delta: float) -> dict:
    if lam is None:
        return {'lambda': None, 'coverage': 0.0, 'selected': 0, 'errors': 0, 'risk': None, 'risk_ucb': None}
    sel = [r for r in rows if r['confidence'] >= lam]
    k, n = sum(r['error'] for r in sel), len(sel)
    return {'lambda': lam, 'coverage': n / len(rows) if rows else 0.0, 'selected': n, 'errors': k,
            'risk': (k / n) if n else None, 'risk_ucb': clopper_pearson_upper(k, n, delta) if n else None}


def risk_coverage(rows: list[dict], points=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> list[dict]:
    order = sorted(rows, key=lambda r: -r['confidence'])
    out = []
    for c in points:
        k = max(1, int(round(c * len(order))))
        sel = order[:k]
        out.append({'coverage': c, 'n': k, 'risk': sum(r['error'] for r in sel) / k})
    return out


def auroc(scores: list[float], positives: list[bool]) -> float | None:
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
