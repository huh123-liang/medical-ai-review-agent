"""Scoring for the router v1 vs v2 comparison (PROTOCOL-V3-ROUTER-V2). Pure functions, no I/O.

Each item has a gold situation (HealthBench category metadata) and, per router version, the situation the router
chose for the gold situation's exclusive group (None if the call failed or it chose "none"). A failure or "none"
counts as wrong, as in production (no situational criterion would be applied).
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

MIN_SITUATION_N = 3          # macro recall averages only gold situations with at least this many items


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b = only v1 right, c = only v2 right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def accuracy(rows: list[dict], version: str) -> float:
    return sum(r[version] == r['gold'] for r in rows) / len(rows) if rows else float('nan')


def recall_by_situation(rows: list[dict], version: str) -> dict[str, dict]:
    by = defaultdict(list)
    for r in rows:
        by[r['gold']].append(r[version] == r['gold'])
    return {g: {'n': len(v), 'recall': sum(v) / len(v)} for g, v in sorted(by.items())}


def macro_recall(rows: list[dict], version: str, min_n: int = MIN_SITUATION_N) -> float:
    rec = [v['recall'] for v in recall_by_situation(rows, version).values() if v['n'] >= min_n]
    return sum(rec) / len(rec) if rec else float('nan')


def bootstrap_diff(rows: list[dict], a: str, b: str, seed: int, reps: int = 4000) -> tuple[float, float]:
    """Percentile 95% CI of accuracy(b) - accuracy(a), resampling items (one item per conversation group)."""
    rng = random.Random(seed)
    n = len(rows)
    d = [(r[b] == r['gold']) - (r[a] == r['gold']) for r in rows]
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return stats[int(0.025 * reps)], stats[int(0.975 * reps) - 1]


def confusion(rows: list[dict], version: str) -> dict[str, dict[str, int]]:
    out: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        out[r['gold']][r[version] or 'none/failed'] += 1
    return {g: dict(c) for g, c in sorted(out.items())}


def auroc(scores: list[float], positives: list[bool]) -> float | None:
    pos = [s for s, p in zip(scores, positives) if p]
    neg = [s for s, p in zip(scores, positives) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def compare(rows: list[dict], v1: str, v2: str, seed: int, min_gain_macro: float = -0.02, alpha: float = 0.05) -> dict:
    """Pre-registered decision: adopt v2 iff accuracy(v2) > accuracy(v1) with exact McNemar p < alpha AND
    macro recall(v2) >= macro recall(v1) + min_gain_macro (i.e. it does not just trade one situation for another)."""
    b = sum(r[v1] == r['gold'] and r[v2] != r['gold'] for r in rows)
    c = sum(r[v2] == r['gold'] and r[v1] != r['gold'] for r in rows)
    acc1, acc2 = accuracy(rows, v1), accuracy(rows, v2)
    mr1, mr2 = macro_recall(rows, v1), macro_recall(rows, v2)
    p = mcnemar_exact(b, c)
    lo, hi = bootstrap_diff(rows, v1, v2, seed)
    adopt = acc2 > acc1 and p < alpha and mr2 >= mr1 + min_gain_macro
    by_group = defaultdict(list)
    for r in rows:
        by_group[r['group']].append(r)
    return {
        'n': len(rows), 'accuracy': {v1: round(acc1, 4), v2: round(acc2, 4)},
        'difference': round(acc2 - acc1, 4), 'difference_ci95': [round(lo, 4), round(hi, 4)],
        'discordant': {'only_v1_right': b, 'only_v2_right': c}, 'mcnemar_exact_p': round(p, 5),
        'macro_recall': {v1: round(mr1, 4), v2: round(mr2, 4)},
        'by_group': {g: {'n': len(rs), v1: round(accuracy(rs, v1), 4), v2: round(accuracy(rs, v2), 4)}
                     for g, rs in sorted(by_group.items())},
        'adopt_v2': adopt,
        'rule': f'adopt v2 iff acc(v2) > acc(v1), exact McNemar p < {alpha}, macro recall(v2) >= macro recall(v1) '
                f'{min_gain_macro:+.2f}',
    }
