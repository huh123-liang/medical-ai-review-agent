"""Agreement statistics for ORDINAL per-dimension scores (v3 profiles), pre-registered in
docs/project/PROTOCOL-V3-ORDINAL.md.

Input rows (one per item, labels joined only AFTER predictions are sealed):
    {'item_id', 'group_id', 'status': 'Completed'|..., 'pred': {dim: int|None},
     'raters': [rater ids], 'scores': {dim: [int per rater]}}

Per dimension:
* primary   - quadratic-weighted kappa (QWK), judge vs EVERY rater rating (one pair per rater), categories =
              the full profile scale; 95 % CI by resampling whole question groups.
* secondary - linear-weighted kappa, exact agreement, within-one agreement, MAE, Spearman rho, confusion matrix,
              item-weighted QWK (first rater only), QWK per rater id.
* physician reference - on items with >= 2 raters: rater-vs-rater QWK and the judge-vs-rater QWK on the SAME items,
              with their difference and a group-bootstrap CI of the difference.
Coverage: failed / not-run items stay in the denominator of coverage; if more than COVERAGE_MAX_FAILURE_SHARE fail,
the primary statistic is withheld (conditional diagnostics only), as in the v2 protocol.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .agreement import group_bootstrap_ci, weighted_kappa

COVERAGE_MAX_FAILURE_SHARE = 0.05


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[int], b: list[int]) -> float | None:
    if len(a) < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return None if va == 0 or vb == 0 else cov / math.sqrt(va * vb)


def _pairs(rows: list[dict], dim: str, first_rater_only: bool = False, rater: str | None = None) -> list[tuple]:
    out = []
    for r in rows:
        p = (r.get('pred') or {}).get(dim)
        if p is None:
            continue
        for k, (rid, g) in enumerate(zip(r['raters'], r['scores'][dim])):
            if first_rater_only and k > 0:
                break
            if rater is not None and rid != rater:
                continue
            out.append((p, g))
    return out


def _qwk(pairs: list[tuple], cats: list[int], weights: str = 'quadratic') -> float | None:
    if not pairs:
        return None
    return weighted_kappa([p for p, _ in pairs], [g for _, g in pairs], weights, categories=cats)


def describe(pairs: list[tuple], cats: list[int]) -> dict:
    n = len(pairs)
    if not n:
        return {'n_pairs': 0}
    k = {c: i for i, c in enumerate(cats)}
    conf = [[0] * len(cats) for _ in cats]
    for p, g in pairs:
        conf[k[g]][k[p]] += 1                     # rows = rater, columns = judge
    return {'n_pairs': n, 'qwk': _qwk(pairs, cats), 'lwk': _qwk(pairs, cats, 'linear'),
            'exact_agreement': sum(p == g for p, g in pairs) / n,
            'within_one': sum(abs(p - g) <= 1 for p, g in pairs) / n,
            'mae': sum(abs(p - g) for p, g in pairs) / n,
            'mean_signed_error_judge_minus_rater': sum(p - g for p, g in pairs) / n,
            'spearman': spearman([p for p, _ in pairs], [g for _, g in pairs]),
            'judge_distribution': dict(sorted(Counter(p for p, _ in pairs).items())),
            'rater_distribution': dict(sorted(Counter(g for _, g in pairs).items())),
            'confusion_rater_rows_judge_cols': conf, 'categories': cats}


def rater_pairs(rows: Iterable[dict], dim: str) -> list[tuple]:
    """All ordered pairs between different raters of the same item (both directions; kappa is symmetric)."""
    out = []
    for r in rows:
        s = r['scores'][dim]
        for i in range(len(s)):
            for j in range(len(s)):
                if i != j:
                    out.append((s[i], s[j]))
    return out


def score_dimension(rows: list[dict], dim: str, cats: list[int], n_boot: int = 2000, seed: int = 0) -> dict:
    done = [r for r in rows if r.get('status') == 'Completed' and (r.get('pred') or {}).get(dim) is not None]
    groups = defaultdict(list)
    for r in done:
        groups[r['group_id']].append(r)
    pairs = _pairs(done, dim)
    out = {'primary': describe(pairs, cats),
           'primary_qwk_group_bootstrap_95ci': group_bootstrap_ci(groups, lambda rs: _qwk(_pairs(rs, dim), cats),
                                                                   n_boot, seed),
           'item_weighted_first_rater': describe(_pairs(done, dim, first_rater_only=True), cats),
           'per_rater': {rid: describe(_pairs(done, dim, rater=rid), cats)
                         for rid in sorted({x for r in done for x in r['raters']})}}
    multi = [r for r in done if len(r['raters']) >= 2]
    if multi:
        mg = defaultdict(list)
        for r in multi:
            mg[r['group_id']].append(r)

        def delta(rs):
            a, b = _qwk(_pairs(rs, dim), cats), _qwk(rater_pairs(rs, dim), cats)
            return None if a is None or b is None else a - b
        out['physician_reference'] = {
            'items': len(multi), 'rater_vs_rater': describe(rater_pairs(multi, dim), cats),
            'judge_vs_rater_same_items': describe(_pairs(multi, dim), cats),
            'delta_qwk_judge_minus_raters': delta(multi),
            'delta_qwk_group_bootstrap_95ci': group_bootstrap_ci(mg, delta, n_boot, seed),
            'note': 'Small sample: the reference describes these items only and is not a population ceiling.'}
    else:
        out['physician_reference'] = None
    return out


def score_ordinal(rows: list[dict], profile, n_boot: int = 2000, seed: int = 0,
                  primary_dimensions: list[str] | None = None) -> dict:
    planned = len(rows)
    failed = sum(r.get('status') != 'Completed' for r in rows)
    share = failed / planned if planned else None
    coverage_ok = share is not None and share <= COVERAGE_MAX_FAILURE_SHARE
    dims = {}
    for d in profile.dimensions:
        cats = list(range(d.scale_min, d.scale_max + 1))
        dims[d.key] = score_dimension(rows, d.key, cats, n_boot, seed)
    primary = primary_dimensions or [d.key for d in profile.dimensions]
    qwks = [dims[k]['primary']['qwk'] for k in primary if dims[k]['primary'].get('qwk') is not None]
    cats = {d.key: list(range(d.scale_min, d.scale_max + 1)) for d in profile.dimensions}
    done = [r for r in rows if r.get('status') == 'Completed']
    groups = defaultdict(list)
    for r in done:
        groups[r['group_id']].append(r)

    def mean_qwk(rs):
        vals = [_qwk(_pairs(rs, k), cats[k]) for k in primary]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)
    return {'profile': profile.profile_id, 'planned_items': planned, 'failed_or_not_run': failed,
            'failure_share': share, 'coverage_ok': coverage_ok,
            'primary_dimensions': primary,
            'mean_primary_qwk': (sum(qwks) / len(qwks)) if (qwks and coverage_ok and len(qwks) == len(primary))
            else None,
            'mean_primary_qwk_group_bootstrap_95ci': group_bootstrap_ci(groups, mean_qwk, n_boot, seed)
            if coverage_ok else None,
            'primary_withheld_reason': None if coverage_ok else
            f'Failure share {share} exceeds {COVERAGE_MAX_FAILURE_SHARE}; per-dimension numbers are conditional.',
            'dimensions': dims}


def compare_runs(rows_a: list[dict], rows_b: list[dict], profile, primary_dimensions: list[str],
                 n_boot: int = 2000, seed: int = 0) -> dict:
    """Paired comparison of two judge configurations scored on the SAME items (development decisions).

    delta = mean primary QWK of B minus A; the CI resamples whole question groups and keeps each item's two
    predictions together. Items failed in either run are dropped from both (reported as ``dropped``)."""
    a = {r['item_id']: r for r in rows_a}
    b = {r['item_id']: r for r in rows_b}
    if set(a) != set(b):
        raise ValueError('The runs do not cover the same items; a paired comparison needs identical item sets.')
    cats = {d.key: list(range(d.scale_min, d.scale_max + 1)) for d in profile.dimensions}
    pairs, dropped = [], 0
    for item_id in sorted(a):
        ra, rb = a[item_id], b[item_id]
        if ra.get('status') != 'Completed' or rb.get('status') != 'Completed':
            dropped += 1
            continue
        pairs.append({'group_id': ra['group_id'], 'raters': ra['raters'], 'scores': ra['scores'],
                      'pred_a': ra['pred'], 'pred_b': rb['pred']})

    def view(rs, side):
        return [{**r, 'pred': r['pred_' + side]} for r in rs]

    def mean_qwk(rs, side):
        vals = [_qwk(_pairs(view(rs, side), k), cats[k]) for k in primary_dimensions]
        return None if any(v is None for v in vals) else sum(vals) / len(vals)

    def delta(rs):
        x, y = mean_qwk(rs, 'a'), mean_qwk(rs, 'b')
        return None if x is None or y is None else y - x
    groups = defaultdict(list)
    for r in pairs:
        groups[r['group_id']].append(r)
    per_dim = {}
    for d in profile.dimensions:
        da, db = describe(_pairs(view(pairs, 'a'), d.key), cats[d.key]), describe(_pairs(view(pairs, 'b'), d.key), cats[d.key])
        per_dim[d.key] = {'a': da, 'b': db,
                          'delta_qwk': None if da.get('qwk') is None or db.get('qwk') is None else db['qwk'] - da['qwk']}
    return {'items': len(a), 'paired_items': len(pairs), 'dropped': dropped, 'groups': len(groups),
            'primary_dimensions': primary_dimensions,
            'mean_primary_qwk_a': mean_qwk(pairs, 'a'), 'mean_primary_qwk_b': mean_qwk(pairs, 'b'),
            'delta_b_minus_a': delta(pairs),
            'delta_group_bootstrap_95ci': group_bootstrap_ci(groups, delta, n_boot, seed),
            'dimensions': per_dim}
