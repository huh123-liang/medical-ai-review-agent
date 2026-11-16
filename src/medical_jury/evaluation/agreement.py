"""Agreement statistics for judge-vs-human meta-evaluation (pure functions, no I/O).

Conventions
-----------
* ``pairs`` is a list of ``(prediction, reference)`` tuples. For binary work both are
  original Python booleans where ``True`` means the criterion is *met*.
* Ordinal statistics (weighted kappa) refuse booleans: HealthBench physician labels are
  binary, so a "weighted kappa" on them is just Cohen's kappa under another name.
* Nothing here reads files, labels or model outputs; callers decide what is compared.
"""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Callable, Hashable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Basic binary statistics
# ---------------------------------------------------------------------------


def _check_binary(pairs: Sequence[tuple]) -> None:
    for p, g in pairs:
        if type(p) is not bool or type(g) is not bool:
            raise TypeError("Binary statistics require original booleans (True = met).")


def confusion(pairs: Sequence[tuple[bool, bool]]) -> dict:
    _check_binary(pairs)
    tp = sum(p and g for p, g in pairs)
    tn = sum((not p) and (not g) for p, g in pairs)
    fp = sum(p and (not g) for p, g in pairs)
    fn = sum((not p) and g for p, g in pairs)
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float | None:
    """Unweighted Cohen's kappa for two aligned nominal label sequences."""
    if len(a) != len(b):
        raise ValueError("Label sequences must be aligned.")
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if pe >= 1:
        return None  # undefined: both raters constant on the same label
    return (po - pe) / (1 - pe)


def gwet_ac1(a: Sequence[Hashable], b: Sequence[Hashable], categories: Iterable[Hashable] | None = None) -> float | None:
    """Gwet's AC1 for two raters; robust to the prevalence paradox of kappa.

    ``categories`` defaults to the observed union; pass ``[True, False]`` for binary
    data so q=2 even when one label never occurs in a small sample.
    """
    if len(a) != len(b):
        raise ValueError("Label sequences must be aligned.")
    n = len(a)
    if n == 0:
        return None
    cats = list(categories) if categories is not None else sorted(set(a) | set(b), key=repr)
    q = len(cats)
    if q < 2:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pi = {k: (ca.get(k, 0) + cb.get(k, 0)) / (2 * n) for k in cats}
    pe = sum(v * (1 - v) for v in pi.values()) / (q - 1)
    if pe >= 1:
        return None
    return (po - pe) / (1 - pe)


def macro_f1(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    c = confusion(pairs)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    scores = []
    for z in (tp, tn):
        denominator = 2 * z + fp + fn
        if denominator == 0:
            return None
        scores.append(2 * z / denominator)
    return sum(scores) / 2


def binary_summary(pairs: Sequence[tuple[bool, bool]]) -> dict:
    """One-stop summary used by every report so the metric family is never cherry-picked."""
    _check_binary(pairs)
    n = len(pairs)
    if n == 0:
        return {"n": 0, "agreement": None, "cohen_kappa": None, "gwet_ac1": None,
                "macro_f1": None, "reference_met_rate": None, "predicted_met_rate": None,
                "confusion": confusion(pairs)}
    pred = [p for p, _ in pairs]
    ref = [g for _, g in pairs]
    c = confusion(pairs)
    return {"n": n,
            "agreement": (c["TP"] + c["TN"]) / n,
            "cohen_kappa": cohen_kappa(pred, ref),
            "gwet_ac1": gwet_ac1(pred, ref, [True, False]),
            "macro_f1": macro_f1(pairs),
            "reference_met_rate": sum(ref) / n,
            "predicted_met_rate": sum(pred) / n,
            "sensitivity_not_met": (c["TN"] / (c["TN"] + c["FP"])) if (c["TN"] + c["FP"]) else None,
            "confusion": c}


# ---------------------------------------------------------------------------
# Ordinal / multi-rater statistics
# ---------------------------------------------------------------------------


def weighted_kappa(a: Sequence[int], b: Sequence[int], weights: str = "quadratic",
                   categories: Sequence[int] | None = None) -> float | None:
    """Cohen's weighted kappa for ORDINAL integer scores (e.g. physician 1-5 ratings).

    Booleans are rejected on purpose: binary met/not_met labels have no ordering
    distance, and weighting them changes nothing except the name of the statistic.
    """
    if len(a) != len(b):
        raise ValueError("Label sequences must be aligned.")
    for v in list(a) + list(b):
        if type(v) is bool or not isinstance(v, int):
            raise TypeError("weighted_kappa needs ordinal integer scores; binary labels -> use cohen_kappa/gwet_ac1.")
    if weights not in {"linear", "quadratic"}:
        raise ValueError("weights must be 'linear' or 'quadratic'.")
    n = len(a)
    if n == 0:
        return None
    cats = sorted(categories if categories is not None else set(a) | set(b))
    k = len(cats)
    if k < 2:
        return None
    index = {c: i for i, c in enumerate(cats)}
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[index[x]][index[y]] += 1
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    power = 2 if weights == "quadratic" else 1
    w = [[(abs(i - j) / (k - 1)) ** power for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * row[i] * col[j] / n for i in range(k) for j in range(k))
    if den == 0:
        return None
    return 1 - num / den


def krippendorff_alpha_nominal(units: Sequence[Sequence[Hashable | None]]) -> float | None:
    """Krippendorff's alpha (nominal) for any number of raters per unit; ``None`` = missing."""
    coincidence: dict[tuple, float] = defaultdict(float)
    for unit in units:
        values = [v for v in unit if v is not None]
        m = len(values)
        if m < 2:
            continue
        for i, x in enumerate(values):
            for j, y in enumerate(values):
                if i != j:
                    coincidence[(x, y)] += 1 / (m - 1)
    totals: dict[Hashable, float] = defaultdict(float)
    for (x, _), value in coincidence.items():
        totals[x] += value
    n = sum(totals.values())
    if n <= 1:
        return None
    disagree_obs = sum(v for (x, y), v in coincidence.items() if x != y)
    disagree_exp = sum(totals[x] * totals[y] for x in totals for y in totals if x != y)
    if disagree_exp == 0:
        return None
    return 1 - (n - 1) * disagree_obs / disagree_exp


# ---------------------------------------------------------------------------
# Resampling and hypothesis tests
# ---------------------------------------------------------------------------


def group_bootstrap_ci(groups: dict[Hashable, list], metric_fn: Callable[[list], float | None],
                       n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                       strata: dict[Hashable, Hashable] | None = None) -> dict:
    """Percentile CI resampling whole conversation groups (optionally within strata).

    ``groups`` maps group_id -> list of rows; ``metric_fn`` receives the flattened rows.
    Resampling groups, not rows, respects the dependence between criteria of one dialogue.
    """
    keys = sorted(groups, key=repr)
    if not keys:
        return {"lower": None, "upper": None, "valid_draws": 0, "n_boot": n_boot}
    by_stratum: dict[Hashable, list] = defaultdict(list)
    for key in keys:
        by_stratum[strata.get(key) if strata else None].append(key)
    rng = random.Random(seed)
    draws = []
    for _ in range(n_boot):
        rows = []
        for members in by_stratum.values():
            for key in rng.choices(members, k=len(members)):
                rows.extend(groups[key])
        value = metric_fn(rows)
        if value is not None and math.isfinite(value):
            draws.append(value)
    if not draws:
        return {"lower": None, "upper": None, "valid_draws": 0, "n_boot": n_boot}
    draws.sort()
    lo = draws[int((len(draws) - 1) * (alpha / 2))]
    hi = draws[int((len(draws) - 1) * (1 - alpha / 2))]
    return {"lower": lo, "upper": hi, "valid_draws": len(draws), "n_boot": n_boot}


def permutation_test_kappa(items: Sequence[tuple], n_perm: int = 5000, seed: int = 0,
                           groups: Sequence[Hashable] | None = None) -> dict:
    """One-sided test of H0 'judge decisions are independent of physician labels' (kappa <= 0).

    ``items``: one entry per JUDGED ITEM: ``(prediction, labels)`` where ``labels`` is a list of the
    physicians' booleans for that item (a bare boolean is accepted as a one-label item). The judge's
    decision is permuted ACROSS ITEMS, never across individual (judge, physician) pairs: pairs from
    the same item share one decision, so pair-level shuffling would understate the null variance.

    ``groups`` (one id per item): permute whole CONVERSATION GROUPS instead. The block of decisions of
    one group is moved together to another group of the same size (blocks are only exchangeable
    between equal-sized groups). Criteria of one dialogue are correlated, so item-level shuffling
    overstates significance when groups hold several items; validation samples 2 items per group.
    """
    norm = []
    for pred, labels in items:
        labels = list(labels) if isinstance(labels, (list, tuple)) else [labels]
        if type(pred) is not bool or any(type(l) is not bool for l in labels):
            raise TypeError("Binary statistics require original booleans (True = met).")
        norm.append((pred, labels))
    if groups is not None and len(groups) != len(norm):
        raise ValueError('groups must give one group id per item.')

    def pooled(preds):
        a = [p for p, (_, labels) in zip(preds, norm) for _ in labels]
        b = [l for _, labels in norm for l in labels]
        return cohen_kappa(a, b)

    preds = [p for p, _ in norm]
    unit = 'item' if groups is None else 'group'
    observed = pooled(preds)
    if observed is None:
        return {"observed_kappa": None, "p_value": None, "n_perm": n_perm, "unit": unit}
    rng = random.Random(seed)
    hits = 0
    if groups is None:
        shuffled = preds[:]
        for _ in range(n_perm):
            rng.shuffle(shuffled)
            value = pooled(shuffled)
            if value is not None and value >= observed:
                hits += 1
    else:
        members: dict[Hashable, list[int]] = defaultdict(list)
        for i, g in enumerate(groups):
            members[g].append(i)
        by_size: dict[int, list[Hashable]] = defaultdict(list)
        for g in sorted(members, key=repr):
            by_size[len(members[g])].append(g)
        for _ in range(n_perm):
            shuffled = preds[:]
            for size, gs in by_size.items():
                blocks = [[preds[i] for i in members[g]] for g in gs]
                rng.shuffle(blocks)
                for g, block in zip(gs, blocks):
                    for i, value in zip(members[g], block):
                        shuffled[i] = value
            value = pooled(shuffled)
            if value is not None and value >= observed:
                hits += 1
    out = {"observed_kappa": observed, "p_value": (hits + 1) / (n_perm + 1), "n_perm": n_perm, "unit": unit}
    if groups is not None:
        out["n_groups"] = len(members)
        out["group_sizes"] = {str(k): len(v) for k, v in sorted(by_size.items())}
    return out


def exact_mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict:
    """Two-sided exact McNemar test on paired per-item correctness of two predictors."""
    if len(correct_a) != len(correct_b):
        raise ValueError("Paired sequences must be aligned.")
    b = sum(x and not y for x, y in zip(correct_a, correct_b))
    c = sum(y and not x for x, y in zip(correct_a, correct_b))
    n = b + c
    if n == 0:
        return {"a_only_correct": b, "b_only_correct": c, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return {"a_only_correct": b, "b_only_correct": c, "p_value": min(1.0, 2 * tail)}


def constant_baselines(pairs: Sequence[tuple[bool, bool]]) -> dict:
    """Trivial predictors on the SAME comparisons; a judge must beat these to mean anything.

    The McNemar p-value here treats (judge, physician) pairs as independent although pairs of one item
    share a decision; it is reported as a descriptive diagnostic only and is NOT an acceptance gate.
    """
    _check_binary(pairs)
    ref = [g for _, g in pairs]
    judge_correct = [p == g for p, g in pairs]
    out = {}
    for name, const in (("always_met", True), ("always_not_met", False)):
        base = [(const, g) for g in ref]
        out[name] = {**binary_summary(base),
                     "judge_vs_baseline_mcnemar": exact_mcnemar(judge_correct, [const == g for g in ref])}
    return out


# ---------------------------------------------------------------------------
# Student t distribution (to avoid a SciPy dependency) and multiple testing
# ---------------------------------------------------------------------------


def _betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1.0 + aa / c if abs(1.0 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_sf(t: float, df: float) -> float:
    """P(T > t) for Student's t with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be positive.")
    tail = 0.5 * _betainc(df / 2.0, 0.5, df / (df + t * t))
    return tail if t > 0 else 1.0 - tail


def benjamini_yekutieli(p_values: Sequence[float], alpha: float = 0.05) -> list[bool]:
    """BY step-up FDR control (valid under arbitrary dependence). Returns reject flags."""
    m = len(p_values)
    if m == 0:
        return []
    c_m = sum(1 / i for i in range(1, m + 1))
    order = sorted(range(m), key=lambda i: p_values[i])
    max_k = 0
    for rank, i in enumerate(order, 1):
        if p_values[i] <= rank * alpha / (m * c_m):
            max_k = rank
    reject = [False] * m
    for rank, i in enumerate(order, 1):
        if rank <= max_k:
            reject[i] = True
    return reject


# ---------------------------------------------------------------------------
# Alternative Annotator Test (Calderon, Reichart & Dror, ACL 2025; arXiv:2501.10970)
# ---------------------------------------------------------------------------


def alt_test(items: Sequence[dict], epsilon: float = 0.2, alpha: float = 0.05,
             min_items_per_annotator: int = 30) -> dict:
    """Can the LLM replace a human annotator?  Discrete-label version of the alt-test.

    ``items``: ``[{"llm": label, "humans": {annotator_id: label, ...}}, ...]``.

    For each annotator j and each item i that j labelled (with >=1 other annotator):
      S(f)   = share of the *other* annotators agreeing with the LLM,
      S(h_j) = share of the *other* annotators agreeing with annotator j,
      W^f = 1[S(f) >= S(h_j)],  W^h = 1[S(h_j) >= S(f)],  d_i = W^f - W^h.
    H0_j: E[d] <= -epsilon (the LLM is worse than j by more than epsilon) is tested with a
    one-sided paired t-test; the p-values are BY-corrected across annotators.
    winning_rate = share of annotators whose H0 is rejected; the test passes if >= 0.5.
    ``advantage_probability`` (mean W^f averaged over annotators) is reported for model comparison.

    epsilon reflects the cost/benefit trade-off; the paper discusses ~0.2 for expensive expert
    annotators and ~0.1 for crowd workers. Pre-register the value; do not tune it after seeing results.
    """
    per_annotator: dict[Hashable, list[tuple[int, int]]] = defaultdict(list)
    for item in items:
        humans = item["humans"]
        if len(humans) < 2:
            continue
        for j, label_j in humans.items():
            others = [v for k, v in humans.items() if k != j]
            s_llm = sum(v == item["llm"] for v in others) / len(others)
            s_h = sum(v == label_j for v in others) / len(others)
            per_annotator[j].append((int(s_llm >= s_h), int(s_h >= s_llm)))
    results, p_values, ids = [], [], []
    for j, rows in sorted(per_annotator.items(), key=lambda kv: repr(kv[0])):
        n = len(rows)
        if n < min_items_per_annotator:
            continue
        d = [wf - wh for wf, wh in rows]
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1)
        rho_f = sum(wf for wf, _ in rows) / n
        rho_h = sum(wh for _, wh in rows) / n
        if var == 0:
            p = 0.0 if mean > -epsilon else 1.0
        else:
            t = (mean + epsilon) / math.sqrt(var / n)
            p = t_sf(t, n - 1)
        ids.append(j)
        p_values.append(p)
        results.append({"annotator": j, "n_items": n, "rho_llm": rho_f, "rho_human": rho_h, "p_value": p})
    reject = benjamini_yekutieli(p_values, alpha)
    for row, r in zip(results, reject):
        row["llm_not_worse_by_epsilon"] = r
    m = len(results)
    winning_rate = (sum(reject) / m) if m else None
    return {"epsilon": epsilon, "alpha": alpha, "eligible_annotators": m,
            "min_items_per_annotator": min_items_per_annotator,
            "winning_rate": winning_rate,
            "passed": (winning_rate is not None and m >= 3 and winning_rate >= 0.5),
            # 'not_evaluable' (too few eligible annotators) is NOT the same finding as 'failed'.
            "status": ("not_evaluable" if m < 3 else
                       ("passed" if winning_rate is not None and winning_rate >= 0.5 else "failed")),
            "advantage_probability": (sum(r["rho_llm"] for r in results) / m) if m else None,
            "per_annotator": results,
            "note": ("Requires >=3 eligible annotators. Passing means 'not worse than a typical annotator "
                     "by more than epsilon on these items', not clinical validity.")}


# ---------------------------------------------------------------------------
# Human reference helpers (physician-physician agreement: a reference point, not an upper bound)
# ---------------------------------------------------------------------------


def pairwise_human_pairs(label_sets: Iterable[Sequence[bool]]) -> list[tuple[bool, bool]]:
    """All ordered physician-physician pairs within each item (symmetric)."""
    pairs = []
    for labels in label_sets:
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j:
                    pairs.append((labels[i], labels[j]))
    return pairs


def leave_one_out_majority_pairs(label_sets: Iterable[Sequence[bool]], min_raters: int = 3) -> list[tuple[bool, bool]]:
    """(held-out physician, strict majority of the others) for items with >= ``min_raters``; ties skipped.

    This is the human reference for the primary judge metric 'judge vs physician majority'.
    """
    pairs = []
    for labels in label_sets:
        if len(labels) < min_raters:
            continue
        for i, own in enumerate(labels):
            others = list(labels[:i]) + list(labels[i + 1:])
            yes = sum(others)
            if yes * 2 == len(others):
                continue
            pairs.append((own, yes * 2 > len(others)))
    return pairs


def majority_label(labels: Sequence[bool]) -> bool | None:
    yes = sum(labels)
    if yes * 2 == len(labels):
        return None
    return yes * 2 > len(labels)


def kappa_standard_error(po: float, pe: float, n: int) -> float | None:
    """Large-sample approximation SE(kappa) ~ sqrt(po(1-po)/n)/(1-pe), for sample-size planning only."""
    if n <= 0 or pe >= 1:
        return None
    return math.sqrt(po * (1 - po) / n) / (1 - pe)
