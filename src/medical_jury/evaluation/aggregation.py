"""Combine several judges' binary votes: majority vote and Dawid-Skene (unsupervised EM).

Dawid & Skene (1979) estimate each annotator's sensitivity/specificity and the class prior
from the votes alone, then weight annotators accordingly. No physician labels are used, so the
model may be FIT on any split's votes; its parameters are frozen before validation scoring.
"""
from __future__ import annotations

import math
from typing import Hashable

EPS = 1e-6


def majority(votes: dict[Hashable, bool | None]) -> bool | None:
    vals = [v for v in votes.values() if v is not None]
    if not vals:
        return None
    yes = sum(vals)
    if yes * 2 == len(vals):
        return None  # tie -> undecided, routed to human review
    return yes * 2 > len(vals)


def _clip(p: float) -> float:
    return min(1 - EPS, max(EPS, p))


def dawid_skene(items: list[dict[Hashable, bool | None]], max_iter: int = 200, tol: float = 1e-8,
                smoothing: float = 1.0) -> dict:
    """items: one dict per item {judge: vote or None}. Returns params + per-item P(met).

    ``smoothing`` adds Beta(s, s) pseudo-counts so a judge with few votes cannot reach 0/1 rates.
    """
    judges = sorted({j for it in items for j in it}, key=repr)
    if not items or not judges:
        raise ValueError('No votes to aggregate.')
    # init: soft majority
    post = []
    for it in items:
        vals = [v for v in it.values() if v is not None]
        post.append((sum(vals) + 0.5) / (len(vals) + 1) if vals else 0.5)
    params: dict = {}
    prior = 0.5
    previous = None
    for iteration in range(max_iter):
        prior = _clip(sum(post) / len(post))
        params = {}
        for j in judges:
            tp = fn = tn = fp = 0.0
            for it, q in zip(items, post):
                v = it.get(j)
                if v is None:
                    continue
                if v:
                    tp += q
                    fp += 1 - q
                else:
                    fn += q
                    tn += 1 - q
            sens = _clip((tp + smoothing) / (tp + fn + 2 * smoothing))
            spec = _clip((tn + smoothing) / (tn + fp + 2 * smoothing))
            params[j] = {'sensitivity': sens, 'specificity': spec}
        post = [posterior(it, params, prior) for it in items]
        ll = sum(_log_marginal(it, params, prior) for it in items)
        if previous is not None and abs(ll - previous) < tol:
            break
        previous = ll
    return {'prior_met': prior, 'judges': params, 'posteriors': post, 'iterations': iteration + 1,
            'log_likelihood': previous,
            'note': 'Unsupervised; identifiability assumes judges are better than chance on average.'}


def _log_terms(votes, params, prior):
    lm, ln = math.log(prior), math.log(1 - prior)
    for j, v in votes.items():
        if v is None or j not in params:
            continue
        s, c = params[j]['sensitivity'], params[j]['specificity']
        lm += math.log(s if v else 1 - s)
        ln += math.log(1 - c if v else c)
    return lm, ln


def posterior(votes: dict[Hashable, bool | None], params: dict, prior: float) -> float:
    """P(met | votes) under frozen Dawid-Skene parameters (used online by the jury workflow)."""
    lm, ln = _log_terms(votes, params, prior)
    m = max(lm, ln)
    return math.exp(lm - m) / (math.exp(lm - m) + math.exp(ln - m))


def _log_marginal(votes, params, prior):
    lm, ln = _log_terms(votes, params, prior)
    m = max(lm, ln)
    return m + math.log(math.exp(lm - m) + math.exp(ln - m))
