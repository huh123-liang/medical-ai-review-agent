"""Outcome-blind sampling with natural distribution.

Iteration 15 drew a fixed 2 groups per theme, which distorted the label prevalence
(37% met in the sample vs ~77% met overall) so that 'always not_met' beat the judge.
Here selection never reads labels, predictions or answer quality. Representativeness
against labels is checked only AFTER the selection manifest is frozen, by the scorer.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Iterable

from ..domain import digest

TOPICS = ('emergency_referrals', 'communication', 'context_seeking', 'health_data_tasks',
          'hedging', 'global_health', 'complex_responses')


def topic_of(category: str) -> str:
    for t in TOPICS:
        if category.startswith('cluster:' + t + '_') or category == 'cluster:' + t:
            return t
    return 'other'


def _distribution(values: Iterable[str]) -> dict:
    counter = Counter(values)
    total = sum(counter.values())
    return {k: round(v / total, 4) for k, v in sorted(counter.items())} if total else {}


def natural_sample(rows: list[dict], n_groups: int, seed: int, split: str,
                   exclude_groups: Iterable[str] = (), items_per_group: int = 1) -> tuple[list[dict], dict]:
    """Uniformly sample conversation groups, then ``items_per_group`` criteria per group.

    Uniform over groups keeps the natural theme mix (no per-theme quota). Rows must be
    prepared public rows ({example_id, case, rubric, category}); labels are never accepted.
    Returns (selected_rows, manifest). The manifest hash is what gets frozen/pre-registered.
    """
    if split not in {'dev', 'validation', 'test'}:
        raise ValueError('Unknown split.')
    if n_groups < 1 or items_per_group < 1:
        raise ValueError('n_groups and items_per_group must be positive.')
    for row in rows[:1]:
        if any(k in row for k in ('binary_labels', 'labels', 'physician_labels')):
            raise ValueError('Sampling must be outcome-blind: labels are not accepted.')
    excluded = set(exclude_groups)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        case = row['case']
        if case['split'] != split or case['group_id'] in excluded:
            continue
        by_group[case['group_id']].append(row)
    keys = sorted(by_group)
    if len(keys) < n_groups:
        raise ValueError(f'INSUFFICIENT_GROUPS: requested {n_groups}, available {len(keys)}.')
    rng = random.Random(seed)
    chosen_groups = rng.sample(keys, n_groups)
    selected = []
    for group in chosen_groups:
        members = sorted(by_group[group], key=lambda r: r['example_id'])
        selected.extend(rng.sample(members, min(items_per_group, len(members))))
    population_rows = [r for k in keys for r in by_group[k]]
    manifest = {
        'sampler': 'natural-uniform-group-v1', 'seed': seed, 'split': split,
        'n_groups': n_groups, 'items_per_group': items_per_group,
        'excluded_groups': len(excluded),
        **({'excluded_groups_sha256': digest(sorted(excluded))} if excluded else {}),
        'example_ids': [r['example_id'] for r in selected],
        'group_ids': chosen_groups,
        'topic_distribution_sample': _distribution(topic_of(r['category']) for r in selected),
        'topic_distribution_population_items': _distribution(topic_of(r['category']) for r in population_rows),
        'outcome_blind': True,
        'note': 'Selection used only split/group/example ids and seed; no labels, outputs or answer quality.'}
    manifest['manifest_sha256'] = digest(manifest)
    return selected, manifest


def panel_sample(rater_rows: list[dict], n_annotators: int, items_per_annotator: int, seed: int,
                 split: str, exclude_groups: Iterable[str] = (), max_items_per_group: int = 1) -> tuple[list[str], dict]:
    """Annotator-panel sample for the Alternative Annotator Test.

    The alt-test needs every compared physician to have >= ~30 judged items. A natural
    sample of a few hundred items spreads ~160 physicians too thinly, so this sampler picks
    the ``n_annotators`` physicians with the most items in ``split`` (by count only) and draws
    ``items_per_annotator`` items each, at most ``max_items_per_group`` items per conversation group
    (physicians rate many criteria inside the same conversations, so 1/group is often infeasible;
    the alt-test t-test then treats correlated items as independent -> report this limitation).

    ``rater_rows`` are rater-index rows with labels REMOVED ({example_id, group_id, split,
    category, physician_ids}); passing labels raises. Returns (example_ids, manifest).
    """
    if any('labels' in r for r in rater_rows[:1]):
        raise ValueError('Panel sampling must be outcome-blind: strip labels first.')
    excluded = set(exclude_groups)
    pool = [r for r in rater_rows if r['split'] == split and r['group_id'] not in excluded]
    counts = Counter(p for r in pool for p in r['physician_ids'])
    panel = [p for p, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n_annotators]]
    rng = random.Random(seed)
    group_use: Counter = Counter()
    chosen_set: set[str] = set()
    chosen: list[str] = []
    shortfall = {}
    for physician in panel:
        candidates = sorted((r for r in pool if physician in r['physician_ids']), key=lambda r: r['example_id'])
        rng.shuffle(candidates)
        taken = 0
        for r in candidates:
            if taken >= items_per_annotator:
                break
            if group_use[r['group_id']] >= max_items_per_group or r['example_id'] in chosen_set:
                continue
            group_use[r['group_id']] += 1
            chosen_set.add(r['example_id'])
            chosen.append(r['example_id'])
            taken += 1
        if taken < items_per_annotator:
            shortfall[physician[:12]] = taken
    manifest = {'sampler': 'annotator-panel-v1', 'seed': seed, 'split': split,
                'n_annotators': len(panel), 'items_per_annotator': items_per_annotator,
                'max_items_per_group': max_items_per_group,
                'panel_ids_sha256': digest(panel), 'example_ids': chosen, 'shortfall': shortfall,
                'topic_distribution_sample': _distribution(
                    topic_of(r['category']) for r in pool if r['example_id'] in set(chosen)),
                'outcome_blind': True,
                'note': ('Physicians chosen by item count only. Panel items over-represent these physicians; '
                         'report natural-sample metrics separately.')}
    manifest['manifest_sha256'] = digest(manifest)
    return chosen, manifest


def representativeness(selected_ids: Iterable[str], labels_by_id: dict[str, list[bool]],
                       population_ids: Iterable[str]) -> dict:
    """Post-freeze check: compare physician met-rate in the sample and its population."""
    def rate(ids):
        values = [v for i in ids for v in labels_by_id.get(i, [])]
        return (sum(values) / len(values)) if values else None
    sample = list(selected_ids)
    sample_rate, population_rate = rate(sample), rate(population_ids)
    return {'sample_label_met_rate': sample_rate, 'population_label_met_rate': population_rate,
            'absolute_difference': (abs(sample_rate - population_rate)
                                    if sample_rate is not None and population_rate is not None else None),
            'note': 'Computed after the selection manifest was frozen; never used to re-select.'}
