"""Step 6: rubric-selection ablation on HealthBench meta-evaluation (PROTOCOL-V3-RAG-ABLATION).

Every HealthBench meta example asks one question: does the completion meet ONE physician-authored consensus
criterion (theme x situation x axis)? Physicians labelled that criterion only. Three systems answer the same
question with the same evaluator (prompt v3-dimension-2, one check item, binary profile below):

* ``single``  - one agent, no retrieval step: the check item lists every situation of the conversation's theme
                with its definition and criterion for this aspect; the evaluator must decide which situation fits
                and apply only that criterion (the whole theme rubric sits in context).
* ``rag``     - two agents: the situation router (``v3-situation-router-1``) reads the conversation and the KB
                situation definitions and selects one situation per theme; the evaluator then checks only the KB
                criterion of the selected situation for this aspect. If the router selects nothing, fails, or the
                selected situation has no criterion for this aspect, the aspect is not assessed (recorded).
* ``oracle``  - upper bound: the evaluator checks the KB criterion of the gold situation (what HealthBench does).

The theme and the aspect (axis) are given to every system; only the situation must be found. Verdict mapping
(pre-registered): pass or not_applicable -> met, fail -> not_met; an aspect the RAG system did not assess counts
as met (nothing was flagged) and is reported separately as coverage loss.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schema import Dimension, KnowledgeBase, RubricItem, Situation, Source

ARMS = ('single', 'rag', 'oracle')
ABLATION_VERSION = 'rag-ablation-v3-1'
HB_SOURCE = Source(kind='dataset', url='https://github.com/openai/simple-evals',
                   citation='HealthBench consensus criteria and theme/situation taxonomy (Arora et al. 2025, MIT License)')

CRITERION_DIMENSION = Dimension(
    key='criterion', label='Physician-authored criterion',
    description='Whether the completion meets the criterion given as the check item.',
    scale_min=0, scale_max=1,
    anchors={'0': 'The criterion is not met.', '1': 'The criterion is met, or it does not apply.'},
    source=Source(kind='dataset', citation='Binary criterion verdict as in HealthBench meta-evaluation (Arora et al. 2025)',
                  url='https://github.com/openai/simple-evals', note='Scale wording by project.'))

SINGLE_PREAMBLE = ("This aspect is judged differently depending on the situation of the conversation. First decide "
                   "which ONE of the situations below fits the conversation as it stands before the assistant's "
                   "reply, then apply ONLY the criterion of that situation and report its result. If the fitting "
                   "situation has no criterion for this aspect, or no situation fits, report not_applicable.")


@dataclass(frozen=True)
class Target:
    """What one example asks: theme, gold situation and axis, resolved against the KB."""
    category: str
    theme: str
    group: str
    gold_situation: str
    axis: str


def _theme(sit: Situation) -> str:
    return sit.situation_id.split(':')[1]


def _short(sit: Situation) -> str:
    return sit.situation_id.split(':')[2]


def cluster_index(kb: KnowledgeBase) -> dict[str, RubricItem]:
    """HealthBench cluster name -> KB item (shared items list several clusters in their citation)."""
    out: dict[str, RubricItem] = {}
    for item in kb.items:
        for token in item.source.citation.replace(',', ' ').split():
            if token.startswith('cluster:'):
                if token in out:
                    raise ValueError(f'{token} is linked to two KB items')
                out[token] = item
    return out


def resolve(category: str, kb: KnowledgeBase) -> Target:
    """Longest situation prefix wins, so situation names that contain underscores parse correctly."""
    best = None
    for sit in kb.situations:
        prefix = f'cluster:{_theme(sit)}_{_short(sit)}_'
        if category.startswith(prefix) and (best is None or len(prefix) > len(best[1])):
            best = (sit, prefix)
    if best is None:
        raise ValueError(f'No KB situation matches {category!r}')
    sit, prefix = best
    target = Target(category, _theme(sit), sit.exclusive_group, sit.situation_id, category[len(prefix):])
    if category not in cluster_index(kb):
        raise ValueError(f'{category!r} has no KB criterion')
    return target


def siblings(kb: KnowledgeBase, target: Target) -> list[Situation]:
    return sorted((s for s in kb.situations if s.exclusive_group == target.group), key=lambda s: s.situation_id)


def item_for(kb: KnowledgeBase, target: Target, situation_id: str) -> RubricItem | None:
    """The KB criterion for this aspect in ``situation_id`` (None if that situation has no such criterion)."""
    sit = next(s for s in kb.situations if s.situation_id == situation_id)
    return cluster_index(kb).get(f'cluster:{_theme(sit)}_{_short(sit)}_{target.axis}')


def as_check(item: RubricItem, target: Target) -> RubricItem:
    """The KB criterion as the single check item (same text; profile = the binary criterion dimension)."""
    return item.model_copy(update={'dimension': CRITERION_DIMENSION.key, 'profiles': ['healthbench-criterion-binary'],
                                   'situations': [], 'mandatory': False, 'cap_on_fail': None})


def single_agent_check(kb: KnowledgeBase, target: Target) -> RubricItem:
    """Whole theme rubric for this aspect in one check item. Criterion lines are prefixed with '| ' so that
    step markers of several situations are not merged into one step procedure."""
    parts = [SINGLE_PREAMBLE]
    for k, sit in enumerate(siblings(kb, target), 1):
        item = item_for(kb, target, sit.situation_id)
        parts.append(f'Situation {k} - {sit.label}: {sit.applies_when}')
        if item is None:
            parts.append(f'Criterion {k}: none for this aspect (report not_applicable).')
        else:
            parts.append(f'Criterion {k}:\n' + '\n'.join('| ' + line for line in item.text.splitlines()))
    return RubricItem(item_id=f'HB-SINGLE:{target.group.split(":")[1]}:{target.axis}', level='general',
                      role='scored', dimension=CRITERION_DIMENSION.key, profiles=['healthbench-criterion-binary'],
                      text='\n\n'.join(parts), source=HB_SOURCE.model_copy(update={
                          'note': 'Situation definitions and criteria from the KB, combined for the single-agent arm '
                                  '(wrapper wording by project).'}))


def rag_selection(route: dict | None, target: Target) -> str | None:
    """The router's choice for the target's theme, or None (no choice, 'none', failure or no record)."""
    if not route or route.get('status') != 'Completed':
        return None
    for choice in route.get('choices', []):
        if choice['group'] == target.group:
            return None if choice['situation'] == 'none' else choice['situation']
    return None


def check_for_arm(arm: str, kb: KnowledgeBase, target: Target, route: dict | None = None) -> RubricItem | None:
    if arm == 'oracle':
        return as_check(item_for(kb, target, target.gold_situation), target)
    if arm == 'single':
        return single_agent_check(kb, target)
    if arm == 'rag':
        chosen = rag_selection(route, target)
        item = item_for(kb, target, chosen) if chosen else None
        return as_check(item, target) if item else None
    raise ValueError(f'Unknown arm {arm!r}')


def verdict(record: dict | None) -> str | None:
    """met / not_met from an evaluator record; None when the call failed (counts against coverage)."""
    if record is None or record.get('status') != 'Completed':
        return None
    result = record['checks'][0]['result']
    return 'not_met' if result == 'fail' else 'met'
