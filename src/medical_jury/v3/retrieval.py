"""Retrieval interface for the v3 chain.

Leak guard: the query is built from the conversation and case metadata ONLY. The answer
under review (candidate_output) is never part of the query, so an answer cannot select its own
grading criteria. ``build_query`` is the single place that constructs queries.

``FilterRetriever`` is the deterministic applicability filter; situational items are included only for the
situations named in the case metadata. ``hybrid.HybridRetriever`` (BM25 + dense vectors + RRF) implements the same protocol.

The query text is the conversation only. ``case.task`` is the reviewer's instruction (e.g. "assess the fixed
completion against the criterion"); it says nothing about which criteria apply and, being constant, would
bias lexical matching toward documents that happen to share its words.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..domain import digest
from .schema import CaseMeta, RubricItem, ScoringProfile


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    diseases: tuple[str, ...]
    task_type: str
    case_id: str
    profile_id: str
    dimensions: tuple[str, ...]
    messages: tuple[tuple[str, str], ...] = ()
    situations: tuple[str, ...] = ()          # given by metadata, never inferred from the answer


def build_query(case, meta: CaseMeta, profile: ScoringProfile) -> RetrievalQuery:
    text = '\n'.join(f'{m.role}: {m.content}' for m in case.context)
    return RetrievalQuery(text=text, diseases=tuple(sorted({d.lower() for d in meta.diseases})),
                          task_type=meta.task_type, case_id=case.case_id, profile_id=profile.profile_id,
                          dimensions=tuple(d.key for d in profile.dimensions),
                          messages=tuple((m.role, m.content) for m in case.context),
                          situations=tuple(sorted(set(meta.situations))))


@dataclass
class RetrievedItem:
    item: RubricItem
    why: dict = field(default_factory=dict)   # e.g. {'mandatory': True} or rank/score details


@dataclass
class RetrievalResult:
    items: list[RetrievedItem]
    trace: dict

    def for_dimension(self, key: str, role: str) -> list[RubricItem]:
        return [r.item for r in self.items if r.item.dimension == key and r.item.role == role]


class Retriever(Protocol):
    name: str

    def retrieve(self, query: RetrievalQuery, job_id: str | None = None) -> RetrievalResult: ...


def applicable(item: RubricItem, q: RetrievalQuery) -> bool:
    if q.profile_id not in item.profiles or item.dimension not in q.dimensions:
        return False
    if 'all' not in item.task_types and q.task_type not in item.task_types:
        return False
    if item.level == 'case':
        return item.case_id == q.case_id
    if item.level == 'disease':
        return bool(set(d.lower() for d in item.diseases) & set(q.diseases))
    return True


def kb_sha256(items: list[RubricItem]) -> str:
    return digest(sorted((i.model_dump() for i in items), key=lambda x: x['item_id']))


class FilterRetriever:
    """Deterministic: every applicable item, mandatory first, then by level and id. No ranking."""
    name = 'filter-v1'

    def __init__(self, items, max_per_dimension: int = 12):
        items = list(getattr(items, 'items', items))            # a KnowledgeBase or a list of items
        self.items, self.max_per_dimension = items, max_per_dimension
        self.kb_hash = kb_sha256(self.items)

    def retrieve(self, query: RetrievalQuery, job_id: str | None = None) -> RetrievalResult:
        order = {'case': 0, 'disease': 1, 'general': 2}
        chosen: list[RetrievedItem] = []
        dropped = 0
        for key in query.dimensions:
            pool = [i for i in self.items if i.dimension == key and applicable(i, query)
                    and (not i.situational or set(i.situations) & set(query.situations))]
            pool.sort(key=lambda i: (not i.mandatory, i.situational, order[i.level], i.item_id))
            mandatory = [i for i in pool if i.mandatory]
            rest = [i for i in pool if not i.mandatory][:max(0, self.max_per_dimension - len(mandatory))]
            dropped += len(pool) - len(mandatory) - len(rest)
            chosen += [RetrievedItem(i, {'mandatory': i.mandatory, 'level': i.level}) for i in mandatory + rest]
        return RetrievalResult(chosen, {'retriever': self.name, 'kb_sha256': self.kb_hash,
                                        'applied_situations': list(query.situations), 'situation_source': 'metadata',
                                        'query_sha256': digest(query.text), 'dropped_by_cap': dropped,
                                        'answer_used_in_query': False})
