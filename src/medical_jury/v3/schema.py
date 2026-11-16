"""Data contracts for pipeline v3 (new models; the legacy domain.Rubric is not modified).

Key rules encoded here (not just documented):
* A dimension's scale and anchor descriptions are defined by its ScoringProfile.
* Every rubric item records where it came from (Source) - provenance is mandatory.
* Disease-level items are REFERENCE notes unless a named reviewer approved them for scoring
  (decision 2026-10-25: unreviewed disease guidance informs the judge but is never scored).
* Reference notes cannot carry scoring side effects (mandatory flags, score caps).
* Situational items (step 2): an item linked to ``situations`` applies only when the retriever selects one
  of those situations for the conversation. Situations in the same ``exclusive_group`` are mutually
  exclusive (e.g. emergent vs non-emergent), so at most one of them is selected per case.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..domain import CaseInput, digest

Level = Literal['general', 'disease', 'case']
Role = Literal['scored', 'reference']
SourceKind = Literal['paper', 'guideline', 'dataset', 'platform', 'project_draft']


class Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class Source(Strict):
    kind: SourceKind
    citation: str = Field(min_length=3, max_length=500)
    url: str | None = Field(default=None, max_length=500)
    note: str = Field(default='', max_length=1000)


class Dimension(Strict):
    key: str = Field(pattern=r'^[a-z][a-z0-9_]{1,40}$')
    label: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=1000)
    scale_min: int = Field(ge=0, le=10)
    scale_max: int = Field(ge=1, le=10)
    anchors: dict[str, str]
    source: Source

    @model_validator(mode='after')
    def _scale(self):
        if self.scale_max <= self.scale_min:
            raise ValueError('scale_max must exceed scale_min')
        expected = {str(v) for v in range(self.scale_min, self.scale_max + 1)}
        if set(self.anchors) != expected:
            raise ValueError(f'anchors must describe every scale point {sorted(expected)}')
        return self

    def valid(self, score: int) -> bool:
        return type(score) is int and self.scale_min <= score <= self.scale_max


class ScoringProfile(Strict):
    profile_id: str = Field(pattern=r'^[a-z0-9][a-z0-9_.-]{2,60}$')
    version: str
    description: str
    dimensions: list[Dimension] = Field(min_length=1, max_length=12)

    @model_validator(mode='after')
    def _unique(self):
        keys = [d.key for d in self.dimensions]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate dimension keys')
        return self

    def dimension(self, key: str) -> Dimension:
        return next(d for d in self.dimensions if d.key == key)

    def sha256(self) -> str:
        return digest(self.model_dump())


class RubricItem(Strict):
    item_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{2,120}$')
    level: Level
    role: Role
    dimension: str = Field(pattern=r'^[a-z][a-z0-9_]{1,40}$')
    profiles: list[str] = Field(min_length=1, description='Profiles this item belongs to.')
    text: str = Field(min_length=3, max_length=8000)   # 8000: step-6 single-agent arm combines a theme rubric
    source: Source
    diseases: list[str] = Field(default_factory=list, description='Normalised disease tags (disease level).')
    task_types: list[str] = Field(default_factory=lambda: ['all'])
    case_id: str | None = None
    mandatory: bool = False
    cap_on_fail: int | None = Field(default=None, description='If this check fails, the dimension score is capped here.')
    approved_by: str | None = Field(default=None, max_length=120)
    version: str = '1'
    situations: list[str] = Field(default_factory=list,
                                  description='If set, the item applies only when one of these situations is selected.')

    @property
    def situational(self) -> bool:
        return bool(self.situations)

    @model_validator(mode='after')
    def _rules(self):
        if self.situations and self.mandatory:
            raise ValueError('mandatory items always apply; they cannot be situational')
        if self.situations and self.level != 'general':
            raise ValueError('only general-level items can be situational (disease/case use metadata filters)')
        if self.level == 'case' and not self.case_id:
            raise ValueError('case-level items must name their case_id')
        if self.level != 'case' and self.case_id:
            raise ValueError('only case-level items may be bound to a case')
        if self.level == 'disease' and not self.diseases:
            raise ValueError('disease-level items need at least one disease tag')
        if self.level == 'disease' and self.role == 'scored' and not self.approved_by:
            raise ValueError('disease-level items are reference-only until a named reviewer approves them')
        if self.role == 'reference' and (self.mandatory or self.cap_on_fail is not None):
            raise ValueError('reference notes cannot be mandatory or cap scores')
        return self


class Situation(Strict):
    """A retrievable applicability unit: WHEN a group of criteria applies (not what they require)."""
    situation_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{2,120}$')
    exclusive_group: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]{1,80}$')
    label: str = Field(min_length=3, max_length=200)
    applies_when: str = Field(min_length=20, max_length=2000)
    source: Source


class KnowledgeBase(Strict):
    items: list[RubricItem]
    situations: list[Situation] = Field(default_factory=list)

    @model_validator(mode='after')
    def _links(self):
        ids = [i.item_id for i in self.items]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate item_id')
        sids = [s.situation_id for s in self.situations]
        if len(sids) != len(set(sids)):
            raise ValueError('duplicate situation_id')
        known = set(sids)
        for item in self.items:
            missing = set(item.situations) - known
            if missing:
                raise ValueError(f'{item.item_id}: unknown situations {sorted(missing)}')
        used = {s for i in self.items for s in i.situations}
        if known - used:
            raise ValueError(f'situations without any item: {sorted(known - used)}')
        return self

    def sha256(self) -> str:
        return digest({'items': sorted((i.model_dump() for i in self.items), key=lambda x: x['item_id']),
                       'situations': sorted((s.model_dump() for s in self.situations),
                                            key=lambda x: x['situation_id'])})


class CaseMeta(Strict):
    diseases: list[str] = Field(default_factory=list)
    situations: list[str] = Field(default_factory=list,
                                  description='Situation ids supplied by the data source or task spec (e.g. a HealthBench '
                                              'category). Situational criteria apply only for these by default.')
    task_type: str = 'medical_dialogue'
    language: str | None = None
    data_source: str = 'unspecified'


class ReviewRequest(Strict):
    case: CaseInput
    meta: CaseMeta = Field(default_factory=CaseMeta)
    profile_id: str


def load_items(path: Path) -> list[RubricItem]:
    items = []
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    items.append(RubricItem.model_validate_json(line))
                except Exception as error:
                    raise ValueError(f'{path.name}:{n}: {error}') from None
    ids = [i.item_id for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f'{path.name}: duplicate item_id')
    return items


def _jsonl(path: Path, model):
    rows = []
    with path.open(encoding='utf-8') as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(model.model_validate_json(line))
                except Exception as error:
                    raise ValueError(f'{path.name}:{n}: {error}') from None
    return rows


def load_kb(path: Path) -> KnowledgeBase:
    """A KB is a single items .jsonl file, or a directory of ``*items*.jsonl`` + ``situations*.jsonl`` files."""
    if path.is_file():
        return KnowledgeBase(items=load_items(path))
    items, situations = [], []
    for f in sorted(path.glob('*.jsonl')):
        if f.name.startswith('situations'):
            situations += _jsonl(f, Situation)
        elif 'items' in f.name:
            items += _jsonl(f, RubricItem)
    if not items:
        raise ValueError(f'{path}: no *items*.jsonl files')
    return KnowledgeBase(items=items, situations=situations)


def load_profile(path: Path) -> ScoringProfile:
    return ScoringProfile.model_validate(json.loads(path.read_text(encoding='utf-8')))
