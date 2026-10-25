"""Model-selectable source IDs. Exact text/offsets are reconstructed locally.

No fuzzy repair, list truncation, deduplication or judgment changes are permitted.
"""
from __future__ import annotations

import hashlib
from pydantic import BaseModel, ConfigDict, Field
from .domain import Judgment

VERSION = 'evidence-ids-v1'
CHUNK_CHARS = 480


class EvidenceIdError(ValueError):
    def __init__(self, kind, field='', value=''):
        super().__init__('Invalid bounded source selection.')
        self.issue = {'kind':kind, 'field':field, 'id':str(value)[:80]}


def segments(text):
    """Lossless contiguous spans, no normalization or dropped whitespace."""
    start = 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            # Prefer paragraph/sentence endings, then a word boundary.
            window = text[start:end]
            choices = [window.rfind(p) + len(p) for p in ('\n\n', '. ', '? ', '! ', '\n', ' ')]
            split = next((x for x in choices if x >= CHUNK_CHARS // 2), None)
            if split is not None:
                end = start + split
        yield start, end
        start = end


def catalog(case):
    rows = []
    for kind, index, text in [('answer',0,case.candidate_output)] + [
        ('context',i,m.content) for i,m in enumerate(case.context)
    ]:
        sha = hashlib.sha256(text.encode()).hexdigest()
        prefix = ('A' if kind=='answer' else f'C{index:02d}') + sha[:6]
        for number,(start,end) in enumerate(segments(text)):
            rows.append({'id':f'{prefix}-{number:03d}', 'kind':kind, 'source_index':index,
                         'text':text[start:end], 'start_char':start, 'end_char':end,
                         'start_utf16':len(text[:start].encode('utf-16-le'))//2,
                         'end_utf16':len(text[:end].encode('utf-16-le'))//2,
                         'source_sha256':sha})
    if len(rows)>600:
        raise EvidenceIdError('catalog_too_large')
    return rows


def payload_sources(case):
    rows=catalog(case)
    return {
        'candidate_segments':[{'id':r['id'],'text':r['text']} for r in rows if r['kind']=='answer'],
        'context':[{'role':m.role,'segments':[{'id':r['id'],'text':r['text']} for r in rows
                                            if r['kind']=='context' and r['source_index']==i]}
                   for i,m in enumerate(case.context)]
    }


class WireIds(BaseModel):
    model_config=ConfigDict(extra='forbid',strict=True,allow_inf_nan=False)
    judgment: str = Field(pattern=r'^(met|not_met|abstain)$')
    reason: str = Field(min_length=1,max_length=2000)
    answer_ids: list[str] = Field(max_length=3)
    context_ids: list[str] = Field(max_length=2)
    review_required: bool


def resolve_ids(wire, case):
    lookup={r['id']:r for r in catalog(case)}
    updates={};trace=[]
    for field,kind,target in [('answer_ids','answer','answer_evidence'),('context_ids','context','context_evidence')]:
        ids=getattr(wire,field)
        if len(ids)!=len(set(ids)):
            raise EvidenceIdError('duplicate_id',field)
        resolved=[]
        for selected in ids:
            row=lookup.get(selected)
            if row is None:
                raise EvidenceIdError('unknown_id',field,selected)
            if row['kind']!=kind:
                raise EvidenceIdError('wrong_source_kind',field,selected)
            source=case.candidate_output if kind=='answer' else case.context[row['source_index']].content
            if source[row['start_char']:row['end_char']]!=row['text'] or hashlib.sha256(source.encode()).hexdigest()!=row['source_sha256']:
                raise EvidenceIdError('source_integrity',field,selected)
            resolved.append(row['text'])
            trace.append({**{k:v for k,v in row.items() if k not in {'text','kind'}},
                          'field':target,'quote_index':len(resolved)-1,'source_quote':row['text'],
                          'method':'provided-source-id','resolution':'explicit_id',
                          'selection_not_semantic_verification':True})
        updates[target]=resolved
    result=Judgment(judgment=wire.judgment,reason=wire.reason,review_required=wire.review_required,**updates)
    return result,{'policy':VERSION,'spans':trace,'ambiguous_count':0,'normalized_count':0,
                   'scope':'Exact source selection, NOT claim correctness or evidence entailment.'}
