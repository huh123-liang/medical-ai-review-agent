"""Deterministic, uniquely anchored evidence; never fuzzy or semantic matching."""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass

from .domain import Judgment


POLICY_VERSION='anchored-v1'
TYPOGRAPHY=str.maketrans({'\u2018':"'",'\u2019':"'",'\u201c':'"','\u201d':'"'})
# Remove balanced inline bold markers only. Not single asterisks, links, code, math, dashes or units.
BOLD=re.compile(r'(?<![\w*])(\*\*)(?=\S)([^\r\n]+?)(?<=\S)\1(?!\*)|(?<![\w_])(__)(?=\S)([^\r\n]+?)(?<=\S)\3(?!_)')


class EvidenceResolutionError(ValueError):
    def __init__(self,kind,field,index,candidates=0):
        super().__init__('Evidence cannot be uniquely anchored.')
        self.issue={'kind':kind,'field':field,'index':index,'candidates':candidates}


@dataclass(frozen=True)
class View:
    text:str
    starts:list[int]
    ends:list[int]


def normalized_view(text:str)->View:
    skipped=set()
    protected=set()
    for code in re.finditer(r'(`{3,})[\s\S]*?\1|`[^`\r\n]*`',text):
        protected.update(range(code.start(),code.end()))
    for match in BOLD.finditer(text):
        if any(i in protected for i in range(match.start(),match.end())):
            continue
        opening=match.start(1) if match.group(1) else match.start(3)
        skipped.update((opening,opening+1,match.end()-2,match.end()-1))
    chars=[]
    starts=[]
    ends=[]
    i=0
    while i<len(text):
        if i in skipped:
            i+=1
            continue
        end=i+1
        while end<len(text) and unicodedata.combining(text[end]) and end not in skipped:
            end+=1
        piece=text[i:end] if i in protected else unicodedata.normalize('NFC',text[i:end]).translate(TYPOGRAPHY)
        for char in piece:
            if char.isspace() and i not in protected:
                if chars and chars[-1]==' ':
                    ends[-1]=end
                    continue
                char=' '
            chars.append(char)
            starts.append(i)
            ends.append(end)
        i=end
    return View(''.join(chars),starts,ends)


def occurrences(text,quote):
    start=0
    while quote:
        found=text.find(quote,start)
        if found<0:
            return
        end=found+len(quote)
        # Do not quote the tail of a larger/signed decimal number as another dose/value.
        left_ok=not (quote[0].isdigit() and found and text[found-1] in '0123456789.+-−')
        right_ok=not (quote[-1].isdigit() and end<len(text) and text[end].isdigit())
        if left_ok and right_ok:
            yield found
        start=found+1


def locate(quote,sources,field,index,allow_ambiguous=False):
    if not quote or not quote.strip():
        raise EvidenceResolutionError('empty_quote',field,index)
    exact=[(source_index,start,start+len(quote),'exact') for source_index,source in sources
           for start in occurrences(source,quote)]
    matches=exact
    if not exact:
        query=normalized_view(quote).text
        if len(query.strip())<12:
            raise EvidenceResolutionError('short_nonexact_quote',field,index)
        matches=[]
        for source_index,source in sources:
            view=normalized_view(source)
            for position in occurrences(view.text,query):
                matches.append((source_index,view.starts[position],view.ends[position+len(query)-1],POLICY_VERSION))
    if len(matches)>1 and allow_ambiguous:
        if len(matches)>32:
            raise EvidenceResolutionError('too_many_candidate_locations',field,index,len(matches))
        candidates=[make_span(quote,sources,field,index,match) for match in matches]
        original_texts={span['source_quote'] for span in candidates}
        # Returning a common text value is not choosing one location. If source texts differ,
        # normalization does not supply enough information to repair the quotation safely.
        if len(original_texts)!=1:
            raise EvidenceResolutionError('ambiguous_normalized_source_text',field,index,len(matches))
        return {'field':field,'quote_index':index,'model_quote':quote,'source_quote':original_texts.pop(),
                'method':'multiple-exact' if exact else 'multiple-normalized','resolution':'ambiguous',
                'candidate_count':len(candidates),'candidates':candidates}
    if len(matches)!=1:
        raise EvidenceResolutionError('ambiguous_quote' if matches else 'unlocatable_quote',field,index,len(matches))
    return make_span(quote,sources,field,index,matches[0])


def make_span(quote,sources,field,index,match):
    source_index,start,end,method=match
    source=dict(sources)[source_index]
    source_quote=source[start:end]
    return {'field':field,'quote_index':index,'source_index':source_index,'method':method,
            'model_quote':quote,'source_quote':source_quote,'start_char':start,'end_char':end,
            'start_utf16':len(source[:start].encode('utf-16-le'))//2,
            'end_utf16':len(source[:end].encode('utf-16-le'))//2,
            'source_sha256':hashlib.sha256(source.encode('utf-8')).hexdigest()}


def resolve_judgment(judgment:Judgment,case,policy=POLICY_VERSION):
    if policy not in {'anchored-v1','anchored-v2'}:
        raise ValueError('Unknown anchor policy.')
    updates={}
    trace=[]
    for field,sources in [('answer_evidence',[(0,case.candidate_output)]),
                          ('context_evidence',list(enumerate(m.content for m in case.context)))]:
        field_trace=[locate(q,sources,field,index,allow_ambiguous=policy=='anchored-v2') for index,q in enumerate(getattr(judgment,field))]
        updates[field]=[span['source_quote'] for span in field_trace]
        trace.extend(field_trace)
    # Judgment and reason are never changed. Normalized anchors require human review.
    updates['review_required']=judgment.review_required or any(s['method']!='exact' for s in trace)
    result={'policy':policy,'spans':trace,
            'normalized_count':sum(s['method'] in {POLICY_VERSION,'multiple-normalized'} for s in trace)}
    if policy=='anchored-v2':
        result.update(trace_schema='evidence-trace-v2',ambiguous_count=sum(s.get('resolution')=='ambiguous' for s in trace))
    return judgment.model_copy(update=updates),result
