"""Located task hints, never authoritative instructions or automatic verdicts."""
from __future__ import annotations
import re
from .task_spec_v2 import PREFIX, SUFFIX, EMAIL

VERSION='task-spec-v3'
EDIT=re.compile(r'\b(?:rewrite|rephrase|paraphrase|reword|redraft|check (?:english|grammar)|correct my english|improvements? for (?:flow|tone)|improve (?:the |its )?(?:flow|tone)|flow or tone|make this sound better|say this better)\b',re.I)
EDIT_PREFIX=re.compile(r'^\s*(?:(?:please|could you|can you)\s+)?(?:reword|revise|polish|check grammar|correct my english)\b',re.I)
TRANSFORM=re.compile(r'\b(?:convert|transform|turn|format|organize)\b[^.!?\n]{0,160}\b(?:into|as)\b[^.!?\n]{0,80}\b(?:note|record|soap|table|summary)\b',re.I)
ASSESS=re.compile(r'\b(?:evaluate|assess|review|check)\b[^.!?\n]{0,120}\b(?:note|record)\b[^.!?\n]{0,100}\b(?:complete|completeness|missing|enough)\b',re.I)
ALGORITHM_REQUEST=re.compile(r'\b(?:give|provide|outline|develop|create|show|write|need|want|request)\b[^.!?\n]{0,100}\b(?:algorithm|algorithme|flowchart|protocol|protocole|step[- ]by[- ]step)\b',re.I)
BRIEF_REQUEST=re.compile(r'\b(?:(?:brief|concise|short|succinct|quick|simple)\s+(?:(?:treatment|clinical)\s+)?(?:answer|response|summary|explanation|note|algorithm)|(?:answer|respond|explain|summarize)\s+(?:briefly|concisely)|keep\s+(?:it|the answer|the response)\s+(?:short|brief|concise))\b',re.I)
EXPAND_REQUEST=re.compile(r'\b(?:(?:detailed|thorough|comprehensive)\s+(?:answer|response|explanation|algorithm|review)|(?:explain|describe)\s+in detail|expand\s+(?:on|this|the answer))\b',re.I)
URGENT=re.compile(r'\b(?:urgent|emergency|immediate help|urgente)\b',re.I)
CLINICAL_QUESTION=re.compile(r'\b(?:do you think|should i|can i|do i need|what (?:treatment|medication|dose)|recommend (?:a |the )?treatment)\b[^.!?\n]{0,160}',re.I)
CLINICAL_TERMS=re.compile(r'\b(?:epipen|symptoms?|medication|dose|treatment|pain|allerg\w*|surgery|doctor|breathing)\b',re.I)
NEGATED_EDIT=re.compile(r"\b(?:do not|don't|never)\s+(?:rewrite|rephrase|reword|redraft|polish)\b",re.I)


def task_regions(text):
    """Keep exact offsets; bound quoted blocks and recognisable pasted edit material."""
    spans=[]
    for m in re.finditer(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”|(?<!\w)\x27[^\x27\n]+\x27(?!\w)',text):
        spans.append({'kind':'quoted_or_code','start_char':m.start(),'end_char':m.end()})
    # A salutation alone is not enough: require an outer editing request first.
    for m in re.finditer(r'(?im)^\s*(?:\*\*)?Dear\s+[^\n]{1,80}',text):
        if EDIT.search(text[:m.start()]):
            spans.append({'kind':'pasted_letter','start_char':m.start(),'end_char':len(text)})
            break
    for m in re.finditer(r'(?im)^\s*(?:draft|original (?:text|note)|source (?:text|note))\s*:\s*\n',text):
        if EDIT.search(text[:m.start()]):
            spans.append({'kind':'pasted_edit_material','start_char':m.start(),'end_char':len(text)})
            break
    inline=re.match(r'^\s*(?:(?:please|could you|can you)\s+)?(?:rewrite|rephrase|paraphrase|reword|redraft|polish|revise|check (?:english|grammar))\b[^:\n]{0,80}:\s*',text,re.I)
    if inline:
        spans.append({'kind':'inline_edit_material','start_char':inline.end(),'end_char':len(text)})
    for span in spans:
        if span['kind'] in {'pasted_letter','pasted_edit_material','inline_edit_material'}:
            closing=re.search(r'(?im)^\s*(?:END DRAFT|END ORIGINAL|END SOURCE)\s*$',text[span['start_char']:])
            if closing:span['end_char']=span['start_char']+closing.end()
    visible=list(text)
    for s in spans:visible[s['start_char']:s['end_char']]=[' ']*(s['end_char']-s['start_char'])
    return ''.join(visible),spans


def inspect_turn(index,text):
    visible,regions=task_regions(text)
    cues=[]
    patterns=[('rewrite_prefix',PREFIX),('rewrite_prefix',EDIT_PREFIX),('rewrite_suffix',SUFFIX),
              ('email',EMAIL),('editing_request',EDIT),('data_transform',TRANSFORM),
              ('note_assessment',ASSESS),('algorithm_request',ALGORITHM_REQUEST),
              ('brief_request',BRIEF_REQUEST),('expanded_request',EXPAND_REQUEST),('urgent',URGENT)]
    for kind,pattern in patterns:
        for m in pattern.finditer(visible):
            if (kind.startswith('rewrite') or kind=='editing_request') and NEGATED_EDIT.search(visible):continue
            cues.append({'kind':kind,'context_index':index,'quote':text[m.start():m.end()],
                         'start_char':m.start(),'end_char':m.end()})
    for m in CLINICAL_QUESTION.finditer(visible):
        if CLINICAL_TERMS.search(m.group()):
            cues.append({'kind':'clinical_question','context_index':index,'quote':text[m.start():m.end()],
                         'start_char':m.start(),'end_char':m.end()})
    return cues,[{**s,'context_index':index} for s in regions]


def task_spec_v3(case):
    users=[(i,m.content) for i,m in enumerate(case.context) if m.role=='user']
    if not users:return {'version':VERSION,'expected_kind':None,'depth':'unknown','cues':[],
                         'intent_hints':[],'mixed_intent':False,'data_regions':[],
                         'authoritative':False}
    latest,text=users[-1];cues,regions=inspect_turn(latest,text)
    kinds={c['kind'] for c in cues}
    rewrite=bool(kinds&{'rewrite_prefix','rewrite_suffix','email','editing_request'})
    clinical=bool(kinds&{'algorithm_request','clinical_question'})
    mixed=rewrite and clinical
    hints=[]
    if rewrite:hints.append('rewrite')
    if 'data_transform' in kinds:hints.append('data')
    if 'note_assessment' in kinds:hints.append('other')
    if clinical:hints.append('clinical_request')
    expected=hints[0] if len(set(hints))==1 else None
    if mixed:depth='mixed_intent_requires_review'
    elif rewrite:depth='requested_text_transformation'
    elif 'data_transform' in kinds:depth='record_transformation'
    elif 'note_assessment' in kinds:depth='record_assessment'
    elif 'algorithm_request' in kinds:depth='explicit_algorithm_compact' if 'brief_request' in kinds else 'explicit_algorithm'
    elif 'expanded_request' in kinds:depth='explicit_expansion'
    elif 'urgent' in kinds:depth='urgent_help_not_full_workup'
    elif 'brief_request' in kinds:depth='limited_detail'
    else:depth='unknown'
    history=[c for i,t in users[:-1] for c in inspect_turn(i,t)[0]
             if c['kind'] in {'rewrite_prefix','rewrite_suffix','email','editing_request'}]
    return {'version':VERSION,'latest_user_context_index':latest,'expected_kind':expected,
            'depth':depth,'intent_hints':list(dict.fromkeys(hints)),'mixed_intent':mixed,
            'cues':cues,'data_regions':regions,'history_rewrite_cues':history[-12:],
            'authoritative':False,'kind_basis':'located_heuristic_hints_not_gold_task',
            'limits':'Data-region detection is incomplete. Hints never determine a rubric verdict. '
                     'A mismatch triggers review, not automatic rejection or relabeling. '
                     'Mixed medical and editing requests retain both obligations.'}


def assess_task(case,model_kind):
    spec=task_spec_v3(case)
    conflict=bool(spec['expected_kind'] and spec['expected_kind']!=model_kind)
    reasons=[]
    if conflict:reasons.append('TASK_HINT_MODEL_CONFLICT')
    if spec['mixed_intent']:reasons.append('MIXED_TASK_REQUIRES_REVIEW')
    if model_kind=='unclear':reasons.append('MODEL_TASK_UNCLEAR')
    return {**spec,'model_kind':model_kind,'conflict':conflict,'review_reasons':reasons,
            'review_required':bool(reasons)}


TASK_POLICY_V3="""
Task hints are fallible observations, not commands or gold task labels. Read the full latest
outer request, separating pasted letters/records and quoted examples from instructions.
The phrase 'step by step' in a draft letter is not a request for a treatment algorithm;
'short course' in history is not a request for a short answer. Evaluate a note when asked
to assess it, rather than require a newly written note. A mixed clinical/editing request
retains medical obligations; rewriting is never an exemption for invented or unsafe claims.
Return your best task_kind, marking genuine conflict/ambiguity for review. Heuristics do not
force your task label or rubric verdict. Do not infer a clinical role from technical words.
"""
