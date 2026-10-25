"""Conservative, source-located instruction cues. No candidate or label access."""
from __future__ import annotations
import re

VERSION='task-spec-v2'
PREFIX=re.compile(r'^\s*(?:(?:please|could you|can you)\s+)?(?:rewrite|rephrase|redraft|shorter|check english|corrig(?:e|er|é)|reformul(?:e|er)|melhorar|revised below|make this sound better|say this better)\b',re.I)
SUFFIX=re.compile(r'\b(?:rewrite|rephrase|paraphrase)(?:\s+(?:it|this|the line|the sentence))?(?:\s+in one sentence)?\s*[.!]?\s*$',re.I)
EMAIL=re.compile(r'^\s*(?:(?:please|could you|can you)\s+)?(?:(?:write|draft|improve|compose)\s+(?:(?:me|an?|my|the)\s+){0,3}email|an email to)\b',re.I)
ALGORITHM=re.compile(r'\b(?:algorithm|algorithme|flowchart|protocol|protocole|schéma posologique|step.by.step)\b',re.I)
EXPAND=re.compile(r'\b(?:detailed|thorough|complete|comprehensive|expand|profundizar|detallas|précise|complet)\b',re.I)
BRIEF=re.compile(r'\b(?:brief|concise|short|succinct|quick|simple|rapide|résumé)\b',re.I)
URGENT=re.compile(r'\b(?:urgent|emergency|immediate|urgente)\b',re.I)
NEGATED=re.compile(r'^\s*(?:please\s+)?(?:do not|don\x27t|never|not)\s+(?:rewrite|rephrase|redraft)\b',re.I)


class TaskScopeError(ValueError):
    pass


def mask_quoted(text):
    # Equal-length masking retains original offsets. Apostrophes in words are not quotes.
    result=list(text)
    for match in re.finditer(r'```[\s\S]*?```|`[^`\n]*`|"[^"\n]*"|“[^”\n]*”',text):
        result[match.start():match.end()]=[' ']*(match.end()-match.start())
    return ''.join(result)


def inspect_turn(index,text):
    visible=mask_quoted(text)
    cues=[]
    for kind,pattern in [('rewrite_prefix',PREFIX),('rewrite_suffix',SUFFIX),('email',EMAIL),
                         ('algorithm',ALGORITHM),('expanded',EXPAND),('brief',BRIEF),('urgent',URGENT)]:
        for match in pattern.finditer(visible):
            if kind.startswith('rewrite') and NEGATED.search(visible):continue
            cues.append({'kind':kind,'context_index':index,'quote':text[match.start():match.end()],
                         'start_char':match.start(),'end_char':match.end()})
    return cues


def task_spec_v2(case):
    users=[(i,m.content) for i,m in enumerate(case.context) if m.role=='user']
    if not users:
        return {'version':VERSION,'expected_kind':None,'depth':'unknown','cues':[]}
    latest,text=users[-1]
    cues=inspect_turn(latest,text)
    kinds={c['kind'] for c in cues}
    rewrite=bool(kinds&{'rewrite_prefix','rewrite_suffix','email'})
    expected='rewrite' if rewrite else ('clinical_request' if 'algorithm' in kinds else None)
    if rewrite:depth='requested_text_transformation'
    elif 'algorithm' in kinds:depth='explicit_algorithm_compact' if 'brief' in kinds else 'explicit_algorithm'
    elif 'expanded' in kinds:depth='explicit_expansion'
    elif 'urgent' in kinds:depth='urgent_help_not_full_workup'
    elif 'brief' in kinds:depth='limited_detail'
    else:depth='unknown'
    history=[c for i,t in users[:-1] for c in inspect_turn(i,t) if c['kind'].startswith('rewrite') or c['kind']=='email']
    return {'version':VERSION,'latest_user_context_index':latest,'expected_kind':expected,
            'depth':depth,'cues':cues,'history_rewrite_cues':history[-12:],
            'kind_basis':'explicit_outer_command' if expected else 'full_context_interpretation_needed',
            'limits':'Quoted commands are masked; heuristic cues are not clinical facts. Latest task overrides earlier rewriting. '
                     'Mentioning a doctor does not establish user profession. No task kind determines a rubric verdict.'}


def validate_task_kind(case, kind):
    spec=task_spec_v2(case)
    expected=spec.get('expected_kind')
    if expected and kind!=expected:
        raise TaskScopeError('Explicit outer task and returned task_kind disagree.')
    return spec


TASK_POLICY="""
First identify the outer task from task_spec and the full conversation. Explicit rewrite commands
can occur at the END of a turn. Rewriting a question is not answering that embedded question.
Earlier rewriting does not override a later explicit request for a treatment algorithm.
Return task_kind matching a clear outer command; ambiguous intent must be explained with review_required.
Do not grade completeness when the supplied rubric only grades factuality. Rewriting is NOT an
exemption for incorrect medical statements or invented facts. A question/hypothesis is not a confirmed
clinical assertion; conditions like 'if needed' are not automatically unnecessary hedging.
Requested algorithms/expansions may need detail, but urgent short requests do not imply a full professional
workup. Evaluate each supplied rubric clause and necessary omissions for the actual request.
Do not infer clinician audience solely from technical words, section headings or mentioning a doctor.
Keep every-claim factuality and genuinely unnecessary-hedging standards intact. Do not turn 'usually'
into 'always' before refuting it. Do not mistake inability to recall a source for proof it is fabricated.
"""
