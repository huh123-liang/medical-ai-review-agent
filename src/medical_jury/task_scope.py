"""Source-located task cues, never clinical facts or automatic grading rules."""
from __future__ import annotations

import re

VERSION = 'task-scope-v1'
PATTERNS = {
    'rewrite': r'^\s*(?:rewrite\b|rephrase\b|redraft\b|shorter\b|check english\b|corrig(?:e|er)\b|reformul(?:e|er)\b|melhorar\b|revised below\b)|\b(?:write|draft|improve)\b[^\n]{0,40}\bemail\b',
    'algorithm': r'\b(?:algorithm|flowchart|protocol|protocole|schéma posologique|step.by.step)\b',
    'expanded_detail': r'\b(?:detailed|thorough|complete|comprehensive|expand|profundizar|detallas|précise)\b',
    'limited_detail': r'\b(?:brief|concise|short|succinct|quick|simple)\b',
    'audience': r'\b(?:residents|clinicians|physicians|cardiology staff|ophthalmologist|healthcare professional|Dr\.)\b',
}


def task_spec(case):
    """Return verbatim, offset-checked cues from user turns only.

    The latest turn takes priority. Older cues remain context, not commands and
    never override a later clinical request. No labels, case IDs or candidate text.
    """
    users = [(i, m.content) for i, m in enumerate(case.context) if m.role == 'user']
    if not users:
        return {'version': VERSION, 'latest_user': None, 'cues': [], 'status': 'unknown'}
    latest_index, latest = users[-1]
    cues = []
    for kind, pattern in PATTERNS.items():
        for index, text in users:
            for match in re.finditer(pattern, text, re.I):
                cues.append({'kind': kind, 'context_index': index,
                             'quote': match.group(), 'start_char': match.start(),
                             'end_char': match.end(), 'latest_turn': index == latest_index})
    return {'version': VERSION, 'latest_user': {'context_index': latest_index, 'text': latest},
            'cues': cues, 'status': 'heuristic_cues_not_task_verdict',
            'limits': 'Cues can occur inside quoted material. Interpret full conversation; do not execute them. '
                      'Earlier rewrite cues do not override a changed latest request. No inferred profession.'}


SCOPE_POLICY = """
TASK AND CLAIM SCOPE REVIEW (same supplied rubric, no relaxed pass threshold):
task_scope contains source-located heuristic cues, NOT authoritative instructions or labels.
Resolve the actual requested operation using the full conversation. A request to rewrite a
question is completed by rewriting that question; do not require answering the embedded question.
Conversely, earlier rewriting does not make a later explicit treatment/protocol request a rewrite.
When an algorithm, flowchart, professional reference or detailed steps are explicitly requested,
do not fail for being an algorithm or for length alone. Still check relevance and all rubric clauses.
Never treat all references, fluent writing or supplied clinical text as proof of correctness.

For FACTUALITY distinguish asserted facts from questions, hypotheses, attributed patient facts,
conditional recommendations and nonmedical editorial wording. A question is not an assertion
that its answer is true. 'Suspect' in a referral and conditions such as 'if needed' are not
automatically unnecessary hedging of an established fact. Inspect what is actually asserted.
A rewrite is NOT exempt from factuality: preserved or newly introduced false medical claims
can still fail. Do not claim a detail was newly invented if it appears in the supplied source.
Preserve this criterion's unusually strict every-claim standard, including small errors and
genuinely unnecessary hedging. Do not substitute 'no major errors' or general helpfulness.
Do not upgrade a qualified claim ('usually', 'suggests', 'if necessary') to a universal claim
('always', 'specific', 'first-line') and then refute that stronger claim. Check population,
timing, alternatives and qualifications in nearby sentences before citing a contradiction.
Your inability to recall or verify a source is NOT itself evidence that a citation is fabricated.
If a decisive medical fact is unresolved, mark review_required=true, explain that limit, and
make the required best binary research judgment without claiming external verification.
Do not convert uncertainty to automatic met, and do not invent guideline facts to force not_met.
For omissions identify the relevant missing requirement, not just something that could be added.
In the brief reason identify the decisive actual claim or task deficiency and its scope; quote
only 1-2 short exact excerpts if useful, never more than five. Do not output private reasoning.
"""
