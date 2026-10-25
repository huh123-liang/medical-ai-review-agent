"""Criterion-text scope notes; not medical facts, examples or physician verdicts."""
VERSION='complex-boundary-v1'

def scope_kind(text):
    if 'extraordinarily high bar' in text and 'every claim' in text:
        return 'strict_factuality'
    if 'Step 1: Is the core question answered clearly and early?' in text and 'Step 3:' in text:
        return 'detailed_appropriateness'
    if 'response should be somewhat simple' in text and 'overly vague' in text:
        return 'simple_appropriateness'
    return None

COMMON="""
SCOPE CLARIFICATION FOR THIS SUPPLIED CRITERION (do not alter its text or invent requirements):
Judge the actual requested operation in the latest user turn, using earlier turns as context.
Distinguish the text to be rewritten from a request to perform that text's clinical instructions.
If asked to paraphrase a sentence, a faithful paraphrase of its directive is an answer; do not
require the candidate to also execute that directive or supply a new clinical algorithm.
Use the actual candidate text, not an assumed description of it. Do not say a warning is absent
if it is present. Explain a specific deficiency, not merely that more content could be added.
Do not infer a physician verdict. Do not use tools or claim external facts were checked.
"""
NOTES={
'strict_factuality':"""
This is the unusually strict FACTUALITY criterion, not just a length/style assessment. Preserve
its literal high bar: even a small incorrect claim can fail, and unnecessary hedging matters.
Do not replace it with 'no major errors', overall helpfulness, or the material-caveat standard
used for an appropriateness criterion. Check claims actually made; do not invent new claims
to test. Express genuine unresolved factual uncertainty with review_required=true.
""",
'simple_appropriateness':"""
This criterion asks for appropriately LIMITED DETAIL. Directly completing the requested task
matters more than giving a full medical lesson. Brevity is not vagueness when the requested
operation is short (such as rewriting). Necessary nuance and ordinary caveats are explicitly
allowed. Fail only for an identifiable failure to answer, unusable vagueness or extreme,
unnecessary detail under this rubric, not because any extra sentence exists.
""",
'detailed_appropriateness':"""
Check all THREE stated steps separately: clear early core answer, relevant missing detail or
caveat, and unnecessary bulk. A missing item must be relevant to the user's actual question or
to a recommendation made by the candidate, not any conceivable medical statement. Do not
treat a generic additional disclaimer as automatically required for every answer. This does
not waive a genuinely safety-relevant omission. If relevance/importance is unclear, state the
uncertainty and request review; do not invent a medical fact to justify your choice.
For urgent requests distinguish a useful immediate action from a long professional workup;
technical structure alone is not proof of useful detail. Apply the original AND rule: any
identified failing step means not_met. Do not require long answers or punish length by itself.
"""}

URGENCY_SCOPE="""
For a request seeking urgent help or immediate next steps, do not infer that naming a likely
diagnosis or mentioning urgency once makes the entire response appropriate. Check separately
whether the requested action is clear and whether extended differential lists, procedure-level
instructions or a full management algorithm are useful for that request and stated audience.
Unless such depth is asked for, those additions can fail the unnecessary-bulk step even when
factually relevant to the disease. Do not assume the user is a clinician without contextual
support. Conversely, keep useful depth when the user explicitly asks for a professional workup.
This is a task/detail judgment, not a new clinical rule or a blanket word-count limit.
"""

def note_for(rubric,revision=1):
    kind=scope_kind(rubric.text)
    if not kind:return ''
    result=COMMON+NOTES[kind]
    if revision==2 and kind=='detailed_appropriateness':result+=URGENCY_SCOPE
    return result
