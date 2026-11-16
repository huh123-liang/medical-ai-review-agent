"""Input screen (v4.2-dev, off by default): the document under review is untrusted text.

Code only, before any model call. It looks for text addressed to the reviewing model rather than to the patient
(instructions to ignore the task, fake chat-role tags, this system's own output field names, hidden characters).
A flag never blocks the review and never changes a finding: it forces escalation, so a human reads the document
knowing the model's verdict may have been steered. Patterns are deliberately narrow (low false-alarm rate on real
clinical prose); they are a tripwire, not a guarantee.
"""
from __future__ import annotations

import re
import unicodedata

GUARD_VERSION = 'input-screen-1'
_I = re.IGNORECASE
PATTERNS = (
    # "ignore all previous instructions": needs all/any, or above/system, so a clinical correction such as
    # "please disregard the previous instructions about metformin" does not trip it
    ('IGNORE_INSTRUCTIONS', re.compile(r'\b(ignore|disregard|forget|override)\s+((all|any)\s+(of\s+)?(the\s+|your\s+)?'
                                       r'(previous|prior|above|earlier|preceding)?\s*|(the\s+)?(above|system)\s+)'
                                       r'(instructions?|prompts?)\b', _I)),
    ('ADDRESSED_TO_MODEL', re.compile(r'\b(as an ai\b|you are (now )?(an? |the )?(ai|language model|llm|chatbot|'
                                      r'reviewer|grader|evaluator)\b|dear (ai|reviewer|grader|model)\b|'
                                      r'note to (the )?(ai|model|reviewer|grader)\b)', _I)),
    ('OUTPUT_STEERING', re.compile(r'\b(return|output|respond with|reply with|answer with)\s+(only\s+)?(an?\s+)?'
                                   r'(empty|no)\s+(list|findings|errors|issues|json)\b|\bmark (all|every) '
                                   r'criteri(a|on) (as )?met\b|\bdo not (report|flag) (any|these) (errors|issues|'
                                   r'findings|problems)\b', _I)),
    ('SCHEMA_KEYS', re.compile(r'"(other_findings|criteria|code_checks)"\s*:', _I)),
    ('ROLE_TAGS', re.compile(r'(<\|?(im_start|im_end)\|?>|<\|(system|assistant|user)\|>|^\s*(system|assistant)\s*:|'
                             r'\[/?(inst|sys)\]|<<\s*sys\s*>>)', _I | re.MULTILINE)),
    ('SYSTEM_PROMPT', re.compile(r'\b(system prompt|jailbreak|prompt injection|developer mode)\b', _I)),
    ('IGNORE_ZH', re.compile(r'(忽略|无视)(以上|上述|之前|前面|先前)?的?(所有|全部)?(指令|提示词)')),
    ('ADDRESSED_ZH', re.compile(r'(你现在是|作为(一个)?(AI|人工智能|语言模型)|系统提示词|'
                                r'不要(报告|标记)(任何)?(错误|问题|发现))')),
)
_HIDDEN = re.compile('[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]')


def screen_input(*texts: str) -> dict:
    flags = []
    for which, text in zip(('case_description', 'document'), texts):
        t = unicodedata.normalize('NFKC', text or '')
        for name, pat in PATTERNS:
            m = pat.search(t)
            if m:
                flags.append({'id': name, 'field': which, 'match': m.group(0)[:60]})
        hidden = _HIDDEN.findall(text or '')
        if hidden:
            flags.append({'id': 'HIDDEN_CHARACTERS', 'field': which, 'match': f'{len(hidden)} invisible/bidi chars'})
    return {'version': GUARD_VERSION, 'flagged': bool(flags), 'flags': flags}
