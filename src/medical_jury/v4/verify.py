"""Independent second opinion on SEVERE open-review claims (v4.2-dev, off by default, never evaluated).

Only claims that come from the model's open review are verified: rule and criteria findings are deterministic and
need no second opinion. The verifier should be a different model family from the checker, so that the two do not share
the same blind spots. Default policy 'flag': a disagreement is shown to the clinician and changes nothing else.
Policy 'downgrade' (to moderate, labelled) exists for an evaluation run only; it must not be used without one.
"""
from __future__ import annotations

import json

from .check import CheckFailure

VERIFY_PROMPT_VERSION = 'v4-verify-0-dev'
VERIFY_MAX_TOKENS = 400
POLICIES = ('flag', 'downgrade')

SYSTEM_VERIFY = (
    'You are an independent second reviewer of chronic-disease management advice. Another reviewer made ONE claim '
    'that the DOCUMENT contains a serious error for THIS patient. Decide whether the claim is correct, using only the '
    'CASE DESCRIPTION and the DOCUMENT. Check the patient\'s actual values against any threshold the claim relies on. '
    'If the document does not actually say what the claim says it says, the claim is not correct.\n'
    'Return one JSON object and nothing else: {"agree": true or false, "reason": "at most 25 words"}.')


class VerifyFailure(CheckFailure):
    """A CheckFailure, so the reviewer records it as a failed call instead of crashing the run."""


def verify_messages(case_description: str, document: str, claim: dict, system: str = SYSTEM_VERIFY) -> list[dict]:
    payload = {'case_description': case_description, 'document': document,
               'claim': {k: claim.get(k) for k in ('category', 'entity', 'issue', 'quote')}}
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]


def parse_verify(raw: str) -> dict:
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        raise VerifyFailure('VERIFY_NOT_JSON', str(e)[:200])
    if not isinstance(obj, dict) or not isinstance(obj.get('agree'), bool):
        raise VerifyFailure('VERIFY_SCHEMA', 'need {"agree": true|false, "reason": "..."}')
    return {'agree': obj['agree'], 'reason': str(obj.get('reason') or '')[:300]}
