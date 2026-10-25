"""Non-default prompt assembly and supplied-output checks. No model/network/storage client.

Candidate-only tri-state research protocol. NOT a legacy API/sidecar adapter.
Checks structure and model-declared logic; does not judge medical entailment.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY_PATH = ROOT / 'experiments/item-evidence-v1/contract.py'
spec = importlib.util.spec_from_file_location('prompt_candidate_legacy_contract', LEGACY_PATH)
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)
VERSION = 'prompt-evidence-research-v1'
MAX_INPUT_BYTES = 160_000
MAX_MESSAGE_BYTES = 480_000
SCHEMA = deepcopy(legacy.SCHEMA)
SCHEMA['title'] = 'Non-default tri-state prompt research output; not legacy Judgment'
for old, new in [('model_verdict', 'criterion_decision'), ('model_reason', 'decision_reason'),
                 ('model_review_required', 'review_required')]:
    SCHEMA['properties'][new] = SCHEMA['properties'].pop(old)
    SCHEMA['required'][SCHEMA['required'].index(old)] = new
SCHEMA['properties']['protocol_version'] = {'const': VERSION}
SCHEMA['properties']['criterion_decision'] = legacy.enum('met', 'not_met', 'unable_to_determine')
SCHEMA['properties']['review_required'] = {'const': True}
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def validate_input(value):
    keys = {'binding', 'documents', 'manifest', 'task_hint', 'context', 'candidate_output', 'rubric_text'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('INPUT_NOT_FROZEN_WHITELIST')
    if len(canonical(value).encode('utf-8')) > MAX_INPUT_BYTES:
        raise ValueError('INPUT_TOO_LARGE')
    if value['manifest'].get('operator') != 'AND':
        raise ValueError('UNSUPPORTED_LOGIC')
    if set(value['manifest']) != {'operator', 'required', 'status'}:
        raise ValueError('MANIFEST_EXTRA_FIELDS')
    docs = value['documents']
    sources = [docs[f'S{i}']['text'] for i in range(1, 1 + sum(k.startswith('S') for k in docs))]
    rebuilt = legacy.prepare_input(value['context'], value['candidate_output'], value['rubric_text'],
                                   value['manifest']['required'], manifest_status=value['manifest']['status'],
                                   source_texts=sources, task_hint=value['task_hint'])
    if rebuilt != value:
        raise ValueError('FROZEN_INPUT_MISMATCH')
    return value


def check_candidate(raw, inputs):
    """Check supplied JSON only. No prediction, verdict repair or medical certification."""
    result = {'protocol_status': 'invalid', 'policy_status': 'NeedsReview',
              'clinical_release': False, 'score': None, 'benchmark_export_eligible': False,
              'criterion_decision': None, 'declared_conjunction': 'undetermined',
              'flags': [], 'errors': [], 'medical_entailment_verified': False,
              'claim_inventory_independently_verified': False, 'automatic_verdict_changes': False}
    try:
        validate_input(inputs)
        out = legacy.strict_loads(raw)
        if isinstance(out, dict):
            result['criterion_decision'] = out.get('criterion_decision')
        if list(VALIDATOR.iter_errors(out)):
            raise ValueError('CANDIDATE_SCHEMA_VIOLATION')
        if out['binding'] != inputs['binding']:
            raise ValueError('TARGET_BINDING_MISMATCH')
        docs = inputs['documents']

        def location(row, kinds):
            doc = docs.get(row['doc_id'])
            if doc is None or doc['kind'] not in kinds:
                raise ValueError('WRONG_EVIDENCE_DOCUMENT')
            a, b = row['start_char'], row['end_char']
            if row['doc_sha256'] != doc['sha256'] or not 0 <= a < b <= len(doc['text']) \
                    or doc['text'][a:b] != row['quote']:
                raise ValueError('EXACT_SPAN_MISMATCH')

        def ids(rows, key):
            values = [r[key] for r in rows]
            if len(values) != len(set(values)):
                raise ValueError('DUPLICATE_ID')
            return set(values)

        claims = ids(out['claims'], 'claim_id')
        obligations = ids(out['obligations'], 'obligation_id')
        expected = {r['id'] for r in inputs['manifest']['required']}
        if obligations - expected:
            raise ValueError('UNKNOWN_OBLIGATION')
        for loc in out['task']['request_locations']:
            location(loc, {'context'})
        for claim in out['claims']:
            location(claim['location'], {'answer'})
        for row in out['obligations']:
            if not set(row['claim_ids']) <= claims or len(row['claim_ids']) != len(set(row['claim_ids'])):
                raise ValueError('BAD_OBLIGATION_CLAIM_REFERENCES')
            for loc in row['locations']:
                location(loc, {'answer', 'context'})
            missing = row['missing_item']
            if missing is not None:
                scope = missing['searched_doc_ids']
                if len(scope) != len(set(scope)) or any(k not in docs or docs[k]['kind'] not in
                        {'answer', 'context'} for k in scope):
                    raise ValueError('BAD_OMISSION_SCOPE')
        seen = set()
        for row in out['checks']:
            pair = (row['claim_id'], row['source_id'])
            if pair in seen or row['claim_id'] not in claims:
                raise ValueError('BAD_CLAIM_SOURCE_PAIR')
            seen.add(pair)
            if row['source_id'] != row['source_location']['doc_id']:
                raise ValueError('SOURCE_LOCATION_ID_MISMATCH')
            location(row['source_location'], {'external_source_fragment'})
        unresolved = out['coverage']['unresolved_claim_ids']
        if not set(unresolved) <= claims or len(unresolved) != len(set(unresolved)):
            raise ValueError('BAD_UNRESOLVED_CLAIM_IDS')

        flags = {'SEMANTIC_SUPPORT_AND_FULL_CLAIM_INVENTORY_NOT_INDEPENDENTLY_VERIFIED'}
        if inputs['manifest']['status'] != 'independently_confirmed':
            flags.add('MANIFEST_NOT_INDEPENDENTLY_CONFIRMED')
        if not out['task']['request_locations'] or any(docs[r['doc_id']]['role'] != 'user'
                for r in out['task']['request_locations']):
            flags.add('TASK_REQUEST_NOT_ANCHORED_TO_USER')
        if out['task']['uncertainties'] or out['task']['kind'] == 'unclear' or (
                inputs['task_hint'] and out['task']['kind'] != inputs['task_hint']):
            flags.add('TASK_SCOPE_NEEDS_REVIEW')
        missing_ids = expected - obligations
        states = [o['declared_state'] for o in out['obligations']]
        declared = ('not_met' if 'fail' in states else 'undetermined'
                    if missing_ids or 'unresolved' in states else 'met')
        if missing_ids:
            flags.add('MISSING_REQUIRED_OBLIGATIONS')
        mapped = 'unable_to_determine' if declared == 'undetermined' else declared
        if out['criterion_decision'] != mapped:
            flags.add('DECISION_VS_DECLARED_ITEMS_CONFLICT')
        unchecked = {c['claim_id'] for c in out['claims'] if c['declared_state'] in {'not_checked', 'unresolved'}}
        for claim in out['claims']:
            paired = [r for r in out['checks'] if r['claim_id'] == claim['claim_id']]
            if claim['declared_state'] in {'supported', 'contradicted'}:
                relation = 'supports' if claim['declared_state'] == 'supported' else 'contradicts'
                if not any(r['relation'] == relation and r['applicability'] == 'applicable' for r in paired):
                    unchecked.add(claim['claim_id'])
                    flags.add('DECLARED_FACT_WITHOUT_MATCHING_SUPPLIED_SOURCE')
            if claim['declared_state'] == 'not_required' and claim['kind'] != 'non_factual':
                unchecked.add(claim['claim_id'])
                flags.add('FACT_CHECK_EXEMPTION_NEEDS_REVIEW')
            if {r['relation'] for r in paired if r['applicability'] == 'applicable'} >= {'supports', 'contradicts'}:
                unchecked.add(claim['claim_id'])
                flags.add('CONFLICTING_SOURCE_RELATIONS')
        if not unchecked <= set(unresolved):
            flags.add('UNRESOLVED_CLAIMS_NOT_REPORTED')
        if any(r['relation'] in {'insufficient', 'not_applicable'} or r['applicability'] != 'applicable'
                for r in out['checks']):
            flags.add('SOURCE_SUPPORT_OR_APPLICABILITY_UNRESOLVED')
        if any(o['missing_item'] is not None for o in out['obligations']):
            flags.add('OMISSION_REQUIRES_SEMANTIC_REVIEW')
        if any(not o['locations'] and not o['missing_item'] for o in out['obligations']):
            flags.add('OBLIGATION_SUPPORT_NOT_LOCATED')
        if out['coverage']['inventory_status'] == 'claimed_complete' and (missing_ids or unchecked or unresolved):
            flags.add('COMPLETE_DECLARATION_WITH_KNOWN_GAPS')
        if any(len(out[k]) == lim for k, lim in [('claims', 40), ('obligations', 12), ('checks', 48)]):
            flags.add('LIST_AT_LIMIT_NOT_PROOF_OF_COMPLETENESS')
        result.update(protocol_status='valid', declared_conjunction=declared, flags=sorted(flags),
                      provided_obligation_coverage={'expected': len(expected), 'recorded': len(obligations)})
    except (ValueError, TypeError, KeyError, legacy.ContractError) as exc:
        result['errors'] = [exc.code if isinstance(exc, legacy.ContractError) else
                            str(exc) if type(exc) is ValueError else type(exc).__name__]
    return result


def build_messages(inputs, *, arm, target_group, target_split='dev', examples=(), enabled=False):
    """Explicit opt-in, pure assembly. Reviewed examples are local synthetic allowlist only."""
    if not enabled:
        return {'status': 'disabled', 'messages': [], 'model_calls': 0}
    validate_input(inputs)
    if arm not in {'B1_zero', 'B2_three'} or target_split != 'dev' or not isinstance(target_group, str) or not target_group.strip():
        raise ValueError('UNAPPROVED_ARM_OR_SPLIT')
    if (arm == 'B1_zero' and examples) or (arm == 'B2_three' and len(examples) != 3):
        raise ValueError('ARM_EXAMPLE_COUNT')
    from fixtures import reviewed_examples
    frozen = reviewed_examples()
    approved = {c['example_id']: digest(c) for c in frozen}
    if arm == 'B2_three' and [c.get('example_id') for c in examples] != [c['example_id'] for c in frozen]:
        raise ValueError('EXAMPLE_NOT_REVIEWED_ALLOWLIST_OR_ORDER')
    seen = set()
    for card in examples:
        key = card.get('example_id')
        if key in seen or key not in approved or digest(card) != approved[key]:
            raise ValueError('EXAMPLE_NOT_REVIEWED_ALLOWLIST')
        seen.add(key)
        if card['group_id'] == target_group or card['input']['binding']['case_sha256'] == inputs['binding']['case_sha256']:
            raise ValueError('EXAMPLE_TARGET_LEAKAGE')
        if card['review']['scope'] != 'nonmedical_synthetic_text_only' or card['split'] != 'dev':
            raise ValueError('EXAMPLE_SCOPE_NOT_APPROVED')
    system = (HERE / 'system.md').read_text(encoding='utf-8') + '\nCANDIDATE_SCHEMA\n' + canonical(SCHEMA)
    payload = {'protocol_version': VERSION, 'demonstrations': deepcopy(list(examples)), 'target': deepcopy(inputs)}
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical(payload)}]
    if len(canonical(messages).encode('utf-8')) > MAX_MESSAGE_BYTES:
        raise ValueError('PROMPT_BYTE_LIMIT')
    return {'status': 'assembled_only_no_generation', 'arm': arm, 'version': VERSION,
            'messages': messages, 'input_sha256': digest(inputs), 'messages_sha256': digest(messages),
            'example_ids': [c['example_id'] for c in examples], 'model_calls': 0,
            'token_count': None, 'token_limit_validated': False,
            'byte_length': len(canonical(messages).encode('utf-8'))}
