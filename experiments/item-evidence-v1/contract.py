"""Isolated, offline candidate contract. Never imported by the running application.

Checks provenance and declared structure, NOT medical entailment or correct grading.
No source retrieval, provider, database, filesystem writes, or verdict repair.
"""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy

from jsonschema import Draft202012Validator

VERSION = 'item-evidence-experiment-v1'
MAX_BYTES = 96_000


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def obj(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def string(max_length=800, pattern=None):
    result = {'type': 'string', 'minLength': 1, 'maxLength': max_length, 'pattern': pattern or r'\S'}
    return result


def enum(*values):
    return {'type': 'string', 'enum': list(values)}


def array(items, maximum=40):
    return {'type': 'array', 'items': items, 'maxItems': maximum}


HASH = string(64, '^[0-9a-f]{64}$')
CID = string(5, r'^C[1-9][0-9]{0,2}$')
OID = string(5, r'^O[1-9][0-9]{0,2}$')
DOC_ID = string(6, r'^(A|C[1-9][0-9]{0,2}|S[1-9][0-9]{0,2})$')
SPAN = obj({'doc_id': DOC_ID, 'doc_sha256': HASH,
            'start_char': {'type': 'integer', 'minimum': 0},
            'end_char': {'type': 'integer', 'minimum': 1}, 'quote': string(800)})
BINDING = obj({k: HASH for k in ['case_sha256', 'rubric_sha256',
                               'obligation_manifest_sha256', 'source_catalog_sha256']})
KINDS = enum('rewrite', 'assess_existing', 'clinical_request', 'explanation', 'mixed', 'other', 'unclear')
MISSING_ITEM = obj({'item': string(), 'searched_doc_ids': array(DOC_ID, 24), 'necessity': string()})
SCHEMA = {
    '$schema': 'https://json-schema.org/draft/2020-12/schema',
    'title': 'Offline item evidence candidate v1, not production Judgment',
    **obj({
        'protocol_version': {'const': VERSION}, 'binding': BINDING,
        'model_verdict': enum('met', 'not_met'), 'model_reason': string(2000),
        'model_review_required': {'type': 'boolean'},
        'task': obj({'kind': KINDS, 'request_locations': array(SPAN, 4), 'uncertainties': array(string(), 6)}),
        'claims': array(obj({'claim_id': CID,
                             'kind': enum('medical_assertion', 'conditional_plan', 'patient_attribute',
                                          'source_attribution', 'text_fidelity', 'non_factual'),
                             'location': SPAN, 'scope_note': string(),
                             'declared_state': enum('supported', 'contradicted', 'unresolved',
                                                    'not_checked', 'not_required')}), 40),
        'obligations': array(obj({'obligation_id': OID, 'declared_state': enum('pass', 'fail', 'unresolved'),
                                 'reason': string(1200), 'claim_ids': array(CID, 40),
                                 'locations': array(SPAN, 6),
                                 'missing_item': {'anyOf': [{'type': 'null'}, MISSING_ITEM]}}), 12),
        'checks': array(obj({'claim_id': CID, 'source_id': DOC_ID, 'source_location': SPAN,
                            'relation': enum('supports', 'contradicts', 'insufficient', 'not_applicable'),
                            'applicability': enum('applicable', 'conditional', 'not_applicable', 'unclear'),
                            'scope_note': string(500)}), 48),
        'coverage': obj({'inventory_status': enum('claimed_complete', 'partial', 'unknown'),
                         'unresolved_claim_ids': array(CID, 40), 'limits': array(string(), 8)}),
    })}
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA)


class ContractError(ValueError):
    def __init__(self, code, path=''):
        super().__init__(code)
        self.code, self.path = code, path


def strict_loads(raw):
    if not isinstance(raw, str):
        raise ContractError('INPUT_NOT_JSON_STRING')
    if len(raw.encode('utf-8')) > MAX_BYTES:
        raise ContractError('JSON_BYTE_LIMIT')

    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ContractError('DUPLICATE_JSON_KEY')
            out[key] = value
        return out

    def nonfinite(_):
        raise ContractError('NONFINITE_JSON_NUMBER')
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (json.JSONDecodeError, RecursionError):
        raise ContractError('INVALID_JSON') from None


def prepare_input(context, answer, rubric, obligations, *, manifest_status='research_draft',
                  source_texts=(), task_hint=None):
    """Explicit fixture/input constructor, not a rubric generator or a task recognizer.

    obligations are externally supplied exact spans of rubric clauses. No labels accepted.
    Runtime adapters need their own trusted construction/authority checks before adoption.
    """
    if manifest_status not in {'research_draft', 'independently_confirmed', 'synthetic_fixture'}:
        raise ContractError('BAD_MANIFEST_AUTHORITY')
    if not isinstance(rubric, str) or not rubric.strip():
        raise ContractError('EMPTY_RUBRIC')
    if not isinstance(answer, str) or not answer.strip() or not context:
        raise ContractError('EMPTY_ORIGINAL_TEXT')
    if not 1 <= len(obligations) <= 12:
        raise ContractError('MANIFEST_REQUIRES_EXTERNAL_DECOMPOSITION')
    if not 1 <= len(context) <= 999 or len(source_texts) > 999:
        raise ContractError('CATALOG_LIMIT')
    if task_hint is not None and task_hint not in KINDS['enum']:
        raise ContractError('BAD_TASK_HINT')
    docs = {'A': {'kind': 'answer', 'role': 'candidate', 'text': answer, 'sha256': text_sha(answer)}}
    for index, turn in enumerate(context, 1):
        if set(turn) != {'role', 'content'} or turn['role'] not in {'user', 'assistant', 'system', 'tool'}:
            raise ContractError('BAD_CONTEXT_SHAPE')
        if not isinstance(turn['content'], str):
            raise ContractError('BAD_CONTEXT_TEXT')
        docs[f'C{index}'] = {'kind': 'context', 'role': turn['role'], 'text': turn['content'],
                            'sha256': text_sha(turn['content'])}
    for index, text in enumerate(source_texts, 1):
        if not isinstance(text, str) or not text.strip():
            raise ContractError('EMPTY_SOURCE_TEXT')
        docs[f'S{index}'] = {'kind': 'external_source_fragment', 'role': 'provided_source',
                            'text': text, 'sha256': text_sha(text)}
    ids = [o['id'] for o in obligations]
    if len(ids) != len(set(ids)):
        raise ContractError('DUPLICATE_MANIFEST_ID')
    for row in obligations:
        if set(row) != {'id', 'start_char', 'end_char', 'quote'} or not re.fullmatch(OID['pattern'], row['id']):
            raise ContractError('BAD_MANIFEST_SHAPE')
        start, end = row['start_char'], row['end_char']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(rubric):
            raise ContractError('BAD_MANIFEST_SPAN')
        if not row['quote'].strip() or rubric[start:end] != row['quote']:
            raise ContractError('BAD_MANIFEST_SPAN')
    manifest = {'status': manifest_status, 'operator': 'AND', 'required': deepcopy(obligations)}
    binding = {'case_sha256': digest({'context': context, 'candidate_output': answer}),
               'rubric_sha256': text_sha(rubric), 'obligation_manifest_sha256': digest(manifest),
               'source_catalog_sha256': digest(docs)}
    return {'binding': binding, 'documents': docs, 'manifest': manifest, 'task_hint': task_hint,
            'context': deepcopy(context), 'candidate_output': answer, 'rubric_text': rubric}


def span(inputs, doc_id, start=0, end=None):
    d = inputs['documents'][doc_id]
    if end is None:
        end = len(d['text'])
    return {'doc_id': doc_id, 'doc_sha256': d['sha256'], 'start_char': start, 'end_char': end,
            'quote': d['text'][start:end]}


def inspect(raw, inputs):
    """Protocol/declared-logic inspection only. Preserves raw verdict and reason.

    Even a perfectly structured output is never medically cleared by this experiment.
    Error reports contain codes/paths, not patient/source text or secret values.
    """
    result = {'protocol_status': 'invalid', 'policy_status': 'NeedsReview', 'clinical_release': False,
              'benchmark_export_eligible': False, 'score': None, 'model_verdict': None,
              'model_reason': None, 'declared_conjunction': 'undetermined', 'structural_coverage': None,
              'consistency_flags': [], 'protocol_errors': [], 'automatic_verdict_changes': False,
              'medical_entailment_verified': False, 'medical_claim_inventory_verified': False}
    try:
        output = strict_loads(raw)
        # Preserve parseable research fields even when another contract field is invalid.
        if isinstance(output, dict):
            for key in ['model_verdict', 'model_reason']:
                if isinstance(output.get(key), str):
                    result[key] = output[key]
        errors = list(VALIDATOR.iter_errors(output))
        if errors:
            e = errors[0]
            raise ContractError('SCHEMA_VIOLATION', '/'.join(str(x) for x in e.absolute_path))
        if output['binding'] != inputs['binding']:
            raise ContractError('FROZEN_INPUT_BINDING_MISMATCH')
        documents = inputs['documents']
        if digest(documents) != inputs['binding']['source_catalog_sha256']:
            raise ContractError('INPUT_CATALOG_CHANGED')
        if digest(inputs['manifest']) != inputs['binding']['obligation_manifest_sha256']:
            raise ContractError('INPUT_MANIFEST_CHANGED')
        if text_sha(inputs['rubric_text']) != inputs['binding']['rubric_sha256']:
            raise ContractError('INPUT_RUBRIC_CHANGED')
        if digest({'context': inputs['context'], 'candidate_output': inputs['candidate_output']}) != inputs['binding']['case_sha256']:
            raise ContractError('INPUT_CASE_CHANGED')

        def validate_span(location, allowed_kinds):
            doc = documents.get(location['doc_id'])
            if doc is None:
                raise ContractError('UNKNOWN_DOCUMENT_ID')
            if doc['kind'] not in allowed_kinds:
                raise ContractError('WRONG_EVIDENCE_KIND')
            if doc['sha256'] != text_sha(doc['text']) or location['doc_sha256'] != doc['sha256']:
                raise ContractError('DOCUMENT_HASH_MISMATCH')
            start, end = location['start_char'], location['end_char']
            if not 0 <= start < end <= len(doc['text']) or doc['text'][start:end] != location['quote']:
                raise ContractError('EXACT_SPAN_MISMATCH')

        def unique_ids(rows, field):
            ids = [r[field] for r in rows]
            if len(ids) != len(set(ids)):
                raise ContractError('DUPLICATE_' + field.upper())
            return set(ids)

        claim_ids = unique_ids(output['claims'], 'claim_id')
        obligation_ids = unique_ids(output['obligations'], 'obligation_id')
        expected = {r['id'] for r in inputs['manifest']['required']}
        if obligation_ids - expected:
            raise ContractError('UNKNOWN_OBLIGATION_ID')
        for loc in output['task']['request_locations']:
            validate_span(loc, {'context'})
        for c in output['claims']:
            validate_span(c['location'], {'answer'})
        for o in output['obligations']:
            if not set(o['claim_ids']) <= claim_ids:
                raise ContractError('UNKNOWN_CLAIM_ID')
            if len(o['claim_ids']) != len(set(o['claim_ids'])):
                raise ContractError('DUPLICATE_OBLIGATION_CLAIM_BINDING')
            for loc in o['locations']:
                validate_span(loc, {'answer', 'context'})
            missing = o['missing_item']
            if missing is not None:
                searched = missing['searched_doc_ids']
                if len(searched) != len(set(searched)):
                    raise ContractError('DUPLICATE_SEARCH_SCOPE_ID')
                if any(d not in documents or documents[d]['kind'] not in {'answer', 'context'} for d in searched):
                    raise ContractError('INVALID_OMISSION_SEARCH_SCOPE')
        pairs = [(r['claim_id'], r['source_id']) for r in output['checks']]
        if len(pairs) != len(set(pairs)):
            raise ContractError('DUPLICATE_CLAIM_SOURCE_PAIR')
        for check in output['checks']:
            if check['claim_id'] not in claim_ids:
                raise ContractError('UNKNOWN_CLAIM_ID')
            if check['source_id'] != check['source_location']['doc_id']:
                raise ContractError('SOURCE_LOCATION_ID_MISMATCH')
            validate_span(check['source_location'], {'external_source_fragment'})
        unresolved = output['coverage']['unresolved_claim_ids']
        if not set(unresolved) <= claim_ids:
            raise ContractError('UNKNOWN_UNRESOLVED_CLAIM_ID')
        if len(unresolved) != len(set(unresolved)):
            raise ContractError('DUPLICATE_UNRESOLVED_CLAIM_ID')

        flags = []
        if inputs['manifest']['status'] != 'independently_confirmed':
            flags.append('OBLIGATION_MANIFEST_NOT_INDEPENDENTLY_CONFIRMED')
        if output['model_review_required']:
            flags.append('MODEL_REVIEW_REQUESTED')
        if not output['task']['request_locations']:
            flags.append('TASK_REQUEST_NOT_LOCATED')
        if any(documents[l['doc_id']]['role'] != 'user' for l in output['task']['request_locations']):
            flags.append('TASK_ANCHORED_TO_NON_USER_TEXT')
        hint = inputs['task_hint']
        if hint is not None and hint != output['task']['kind']:
            flags.append('NONAUTHORITATIVE_TASK_HINT_CONFLICT')
        if output['task']['kind'] == 'unclear' or output['task']['uncertainties']:
            flags.append('TASK_SCOPE_UNRESOLVED')
        missing_ids = sorted(expected - obligation_ids)
        if missing_ids:
            flags.append('MISSING_REQUIRED_OBLIGATION_RECORDS')
        any_fail = any(o['declared_state'] == 'fail' for o in output['obligations'])
        any_unresolved = any(o['declared_state'] == 'unresolved' for o in output['obligations'])
        declared = 'not_met' if any_fail else 'undetermined' if missing_ids or any_unresolved else 'met'
        if declared != 'undetermined' and declared != output['model_verdict']:
            flags.append('MODEL_VERDICT_VS_DECLARED_ITEMS_CONFLICT')
        if any_unresolved:
            flags.append('UNRESOLVED_OBLIGATION_ITEMS')
        if declared == 'undetermined':
            flags.append('DECLARED_CONJUNCTION_UNDETERMINED')
        for o in output['obligations']:
            if not o['locations'] and not o['missing_item']:
                flags.append('OBLIGATION_WITHOUT_LOCATED_SUPPORT')
            if o['missing_item']:
                flags.append('OMISSION_ASSERTION_REQUIRES_SEMANTIC_REVIEW')
                if not o['missing_item']['searched_doc_ids']:
                    flags.append('OMISSION_SEARCH_SCOPE_NOT_RECORDED')
            if o['declared_state'] == 'pass' and o['missing_item']:
                flags.append('PASS_WITH_DECLARED_MISSING_ITEM')
        by_claim = {cid: [r for r in output['checks'] if r['claim_id'] == cid] for cid in claim_ids}
        unverified = set()
        for c in output['claims']:
            cid, state = c['claim_id'], c['declared_state']
            suitable = [r for r in by_claim[cid] if r['applicability'] == 'applicable']
            if state in {'unresolved', 'not_checked'}:
                unverified.add(cid)
                flags.append('CLAIM_STATE_UNRESOLVED')
            if state in {'supported', 'contradicted'}:
                relation = 'supports' if state == 'supported' else 'contradicts'
                if not any(r['relation'] == relation for r in suitable):
                    flags.append('DECLARED_FACT_STATE_WITHOUT_MATCHING_SOURCE_RELATION')
                    unverified.add(cid)
            if state == 'not_required' and c['kind'] != 'non_factual':
                flags.append('FACT_CHECK_EXEMPTION_REQUIRES_REVIEW')
                unverified.add(cid)
            if {r['relation'] for r in suitable} >= {'supports', 'contradicts'}:
                flags.append('CONFLICTING_SOURCE_RELATIONS')
                unverified.add(cid)
        if not unverified <= set(unresolved):
            flags.append('UNRESOLVED_CLAIMS_NOT_LISTED_IN_COVERAGE')
        if any(r['relation'] in {'insufficient', 'not_applicable'} or r['applicability'] != 'applicable'
               for r in output['checks']):
            flags.append('SOURCE_RELATION_OR_APPLICABILITY_UNRESOLVED')
        if not output['claims']:
            flags.append('NO_CLAIM_INVENTORY')
        if output['coverage']['inventory_status'] != 'claimed_complete':
            flags.append('CLAIM_INVENTORY_PARTIAL_OR_UNKNOWN')
        if output['coverage']['inventory_status'] == 'claimed_complete' and (missing_ids or unverified or unresolved):
            flags.append('COMPLETE_DECLARATION_WITH_KNOWN_GAPS')
        if any(len(output[k]) == limit for k, limit in [('claims', 40), ('obligations', 12), ('checks', 48)]):
            flags.append('DECLARED_LIST_AT_EXPERIMENT_BOUNDARY')
        if not output['checks']:
            flags.append('NO_BOUND_EXTERNAL_SOURCE_CHECKS')
        flags.append('SEMANTIC_ENTAILMENT_AND_FULL_CLAIM_COVERAGE_NOT_INDEPENDENTLY_VERIFIED')
        result.update(protocol_status='valid', model_verdict=output['model_verdict'], model_reason=output['model_reason'],
                      model_review_required=output['model_review_required'], declared_conjunction=declared,
                      structural_coverage={'unit': 'provided_obligation_ids_only', 'recorded': len(obligation_ids),
                                           'expected': len(expected), 'missing_ids': missing_ids,
                                           'clinical_coverage_claim': False},
                      consistency_flags=sorted(set(flags)))
    except ContractError as exc:
        result['protocol_errors'] = [{'code': exc.code, 'path': exc.path}]
    return result


def inspect_legacy(parsed):
    """Compatibility inventory only; never invent obligations or change old verdicts."""
    if not isinstance(parsed, dict):
        raise ContractError('LEGACY_NOT_OBJECT')
    return {'compatibility': 'legacy_requires_new_instrumentation',
            'historical_record_status_changed': False,
            'original_verdict': parsed.get('judgment'), 'original_reason': parsed.get('reason'),
            'item_evidence': None, 'new_coverage': 'unknown', 'clinical_release': False,
            'new_prediction': False, 'grading_performed': False,
            'note': 'No new fields inferred from old reasons; old format is not retroactively failed.'}
