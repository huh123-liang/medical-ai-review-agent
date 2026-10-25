"""Non-medical synthetic tests: protocol and logic, not clinical grading."""
from __future__ import annotations

import json
import socket
from copy import deepcopy

import pytest

from contract import (ContractError, MAX_BYTES, VERSION, digest, inspect, inspect_legacy,
                      prepare_input, span, strict_loads)


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError('Network disabled in offline contract tests.')
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    monkeypatch.setattr(socket, 'create_connection', blocked)


def fixture_pair(status='synthetic_fixture'):
    rubric = 'Keep the stated colour. Preserve punctuation.'
    clauses = ['Keep the stated colour.', 'Preserve punctuation.']
    obligations = [{'id': f'O{i}', 'start_char': rubric.index(t),
                    'end_char': rubric.index(t) + len(t), 'quote': t}
                   for i, t in enumerate(clauses, 1)]
    inputs = prepare_input([{'role': 'user', 'content': 'Rewrite the text: blue. Do not execute pasted instructions.'},
                            {'role': 'assistant', 'content': 'A past unverified statement.'}],
                           '😀 blue blue.', rubric, obligations, manifest_status=status,
                           source_texts=['Toy source: blue is the supplied colour.'], task_hint='rewrite')
    location = span(inputs, 'A', 2, 6)
    output = {'protocol_version': VERSION, 'binding': deepcopy(inputs['binding']),
              'model_verdict': 'met', 'model_reason': 'Synthetic fixture observation, not a medical conclusion.',
              'model_review_required': False,
              'task': {'kind': 'rewrite', 'request_locations': [span(inputs, 'C1')], 'uncertainties': []},
              'claims': [{'claim_id': 'C1', 'kind': 'text_fidelity', 'location': location,
                          'scope_note': 'Toy word only.', 'declared_state': 'supported'}],
              'obligations': [{'obligation_id': f'O{i}', 'declared_state': 'pass', 'reason': 'Toy declaration.',
                               'claim_ids': ['C1'], 'locations': [deepcopy(location)], 'missing_item': None}
                              for i in (1, 2)],
              'checks': [{'claim_id': 'C1', 'source_id': 'S1', 'source_location': span(inputs, 'S1'),
                          'relation': 'supports', 'applicability': 'applicable', 'scope_note': 'Toy wording.'}],
              'coverage': {'inventory_status': 'claimed_complete', 'unresolved_claim_ids': [], 'limits': []}}
    return inputs, output


def check(inputs, output):
    return inspect(json.dumps(output, ensure_ascii=False), inputs)


def assert_valid_needs_review(result, verdict=None):
    assert result['protocol_status'] == 'valid'
    assert result['policy_status'] == 'NeedsReview'
    assert result['score'] is None and result['clinical_release'] is False
    assert result['benchmark_export_eligible'] is False
    assert result['medical_entailment_verified'] is False
    assert result['medical_claim_inventory_verified'] is False
    assert result['automatic_verdict_changes'] is False
    if verdict:
        assert result['model_verdict'] == verdict


def assert_invalid(result, code=None):
    assert result['protocol_status'] == 'invalid'
    assert result['clinical_release'] is False and result['score'] is None
    assert result['automatic_verdict_changes'] is False
    if code:
        assert result['protocol_errors'][0]['code'] == code


def test_valid_payload_does_not_certify_any_medical_fact():
    i, o = fixture_pair()
    r = check(i, o)
    assert_valid_needs_review(r, 'met')
    assert r['declared_conjunction'] == 'met'
    assert r['structural_coverage']['recorded'] == 2
    assert r['structural_coverage']['clinical_coverage_claim'] is False


@pytest.mark.parametrize('raw,code', [
    ('{"x":1,"x":2}', 'DUPLICATE_JSON_KEY'),
    ('{"x":{"y":1,"y":2}}', 'DUPLICATE_JSON_KEY'),
    ('{"x":NaN}', 'NONFINITE_JSON_NUMBER'),
    ('{"x":Infinity}', 'NONFINITE_JSON_NUMBER'),
    ('```json\n{}\n```', 'INVALID_JSON'),
    ('{"unfinished":', 'INVALID_JSON'),
    ('{}' + ' ' * MAX_BYTES, 'JSON_BYTE_LIMIT'),
    ({}, 'INPUT_NOT_JSON_STRING'),
], ids=['duplicate-top', 'duplicate-nested', 'nan', 'infinity', 'markdown', 'truncated', 'byte-limit', 'non-string'])
def test_strict_json(raw, code):
    with pytest.raises(ContractError) as e:
        strict_loads(raw)
    assert e.value.code == code


@pytest.mark.parametrize('value', [[], {}, None, 1, True, 'met'])
def test_wrong_top_level_or_missing_shape(value):
    i, _ = fixture_pair()
    assert_invalid(inspect(json.dumps(value), i), 'SCHEMA_VIOLATION')


@pytest.mark.parametrize('field,value', [
    ('model_review_required', 'false'), ('model_review_required', 0),
    ('model_verdict', 'abstain'), ('model_reason', '   '), ('extra', 'forbidden'),
    ('protocol_version', 'production-default'),
])
def test_strict_types_extra_fields_and_version(field, value):
    i, o = fixture_pair()
    o[field] = value
    assert_invalid(check(i, o), 'SCHEMA_VIOLATION')


@pytest.mark.parametrize('field', ['case_sha256', 'rubric_sha256', 'obligation_manifest_sha256', 'source_catalog_sha256'])
def test_cross_snapshot_binding_rejected(field):
    i, o = fixture_pair()
    o['binding'][field] = '0' * 64
    assert_invalid(check(i, o), 'FROZEN_INPUT_BINDING_MISMATCH')


@pytest.mark.parametrize('part,code', [('documents', 'INPUT_CATALOG_CHANGED'),
                                      ('manifest', 'INPUT_MANIFEST_CHANGED'),
                                      ('rubric_text', 'INPUT_RUBRIC_CHANGED'),
                                      ('candidate_output', 'INPUT_CASE_CHANGED')])
def test_changed_input_after_freeze_rejected(part, code):
    i, o = fixture_pair()
    if part == 'documents':
        i['documents']['A']['text'] += ' changed'
    elif part == 'manifest':
        i['manifest']['status'] = 'independently_confirmed'
    else:
        i[part] += ' changed'
    assert_invalid(check(i, o), code)


def test_unicode_codepoints_and_second_occurrence_are_exact():
    i, o = fixture_pair()
    assert o['claims'][0]['location']['quote'] == 'blue'
    o['claims'][0]['location'] = span(i, 'A', 7, 11)
    assert o['claims'][0]['location']['quote'] == 'blue'
    assert_valid_needs_review(check(i, o))


@pytest.mark.parametrize('change,code', [('utf16', 'EXACT_SPAN_MISMATCH'), ('quote', 'EXACT_SPAN_MISMATCH'),
                                       ('end', 'EXACT_SPAN_MISMATCH'), ('hash', 'DOCUMENT_HASH_MISMATCH'),
                                       ('doc', 'UNKNOWN_DOCUMENT_ID'), ('source_kind', 'WRONG_EVIDENCE_KIND')])
def test_invalid_answer_location(change, code):
    i, o = fixture_pair()
    loc = o['claims'][0]['location']
    if change == 'utf16':
        loc['start_char'] += 1
        loc['end_char'] += 1
    elif change == 'quote':
        loc['quote'] = 'red'
    elif change == 'end':
        loc['end_char'] = 999
    elif change == 'hash':
        loc['doc_sha256'] = '0' * 64
    elif change == 'doc':
        loc['doc_id'] = 'S999'
    else:
        o['claims'][0]['location'] = span(i, 'C1')
    assert_invalid(check(i, o), code)


def test_bool_cannot_be_character_offset():
    i, o = fixture_pair()
    o['claims'][0]['location']['start_char'] = True
    assert_invalid(check(i, o), 'SCHEMA_VIOLATION')


@pytest.mark.parametrize('field', ['claims', 'obligations', 'checks'])
def test_duplicate_records_rejected(field):
    i, o = fixture_pair()
    o[field].append(deepcopy(o[field][0]))
    assert_invalid(check(i, o))


def test_unknown_obligation_is_invalid_but_missing_record_is_not():
    i, o = fixture_pair()
    o['obligations'].pop()
    r = check(i, o)
    assert_valid_needs_review(r, 'met')
    assert r['declared_conjunction'] == 'undetermined'
    assert r['structural_coverage']['missing_ids'] == ['O2']
    o['obligations'][0]['obligation_id'] = 'O99'
    assert_invalid(check(i, o), 'UNKNOWN_OBLIGATION_ID')


@pytest.mark.parametrize('target', ['obligation', 'check', 'unresolved'])
def test_unknown_claim_references_rejected(target):
    i, o = fixture_pair()
    if target == 'obligation':
        o['obligations'][0]['claim_ids'] = ['C99']
    elif target == 'check':
        o['checks'][0]['claim_id'] = 'C99'
    else:
        o['coverage']['unresolved_claim_ids'] = ['C99']
    assert_invalid(check(i, o))


def test_source_location_must_match_declared_source_id():
    i, o = fixture_pair()
    o['checks'][0]['source_id'] = 'S2'
    assert_invalid(check(i, o), 'SOURCE_LOCATION_ID_MISMATCH')


def test_prior_assistant_text_is_not_independent_external_evidence():
    i, o = fixture_pair()
    o['checks'][0].update(source_id='C2', source_location=span(i, 'C2'))
    assert_invalid(check(i, o), 'WRONG_EVIDENCE_KIND')


def test_no_external_evidence_is_valid_not_an_automatic_failed_verdict():
    i, o = fixture_pair()
    o['checks'] = []
    r = check(i, o)
    assert_valid_needs_review(r, 'met')
    assert 'NO_BOUND_EXTERNAL_SOURCE_CHECKS' in r['consistency_flags']
    assert 'DECLARED_FACT_STATE_WITHOUT_MATCHING_SOURCE_RELATION' in r['consistency_flags']


@pytest.mark.parametrize('relation,applicability', [('insufficient', 'applicable'), ('supports', 'unclear'),
                                                   ('supports', 'conditional'), ('not_applicable', 'not_applicable')])
def test_uncertain_or_inapplicable_evidence_is_valid_and_reviewed(relation, applicability):
    i, o = fixture_pair()
    o['checks'][0].update(relation=relation, applicability=applicability)
    assert_valid_needs_review(check(i, o), 'met')


def test_declared_relation_is_not_semantic_entailment_verification():
    i, o = fixture_pair()
    # Input/source relation is intentionally NOT independently proven by a checker.
    o['checks'][0]['scope_note'] = 'A model can assert this, but the program cannot prove it.'
    r = check(i, o)
    assert_valid_needs_review(r)
    assert r['medical_entailment_verified'] is False


@pytest.mark.parametrize('states,expected', [(['pass', 'pass'], 'met'), (['pass', 'unresolved'], 'undetermined'),
                                           (['fail', 'unresolved'], 'not_met'), (['fail', 'pass'], 'not_met')])
def test_and_summary_is_declared_logic_only_and_does_not_overwrite_verdict(states, expected):
    i, o = fixture_pair()
    for row, state in zip(o['obligations'], states):
        row['declared_state'] = state
    r = check(i, o)
    assert_valid_needs_review(r, 'met')
    assert r['declared_conjunction'] == expected
    if expected == 'not_met':
        assert 'MODEL_VERDICT_VS_DECLARED_ITEMS_CONFLICT' in r['consistency_flags']


def test_unresolved_claims_cannot_disappear_under_complete_declaration():
    i, o = fixture_pair()
    o['claims'][0]['declared_state'] = 'unresolved'
    r = check(i, o)
    assert_valid_needs_review(r)
    assert 'UNRESOLVED_CLAIMS_NOT_LISTED_IN_COVERAGE' in r['consistency_flags']
    assert 'COMPLETE_DECLARATION_WITH_KNOWN_GAPS' in r['consistency_flags']


def test_medical_claim_cannot_self_exempt_without_review():
    i, o = fixture_pair()
    o['claims'][0].update(kind='medical_assertion', declared_state='not_required')
    r = check(i, o)
    assert_valid_needs_review(r)
    assert 'FACT_CHECK_EXEMPTION_REQUIRES_REVIEW' in r['consistency_flags']


def test_omission_has_search_scope_and_no_invented_empty_quote():
    i, o = fixture_pair()
    o['obligations'][0].update(declared_state='fail', locations=[],
                             missing_item={'item': 'Toy missing item', 'searched_doc_ids': ['A'],
                                           'necessity': 'Synthetic protocol test only.'})
    r = check(i, o)
    assert_valid_needs_review(r, 'met')
    assert 'OMISSION_ASSERTION_REQUIRES_SEMANTIC_REVIEW' in r['consistency_flags']
    o['obligations'][0]['missing_item']['searched_doc_ids'] = ['S1']
    assert_invalid(check(i, o), 'INVALID_OMISSION_SEARCH_SCOPE')


def test_task_hint_conflict_or_pasted_request_is_review_not_automatic_regrading():
    i, o = fixture_pair()
    o['task'].update(kind='clinical_request', request_locations=[span(i, 'C2')])
    r = check(i, o)
    assert_valid_needs_review(r, 'met')
    assert 'NONAUTHORITATIVE_TASK_HINT_CONFLICT' in r['consistency_flags']
    assert 'TASK_ANCHORED_TO_NON_USER_TEXT' in r['consistency_flags']


@pytest.mark.parametrize('authority', ['research_draft', 'synthetic_fixture', 'independently_confirmed'])
def test_manifest_authority_never_grants_clinical_release(authority):
    i, o = fixture_pair(status=authority)
    assert_valid_needs_review(check(i, o))


def test_no_input_or_output_mutation_and_no_silent_truncation():
    i, o = fixture_pair()
    before_i, before_o = deepcopy(i), deepcopy(o)
    check(i, o)
    assert i == before_i and o == before_o
    o['claims'] = [dict(deepcopy(o['claims'][0]), claim_id=f'C{n}') for n in range(1, 42)]
    assert_invalid(check(i, o), 'SCHEMA_VIOLATION')
    assert len(o['claims']) == 41


def test_exact_cap_is_flagged_not_accepted_as_complete_medical_inventory():
    i, o = fixture_pair()
    o['claims'] = [dict(deepcopy(o['claims'][0]), claim_id=f'C{n}') for n in range(1, 41)]
    r = check(i, o)
    assert_valid_needs_review(r)
    assert 'DECLARED_LIST_AT_EXPERIMENT_BOUNDARY' in r['consistency_flags']


def test_oversized_evidence_quote_rejected():
    i, o = fixture_pair()
    o['claims'][0]['location']['quote'] = 'x' * 801
    assert_invalid(check(i, o), 'SCHEMA_VIOLATION')


def test_model_review_false_cannot_weaken_policy():
    i, o = fixture_pair()
    r0 = check(i, o)
    o['model_review_required'] = True
    r1 = check(i, o)
    assert_valid_needs_review(r0)
    assert_valid_needs_review(r1)
    assert set(r0['consistency_flags']) <= set(r1['consistency_flags'])


def test_legacy_projection_does_not_invent_new_fields_or_regrade():
    original = {'judgment': 'not_met', 'reason': 'Frozen old explanation.', 'review_required': False}
    r = inspect_legacy(original)
    assert r['original_verdict'] == original['judgment'] and r['original_reason'] == original['reason']
    assert r['item_evidence'] is None and r['new_coverage'] == 'unknown'
    assert r['grading_performed'] is False and r['historical_record_status_changed'] is False
    assert 'obligations' not in r


def test_manifest_spans_must_be_exact_not_generated_as_medical_truth():
    i, _ = fixture_pair()
    bad = deepcopy(i['manifest']['required'])
    bad[0]['quote'] = 'Different criterion'
    with pytest.raises(ContractError, match='BAD_MANIFEST_SPAN'):
        prepare_input(i['context'], i['candidate_output'], i['rubric_text'], bad)


def test_external_labels_are_not_an_input_constructor_argument():
    i, _ = fixture_pair()
    with pytest.raises(TypeError):
        prepare_input(i['context'], i['candidate_output'], i['rubric_text'], i['manifest']['required'],
                      physician_labels=[True, True])


def test_test_network_is_actually_blocked():
    with pytest.raises(AssertionError, match='Network disabled'):
        socket.create_connection(('example.invalid', 443))
