"""Offline checks before paid execution; no provider requests or physician material used."""
from __future__ import annotations

import ast
import inspect
import json
from copy import deepcopy

import pytest

import common
from common import (PER_TOPIC, ROOT, TOPICS, candidate_input, candidate_messages, digest, select_rows)
from metrics import binary_metrics, score


def toy_row(index, topic, split='validation'):
    return {'example_id': f'TOY-{topic}-{index}', 'category': 'cluster:' + topic + '_toy',
            'case': {'case_id': f'TOY-C-{index}-{topic}', 'group_id': f'TOY-G-{index}-{topic}',
                     'split': split, 'task': 'Rewrite text', 'context': [{'role': 'user', 'content': 'Rewrite blue.'}],
                     'candidate_output': 'blue.'},
            'rubric': {'text': 'Preserve blue.', 'rubric_id': 'TOY-R'}}


def test_selection_is_label_blind_and_order_stable():
    rows = [toy_row(i, t) for t in TOPICS for i in range(6)]
    a = select_rows(rows)
    b = select_rows(list(reversed(rows)))
    assert a == b and len(a) == len(TOPICS) * PER_TOPIC
    assert len({r['case']['group_id'] for r in a}) == len(a)
    assert 'labels' not in inspect.signature(select_rows).parameters


def test_selection_excludes_prior_groups_and_other_splits():
    rows = [toy_row(i, t) for t in TOPICS for i in range(6)]
    excluded = {r['case']['group_id'] for r in select_rows(rows)}
    rows += [toy_row(100, t, 'test') for t in TOPICS]
    rows += [toy_row(101, t, 'dev') for t in TOPICS]
    assert all(r['case']['group_id'] not in excluded and r['case']['split'] == 'validation'
               for r in select_rows(rows, excluded))


def test_insufficient_groups_never_silent_backfill():
    with pytest.raises(ValueError):
        select_rows([toy_row(0, t) for t in TOPICS])


def test_candidate_preserves_original_criterion_without_fake_clinical_decomposition():
    row = toy_row(0, TOPICS[0])
    row['rubric']['text'] = 'Preserve blue AND punctuation; do not change the condition.'
    i = candidate_input(row)
    assert i['manifest']['status'] == 'research_draft'
    assert i['manifest']['required'] == [{'id': 'O1', 'start_char': 0, 'end_char': len(row['rubric']['text']),
                                        'quote': row['rubric']['text']}]
    assert not any(k.startswith('S') for k in i['documents'])


def test_eval_assembly_reuses_frozen_B1_system_and_no_examples():
    row = toy_row(0, TOPICS[0])
    messages = candidate_messages(row)
    c = common.candidate_module()
    expected = c.build_messages(candidate_input(row), arm='B1_zero', target_group='toy-dev', enabled=True)
    # Message content same; evaluation split is truthfully governed by NEW wrapper, not falsified.
    assert messages == expected['messages']
    assert row['case']['split'] == 'validation'
    assert json.loads(messages[1]['content'])['demonstrations'] == []
    assert c.build_messages.__defaults__ is None


@pytest.mark.parametrize('split', ['dev', 'test', 'adhoc'])
def test_eval_wrapper_rejects_other_splits(split):
    with pytest.raises(ValueError):
        candidate_messages(toy_row(0, TOPICS[0], split))


def test_no_physician_fields_in_model_payload():
    data = json.loads(candidate_messages(toy_row(0, TOPICS[0]))[1]['content'])
    banned = {'binary_labels', 'anonymized_physician_ids', 'physician_label', 'expected_decision'}

    def walk(x):
        if isinstance(x, dict):
            assert not banned & set(x)
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            for value in x:
                walk(value)
    walk(data)


@pytest.mark.parametrize('pairs,agreement,f1,kappa', [
    ([(True, True), (False, False)], 1, 1, 1),
    ([(True, False), (False, True)], 0, 0, -1),
    ([(True, True), (True, False), (False, False), (False, True)], .5, .5, 0),
    ([(True, True), (True, False), (False, False)], 2/3, 2/3, .4)])
def test_metrics_reconcile_with_project_helper(pairs, agreement, f1, kappa):
    from medical_jury.benchmark import binary_metrics as project_metrics
    actual = binary_metrics(pairs)
    assert actual['agreement'] == pytest.approx(agreement)
    assert actual['macro_f1'] == pytest.approx(f1)
    assert actual['cohen_kappa'] == pytest.approx(kappa)
    other = project_metrics(pairs)
    for key in ['agreement', 'macro_f1', 'cohen_kappa']:
        assert actual[key] == pytest.approx(other[key])


def test_empty_and_constant_metrics_are_not_fabricated():
    assert binary_metrics([])['agreement'] is None
    assert binary_metrics([(True, True)])['cohen_kappa'] is None


@pytest.mark.parametrize('bad', [[('met', True)], [(True, 1)], [(True, None)]])
def test_bad_labels_not_cast_to_bool(bad):
    with pytest.raises(ValueError):
        binary_metrics(bad)


def test_unknown_failure_and_disagreement_keep_full_denominators():
    items = [{'topic': 'a', 'labels': [True, False], 'arms': {'x': {'status': 'Completed', 'decision': 'met'}}},
             {'topic': 'b', 'labels': [True, True], 'arms': {'x': {'status': 'Completed', 'decision': 'unable_to_determine'}}},
             {'topic': 'b', 'labels': [False, False], 'arms': {'x': {'status': 'Failed', 'decision': None}}}]
    r = score(items, 'x', bootstrap=20)
    assert r['primary_metrics'] is None and r['planned_physician_comparisons'] == 6
    assert r['decided_cases'] == r['undetermined_cases'] == r['failed_or_not_run_cases'] == 1
    assert r['conditional_decided_metrics']['n'] == 2 and r['conditional_decided_metrics']['agreement'] == .5
    assert r['delivered_label_match_fraction'] == pytest.approx(1/6)


def test_score_retains_each_doctor_not_majority_collapse():
    items = [{'topic': 'a', 'labels': [True, True, False], 'arms': {'x': {'status': 'Completed', 'decision': 'met'}}}]
    r = score(items, 'x', bootstrap=20)
    assert r['primary_metrics']['n'] == 3
    assert r['primary_metrics']['agreement'] == pytest.approx(2/3)


def test_prediction_code_does_not_deserialize_sealed_references():
    source = (common.HERE / 'predict.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'references.json' not in source and 'binary_labels' not in source and 'physician-labels.jsonl' not in source
    assert "p.add_argument('--real', action='store_true'" in source


def test_group_bootstrap_deterministic():
    items = [{'topic': str(i % 2), 'labels': [bool(i % 2), True],
              'arms': {'x': {'status': 'Completed', 'decision': 'met' if i < 3 else 'not_met'}}} for i in range(4)]
    assert score(items, 'x', bootstrap=40) == score(deepcopy(items), 'x', bootstrap=40)
