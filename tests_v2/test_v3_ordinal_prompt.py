"""Step 5 (after the dev-A pilot): anchor-only prompt v3-ordinal-1, dev-A / dev-B subsets and the paired
comparison. Synthetic data and the offline mock judge only."""
from __future__ import annotations

import hashlib
import json

import pytest

from medical_jury.domain import CaseInput
from medical_jury.evaluation import ordinal_run as R
from medical_jury.evaluation.ordinal import compare_runs
from medical_jury.v3.evaluator import (ORDINAL_PROMPT_VERSION, PROMPT_VERSIONS, build_messages_group,
                                       build_messages_v3, mock_responder_v3, parse_group, system_prompt)
from medical_jury.v3.schema import RubricItem

from test_v3_ordinal_run import make_root, runner
from v2_fixtures import load_script


def test_earlier_prompts_are_unchanged():
    sha = lambda s: hashlib.sha256(s.encode()).hexdigest()
    assert sha(system_prompt('v3-dimension-1')) == 'fc64d2061b18eefc91cdf041d637baff7985ae6f5f93d2ce21b05f2649bde1ab'
    # grouped v3-dimension-2 exactly as used by the 2026-10-26 dev-A pilot
    assert sha(system_prompt('v3-dimension-2', grouped=True)) == \
        '94bdb5f6f2875acf89fa0f406bedd4ba46c3635ae2c019798627573a60ab9543'
    assert ORDINAL_PROMPT_VERSION in PROMPT_VERSIONS


def test_ordinal_prompt_is_anchor_only_and_has_no_item_rules():
    for grouped in (False, True):
        text = system_prompt(ORDINAL_PROMPT_VERSION, grouped=grouped)
        assert '"checks": []' in text and 'clinical significance' in text
        assert 'doubtful' not in text and 'steps' not in text and 'Reference notes' not in text


def _case():
    return CaseInput.model_validate({'case_id': 'c1', 'group_id': 'g1', 'task': 'Answer the question.',
                                     'context': [{'role': 'user', 'content': 'Is ibuprofen safe with a cold?'}],
                                     'candidate_output': 'Usually yes, at the labelled dose.'})


def test_ordinal_prompt_refuses_check_items(tmp_path):
    root = make_root(tmp_path)
    profile = R.profile_for(root, 'bionlp-trec')
    dim = profile.dimensions[0]
    line = (root / 'kb/v3/general-items.jsonl').read_text(encoding='utf-8').splitlines()[0]
    item = RubricItem.model_validate(json.loads(line))
    with pytest.raises(ValueError, match='anchor-only'):
        build_messages_group(_case(), [(dim, [item], [])], ORDINAL_PROMPT_VERSION)
    with pytest.raises(ValueError, match='anchor-only'):
        build_messages_v3(_case(), dim, [item], [], ORDINAL_PROMPT_VERSION)


def test_ordinal_prompt_output_parses(tmp_path):
    root = make_root(tmp_path)
    profile = R.profile_for(root, 'bionlp-trec')
    specs = [(d, [], []) for d in profile.dimensions]
    messages = build_messages_group(_case(), specs, ORDINAL_PROMPT_VERSION)
    parsed = parse_group(mock_responder_v3(messages), _case(), specs)
    assert set(parsed) == {d.key for d in profile.dimensions}
    assert all(isinstance(p, dict) and p['checks'] == [] for p in parsed.values())


def test_complement_is_the_disjoint_remainder_of_whole_groups():
    rows = [{'item_id': f'{g}-{k}', 'group_id': g} for g in 'abcdefghij' for k in range(3)]
    a, b = R.subset(rows, 7, 's'), R.subset(rows, 7, 's', complement=True)
    assert {r['item_id'] for r in a}.isdisjoint({r['item_id'] for r in b})
    assert len(a) + len(b) == len(rows)
    assert {r['group_id'] for r in a}.isdisjoint({r['group_id'] for r in b})
    with pytest.raises(ValueError):
        R.subset(rows, None, 's', complement=True)


def test_complement_needs_n(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(SystemExit, match='--complement'):
        runner().main(['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'dev', '--judge', 'mock',
                       '--complement'])


def _run(root, *extra):
    before = set((root / 'reports').glob('ordinal-v3-*')) if (root / 'reports').exists() else set()
    assert runner().main(['--root', str(root), '--dataset', 'bionlp-trec', '--split', 'dev', '--judge', 'mock',
                          '--bootstrap', '30', *extra]) == 0
    (new,) = set((root / 'reports').glob('ordinal-v3-*')) - before
    return new


def test_dev_b_runs_of_two_prompts_compare_with_the_preregistered_rule(tmp_path, capsys):
    root = make_root(tmp_path)
    old = _run(root, '--n', '4', '--complement')
    new = _run(root, '--n', '4', '--complement', '--prompt-version', ORDINAL_PROMPT_VERSION)
    assert old.name.startswith('ordinal-v3-bionlp-trec-dev-complement-')
    plan = json.loads((new / 'plan.json').read_text())
    assert plan['subset'] == {'n': 4, 'seed': 'ordinal-v3-dev-subset-1', 'complement': True}
    dev_a = R.subset(R.load_items(root, 'bionlp-trec', 'dev'), 4, 'ordinal-v3-dev-subset-1')
    seen = {json.loads(l)['item_id'] for l in (new / 'predictions.jsonl').read_text().splitlines()}
    assert seen.isdisjoint({i['item_id'] for i in dev_a})
    capsys.readouterr()
    assert load_script('compare_ordinal_runs_v3').main(['--root', str(root), '--a', str(old), '--b', str(new),
                                                        '--bootstrap', '30']) == 0
    out = json.loads(capsys.readouterr().out)
    result = json.loads((next((root / 'reports').glob('ordinal-v3-compare-*')) / 'comparison.json').read_text())
    assert result['a']['prompt_version'] == 'v3-dimension-2' and result['b']['prompt_version'] == ORDINAL_PROMPT_VERSION
    assert out['decision']['adopt_b'] == (result['delta_b_minus_a'] is not None and result['delta_b_minus_a'] > 0)


def test_compare_refuses_different_items_or_tampered_predictions(tmp_path, capsys):
    root = make_root(tmp_path)
    a = _run(root, '--n', '4', '--complement')
    b = _run(root, '--n', '4')
    with pytest.raises(SystemExit, match='item_ids_sha256'):
        load_script('compare_ordinal_runs_v3').main(['--root', str(root), '--a', str(a), '--b', str(b)])
    c = _run(root, '--n', '4', '--complement', '--prompt-version', ORDINAL_PROMPT_VERSION)
    with (c / 'predictions.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(SystemExit, match='seal'):
        load_script('compare_ordinal_runs_v3').main(['--root', str(root), '--a', str(a), '--b', str(c)])


def test_compare_runs_statistics():
    class P:
        dimensions = [type('D', (), {'key': 'x', 'scale_min': 0, 'scale_max': 2})()]
    gold = [0, 1, 2, 2, 1, 0, 2, 1]
    base = [{'item_id': f'i{k}', 'group_id': f'g{k // 2}', 'status': 'Completed', 'raters': ['A'],
             'scores': {'x': [g]}} for k, g in enumerate(gold)]
    a = [{**r, 'pred': {'x': 0}} if k % 2 else {**r, 'pred': {'x': gold[k]}} for k, r in enumerate(base)]
    b = [{**r, 'pred': {'x': gold[k]}} for k, r in enumerate(base)]
    same = compare_runs(a, a, P, ['x'], n_boot=20)
    assert same['delta_b_minus_a'] == 0
    better = compare_runs(a, b, P, ['x'], n_boot=20)
    assert better['mean_primary_qwk_b'] == pytest.approx(1.0) and better['delta_b_minus_a'] > 0
    failed = [{**b[0], 'status': 'Failed', 'pred': {}}] + b[1:]
    assert compare_runs(a, failed, P, ['x'], n_boot=20)['dropped'] == 1
    with pytest.raises(ValueError):
        compare_runs(a, b[1:], P, ['x'])
