import pytest

from medical_jury.evaluation.sampling import natural_sample, panel_sample, representativeness, topic_of


def rows(n_groups=40, per_group=5, split='dev'):
    out = []
    topics = ['hedging', 'communication', 'emergency_referrals', 'global_health']
    for g in range(n_groups):
        for i in range(per_group):
            out.append({'example_id': f'E{g:03d}-{i}', 'category': f'cluster:{topics[g % 4]}_x',
                        'case': {'group_id': f'G{g:03d}', 'split': split}, 'rubric': {}})
    return out


def test_topic_parser():
    assert topic_of('cluster:hedging_any_reducible') == 'hedging'
    assert topic_of('cluster:context_seeking_enough') == 'context_seeking'
    assert topic_of('theme:other') == 'other'


def test_natural_sample_one_item_per_group_and_deterministic():
    data = rows()
    a, ma = natural_sample(data, 10, 7, 'dev')
    b, mb = natural_sample(data, 10, 7, 'dev')
    assert [r['example_id'] for r in a] == [r['example_id'] for r in b]
    assert ma['manifest_sha256'] == mb['manifest_sha256']
    assert len({r['case']['group_id'] for r in a}) == 10 == len(a)


def test_natural_sample_respects_split_and_exclusions():
    data = rows() + rows(split='validation')
    sel, m = natural_sample(data, 39, 1, 'dev', exclude_groups={'G000'})
    assert all(r['case']['split'] == 'dev' and r['case']['group_id'] != 'G000' for r in sel)
    with pytest.raises(ValueError, match='INSUFFICIENT'):
        natural_sample(data, 40, 1, 'dev', exclude_groups={'G000'})


def test_natural_sample_refuses_labels():
    data = rows()
    data[0]['binary_labels'] = [True]
    with pytest.raises(ValueError, match='outcome-blind'):
        natural_sample(data, 3, 1, 'dev')


def test_panel_sample_outcome_blind_and_one_item_per_group():
    rater_rows = []
    for g in range(200):
        rater_rows.append({'example_id': f'E{g}', 'group_id': f'G{g}', 'split': 'validation',
                           'category': 'cluster:hedging_x', 'physician_ids': ['p1', f'q{g % 5}']})
    ids, m = panel_sample(rater_rows, 3, 20, 5, 'validation')
    assert len(ids) == len(set(ids)) == 60
    assert m['n_annotators'] == 3 and not m['shortfall']
    with pytest.raises(ValueError):
        panel_sample([{**rater_rows[0], 'labels': [True]}], 1, 1, 1, 'validation')


def test_representativeness_after_freeze():
    r = representativeness(['a'], {'a': [True, False], 'b': [True, True]}, ['a', 'b'])
    assert r['sample_label_met_rate'] == 0.5 and r['population_label_met_rate'] == 0.75
