import pytest

from medical_jury.stratified import LABEL_QUOTAS, TOPICS, scenario_tag, select_rows


def fixtures():
    rows,refs=[],{}
    for topic in TOPICS:
        for pattern,labels in [('positive',[True,True]),('negative',[False,False]),('split',[True,False])]:
            for index in range(15):
                key=f'{topic}-{pattern}-{index}'
                row={'example_id':key,'category':'cluster:'+topic+'_fixture','case':{
                     'split':'dev','group_id':'group-'+key,'context':[{'content':'diabetes' if index%2 else 'general question'}]}}
                rows.append(row)
                refs[key]={'binary_labels':labels}
    return rows,refs


def test_sample_is_deterministic_and_independent_of_source_order():
    rows,refs=fixtures()
    a=select_rows(rows,refs)
    b=select_rows(list(reversed(rows)),refs)
    assert [r['example_id'] for r in a]==[r['example_id'] for r in b]
    assert len(a)==70 and len({r['case']['group_id'] for r in a})==70


def test_prior_groups_excluded_and_reference_labels_never_added():
    rows,refs=fixtures()
    excluded={select_rows(rows,refs)[0]['case']['group_id']}
    selected=select_rows(rows,refs,excluded_groups=excluded)
    assert not excluded & {r['case']['group_id'] for r in selected}
    assert all('binary_labels' not in r and 'label_pattern' not in r['case'] for r in selected)


def test_missing_stratum_fails_instead_of_silent_replacement():
    rows,refs=fixtures()
    with pytest.raises(ValueError,match='Insufficient'):
        select_rows([r for r in rows if not r['category'].startswith('cluster:hedging_')],refs)


def test_scenario_tag_is_context_based_not_candidate_based():
    assert scenario_tag({'context':[{'content':'insulin and diabetes'}],'candidate_output':'anxiety'})=='chronic_metabolic_renal'
