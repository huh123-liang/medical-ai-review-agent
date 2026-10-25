import importlib.util
import sys
from pathlib import Path
import pytest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
from analyze_iteration6 import compare, verify_metrics
from run_iteration6 import code_hashes


def item(index, gold, a='met', b='met'):
    def prediction(value):
        return {'status':'Completed' if value is not None else 'Failed','judgment':value}
    return {'index':index,'category':'one','labels':gold,'arms':{'A':prediction(a),'B':prediction(b)}}


def test_failed_predictions_remain_in_planned_denominator():
    s=compare([item(1,[True,True]),item(2,[False,True],None,'not_met')])
    assert s['A']['delivered_physician_match_rate']==.5
    assert s['B']['delivered_physician_match_rate']==.75
    assert s['common_valid_examples']==1


def test_each_doctor_retained_not_majority_voted():
    s=compare([item(1,[True,False])])
    assert s['common_valid_metrics']['A']['agreement']==.5
    assert s['common_valid_metrics']['A']['n_physician_comparisons']==2


def test_kappa_independently_checked():
    s=compare([item(1,[True,True]),item(2,[False,False],'not_met','met')])
    verify_metrics(s)
    assert s['common_valid_metrics']['A']['cohen_kappa']==1
    assert s['common_valid_metrics']['B']['cohen_kappa']==0


def test_all_failures_do_not_fabricate_binary_predictions():
    s=compare([item(1,[True,True],None,None)])
    verify_metrics(s)
    assert s['common_valid_metrics']['A']['agreement'] is None
    assert s['A']['coverage']==0


def test_task_module_is_part_of_source_freeze():
    assert 'task_scope.py' in code_hashes()
