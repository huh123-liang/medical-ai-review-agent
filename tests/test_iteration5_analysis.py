import runpy
from pathlib import Path

FUNCTIONS=runpy.run_path(str(Path(__file__).parents[1]/'scripts/analyze_iteration5.py'))

def item(a_status='Completed',a='met',b_status='Completed',b='met',labels=None):
    return {'category':'fixture','labels':labels or [True,False],
            'arms':{'A':{'status':a_status,'judgment':a},'B':{'status':b_status,'judgment':b}}}

def test_failure_counts_as_undelivered_not_a_false_binary_prediction():
    scores=FUNCTIONS['compare']([item(labels=[False,False],b_status='Failed',b=None)])
    assert scores['B']['delivered_physician_match_rate']==0
    assert scores['B']['completed']==0 and scores['common_valid_examples']==0
    assert scores['common_valid_metrics']['B']['agreement'] is None

def test_each_physician_keeps_weight_not_majority_collapsed():
    scores=FUNCTIONS['compare']([item(labels=[True,True,False])])
    assert scores['A']['planned_label_comparisons']==3
    assert scores['A']['delivered_physician_match_rate']==2/3

def test_bootstrap_equal_arms_zero_delta_with_missingness():
    result=FUNCTIONS['bootstrap']([item(),item(a_status='Failed',a=None,b_status='Failed',b=None)],iterations=20)
    assert result['delivered_match_rate_delta_95ci']==[0,0]
