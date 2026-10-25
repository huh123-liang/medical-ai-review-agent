"""Offline scope check, not clinical sensitivity/specificity measurement."""
import json
from pathlib import Path
from medical_jury.domain import CaseInput
from medical_jury.safety import screen_case,release_gate
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]

def main():
    _,rows=load_plan(ROOT,ROOT/'reports/iteration-3/paired/manifest.json')
    records=[]
    for index,row in enumerate(rows,1):
        screening=screen_case(CaseInput.model_validate(row['case']))
        if screening['findings']: records.append({'index':index,'screening':screening})
    assert any(r['index']==27 for r in records), 'Known observed missed hazard was not caught.'
    # Check the downstream safety gate, without editing the historical model's met verdict.
    assert release_gate(1,screen_case(CaseInput.model_validate(rows[26]['case'])),[])['weighted_score'] is None
    out=ROOT/'reports/iteration-4'
    result={'protocol':'OFFLINE_NARROW_RULE_SCREEN_NOT_CLINICAL_VALIDATION','screened_cases':70,
            'flagged_indices':[r['index'] for r in records],'known_unsafe_advice_now_requires_review':True,
            'model_binary_judgment_changed':False,'clinical_sensitivity':None,'clinical_specificity':None,
            'paid_calls':0,'findings':records}
    (out/'safety-replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'gate-2.json').write_text(json.dumps({'status':'PASS_TO_COMPLEX_RESPONSE_EXPERIMENT',
        'objective_alignment':'Prevent known high-risk advice from silently yielding a usable score, not claim complete medical correctness.',
        'architecture_alignment':'One source-backed deterministic screen in the same workflow, no autonomous tool or second Agent.',
        'platform_alignment':'Separate research judgment, withheld usable score, source-backed alert and human-review status for future Admin.',
        'label_isolation':True,'historical_judgments_unchanged':True,'clinical_validation':False,
        'paid_calls_added':0,'scope_limit':'One English-language risk rule; absence of alerts does not mean safe.',
        'next':'Test single-Agent structured criterion checks; retain baseline unless controlled evidence supports promotion.'},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='findings'},indent=2))

if __name__=='__main__': main()
