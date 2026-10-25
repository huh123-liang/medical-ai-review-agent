"""Save a source-grounded, non-clinical boundary review before new-case selection."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    baseline=json.loads((ROOT/'reports/iteration-4/paired-analysis.json').read_text(encoding='utf-8'))
    out=ROOT/'reports/iteration-5';out.mkdir(exist_ok=True)
    ids=(61,62,64,66,67)
    review={'reference':'docs/ITERATION-5-BOUNDARIES-AND-PROTOCOL.md','review_type':'Task/rubric interpretation, not new physician adjudication',
            'original_labels_modified':False,'rubric_text_modified':False,
            'old_case_items':[baseline['items'][i-1] for i in ids],
            'findings':['Rewrite-task misinterpretation is observable from the request.',
                        'Strict factuality cannot be weakened to material-errors-only.',
                        'The urgent headache answer already mentions urgent assessment in its opening.',
                        'Missing-caveat and added-sentence thresholds still require reviewer calibration.']}
    (out/'boundary-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'gate-1.json').write_text(json.dumps({'status':'PASS_TO_BOUNDED_CANDIDATE_CALIBRATION',
        'objective_alignment':'Reduce task/rubric boundary errors, not memorize labels or certify clinical truth.',
        'architecture_alignment':'One Agent, same immutable input/rubric, deterministic scope classification and output validation.',
        'platform_alignment':'No original platform changes; future Jury remains separate from doctor scores.',
        'remaining_ambiguities':['Clinical materiality of omitted caveats.','Reference-label factual disputes.'],
        'next':'Implement one candidate and test only old complex-response cases before freezing new development selection.'},indent=2),encoding='utf-8')
    print(json.dumps({'gate_1':'PASS_WITH_EXPLICIT_REFERENCE_LIMITS','new_cases_viewed':0,'paid_calls':0}))

if __name__=='__main__':main()
