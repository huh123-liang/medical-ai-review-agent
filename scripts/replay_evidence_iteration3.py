"""Replay saved public outputs through deterministic anchoring without LLM calls."""
import json
from collections import Counter
from pathlib import Path

from medical_jury.provider import parse_judgment
from medical_jury.evidence import EvidenceResolutionError
from medical_jury.domain import CaseInput
from medical_jury.stratified import load_plan


def main():
    root=Path(__file__).resolve().parents[1]
    baseline=root/'reports/stratified-20261024-194713'
    plan,rows=load_plan(root,baseline/'manifest.json')
    summary=json.loads((baseline/'summary.json').read_text(encoding='utf-8'))
    results=[]
    for i,(row,pred) in enumerate(zip(rows,summary['predictions']),1):
        raw=None
        if pred['status']=='Completed':
            report=json.loads((root/'reports'/pred['job_id']/'report.json').read_text(encoding='utf-8'))
            judgment=report['judgments'][0]
            raw=json.dumps({k:v for k,v in judgment.items() if k not in {'rubric_id','weight'}},ensure_ascii=False)
        else:
            for path in (root/'data/private/provider-diagnostics'/pred['job_id']).glob('*.json'):
                log=json.loads(path.read_text(encoding='utf-8'))
                if not log.get('excerpt_truncated'):
                    raw=log.get('redacted_output_excerpt')
        out={'index':i,'example_id':row['example_id'],'baseline_status':pred['status'],
             'baseline_judgment':pred['judgment']}
        if raw is None:
            out.update(status='Unavailable',reason='Historical raw output was truncated; not fabricated or regenerated.')
        else:
            trace={}
            try:
                value=parse_judgment(raw,CaseInput.model_validate(row['case']),True,'anchored-v1',trace.update)
                original=json.loads(raw)['judgment']
                assert value.judgment==original
                assert value.reason==json.loads(raw)['reason']
                out.update(status='Anchored',judgment=value.judgment,trace=trace)
            except EvidenceResolutionError as error:
                out.update(status='Rejected',issue=error.issue)
        results.append(out)
    output={'mode':'OFFLINE_REPLAY_NOT_NEW_MODEL_EVALUATION','paid_calls':0,
            'counts':dict(Counter(r['status'] for r in results)),
            'rescued_saved_failures':[r['index'] for r in results if r['baseline_status']=='Failed' and r['status']=='Anchored'],
            'original_decisions_and_reasons_unchanged':True,'items':results}
    folder=root/'reports/iteration-3'
    folder.mkdir(exist_ok=True)
    path=folder/'step1-evidence-replay.json'
    with path.open('x',encoding='utf-8') as file:
        json.dump(output,file,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in output.items() if k!='items'},ensure_ascii=False,indent=2))
    print(json.dumps({'rejected':[{k:v for k,v in r.items() if k!='trace'} for r in results if r['status']=='Rejected']},indent=2))


if __name__=='__main__':
    main()
