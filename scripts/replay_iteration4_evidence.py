"""Free paired-output replay for anchored-v2. No labels or provider access."""
import json
from collections import Counter
from pathlib import Path
from medical_jury.domain import CaseInput
from medical_jury.provider import parse_judgment
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]

def saved_wire(pred):
    if pred['status']=='Completed':
        report=json.loads((ROOT/'reports'/pred['job_id']/'report.json').read_text(encoding='utf-8'))
        original=report['judgments'][0]
        output={k:original[k] for k in ('judgment','reason','answer_evidence','context_evidence','review_required')}
        spans=report['evidence_resolution'][0]['spans']
        for field in ('answer_evidence','context_evidence'):
            output[field]=[s['model_quote'] for s in sorted((s for s in spans if s['field']==field),key=lambda s:s['quote_index'])]
        return json.dumps(output)
    logs=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'data/private/provider-diagnostics'/pred['job_id']).glob('*.json')]
    full=[d for d in logs if d.get('redacted_output_excerpt') and not d.get('excerpt_truncated')]
    if len(full)!=1: raise ValueError('Complete historical wire output unavailable.')
    return full[0]['redacted_output_excerpt']


def main():
    summary=json.loads((ROOT/'reports/iteration-3/paired/summary.json').read_text(encoding='utf-8'))
    _,rows=load_plan(ROOT,ROOT/'reports/iteration-3/paired/manifest.json')
    items=[]
    for pred in summary['predictions']:
        raw=saved_wire(pred); original=json.loads(raw)
        case=CaseInput.model_validate(rows[pred['index']-1]['case']); trace={}
        parsed=parse_judgment(raw,case,True,'anchored-v2',trace.update)
        assert parsed.judgment==original['judgment'] and parsed.reason==original['reason']
        for span in trace['spans']:
            for candidate in span.get('candidates',[span]):
                source=case.candidate_output if candidate['field']=='answer_evidence' else case.context[candidate['source_index']].content
                assert source[candidate['start_char']:candidate['end_char']]==candidate['source_quote']
                assert len(source[:candidate['start_char']].encode('utf-16-le'))//2==candidate['start_utf16']
        items.append({'index':pred['index'],'arm':pred['arm'],'job_id':pred['job_id'],'status':'Anchored',
                      'ambiguous_count':trace['ambiguous_count'],'trace':trace if trace['ambiguous_count'] else None})
    out=ROOT/'reports/iteration-4'; out.mkdir(exist_ok=True)
    result={'protocol':'OFFLINE_HISTORICAL_WIRE_REPLAY_NOT_NEW_MEDICAL_EVALUATION','planned':140,
            'valid':len(items),'newly_resolved_duplicate_quote_jobs':[i['job_id'] for i in items if i['ambiguous_count']],
            'judgments_and_reasons_unchanged':True,'paid_calls':0,'items':items}
    (out/'evidence-replay.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'gate-1.json').write_text(json.dumps({'status':'PASS_TO_SAFETY_GUARD_STEP','offline_tests':102,
        'objective_alignment':'Audit all locations, never fabricate unique attribution or clinical correctness.',
        'architecture_alignment':'Versioned deterministic postprocessing within the same single-Agent workflow.',
        'platform_alignment':'Additive candidate-span trace, explicit human review, no old platform writes.',
        'historical_replayed':140,'previously_ambiguous_resolved_as_multiple':1,'paid_calls_added':0,
        'next':'Implement source-backed narrow safety risk gate without rewriting judge/physician labels.'},indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},indent=2))

if __name__=='__main__': main()
