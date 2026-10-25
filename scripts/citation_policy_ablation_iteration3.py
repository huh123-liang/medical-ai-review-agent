"""Citation-only policy ablation on the same saved wire quotes. Zero provider calls."""
import json
from collections import Counter
from pathlib import Path
from medical_jury.domain import CaseInput
from medical_jury.provider import parse_judgment
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-3'


def main():
    _,rows=load_plan(ROOT,OUT/'paired/manifest.json')
    summary=json.loads((OUT/'paired/summary.json').read_text(encoding='utf-8'))
    assert summary['state']=='Finished'
    result=[]
    for pred in summary['predictions']:
        case=CaseInput.model_validate(rows[pred['index']-1]['case'])
        if pred['status']=='Completed':
            report=json.loads((ROOT/'reports'/pred['job_id']/'report.json').read_text(encoding='utf-8'))
            saved=report['judgments'][0]
            # The new policy changes no decision/reason. Reconstruct wire quotations from trace,
            # not the normalized source slice. review_required may already be raised, but its
            # value cannot change citation validity in either parser.
            output={k:saved[k] for k in ('judgment','reason','answer_evidence','context_evidence','review_required')}
            spans=report['evidence_resolution'][0]['spans']
            for field in ('answer_evidence','context_evidence'):
                output[field]=[s['model_quote'] for s in sorted((s for s in spans if s['field']==field),key=lambda s:s['quote_index'])]
            raw=json.dumps(output)
        else:
            logs=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'data/private/provider-diagnostics'/pred['job_id']).glob('*.json')]
            usable=[d for d in logs if d.get('redacted_output_excerpt') and not d.get('excerpt_truncated')]
            if len(usable)!=1: raise ValueError('Cannot reconstruct full saved output; do not invent it.')
            raw=usable[0]['redacted_output_excerpt']
        item={'index':pred['index'],'arm':pred['arm'],'job_id':pred['job_id']}
        for policy in ('strict-v2','anchored-v1'):
            try:
                parsed=parse_judgment(raw,case,True,policy)
                item[policy]='Valid'
            except ValueError as error:
                item[policy]=type(error).__name__
        assert (item['anchored-v1']=='Valid')==(pred['status']=='Completed')
        result.append(item)
    counts={arm:{policy:dict(Counter(r[policy] for r in result if r['arm']==arm)) for policy in ('strict-v2','anchored-v1')} for arm in ('A','B')}
    artifact={'protocol':'OFFLINE_SAME_SAVED_WIRE_QUOTES_CITATION_VALIDITY_ONLY','paid_calls':0,'counts':counts,'items':result,
        'limitations':['This compares parser acceptance, not medical validity or clinical correctness.',
            'Normalized review flags reconstructed as true where raised; citation validity is independent of that flag.',
            'Original raw decisions, reasons, labels and historical jobs are not edited.']}
    (OUT/'citation-policy-ablation.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'counts':counts,'paid_calls':0,'changed':[r for r in result if (r['strict-v2']=='Valid')!=(r['anchored-v1']=='Valid')]},indent=2))

if __name__=='__main__': main()
