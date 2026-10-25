"""Reconcile local experiment inputs and labels directly to original public rows."""
from __future__ import annotations
import hashlib
import json
from collections import Counter
from medical_jury.benchmark import convert_row
from medical_jury.domain import EvaluationRequest
from medical_jury.task_spec_v2 import task_spec_v2
from run_iteration7 import ROOT,OUT,load_rows,dump,check_freeze


def main():
    check_freeze()
    wanted={r['example_id']:r for p in ('fixed-first','fixed-second','new') for r in load_rows(p)[1]}
    labels={r['example_id']:r for r in map(json.loads,(ROOT/'data/private/physician-labels.jsonl').read_text(encoding='utf-8').splitlines()) if r['example_id'] in wanted}
    manifest=json.loads((ROOT/'data/public/source-manifest.json').read_text())
    rawfile=ROOT/'data/public/healthbench-meta.jsonl'
    hasher=hashlib.sha256()
    with rawfile.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):hasher.update(chunk)
    assert hasher.hexdigest()==manifest['sha256']
    found={}
    with rawfile.open(encoding='utf-8') as f:
        for line_no,line in enumerate(f,1):
            raw=json.loads(line)
            try:public,private=convert_row(raw)
            except (ValueError,KeyError,TypeError):continue
            eid=public['example_id']
            if eid not in wanted:continue
            assert public==wanted[eid] and private==labels[eid]
            found[eid]={'original_line':line_no,'original_labels':raw['binary_labels'],
                        'anonymized_raters':raw['anonymized_physician_ids']}
    assert set(found)==set(wanted)
    failure_counts={};preserved_complete_public_outputs=0;truncated_outputs=0
    for phase in ('fixed','new'):
        result=json.loads((OUT/phase/'analysis.json').read_text(encoding='utf-8'))
        reasons=Counter()
        for item in result['items']:
            for arm,p in item['arms'].items():
                if p['status']=='Completed':continue
                for file in (ROOT/'data/private/provider-diagnostics'/p['job_id']).glob('*.json'):
                    diag=json.loads(file.read_text(encoding='utf-8'))
                    if not diag.get('redacted_output_excerpt'):continue
                    truncated_outputs+=int(diag.get('excerpt_truncated',True))
                    if diag.get('excerpt_truncated',True):continue
                    preserved_complete_public_outputs+=1
                    try:wire=json.loads(diag['redacted_output_excerpt'])
                    except ValueError:continue
                    if p['error_code']=='OUTPUT_FACT_SOURCE_INVALID' and wire.get('fact_checks')==[] and wire.get('review_required') is False:
                        reasons['strict_fact_no_card_review_false']+=1
                    if p['error_code']=='OUTPUT_TASK_KIND_MISMATCH':
                        row=wanted[item['example_id']]
                        spec=task_spec_v2(EvaluationRequest(case=row['case'],rubrics=[row['rubric']]).case)
                        reasons[f'heuristic_expected_{spec.get("expected_kind")}_model_{wire.get("task_kind")}']+=1
        failure_counts[phase]=dict(reasons)
    report={'status':'ORIGINAL_PUBLIC_ROW_RECONCILIATION_PASSED','examples_checked':len(wanted),
            'original_file_sha256':hasher.hexdigest(),'original_version':manifest['version'],
            'inputs_rubrics_labels_match_original':True,'original_rows':found,
            'failure_output_audit':failure_counts,'complete_redacted_failure_outputs':preserved_complete_public_outputs,
            'truncated_failure_outputs':truncated_outputs,
            'limits':['Exact reconciliation validates lineage, not medical truth.',
                      'Failures retain bounded redacted public excerpts; private inputs retain hashes only.',
                      'Missing/truncated original outputs cannot be reconstructed from hashes.']}
    dump(OUT/'originals-audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='original_rows'},indent=2))


if __name__=='__main__':main()
