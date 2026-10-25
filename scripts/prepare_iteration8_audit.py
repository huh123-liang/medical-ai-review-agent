"""Freeze a root-cause-selected, offline-only 12-case audit. No provider calls."""
from __future__ import annotations
import hashlib
import json
import sqlite3
from pathlib import Path
from medical_jury.domain import digest

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/iteration-8-offline'
SELECTION=[
 ('fixed',26,'mixed_intent_safety','Medical question plus English correction: do not excuse safety content as rewriting.'),
 ('fixed',35,'incidental_short_and_invented_record','Short course is history, not a request for a short answer; note contains unsupported details.'),
 ('fixed',38,'evaluate_vs_create','Assess an existing note rather than demand creation of a new note.'),
 ('fixed',62,'source_label_conflict','A sourced conflict with the physician labels requires adjudication, not label rewriting.'),
 ('fixed',63,'valid_output_insufficient_evidence','Both original JSON outputs exist but lack the required review flag.'),
 ('fixed',64,'stable_suffix_rewrite_control','Positive control: explicit suffix paraphrase should remain a rewrite.'),
 ('fixed',65,'no_source_not_false_and_truncated_log','Unknown medical evidence must not become proof of falsehood; one raw log is truncated.'),
 ('fixed',67,'urgent_scope_vs_actual_bulk','An urgent task does not license either demanding a full workup or overlooking bulk.'),
 ('new',3,'source_applicability_and_missing_review','Different numerical thresholds and guideline applicability need source review, not memory certainty.'),
 ('new',5,'internal_consistency_vs_external_verification','Keep internal numerical contradictions distinct from externally sourced medical truth.'),
 ('new',11,'instruction_vs_pasted_letter','A parent-letter closing says step by step; the outer task is editing.'),
 ('new',22,'semantic_regression_negative_control','A valid protocol must not be misreported as a repaired semantic verdict.'),
]


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def ledger(db):
    costs=[dict(r) for r in db.execute('SELECT * FROM costs ORDER BY id')]
    return {'jobs':db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0],
            'cost_records':len(costs),'costs_sha256':digest(costs),
            'accounted_cny':round(sum(r['accounted'] for r in costs),6),
            'reserved':sum(r['state']=='reserved' for r in costs)}


def main():
    OUT.mkdir(exist_ok=False)
    archive=OUT/'before-code';archive.mkdir()
    hashes={}
    for p in sorted((ROOT/'src/medical_jury').glob('*.py')):
        (archive/p.name).write_bytes(p.read_bytes());hashes[p.name]=sha(p)
    originals=read(ROOT/'reports/iteration-7/originals-audit.json')
    assert originals['inputs_rubrics_labels_match_original']
    analyses={p:read(ROOT/f'reports/iteration-7/{p}/analysis.json') for p in ('fixed','new')}
    db=sqlite3.connect(f'file:{(ROOT/"data/jury.sqlite").as_posix()}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    cases=[]
    for phase,index,root_cause,why in SELECTION:
        item=analyses[phase]['items'][index-1];assert item['index']==index
        arms={}
        for a,p in item['arms'].items():
            job=dict(db.execute('SELECT * FROM jobs WHERE id=?',(p['job_id'],)).fetchone())
            snap=json.loads(job['snapshot_json'])
            assert digest(snap['case'])==digest(json.loads(job['request_json'])['case'])
            files=list((ROOT/'data/private/provider-diagnostics'/p['job_id']).glob('*.json'))
            raw_info={'kind':'unavailable_success_raw_not_retained','raw':None}
            if files:
                assert len(files)==1
                diag=read(files[0]);raw=diag.get('redacted_output_excerpt')
                exact=bool(raw and not diag.get('excerpt_truncated') and hashlib.sha256(raw.encode()).hexdigest()==diag.get('output_sha256'))
                raw_info={'kind':'exact_original_output' if exact else 'truncated_or_redacted_not_exact',
                          'raw':raw if exact else None,'diagnostic_file':str(files[0].relative_to(ROOT)),
                          'diagnostic_sha256':sha(files[0]),'original_output_sha256':diag.get('output_sha256')}
            report=json.loads(job['report_json']) if job['report_json'] else None
            arms[a]={'job_id':p['job_id'],'original_status':p['status'],'original_error':p['error_code'],
                     'snapshot':snap,'snapshot_file_hash':digest(snap),'raw_output':raw_info,
                     'report':report}
        first=arms['A']['snapshot']
        assert first['case']==arms['B']['snapshot']['case'] and first['rubrics']==arms['B']['snapshot']['rubrics']
        cases.append({'audit_id':f'{phase}-{index:02d}','phase':phase,'index':index,
                      'example_id':item['example_id'],'category':item['category'],
                      'root_cause':root_cause,'selection_reason':why,
                      'original_line':originals['original_rows'][item['example_id']]['original_line'],
                      'original_physician_labels':item['labels'],'case':first['case'],
                      'rubric':first['rubrics'][0],'arms':arms})
    plan={'stage':'offline-deep-audit-before-repair','selection_basis':'Root-cause coverage plus positive/negative controls, not score maximization. These are seen diagnostic cases, not a new validation set.',
          'selected_cases':len(cases),'selected_ids':[c['audit_id'] for c in cases],
          'core_before':hashes,'original_file_sha256':originals['original_file_sha256'],
          'budget_before':ledger(db),'paid_calls_authorized':False,'original_platform_changes_authorized':False,
          'cases_hash':digest(cases)}
    write(OUT/'cases.json',cases);write(OUT/'selection-manifest.json',plan)
    db.close()
    print(json.dumps({'selected':len(cases),'raw_availability':{k:sum(a['raw_output']['kind']==k for c in cases for a in c['arms'].values()) for k in ('exact_original_output','unavailable_success_raw_not_retained','truncated_or_redacted_not_exact')},'budget':plan['budget_before']},indent=2))


if __name__=='__main__':main()
