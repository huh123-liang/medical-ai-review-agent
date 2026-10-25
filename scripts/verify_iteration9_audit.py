"""Offline provenance and obligation checks, NOT an independent medical adjudicator."""
from __future__ import annotations
import hashlib
import json
import re
import socket
import sqlite3
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from prepare_iteration8_audit import ROOT,read,write,sha,ledger
from medical_jury.domain import digest
from criterion_boundary_rules import audit_conjunction

OUT=ROOT/'reports/iteration-9-source-audit'


def source_text(record):
    path=(ROOT/record['text_file']).resolve()
    assert path.is_relative_to(ROOT.resolve())
    assert sha(path)==record['text_sha256']
    return path.read_text(encoding='utf-8')


def original_reason(entry):
    if entry['report']:return entry['report']['judgments'][0]['reason']
    info=entry['raw_output']
    if info['kind']=='exact_original_output':
        assert hashlib.sha256(info['raw'].encode()).hexdigest()==info['original_output_sha256']
        return json.loads(info['raw'])['reason']
    raise ValueError('No complete original judge reason available; cannot fabricate it.')


def main():
    freeze=read(OUT/'scope-freeze.json')
    assert sha(ROOT/'reports/iteration-8-offline/cases.json')==freeze['same_cases_sha256']
    cases={c['audit_id']:c for c in read(ROOT/'reports/iteration-8-offline/cases.json')}
    annotations=read(OUT/'case-boundaries.json')['cases']
    assert len(annotations)==12 and {c['id'] for c in annotations}==set(cases)
    records=read(OUT/'source-manifest.json')['records']+read(OUT/'source-addendum.json')['records']
    sources={r['id']:r for r in records}
    for r in records:
        if r.get('raw_file'):
            p=(ROOT/r['raw_file']).resolve();assert p.is_relative_to(ROOT.resolve()) and sha(p)==r['sha256']
        if r.get('text_file'):source_text(r)
    articles={a['id']:a for a in read(OUT/'primary-publications.json')}
    for a in articles.values():assert sha(ROOT/a['source_snapshot'])==a['source_sha256']
    web={a['id']:a for a in read(OUT/'web-source-observations.json')['records']}
    known=set(sources)|set(articles)|set(web)|{'original-user-record','original-candidate-record'}
    assert 'An alternative diagnosis is at least as likely as DVT -2' in source_text(sources['nice-wells'])
    assert 'Do not use EEG to exclude a diagnosis of epilepsy' in source_text(sources['nice-epilepsy'])
    assert 'level of consciousness' in source_text(sources['nice-observations'])
    assert '4AT' in source_text(sources['nice-delirium'])
    assert articles['pmid-32417976']['journal'].casefold()=='journal of anesthesia'
    assert articles['pmid-30292760']['journal']=='Chest'
    assert 'phase IIb' in articles['pmid-26363333']['title']
    assert 'phase III' in articles['pmid-29723512']['title']
    assert articles['pmid-29229594']['journal']=='Blood'
    assert '2018' in articles['pmid-29229594']['publication_date']
    assert 'objective response rate of 82%' in ' '.join(articles['pmid-29229594']['abstract'])
    assert 'was 85%, with 67%' in ' '.join(articles['pmid-33827139']['abstract'])
    assert 'nonulcerated lesions < 0.8 mm' in ' '.join(articles['pmid-29232171']['abstract'])
    rows=[]
    def offline(*args,**kwargs):raise AssertionError('No network during audit verification.')
    with patch.object(socket.socket,'connect',offline),patch.object(socket,'create_connection',offline):
        for a in annotations:
            case=cases[a['id']];candidate=case['case']['candidate_output'];rubric=case['rubric']['text']
            if a['scope']=='strict_factuality':assert 'extraordinarily high bar' in rubric
            if a['scope']=='detailed_appropriateness':
                assert 'Step 1: Is the core question answered clearly and early?' in rubric
                assert len(a['obligations'])==3
            if a['scope']=='simple_appropriateness':assert 'response should be somewhat simple' in rubric
            findings=[]
            for f in a['findings']:
                assert f['sources'] and all(s in known for s in f['sources'])
                for sid in f['sources']:
                    if sid in sources:assert sources[sid]['status']!='UNAVAILABLE_NOT_VERIFIED'
                text=candidate if f['target']=='candidate' else original_reason(case['arms'][f['target'][-1]])
                start=text.find(f['needle']);assert start>=0,(a['id'],f['needle'])
                findings.append({**f,'source_text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                                 'start_char':start,'end_char':start+len(f['needle']),
                                 'verification':'Exact span/source inventory only; clinical relation is researcher assessment.'})
            fids={f['id'] for f in findings}
            assert all(set(o.get('counterexample_ids',[]))<=fids for o in a['obligations'])
            result=audit_conjunction(a['obligations'],strict_factuality=a['scope']=='strict_factuality',
                                     full_claim_coverage=a['full_claim_coverage'])
            rows.append({'audit_id':a['id'],'original_rubric_sha256':digest(case['rubric']),
                         'findings':findings,'obligations':a['obligations'],**result})
    db=sqlite3.connect('file:'+(ROOT/'data/jury.sqlite').as_posix()+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    financial=ledger(db);db.close();assert financial==freeze['budget_before']
    assert {p.name:sha(p) for p in (ROOT/'src/medical_jury').glob('*.py')}==freeze['core_before']
    previous=read(ROOT/'reports/iteration-7/new-manifest.json')
    assert sha(ROOT/'data/private/physician-labels.jsonl')==previous['labels_file_sha256']
    assert sha(ROOT/'data/prepared/dev.jsonl')==previous['dev_file_sha256']
    summary={'status':'SOURCE_PROVENANCE_AND_AUDIT_LOGIC_CHECKED_NOT_CLINICAL_VALIDATION',
             'seen_cases':12,'research_annotations_blinded':False,
             'research_suggestions':dict(Counter(r['audit_suggestion'] for r in rows)),
             'exact_findings_located':sum(len(r['findings']) for r in rows),
             'independent_physician_adjudication':False,'new_model_predictions':0,'new_paid_calls':0,
             'clinical_releases':0,'original_model_or_physician_verdicts_changed':0,
             'runtime_core_unchanged':True,'default_promoted':False,
             'financial_state_unchanged':financial,
             'limits':['Annotations are non-blind researcher interpretations, not new ground truth.',
                       'Source inventory and spans do not independently prove semantic entailment.',
                       'PubMed captures are metadata/abstracts; web-only observations are not full local snapshots.',
                       'Unresolved cases remain unresolved; no new agreement/Kappa or clinical-safety claim.']}
    write(OUT/'located-evidence-and-boundaries.json',rows)
    write(OUT/'verification.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
