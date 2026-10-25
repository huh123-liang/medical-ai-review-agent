"""Final read-only source/database checks; writes only this iteration's evidence."""
import json
import sqlite3
import xml.etree.ElementTree as ET
from prepare_iteration8_audit import ROOT,OUT,read,write,sha,ledger
from medical_jury.config import Settings


def main():
    frozen=read(OUT/'replay-freeze.json')
    assert {p.name:sha(p) for p in (ROOT/'src/medical_jury').glob('*.py')}==frozen['core']
    assert sha(ROOT/'scripts/replay_iteration8_offline.py')==frozen['replay_script_sha256']
    assert sha(OUT/'selection-manifest.json')==frozen['selection_manifest_sha256']
    assert sha(OUT/'逐条深度审计.md')==frozen['audit_sha256']
    old=read(ROOT/'reports/iteration-7/freeze.json')
    assert all(sha(ROOT/'data/facts/v1'/n)==h for n,h in old['fact_files'].items())
    prior=read(ROOT/'reports/iteration-7/new-manifest.json')
    assert sha(ROOT/'data/private/physician-labels.jsonl')==prior['labels_file_sha256']
    assert sha(ROOT/'data/prepared/dev.jsonl')==prior['dev_file_sha256']
    db=sqlite3.connect('file:'+(ROOT/'data/jury.sqlite').as_posix()+'?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    budget=ledger(db);db.close()
    assert budget==read(OUT/'selection-manifest.json')['budget_before']
    s=Settings.load()
    assert s.model=='deepseek-v4-pro' and s.judge_profile=='scoped-review-v1'
    xml=ET.parse(OUT/'offline-tests.xml').getroot()
    suites=xml.findall('testsuite') if xml.tag=='testsuites' else [xml]
    stats={key:sum(int(suite.attrib[key]) for suite in suites) for key in ('tests','errors','failures','skipped')}
    assert stats=={'tests':233,'errors':0,'failures':0,'skipped':0}
    replay=read(OUT/'replay-summary.json')
    assert replay['paid_calls']==0 and replay['exact_raw_replays_accepted_for_review']==8
    report={'status':'OFFLINE_CONTROL_LAYER_CHECKS_PASSED_NOT_CLINICAL_VALIDATION',
            'offline_tests':stats,'test_report_sha256':sha(OUT/'offline-tests.xml'),
            'candidate_frozen_after_replay':True,'original_physician_labels_unchanged':True,
            'development_inputs_unchanged':True,'source_snapshots_unchanged':True,
            'selection_and_audit_unchanged':True,'default_profile':s.judge_profile,'model':s.model,
            'primary_database_financial_state_unchanged':budget,'new_paid_calls':0,
            'candidate_profile':'grounded-review-v2','default_promoted':False,
            'original_platform_accessed_or_modified':False,
            'scope':'Task and evidence review policy candidate only; no original-platform commands, no extra agents, no new sources or index writes.'}
    write(OUT/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
