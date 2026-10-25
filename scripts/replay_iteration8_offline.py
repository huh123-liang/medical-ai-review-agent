"""Exact saved-output replay, separately labelled report projections, NO network/API."""
from __future__ import annotations
import json
import socket
import sqlite3
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import httpx
from medical_jury.domain import EvaluationRequest,digest,validate_judgment
from medical_jury.provider import unique_object
from medical_jury import structured_review
from medical_jury.review_policy import decode_with_sources
from medical_jury.task_spec_v2 import TaskScopeError,task_spec_v2
from medical_jury.task_spec_v3 import task_spec_v3
from medical_jury.fact_sources import FactSourceError
from prepare_iteration8_audit import ROOT,OUT,read,sha,write,ledger


def no_network(*args,**kwargs):raise AssertionError('Network forbidden during saved-output replay.')


def projection(report):
    """Not original model text. Used only for normalized-success compatibility controls."""
    j=report['judgments'][0];r=report['structured_reviews'][0]
    spans=report['evidence_resolution'][0]['spans']
    ids={}
    for field in ('answer_evidence','context_evidence'):
        rows=sorted([s for s in spans if s['field']==field],key=lambda s:s['quote_index'])
        assert [s['source_quote'] for s in rows]==j[field]
        ids[field]=[s['id'] for s in rows]
    return {'judgment':j['judgment'],'reason':j['reason'],
            'answer_ids':ids['answer_evidence'],'context_ids':ids['context_evidence'],
            'review_required':j['review_required'],'task_kind':r['task_kind'],
            'fact_checks':r['fact_checks']}


def main():
    assert read(OUT/'gate-2.json')['status'].startswith('PASS')
    cases=read(OUT/'cases.json');manifest=read(OUT/'selection-manifest.json')
    assert digest(cases)==manifest['cases_hash'] and len(cases)==12
    assert read(OUT/'gate-1.json')['paid_calls']==0
    archive=OUT/'after-code';archive.mkdir(exist_ok=False)
    hashes={}
    for p in sorted((ROOT/'src/medical_jury').glob('*.py')):
        (archive/p.name).write_bytes(p.read_bytes());hashes[p.name]=sha(p)
    frozen={'core':hashes,'replay_script_sha256':sha(Path(__file__)),
            'cases_hash':manifest['cases_hash'],'selection_manifest_sha256':sha(OUT/'selection-manifest.json'),
            'audit_sha256':sha(OUT/'逐条深度审计.md'),
            'purpose':'Offline parser and policy replay only, not new model generation or clinical validation.'}
    write(OUT/'replay-freeze.json',frozen)
    records=[];tasks=[]
    db=sqlite3.connect(f'file:{(ROOT/"data/jury.sqlite").as_posix()}?mode=ro',uri=True);db.row_factory=sqlite3.Row
    assert ledger(db)==manifest['budget_before']
    with patch.object(socket.socket,'connect',no_network),patch.object(socket,'create_connection',no_network),patch.object(httpx.Client,'send',no_network):
        for case in cases:
            req=EvaluationRequest(case=case['case'],rubrics=[case['rubric']])
            before=task_spec_v2(req.case);after=task_spec_v3(req.case)
            tasks.append({'audit_id':case['audit_id'],'old':before,'new':after})
            for arm,entry in case['arms'].items():
                snap=entry['snapshot'];assert digest(snap)==entry['snapshot_file_hash']
                info=entry['raw_output'];source=info['kind']
                row={'audit_id':case['audit_id'],'arm':arm,'source_kind':source,
                     'historical_job_id':entry['job_id'],'historical_status':entry['original_status'],
                     'historical_error':entry['original_error']}
                if source=='truncated_or_redacted_not_exact':
                    row.update(replay_status='SKIPPED_NO_EXACT_OUTPUT',reason='No reconstruction or paid retry.')
                    records.append(row);continue
                if source=='exact_original_output':
                    assert sha(ROOT/info['diagnostic_file'])==info['diagnostic_sha256']
                    import hashlib
                    assert hashlib.sha256(info['raw'].encode()).hexdigest()==info['original_output_sha256']
                    parsed=json.loads(info['raw'],object_pairs_hook=unique_object)
                    try:
                        with patch.object(structured_review,'fact_review',return_value=snap['fact_review']):
                            old,_,_=structured_review.decode(parsed,req.case,req.rubrics[0],snap['judge_profile'])
                        validate_judgment(old,req.case,True);old_error=None
                    except TaskScopeError:old_error='OUTPUT_TASK_KIND_MISMATCH'
                    except FactSourceError:old_error='OUTPUT_FACT_SOURCE_INVALID'
                    assert old_error==entry['original_error']
                    row['reproduced_old_error']=old_error
                else:
                    parsed=projection(entry['report'])
                    row['source_kind']='normalized_success_report_projection_NOT_RAW'
                expected_verdict=parsed['judgment'];expected_reason=parsed['reason']
                current,trace,policy=decode_with_sources(parsed,req.case,req.rubrics[0],snap['fact_review'])
                validate_judgment(current,req.case,True)
                assert current.judgment==expected_verdict and current.reason==expected_reason
                assert policy['model_review_required']==parsed['review_required']
                assert current.review_required and policy['policy_status']=='NeedsReview'
                assert not policy['automatic_verdict_changes'] and not policy['medical_claims_fully_verified']
                row.update(replay_status='VALID_RESEARCH_RESULT_NEEDS_REVIEW',judgment=current.model_dump(),
                           structured_review=policy,evidence_trace=trace,
                           verdict_and_reason_preserved=True,usable_weighted_score=None)
                records.append(row)
    after_budget=ledger(db);db.close()
    assert after_budget==manifest['budget_before']
    # Fixed negative controls retain disagreement; the parser is not a score optimizer.
    for case_id in ('fixed-26','fixed-67','new-22'):
        pair={r['arm']:r['judgment']['judgment'] for r in records if r['audit_id']==case_id}
        assert pair=={'A':'not_met','B':'met'}
    new_tasks={t['audit_id']:t['new'] for t in tasks}
    assert new_tasks['new-11']['expected_kind']=='rewrite'
    assert not any(c['kind']=='algorithm_request' for c in new_tasks['new-11']['cues'])
    assert new_tasks['fixed-35']['depth']=='record_transformation'
    assert not any(c['kind']=='brief_request' for c in new_tasks['fixed-35']['cues'])
    assert new_tasks['fixed-38']['depth']=='record_assessment'
    assert new_tasks['fixed-26']['mixed_intent']
    assert new_tasks['fixed-64']['expected_kind']=='rewrite'
    assert new_tasks['fixed-67']['depth']=='urgent_help_not_full_workup'
    summary={'status':'OFFLINE_REPLAY_PASSED_NOT_MODEL_OR_CLINICAL_VALIDATION',
             'selected_cases':len(cases),'records':len(records),
             'exact_raw_replays':sum(r['source_kind']=='exact_original_output' for r in records),
             'exact_raw_replays_accepted_for_review':sum(r['source_kind']=='exact_original_output' and r['replay_status']=='VALID_RESEARCH_RESULT_NEEDS_REVIEW' for r in records),
             'success_projection_checks':sum(r['source_kind']=='normalized_success_report_projection_NOT_RAW' for r in records),
             'skipped_without_exact_output':sum(r['replay_status']=='SKIPPED_NO_EXACT_OUTPUT' for r in records),
             'automatic_verdict_or_reason_changes':0,'clinical_releases':0,'paid_calls':0,
             'budget_unchanged':after_budget,'network':'Socket connections and httpx send blocked during replay.',
             'limitations':['Seen root-cause-selected diagnostic cases, not an independent sample.',
                            'Old model outputs did not see the new task hints or prompt.',
                            'Normalized report projections are not original response replays.',
                            'Valid-with-review does not mean correct clinical judgment or improved physician agreement.',
                            'All candidate results require review; selective coverage/calibration remains unvalidated.']}
    write(OUT/'replay-records.json',records);write(OUT/'task-hints-before-after.json',tasks)
    write(OUT/'replay-summary.json',summary)
    lines=['# 原始输出离线回放验证','',
           '仅解析与策略回放，不重新生成、不改判、不计算新的医生一致率。','',
           '| 检查 | 数量 |','| --- | --- |',
           f'| 审计案例 | {len(cases)} |',
           f'| 完整原始输出回放，旧错误复现后新策略进入需复核 | {summary["exact_raw_replays_accepted_for_review"]}/{summary["exact_raw_replays"]} |',
           f'| 成功报告投影兼容性检查（非原始输出） | {summary["success_projection_checks"]} |',
           f'| 缺完整输出跳过 | {summary["skipped_without_exact_output"]} |',
           '| 原研究判定/理由自动更改 | 0 |','| 临床放行 | 0 |','| 新付费调用 | 0 |','',
           'new-11恢复为改写提示；fixed-35不再把short course当篇幅要求；fixed-38区分病历评估与代写；fixed-26保留混合意图。任务提示均非权威金标准。','',
           'fixed-26、fixed-67、new-22的A not_met/B met原分歧仍存在，未用规则追平医生标签。fixed-65的来源不足理由没有通过解析器变成医学正确；仍需进一步语义核查。','',
           f'费用账本和任务计数完全未变：{after_budget}。','',
           '旧任务状态、原标签及旧结果未更新；修复只存在版本化候选grounded-review-v2，默认配置未推广。']
    (OUT/'回放验证结果.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
