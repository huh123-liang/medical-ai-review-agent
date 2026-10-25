"""Final local-only verification and architecture checkpoints. Zero paid requests."""
import hashlib
import json
import subprocess
from pathlib import Path
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.domain import Rubric
from medical_jury.store import Store

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'reports/iteration-4'

def main():
    paired=json.loads((OUT/'paired-analysis.json').read_text(encoding='utf-8'))
    smoke=json.loads((OUT/'scoped-smoke/summary.json').read_text(encoding='utf-8'))
    assert smoke['state']=='Finished' and len(smoke['results'])==6 and all(r['status']=='Completed' for r in smoke['results'])
    for path,sha in smoke['preflight']['code_sha256'].items():
        assert hashlib.sha256((ROOT/'src/medical_jury'/path).read_bytes()).hexdigest()==sha
        assert hashlib.sha256((OUT/'scoped-smoke/frozen-code'/path).read_bytes()).hexdigest()==sha
    settings=Settings.load();store=Store(ROOT);budget=Budget(settings,store).total()
    assert settings.judge_profile=='scoped-review-v1' and settings.evidence_policy=='anchored-v2'
    with store.connect() as db:
        count=db.execute('SELECT COUNT(*) FROM rubrics').fetchone()[0]
        our_jobs=[p['job_id'] for p in json.loads((OUT/'paired/summary.json').read_text())['predictions']]+[p['job_id'] for p in smoke['results']]
        cost=sum(db.execute('SELECT SUM(accounted) FROM costs WHERE job_id=?',(job,)).fetchone()[0] for job in our_jobs)
        checked=store.get_job(next(r['job_id'] for r in smoke['results'] if r['index']==27))
    assert checked['report']['weighted_score'] is None and checked['report']['release_status']=='NeedsReview'
    assert abs(cost-paired['batch_accounted_cny']-smoke['batch_accounted_cny'])<1e-6
    assert budget['accounted_cny']<=30 and count==100
    proposed=json.loads((ROOT/'examples/proposed-rubric-v2.json').read_text(encoding='utf-8'))
    assert proposed['approval_status']=='draft' and proposed['eligible_for_current_index'] is False
    try:Rubric.model_validate(proposed)
    except ValueError:pass
    else:raise AssertionError('Design draft unexpectedly became an ingestible approved rubric.')
    git=Path('C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe')
    platform='C:/Users/Lenovo/Desktop/NTU/NTU AI security agents/MVP2'
    status=subprocess.run([str(git),'-C',platform,'status','--porcelain'],check=True,capture_output=True,text=True,encoding='utf-8').stdout.splitlines()
    expected=[' D data-source/data500_v5_Pat1to500.zip',' D design-qa-comparison-final.png',' D design-qa-focus-evaluation.png',
        ' D design-references/selected-evidence-lens-v1.png',' D design-references/selected-safety-review-cockpit.png',
        ' D output/pdf/AI_Medical_Agent_Evaluation_Platform_Showcase_Handbook_CN.pdf',
        ' D output/pdf/AI_Medical_Agent_Evaluation_Platform_Showcase_Handbook_EN.pdf']
    assert status==expected,'Platform status differs from pre-existing baseline; investigate before claiming unchanged.'
    result={'status':'ENGINEERING_GUARDS_AND_PLAN_COMPLETE_SEMANTIC_RELIABILITY_PARTIAL',
        'offline_tests_passed':133,'historical_evidence_replayed':140,'narrow_safety_replay_cases':70,
        'paired_development':{'A_completed':69,'B_completed':67,'planned_per_arm':70,'common_valid':66,
            'A_agreement':paired['common_valid_metrics']['A']['agreement'],'B_agreement':paired['common_valid_metrics']['B']['agreement'],
            'complex_task_A_agreement':paired['by_topic_common_valid']['complex_responses']['A']['agreement'],
            'complex_task_B_agreement':paired['by_topic_common_valid']['complex_responses']['B']['agreement'],
            'global_v4_promoted':False},
        'default_profile':settings.judge_profile,'default_evidence_policy':settings.evidence_policy,
        'targeted_smoke_completed':6,'smoke_is_accuracy_benchmark':False,
        'unchanged_v2_input_count_on_original70':smoke['preflight']['unchanged_v2_input_count'],
        'clinical_validation':False,'complex_response_semantic_errors_fully_resolved':False,
        'existing_case_bound_rubric_records':count,'new_general_rubric_index_built':False,
        'original_platform_status_unchanged':True,'original_platform_preexisting_status':status,
        'github_changed':False,'paid_calls_this_iteration':len(our_jobs),'iteration_accounted_cny':round(cost,6),'budget':budget,
        'remaining_work':['Resolve rubric task/meaning ambiguities with additional reviewed examples.',
            'Validate safety scope outside known English examples; do not claim full safety detection.',
            'Implement applicability/retrieval plan only after review; do not relabel case-bound rules as universal.',
            'Fix genuine quote copying errors without weakening number/unit/pronoun checks.']}
    (OUT/'completion-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    gate3=json.loads((OUT/'gate-3.json').read_text(encoding='utf-8'))
    gate3.update(status='ENGINEERING_PASS_SEMANTIC_RELIABILITY_PARTIAL',offline_tests_passed=133,
        scoped_smoke_completed=6,complex_response_semantic_errors_fully_resolved=False,
        global_v4_promoted=False,default_profile=settings.judge_profile,
        next='Only deliver planned retrieval contract; no actual index migration or platform integration.')
    (OUT/'gate-3.json').write_text(json.dumps(gate3,indent=2),encoding='utf-8')
    (OUT/'gate-4.json').write_text(json.dumps({'status':'PASS_DESIGN_ONLY_NOT_RETRIEVAL_IMPLEMENTATION',
        'objective_alignment':'Define applicability and fair retrieval evaluation before cross-case reuse.',
        'architecture_alignment':'Separate rubric selection, medical facts, and private doctor labels; mandatory rules never replaced by vector top-k.',
        'platform_alignment':'Future Node-to-Jury Admin-only integration with independent automatic and doctor scores, no present platform changes.',
        'draft_not_ingestible':True,'current_index_records':count,'new_index_built':False,
        'plan':'docs/RUBRICS-APPLICABILITY-AND-RETRIEVAL-TEST-PLAN.md',
        'next':'Discuss reviewer availability and new development cases before implementing the next phase.'},indent=2),encoding='utf-8')
    lines=['# 本轮完成情况与下一步','','## 交付判断','',
        '工程防护与检索方案已完成；复杂回答的语义评分稳定性尚未修好，不能宣称所有问题解决。',
        '- 引用歧义：保留全部候选并要求复核，140份历史回放通过；无出处或改词/改单位仍拒绝。',
        '- 安全漏检：窄来源规则可阻止原第27条被静默放行，默认作用域路线真实试跑也给出not_met。阴性不等于医学安全。',
        '- 复杂回答：全局v4在共同10条上70%→50%，没有获准推广。默认无告警任务保留v2相同输入，以阻止新版本退步扩散；旧有错判仍在。',
        '- 检索：已交付可执行的字段契约、四阶段测试、指标口径与拟定门槛，尚未建设通用库。',
        '- 133项离线测试通过；6条定点接口冒烟完成，不把它当准确率验证。','',
        '## 费用','',f'本轮146次付费请求，保守记账{cost:.6f}元；累计{budget["accounted_cny"]:.6f}元，剩余{budget["remaining_cny"]:.6f}元。供应商账单为准。','',
        '## 项目架构核对','',
        '目标不变：建立有据、可复核的单Jury，不靠修改医生标签刷分。',
        '结构仍是输入→确定性筛查/准则获取→冻结→单次LLM→引用/检查表校验→研究计分/风险门→审计。没有新增第二Agent、自主联网工具或患者数据传输。',
        '未来仍由平台Node后端调用独立服务，Jury供Admin，医生评分与自动分数独立，锁定人工结果不覆盖。原平台git状态与原有7项删除记录一致，本轮没有改动它们；未上传GitHub。','',
        '## 下一步，尚未执行','',
        '1. 先复核复杂准则中的任务意图和必要缺失边界，收集新的开发样本，避免反复针对原70条调Prompt。',
        '2. 按已制定的适用性方案审核候选准则，先做零付费契约/过滤测试，再做真实本地检索测试。',
        '3. 方法冻结后才启用独立验证；暂不接平台或扩展多Agent。','',
        '详见[同题对照](第四轮结果与架构核对.md)、[方案](../../docs/RUBRICS-APPLICABILITY-AND-RETRIEVAL-TEST-PLAN.md)、[当前默认行为](../../docs/ITERATION-4-RELIABILITY.md)。','']
    (OUT/'本轮完成情况与下一步.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
