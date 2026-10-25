"""Final evidence and architecture gate. No provider calls and no platform writes."""
import hashlib
import json
import subprocess
from pathlib import Path
from medical_jury.budget import Budget
from medical_jury.config import Settings
from medical_jury.store import Store

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-5';V2=OUT/'candidate-v2'

def main():
    v1=json.loads((OUT/'calibration/analysis.json').read_text(encoding='utf-8'))
    cal=json.loads((V2/'calibration/analysis.json').read_text(encoding='utf-8'))
    new=json.loads((V2/'new/analysis.json').read_text(encoding='utf-8'))
    manifest=json.loads((V2/'new/manifest.json').read_text(encoding='utf-8'))
    cpre=json.loads((V2/'calibration/preflight.json').read_text(encoding='utf-8'))
    npre=json.loads((V2/'new/preflight.json').read_text(encoding='utf-8'))
    assert cpre['code_sha256']==npre['code_sha256'],'Candidate changed between old regression and new cases.'
    for name,sha in npre['code_sha256'].items():
        assert hashlib.sha256((ROOT/'src/medical_jury'/name).read_bytes()).hexdigest()==sha
        assert hashlib.sha256((V2/'new/frozen-code'/name).read_bytes()).hexdigest()==sha
    assert len(new['items'])==80 and len({i['group_id'] for i in new['items']})==80
    assert not {i['group_id'] for i in new['items']}&set(manifest['excluded_groups'])
    assert hashlib.sha256((ROOT/'data/private/physician-labels.jsonl').read_bytes()).hexdigest()==manifest['labels_file_sha256']
    prior=json.loads((ROOT/'reports/iteration-4/completion-summary.json').read_text(encoding='utf-8'))
    git='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
    platform='C:/Users/Lenovo/Desktop/NTU/NTU AI security agents/MVP2'
    status=subprocess.run([git,'-C',platform,'status','--porcelain'],check=True,capture_output=True,text=True,encoding='utf-8').stdout.splitlines()
    assert status==prior['original_platform_preexisting_status']
    settings=Settings.load();store=Store(ROOT);budget=Budget(settings,store).total()
    assert settings.judge_profile=='scoped-review-v1' and settings.evidence_policy=='anchored-v2'
    assert budget['accounted_cny']<=30
    with store.connect() as db:rubrics=db.execute('SELECT COUNT(*) FROM rubrics').fetchone()[0]
    assert rubrics==100
    cost=round(sum(a['phase_accounted_cny'] for a in (v1,cal,new)),6)
    assert abs(budget['accounted_cny']-prior['budget']['accounted_cny']-cost)<1e-6
    score=new['scores'];ci=new['uncertainty']['common_agreement_delta_95ci']
    result={'status':'REVIEW_AND_NEW_SAMPLE_TEST_COMPLETE_QUALITY_GATE_NOT_PASSED',
        'offline_tests_passed':141,'new_examples':80,'new_distinct_groups':80,'excluded_prior_groups':118,
        'sampling_label_blind':True,'candidate_changed_after_new_results':False,
        'calibration_v1_A_agreement':v1['scores']['common_valid_metrics']['A']['agreement'],
        'calibration_v1_B_agreement':v1['scores']['common_valid_metrics']['B']['agreement'],
        'calibration_v2_A_agreement':cal['scores']['common_valid_metrics']['A']['agreement'],
        'calibration_v2_B_agreement':cal['scores']['common_valid_metrics']['B']['agreement'],
        'new_sample_scores':score,'new_common_agreement_delta_95ci':ci,
        'complex_response_regression_fully_resolved':False,'candidate_default_promoted':False,
        'default_profile':settings.judge_profile,'original_platform_unchanged':True,'existing_index_records':rubrics,
        'github_changed':False,'paid_attempts_this_iteration':200,'iteration_accounted_cny':cost,'budget':budget,
        'quality_findings':['Old10 additional regression restored to baseline, not proof of generalization.',
            'New80 shows modest uncertain improvement, with per-category regressions and output failures.',
            'Detailed strict-factuality candidate predicted not_met for all20; not acceptable as solved.',
            'Always-met sanity baseline exceeds both models on raw agreement because labels are imbalanced; models have better macro-F1.',
            'Explicit algorithm requests and diagnostic questions can still be misclassified.'],
        'next':'Review TaskSpec and a small expert/source-adjudicated calibration set before another candidate; keep new80 as used dev data.'}
    (OUT/'completion-summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    gate={'status':'ENGINEERING_AND_EXPERIMENT_PASS_QUALITY_HOLD_NO_DEFAULT_PROMOTION',
        'objective_alignment':'Expose and reduce boundary errors with frozen, label-isolated comparisons, not fabricate success.',
        'architecture_alignment':'One Agent, original supplied rubric/five fields, deterministic safety/evidence gates, no extra tools or model.',
        'platform_alignment':'MVP2/GitHub/index unchanged; future automatic scores remain separate from doctor labels.',
        'candidate_unchanged_between_calibration_and_new':True,'doctor_labels_unchanged':True,
        'quality_gate_passed':False,'complex_regression_fully_resolved':False,'default_promoted':False,
        'reason':'Confidence interval includes no improvement, strict factuality regresses and task-intent mistakes remain.',
        'budget':budget,'next':'Stop this paid batch; review explicit task scope and clinical reference disputes before a new protocol.'}
    (V2/'gate-4.json').write_text(json.dumps(gate,indent=2),encoding='utf-8')
    lines=['# 第5轮结果：旧回归恢复，但新样本尚未通过质量门','','## 结论','',
        '**边界复核和新80条对照已完成；评分可靠性仍未解决，因此complex-boundary-v2不设为默认。**',
        '默认仍为scoped-review-v1：无来源安全告警时保留v2输入；原平台、原索引和GitHub不变。','',
        '## 按顺序完成与核对','',
        '1. 复核三类准则：严格事实标准不放宽、改写不额外求临床解法、详略三步骤分开。纠正了“头痛原文开头没有紧急提示”的错误概括。gate-1通过研究边界核对，临床尺度争议保留。',
        '2. v1旧10条70%→60%，保留失败候选并暂停；只作一次v2修订后，旧回归恢复为70%→70%，两组均10/10完成。不是泛化提升。',
        '3. 冻结候选后按类别各20条抽取新80个开发对话组，排除118个历史/索引组，不按标签挑样本；新结果出来后未改候选、未重跑。',
        '4. 统一核对覆盖、原始医生标签、逐类退步、朴素基线及区间；质量门没有通过，未继续推广。','',
        '| 新80条指标 | 原v2 | 边界候选v2 |','| --- | --- | --- |',
        f'| 完整通过 | {score["A"]["completed"]}/80 | {score["B"]["completed"]}/80 |',
        f'| 共同77条逐医生一致率 | {score["common_valid_metrics"]["A"]["agreement"]:.2%} | {score["common_valid_metrics"]["B"]["agreement"]:.2%} |',
        f'| 共同77条Cohen Kappa | {score["common_valid_metrics"]["A"]["cohen_kappa"]:.4f} | {score["common_valid_metrics"]["B"]["cohen_kappa"]:.4f} |',
        f'| 共同77条macro-F1 | {score["common_valid_metrics"]["A"]["macro_f1"]:.4f} | {score["common_valid_metrics"]["B"]["macro_f1"]:.4f} |',
        f'| 含全部计划失败的可交付标签匹配率 | {score["A"]["delivered_physician_match_rate"]:.2%} | {score["B"]["delivered_physician_match_rate"]:.2%} |','',
        f'共同有效77条、154次医生比较；差异95%区间约[{ci[0]*100:.2f}, {ci[1]*100:.2f}]个百分点，跨0。两组均有失败，整批二分类主要分数按预定协议保留为空。不能把57.79/60.39当成全80条主要成绩。',
        '可交付标签匹配率以全部160条医生标签为分母，失败只表示未交付匹配，未伪造成not_met。',
        '新样本标签127/160为met，始终预测met的朴素基线一致率79.38%、macro-F1约0.4425；两组macro-F1较高，但原始一致率仍低于该朴素基线，不能说已达到医生水平。',
        '详略适当性观察到改善；详细严格事实子类42.5%→37.5%，候选对20条全部判not_met；严格事实类仍是明显短板。','',
        '## 未解决原因及更具体的下一步','',
        '- 条件式Prompt仍可能把明确要求算法的回答当成展开过多，把诊断疑问/假设当成事实hedging；需要可审阅的TaskSpec及来源，而不是继续无限叠加提示词。',
        '- 严格事实类把知识不确定、事实错误、参考标签尺度混在一起，必须做少量有来源/专家审核的事实与判分边界校准。保持原标签，同时记录争议，不强行刷分。',
        '- 1条候选因证据超过5项失败、2条原v2引文无法精确定位；没有事后放松校验或重试来提高分数。',
        '- 这80条现在是已用开发数据，不再称未见验证；下一候选需新的未见样本。未动validation/test、隐藏测试、RAG索引或原平台。','',
        '## 费用与交付','',f'本轮200次首次请求，总保守记账{cost:.6f}元；累计{budget["accounted_cny"]:.6f}元，剩余{budget["remaining_cny"]:.6f}元；供应商账单为准。141项离线测试通过。',
        '详细新样本对照：[结果解读](candidate-v2/new/结果解读.md)；逐条退步：[复核](新样本退步复核.md)；原始输出、hash和费用均可追溯。','']
    (OUT/'第五轮结论与后续.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in {'new_sample_scores','quality_findings'}},indent=2))

if __name__=='__main__':main()
