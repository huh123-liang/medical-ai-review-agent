"""No network: reconcile ledger, immutable experiments and the final research decision."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from medical_jury.config import Settings
from medical_jury.budget import Budget
from medical_jury.store import Store
from medical_jury.stratified import load_plan
from medical_jury.provider import build_messages
from medical_jury.domain import EvaluationRequest, digest

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-6'


def main():
    settings=Settings.load();store=Store(ROOT)
    fixed=json.loads((OUT/'fixed/analysis.json').read_text(encoding='utf-8'))
    new=json.loads((OUT/'new/analysis.json').read_text(encoding='utf-8'))
    total=Budget(settings,store).total();cost=fixed['phase_accounted_cny']+new['phase_accounted_cny']
    assert abs(3.739158+cost-total['accounted_cny'])<1e-6
    assert total['accounted_cny']<=30 and total['paid_attempts']==786
    assert settings.model=='deepseek-v4-pro' and settings.judge_profile=='scoped-review-v1'
    sources=[]
    for phase in ('fixed','new'):
        folder=OUT/phase
        pre=json.loads((folder/'preflight.json').read_text())
        assert all(hashlib.sha256((ROOT/'src/medical_jury'/n).read_bytes()).hexdigest()==sha for n,sha in pre['code_sha256'].items())
        plan,rows=load_plan(ROOT,folder/'manifest.json')
        for item in pre['schedule']:
            row=rows[item['index']-1];req=EvaluationRequest(case=row['case'],rubrics=[row['rubric']])
            assert digest(build_messages(req.case,req.rubrics[0],True,pre['profiles'][item['arm']]))==item['messages_hash']
        sources.append({'phase':phase,'manifest_hash':plan['manifest_hash'],'planned':len(rows),'calls':len(pre['schedule'])})
    assert json.loads((OUT/'fixed/preflight.json').read_text())['code_sha256']==json.loads((OUT/'new/preflight.json').read_text())['code_sha256']
    error_counts={phase:{a:dict(Counter(i['arms'][a]['error_code'] for i in data['items'] if i['arms'][a]['status']!='Completed'))
                         for a in ('A','B')} for phase,data in (('fixed',fixed),('new',new))}
    with store.connect() as db:
        pending=db.execute("SELECT COUNT(*) FROM costs WHERE state='reserved'").fetchone()[0]
        rubrics=db.execute('SELECT COUNT(*) FROM rubrics').fetchone()[0]
    assert pending==0
    gate={'status':'NOT_PASSED_NO_PROMPT_PROMOTION','goal_alignment':'No label editing or relaxed criteria; negative results retained.',
          'architecture_alignment':'Single bounded judge; source-located TaskSpec, schema/evidence validation, deterministic scoring and audit retained.',
          'platform_alignment':'No original MVP2 writes or integration. Binary research results are not six-dimension clinician scores.',
          'evidence':'Fixed regression worsened; unseen complex matching unchanged with incomplete coverage.',
          'next':'Stop paid calls. First design bounded evidence-ID output and auditable fact verification; require a new frozen protocol and fresh dev groups before testing.',
          'model':'deepseek-v4-pro','default_prompt':'scoped-review-v1','candidate_profile':'task-scope-v1',
          'candidate_promoted':False,'budget':total,'iteration_cost_cny':round(cost,6)}
    (OUT/'gate-3.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={**gate,'offline_tests_passed':159,'paid_calls_this_iteration':220,'sources':sources,
             'error_counts':error_counts,'unsettled_reservations':pending,'indexed_rubric_records':rubrics,
             'labels_unchanged':True,'candidate_unchanged_between_phases':True,
             'new_sample_definition':'40 new-to-project public development conversation groups, not hidden or clinical validation.'}
    (OUT/'completion-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 第6轮结论与下一步','',
           '**结论：完成逐条分析、TaskSpec候选实现和V4 Pro验证，但未证明修好复杂回答过度拒绝。候选不推广。**','',
           '## 完成内容','',
           '- 对前轮A/B任一组有分歧或失败的全部48条逐条审阅；保留原回答、原准则、原医生标签和裁判理由。不是48条都确定Agent错。',
           '- 增加可定位TaskSpec：最新请求、历史任务线索及字符位置；区别问题/假设与事实断言，禁止把有限命题夸成全称命题。没有新增事实搜索Agent或标签输入。',
           '- 本机模型切换为deepseek-v4-pro，实际220次provider元数据均回报该模型；关闭thinking、1200输出token，两组一致。非复杂准则输入保持不变。',
           '- 159项离线测试通过；覆盖标签隔离、报价绑定、证据归属、原schema及失败分母。离线通过不代表医学能力通过。','',
           '## 同模型同题结果','',
           '| 阶段 | A完整 | B完整 | 共同有效 | A一致率 | B一致率 | A/B Kappa |',
           '| --- | --- | --- | --- | --- | --- | --- |']
    for name,d in (('固定70条',fixed),('新complex40条',new)):
        s=d['scores'];m=s['common_valid_metrics']
        lines.append(f'| {name} | {s["A"]["completed"]}/{d["planned"]} | {s["B"]["completed"]}/{d["planned"]} | {s["common_valid_examples"]} | {m["A"]["agreement"]:.2%} | {m["B"]["agreement"]:.2%} | {m["A"]["cohen_kappa"]:.4f}/{m["B"]["cohen_kappa"]:.4f} |')
    lines+=['',
        'A=complex-boundary-v2；B=task-scope-v1；均为Pro、给定原准则、首次请求一次。不同阶段是不同试卷，不跨阶段声称提高/下降。有失败时上述agreement/Kappa仅为共同有效子集诊断；整批正式二分类主指标留空。',
        '固定70条：全计划可交付匹配率74.83%→73.43%，共同有效一致率81.75%→80.16%。其中旧complex10条共同有效9条，FN 3→2但FP 3→6；减少拒绝伴随更多错误放行，不推广。',
        '新40条：全计划可交付匹配率43.75%→45.00%；共同有效29条、58次逐医生比较，两组均56.90%、Kappa 0.1560、FN22、FP3。配对一致率变化95%区间[-10.00,+9.38]个百分点。',
        '新40条：详细适当性58.33%→75.00%，简单适当性71.43%→57.14%；严格事实两子类均无提升。不能挑改善子类作为总体成功。','',
        '## 仍未解决的具体问题','',
        '1. 新第12条末尾明确rewrite，B却称候选不是改写；当前TaskSpec的改写启发式偏向句首，也不能强制LLM遵守全文意图。该例表明仅加提示词未修好任务判定，不把该条答案硬编码。',
        '2. 新第22条从多轮改写转为URGENT临床算法，B对详略判定匹配医生；这只是一个有利案例，未核验算法内全部医疗事实。',
        '3. 固定第67条急诊短问下长篇workup被B放行，A正确扣bulk；专业结构不能自动代表请求所需深度。',
        '4. 固定第62条两组都错误归属aripiprazole说明书；已对照DailyMed 8.7节，见《事实来源复核》。模型记忆不是事实来源，不凭原标签自动给整答翻案。',
        '5. 输出列表超过5项或引用并非原文仍很多；新40条A失败7、B失败9。均为协议/引用失败，不是网络中断。没有重试、截断或事后补判。','',
        '## 各步架构核对','',
        '- gate-1：逐条审阅后核对目标与边界；没有临床重标。',
        '- 离线测试后：保留单Agent、五字段及来源校验，不扩大权限。',
        '- gate-2：固定回归质量未过，仅继续一次冻结新开发诊断，不推广。',
        '- gate-3：新样本亦未证明改善，停止付费调用，不把研究候选接入原平台。','',
        '## 建议下一轮（尚未实施）','',
        '1. 先解决Pro结构化引用：从冻结原文生成可选择的证据ID，由程序还原精确片段，严格校验ID/上限；不是让模型改写引文后再强行匹配，也不是截掉失败条目刷覆盖率。',
        '2. TaskSpec需要可校验的任务类型及归属，包括句尾改写命令、跨轮任务切换、明确请求专业深度与紧急短问；不依据病例编号路由答案。先用合成/已见开发回归检验。',
        '3. Rubrics库和医学事实库分开；少量权威指南/药品标签候选先人工审核来源、版本、人群与时间，未知事实进入复核而非自信编造。当前模型仍无自主工具权限。',
        '4. 再冻结新方案，用全新开发组同题对照，保留负例与非复杂回归；验证集/隐藏测试继续封存。DSPy/TextGrad、多Agent及平台接入都排在可靠基线之后。','',
        f'## 预算与边界\n\n本轮220次请求，保守结算{cost:.6f}元；累计{total["accounted_cny"]:.6f}元，剩余{total["remaining_cny"]:.6f}元。无未结算预留；保守记账不是供应商发票。',
        f'原MVP2、GitHub及原100条准则索引未修改；原始数据/标签hash核对通过，当前rubrics记录{rubrics}条。默认Prompt仍scoped-review-v1，模型按用户要求切成Pro。','']
    (OUT/'第六轮结论与下一步.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in {'sources','error_counts'}},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
