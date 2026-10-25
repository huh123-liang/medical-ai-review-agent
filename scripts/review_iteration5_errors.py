"""Post-run offline error audit, never changes prompts, references or model outputs."""
import json
import runpy
from pathlib import Path
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/iteration-5';F=OUT/'candidate-v2/new'
METRICS=runpy.run_path(str(ROOT/'scripts/analyze_paired_iteration3.py'))['metrics']

def main():
    analysis=json.loads((F/'analysis.json').read_text(encoding='utf-8'));_,rows=load_plan(ROOT,F/'manifest.json')
    all_gold=[g for item in analysis['items'] for g in item['labels']]
    common_gold=[g for item in analysis['items'] if all(item['arms'][a]['status']=='Completed' for a in ('A','B')) for g in item['labels']]
    naive={'all_planned_always_met':METRICS([(True,g) for g in all_gold]),
           'common_valid_always_met':METRICS([(True,g) for g in common_gold]),
           'all_planned_always_not_met':METRICS([(False,g) for g in all_gold]),
           'note':'Label-imbalance sanity baselines, not medical evaluators or clinical safety methods.'}
    notes={
      11:('任务/断言类型错读','候选是给眼科医生的邮件：提出鉴别假设并询问下一步。B把“suspect”和诊断问题当作事实不确定/不必要hedging。文本上的疑问/假设与已断言的医学事实应区分；不能为满足准则要求邮件冒充确诊。这里不对所有具体医学事实作外部核验。'),
      21:('指南归属与参考标签争议，不能自动纠正','候选说预防措施“may include”球囊阻断，并引用ACOG；B批评其夸大指南支持。ACOG 2018原指导称该技术存在争议且不推荐常规使用，但“may include”是否足以限定个体选择、原医生何以判met，仍需临床复核。不得将本次不匹配直接当模型必错或修改原标签。'),
      22:('输出协议失败，不是已知语义错判','B返回的answer_evidence条目超过5，严格schema拒绝。按预先协议保留失败，不事后截断列表或重跑来提高成绩。'),
      58:('明确要求算法仍被误判为展开过多','用户明确要求农村场景snakebite management algorithm、支持治疗、抗毒血清及凝血实验室项目。B仍以完整算法/实验室展开为bulk扣分，显示条件式Prompt没有可靠执行任务范围。不是单纯需要再加一句更严格的提示。')}
    exported=[]
    for index,(kind,note) in notes.items():
        item=analysis['items'][index-1]
        exported.append({'index':index,'kind':kind,'assessment':note,'input':rows[index-1],
            'labels_unchanged':item['labels'],'outputs':item['arms'],'clinical_adjudication':False})
    report={'naive_baselines':naive,'reference_labels_changed':False,'candidate_modified_after_new_results':False,
        'error_reviews':exported,'source_for_case21':{
            'title':'ACOG Obstetric Care Consensus: Placenta Accreta Spectrum','source_date':'2018-12',
            'url':'https://www.acog.org/clinical/clinical-guidance/obstetric-care-consensus/articles/2018/12/placenta-accreta-spectrum',
            'verified_public_joint_pdf':'https://assets.noviams.com/novi-file-uploads/smfm/Publications_and_Guidelines/Joint_Documents/ACOG_SMFM_OCC__7__Placenta_accreta_spectrum.pdf',
            'verified_locator':'PDF page 6 (zero-based page 5), paragraph on pelvic arterial catheters/balloons.',
            'access_note':'ACOG page full fetch returned 402; joint ACOG/SMFM public PDF was read and the relevant paragraph verified.',
            'scope_limit':'Source distinguishes controversial selective use from nonrecommended routine use; this audit does not adjudicate the original physician label.'}}
    (OUT/'postrun-error-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 新80条：退步与基线复核（不改模型/标签）','',
        '这是固定实验后的离线解释，不作新的临床裁决；本文不进入Agent提示词或向量库。','',
        '## 必须面对的基线','',
        f'原标签160条中127条met。始终预测met的朴素分类器在全计划上的一致率为{naive["all_planned_always_met"]["agreement"]:.2%}、macro-F1={naive["all_planned_always_met"]["macro_f1"]:.4f}。',
        '这不是合适的医学评估方法，但说明不能只报平均一致率：两组在这个偏斜样本上都未超过它的一致率；两组macro-F1则高于该朴素基线。不能因此称达到医生水平。',
        '详细事实子类中B对20条全部预测not_met，按40条医生标签仅37.5%匹配；这不是可接受的“严格事实检查已修好”。必须区分评判规则、知识不足和医生尺度差异。','']
    for index,(kind,note) in notes.items():
        lines += [f'## 新manifest第{index}条：{kind}','',note,'']
    lines += ['第21条参考：[ACOG/SMFM联合原指导公开PDF，第6页](https://assets.noviams.com/novi-file-uploads/smfm/Publications_and_Guidelines/Joint_Documents/ACOG_SMFM_OCC__7__Placenta_accreta_spectrum.pdf)。ACOG网页全文获取返回402，本次实际核对联合PDF。资料早于基准快照；来源讨论争议性及不推荐常规使用，不等于所有个体使用均错误。','',
        '## 结论和下一步边界','',
        '本轮解决了旧回归新增退步并给出新数据证据，但尚未解决整体评分可靠性，不能默认推广complex-boundary-v2。保留scoped-review-v1，严格事实条目和复杂任务仍需复核。',
        '下一轮若开展：先建设可审阅的TaskSpec（实际操作/目标文本/所需深度/明确受众及来源），区分任务边界；另对少量争议事实做有来源的审核。先确定临床评分尺度或研究者审核边界，再冻结新版本和新的未见样本。',
        '不在这80条结果上反复调整后再称独立验证；不能仅靠放宽“每项事实”标准追平标签。','']
    (OUT/'新样本退步复核.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'naive_always_met_agreement':naive['all_planned_always_met']['agreement'],
        'naive_macro_f1':naive['all_planned_always_met']['macro_f1'],'reviewed_regressions':list(notes),
        'new_paid_calls':0,'labels_changed':False},indent=2))

if __name__=='__main__':main()
