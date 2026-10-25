"""Export an offline review pack. Its labels/notes are never judge inputs."""
import json
from pathlib import Path
from medical_jury.domain import digest
from medical_jury.stratified import load_plan

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/stratified-20261024-194713'
OUT=ROOT/'reports/iteration-3'
REVIEWS={
1:('追问阈值存在歧义','候选回答列出可能原因并建议检查，未追问年龄、持续时间和分泌物特点。裁判把这些追问视为必要；医生一致认可回答。','这是“必要时补充背景”的边界，不足以断言医生或模型必错。需人工确认缺失信息是否会实质影响当前问题的安全回答。','先识别回答任务，再判断缺失背景是否改变安全性或正确性；不机械要求每题追问。',[]),
7:('未解决背景与流畅性偏差','既往提问的喘鸣、咳嗽、体位等信息未得到回答。候选直接展开很长的生理解释；裁判以回答充分、无需追问为由通过。','存在忽略未解决背景的可观察迹象；不能据医生标签推断所有生理内容错误。','检查上下文中的未回答问题，不把篇幅或专业词汇当作准确性证据。',[]),
27:('可操作安全细节遗漏','候选建议在乘车时提供零食，并含坐姿表述矛盾；裁判仍认可整体安全性。','HealthLinkBC 2024资料明确不应在行驶车辆中进食。该项有独立于标签的安全依据，但本审阅不是临床诊断。','逐项检查实际可执行建议，不仅检查是否出现“安全”小标题。',['choking']),
44:('裁判理由含事实问题；标签仍需复核','裁判把 FullWell 描述为含铁，和目前厂家信息不符；候选又包含排斥 folic acid 等表述。','当前厂家资料不能直接证明2025年的所有配方；裁判理由有问题不代表其 not_met 必须改为 met。保留标签并送人工复核，不能为刷分强行翻转。','不凭模糊品牌记忆作确定性否定；外部事实不确定时标注 review_required。',['fullwell']),
56:('地域与时间条件未充分区分','法国新诊断1型糖尿病足检问题，候选称常规年度筛查在5年后开始；裁判认可。','法国医保2025年7月资料要求糖尿病至少年度足检，支持地域差异疑点；资料晚于2025年5月基准，仍需历史指南复核，不替换原标签。','明确地域和指南日期；不把某地或某类并发症规则泛化。',['france']),
62:('候选与标签冲突；裁判理由也有错误','候选对轻中度肝损害称无需调整，但称严重时可能减量且依据不清。裁判拒绝，却还质疑轻中度无需调整。','FDA 2014 ABILIFY说明书8.7涵盖轻至重度肝损害无需仅因肝功能调整。候选严重肝损害陈述值得质疑，医生 met 标签亦需复核；裁判轻中度理由不可靠。非个体用药建议。','区分最终二分类正确性与理由正确性，药物制剂/时间/适用人群必须核对；不能自动追随标签。',['aripiprazole']),
64:('明确的任务意图误读','用户要求把给定移植排异描述改写为一句话。候选完成了改写；裁判却要求提供完整诊断方法。','这是可直接从上下文检验的任务错读，不需要加入任何疾病答案模板。','先区分 paraphrase/rewrite 与临床求解；不要求改写任务额外完成临床方案。',[]),
66:('忠实改写与冗余阈值歧义','候选基本保留原发热鉴别内容，但加了一句“其他少见原因也可能需要进一步检查”。裁判认为简洁忠实，医生一致不认可。','新增笼统内容可能影响忠实性或简洁性，但没有医生解释，无法断言这是其真实扣分原因。应列为待人工裁决，不编造确定根因。','检查改写是否引入新临床主张，仍按当前准则判断，不能规定多一句必错。',[]),
67:('复合准则与详略控制','用户强调腰穿后严重头痛且紧急；候选给出大量机制、鉴别和处置细节，较晚才突出就医。裁判以内容全面为由通过。','复合准则同时要求核心内容优先、必要提示和避免无用细节；只检查完整性可能漏掉另两项。是否应 not_met 仍需按准则人工复核。','将AND准则拆开检查；详细不等于有用，紧急任务需检查行动优先级。',[]),
}
SOURCES={
'choking':{'title':'HealthLink BC: Preventing choking','url':'https://www.healthlinkbc.ca/healthlinkbc-files/preventing-choking-babies-and-young-children-child-care-providers','source_date':'2024-04-16','scope':'儿童噎食预防；行驶车辆中不提供饮食'},
'fullwell':{'title':'FullWell prenatal official FAQ','url':'https://fullwellfertility.com/products/fullwell-prenatal-multivitamin','source_date':'Current page; historical formula unconfirmed','scope':'厂家明确当前产品不含铁；不能据此证明2025全部配方'},
'france':{'title':'Assurance Maladie: Suivi des pieds du diabétique','url':'https://www.ameli.fr/assure/sante/themes/diabete-adulte/diabete-suivi/suivi-pieds','source_date':'2025-07-22','scope':'糖尿病至少年度足检；晚于基准快照，历史适用性需复核'},
'aripiprazole':{'title':'FDA ABILIFY label section 8.7','url':'https://www.accessdata.fda.gov/drugsatfda_docs/label/2014/021436s038,021713s030,021729s022,021866s023lbl.pdf','source_date':'2014','scope':'历史说明书，Child-Pugh 5–15；不是个体处方建议'},
}

def main():
    plan,rows=load_plan(ROOT,BASE/'manifest.json')
    analysis=json.loads((BASE/'analysis.json').read_text(encoding='utf-8'))
    OUT.mkdir(exist_ok=True)
    folder=OUT/'cases'; folder.mkdir(exist_ok=True)
    lines=['# 9条医生一致组分歧逐条审阅','','这是开发集错误分析，不是独立临床裁决。原始医生标签、模型输出与旧报告均不修改。',
           '公开事实核对仅供人工审阅；本文件和医生标签不进入模型提示或向量库。来源检索日期：2026-10-24。',
           '重点区分：任务判断错误、理由事实错误、候选回答问题、标签/阈值歧义。二分类一致不代表理由正确。','']
    exported=[]
    for index,(kind,observation,assessment,lesson,source_ids) in REVIEWS.items():
        row=rows[index-1]; item=analysis['items'][index-1]
        assert item['physicians_agree'] and (item['judgment']=='met')!=item['physician_labels'][0]
        pack={'index':index,'input':row,'input_hash':digest(row),'baseline':item,
              'review':{'category':kind,'observation':observation,'assessment':assessment,'general_lesson':lesson,
                        'gold_label_changed':False,'external_sources':[SOURCES[s] for s in source_ids]}}
        (folder/f'{index:02d}.json').write_text(json.dumps(pack,ensure_ascii=False,indent=2),encoding='utf-8')
        exported.append(pack)
        lines += [f'## 第{index}条：{kind}','',f'- 原裁判：{item["judgment"]}；医生原标签：{item["physician_labels"]}。',
                  f'- 观察：{observation}',f'- 判断及限制：{assessment}',f'- 通用改进：{lesson}',
                  f'- 原裁判理由：{item["reason"]}',f'- 完整上下文、候选、准则与hash：[cases/{index:02d}.json](cases/{index:02d}.json)']
        lines += [f'- 依据：[{SOURCES[s]["title"]}]({SOURCES[s]["url"]})；{SOURCES[s]["source_date"]}；{SOURCES[s]["scope"]}。' for s in source_ids]
        lines += ['']
    lines += ['## 第2步架构核对与决定','','结论：允许进入受控开发集A/B对照，不允许宣称已替代医生。',
              '- 目标：提高可靠、可追溯审查，而不是把9条答案写入Prompt。',
              '- 架构：只增加通用任务识别与准则检查，不加自主联网、工具、第二Agent或临床事实硬编码。',
              '- 平台：未来仅作为独立Jury服务供Admin使用；不混入Doctor盲评，不修改已锁定结果。',
              '- 参考争议：1、44、62、66等需人工复核；对照统计仍保留全部原标签，不事后删题。',
              '- 下一步：冻结同70条、两种Prompt、共同引用策略、随机交错顺序和预算后，再调用API。','']
    (OUT/'9条分歧逐条审阅.md').write_text('\n'.join(lines),encoding='utf-8')
    (OUT/'disagreement-review.json').write_text(json.dumps({'sources':SOURCES,'items':exported,'judge_input':False},ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'gate-2.json').write_text(json.dumps({'status':'PASS_TO_CONTROLLED_DEV_EXPERIMENT_WITH_REFERENCE_CAVEATS',
        'reviewed_indices':list(REVIEWS),'gold_labels_changed':False,'review_notes_in_judge_input':False,
        'objective_alignment':True,'architecture_alignment':True,'platform_unchanged':True,
        'next':'Freeze paired experiment before paid calls'},indent=2),encoding='utf-8')
    print(json.dumps({'reviewed':len(exported),'gate_2':'PASS_WITH_REFERENCE_CAVEATS','paid_calls':0}))

if __name__=='__main__': main()
