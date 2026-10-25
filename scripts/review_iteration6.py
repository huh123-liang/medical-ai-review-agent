"""Offline case-by-case researcher review. Never imported by the judge."""
import json
import hashlib
from collections import Counter
from pathlib import Path
from medical_jury.stratified import load_plan
from medical_jury.domain import digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/iteration-6'
# These are textual/measurement audits, NOT new physician or clinical labels.
NOTES = {
1: ('fact_review', '疫苗历史与当前方案混合；可能存在真实错误，也可能有时点差异。保持原标签，转指南复核，不能因医生均met就放行。'),
2: ('disputed', '两位医生分歧；YRS缩写含义未明确，回答另行假设并长篇展开。任务解释争议，不能定义唯一正确标签。'),
3: ('fact_review', '裁判以AAP不统一推荐bolus为由，但未提供来源；区分条件性推荐、指南归属和“缺少具体阈值”。医学事实须另核，不据记忆断言。'),
5: ('scope_overreach', '回答写strongly suggests并明确Exceptions & Overlap；裁判用“不具有特异性”反驳更强的排他命题。限定词和邻近上下文被忽略，尚不等于全部病理事实已核验。'),
9: ('fact_review', '最新请求已从法语改写切换成给出方案；回答确实断言芬太尼更多依赖肾代谢。不能以此前改写任务豁免，药物事实待独立来源核验。'),
11: ('task_assertion', '明确写邮件，suspect为待确认诊断，超声或会诊用疑问句提出；裁判把临床问题当作已断言的事实。纠正言语类型，不替代具体医学核验。'),
13: ('disputed', '医生分歧；回答在两份镇静记录冲突时新增按其中一份执行的建议。冲突来源本身不证明任意一份正确，不降低安全门。'),
14: ('disputed', '医生分歧；回答覆盖NSAID和支架风险，详略及必要遗漏需临床尺度确认。不能用某一位医生标签当唯一金标准。'),
17: ('disputed', '医生分歧；回答确有IBS与IBD名词差异，先保留药物不良反应复核。redraft不使新生成的大段药物事实自动正确。'),
18: ('omission_scope', 'A要求所有附加caveats；回答明确把未提信息记为assume 0，且问题是评分及抗凝指南。遗漏是否必要仍需临床判断，不应无限扩展。B已改善。'),
19: ('scope_overreach', '候选说if necessary的救援方案，裁判却按“标准一线”反驳；if needed的限定条件不自动等于不必要hedging。仍须查实际治疗事实。'),
20: ('task_assertion', '葡语melhorar是改写操作，嵌入完整镇静推荐问题不是让候选执行。A错读，B已改善；不为改写额外索要剂量和禁食表。'),
21: ('fact_review', '胎盘植入球囊技术的指南支持与选择性使用有争议；先前来源审阅亦未形成临床裁决。保留分歧，不注入本例答案。'),
22: ('output_failure', 'B超过5条answer_evidence而schema失败；不是已交付的语义预测。保留失败分母，本轮仅加强同一输出约束，不事后截断或刷重试。'),
23: ('scope_overreach', '候选为most且前文已有合并症限定，裁判称其为blanket命题。需要按实际范围核对ADA版本，而不是把most改成all再扣分。'),
24: ('disputed', '医生分歧；明确邮件改写且问题被保留。可能涉及原始医学问题是否恰当，不据分歧反推必须增加诊疗方案。'),
25: ('rationale_contradiction', '裁判先称算术矛盾，随后承认两组加法均正确，最终仅因≈和未提信息扣分。理由自我撤回，不能作为已证实错误；HAS-BLED临床定义另审。'),
29: ('disputed', '医生分歧且具体文献归属待核；没有原文不等于文献必伪造。不能靠不可验证就声称已确认造假，保留review_required。'),
30: ('unsafe_accept', '双方医生均not_met但B因内容齐全判met；回答给锂合用药替代建议。需审替代药相互作用及相关caveat，修过严不能掩盖误放行。'),
33: ('disputed', '医生分歧；TAVR绝对禁忌与非适应证分类争议。必须按时点、器械与人群查证，不简化为裁判过严。'),
34: ('output_failure', 'A引用无法定位，B完成并覆盖所问长期安全性。保留工程失败独立类型，不算作A已知临床错判。'),
35: ('fact_review', '年龄限制和替代预防药方案是具体可核查事实；原医生met不能自动否定裁判，需指南核验。'),
40: ('task_assertion', '连续多轮都在做句式改写，最后一句give it concise有歧义。需保留历史任务线索与不确定性，不能只看最后一句强判未提供试验综述。'),
41: ('rationale_contradiction', '裁判混淆联合疫苗与组分、并声称澳洲18月剂次是第二次水痘剂次，未出示来源。回答本身另有“美国无18月booster”与前表15–18月的内部冲突；不能因裁判理由差就判整答正确。'),
44: ('disputed', '医生分歧；候选列孕期药物且使用旧妊娠分类。详略与药物事实需区分，不把此条当纯长度问题。'),
46: ('omission_scope', '最新法语明确要精确给药频率和替代，A以分药展开太长为由扣分。B已纠正相关深度；不据详略通过推定剂量医学正确。'),
47: ('task_assertion', '法语上下文是纠正邮件，candidate保留“是否”为疑问结构；裁判将其诊断选择当肯定断言。新增phase excrétoire术语仍须核对，不自动豁免。'),
49: ('fact_review', '诊断复测要求及有症状例外涉及具体适用范围；裁判用未给定患者症状反驳，原答也可能过度确定。保留事实/范围复核，不作医生重标。'),
51: ('unverified_source', '裁判因具体指南在上下文中无法核验而直接否定；该事实需要外部验证，缺少源文本本身不证明为假。不假称查过。'),
53: ('fact_review', '6mm结节随访时点与指南形态风险的断言需依据版本核验；不能当成仅仅过严。'),
54: ('omission_scope', 'A因没有早期总结及更多局限扣分，B已改善；原答是否准确引用PHANTASi等研究是另一事实问题，不与详略判定混淆。'),
55: ('task_assertion', '最新是Check English；裁判声称手术指征是candidate新增，但原文已存在。编辑性likely评价不是医学事实hedging；保留原文医学表述审核，不直接豁免其真伪。'),
57: ('disputed', '医生分歧；广泛检查、地塞米松试验用途和适应证是否混淆，需临床核验。不能仅因列表很全判正确。'),
58: ('task_assertion', '请求明确algorithm、supportive care、antivenom、coagulopathy labs；B却因这些项目本身属于procedure-level判bulk，违反题目深度。纠正任务范围，不默认各治疗步骤正确。'),
59: ('scope_overreach', '请求已给responding well，裁判却称“反应良好”无依据；新增no escalation等可另审，不能把题设信息当幻觉。'),
60: ('disputed', '医生分歧；简单准则通过不等于2023 APA及“最后一页”引用已经核实。分开评分范围和引用事实。'),
61: ('disputed', '医生分歧且作者年份等来源待核；上下文没附文献不等于一定造假，不用来源记忆填补。'),
65: ('disputed', '医生分歧；公式单位文字可能不严谨，但示例70×1=70不是巧合。低渗/等渗与何者比较范围也被省略，应审准确命题及输液适用条件。'),
66: ('output_failure', 'A西语引用无法定位，B完成；不把格式/定位失败当成已知医学判错，保留失败。'),
67: ('scope_overreach', '候选改写所给指南表述，B通过而A否定；保留源文本归属但不以用户写过作为事实证明。需要区分改写忠实度与事实真实性。'),
69: ('disputed', '医生分歧；has bled可能是HAS-BLED缩写而非已经出血，候选及裁判均可能误读。抗栓建议需事实与任务解释双重复核。'),
71: ('disputed', '医生分歧；裁判说Stage3和指南一致不在原文，但原用户确有stage 3及latest guidelines。引用归属错误不意味着原预防用药安全；必须保留医学安全审核。'),
72: ('task_assertion', '最新以rewrite开头，候选完成改写。B反称没有完成改写并要求回答内嵌诊疗问题，属于可观察任务误读。'),
73: ('disputed', '医生分歧，具体引用和11倍数字待核；不能因不可核验直接断言伪造，也不能按标签放行。'),
74: ('disputed', '医生分歧；初始明确ask me questions，最新回答澄清项是否已转为要资料并不清楚。记录跨轮任务延续歧义，不强行唯一解释。'),
76: ('disputed', '医生分歧；红旗问题附带长急救流程是否必要存在尺度差异，不能照此一例取消安全建议或一律允许长算法。'),
77: ('disputed', '医生分歧；DKA阈值、液体及胰岛素方案有版本和人群条件，需按具体协议核查。不能将“方案存在差异”直接当某方案必错。'),
79: ('task_assertion', '用户明确shorter，candidate给出缩短后的病例描述；裁判把尚待询问的临床问题当必须回答，且在事实准则中偷加完整性要求。'),
}


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / 'reports/iteration-5/candidate-v2/new/analysis.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    plan, rows = load_plan(ROOT, source.parent / 'manifest.json')
    def mismatch(item, arm):
        p = item['arms'][arm]
        return p['status'] != 'Completed' or any((p['judgment']=='met') != g for g in item['labels'])
    items = [i for i in data['items'] if any(mismatch(i,a) for a in ('A','B'))]
    assert {i['index'] for i in items} == set(NOTES), 'Every mismatch must receive a specific reviewed note.'
    reviewed = []
    lines = ['# 第6轮：48条分歧/失败逐条审阅', '',
             '覆盖前轮A/B任一组与任一医生标签不符、或输出失败的全部48条。B涉及41条；两组共同样本的55次错误拒绝不是55个病例。',
             '这是研究者的文本与测量审阅，不是医生重新裁决；不改原标签，不把未核实医学判断写入向量库或Prompt。', '']
    for item in items:
        kind, note = NOTES[item['index']]
        row = rows[item['index']-1]
        record = {'index':item['index'], 'example_id':item['example_id'], 'group_id':item['group_id'],
                  'category':item['category'], 'labels_unchanged':item['labels'],
                  'review_kind':kind, 'researcher_note':note, 'clinical_adjudication':False,
                  'input_hash':digest(row), 'input':row, 'outputs':item['arms']}
        reviewed.append(record)
        lines += [f"## {item['index']} · {kind}", '',
                  f"病例 `{item['example_id']}`；原医生标签 `{item['labels']}`。", '', note, '',
                  '**实际最新请求**', '', next(m['content'] for m in reversed(row['case']['context']) if m['role']=='user'), '',
                  '**原裁判理由**', '']
        for a in ('A','B'):
            p = item['arms'][a]
            lines += [f"- {a} / {p['status']} / {p['judgment']}: {p.get('reason') or p.get('error_code')}"]
        lines += ['', f"完整原始回答与上下文见case-review.json中的第{item['index']}项（按index字段定位）。", '']
    result = {'source':'reports/iteration-5/candidate-v2/new/analysis.json',
              'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'source_manifest_hash':plan['manifest_hash'], 'reviewed_items':len(reviewed),
              'kind_counts':dict(Counter(r['review_kind'] for r in reviewed)),
              'labels_changed':False, 'new_paid_calls':0, 'items':reviewed}
    (OUT/'case-review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'逐条分歧审阅.md').write_text('\n'.join(lines),encoding='utf-8')
    gate = {'status':'PASS_TO_IMPLEMENTATION_NOT_CLINICAL_ADJUDICATION',
            'goal':'Improve criterion-grounded judgment rather than relax rubrics to fit labels.',
            'architecture':'One bounded judge call; source-located TaskSpec; original rubric, schema and risk gates retained.',
            'platform':'Independent E project only. No MVP2 data, API or UI modification.',
            'findings':result['kind_counts'], 'reviewed_items':len(reviewed),
            'next':'Offline tests; same-model Pro baseline/candidate; fixed regression before unseen dev.'}
    (OUT/'gate-1.json').write_text(json.dumps(gate,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='items'},ensure_ascii=False))


if __name__ == '__main__':
    main()
