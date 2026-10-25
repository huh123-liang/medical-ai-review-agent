# 逐项证据模板 v1.1：填写与版本边界

用途：研究评审表及未来候选Prompt的字段需求，不是运行中的模型输出协议。空白结构见 [blank-review-record.json](blank-review-record.json)。空值表示未填写，不能解释为未满足准则。

## A. 冻结记录头

- review_id / review_schema_version / record_status：唯一编号、版本、blank_template/research_draft/independent_review_frozen。
- case_reference / group_id：保留对话/重复组；不在模型输入中提供医生标签或目标评分。
- input_bindings：完整对话与回答、准则、外部义务清单、来源目录、Prompt候选的哈希。未存在的候选Prompt哈希留空，不能伪造。
- provenance：材料类型、填写者角色、审核状态、既往暴露、是否为历史原始JSON或报告投影。
- scope：仅研究、clinical_release=false、official_score=null；七例始终保留未决。

## B. 任务与准则清单

记录最新真实请求的doc_id、角色、原句、精确坐标、任务类型、受众及未知项。任务线索/关键词不能当专业确认。

逐字保留准则、极性（positive/negative/unclear）、逻辑（AND/OR/conditional/other/unclear）、例外与歧义。每个义务记录原文位置、是否必选、适用性及其依据、审核状态。任意OR/条件句目前不能映射现有AND候选契约。

## C. 逐项断言表

每个主张一行，字段包括：

| 字段 | 如何填写 | 来源 / 谁负责 |
| --- | --- | --- |
| claim_id / answer_location | 唯一ID、回答逐字引用、文档SHA、Unicode坐标 | 原始回答；程序可验证定位 |
| claim_kind / qualifiers | 医学断言、条件计划、患者属性、归属、文字忠实性或非事实；条件/否定/单位/时间 | 研究或模型声明；不能当已确认真值 |
| obligation_ids | 影响哪些准则义务，为什么 | 外部清单+解释；保留歧义 |
| fact_check_needed / rationale | 是否需外部核查，不能用改写任务豁免新增医学主张 | 待专业确认的研究记录 |
| declared_state | supported/contradicted/unresolved/not_checked/not_required | 填写者声明，不等于核查已完成 |
| independent_verification | 谁、何时、依据什么核查；未核查留空 | 独立评审记录，不由模型自授权限 |

## D. 主张—来源配对表

每个配对记录claim_id/source_id、来源原文位置、URL/版本/日期、适用人群/场景/时间、支持关系、适用性、局限与审核者。

四个检查分别保存：`locator_check`、`semantic_support_review`、`applicability_review`、`claim_inventory_review`。只通过定位时其余保持未核实。来源本身可信不等于支持本条主张；一条文献不能自动覆盖整份回答。

## E. 义务与遗漏表

每项义务记录关联主张、pass/fail/unresolved/not_applicable、证据和理由。若遗漏：记录 `missing_item`、`searched_doc_ids`、检查范围和原任务必要性；不填虚构空引用。若N/A：提供原准则许可依据，不自动删项。

## F. 结论和复核

分别保留原模型原判、研究结论、协议状态、复核原因和未决项。只有结构检查时不得填写“医学验证成功”。示例：已定位到出处但语义未审，可为valid+NeedsReview，而不是not_met或临床通过。

说明何项失败足以决定整条、何项仍未知、哪项原准则解释待裁定。置信度如填写只能是未经校准的主观等级，不能称概率。现有7例不填新判分，也不生成任何临床共识指标。

## G. 示例记忆卡（以后另批，不是本轮已填内容）

建议最小字段：example_id、origin_group_id、dev split、task、rubric_version、回答片段、已核查结论、短理由、来源、review_status、reusable_scope、排除条件。只有批准的开发材料可入候选示例库；验证/隐藏测试和七例不得入库。检索不根据当前目标标签选择例子；固定示例ID与顺序，避免事后挑最好结果。

## H. 与旧契约的映射

| 本模板 | 既有item-evidence-experiment-v1 | 当前能做 / 不能做 |
| --- | --- | --- |
| input_bindings前四项 | binding | 可按既有定义绑定；Prompt哈希只在本研究记录 |
| 原模型原判 | model_verdict/model_reason | 仅保留原始值；不从研究建议补填 |
| 任务/主张/义务/来源配对 | task/claims/obligations/checks | 只接受旧协议字段和范围；本模板不是无损可自动转换格式 |
| 三态研究结论、N/A、OR等 | 无等价wire字段 | 留在人审/研究旁表，不改旧API或强制二元 |
| 来源审核/审核人/许可证 | 旧source目录主要是片段绑定 | 另存研究记录，不能声称线上已有完整来源治理 |
| coverage自报 | coverage | 不是独立证实的医学覆盖率 |

填写者应给简短、可核验的理由，不要求或保存不可验证的长篇内部思维链。模板保留必要证据、边界和决定性依据即可。
