# 第4轮可靠性修正：当前行为与边界

## 默认配置

`JURY_JUDGE_PROFILE=scoped-review-v1`，`JURY_EVIDENCE_POLICY=anchored-v2`。

默认路由由确定性安全筛查决定，不读病例ID、医生标签或评分历史：无告警时发送与v2完全相同的输入；有窄范围来源告警时使用v4七字段检查表。每个准则仍只调用同一个Flash模型一次，没有第二Agent或额外工具循环。

全局v4对照没有改善平均分，复杂回答反而退步，因此不作为所有任务的默认。保守路由在旧70条上69条输入保持v2，只有安全规则命中的1条使用v4。此设计限制改动传播，不保证v2本身判断正确，也不是跨病例临床验证。v2、v3、v4历史版本均保留。

独立API若正在运行，需要停止后重新启动本项目 `Start-Agent.cmd` 才会加载新默认；显式环境变量仍优先。原MVP2的启动方式、UI、数据库和密钥未改动。

## 引用v2

- 唯一出处继续输出原文、hash及Python/UTF16偏移。
- 同一引文有多个位置时保留全部候选。只有候选原文一致才可作为多位置引用，标记ambiguous和人工复核；不擅自选择第一处。
- 多个候选规范化后的原文不一致、找不到原文、数字/单位变化等继续拒绝。未引入模糊匹配或case-fold纠错。
- trace_schema升级为evidence-trace-v2。模糊位置的外层不含伪造start/end，真实偏移在candidates中；未来UI必须展示候选而非只取数组首项。
- 歧义引用不会使原模型判定自动变正确；release gate暂停可用汇总分。

## 安全筛查与分数

当前只有一条英文窄范围幼儿乘车进食风险规则，依据[HealthLink BC 110b，2024-04-16](https://www.healthlinkbc.ca/healthlinkbc-files/preventing-choking-babies-and-young-children-child-care-providers)。来源规则是研究者编码，不等于全部医学事实经过验证。年龄未知、否定、引用或不同语言仍可能误检/漏检，阴性不代表安全。

- `judgments`：原始LLM逐项判断，原样保留。
- `research_weighted_score`：用于审计的研究分数，不是放行意见。
- `weighted_score`：存在安全提示或引用歧义时为null，避免直接使用。
- `release_status`：NeedsReview或ResearchOnlyNotClinicallyValidated；后者也不是临床通过。
- `safety_screen`：规则、出处、片段、适用性和局限。
- `criterion_checklists`：任务类型、逐条准则判断，属于模型输出而非外部证据。
- `review_required`：保留模型标志及程序风险标志；必须连同状态读取，不能仅看met或分数。

## 逐条检查协议

v4从原准则显式Step行建立C1–C6；没有Step则只建立一个原准则检查，不编造新医学要求。模型输出task_kind及每项状态/basis，本地校验覆盖、重复和AND矛盾。不收集隐藏思维链，只保留简短可审阅结论。

第4轮实验仍显示错误，例如改写任务被额外要求给临床方法，复杂准则“任何缺失”边界导致过度扣分。检查表提升可审计性不等于提升判断正确率；这些未解决问题已在结果报告列明，不能强改标签或宣称修复。

## 验证与历史

- 140份旧输出离线回放：全部可定位；仅修复位置表达，不改变判断或理由。
- 70条旧病例安全回放：原漏检第27条被标记复核；不报告临床敏感度/特异度。
- 固定70条A/B：两组各70次首次请求，无重试；全局v4没有获准推广。
- 定点6条真实接口冒烟：只检验保守路由及审计报告，不是新准确率基准。
- 已有job、账本、医生标签和旧报告均不覆盖。完整实验源代码位于各自frozen-code。

## 未来平台与检索方案

未来仍是平台Node后端调用独立API，Jury结果供Admin，医生评分独立。API使用者须适配score=null、候选证据位置、release_status和版本，不可将模型met直接解释为“临床安全”。本轮不接入平台。

下一步方案见 `docs/RUBRICS-APPLICABILITY-AND-RETRIEVAL-TEST-PLAN.md`；这是设计稿，现有向量索引没有被批量更改，病例绑定准则没有自动提升为通用标准。
