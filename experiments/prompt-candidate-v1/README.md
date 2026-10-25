# 第14轮：非默认候选Prompt与离线最小对照

状态：实验目录、默认禁用、未接运行核心/API/原平台。仅准备消息、审核非医疗合成示例、检查人工提供的候选JSON；没有模型客户端，没有新模型预测。

## 两版设计

- **B1_zero**：统一system说明 + 完整冻结输入 + 空demonstrations。
- **B2_three**：同一system、同一schema、同一冻结输入；只多3份固定顺序的非医疗示例。
- 三例分别演示：文字保留、明确文字不一致、证据不足而未定。来源是本地手工构造文本；检查方式为字符串断言和研究协议检查。不是医生审核或临床金标准，不是从7例提取的标签。
- 两版是以后做实际对照的材料，不是两版模型已生成的结果。无法据本轮回答“哪版医学表现更好”。

报告目录：`E:\AI agnet\reports\iteration-14-candidate-prompts`。包含两版完整可读Prompt、机器消息、schema、3张示例卡与核验依据、手工非医疗检查回执、最小测试结果及反思。

## 本轮发现并处理的兼容性冲突

旧 `item-evidence-experiment-v1` 的model_verdict只支持met/not_met；若强迫证据不足输出二元结果，会违背上一轮的三态规范。

因此候选使用**新的独立研究协议** `prompt-evidence-research-v1`，字段为criterion_decision=met/not_met/unable_to_determine、decision_reason、review_required=true。其余定位、来源、义务字段沿用旧设计形状，仍只支持外部AND清单。使用独立schema并做最小检查，**没有更改旧文件，也没有转换后提交旧影子入口**。

旧协议拒绝新候选是预期兼容性边界，不是本轮测试失败。此候选不能直接用于旧API、旧Provider、正式计分或平台。将来如需接线，须另行评审版本化映射；不能把unable_to_determine偷偷转成not_met或met。

## 新增内容及其边界

| 文件 | 实际作用 | 不能证明 |
| --- | --- | --- |
| system.md | 候选任务、极性、证据和未知边界 | 模型一定遵守、能识别任务或抵抗注入 |
| candidate.py/build_messages | 显式enabled后纯消息组装；schema、输入白名单、固定示例、同组/同输入泄漏检查、字节上限 | 不调用模型、不检索RAG、不估算真实tokens |
| candidate.py/check_candidate | 检查提供的JSON、来源定位、引用角色及自报AND状态；保留候选原决定 | 不判断医学语义、真实遗漏、完整断言或源适用性 |
| fixtures.py | 3个非医疗文本例及独立目标样例，输出均手工构造 | 不是新DeepSeek输出、医学标签或评分改进 |
| test_candidate.py | 两版单因素、边界、协议及最小对照的程序测试 | 不是盲审、临床质量、ICL效果或线上负载验证 |
| coverage-map.json | 与上轮A01–A30逐项对应，明确partial/text-only/deferred | 不把旧计划30例全改成“已实施通过” |
| verify.py | 新报告与输入/代码/账本冻结、阻网测试和交付清单 | 哈希不是医学真实性或强防篡改 |

## 输入、示例和记忆边界

模型输入不包含当前病例医生标签、预期结论或旧模型原判。旧的原判只在原流程保存，不送入新候选，避免裁判被先前答案引导。build_messages拒绝冻结输入之外的额外字段，并通过重建校验来源/义务/原文绑定。

三例固定为开发分组，审核状态为 `deterministically_checked_not_physician_reviewed`。B2只接受本目录确定性生成的白名单卡片、固定顺序，禁止改标签、换成7例、加隐藏标签或提升为临床批准。同组和相同case内容哈希均阻止。生产中仍需更广的近重复/患者血缘检查；当前不声称任意语义重复都能检出。

所有目标只允许显式dev。B1和B2共享目标hash；两版唯一消息差异是demonstrations内容。示例不是自动学习记忆，不写向量库、业务数据库或历史判定。

## 输出与资源限制

三态决定、协议有效性、复核状态和医学真实性分离；无论候选met/not_met/unknown，检查回执始终NeedsReview、clinical_release=false、score=null、benchmark_export_eligible=false。程序只汇总自报项，不改写原候选结论。

输入160KB、消息480KB、候选96KB，沿用40主张/12义务/48配对上限，超限拒绝而非截断。这些是**字节/结构限制，不是token预算**。消息schema和三例可能占用较多tokens，未做真实V4 Pro token/延迟/费用验证；暂不视为可直接上线的Prompt。未来首次调用前必须确认模型token上限与预算预约。

OR/条件逻辑不进入AND构造器；负向极性规则只写入Prompt且保留原命题，不声称已实现自动解析。示例的支持关系仅针对显式文字，医学支持关系仍需独立来源和专业审核。

## 运行与审批门

`verify.py`只写新报告及专用测试临时目录；重新运行需新的版本化output-name，不覆盖旧报告。`verify.py --verify`只复核交付/账本/旧文件；需本机对独立Agent目录有相应访问权。正式服务启动不会导入本目录。

完成后先核对目标和架构。建议下一部分先制定小额真实生成的预注册方案：明确开发组、输出/token上限、字段失败停止条件、调用成本和临床标签边界；是否真实调用由用户另批。没有新预测前，不报告两版一致率、Kappa、医疗正确率、稳健性改善或哪版更优。7例继续未决；不改原平台。
