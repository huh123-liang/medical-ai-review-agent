# v2输出约束、失败诊断与分层实验

## 已实施的输出约束

`medical-jury-criterion-v2-strict` 使用明确JSON模板和本地严格验证。接口仍为DeepSeek支持的 `response_format: json_object`，**没有冒充供应商原生强制json_schema**。

五个字段全部必填：`judgment`、`reason`、`answer_evidence`、`context_evidence`、`review_required`。拒绝额外字段、重复键、自动类型转换、字符串冒充数组、字符串冒充boolean、Markdown围栏及非JSON内容。基准判断只允许met/not_met；引用须为原文精确子串。

原有结果不会重新解释为v2。首轮v1快照及失败记录均保留。失败不自动生成第二答案，不使用医生标签修复模型输出。

## 失败日志

每个失败请求在SQLite保留审计事件，详细诊断位于 `data/private/provider-diagnostics/<job_id>/<reservation_id>.json`。

错误包括：`OUTPUT_JSON_SYNTAX`、`OUTPUT_JSON_DUPLICATE_KEYS`、`OUTPUT_SCHEMA_INVALID`、`OUTPUT_EVIDENCE_OR_PROTOCOL_INVALID`、`EMPTY_MODEL_OUTPUT`、`INCOMPLETE_MODEL_OUTPUT`、`PROVIDER_TRANSPORT_ERROR`、`PROVIDER_ENVELOPE_INVALID` 和 `PROVIDER_HTTP_*`。

- 语法错误：行、列及输出hash。
- 字段错误：安全字段名、错误类型；不保存Pydantic输入值或原始错误消息。
- 引用错误：字段、序号、引用长度及hash；不保存私人病例的引用文本。
- 仅公开HealthBench基准：允许保存已脱敏的最多1200字符输出片段；v2诊断还分别保存每条最多300字符、每字段最多5条的公开证据片段，避免长reason挤掉引用线索。
- 不保存请求头、API Key、Admin token或原始HTTP错误正文；诊断不上传GitHub。

日志脱敏不是完整DLP或临床合规认证。未来接入私有数据时应保持无原文默认策略，另外设计访问控制、保留周期与删除机制。

## 本轮实验

固定70条、70个不同对话，7个官方任务大类各10条，覆盖34个子类；排除首轮5个对话。每类预先抽4条医生一致满足、4条一致不满足、2条分歧项。病例上下文关键词仅辅助覆盖多样性，不是官方疾病标签。

清单及输入hash在付费请求前冻结：`reports/stratified-v2-dev-manifest.json`。医生标签参与抽样和离线核对，但不进入Agent请求。仍是oracle给定准则测试，不是RAG检索验证。

本轮上限5元，累计30元；每条首次请求、最多0.50元，预调用保守费用合计2.742448元。完整执行后本轮保守记账0.421992元，累计0.463245元。供应商账单为最终依据。

结果：JSON及字段校验70/70通过，整体67/70通过，3条引用不匹配。最后增加的引用字段/序号诊断只影响未来日志，不修改本轮旧日志或判断。本轮没有重试这3条，也没有将它们排除出覆盖率。

完整批次质量主指标仍为空。67条成功子集的诊断性agreement约76.6%、Cohen Kappa约0.532、macro-F1约0.766，**不是整批正式成绩**。存在样本选择及缺失偏倚；不能与第一轮10条直接做因果提升比较。

详细结果：`reports/stratified-20261024-194713/第二轮测试解读.md`。
