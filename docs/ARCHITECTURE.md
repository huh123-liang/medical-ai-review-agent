# 架构与边界

当前版本：独立单Agent研究原型，位于 `E:\AI agnet`。不修改或共享旧平台数据库。

## 在线评判主线

`FastAPI/CLI → 输入校验 → 准则获取 → 快照冻结 → DeepSeek逐项判断 → 格式/引用位置校验 → 程序计分 → SQLite审计与报告`

LangGraph组织显式节点，任务状态、输入、选中准则和已完成的部分判断保存在SQLite。本版没有启用LangGraph原生checkpoint续跑；进程重启时旧任务标记Interrupted，重试新建任务，不冒充原任务无缝恢复。

## 两种准则协议

- `oracle`：直接提供已绑定病例的医生准则，单独验证裁判能力。不是检索泛化实验。
- `retrieval`：必检项直载，补充项经本地BGE向量和Chroma过滤检索；不使用待评回答作为准则选择查询。没有适用准则则停止，不跨患者兜底。

SQLite是真实准则记录，向量库是派生索引。准则ID不可覆盖，更新需新版本/新ID。开发集以外的准则禁止进入当前开发索引。

首轮100条索引记录是公开开发集的**病例绑定准则记录**，不是100套经过医生确认的通用标准。跨病例复用需要逐条确认适用条件。`data/rubric-candidates-for-review.jsonl`只包含待审候选，不能直接当成可用医学准则。

## 证据与输出

第8轮新增注册候选grounded-review-v2：task-spec-v3将可识别的粘贴信件/编辑材料和请求操作分开，线索冲突不再强制输出失败；review-policy-v2独立记录model_verdict、model_review_required、protocol_status、evidence_status、policy_status和review_reasons。公共Judgment仍五字段，模型原判定/理由不改。证据不足进入NeedsReview，真正无效的字段/ID/来源绑定仍失败。当前所有候选结果都需复核，尚无自动临床放行能力。仅离线原输出/报告回放及单元验证，未推广默认、未新增付费测试或原平台集成。详见OFFLINE-REVIEW-POLICY-V2.md。

第7轮实验增加无损source-ID、TaskSpec v2与4张有限来源卡，同一次V4 Pro调用返回内部七字段，转回原五字段Judgment并另存审计。来源和任务组件没有通过质量门，默认scoped-review-v1不变。grounded配置的可放行分数为空，保留研究判定。该结构不是新Agent或平台集成，原100条Rubrics索引未改。来源卡当前为本地主题匹配，不是通用医学事实向量库。

第7轮提出的协议/任务冲突/证据策略分离，已在第8轮候选实现并离线回放；原子主张完整覆盖、独立临床裁定和自动风险校准仍未实现。旧冻结实现与历史结果保留，不以候选回放覆盖原测试成绩。详见STRUCTURED-EVIDENCE-TASK-FACTS.md与第7轮深度重构建议。

第6轮在独立项目中测试`task-scope-v1`：确定性提取最新用户请求、历史任务线索和字符定位，将其作为不可信上下文提示，同一次LLM判断时参考；不是第二Agent或医学事实证据。五字段输出、引用定位、安全门不变，非复杂准则消息保持原样。两组均V4 Pro进行了固定/新样本对照，但候选未通过质量门，因此默认Prompt仍scoped-review-v1；本机模型配置独立切为Pro。事实来源研究审阅不进入本轮提示词或向量索引。

第4轮增加同一工作流中的确定性安全筛查与放行门；默认scoped-review-v1在无来源告警时使用与v2相同的输入，仅告警触发时启用单次v4逐条检查。anchored-v2支持多位置候选而非伪造唯一定位；风险/歧义时weighted_score为null，research_weighted_score保留。研究分数、LLM判定、医学来源和医生标签仍分开。详见`ITERATION-4-RELIABILITY.md`；复杂回答稳定性尚未得到验证，不能称临床系统。

输出包括met/not_met、简短理由、回答原文片段、上下文原文片段和人工复核标记。严格基准模式不允许abstain；非基准模式可保留abstain，总分停止计算并报告覆盖率。引用定位有效不等于医学结论正确。

本版没有完整医学指南事实库，仅第7轮4张实验来源卡；不提供诊断、处方、临床安全认证或“全面替代医生”的结论。模型无法访问shell、任意文件、医生标签或自主网络工具。模型返回内容不能修改准则、输入和费用配置。

## API与未来平台

API只绑定127.0.0.1，默认4205。除健康检查外需要本机Admin token；无浏览器跨域支持、无公开Swagger界面。旧平台未来从Node后端调用此服务，不让浏览器接触Provider Key或Admin token。自动结果与人工评分独立存储。

- `GET /health`：服务身份与健康。
- `GET /v1/budget`：保守费用账本。
- `POST /v1/evaluations`：提交任务，建议携带`Idempotency-Key`防重复。
- `GET /v1/evaluations/{job_id}`：状态与结果。
- `POST /v1/evaluations/{job_id}/cancel`：请求取消；在途HTTP调用可能需等待超时或返回。
- `POST /v1/evaluations/{job_id}/retry`：失败/取消/中断任务新建尝试。
- `GET /v1/evaluations/{job_id}/audit`：阶段与部分判断审计。

单个runner文件锁防止多个执行服务互相打断状态。API目前只有Admin角色，无Doctor访问。没有修改旧平台UI、权限或最终锁定规则。

## 多Agent预留

目前Retrieval/Validation是确定性模块，不是独立自由行动Agent。以后可在一致性实验显示收益后添加Verifier，复用任务ID、快照、固定schema和权限边界。DSPy/TextGrad、本地Prometheus、正式平台接入均未实施。
