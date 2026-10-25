# 引用修复、默认配置与第3轮实验

## 已启用与没有启用

独立Agent默认使用 `judge_profile=v2-strict`、`evidence_policy=anchored-v1`。没有把本轮收益尚不稳定的v3设为默认，没有改动MVP2。若独立Agent服务正在运行，停止后重新运行本项目 `Start-Agent.cmd` 才会加载新默认。

显式环境变量 `JURY_EVIDENCE_POLICY` / `JURY_JUDGE_PROFILE` 优先于代码默认。`strict-v2` 保留历史严格引用行为；`v3-rubric-first` 仅作为可切换实验版本，不代表临床批准。

## 引用定位如何修复

1. 优先在原回答或单条上下文消息中精确查找。
2. 找不到时，仅允许NFC、弯直引号、空白及受约束的Markdown加粗排版等价；代码片段不改。非精确引文至少12个字符。
3. 必须得到唯一原文位置；拒绝模糊匹配、改变数字/单位/符号/否定、跨消息拼接、重复位置擅选第一个。
4. 输出恢复为原文片段，记录模型引文、原文引文、source_index、SHA256、Python字符偏移、浏览器UTF16偏移和规则版本。经排版定位者设置人工复核标志。
5. 不修改模型判定、理由、输入、准则或医生标签。成功定位只证明出处位置，不证明医学事实。

9条分歧的完整分析保存在 `reports/iteration-3/cases/`，不导入向量库、不传入Agent提示词。

## 实验与审计入口

- 历史引用回放：`reports/iteration-3/step1-evidence-replay.json`；69条可回放，1条旧原始输出截断，不能补造。
- 9条分析：`reports/iteration-3/9条分歧逐条审阅.md`。
- 同70条A/B：`reports/iteration-3/第三轮对照结果.md`。
- 费用、请求顺序与hash：`reports/iteration-3/paired/preflight.json`；每条job另存快照。
- 实验使用代码：`reports/iteration-3/paired/frozen-code/`；对应原始hash，不与实验后默认配置变更混淆。
- 相同引文的免费解析器对照：`reports/iteration-3/citation-policy-ablation.json`。
- 逐步目标/架构判断：`reports/iteration-3/gate-1.json` 至 `gate-4.json`。

运行 `python scripts/run_paired_iteration3.py estimate` 只做费用估算；`run` 是付费命令，既有实验目录存在时拒绝覆盖和重复运行。分析脚本无付费调用，不要删账本重置预算。

## 结果判断与平台方向

A完整70/70；B完整69/70。共同69条上，一致率75.89%→78.01%，Kappa 0.5177→0.5604；一致率变化区间约−4.26至+8.57个百分点，不能认为有稳定提升。B增加重复标题歧义失败，并有退步；因此不推广v3。

仍处于“单Agent公开开发基准和可靠性修正”阶段。后续先处理事实依据/参考标签争议、引用歧义和未识别安全细节，再冻结方法做未见数据验证；独立验证、真实RAG端到端验证、多Agent、本地模型与平台接入尚未完成。

将来保持 `平台Node后端 → 独立Jury API → Admin审查`；Jury报告与Doctor评分独立，不暴露同伴评分、不覆盖锁定人工结果。不应把医院患者数据直接用于当前公开基准测试流程。
