# 医疗AI检测Agent

Medical AI Review Agent / AI Jury：一个独立的医疗大模型输出评估研究原型。

它的任务是**审查其他模型已经生成的回答**，依据固定Rubrics记录判断、证据位置和复核状态；不是给患者自动诊疗，也不是已经证明能替代医生的系统。

> 当前版本：2026-10-25，第15轮公开医生标签小规模盲评后快照。最新试验未达到可靠使用条件；失败和未完成部分均公开说明。原医疗评估平台是独立项目，本仓库尚未与其集成。

## 从这里开始

| 想了解什么 | 入口 |
|---|---|
| 为什么做这个项目、边界是什么 | [项目目标](docs/project/OVERVIEW.md) |
| 整个Agent怎样运行 | [架构](docs/project/ARCHITECTURE.md) · [端到端流程](docs/project/WORKFLOW.md) |
| 已经完成什么、没有完成什么 | [当前状态](docs/project/STATUS.md) |
| 每一阶段做了什么、为什么转向 | [阶段记录](docs/project/ITERATIONS.md) |
| 最新真实测试结果 | [公开医生对照结果](docs/project/EVALUATION.md) · [机器可读汇总](benchmarks/summaries/latest.json) |
| 如何安装、启动、测试 | [安装与使用](docs/project/SETUP.md) |
| 哪些论文/数据可参考 | [研究依据](docs/project/RESEARCH.md) |
| 下一步怎么做 | [路线图](docs/project/ROADMAP.md) |
| 数据、密钥和付费调用限制 | [安全与数据说明](SECURITY.md) |

## 已实现的主线

```text
固定病例上下文 + 待评回答
            ↓
输入校验 → 给定准则 / 本地准则检索
            ↓
冻结病例、准则、模型和Prompt版本
            ↓
LangGraph 单Jury逐条评判（DeepSeek API或离线Mock）
            ↓
程序检查格式、引用位置、条目覆盖及预算
            ↓
SQLite保存研究结果、审计与费用 → API / CLI / 报告

独立评分器：预测封存后读取隔离的公开医生标签，计算一致性
```

检索使用本地BGE Embedding和Chroma。**有向量库不等于有完整医学事实库**；当前原型中的病例绑定准则不能自动作为跨病例通用标准。医生标签不作为裁判输入或检索内容。

## 最新结果：不能包装成“已达到医生水平”

本轮使用14个不同的公开验证对话组、每组1条准则，共29次原医生标签比较。与患者数量、独立医生人数不是同一计数单位。

| 项目 | 当前默认基线 | 新三态证据候选B1 |
|---|---|---|
| 计划 | 14条 | 14条 |
| 有效可比较输出 | 13条 | 0条 |
| 失败 / 停止后未运行 | 1条格式失败 | 3条截断，随后11条未运行 |
| 有效子集逐医生一致率 | **13/27 = 48.15%** | 无可报告结果 |
| 有效子集Macro-F1 / Cohen Kappa | **0.481 / 0.045** | 无可报告结果 |
| 整批主指标 | 因覆盖不完整，保留为空 | 保留为空 |

这些是**小规模、给定准则、已完成子集**的研究结果，不是总体临床准确率，也不是完整HealthBench复现。两臂输出协议/token上限不同，不构成严格单因素Prompt消融。数据、分母、失败和费用解释见[评测说明](docs/project/EVALUATION.md)。

## 目录一览

```text
src/medical_jury/        当前运行核心：工作流、API、模型、检索、预算和审计
tests/                  核心离线回归；不依赖真实API Key
examples/               明确标注的合成研究输入与草案准则
experiments/            非默认候选、逐项契约、影子层和盲评实验代码
scripts/                历次研究脚本；部分需要作者未公开的本地报告
tools/                  本公开版本的离线测试、文档和发布安全核验入口
docs/project/           维护中的结构化项目文档（当前状态以此为准）
docs/                   原设计/迭代规范；历史描述保留，不代表最新成绩
benchmarks/summaries/   经人工字段白名单整理的匿名聚合结果，不含逐例数据
docs/automation/        不使用模型凭据的离线CI模板（未启用）
```

`data/`、`reports/`、`.runtime/`、`.venv/`、`.cache/`在本机运行后生成，**不在GitHub中**。官方数据由使用者自行下载，并遵守原始许可；本仓库不再分发患者资料、医生标签或原始模型回答。

## 快速开始：先离线，不收费

建议Python 3.12，支持3.11–3.13。Windows可双击 `First-Time-Setup.cmd`；其他系统或手动操作：

```bash
python -m venv .venv
# 激活环境后：
python -m pip install -e ".[test]"
python -m medical_jury.cli init
python tools/run_offline_checks.py
python -m medical_jury.cli evaluate --input examples/synthetic-request.json --provider mock
```

启动本地API：`python -m medical_jury.cli serve --port 4205`，或Windows双击 `Start-Agent.cmd`。这是API/研究服务，不是原Doctor/Admin网页平台。启动和Mock不发送付费模型请求。

真实模型测试必须显式配置自己的本地凭据、模型价格、费用上限与有效期，详见[安装说明](docs/project/SETUP.md)。**配置器默认Flash；作者最近实验用V4 Pro，二者不能共用费率。** 旧价格有效期不是永久保证，过期会阻止付费调用。

## 当前未完成

- 医学语义正确性、复杂回答评分可靠性和有代表性的独立临床验证。
- 稳定的紧凑三态证据输出、经审核的通用/疾病/病例三级Rubrics与完整事实RAG。
- DSPy/TextGrad、DeepEval、Prometheus本地模型、多Agent、MCP及原平台接入。

本仓库公开源码便于审阅，不意味着已选择开源许可或授予第三方数据再分发权，见[许可与引用说明](LICENSE-NOTICE.md)。
