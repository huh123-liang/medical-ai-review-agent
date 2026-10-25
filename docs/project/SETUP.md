# 安装与使用

## 1. 环境

Python 3.11–3.13，推荐3.12；Git用于克隆。Windows使用独立`.venv`，不复用旧平台环境。安装依赖需要网络；模型API调用与安装是两回事。

```bash
git clone https://github.com/huh123-liang/medical-ai-review-agent.git
cd medical-ai-review-agent
python -m venv .venv
```

Windows激活：`.venv\Scripts\activate`；macOS/Linux：`source .venv/bin/activate`。

```bash
python -m pip install -e ".[test]"
python -m medical_jury.cli init
python tools/run_offline_checks.py
python -m medical_jury.cli evaluate --input examples/synthetic-request.json --provider mock
```

作者环境的`constraints-tested.txt`是依赖版本快照；可以用`pip install -c constraints-tested.txt -e ".[test]"`尝试相同依赖。不同系统/未来软件源不保证完整复现该Windows环境；发生安装冲突时先保存错误，不静默绕过安全校验。

Windows也可双击`First-Time-Setup.cmd`；它建立环境、安装依赖并初始化，不调用付费模型。`Run-Offline-Test.cmd`验证本地流程；统一检查入口`tools/run_offline_checks.py`还分进程验证实验模块。

## 2. 启动本地API

```bash
python -m medical_jury.cli serve --port 4205
```

或双击`Start-Agent.cmd`。只绑定127.0.0.1；`GET /health`可检查服务。评估/费用接口要求本机Admin token，存于`.runtime/admin-token`，不要发送给他人。默认关闭Swagger/OpenAPI网页；这不是Doctor Portal登录页。

主要接口：`POST /v1/evaluations`、`GET /v1/evaluations/{job_id}`、`GET /v1/budget`。字段以`domain.py`和`api.py`为准；浏览器跨域请求默认禁止，未来平台应通过受控后端调用。

## 3. 下载公开基准（不调用付费模型）

```bash
python -m medical_jury.cli download
python -m medical_jury.cli prepare
python -m medical_jury.cli benchmark --provider mock --split dev --limit 10
```

下载较大的官方JSONL，拆分输入和医生标签。原数据、许可和上游版本由使用者自行核对，不随仓库分发。Mock分数不能当医学成绩。

## 4. 可选：本地准则向量库

```bash
python -m medical_jury.cli index --limit 100
python -m medical_jury.cli query --top-k 4
```

第一次可能下载Embedding模型。只把dev病例绑定准则作为链路演示，不是通用医学知识或经医生批准的跨病例RAG。

## 5. 可选：真实DeepSeek调用（收费）

运行`python -m medical_jury.cli configure`或`Configure-DeepSeek.cmd`，凭据隐藏输入，只保存在自己的`.env.local`；不要提交或发到聊天。配置器默认Flash并可能覆盖原配置，切换Pro前核对模型与保守费率。

作者最近实验使用：`deepseek-v4-pro`，保守输入13.2/输出39.6元每百万token，默认单次尝试。它们是本项目规划上限，不是永久官方报价。请核对[官方人民币价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)和本地有效期；不得为解除阻断而随便填未来日期或给Pro套Flash费率。

确认协议、模型、价格和费用上限后才显式运行：

```bash
python -m medical_jury.cli benchmark --provider deepseek --split dev --limit 10 --max-batch-cost 0.50
python -m medical_jury.cli budget
```

0.50元不足时可能只运行部分并停止；不能承诺10条都能完成。不要删SQLite重置预算或自动重试失败。隐藏test需要显式`--allow-heldout`，但只有冻结方案后才可使用。

## 6. 历史实验与新电脑的区别

本仓库包含研究脚本，但不包含作者`data/`、`reports/`、费用账本及逐例材料。部分历史verifier有作者路径和冻结摘要门，故只作为可阅读的历史实现，不是新电脑即点即跑的入口。

可直接从干净仓库验证的是：核心离线测试、合成Mock输入、当前API/CLI、本地公开数据适配，以及纯实验模块测试。若要重新执行第15轮研究，请先重新设计独立样本/封存协议；不要移除旧保护门后假称复现了作者原封存实验。

公开版不带4份医学来源快照，对应一项来源完整性集成测试会明确跳过；其余测试仍使用合成夹具验证定位、校验、存储和预算。不能把这项跳过计为医学事实核查通过。
