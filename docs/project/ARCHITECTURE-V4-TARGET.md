# v4 架构升级：落地能力（2026-11-16）

目标：在不改变已验收结果（`review-v4.1`，P1/P2/P3）的前提下，把 v4 从「能跑通的研究流水线」推进到「可运维的服务内核」。
全部为纯代码改动，离线验证，未发生任何付费调用。架构图见 `docs/figures/v4-langgraph.pdf`。

## 结论

| 组 | 步骤 | 状态 | 对已验收结果的影响 |
| --- | --- | --- | --- |
| A | 1. 取消与更正（`cancel` / `revise`） | 完成 | 无 |
| A | 2. 持久化检查点 + 崩溃恢复（`open_durable` / `recover`） | 完成 | 无 |
| A | 3. 调用截止层（总耗时上限，取消时立即停止等待） | 完成 | 无 |
| A | 4. 声明式规则库（`kb/v4/rules.json`） | 完成 | 无（行为摘要固定） |
| A | 5. 节点追踪 + 回归闸门（模拟 + 真实记录重放） | 完成 | 无 |
| B | 6. 输入防护（`screen_input`） | 开发版，默认关闭，未评测 | 关闭时无 |
| B | 7. 漏写类准则（病期停药、CKD 合并 T2D 的 SGLT2i） | 开发版准则库，未评测 | 只在加载开发库时生效 |
| B | 8. 严重断言独立复核（`verify_severe`） | 开发版，默认关闭，未评测 | 关闭时无 |

「无影响」的证据：
- 同一病例经 LangGraph 与原 `ReviewerV4` 产出的记录逐字段一致（除耗时）；
- P1 的 322 条、P3 的 158 条真实封存记录，经当前代码重放模型之后的全部代码层，**480/480 一致**；
- 规则引擎与原手写实现在 480 个测试集病例和 20 000 个边界随机病例上逐条一致，每条规则都被触发过；
- 离线测试 6 套全部通过，jury_v2 从 418 增至 481。

## 各步要点

**1. 取消与更正**（`src/medical_jury/v4/graph.py`）
- 按任务取消，不影响同进程其他任务。开始前取消：`start()` 拒绝；运行中：不再发起新调用，直接收尾为 `Cancelled`、不出任何发现；等待医生时：关闭等待并收尾。已完成的任务不能取消。
- `revise()` 从暂停点分叉写入新决定，新决定注明取代哪条，旧决定保留在检查点历史中；`resume()` 不接受伪造的更正链。
- 每个任务一把锁：多人同时恢复同一任务，只有一个决定生效。
- 「撤销」的范围：Agent 没有任何工具，不写外部系统，所以无需补偿操作；已花的钱不能退，靠事前预算闸门控制。

**2. 持久化**
- `open_durable(path)`：检查点与取消登记放在同一个 SQLite 文件（WAL），重启后暂停的任务可继续，取消依然有效。
- `recover(job_id)`：崩溃后从最后检查点续跑，已完成的模型调用不重发（测试证明只补发了未完成的那一次）。

**3. 调用截止层**（`DeadlineBackend`）
- 原有 httpx 超时（90 s）按阶段计，不是总时长；新增每次调用的总时长上限（默认 150 s）。
- 取消时立即停止等待；后台请求继续到结束，预算账本照常结算，花费不会漏记。

**4. 声明式规则库**（`kb/v4/rules.json` + `src/medical_jury/v4/rules.py`）
- 每条规则带指南出处和医生签核字段（目前全部为 `not_reviewed`，如实标注）。
- 行为摘要只覆盖会改变发现的内容；签核、出处属元数据，签核不需要升版本，改阈值必须升版本，否则测试失败。

**5. 追踪与回归闸门**
- `JsonlTracer`：每个节点一条记录（OpenTelemetry 风格字段），只含 ID、版本、耗时、调用状态、用量和哈希，**不含病例原文**（有测试检查）。
- `scripts/regression_gate.py`：
  - 模拟闸门：480 个病例全流程，约 5 秒；行为变了而版本没变 → 拒绝，`--accept` 也绕不过；
  - 重放闸门：`--replay records.jsonl`，在真实模型输出上重跑模型之后的全部代码。
- 你电脑上双击 `V4-Regression-Gate.cmd` 即可运行两道闸门（免费、离线）。

**6. 输入防护**（`src/medical_jury/v4/guard.py`）
- 检测指向模型的文本：「忽略以上所有指令」、伪造角色标签、本系统输出字段名、隐藏/双向控制字符，中英文。
- 命中只强制升级、不改任何发现。11 条攻击样例全部命中；10 条临床难负例（如「请忽略之前关于二甲双胍的用药指示」）、771 篇真实渲染文档和 480 篇模板文档零误报。

**7. 漏写类准则**（`kb/v4/disease-items-v4-3-dev.json`）
- `T_SICK_DAY`（模型判断）：用了二甲双胍 / SGLT2i / ACEI / ARB 时，要求写明生病时暂停并求助。
- `T_SGLT2I_CKD`（代码判断）：T2D + CKD、eGFR ≥ 20、未怀孕且完全未提 SGLT2i 时报出；文档明确停用/避免则不报。
- 只在训练集上检查：50 个无错病例零误报，病期停药准则触发的 34 例参考方案都写了相应建议。

**8. 严重断言独立复核**（`src/medical_jury/v4/verify.py`）
- 只复核模型开放复查中的严重断言，规则层发现不送复核。
- 默认策略 `flag`：不同意时只提示医生，不改严重度；`downgrade` 仅供评测使用，会标注原严重度。
- 复核模型返回乱码时记为失败调用，不会让流程崩溃。

## 已知局限（如实记录）

1. **多进程并发**：任务锁只在单进程内有效；多个进程同时操作同一任务需要 Postgres 检查点 + 数据库层任务租约。
2. **进行中的调用照常计费**：取消只停止等待，不能撤回已发出的请求。
3. **复核成本未进付费脚本**：`run_review_v4.py` 不支持开发功能；接入前必须先把复核调用加入成本上界。
4. **输入防护是报警线**：模型仍会读到注入文本，靠升级给人工兜底。
5. **开发版准则库下 O06/O09 不再是留出类型**：要证明新准则有效，需要新设计留出类型并做预注册付费运行。
6. **全部为合成数据**：误报率、命中率不能直接外推到真实平台输出；规则和模板仍未经医生审核。

## 使用示例

```python
from medical_jury.v4 import criteria as CR
from medical_jury.v4.graph import JsonlTracer, ReviewGraphV4, open_durable

lib = CR.load_library('.')                                  # 冻结库；开发库：CR.load_library('.', CR.DISEASE_DEV)
g = ReviewGraphV4(backend, lib, **open_durable('reports/v4-service.sqlite'),
                  tracer=JsonlTracer('reports/v4-trace.jsonl'))
out = g.start(case_description, document, job_id)          # 'Completed' | 'AwaitingHumanReview' | 'Cancelled'
g.resume(job_id, {'decision': 'accept', 'reviewer': 'Dr A'})
g.revise(job_id, {'decision': 'override', 'final_triage': 'pass', 'reviewer': 'Dr A', 'note': 'mis-click'})
g.cancel(other_job, reason='patient withdrew')
g.recover(crashed_job)
```

## 下一步（需要你或导师提供）

真实平台数据与医生审核规则 → 设计新的留出类型并预注册 → 付费评测开发功能（预算需另批）→ Postgres + 任务租约 → 合规文件。
