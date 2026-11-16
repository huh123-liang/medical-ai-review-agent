# Jury v2 运行手册（方案 3：DeepSeek Pro 单判官 + DSPy）

在项目根目录（有 `data/` 的那个目录）打开 PowerShell，激活 `.venv` 后按顺序执行。**带 `--authorized` 的命令才会真正调用 API 并产生费用**，其余命令都是离线的。协议见 [PROTOCOL-V2](PROTOCOL-V2.md)。

## 0. 一次性准备（离线）

```powershell
python -m pip install -e ".[test,jury2]"
copy examples\jury-v2.example.json jury-v2.json
python -m pytest
python scripts\human_ceiling.py --no-index      # 只在内存里算医生间一致性参照，data/ 下什么都不写
# 需负责人批准后才执行下一行：新建派生的评分索引（只新建，不覆盖；同时写 .meta.json 记录来源哈希）
python scripts\human_ceiling.py
python scripts\gate_sanity_check.py
```

- `--no-index` 只写 `reports/human-ceiling/`。
- 不带 `--no-index` 时会新建 `data/private/physician-rater-index.jsonl` 和同名 `.meta.json`。如果文件已经存在，脚本先核对来源哈希和内容哈希：一致则复用，不一致则拒绝运行，绝不覆盖。按当前代码，内容的 SHA256 应为 `b7a3f1c0f11476d08f488911231c5fa904861a69d1ecb780dba6fca9ac525fd2`。
- 所有付费脚本在调用模型**之前**都会检查评分索引是否就绪；没有就绪就直接退出，不会先花钱再发现缺索引。
- 付费脚本的计划会显示账本已用、剩余额度、费用上界、写入位置和停止条件。费用上界超过剩余额度时默认拒绝运行；`--allow-partial` 表示到上限就停、其余条目记为 NotRun，只有负责人明确批准时才能使用。
- 预测是逐条写进 `predictions.partial.jsonl` 的。预算被拦截、触发停止规则或按 Ctrl+C 时，已完成的结果全部保留，其余条目记为 NotRun，分母保持完整。
- 停止规则见 [PROTOCOL-V2 第 6 节](PROTOCOL-V2.md)：401/402/403/429 立即停止；连续 3 条失败停止；前 50 条失败超过 5 条停止；预算拦截停止。`summary.json` 里的 `stop_reason` 写明停止原因，`output_length` 给出输出 token 数（常规中位数）、`finish_reasons` 分布和真正的 `finish_reason=length` 截断数。
- 每条预测都记录 `raw_chars` 和 `raw_sha256`（原始输出的长度和哈希），但不保存原文。要保存原文需要另行批准存储方案。
- 同一样本再次运行时必须加 `--rerun-of <上次运行目录>`，否则脚本拒绝运行。
- v2 与旧流程共享费用账本，累计上限 **200 元**（负责人 2026-10-26 14:04 批准，超出原 100 元的部分由负责人自费；此前依次为 30 元、100 元）。可以用 `JURY_V2_BUDGET_CNY` 设置更低的正数上限，但不得超过 200 元。每次运行先扣除已有支出，再检查本次计划的费用上限；余额不足就不能继续，也不能把这 200 元理解为「新增 200 元」。如果 `.env.local` 里写了 `JURY_V2_BUDGET_CNY=100`（或 30），计划里的 `ceiling_cny` 仍会是那个值，需要负责人自己改成 200 或删掉这一行。
- 新的 dev 运行默认排除已经发给模型的对话组（计划里的 `excluded_used_groups`）；`--rerun-of` 只接受本样本上已开始过的运行目录。
- 蜕变测试和重复调用另写 `metamorphic.partial.jsonl`、`repeats.partial.jsonl`，同样逐条落盘、共用停止规则。

## 1. Dev 基线与错误分析

```powershell
python scripts\run_dev_v2.py --backend panel --judge deepseek-pro --n-groups 200 --metamorphic-per-kind 20 --repeats 3
python scripts\run_dev_v2.py --backend panel --judge deepseek-pro --n-groups 200 --metamorphic-per-kind 20 --repeats 3 --authorized
python scripts\export_error_review.py reports\dev-v2-deepseek-pro-<时间戳> --limit 100
```

第一条命令只打印计划和费用上界，第二条才真正运行。

**人工步骤（不能省）**：打开 `error-review.csv`，逐条填写 `error_type` 和 `notes`。根据错误类型修改 prompt 或签名说明，每次修改都只在 dev 上验证。

## 2. DSPy 优化（只用 dev）

```powershell
python scripts\optimize_dspy.py --judge deepseek-pro
python scripts\optimize_dspy.py --judge deepseek-pro --authorized --run-limit 40 --max-calls 1500
```

同样，第一条只出计划，第二条才运行。产出物：`program.json`、`threshold.json`（dev 上拟合的阈值）、`summary.json`（优化前后的 dev 对比）。

## 3.（可选）Pro vs Flash 消融

```powershell
python scripts\run_dev_v2.py --backend panel --judge deepseek-flash --n-groups 200 --authorized
python scripts\ablation_jury.py reports\dev-v2-deepseek-pro-<t> reports\dev-v2-deepseek-flash-<t>
```

Flash 必须使用和第 1 步**相同的** `--n-groups` 和默认 seed，否则两次运行的样本对不上。只有在 dev 上明显优于 Pro 单判官时，才改用多判官。

## 4. 冻结（离线）

```powershell
python scripts\freeze_protocol.py --judges deepseek-pro --program deepseek-pro=reports\dspy-opt-deepseek-pro-<t>\program.json --threshold <threshold.json 里的 chosen_threshold>
```

`freeze.json` 已经对代码树、协议、DSPy 程序和样本清单做了哈希，冻结之后改动其中任何一样，第 5 步都会拒绝运行。

需要对外留证时，把 `freeze.json` 复制到 git 仓库（`github-release\medical-ai-review-agent`）的 `benchmarks/freezes/` 下，然后 commit 并 `git tag protocol-v2-<freeze_id>`。这里的时间戳证明你是在看到结果**之前**冻结的。

## 5. 一次性验证（每种只能跑一次）

```powershell
python scripts\run_validation_once.py reports\protocol-freeze\<id>\freeze.json --kind natural
python scripts\run_validation_once.py reports\protocol-freeze\<id>\freeze.json --kind natural --authorized --run-limit 150
python scripts\run_validation_once.py reports\protocol-freeze\<id>\freeze.json --kind panel --authorized --run-limit 150
```

不带 `--authorized` 时只出计划，不加锁；带上之后才会加锁并调用。两个都通过（Δκ 非劣效 + alt-test）才进入第 6 步。没通过就回到 dev 修改方法，重新冻结；这份 validation 从此算作开发证据。

## 6. 最终测试（只跑一次）

```powershell
python scripts\run_test_once.py reports\protocol-freeze\<id>\freeze.json --kind natural --final --authorized
```

## 7. 外部验证：MedExpert（医生标注的遗漏 / 事实错误）

数据已经放在 `data/external/medexpert/medexpert-all.jsonl`，来自 Hugging Face 公开数据。

```powershell
python scripts\run_external_medexpert.py --freeze reports\protocol-freeze\<id>\freeze.json
python scripts\run_external_medexpert.py --freeze reports\protocol-freeze\<id>\freeze.json --authorized
```

第一条只出计划，第二条才运行。共 1080 次调用，保守上界约 120 元，实际通常低得多。每个系统只能跑一次。

还没冻结时，也可以先用当前判官看一个**初步**结果：

```powershell
python scripts\run_external_medexpert.py --judge deepseek-pro --authorized
```

结果在 `reports/external-runs/medexpert-*/summary.json`，会标注为 PRELIMINARY。

## 结果写到哪里

| 路径 | 内容 |
|---|---|
| `reports/frozen-runs/<id>-<split>-<kind>/summary.json` | 完整指标、验收结果、常数基线、按主题拆分 |
| `predictions.jsonl` | 逐条预测 |

这些文件都不提交到 Git，只把聚合数字整理进 `benchmarks/summaries/`。

## 简历上可以怎么写（只填实测数）

"在 HealthBench 公开医生标注上，先测出医生间一致性参照（κ=0.39）；在 N 条预注册验证集上，Jury 与医生的 κ 为 X（Δκ 95% CI [a, b]），alt-test 结果为 [通过/未通过]。"

不通过也照实写。这本身就是一个可以在面试中讲清楚的严谨结果。
