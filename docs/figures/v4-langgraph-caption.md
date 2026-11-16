# Figure: LangGraph orchestration of the v4 review agent (review-v4-graph-3)

Files: `v4-langgraph.pdf` (vector, for submission), `v4-langgraph.svg` (vector, editable), `v4-langgraph.png` (600 dpi).
Source: `scripts/plot_v4_graph.py`; the drawn topology is the one asserted against the compiled graph by
`tests_v2/test_v4_graph.py::test_topology_matches_the_documented_graph`.
Size: 183 mm × 124 mm (double-column). Fonts: Liberation Sans (Arial metrics) and Liberation Mono, embedded as TrueType.
Palette: Okabe–Ito (colour-blind safe).

## Caption (English)

**LangGraph orchestration of the v4 error-detection agent.** Each case passes through a state graph in which only two
nodes call the language model in the evaluated configuration (blue): `extract` turns the case description and the
advice document into structured patient facts and medicines with verbatim quotes, and `check` judges every applicable
criterion and then performs an open review. All other nodes are deterministic code (white). After extraction, three
code nodes run in one parallel superstep (fork/join): `verify_extraction` checks that quotes occur in the source and that
every mentioned drug was extracted, `rule_layer` applies 10 prescribing rules read from a declarative, versioned rule
file, and `select_criteria` instantiates general and disease-specific criteria with the patient's values. `aggregate`
merges the findings and downgrades severe model claims that contradict checks already made by code; `triage` assigns
*escalate*, *pass with findings* or *pass*. Escalated cases pause at `human_review` (orange) through `interrupt()`.
Every job is checkpointed in SQLite under its own thread, so a paused or crashed job survives a restart (`recover`), a
job can be cancelled at any point without further model calls (`cancel`), and a clinician decision can be replaced
without overwriting the earlier one (`revise`). Dashed boxes are development features that are off by default and
were not evaluated (`screen_input`: a code tripwire for text addressed to the model; `verify_severe`: a second model
re-examines severe open-review claims). Dashed arrows are conditional edges: the arm router (full pipeline or the
model-only ablation `check_only`) and the extraction-failure path. Bottom: channels of the shared graph state; every
node returns a partial update and emits one trace span that contains no case text.

## 图注（中文）

**v4 纠错 Agent 的 LangGraph 编排。** 在已评测的配置中，只有两个节点调用大模型（蓝色）：`extract` 把病例描述和建议文档读成带原文引用的结构化患者事实与用药；`check` 对每条适用准则作出判定并做开放复查。其余节点均为确定性代码（白色）。提取完成后，三个代码节点在同一个并行超步中运行（fork/join）：`verify_extraction` 核对引用是否出自原文、提到的药是否都被提取；`rule_layer` 从带版本的声明式规则文件执行 10 条用药规则；`select_criteria` 把通用准则和疾病准则代入患者数值。`aggregate` 合并发现，并把与代码检查矛盾的模型严重断言降级；`triage` 给出升级、带发现通过或通过。升级的病例在 `human_review`（橙色）处通过 `interrupt()` 暂停。每个任务在 SQLite 中按独立线程保存检查点：暂停或崩溃的任务重启后可继续（`recover`），任务可随时取消且不再发起模型调用（`cancel`），医生决定可以更正而不覆盖原决定（`revise`）。虚线框为默认关闭、未经评测的开发功能（`screen_input`：检测指向模型的文本的代码报警线；`verify_severe`：第二个模型复核开放复查中的严重断言）。虚线箭头为条件边：分组路由（完整流程或只用模型的消融组 `check_only`）和提取失败路径。底部为共享图状态的各通道；每个节点只返回部分更新，并产生一条不含病例原文的追踪记录。
