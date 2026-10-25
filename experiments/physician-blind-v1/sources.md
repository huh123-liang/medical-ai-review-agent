# 有外部医疗评审标注的公开基准：本轮核对

查阅日期：2026-10-25；原始来源优先。公开可获取不等于任意可再分发，本轮仅本地研究，不推GitHub。

| 基准及原文 | 能确认的材料 | 本轮适合程度 |
|---|---|---|
| [HealthBench meta-evaluation代码](https://raw.githubusercontent.com/openai/simple-evals/main/healthbench_meta_eval.py)、[论文§8.1](https://arxiv.org/html/2505.08775v1) | 官方INPUT_PATH指向固定JSONL；rubric、prompt、completion、binary_labels、anonymized_physician_ids可形成同一回答的逐医生对照。论文明确比较原医生met/not-met与裁判 | 最适合本轮：与现有二分类任务兼容，有本地可追溯原始文件，能冻结标签后实测。只验证给定准则的裁判，不证明检索或临床事实全覆盖 |
| [MORQA论文](https://arxiv.org/abs/2509.12405)、[作者仓库](https://github.com/wyim/morqa-eval)、[标注入口](https://osf.io/kcv2n/overview) | 作者仓库说明EN woundcare/iiyi模型回答有执业医生多维0/0.5/1评分及部分评语；部分中文为医学院背景领域专家，liveqa为NLM研究员，不能统称所有标签都由执业医生给出 | 有价值的第二来源，适合以后多维/中英扩展。需核对具体文件、响应ID对齐、评审者身份、许可、序数评价协议；本轮未下载/适配或虚构测试结果，不把0.5强改成二分类 |
| [LLMEval-Med作者仓库](https://github.com/llmeval/LLMEval-Med)、[公开dataset.json](https://raw.githubusercontent.com/llmeval/LLMEval-Med/main/dataset/dataset.json) | 公开样例包含problem、sanswer、checklist等；参考答案来自专家。论文有人工与机器评价，但这份公开文件不是“某固定模型回答+医生评分”的成套元评估记录 | 可参考任务/检查清单；不能仅凭专家参考答案就声称有独立医生对Agent预测的评分标签。本轮不将其强行当二分类gold |

本轮实际实验只用HealthBench，不把文献中的0.709、92.47%或其他任务成绩当本项目结果。原文件包含上游canary等元信息，本地原始数据保持不变，不把这些元信息或医生评分作为裁判输入。

在当前资料中，医生分歧必须原样保留。外部标注能提供可复核参考，而非证明所有医学争议已解决。模型输出与医生不同需后续逐项复核，不自行改标签。
