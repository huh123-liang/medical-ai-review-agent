# 论文、公开基准与方法取舍

详细历史对照表见[论文与缺口对应](../review-methods-v1/paper-gap-matrix.md)。下列是采用理由，不代表相关框架已全部接入。

| 资料 | 对本项目的用途 | 不能直接迁移的结论 |
|---|---|---|
| [HealthBench](https://arxiv.org/abs/2505.08775) | 原医生逐项标注、裁判meta-evaluation、主题分层与标签隔离 | 论文Macro-F1不等于本项目成绩或临床准确率 |
| [LLMEval-Med](https://arxiv.org/abs/2506.04078) | 核心/次要checklist、开放回答的人工复核 | 有专家参考答案不等于有某固定模型回答的医生评分 |
| [TALEC](https://arxiv.org/abs/2407.10999) | 分准则判断、例子和zero/few-shot对照 | 非医疗结果不保证医疗表现，不必照搬多轮调用成本 |
| [Many-shot ICL evaluators](https://arxiv.org/abs/2406.11629) | 经过核查的示例与稳定性检查 | 数学任务、部分模型评价；稳定不等于正确 |
| [TextGrad](https://arxiv.org/abs/2406.07496) | 后续候选Prompt、反馈、验证与回滚思路 | 没有在本项目运行；不是训练DeepSeek权重 |
| [PH-LLM](https://arxiv.org/abs/2406.06474) | 纵向个人健康任务和专家Rubrics组织 | 可穿戴/睡眠运动成绩不是通用医疗裁判一致性 |
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | 将引用定位、支持关系与覆盖分开 | 自动蕴含判断不是医生真值或临床安全认证 |
| [MORQA](https://github.com/wyim/morqa-eval) | 未来部分英语子集医生多维评分；中文评审者身份另行区分 | 本项目未下载适配，不混算不同标注尺度 |

## 本轮具体借鉴

逐条保留原准则；证据不足与明确错误分开；定位不当语义真值；原医生标签预测后才读；失败、未定和未运行都在计划分母；按组隔离和bootstrap。

## 数据与许可

公开论文链接只用于方法说明，不复制整篇PDF。官方数据使用各自原始许可；项目不自动再分发原病例、匿名医生标注或原模型回答。作者原始数据版本/哈希在评测说明与代码里可追溯。
