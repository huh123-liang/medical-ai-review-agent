# 仅一次旧回归修订：complex-boundary-v2

v1在旧10条回归从70%降至60%，两组均10/10完成。唯一新增医生一致组退步是旧67条。原输出与20次费用全部保留，未把v1设为默认，未抽新样本。

可观察问题：回答开头提及紧急评估，不代表其余长篇鉴别/操作流程都适合“urgent”请求。v1对“无用展开”检查仍过于宽松。

只对detailed_appropriate追加任务/受众与行动优先级说明；strict factuality及simple_appropriate的v1提示词保持相同。没有加入腰穿、药物、病例编号、医生答案或长度硬阈值。

最多再做这一次旧10条v2回归。之后冻结候选，不能为使新80条成绩好看再调整；新样本抽样不读标签。若仍不支持推广，保留旧默认并报告，而不删除失败或无限调Prompt。

v1在 `reports/iteration-5/calibration/`；v2在 `reports/iteration-5/candidate-v2/`，分别保留manifest、代码快照与费用。两个旧回归阶段均计入同一10元本轮上限和30元累计上限。
