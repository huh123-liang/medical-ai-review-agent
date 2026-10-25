# 第15轮 r2：公开医生标注的冻结与真实Agent盲评

首次离线门为20通过、1测试导入失败，未付费。原实验和失败报告完整保留在physician-blind-v1及iteration-15-physician-blind。r2只为该对照测试补相邻fixtures模块的导入路径；Prompt、固定seed、抽样、模型、标签规则与预算均不改。r2采用新实验/报告/private目录，不覆盖首遍。

本轮用户授权：寻找有独立医生标注的公开医疗基准，冻结标签，再让现有Agent预测并与医生比较。沿用V4 Pro；累计30元以内，原平台与既有7例不动。

## 选择的数据与“正确答案”边界

采用[HealthBench官方meta-evaluation代码](https://github.com/openai/simple-evals/blob/main/healthbench_meta_eval.py)所指向的公开JSONL。原文[§8.1](https://arxiv.org/html/2505.08775v1)说明其元样本包含准则、对话、固定回答和医生met/not-met判断。不是仅有医生撰写参考答案，也不是另一个LLM生成的评分标签。

原文件已在本项目，使用现有版本并复核SHA256和抽中记录；不重写原始数据、标签或分区。标签代表原医生对指定准则的判断，不等同于所有医学事实的绝对真值。医生可能彼此分歧，逐人保留，不先多数投票抹平。

更多候选与适配边界见[sources.md](sources.md)。本轮不同时接多数据集，避免不同标注尺度混算。

## 冻结与预测顺序

1. 从validation中按固定seed、7主题各2个未使用对话组抽取，共14组，一组一条准则。不根据医生标签/旧预测/回答质量筛选；排除历史运行和索引组。
2. 冻结完整原对话、固定回答、原准则及每个消息hash。原始医生标签单独放在 `data/private/iteration-15-physician-blind-r2/references.json`，保留匿名评审ID、上游行号及来源hash。
3. 运行前冻结协议、消息、脚本、原代码与预算；先跑阻网单元测试。
4. 预测器只反序列化label-free inputs/schedule；可校验标签文件hash但不读取标签值。模型没有文件/工具访问。原医生标签、研究建议、旧模型判定不进入消息。
5. 真实预测及所有失败/未运行状态全部封存后，独立analyze.py才读取医生标签，逐人比对、计算指标并输出逐条分歧。

这是代码和流程隔离，不是对本机所有者/研究人员的密码学盲法；公开资料也可能已进入基础模型训练。SHA证明未改，不能证明无人看过或医学上正确。

## 两臂与范围

- A：现有正式默认单Agent（scoped-review-v1、anchored-v2），原1200输出tokens。运行现有Engine，不修改核心。
- B：第14轮冻结的B1无示例三态研究候选，2400输出tokens。使用单独实验适配器，不注册到旧API。
- B的原开发构造器仅允许dev，不能伪称validation是dev绕过。新common.py提供明确的**仅验证用**组装入口，逐字复用冻结B1 system/schema、空示例及原输入，单元测试核对一致性；无训练/记忆写入，原构造器和门禁不改。
- B没有可靠原子临床拆解，故原准则全文只作为O1，manifest保持research_draft，外部来源为空。不声称已逐原子医学义务核查、已完成事实RAG或指南验证；因此可能高比例unable_to_determine。
- 输出格式与token上限不同，两臂是当前产品基线与新研究候选的描述性对照，不是严格单因素Prompt消融。B2三示例版本轮不跑。

## 预算和停止条件

计划14×2=28次、单次只尝试1次；本轮保守上限8元，累计不超过30元。按UTF-8字节+2048 framing及最大输出token预留，不假设缓存/闲时折扣。运行前必须整批上界可负担，否则不调用、不静默换样本。

已复核DeepSeek官方中文[价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)，2026-10-25页面Pro高峰输入9元/百万、输出27元/百万；继续用既有更保守13.2/39.6元上限，不修改.env。网络/失败费用保留；只有成功usage可释放预留。

401/402/403、预算异常或连续3次连接失败停止全部；候选连续3次格式/截断/信封失败停止候选臂，基线继续；不修改Prompt后重试。所有未完成项保留完整计划分母。

## 结果口径

- 逐医生agreement、二分类Macro-F1、Cohen Kappa与混淆矩阵；不是1–5评分，因此不编造Spearman、MAE或加权Kappa。
- 若一臂二元可比较覆盖不足，整批主指标保持空，另报已决条件性诊断与全部计划覆盖、失败和未定。
- 报7主题、医生分歧、always-met/not-met朴素基线、按对话组分层重抽样区间；14组只能做小规模先导，不外推医院总体准确率。
- 引用准确定位、医学生命题支持和临床安全不是同一件事。本轮不新增医生裁定，不把模型理由当事实来源。

## 文件及操作门

- prepare.py：协调员抽样/封存，读取原标注但不发送模型。
- predict.py：默认仅preflight；显式--real且离线测试门通过后才收费，不反序列化医生标签。
- analyze.py：必须存在prediction-seal.json后才读封存标签，不覆盖既有分析。
- metrics.py / test_protocol.py：纯计算及阻网测试，不生成模型答案。

项目Python下先prepare，再离线测试，审核preflight后才predict --real，最后analyze。报告在 `reports/iteration-15-physician-blind-r2`；原标签在private，原core/.env/旧实验/平台不变，正式Agent账本只追加本轮真实任务及费用。

每阶段都核对：单Jury、原文/准则不改、标签不进提示、费用有界、失败留痕、7例未决。完成真实对照后停止，列下一步等用户审批；不根据本轮validation结果立即改Prompt并重跑刷分。
