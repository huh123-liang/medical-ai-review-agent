# 逐项证据候选：仅离线实验

不被运行中的医疗Agent导入，不调用模型、不访问网络、不写数据库，不修改原平台。

- 设计：`E:\AI agnet\docs\ITEM-EVIDENCE-DESIGN-V1.md`
- contract.py：严格JSON、冻结绑定、精确定位、来源配对、逐项覆盖及声明逻辑检查。输出始终NeedsReview，不认证医学事实。
- test_contract.py：仅非医疗合成文本，预期来自协议和逻辑；不是医生标签或临床正确答案。
- 旧输出只能盘点兼容性，不能补造逐项医学证据或覆盖历史结果。

由 `E:\AI agnet\scripts\verify_iteration11_item_evidence.py` 统一冻结和运行；报告保存在新目录 `E:\AI agnet\reports\iteration-11-item-evidence`。生成器拒绝覆盖已有结果。运行需要本项目环境中已有的jsonschema/pytest，不自动安装依赖。

测试通过只证明这些结构边界按预期执行，不证明任务识别、义务拆解、事实核查或医学判分正确。7条未决样本继续等待独立专业评审，不作为优化目标或新金标准。
