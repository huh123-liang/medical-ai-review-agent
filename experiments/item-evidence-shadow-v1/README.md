# Offline shadow adapter / 默认关闭

详细说明：`E:\AI agnet\docs\SHADOW-ADAPTER-V1.md`。

本轮最终报告默认目录：`reports/iteration-12-shadow-adapter-r3`。前两次测试/报告问题的记录保留，不覆盖。

- shadow.py：被动旁路，只检查调用方提供的离线JSON，不调用模型。
- offline_cli.py：独立命令，默认不读文件、不写存储。`--enable-shadow` 才启用；没有挂到旧API或启动入口。
- test_shadow.py：非医疗工程测试，原API/分数不变，7例未决不动。

运行本轮验证（拒绝覆盖已有报告）：

```powershell
& 'E:\AI agnet\.venv\Scripts\python.exe' -X utf8 -B 'E:\AI agnet\scripts\verify_iteration12_shadow.py'
```

已有报告的只读复核：

```powershell
& 'E:\AI agnet\.venv\Scripts\python.exe' -X utf8 -B 'E:\AI agnet\scripts\verify_iteration12_shadow.py' --verify
```

独立CLI默认关闭：

```powershell
& 'E:\AI agnet\.venv\Scripts\python.exe' -B 'E:\AI agnet\experiments\item-evidence-shadow-v1\offline_cli.py'
```

离线显式启用需要 `--enable-shadow --primary-file <项目内JSON> --request-id <唯一尝试编号>`；逐项候选另外提供 `--snapshot-file` 与 `--candidate-file`。缺少新候选只记缺项，不能自动从历史回答“补齐”。存储只能在 data/shadow-item-evidence-v1 下。

请勿将本目录生成的研究记录当训练金标准或正式评分。原DeepSeek配置、预算及旧平台均保持不变。
