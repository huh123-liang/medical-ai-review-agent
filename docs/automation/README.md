# 可选的GitHub离线测试模板（未启用）

`offline-checks.yml.example`保留自动测试配置，但目前不作为活动工作流运行。本次发布没有要求增加GitHub账户的`workflow`权限，也没有上传任何模型密钥。

项目所有者以后可以在GitHub网页中审阅此文件，明确授权后复制到`.github/workflows/offline-checks.yml`，再自行提交。需要确认依赖安装的外网访问、Actions额度和触发条件。不需要给工作流添加DeepSeek API Key。

模板只安装测试依赖，执行`tools/check_publication.py`及`tools/run_offline_checks.py`；测试阶段阻断外部网络。不调用收费模型，不下载医疗基准，不上传逐例病例/标签/模型回答。

目前已验证的是本机导出副本及干净Git克隆的离线测试；没有声称远端CI运行成功。
