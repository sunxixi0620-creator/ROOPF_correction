# Frozen prior correction v1

先将 `artifacts/fewshot_prior_v1` 所有ZIP解压到仓库根目录以恢复父模型及身份依赖，再将本目录所有ZIP解压到同一根目录。本归档包含648条完整搜索、36个预测诊断、契约、冻结代码/协议和结果报告；MANIFEST.json逐文件及逐包记录SHA256。

复核已有结果（0次新目标调用、0轮训练）：

```bash
.venv/bin/python scripts/report_frozen_prior_correction.py verify --workers 24
.venv/bin/python scripts/report_frozen_prior_correction.py report
```

结果见 `docs/revision/frozen_prior_correction/CONCLUSIONS.zh-CN.md`，研究边界见同目录 `NEXT_DECISION.zh-CN.md`。原运行总397424次目标调用；新增神经网络训练0轮。代码协议在df334f3冻结，报告脚本为运行后只读统计/复核工具。
