# Cold-start feasibility v1

按依赖顺序将 `artifacts/fewshot_prior_v1`、`artifacts/frozen_prior_correction_v1` 和本目录的全部ZIP解压至仓库根目录。父实验提供冻结模型、GP实现和身份依赖。本轮包含完整轨迹、诊断、契约、源代码/协议和统计结果。MANIFEST.json记录每个文件与ZIP的SHA256。

只读复核（0次新增目标评估）：

```bash
.venv/bin/python scripts/report_cold_start.py verify --workers 24
.venv/bin/python scripts/report_cold_start.py report
```

首次执行的顺序是 `scripts/cold_start.py freeze`、`contracts`、`diagnostics --workers 24`、`run --workers 24`。已有结果按身份及校验和恢复，不重复计费。全新运行需要481428次目标调用，无新增神经训练。冻结协议见 `docs/experiments/COLD_START_PROTOCOL.md`，结果和边界见 `docs/revision/cold_start/`。本轮仅属开发证据。
