# Cold-start coverage v1

先恢复 `artifacts/cold_start_v1` 及其列明的父依赖，再将本目录全部ZIP解压到仓库根目录。包括1728条完整轨迹、冻结协议/代码、结果与零调用重放核验；MANIFEST逐文件和分包记录SHA256。

```bash
.venv/bin/python scripts/verify_cold_start_coverage.py
.venv/bin/python scripts/cold_start_coverage.py report
```

以上复核不新增目标调用。本轮新运行1036845次评估、0轮神经训练。固定10个Sobol点与20个融合选点的早期分配，40次后退出，不能称未见函数族确认。原冷启动开发结果保持独立。
