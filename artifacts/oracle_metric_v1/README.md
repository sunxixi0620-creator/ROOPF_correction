# Restricted oracle metric diagnostic

恢复 `artifacts/cold_start_components_v1` 及其父依赖后，将本目录所有ZIP解压到仓库根目录。包含720条完整轨迹、真实/随机度量、契约、代码/协议、统计和输入身份。MANIFEST逐文件及分包记录SHA256。

```bash
.venv/bin/python scripts/report_oracle_metric.py verify
.venv/bin/python scripts/report_oracle_metric.py report
```

复核0次目标调用；原运行432000主调用＋90契约＝432090次，0轮训练。真实方向/轴谱是特权信息，不是可部署算法，也不是完整结构性能上界。结果与边界见 `docs/revision/oracle_metric/`。
