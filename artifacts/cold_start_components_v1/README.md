# Frozen cold-start component study v1

先恢复 `artifacts/cold_start_v1` 及其列明的父依赖，再将本目录全部ZIP解压到同一仓库根目录。包含3240条完整搜索轨迹、契约、冻结代码/协议、统计及零调用核验。MANIFEST记录每个文件和ZIP的SHA256。

复核既有结果，不新增目标调用：

```bash
.venv/bin/python scripts/report_cold_start_components.py verify
.venv/bin/python scripts/report_cold_start_components.py report
```

全新实验顺序是 `scripts/cold_start_components.py freeze`、`contracts`、`run`；已有结果按身份和校验和复用。本轮主搜索1944000次＋契约180次＝1944180次目标调用，新增神经训练0轮。角色cold_start_components_v1不同于之前实验，但任务家族已有开发接触，不冒称未见函数族确认。

结果见 `docs/revision/cold_start_components/CONCLUSIONS.zh-CN.md`；方法细节见同目录 `METHOD.zh-CN.md`。每个端点内的六比较使用99.1667%区间，不是全部18端点的联合覆盖保证。前序未通过的严格验收结果不被改判。
