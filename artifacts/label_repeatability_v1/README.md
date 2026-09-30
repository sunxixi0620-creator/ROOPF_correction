完整实验归档：解压全部ZIP到同一仓库根目录。包含原始数据、源身份、冻结协议、结果与核验。

复核（不调用目标函数、不重新训练）：`.venv/bin/python scripts/verify_prior_studies.py --study label`。

标签审查还依赖 `artifacts/long_value_v1` 的父数据与其已归档依赖。
