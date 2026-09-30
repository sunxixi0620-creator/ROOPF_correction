冻结先验闭环融合完整归档。将全部ZIP解压到同一仓库根目录；先恢复fewshot_prior_v1，再恢复本归档，以最新核验脚本为准。包含144个任务批次、2016条完整轨迹、冻结协议/身份、结果与核验。先验checkpoint见父归档artifacts/fewshot_prior_v1。

复核（0目标查询、不重训）：`.venv/bin/python scripts/verify_prior_studies.py --study fusion`。
