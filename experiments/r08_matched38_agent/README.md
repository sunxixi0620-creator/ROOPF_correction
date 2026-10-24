# R08：以 matched38 为默认底座的完整 LLM Agent

按用户确认，保留离线案例检索、历史在线反馈、五轮回顾、Anchor 温度、局部步长和全部八个联合动作。默认及 API 失败回退使用 `matched38`；`release` 仍是 LLM 可以明确选择的原版动作。动作窗口之外仍按冻结 matched38 原有日程使用 release。

这次回滚指恢复冻结 matched38 的实际搜索行为，不删除旧版本。回滚前的代码存于 `PRE_MIGRATION_CODE.tar.gz`，文件及旧科学记录的 SHA256 存于 `PRE_MIGRATION.json`。R07 代码和结果保留，R08 使用独立分支和目录。

离线案例必须重新采集，标签参考动作由原版 release 改为 matched38。原案例不会直接改名冒充新策略标签。仍采用36生成函数×4种子，29训练函数进入检索、7函数隔离保留，5/10轮配对收益，不按效果筛选动作。

完整运行：

```bash
conda activate ooneurgo
cd /home/skq/repos/roopf_agent/ROOPF
bash scripts/run_r08_hardware.sh all --workers 24
```

此命令依次完成全量采集、案例库、六臂35函数×100轨迹、真实API匹配核验、报告与归档。D10、300次评价含初始100、模型配置、API上限600次、CEC结构锁、BBOB20单列等设置保持。仍使用CPU搜索与评价、固定请求形状的GPU在线代理计算和实时资源进度。

报告同时与冻结原版及 matched38 比较。报告前要求本轮 matched38 的100×300评价值逐函数与R07冻结 matched38 完全一致；否则拒绝完成报告。

工程检查通过不等于全量实验完成，也不代表优化改善。工程检查中的失败回退合约使用了明确标记的 mock；正式 `llm_live` 只接收配置中的真实API响应，失败按 matched38 回退且不自动付费重试。

底座是在观察R07结果后选定，本轮属于开发验证。完整实验结束前不修改评分系数、动作、门槛或测试设置。
