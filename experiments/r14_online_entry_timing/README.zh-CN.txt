R14：在线候选介入评价时机的单参数实验
冻结参照：final-v1.1.0-cec-unlocked-20261025；实际阈值210FE。
参数：100、130、150、180、230、260；增加210FE同期对照。
每组35函数×100条配对轨迹，D10，300FE含初始100。当前Full的权重、案例、CEC解锁、算子生成、代理、Residual、风险门、LLM调用220/240及5轮融合/4轮延迟生成窗口保留。
只改变在线候选何时可替代第二个竞争Anchor的位置，保护Anchor永远保留。生成控制仍从210开始；更早开放时沿用原生候选初选和数值保护，不提前调用Agent；更晚开放时Agent照常预测与调用，但入场门强制第二个点仍为竞争Anchor，反馈仅由实际评价点产生。
默认210已在工程检查中与未修改Full逐值、逐坐标和逐决策核对一致。参数实验通过独立variant_model.py装载，不重写冻结roopf/model.py。
总共24500条新轨迹，2450个条件批次，4900个实际API请求窗口；冻结基线每个配对批次另行核验。精确目标缓存只命中形状、坐标与随机流相同的请求；API实际请求与重用分别记录，未知请求不自动重试。
启动/恢复：bash scripts/run_r14_timing.sh all --workers 24 --progress-seconds 10
查看进度：tail -f experiments/r14_online_entry_timing/LIVE_RUN.log
部分汇总：python scripts/report_r14_timing.py
最终核验：bash scripts/run_r14_timing.sh verify
结果：FUNCTION_STATISTICS_AND_COMPARISONS.csv、PAIRED_TRAJECTORIES.csv、SUMMARY.csv、PARAMETER_RANKINGS.csv、FINAL_RESULT.json。
BBOB20按既定批次随机语义单列；主要汇总为另外34函数。胜平负和平均秩为描述性结果，不是显著性检验；在已有测试集选择时机属于探索性调参，后续应独立确认。
真实API失败保留并按既定matched38回退；如某函数某参数没有任何有效返回，标记execution_complete_api_validation_incomplete，不声称完成真实LLM验证。不根据结果修改门限。
