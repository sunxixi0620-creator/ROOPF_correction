# 强在线参照：运行说明

当前批次为两个新在线参照：标准拟合GP／LogEI，以及TuRBO-1／LogEI变体。
固定36配置×2既有实例，144条轨迹，300次预算、相同初始10点。
此批不重训学习先验，不新增W样本，不是独立未见函数族确认。

V1发生TuRBO暖初始化线搜索失败。已完整保留六条完成GP轨迹；165点的
确定性复现定位失败，冷初始化同一模型能继续拟合。V2只在拟合失败后
用默认参数重建同一GP并重新拟合，记录恢复次数；不增加目标查询。
逐步原子检查点和断点续跑契约已通过。V1未完成任务的调用数只能给出
140～4,200范围，不能精确重建；正式报告须保留这项限制。

执行使用16个CPU worker。小GP在本机CPU比GPU更快，GPU用于独立后验
分数重放。主运行完成后命令队列依次进行全144条记账/边界/状态审查、
CPU/GPU重放、固定统计报告和ZIP哈希归档。出现未恢复错误时不生成
最终成绩表，保留失败点供继续诊断。

状态文件：results/strong_online_baselines_v2/STATUS.json
逐任务进度：results/strong_online_baselines_v2/progress/*.json
日志：results/strong_online_baselines_v2/run.log

只有RESULTS.json及VERIFICATION.json均生成后才能给出本批完整比较结论。
既有“学习先验相对解析先验的增量未达门槛”结论不会因本批启动而改变。
