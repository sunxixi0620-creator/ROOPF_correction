# 可写入论文的结论与未完成证据

## 建议当前定位

研究跨相关任务预训练的条件函数先验，是否能提高昂贵黑盒优化在小样本阶段的评估效率。离线先验只在前期提供预测，在线GP纠偏；固定窗口结束后，撤去先验并在所有付费观测上重新拟合在线模型。研究重点是早期样本效率，而非声称全预算、全部任务和全部组件均优越。

## 证据映射

|命题|当前状态|依据或缺口|
|---|---|---|
|离线辅助能加快同分布相关任务的早期搜索|有重复开发证据|10点初始化、40次退出在扩大实例复核中相对O节省15.34次、相对S节省16.29次|
|原固定窗口方案终局无实质退步|支持预设容忍度内非劣|600次有界效用差下界高于−0.005；不等于零损失保证|
|辅助不降低固定预算内达到率|未通过原保护条件|复核中相对O/S均值略低；不能删去此结果|
|固定空间填充保护修复了不足|不支持|H相对同批W更慢，达到率未提高|
|收益由有效离线知识而非通用偏置产生|未闭合|有正确/打乱先验的离线诊断；缺匹配闭环确认|
|在线适应必要|有开发对照支持，待确认|早期P/F/W对照不能代替冻结后独立确认|
|40次退出或自适应退出必要|未证明|固定退出未与充分独立的持续融合对照确认；未开发自适应退出|
|泛化至全新任务家族或应用|未验证|当前均为历史开发函数家族的新实例|
|旧离线residual或原anchor机制必要|未验证|当前模型身份已是函数先验＋在线GP，不是原三模块实现|

## 可用英文结果表述

On additional instances from the development task distribution, a frozen conditional prior used during the first 40 objective evaluations reduced the restricted time to a prespecified target relative to both an online GP and a space-filling initialization control. The mean savings were 15.34 and 16.29 evaluations, respectively. Terminal performance satisfied a prespecified noninferiority tolerance. However, target attainment at 300 evaluations was slightly lower, so the joint acceptance criterion was not met. A separately frozen spatial-coverage variant did not resolve this limitation and was not adopted.

These results support early sample-efficiency benefits under the studied distribution, rather than universal superiority or the necessity of every component. The task families have development exposure; new-instance replication is not evidence of generalization to untouched benchmark families.

## 下一轮不应直接做什么

不继续搜索Sobol比例、退出时间或神经训练轮次。先决定可接受的科学主张和实用损失容忍度。若另行采用达标率非劣容忍度，必须基于应用代价预先论证、在全新协议和数据上检验；绝不能用来重新判定本轮通过。对本文原有充分性门槛的失败应保留。

本轮完成有界改进闭环：复核假设、测试一个修复机制、验证和归档正负证据。结果没有达到全面优越，不能称完美；继续推进必须提出新的可证伪机制，而不是无限追求同一表格全胜。
