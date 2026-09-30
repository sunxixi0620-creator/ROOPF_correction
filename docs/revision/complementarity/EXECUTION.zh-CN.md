# 充分训练与四组互补性实验：执行记录

用户于2026-09-29确认四组方案并授权继续。当前属于新一轮实验，不覆盖已完成修订的原始记录。

2026-09-30更新：本页所述训练与四组比较已全部完成，见[结果与取舍](CONCLUSIONS.zh-CN.md)。下文保留启动时的流程说明；最新科学结论以完成报告为准。

## 分组

| 代号 | 定义 |
|---|---|
| A | 仅离线：验证选定的冻结anchor，承担所有评估槽位 |
| O | 仅在线：固定初始化的共同算子和proxy，不加载预训练anchor/residual；两个均匀随机提案替代anchor提案，38选2 |
| F | 融合：与A完全相同的anchor，加在线候选选择；不执行residual |
| F+R（代码FR） | 完整融合：与F相同的anchor和规则，加在该anchor对应数据上重训的residual |

O可从100点初始化后立即使用全部评估槽位，不设置anchor等待期。其固定神经映射没有经过离线训练。O不被当作三个独立训练种子：每个任务/初始化只运行一次，作为三个anchor种子的共享对照。

## 已启动

1. 已校验并复制六份完整训练状态，保留模型、Adam状态和历史验证选择：10维终止轮次80/30/80，20维80/80/80。
2. 续训至1000轮，保持width200、batch16、常数学习率0.001、原损失和角色随机流；取消旧的提前停止规则。每轮9次参数更新，每5轮验证，并保存80/160/320/640/1000轮状态。此处1000轮是上限预算，不预设会收敛或必然获益。
3. GPU同时运行三个训练进程，先20维后10维。训练结束后，四组监督进程会核验全部完成记录，冻结选中anchor，采集对应标签、训练六个residual，再执行四组测试与串行计时。

在线控制和共同模块的10/20维契约检查已通过，详见CONTRACTS.json。核验包括不读取checkpoint、同状态共同36候选及分数逐元素一致、301次奇数预算、任务更名不改变结果、A路径一致，以及教师标签不改变轨迹/RNG。四组集成检查见INTEGRATION.json；使用旧权重仅检验执行合同，不计入本轮性能样本。

## 完成前不能声称的内容

本页是启动/执行说明，不是结果报告。训练完成只由训练目录COMPLETE.json确认；四组结果完成只由其自己的COMPLETE.json确认。中途验证值上升不证明融合互补或residual有效，也不等于80轮与1000轮外部性能结论。

本轮固定四组测试为同12个配方族的新参数实例：10维300、20维300/600，每条件2880条，共8640条主轨迹。不能称为全新函数族或新的CEC独立测试。F与F+R分别必须同时超过A和O，residual还需通过FR−F的独立验收。区间与实用阈值事前固定，完整保留不支持假设的结果。

容量扩大不与时长实验混做。当前不自动把width200改为400；若完整曲线和拟合证据明确指向容量瓶颈，再单独规定有限容量对照，不能凭参数少或确认集表现选择扩网。

## 路径与恢复

- 训练：`results/complementarity_20260929`；各`d20_s*/PROGRESS.json`、`d10_s*/PROGRESS.json`为逐轮原子进度。
- 四组：`results/complementarity_four_group_20260929/STATUS.json`；错误写入`FAILED.json`，不会伪造完成记录。
- 训练入口：`scripts/complementarity_training.py`；四组入口：`scripts/complementarity_four_group.py`。
- 起始状态归档：`artifacts/complementarity_training_start_v1`。所有运行身份记录代码/协议/起点哈希；已完成旧实验的源代码不改写。

恢复前先检查现有监督进程，避免重复启动。确认旧进程已退出后，用相同源码与环境执行：

```bash
.venv/bin/python scripts/complementarity_training.py train --run results/complementarity_20260929 --workers 3
.venv/bin/python scripts/complementarity_four_group.py all --run results/complementarity_four_group_20260929 --workers 12
```

第二条可在训练时并行启动，它会等待训练完成。训练以每轮提交状态恢复；四组案例通过锁与数据哈希复用。已开始的执行源文件不得修改；修复须明确使用新身份或在产生比较结果前重新冻结，不能绕过缓存检查。

预登记规则：[训练](../../experiments/COMPLEMENTARITY_TRAINING_PROTOCOL.md)、[四组比较](../../experiments/COMPLEMENTARITY_FOUR_GROUP_PROTOCOL.md)。
