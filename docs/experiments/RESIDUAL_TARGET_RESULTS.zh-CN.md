> 后续链路审查提示：本轮依赖的训练数据中4个实例使用旧预检种子配置，实际来源已精确恢复；训练日志score与线上特征也存在定义差异。原始数值保留，但统一配置训练与目标效果归因须按[总审查](../revision/EXPERIMENT_CHAIN_REVIEW.zh-CN.md)收紧，暂不作为干净协议下的论文主结论。

# Residual训练目标对照结果

Two training seeds; known36 families, fresh instance splits; no external test selection

原始最终模型保持不变。新模型仅改变residual训练数据/目标；同一新数据上的两种目标使用相同结构和随机种子。新数据包含anchor行，与历史训练集不同，因此不能把新旧模型全部差异归因于标签。

## 早停与选模

| 模型 | 实际训练轮数 | 选中轮次 | 验证BCE |
|---|---:|---:|---:|
| incumbent_20260928 | 10 | 2 | 0.221852 |
| incumbent_20260929 | 12 | 4 | 0.220548 |
| second_20260928 | 22 | 14 | 0.271262 |
| second_20260929 | 26 | 18 | 0.275085 |

## 完整ROOPF确认评测

每个方法36个新实例×4条轨迹，300NFE。得分是归一化改善的有界变换，不是成功率。胜平负按36个函数实例均值计算，未作显著性声明。

| 模型 | 得分↑ | 相对原最终版胜/平/负 | 相对无residual胜/平/负 |
|---|---:|---|---|
| incumbent_20260928 | 0.242949 | 14/9/13 | 14/2/20 |
| incumbent_20260929 | 0.242218 | 15/5/16 | 12/3/21 |
| no_residual | 0.243068 | 20/2/14 | 0/36/0 |
| original | 0.243355 | 0/36/0 | 14/2/20 |
| second_20260928 | 0.243830 | 18/4/14 | 8/23/5 |
| second_20260929 | 0.243662 | 18/3/15 | 8/24/4 |

## 后期候选标签表现

以下为所有38候选的Brier；top5仅取36个portfolio候选。不同标签对应不同任务，BCE/Brier不能跨任务直接解释为同一种性能。

| 模型 | 目标 | 正例率 | Brier↓ | 训练先验常数Brier↓ | portfolio top5精度 | portfolio正例率 |
|---|---|---:|---:|---:|---:|---:|
| original | incumbent | 0.1017 | 0.1122 | 0.0913 | 0.4780 | 0.1073 |
| original | second | 0.8139 | 0.4494 | 0.1591 | 0.9992 | 0.8457 |
| incumbent_20260928 | incumbent | 0.1017 | 0.0665 | 0.0913 | 0.4754 | 0.1073 |
| incumbent_20260928 | second | 0.8139 | 0.6628 | 0.1591 | 0.9985 | 0.8457 |
| incumbent_20260929 | incumbent | 0.1017 | 0.0663 | 0.0913 | 0.4805 | 0.1073 |
| incumbent_20260929 | second | 0.8139 | 0.6646 | 0.1591 | 0.9990 | 0.8457 |
| second_20260928 | incumbent | 0.1017 | 0.6472 | 0.0913 | 0.3180 | 0.1073 |
| second_20260928 | second | 0.8139 | 0.0606 | 0.1591 | 0.9990 | 0.8457 |
| second_20260929 | incumbent | 0.1017 | 0.6484 | 0.0913 | 0.3129 | 0.1073 |
| second_20260929 | second | 0.8139 | 0.0605 | 0.1591 | 0.9990 | 0.8457 |

校准分箱详见归档calibration.csv。标签确认集由无residual行为策略产生，不能等同于新模型部署后的状态分布；完整优化确认评测单独执行以检验实际收益。验证与确认仍为已知函数族的新实例。

数据预算：
```json
{
  "training_candidate_rows": 1094400,
  "validation_candidate_rows": 547200,
  "confirmation_candidate_rows": 547200,
  "collection_main_points": 172800,
  "collection_teacher_points": 2188800,
  "confirmation_optimization_points": 259200,
  "confirmation_initial_points": 14400,
  "formal_trajectory_executions": 1440,
  "preflight_parity_main_points": 2400,
  "preflight_parity_teacher_points": 15200
}
```
