# 训练充分性开发实验结果

本轮是同一80轮训练状态的配对续训，不是从头重训或外部基准领先证明。原最终10-D权重不变。
验证集与确认集均为已知36生成函数的新实例，独立参数种子；不声称全新函数族。只有一个续训随机流。

## 验证选模

| 训练目标 | 验证选中总轮次 | 验证得分 |
|---|---:|---:|
| legacy | 80 | 0.027926 |
| shared_scale | 160 | 0.067479 |

停止轮数不影响衰减日程：两组均在80轮后令多样性项权重为0；其余训练设置相同。选模指标为归一化真实改善的有界变换，而不是训练loss。

## 冻结选择后的确认集

| 方法 | 确认得分 | 相对80轮起点的72条件均值胜/平/负 |
|---|---:|---|
| original_final | 0.098064 | 50/20/2 |
| start_epoch80 | 0.026887 | 0/72/0 |
| legacy_selected | 0.026887 | 0/72/0 |
| shared_scale_selected | 0.067085 | 43/23/6 |

胜平负是描述性比较，没有多重比较显著性声明。实例与搜索初始化不能当成多个训练种子。

## 未训练gate的必要性对照

- memory_only相对full：72条件均值胜/平/负为35/20/17。
- uniform相对full：72条件均值胜/平/负为43/4/25。

memory_only仅移除随机gate logits、保留成功记忆；uniform同时去除成功记忆。本轮未训练新的辅助网络，也未移除候选生成网络。

## Residual标签

恢复的原3,240,000行日志没有anchor真值，无法直接重标为实际替换标签。新诊断保留两个anchor真值，明确比较第二个anchor；额外教师求值不更新在线状态。
- all：incumbent与第二anchor标签分歧率62.96%；best-of-two与第二anchor标签分歧率0.37%。
- late：incumbent与第二anchor标签分歧率72.72%；best-of-two与第二anchor标签分歧率0.15%。
- residual_holdout_late：incumbent与第二anchor标签分歧率73.33%；best-of-two与第二anchor标签分歧率0.14%。

不同标签的正例率、Brier和top5精度见机器可读REPORT.json。标签分歧本身不证明改标签就能提高最终优化效果；本轮没有在验证集上重新训练residual。

## 范围与交付

{
  "anchor_extra_training_points": 221184000,
  "validation_runs": 5184,
  "confirmation_runs": 1152,
  "auxiliary_runs": 864,
  "auxiliary_teacher_points": 1094400,
  "dataset_initial_value_points": 57600,
  "preflight": "288anchor validation trajectories and two four-trajectory full runs, with15200teacher points; excluded from formal counts"
}

图见TRAINING_VALIDATION_CURVES.pdf；数据、模型与哈希见artifacts/training_validation。所有原有外部测试成绩均未用于选模。
