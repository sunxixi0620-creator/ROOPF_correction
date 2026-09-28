# 两轮排序评分一致性诊断

所有36×2配置均通过诊断开关点、轨迹与RNG一致性检查。组件回退为同状态数值诊断，各组件影响可能重叠，不能相加归因。

| 配置 | 状态数 | 短名单内顺序反转 | 反转后真实胜/平/负 | 保留prior后改选 | 改选真实胜/平/负 |
|---|---:|---:|---|---:|---|
| full | 6480 | 234 | 137/0/97 | 213 | 82/0/131 |
| no_residual | 6480 | 292 | 178/0/114 | 331 | 126/0/205 |

## 组件变化

平均绝对变化为原始评分单位，不能跨函数解释为统一效果。回退数表示只恢复该项的池内值时，多少次原反转不再发生。

| 配置 | 组件 | 平均绝对变化 | 回退后消除的反转数 |
|---|---|---:|---:|
| full | mu | 7.64991e-08 | 0 |
| full | uncertainty | 5.13549e-08 | 0 |
| full | prior | 0.04376 | 159 |
| full | other | 0.00139999 | 9 |
| full | residual | 0.0122417 | 51 |
| no_residual | mu | 7.78248e-08 | 0 |
| no_residual | uncertainty | 4.80999e-08 | 0 |
| no_residual | prior | 0.0455286 | 287 |
| no_residual | other | 0.00132732 | 8 |
| no_residual | residual | 0 | 0 |

other包含结构、guard和operator修正；residual为开启与关闭learned router的分差。候选顺序变化不自动等于错误；真值胜负为第二个候选相对第一个候选。

```json
{
  "main_trajectories": 576,
  "main_points": 172800,
  "teacher_points": 38880
}
```
