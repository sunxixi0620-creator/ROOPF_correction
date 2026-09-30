# 共享方向结构的可迁移性试验

预设20项matched比较全部通过；仅证明受控共享方向分布的可迁移信息，不是通用融合或原ROOPF成功。

| matched主比较 | 平均差 | 99.75%区间 | 通过 |
|---|---:|---|---|
| box:mse:Learned-Random | +0.92789 | [+0.91999,+0.93508] | True |
| box:regret:Learned-Random | +0.89466 | [+0.76113,+1.01983] | True |
| box:mse:Learned-Wrong | +0.92639 | [+0.91650,+0.93586] | True |
| box:regret:Learned-Wrong | +0.87470 | [+0.78442,+0.96687] | True |
| box:mse:Learned-RadialLinear | +0.80902 | [+0.78701,+0.82939] | True |
| box:regret:Learned-RadialLinear | +0.66849 | [+0.54285,+0.81189] | True |
| box:mse:Learned-OnlineFull | +0.71520 | [+0.69172,+0.74140] | True |
| box:regret:Learned-OnlineFull | +0.30389 | [+0.20648,+0.41668] | True |
| box:mse:Learned-OnlineRidge | +0.73764 | [+0.71288,+0.76248] | True |
| box:regret:Learned-OnlineRidge | +0.62087 | [+0.47483,+0.78594] | True |
| shell:mse:Learned-Random | +0.93500 | [+0.92346,+0.94556] | True |
| shell:regret:Learned-Random | +0.30773 | [+0.27676,+0.33897] | True |
| shell:mse:Learned-Wrong | +0.93459 | [+0.91715,+0.94748] | True |
| shell:regret:Learned-Wrong | +0.29884 | [+0.25928,+0.34391] | True |
| shell:mse:Learned-RadialLinear | +0.90562 | [+0.88828,+0.92106] | True |
| shell:regret:Learned-RadialLinear | +0.14475 | [+0.12392,+0.16765] | True |
| shell:mse:Learned-OnlineFull | +0.89819 | [+0.88054,+0.91344] | True |
| shell:regret:Learned-OnlineFull | +0.10098 | [+0.08061,+0.12220] | True |
| shell:mse:Learned-OnlineRidge | +0.90087 | [+0.88490,+0.91504] | True |
| shell:regret:Learned-OnlineRidge | +0.11700 | [+0.09974,+0.13439] | True |

MSE项是相对降低，regret项是候选池归一化选择遗憾的绝对降低；不是全局最优regret或终局效用。
学到的子空间投影误差平均0.039232，仅用于事后核验；算法未读取真实子空间。
12个独立方向组共享同一解析生成结构，不能称12个不同函数族。三个源数据种子各自闭式拟合，无神经网络epoch、无测试调参。
目标context10/20/40；同半径候选均在半径4.5球面。源训练184,320次、目标标签423,936次、契约128次，共608,384次；候选标签采集全部算作研究成本。

| mismatch诊断 | 平均差 | 99.75%区间 |
|---|---:|---|
| box:mse:Learned-Random | -0.02519 | [-0.08114,+0.03287] |
| box:regret:Learned-Random | +0.06793 | [-0.08270,+0.23234] |
| box:mse:Learned-Wrong | -0.00820 | [-0.07374,+0.05949] |
| box:regret:Learned-Wrong | +0.03631 | [-0.12414,+0.22361] |
| box:mse:Learned-RadialLinear | -1.23160 | [-1.43410,-1.07967] |
| box:regret:Learned-RadialLinear | -0.24718 | [-0.35520,-0.12308] |
| box:mse:Learned-OnlineFull | -2.55177 | [-2.85109,-2.34569] |
| box:regret:Learned-OnlineFull | -0.60051 | [-0.76413,-0.43756] |
| box:mse:Learned-OnlineRidge | -2.20300 | [-2.48189,-1.99476] |
| box:regret:Learned-OnlineRidge | -0.32755 | [-0.51965,-0.13452] |
| shell:mse:Learned-Random | -0.05090 | [-0.17363,+0.05309] |
| shell:regret:Learned-Random | -0.01586 | [-0.05237,+0.01926] |
| shell:mse:Learned-Wrong | +0.00418 | [-0.18010,+0.13852] |
| shell:regret:Learned-Wrong | -0.00184 | [-0.03933,+0.04918] |
| shell:mse:Learned-RadialLinear | -0.65300 | [-0.84582,-0.45682] |
| shell:regret:Learned-RadialLinear | -0.17580 | [-0.21298,-0.14497] |
| shell:mse:Learned-OnlineFull | -0.53709 | [-0.67700,-0.44055] |
| shell:regret:Learned-OnlineFull | -0.21848 | [-0.25433,-0.18528] |
| shell:mse:Learned-OnlineRidge | -0.53853 | [-0.64003,-0.45084] |
| shell:regret:Learned-OnlineRidge | -0.20580 | [-0.24281,-0.16793] |

mismatch通过固定坐标置换破坏源目标方向对应，目标分布的旋转对称边际仍相同。失配条件不得隐藏，也不用于更换matched主验收。
没有完整A/O/F/FR搜索，不自动把条件化预测器叫作纯离线模型；下一项须单独定义在线修正及失配机制后冻结验证。
本轮耗时约0.15分钟，源拟合与分组评测使用单线程CPU进程并行，实际资源见RESOURCES.json。
