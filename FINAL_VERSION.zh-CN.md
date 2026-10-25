# 最终版本说明：CEC 统一在线流程

用户于 2026 年 10 月 25 日确认，以已完成真实 API 重测的 CEC 解锁版作为最终代码。本发布固定该次测量所用的代码、权重、案例、门限和动作时序，仅更新正式发布标识及结果索引；本次发布没有新增算法改动、训练或 API 请求。

## 发布标识

- 最终分支：`release/final-20261025`
- 最终标签：`final-v1.1.0-cec-unlocked-20261025`
- 已测代码与结果提交：`b311a0ebffd4238d92b124f560892ffdc2d338ef`
- CEC 执行策略：`r13_cec_same_execution_path_as_bbob_v1`
- Agent 策略：`r10_current_pool_fusion_delayed_generation_v1`
- 发布清单：[release/FINAL_MANIFEST.json](release/FINAL_MANIFEST.json)
- 发布检查：[release/FINAL_PREFLIGHT.json](release/FINAL_PREFLIGHT.json)

CEC 最终执行入口为 `scripts/run_r13_cec_uniform.sh`，由 `scripts/experiment_r13_cec_uniform.py` 在工作进程中安装 `roopf/cec_uniform.py` 的执行策略。使用这一入口时，CEC 按与 BBOB 相同的生成激活、Residual、Agent 候选池、代理保护和最终选择路径运行。该策略明确安装在当前模型及严格消融模块，独立加载的冻结基线继续保持其既定实现。

## 方法与数据流

离线准备：36 个生成训练函数 → Anchor 预训练及冻结 → 完整候选池日志 → Residual 训练及冻结 → 同状态 22 动作分支案例。共 576 个上下文，其中 464 条用于检索、112 条用于覆盖校准；不对 LLM 微调。

在线每条轨迹先真实评价 100 个初始点，种群保留 100 点。保护竞争阶段由六个算子各生成六点，加上竞争 Anchor 和额外提议，共 38 个竞争候选；保护 Anchor 单独保留。档案拟合在线代理，冻结 Residual 提供改进支持，再结合来源可靠性、案例支持及保护规则完成选择。

LLM 在评价前 FE=220、240 决策。融合倍率在当前轮起生效五轮；额外 Anchor 温度和指定局部生成尺度从下一轮起生效四轮。默认两个基础 Anchor 点保持保护及竞争回退职责。通常每轮只对保护点和获准竞争点进行真实评价，再更新种群、档案、成功记忆和已测反馈，达到 300 FE 停止。生成、代理预测、Residual 推理及 LLM 调用均不计为目标 FE。

CEC 当前允许 Agent 按相同规则介入；动作准入仍由冻结案例和既定支持门决定，并不保证每个状态都允许非默认动作。失败按既定规则回退，不追加 API 重试。

## 最终结果采用范围

最终表按 BBOB24、CEC-style6、UAV5 分组。BBOB 与 UAV 沿用已冻结的完整测量记录；CEC 采用本轮 6 函数 × 每函数 100 轨迹 × 五条件、D10、300 FE（含初始100）的完整重测，合计 3,000 条轨迹。本次没有将三个问题组重新合并跑一轮实验，也不添加跨类别效能合计。

CEC 条件为 Full、冻结基线、移除 Anchor、移除在线生成算子、移除 Agent。胜/平/负均从 Full 角度统计；函数比较最终均值，轨迹比较配对终值：

| 比较 | 函数胜/平/负 | 配对轨迹胜/平/负 |
|---|---:|---:|
| Full vs 冻结基线 | 1/5/0 | 1/599/0 |
| Full vs 移除 Anchor | 6/0/0 | 600/0/0 |
| Full vs 移除在线生成算子 | 1/5/0 | 1/599/0 |
| Full vs 移除 Agent | 1/5/0 | 1/599/0 |

Full 执行 358 个非默认动作，但整体终值几乎未变。唯一改善为 CEC3 的一条轨迹，改善很小且其两次执行动作均为 matched38，不能归因于 LLM 主动调参。本轮未建立明显的 Agent 额外优化收益。

移除 Anchor 是移除预训练网络及其候选生成路径，以原生信赖域提议补位保护与竞争位置；移除在线生成算子是完整移除六个生成算子及额外原生提议，保留代理、Residual、Agent 和冻结案例门；移除 Agent 是移除整套 Agent 控制框架。CEC 的移除在线生成算子条件仍真实请求 API，但 1,200 个决策上下文的 allowed 均仅有默认动作，非默认动作和实际改点为零。

## 真实 API、预算与可靠性记录

CEC 正式测量发送 360 次全新 API 请求：251 次完整有效、2 次部分有效、107 次失败；2,526 条有效动作记录、1,074 条回退记录。请求、响应及实际执行决策已匹配，未复用往轮 LLM 响应。输入 token 为 5,192,189，输出 token 为 99,090；服务商返回的 credit 累计 158.28，仅 253 条响应包含该字段，不推断币种或完整账单金额。

CEC 正式逻辑 FE 为 900,000，其中原生评价 191,030、精确缓存评价 708,970；缓存校验完整请求身份与随机状态。工程检查另有 15,000 逻辑 FE、零 API，未混入正式结果。每函数各条件均保留均值、总体标准差、方差、配对胜平负、评价坐标差异和原生/缓存 FE；表格数值明细仍按算法分类，仅展示函数、均值和方差。

## 重要材料与科学证据

- [最终 Excel：论文对照、消融与均值方差](exports/当前Full_论文对照与消融_BBOB24_CEC统一流程重测_R13.xlsx)
- [CEC 完整结果报告 PDF](exports/CEC统一流程重测_R13_完整结果报告.pdf)
- [结果材料包：Excel、PDF、统计与核验记录](exports/CEC统一流程重测_R13_结果材料.zip)
- [CEC 实验目录](experiments/r13_cec_uniform/)：冻结协议、代码绑定、逐函数统计、配对轨迹、API审计、独立核验、运行硬件记录和远端读回回执。
- [CEC 发布证据目录](experiments/r13_cec_uniform/publication/)：13 个归档，保留目标值、实际坐标、原生调用、初始种群、随机流、真实 API 请求/响应与执行记录。未选候选的全量诊断保留在本机完整归档；共享重要证据可独立复核。
- BBOB/UAV 的 Full 证据：[R10](experiments/r10_current_candidate_fusion/)；移除 Anchor：[R11](experiments/r11_anchor_ablation/)；移除在线生成算子/Agent：[R12](experiments/r12_strict_component_ablations/)。其中 CEC 结果在最终表中采用 R13。
- 流程与训练材料保留在 `exports/最终版本_端到端流程_20261025/`；涉及 CEC 开关、执行入口及测量结果时，以本说明和 R13 报告为准。

Excel 的 BBOB/UAV 八个结果页保持不变；论文算法采用归档均值形成参考排名，未在本轮重新运行。BBOB20 按用户要求合入 BBOB24 展示，其跨批求和和随机符号语义仍需披露，冻结证据中的单列辅助记录继续保留。CEC 为项目定义的六个 CEC-style 目标，不等同于完整官方 CEC 竞赛协议；与论文归档的偏移一致性尚未确认。UAV 沿用恢复的代理协议。

## 环境与运行入口

使用 `ooneurgo` conda 环境。该机器解释器为 `/home/skq/.conda/envs/ooneurgo/bin/python`。在保留冻结执行数据的工作区启动或续跑 CEC 五组测量，实时输出并保存日志：

```bash
conda activate ooneurgo
bash scripts/run_r13_cec_uniform.sh all --workers 24 --progress-seconds 10
```

该入口检查已有进度及冻结身份，不能把已完成目录当作一套新的空实验。异机恢复须准备绑定的资产及配对执行数据，按既定规则恢复进度。API 凭据从本机配置读取，不随仓库提交。

只复核已发布测量与实际 API 记录，无需发起 API：

```bash
conda activate ooneurgo
python scripts/verify_r13_publication.py
```

CPU 负责求解器与随机流，五成员岭回归代理使用共享 CUDA 后端；24 个工作进程、每进程一线程、GPU 并发二、API 并发二、float32、关闭 TF32。异机的设备数值差异需单独记录。

## 发布保留

此前发布标签 `final-v1.0.0-20261025`、独立基线分支及所有冻结实验记录继续保留。此前发布说明与清单另存于 [release/final-v1.0.0-20261025/](release/final-v1.0.0-20261025/)。本发布按用户确认固定当前 CEC 解锁版，完整 API 执行不构成优化改进证明。
