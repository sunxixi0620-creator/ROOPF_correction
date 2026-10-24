# R07 CPU/GPU 协同运行

使用 `ooneurgo` conda 环境。已在该环境确认 PyTorch 2.11.0+cu128、CUDA 12.8 和 RTX 5060 Ti 可用。

```bash
conda activate ooneurgo
cd /home/skq/repos/roopf_agent/ROOPF
bash scripts/run_r07_hardware.sh all
```

这是完整流程：36 个生成函数 × 4 种子的离线采集、案例库构建、六个对照臂各 35 个函数 × 100 条轨迹的实验、API 执行核验、统计报告和并行归档。维度仍为 10，每条轨迹仍为 300 次评价（含初始 100 次）。没有使用 quick 参数或缩减正式协议。

本项目加载冻结的 Anchor 和 Residual 权重；离线采集构建的是 Agent 案例库，不是从零训练这两个网络。

## 资源分配

- 自动根据 CPU affinity 和可用内存选择进程数，最多 24 个；每进程一个计算线程，防止进程和 BLAS 线程相乘造成过量调度。
- 单轨迹离线任务使用 CPU。工程测量显示这些小任务传输到 GPU 更慢，因此不强制迁移。
- 含 Agent、批大小至少为 2 的在线代理拟合与预测通过独立 CUDA 进程执行，默认两个 CUDA stream。集合成员和同一请求内的轨迹批量求解。
- 原版冻结回放和 Agent-off 始终使用历史 CPU 路径；目标评价、候选生成和随机采样也留在 CPU，保留原配对种群和求解器随机流。
- 不合并不同任务的代理请求，避免队列顺序改变浮点结果。GPU 错误会使任务失败，不会悄悄切换计算设备。
- 结果的 xz 归档使用并行线程；每个函数写独立文件。

若要明确指定本机使用 24 个 CPU 工作进程：

```bash
bash scripts/run_r07_hardware.sh all --workers 24
```

可用 `--gpu-concurrency 1` 或 `2` 指定 GPU stream 数。工作进程数、设备、环境版本与代码都绑定到该次运行；同目录续跑时必须一致。变更配置需提供新目录，例如：

```bash
bash scripts/run_r07_hardware.sh all --workers 16 \
  --experiment-dir experiments/r07_hardware_accelerated_w16
```

入口固定使用 `/home/skq/.conda/envs/ooneurgo/bin/python`，包括所有 Python 子进程。若同名环境安装位置不同，可设置 `ROOPF_PYTHON` 指向该环境的 Python；启动器仍会检查环境名。

## 进度与输出

每 5 秒输出阶段、已完成任务数、百分比、完成部分任务后的 ETA、CPU 核心使用量、进程树 RSS、GPU 利用率及显存、已完成 GPU 代理请求数。还保留原有批次与函数完成日志。CPU 采集、基线回放、API 等待阶段的 GPU 利用率可能较低；资源占用并不等于有效加速。

默认新结果目录是 `experiments/r07_hardware_accelerated_v1/`。终端完整日志在 `results/hardware_logs/`；阶段及资源记录写入实验目录的 `PROGRESS.jsonl`，设备记录、运行设置和每次 GPU 服务统计也单独保存。不会写入旧实验结果目录。

相同命令可继续已完成的采集任务、函数和批次。冻结绑定不一致会拒绝继续，不应删除绑定文件强行续跑。失败记录保存在 `FAILURE_*.json`。

也可以按顺序分阶段执行，其他参数须与首次运行保持一致：

```bash
bash scripts/run_r07_hardware.sh collect
bash scripts/run_r07_hardware.sh memory
bash scripts/run_r07_hardware.sh run
bash scripts/run_r07_hardware.sh report
```

## 工程检查与实验边界

`experiments/r07_hardware_checks_v4/` 记录了以下工程检查：CPU/GPU 代理预测误差、16 个并发请求的逐位复现、CPU RNG 不变、冻结基线一个完整批次（10 条轨迹、各 300 次评价）的精确回放、两个 GPU 离线动作分支的同前缀检查，以及两个真实离线采集任务通过 spawn 多进程入口完成。检查没有调用 LLM API，不能替代完整验证。

当时的单请求稳态测量（包含 GPU IPC，不含启动）为：

| 批大小 | CPU | GPU + IPC | CPU 时间 / GPU 时间 |
|---|---:|---:|---:|
| 1 | 0.755 ms | 2.032 ms | 0.37 |
| 10 | 6.896 ms | 2.691 ms | 2.56 |

这些是代理模块的局部计时，不是整套实验加速比，也不保证高并发时的整体吞吐量。实际速度还受基线回放、日志、目标评价与 API 等待影响。

GPU float32 结果不与历史 CPU 结果逐位等同，可能改变后续候选选择；GPU 数值版本作为独立运行保存，不能当作旧 CPU 实验的直接续跑。初次跨请求合批检查失败的代码和记录保存在 `r07_hardware_checks_v1/`，没有删除。

正式报告前，新增核验要求有效 API 响应与执行的配置及生成上下文相符，并报告失败回退与提供方原始账单字段；不推断币种。原报告继续提供逐函数均值、总体标准差、配对胜平负、changed_points、原生/缓存评价与 token 统计。BBOB20 排除在主效果汇总之外，CEC 结构锁保持。

完整 35 函数 × 100 轨迹的真实 LLM 实验由上面的 `all` 命令启动。工程检查通过不表示该实验已完成，也不表示优化效果改善。没有自动提交、推送或发布代码及结果。
