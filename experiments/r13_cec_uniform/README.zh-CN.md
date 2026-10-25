# CEC统一流程重测（R13）

本轮只重测6个CEC-style函数，每函数100条轨迹，D10，300 FE（初始100）。对照条件：冻结CPU基线、Full、移除Anchor、移除在线生成算子、移除Agent。所有新条件采用与BBOB相同的非结构锁执行路径；保留函数名、目标实现、边界、权重、案例、门限、配对初始种群和随机流。冻结基线不解除结构锁，保留其原始定义。

新增执行适配独立位于roopf/cec_uniform.py，不覆盖冻结源码和旧实验结果。只有当前模型和严格消融模块的CEC结构判定在本轮工作进程中被覆盖，档案基线模块保持独立。

真实API预期360次（三个LLM条件各120次），失败不自动重试。有效默认动作仍可能不改点，案例支持不足时allowed可仅包含默认动作；不强制干预，不在观察结果后更改门限。

命令：bash scripts/run_r13_cec_uniform.sh all --workers 24 --progress-seconds 10

本轮已完成全部3000条轨迹、360次真实API请求，FINAL_RESULT.json与INDEPENDENT_VERIFICATION.json核验通过。Full对冻结基线为1个函数胜、5平，轨迹1胜599平；唯一改善轨迹的两次动作均为matched38，不能归因于LLM主动调参。358个非默认Full动作未带来明显终值整体增益。工程检查不替代正式测量，单独保存。新Excel与结果报告位于exports，旧工作流文档和旧Excel保持冻结。

正式测量逻辑FE为900,000，其中原生191,030、精确缓存708,970；工程检查另有15,000逻辑FE、0次API，未并入正式测量计数。所有正式API请求均为新请求，未复用旧轮LLM响应。

publication/保存完整目标值、实际坐标、原生目标调用、初始种群与随机流、所有真实API请求及响应；指定请求轮保留完整决策记录，其余轮保留执行点、动作、时序及完整记录哈希。完整未选候选预测诊断和全量归档保留本地results/及raw_full_archives/。使用scripts/verify_r13_publication.py可仅凭发布文件复核，无需调用API。

远端基线：baseline/r13-cec-uniform-20261025；实验分支：experiment/r13-cec-uniform-20261025。
