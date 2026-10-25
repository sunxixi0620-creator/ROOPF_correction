# CEC统一流程重测（R13）

本轮只重测6个CEC-style函数，每函数100条轨迹，D10，300 FE（初始100）。对照条件：冻结CPU基线、Full、移除Anchor、移除在线生成算子、移除Agent。所有新条件采用与BBOB相同的非结构锁执行路径；保留函数名、目标实现、边界、权重、案例、门限、配对初始种群和随机流。冻结基线不解除结构锁，保留其原始定义。

新增执行适配独立位于roopf/cec_uniform.py，不覆盖冻结源码和旧实验结果。只有当前模型和严格消融模块的CEC结构判定在本轮工作进程中被覆盖，档案基线模块保持独立。

真实API预期360次（三个LLM条件各120次），失败不自动重试。有效默认动作仍可能不改点，案例支持不足时allowed可仅包含默认动作；不强制干预，不在观察结果后更改门限。

命令：bash scripts/run_r13_cec_uniform.sh all --workers 24 --progress-seconds 10

当前仅为预注册方案；是否完成及结果以本轮FINAL_RESULT.json和INDEPENDENT_VERIFICATION.json为准。工程检查不替代完整真实LLM实验。旧工作流文档和Excel仍描述冻结的结构锁方案，待本轮完成另行生成新结果文件，不覆盖旧证据。
