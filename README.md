# ROOPF Reproducibility Package

**冷启动方向扩大复核与单一覆盖保护实验已完成。** 新增3168条轨迹：原固定窗口的早期加速重复出现，但达标率保护条件仍未通过；固定10个Sobol槽位的候选未修复不足，不采用、不扫描比例。[两轮汇总与曲线](docs/revision/cold_start_direction/CONCLUSIONS.zh-CN.md) · [论文证据与缺口](docs/revision/cold_start_direction/PAPER_EVIDENCE.zh-CN.md)。原模型及原失败结论保持不变。

**冷启动失败轨迹审查已完成（新增评估0次）。** 相对空间填充的达标率差来自7个模型种子配对、4个任务实例，全部于324～358次追上目标。采样展开度存在探索性关联，尚不足以设计可靠门控；不更改300次主验收。[审查结果](docs/revision/cold_start_failure_audit/CONCLUSIONS.zh-CN.md) · [研究决策](docs/revision/cold_start_failure_audit/NEXT_DECISION.zh-CN.md)。

**10点冷启动实验已完成。** 固定辅助至40次再退出，相对纯在线平均节省15.37次评估，终局满足预设非劣条件；但相对Sobol初始化的目标达到率低3.24个百分点，联合验收未通过。保留早期效率正向证据，不启动自适应退出或追加调参。[结果](docs/revision/cold_start/CONCLUSIONS.zh-CN.md) · [结论与边界](docs/revision/cold_start/NEXT_DECISION.zh-CN.md) · [归档](artifacts/cold_start_v1/README.md)。

**固定先验＋在线GP修正实验已完成，联合门槛未通过。** 648条20D/600NFE轨迹：融合改善纯先验与固定混合，但低于使用全部观测的纯在线GP。初始预测MSE降低约28.8%，未转化为终局收益。按冻结协议停止该候选，原解锁权重保持不变。[结果](docs/revision/frozen_prior_correction/CONCLUSIONS.zh-CN.md) · [研究决策](docs/revision/frozen_prior_correction/NEXT_DECISION.zh-CN.md) · [复现归档](artifacts/frozen_prior_correction_v1/README.md)。

**跨数据集回归任务资格审查已完成。** 六项检查中源相关性、任务差异、稳定性等五项通过，但开发任务难度未通过；仅船体阻力达到难度要求，汽车油耗和红酒质量仍较容易。不启动该集合上的融合排名实验，确认候选保持封存。[结果与决策](docs/revision/regression_qualification/NEXT_DECISION.zh-CN.md) · [审查表](docs/revision/regression_qualification/CONCLUSIONS.zh-CN.md)。

**真实数据 HPO 小规模验证已完成，联合门槛未通过。** 融合相对纯在线略有改善，但没有超过固定离线排序；85.9%的轨迹在共同初始化内已达到目标，任务区分力不足。停止本原型在该数据集上的调参，不回填原ROOPF组件必要性。[结果与下一步边界](docs/revision/real_hpo_prior/NEXT_DECISION.zh-CN.md) · [完整结果](docs/revision/real_hpo_prior/CONCLUSIONS.zh-CN.md)。

**结构迁移与失配选择研究已完成。** 已有先验不只是中心偏好；共享方向正对照通过，但失配会退化。留一选择缓解了受控失配，仍未通过300NFE相对纯在线的终局门槛。保持原模型及主终点。[本轮证据与研究决策](docs/revision/STRUCTURE_ALIGNMENT_CONCLUSIONS.zh-CN.md)。

**标签审查与预测先验研究已完成。** 旧终局标签路线未通过跨随机流审查；新先验通过少样本预测与单次选点门槛，但闭环融合未通过相对纯在线与解析先验的联合验收。按协议停止局部搜索，原解锁权重保持不变。[三阶段结论与下一步边界](docs/revision/PREDICTIVE_PRIOR_CONCLUSIONS.zh-CN.md)。

**长期价值提前识别实验已完成。** 当前长期价值模型未通过预设的提前识别门槛。按协议停止此模型的特征/损失/干预率搜索，不启动大规模重训或完整搜索扩展。 [详细结果](docs/revision/long_value/CONCLUSIONS.zh-CN.md) · [归档](artifacts/long_value_v1/README.md)。

**区域方向的长期潜力诊断已完成。** 离线方向候选存在达到预设门槛的事后长期潜力，但没有通过相对所有非学习探索的增量验收。当前证据不足以支撑对该生成器继续重训。 [完整结果](docs/revision/region_potential/CONCLUSIONS.zh-CN.md) · [归档](artifacts/region_potential_v1/README.md)。

**低维 ES 匹配诊断已完成。** 200 参数低维更新没有同时通过两组方向的局部可靠性门槛。按冻结协议停止该方案，不追加完整重训或秩/步长搜索。 [详细结果](docs/revision/lowdim_es/CONCLUSIONS.zh-CN.md) · [归档](artifacts/lowdim_es_v1/README.md)。

**ES 更新可靠性审计已完成。** 两组更新没有同时通过局部可靠性门槛。当前全参数、少方向 ES 的稳定一步改进能力未获支持；不能据此认定网络结构或离线融合没有潜力。下一步优先限定一个低维更新方案，先检查更新可靠性，再决定是否完整重训。 [结果与资源实测](docs/revision/es_reliability/CONCLUSIONS.zh-CN.md) · [归档](artifacts/es_reliability_v1/README.md)。

**终局收益闭环重训已完成，未通过可行性门槛。** [结果](docs/revision/terminal_training/CONCLUSIONS.zh-CN.md)：三个种子均完成30次ES更新（约37分22秒），选中更新0/20/10。选模验证集相对O平均增益0.001323、相对重训前0.000529，低于预设0.005/0.001门槛；种子0无提升。确认集未打开，不追加训练变体，不替换原模型。选模集区间不是独立显著性证据。[后续研究决策](docs/revision/terminal_training/NEXT_DECISION.zh-CN.md) · [完整归档](artifacts/terminal_training_v1/README.md)。

**单次有益提案的分支续跑已完成。** [结果](docs/revision/branch_continuation/CONCLUSIONS.zh-CN.md)：432条原轨迹、1,296条分支，固定170/310/520 NFE位置。61次即时有益干预最终30胜31负，全部状态平均终局增益约−0.0000065，95%区间跨零且上界远低于预设0.001阈值。不支持继续优先优化即时提案排序；这是单次事后真值辅助诊断，不能推断全部融合方法不可能。[归档](artifacts/branch_continuation_v1/README.md)。

**固定提案的选择校准实验已完成。** [结果](docs/revision/selection_calibration/CONCLUSIONS.zh-CN.md)：三个769参数校准器、2,016条完整搜索轨迹。验证单步收益改善，但最终相对当前排序和纯在线O的平均差分别为−0.000456、−0.000137，调整后区间均跨零，未通过预定验收。按协议停止，校准版不替换当前版本；原解锁权重保持不变。[归档](artifacts/selection_calibration_v1/README.md)。

**冻结提案的同状态诊断已完成。** [诊断结论](docs/revision/proposal_diagnostic/CONCLUSIONS.zh-CN.md)：288批次逐项复现原轨迹，13,824条状态记录；新提案在自身状态仍有增量价值，但首次决策之后490次改进机会仅兑现139次。改进机会也随搜索推进减少，不能仅归因于训练不足或状态分布变化。原模型与成绩保持不变；下一步优先研究固定提案下的排序识别，不自动追加训练。[完整诊断归档](artifacts/proposal_diagnostic_v1/README.md)。

**互补提案机制试验已完成（2026-09-30）。** [结果与限制](docs/revision/complementary_proposal/CONCLUSIONS.zh-CN.md)：三个上下文提案模型、2,880条正式搜索轨迹。预留实例上的提案潜力通过门槛；20D/600完整搜索相对O和旧自由选择版本分别提升0.000666、0.000480，调整后区间在零以上，但均未达到预设0.005实质增益要求，按[冻结协议](docs/experiments/COMPLEMENTARY_PROPOSAL_PROTOCOL.md)停止。原最终权重保持不变；这些是同族新实例上的机制证据。[归档](artifacts/complementary_proposal_v1/README.md)。

**2026-09-30进度：1000轮训练、四组实验与自由分配对照均已完成。** [四组结论](docs/revision/complementarity/CONCLUSIONS.zh-CN.md)：F/F+R均改善同一anchor，但未证明优于纯在线O，residual独立增益未通过。[新增2592条分配实验](docs/revision/free_allocation/CONCLUSIONS.zh-CN.md)显示20维600预算下解除固定分配改善F，但仍未证明优于O；按[固定协议](docs/experiments/FREE_ALLOCATION_PROTOCOL.md)停止，未触发新实例确认。20维300预算的正向结果保留为开发证据。下面是此前已完成的修订证据。

**当前结果：已完成用户确认的四阶段修订实验。** 共完成 10,584 条主评测轨迹（开发消融 5,184 条，冻结后的外部评测 5,400 条），实际预算及数据身份核验通过。本轮统一候选的在线补充改善了同一 anchor；residual 和门槛式保护的独立收益未获支持。外部三个条件均支持相对同一 anchor 的收益，20D/600 NFE 支持相对 GP-EI 的收益，未证明稳定优于 CMA-ES。测试来源存在已披露的历史景观数据重用，不称为完全未见函数族验证。

- [最终结论与论文取舍](docs/revision/UNIFIED_CONCLUSIONS.zh-CN.md)
- [审稿问题逐条对应](docs/revision/UNIFIED_REVIEW_CLOSURE.zh-CN.md)、[英文方法、结果与限制材料](docs/revision/UNIFIED_METHOD.en.md)
- [机制消融](docs/revision/unified_execution/RESULTS.zh-CN.md)、[完整外部评测](docs/revision/unified_external/RESULTS.zh-CN.md)、[复现说明](docs/experiments/REPRODUCE_UNIFIED.md)
- [本轮外部模型、完整轨迹与源代码归档](artifacts/unified_external_v1/README.md)

原解锁最终版及其权重保持不变。本轮独立训练的研究候选按预定规则保留 `no_residual`，没有静默替换原模型，也没有将外部成绩追溯归给原权重。以下早期补充报告保留历史身份，其中尚未执行的局部建议不再作为当前计划。

**链路审查与受控修订：** [实验链路总审查](docs/revision/EXPERIMENT_CHAIN_REVIEW.zh-CN.md)发现的旧缓存和特征定义问题已纳入新入口的验收；详见[阶段一检查](docs/revision/unified_execution/STAGE1.json)和[实际算法](docs/revision/unified_execution/ALGORITHM.zh-CN.md)。原始最终权重及历史结果保持不变，受影响旧报告保留限制说明。按用户确认的主线只推进一个统一候选，暂停旧计划中的连续局部变体搜索。

**已确认的推进主线：** [修订主线与阶段验收](docs/revision/EXECUTION_DIRECTION.zh-CN.md)。限定一个新候选，依次完成链路修复、协议冻结、完整机制验收及最终验证；旧报告中的局部“下一步建议”不自动触发新实验。原最终版继续冻结保存。新实验按[统一候选预登记协议](docs/experiments/UNIFIED_REVISION_PROTOCOL.md)执行。

当前 `run_roopf.py` 默认使用显式 `final_unlocked` 配置，与补充实验共用 `roopf.factory.build_final`。复现历史锁定入口须指定 `--profile legacy_locked`；算法配置身份不等于基准实例协议，CEC移位等设置仍需按对应实验清单指定。以下旧发布说明与参考表属于原归档口径，不直接充当新候选结果。

**历史补充实验入口（2026-09-28，四阶段修订之前）。** 这些报告以选定的最终 10-D ROOPF 为主方法，原 checkpoint 保持冻结；20-D 和训练顺序研究使用单独保存的受控重训模型。当前修订结论以上方报告为准。

- [最终补充实验报告与论文结论](docs/experiments/FINAL_RESULTS_20260928.zh-CN.md)
- [两轮评分一致性诊断](docs/experiments/SCORE_CONSISTENCY_RESULTS.zh-CN.md)、[结论](docs/experiments/SCORE_CONSISTENCY_DECISIONS.zh-CN.md)和[复现说明](docs/experiments/SCORE_CONSISTENCY_REPRODUCE.md)：组件记录与保留prior的同状态对照。
- [共享状态分阶段诊断](docs/experiments/RESIDUAL_STAGE_RESULTS.zh-CN.md)、[机制结论](docs/experiments/RESIDUAL_STAGE_DECISIONS.zh-CN.md)和[复现说明](docs/experiments/RESIDUAL_STAGE_REPRODUCE.md)：短名单、二次排序和门控的局部误差分解。
- [最优候选优先训练实验](docs/experiments/RESIDUAL_TOP_RESULTS.zh-CN.md)、[结论与取舍](docs/experiments/RESIDUAL_TOP_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_TOP_REPRODUCE.md)：统一选模标准、三种目标各两个种子。
- [Residual排序与改善幅度实验](docs/experiments/RESIDUAL_RANKING_RESULTS.zh-CN.md)、[结论](docs/experiments/RESIDUAL_RANKING_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_RANKING_REPRODUCE.md)：两个训练种子、新确认实例与实际改选诊断。
- [Residual决策与干预诊断](docs/experiments/RESIDUAL_DECISION_RESULTS.zh-CN.md)、[解释与后续决策](docs/experiments/RESIDUAL_DECISION_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_DECISION_REPRODUCE.md)：同状态反事实与首次改选轮次重放。
- [Residual训练目标对照结果](docs/experiments/RESIDUAL_TARGET_RESULTS.zh-CN.md)、[解释与后续决策](docs/experiments/RESIDUAL_TARGET_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_TARGET_REPRODUCE.md)：两个训练种子、验证早停及完整优化确认。
- [训练充分性开发实验结果](docs/experiments/TRAINING_VALIDATION_RESULTS.zh-CN.md)、[结论与后续决策](docs/experiments/TRAINING_VALIDATION_DECISIONS.zh-CN.md)和[复现说明](docs/experiments/TRAINING_VALIDATION_REPRODUCE.md)：原最终权重保持不变。
- [审稿意见逐条对应](docs/experiments/REVIEW_RESPONSE_COMPLETION.zh-CN.md)与[英文修订材料](docs/experiments/MANUSCRIPT_POSITIONING.en.md)
- [精确在线算法](docs/experiments/ALGORITHM_EXACT.zh-CN.md)、[复现步骤](docs/experiments/REPRODUCE_REMAINING.md)和[逐条件统计表](docs/experiments/remaining_tables/README.md)
- [前轮数据归档](artifacts/supplement_20260928/README.md)与[本轮完整归档](artifacts/remaining_20260928/README.md)

历史补充实验记录了相对同一 anchor 的增益，以及相对强基线、保护门控、概率校准和 UAV 可行性的负面或混合证据。以下为原始推理复现包说明；本轮统一候选的最终解释及补充协议以页首链接为准。

This anonymous artifact contains the implementation and frozen artifacts used
to reproduce the 10-dimensional BBOB and CEC-style results reported for
**ROOPF (Reliable Offline-Online Proxy Fusion)**.

The package is intentionally self-contained for evaluation. It includes the
ROOPF optimizer, the frozen anchor-policy and residual-selector checkpoints,
the fixed BBOB transformations, the six CEC-style structured objectives, and
the archived result tables used for verification. The large offline rollout
logs are not required to reproduce the evaluation results and are therefore
not included.

## 1. Package layout

```text
ROOPF/
|-- checkpoints/
|   |-- anchor_policy_d10.pt
|   `-- residual_selector_generated36_d10.pt
|-- configs/
|   `-- paper_protocol.json
|-- data/
|   `-- bbob_offsets_d10.pkl
|-- reference_results/
|   |-- roopf_bbob_d10_nfe300.csv
|   `-- roopf_cec_style_d10_nfe300.csv
|-- roopf/
|   |-- model.py
|   |-- anchor_backbone.py
|   `-- benchmarks/
|       |-- bbobfunctions.py
|       |-- cecfunctions.py
|       `-- utils.py
|-- scripts/
|   |-- compare_results.py
|   |-- run_bbob.sh
|   |-- run_cec.sh
|   `-- verify_installation.py
|-- MANIFEST.sha256
|-- requirements.txt
|-- requirements-tested.txt
`-- run_roopf.py
```

## 2. Environment

Python 3.10 or newer is recommended. A CUDA-capable GPU substantially reduces
runtime, but the released code also runs on CPU. ROOPF selects CUDA
automatically when PyTorch reports that it is available.

Create an isolated environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the generic PyTorch wheel is not suitable for the local CUDA installation,
install the matching PyTorch build first and then install the remaining
requirements. `requirements-tested.txt` records the versions used for the
final package validation; it is provided for reference rather than as a strict
platform-independent lock file.

## 3. Verify the package

Run the static integrity and import checks before launching experiments:

```bash
python scripts/verify_installation.py
```

Then run one short end-to-end test for each benchmark group:

```bash
python run_roopf.py --benchmark bbob --quick --output-dir results/smoke_bbob
python run_roopf.py --benchmark cec  --quick --output-dir results/smoke_cec
```

The smoke tests use function 1, one trajectory, and 120 true objective
evaluations. They test the complete execution path but are not the paper
protocol.

## 4. Reproduce the BBOB and CEC-style results

The frozen paper configuration is:

- dimension: 10;
- population size: 100;
- true objective evaluation budget: 300;
- anchor candidates per decision: at most 2;
- runtime operator bank: 6 operators with 6 candidates each;
- portfolio size: 36;
- online proxy ensemble size: 5;
- residual weight: 0.008;
- random seed: 20260630;
- aggregation: 10 repeat batches, each with 10 independently initialized
  trajectories, for 100 final values per function.

Reproduce all 24 BBOB functions:

```bash
python run_roopf.py \
  --benchmark bbob \
  --output-dir results/paper_bbob \
  --save-trails
```

Reproduce all six CEC-style structured functions:

```bash
python run_roopf.py \
  --benchmark cec \
  --output-dir results/paper_cec \
  --save-trails
```

The convenience scripts run the same commands:

```bash
bash scripts/run_bbob.sh
bash scripts/run_cec.sh
```

To run both groups sequentially without saving trajectories:

```bash
python run_roopf.py --benchmark all --output-dir results/paper_all
```

A function subset can be selected with `--functions`, for example:

```bash
python run_roopf.py --benchmark bbob --functions 1,5,10 --output-dir results/subset
python run_roopf.py --benchmark cec  --functions 1,3,6  --output-dir results/subset
```

## 5. Outputs and result checking

Every run writes:

- one CSV containing the mean, population standard deviation, evaluation
  count, and protocol fields for each function;
- `run_metadata.json` containing the arguments, software versions, device,
  and SHA-256 hashes of the frozen artifacts;
- optional trajectory files when `--save-trails` is enabled.

Compare a reproduced table with the archived reference table:

```bash
python scripts/compare_results.py \
  --reference reference_results/roopf_bbob_d10_nfe300.csv \
  --candidate results/paper_bbob/roopf_bbob_d10_nfe300.csv

python scripts/compare_results.py \
  --reference reference_results/roopf_cec_style_d10_nfe300.csv \
  --candidate results/paper_cec/roopf_cec_style_d10_nfe300.csv
```

The comparison utility reports per-function relative differences and a
standard-error diagnostic based on the two sets of trajectory-level standard
deviations. A row is marked `consistent` when a difference larger than the
practical relative tolerance remains within the default 1.96 standard-error
threshold; `improved` and `review` mark statistically clear changes in the
favorable and unfavorable directions, respectively. Exact floating-point
identity is not expected across different PyTorch, CUDA, GPU, or CPU
configurations. The fixed seed, checkpoints, BBOB transformations, and
protocol are recorded so that any remaining platform variation is auditable.

## 6. Benchmark scope

`BBOB` denotes the 24 noiseless continuous functions used in the paper with
the fixed 10-D transformations distributed in `data/bbob_offsets_d10.pkl`.

`CEC-style` denotes the six structured analytic proxy objectives used in the
paper. They are included in `roopf/benchmarks/cecfunctions.py`. This label does
not claim that the six functions constitute an entire official CEC competition
suite.

## 7. Frozen model artifacts

The released anchor policy and residual selector are fixed during benchmark
evaluation. Both are the paper artifacts trained from the audited suite of 36
generated training functions. The online proxy is not pretrained: it is fitted
from the evaluated archive at each optimization iteration. The runtime
operator bank also has no separately loaded checkpoint.

The expected artifact hashes are recorded in `configs/paper_protocol.json` and
`MANIFEST.sha256`. Do not replace the checkpoints or BBOB offset file when
attempting to reproduce the archived tables.

## 8. Runtime notes

- Full reproduction is compute-intensive because it evaluates 30 functions,
  with 100 trajectories and 300 true evaluations per function.
- CPU execution is supported but can be considerably slower than GPU
  execution.
- To force CPU execution, hide CUDA devices before starting Python, for
  example: `CUDA_VISIBLE_DEVICES='' python run_roopf.py ...`.
- Run BBOB and CEC-style experiments in separate processes when GPU memory is
  limited.
