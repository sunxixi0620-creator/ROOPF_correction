# ROOPF Reproducibility Package

**补充实验入口（2026-09-28）。** 本分支以选定的最终 10-D ROOPF 为主方法，原 checkpoint 保持冻结；20-D 和训练顺序研究使用单独保存的受控重训模型。

- [最终补充实验报告与论文结论](docs/experiments/FINAL_RESULTS_20260928.zh-CN.md)
- [Residual排序与改善幅度实验](docs/experiments/RESIDUAL_RANKING_RESULTS.zh-CN.md)、[结论](docs/experiments/RESIDUAL_RANKING_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_RANKING_REPRODUCE.md)：两个训练种子、新确认实例与实际改选诊断。
- [Residual决策与干预诊断](docs/experiments/RESIDUAL_DECISION_RESULTS.zh-CN.md)、[解释与后续决策](docs/experiments/RESIDUAL_DECISION_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_DECISION_REPRODUCE.md)：同状态反事实与首次改选轮次重放。
- [Residual训练目标对照结果](docs/experiments/RESIDUAL_TARGET_RESULTS.zh-CN.md)、[解释与后续决策](docs/experiments/RESIDUAL_TARGET_DECISIONS.zh-CN.md)和[复现步骤](docs/experiments/RESIDUAL_TARGET_REPRODUCE.md)：两个训练种子、验证早停及完整优化确认。
- [训练充分性开发实验结果](docs/experiments/TRAINING_VALIDATION_RESULTS.zh-CN.md)、[结论与后续决策](docs/experiments/TRAINING_VALIDATION_DECISIONS.zh-CN.md)和[复现说明](docs/experiments/TRAINING_VALIDATION_REPRODUCE.md)：原最终权重保持不变。
- [审稿意见逐条对应](docs/experiments/REVIEW_RESPONSE_COMPLETION.zh-CN.md)与[英文修订材料](docs/experiments/MANUSCRIPT_POSITIONING.en.md)
- [精确在线算法](docs/experiments/ALGORITHM_EXACT.zh-CN.md)、[复现步骤](docs/experiments/REPRODUCE_REMAINING.md)和[逐条件统计表](docs/experiments/remaining_tables/README.md)
- [前轮数据归档](artifacts/supplement_20260928/README.md)与[本轮完整归档](artifacts/remaining_20260928/README.md)

新实验支持相对同一 anchor 的增益，同时记录相对强基线、保护门控、概率校准和 UAV 可行性的负面或混合证据。以下为原始推理复现包说明；最终解释及补充协议以上述报告为准。

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
