# ROOPF：最终 Agent 方法（CEC 统一在线流程）

最终版本为 **`final-v1.1.0-cec-unlocked-20261025`**，正式分支为 **`release/final-20261025`**。CEC 采用与 BBOB 相同的在线生成、Residual、Agent 和保护选择流程，已完成六个函数、五个条件、每条件每函数一百条轨迹的真实 API 重测。

- [最终版本说明、运行入口与结果口径](FINAL_VERSION.zh-CN.md)
- [最终代码与科学资产清单](release/FINAL_MANIFEST.json)
- [最终 Excel](exports/当前Full_论文对照与消融_BBOB24_CEC统一流程重测_R13.xlsx)
- [CEC 结果报告 PDF](exports/CEC统一流程重测_R13_完整结果报告.pdf)
- [结果材料包](exports/CEC统一流程重测_R13_结果材料.zip)
- [逐函数、配对轨迹与真实 API 证据](experiments/r13_cec_uniform/)

在 `ooneurgo` 环境中使用 CEC 最终入口（实时进度及日志）：

```bash
conda activate ooneurgo
bash scripts/run_r13_cec_uniform.sh all --workers 24 --progress-seconds 10
```

该入口显式安装 CEC 统一执行策略，保持已测代码、权重、案例及门限不变。Full 对冻结基线为 1 个函数胜、5 平，配对轨迹为 1 胜、599 平；唯一改善轨迹均执行默认动作，本轮尚未证明明显的 Agent 额外收益。BBOB/UAV 使用冻结测量记录，最终 Excel 的八个相关结果页保持不变。

下面保留论文复现包的通用安装与说明；最终 Agent 方法的入口、CEC 执行策略和最新结果见上方链接。

---

# ROOPF Reproducibility Package

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
