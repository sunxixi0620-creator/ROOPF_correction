#!/usr/bin/env bash
set -euo pipefail
ROOPF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOPF_PYTHON="${ROOPF_PYTHON:-/home/skq/.conda/envs/ooneurgo/bin/python}"
cd "$ROOPF_ROOT"
mkdir -p results/hardware_logs
ROOPF_LOG="results/hardware_logs/r13_cec_uniform_$(date +%Y%m%d_%H%M%S)_$$.log"
export PYTHONUNBUFFERED=1
echo "使用环境：$ROOPF_PYTHON"
echo "实时日志：$ROOPF_LOG"
"$ROOPF_PYTHON" -u scripts/experiment_r13_cec_uniform.py "$@" 2>&1 | tee "$ROOPF_LOG"
