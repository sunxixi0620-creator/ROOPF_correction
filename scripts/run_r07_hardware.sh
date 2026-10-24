#!/usr/bin/env bash
set -euo pipefail
ROOPF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOPF_PYTHON="${ROOPF_PYTHON:-/home/skq/.conda/envs/ooneurgo/bin/python}"
if [[ ! -x "$ROOPF_PYTHON" ]]; then
    echo "找不到 ooneurgo 环境的 Python：$ROOPF_PYTHON" >&2
    exit 1
fi
cd "$ROOPF_ROOT"
mkdir -p results/hardware_logs
ROOPF_LOG="results/hardware_logs/run_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "使用环境：$ROOPF_PYTHON"
echo "实时日志：$ROOPF_LOG"
export PYTHONUNBUFFERED=1
"$ROOPF_PYTHON" -u scripts/run_r07_hardware.py "$@" 2>&1 | tee "$ROOPF_LOG"
