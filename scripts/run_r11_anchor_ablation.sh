#!/usr/bin/env bash
set -euo pipefail
ROOPF_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
ROOPF_PYTHON="${ROOPF_PYTHON:-/home/skq/.conda/envs/ooneurgo/bin/python}"
cd "$ROOPF_ROOT"
mkdir -p results/hardware_logs
ROOPF_LOG="results/hardware_logs/r11_anchor_$(date +%Y%m%d_%H%M%S)_$$.log"
echo "使用环境：$ROOPF_PYTHON"
echo "实时日志：$ROOPF_LOG"
export PYTHONUNBUFFERED=1
"$ROOPF_PYTHON" -u scripts/experiment_r11_anchor_ablation.py "$@" 2>&1 | tee "$ROOPF_LOG"
