#!/usr/bin/env bash
set -euo pipefail
cd /home/skq/repos/roopf_agent/ROOPF
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec /home/skq/.conda/envs/ooneurgo/bin/python -u scripts/experiment_r15_schedule.py "${@:-all}"
