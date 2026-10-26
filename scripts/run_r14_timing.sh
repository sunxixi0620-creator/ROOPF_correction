#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec /home/skq/.conda/envs/ooneurgo/bin/python -u scripts/experiment_r14_timing.py "$@"
