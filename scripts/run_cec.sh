#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python run_roopf.py \
  --benchmark cec \
  --output-dir results/paper_cec \
  --save-trails

