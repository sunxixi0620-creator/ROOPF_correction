#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python run_roopf.py \
  --benchmark bbob \
  --output-dir results/paper_bbob \
  --save-trails

