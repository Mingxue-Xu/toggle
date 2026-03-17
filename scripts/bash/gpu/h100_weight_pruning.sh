#!/usr/bin/env bash
# Strategy 2: Weight-Metrics-Guided Block Pruning for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 Weight-Guided Block Pruning"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_weight_pruning.py" \
  --config "${ROOT}/config/h100_weight_pruning.yaml" \
  "$@"
