#!/usr/bin/env bash
# Strategy: Weight Pruning for H200 GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Weight Pruning"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h200_weight_pruning.py" "$@"
