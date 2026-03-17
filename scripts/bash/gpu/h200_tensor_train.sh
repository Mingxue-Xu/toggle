#!/usr/bin/env bash
# Strategy: Tensor-Train Decomposition for H200 GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Tensor-Train Decomposition"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h200_tensor_train.py" "$@"
