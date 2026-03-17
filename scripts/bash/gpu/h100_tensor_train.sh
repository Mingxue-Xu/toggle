#!/usr/bin/env bash
# Strategy 6: Tensor-Train Decomposition for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 Tensor-Train Decomposition"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_tensor_train.py" \
  --config "${ROOT}/config/h100_tensor_train.yaml" \
  "$@"
