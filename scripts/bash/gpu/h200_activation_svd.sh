#!/usr/bin/env bash
# Strategy: Activation-Guided SVD Compression for H200 GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Activation-Guided SVD Compression"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h200_activation_svd.py" "$@"
