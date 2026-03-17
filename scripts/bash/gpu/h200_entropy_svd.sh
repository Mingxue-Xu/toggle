#!/usr/bin/env bash
# Strategy: Entropy-Based SVD for H200 GPU
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Entropy-Based SVD"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h200_entropy_svd.py" "$@"
