#!/usr/bin/env bash
# Strategy 1: Activation-Guided SVD Compression for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 Activation-Guided SVD Compression"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_activation_svd.py" \
  --config "${ROOT}/config/h100_activation_svd.yaml" \
  "$@"
