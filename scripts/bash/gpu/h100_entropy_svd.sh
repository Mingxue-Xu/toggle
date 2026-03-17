#!/usr/bin/env bash
# Strategy 5: Entropy-Based SVD Compression for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 Entropy-Based SVD Compression"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_entropy_svd.py" \
  --config "${ROOT}/config/h100_entropy_svd.yaml" \
  "$@"
