#!/usr/bin/env bash
# Strategy 7: MI-Guided Hybrid Compression for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 MI-Guided Hybrid Compression"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_hybrid_mi.py" \
  --config "${ROOT}/config/h100_hybrid_mi.yaml" \
  "$@"
