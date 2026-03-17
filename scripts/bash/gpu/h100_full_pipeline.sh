#!/usr/bin/env bash
# Strategy 8: Full Pipeline Compression for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 Full Pipeline Compression"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_full_pipeline.py" \
  --config "${ROOT}/config/h100_full_pipeline.yaml" \
  "$@"
