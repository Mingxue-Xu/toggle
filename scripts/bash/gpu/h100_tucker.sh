#!/usr/bin/env bash
# Strategy 3: Tucker Decomposition for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 Tucker Decomposition"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_tucker.py" \
  --config "${ROOT}/config/h100_tucker.yaml" \
  "$@"
