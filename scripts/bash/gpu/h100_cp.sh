#!/usr/bin/env bash
# Strategy 4: CP Decomposition for H100 80GB
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H100 CP Decomposition"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/h100_cp.py" \
  --config "${ROOT}/config/h100_cp.yaml" \
  "$@"
