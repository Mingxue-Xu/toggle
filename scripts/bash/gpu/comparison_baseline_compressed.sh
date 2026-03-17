#!/usr/bin/env bash
# H200 GPU Baseline vs Compressed Comparison
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Baseline vs Compressed Comparison"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/comparison_baseline_compressed.py" "$@"
