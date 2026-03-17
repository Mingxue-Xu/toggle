#!/usr/bin/env bash
# H200 GPU Benchmark Reproducibility Runs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Benchmark Reproducibility"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/benchmark_reproducibility.py" "$@"
