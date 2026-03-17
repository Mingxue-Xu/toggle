#!/usr/bin/env bash
# H200 GPU Memory Profiling
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Memory Profiling"
echo "=========================================="

/usr/bin/time -v python "${ROOT}/scripts/examples/gpu/memory_profiling.py" "$@"
