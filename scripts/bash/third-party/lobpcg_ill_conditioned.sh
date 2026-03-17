#!/bin/bash
# §1.3 LOBPCG for Ill-Conditioned Layers
#
# Shows LOBPCG superiority for problematic layers with high condition number.
#
# Usage:
#   ./scripts/bash/third-party/lobpcg_ill_conditioned.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WORKSPACE="${ROOT_DIR}/logs/third-party/lobpcg_ill_conditioned"

echo "============================================================"
echo "LOBPCG for Ill-Conditioned Matrices"
echo "============================================================"
echo "Workspace: ${WORKSPACE}"
echo "============================================================"

# Activate virtual environment if available
if [ -f "/home/mercy/act/env/bin/activate" ]; then
    source /home/mercy/act/env/bin/activate
fi

# Run benchmark
python "${ROOT_DIR}/scripts/examples/third-party/lobpcg_ill_conditioned.py" \
    --workspace "${WORKSPACE}" \
    --size 512 \
    --rank 32 \
    --condition-numbers "1e2,1e4,1e6,1e8" \
    --max-iters 200

echo ""
echo "Results saved to: ${WORKSPACE}/lobpcg_benchmark_results.json"
