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

# Activate an existing virtualenv or a repo-local .venv if available.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VIRTUAL_ENV}/bin/activate"
elif [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate"
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
