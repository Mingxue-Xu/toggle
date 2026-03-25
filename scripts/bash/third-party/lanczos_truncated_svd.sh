#!/bin/bash
# §1.2 Lanczos Algorithm Showcase for Truncated SVD
#
# Demonstrates Lanczos advantages for truncated SVD (k << min(m,n)).
#
# Usage:
#   ./scripts/bash/third-party/lanczos_truncated_svd.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WORKSPACE="${ROOT_DIR}/logs/third-party/lanczos_showcase"

echo "============================================================"
echo "Lanczos Algorithm Showcase for Truncated SVD"
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
python "${ROOT_DIR}/scripts/examples/third-party/lanczos_truncated_svd.py" \
    --workspace "${WORKSPACE}" \
    --sizes "512,1024,2048,4096" \
    --ranks "16,32,64,128" \
    --bench-iters 3 \
    --max-lanczos-iters 100

echo ""
echo "Results saved to: ${WORKSPACE}/lanczos_benchmark_results.json"
