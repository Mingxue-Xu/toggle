#!/bin/bash
# §1.1 CoLA vs Torch SVD Backend Comparison
#
# Compares runtime, memory, and accuracy of CoLA vs PyTorch SVD backends.
#
# Usage:
#   ./scripts/bash/third-party/cola_vs_torch_benchmark.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/cola_vs_torch_benchmark.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
WORKSPACE="${ROOT_DIR}/logs/third-party/cola_vs_torch"

echo "============================================================"
echo "CoLA vs Torch SVD Backend Benchmark"
echo "============================================================"
echo "Model: ${MODEL}"
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
python "${ROOT_DIR}/scripts/examples/third-party/cola_vs_torch_benchmark.py" \
    --model "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --ranks "32,64,128" \
    --layers "0,4,8" \
    --warmup-iters 2 \
    --bench-iters 5

echo ""
echo "Results saved to: ${WORKSPACE}/benchmark_results.json"
