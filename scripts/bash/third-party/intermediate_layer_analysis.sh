#!/bin/bash
# §2.2 Intermediate Layer Analysis
#
# Reproduces paper's key finding that intermediate layers outperform final layers.
#
# Usage:
#   ./scripts/bash/third-party/intermediate_layer_analysis.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/intermediate_layer_analysis.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
WORKSPACE="${ROOT_DIR}/logs/third-party/intermediate_layer_analysis"

echo "============================================================"
echo "Intermediate Layer Analysis"
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

# Run analysis
python "${ROOT_DIR}/scripts/examples/third-party/intermediate_layer_analysis.py" \
    --model "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --n-samples 64 \
    --seq-length 128

echo ""
echo "Results saved to: ${WORKSPACE}/layer_analysis_results.json"
