#!/bin/bash
# §2.4 Layer Sensitivity Correlation
#
# Verifies info_flow metrics correlate with compression sensitivity.
#
# Usage:
#   ./scripts/bash/third-party/layer_sensitivity_correlation.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/layer_sensitivity_correlation.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
WORKSPACE="${ROOT_DIR}/logs/third-party/layer_sensitivity_correlation"

echo "============================================================"
echo "Layer Sensitivity Correlation"
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

# Run correlation analysis
python "${ROOT_DIR}/scripts/examples/third-party/layer_sensitivity_correlation.py" \
    --model "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --compression-rank 32 \
    --n-samples 32

echo ""
echo "Results saved to: ${WORKSPACE}/correlation_results.json"
