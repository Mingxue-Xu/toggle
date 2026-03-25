#!/bin/bash
# §2.3 Metric-Guided Compression Quality
#
# Shows info_flow metrics improve compression quality vs uniform allocation.
#
# Usage:
#   ./scripts/bash/third-party/metric_guided_compression.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/metric_guided_compression.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
WORKSPACE="${ROOT_DIR}/logs/third-party/metric_guided_compression"

echo "============================================================"
echo "Metric-Guided Compression Quality"
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

# Run compression comparison
python "${ROOT_DIR}/scripts/examples/third-party/metric_guided_compression.py" \
    --model "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --target-ratio 0.7 \
    --n-samples 32

echo ""
echo "Results saved to: ${WORKSPACE}/compression_results.json"
