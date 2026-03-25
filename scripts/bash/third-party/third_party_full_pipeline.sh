#!/bin/bash
# §3.1 Full Pipeline: Info Flow + CoLA Integration
#
# End-to-end demonstration of both third-party integrations.
#
# Usage:
#   ./scripts/bash/third-party/third_party_full_pipeline.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/third_party_full_pipeline.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
WORKSPACE="${ROOT_DIR}/logs/third-party/full_pipeline"

echo "============================================================"
echo "Full Third-Party Integration Pipeline"
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

# Run full pipeline
python "${ROOT_DIR}/scripts/examples/third-party/third_party_full_pipeline.py" \
    --model "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --target-ratio 0.7 \
    --n-samples 64 \
    --svd-backend "cola" \
    --eval-tasks "hellaswag,arc_easy" \
    --eval-limit 100

echo ""
echo "Results saved to: ${WORKSPACE}/full_pipeline_report.json"
