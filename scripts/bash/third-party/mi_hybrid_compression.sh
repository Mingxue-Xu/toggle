#!/bin/bash
# §3.2 Mutual Information Hybrid Compression
#
# Uses MI metric to guide compression method selection (SVD vs Tucker).
#
# Usage:
#   ./scripts/bash/third-party/mi_hybrid_compression.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/mi_hybrid_compression.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
WORKSPACE="${ROOT_DIR}/logs/third-party/mi_hybrid_compression"

echo "============================================================"
echo "MI-Guided Hybrid Compression"
echo "============================================================"
echo "Model: ${MODEL}"
echo "Workspace: ${WORKSPACE}"
echo "============================================================"

# Activate virtual environment if available
if [ -f "/home/mercy/act/env/bin/activate" ]; then
    source /home/mercy/act/env/bin/activate
fi

# Run hybrid compression
python "${ROOT_DIR}/scripts/examples/third-party/mi_hybrid_compression.py" \
    --model "${MODEL}" \
    --workspace "${WORKSPACE}" \
    --target-ratio 0.7 \
    --mi-threshold 0.5 \
    --n-samples 32

echo ""
echo "Results saved to: ${WORKSPACE}/mi_hybrid_results.json"
