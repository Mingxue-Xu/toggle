#!/usr/bin/env bash
# H200 GPU Perplexity Evaluation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU Perplexity Evaluation"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/perplexity_evaluation.py" "$@"
