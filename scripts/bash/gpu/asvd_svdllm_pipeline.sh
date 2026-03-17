#!/usr/bin/env bash
# H200 GPU ASVD/SVD-LLM Pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

echo "=========================================="
echo "H200 GPU ASVD/SVD-LLM Pipeline"
echo "=========================================="

python "${ROOT}/scripts/examples/gpu/asvd_svdllm_pipeline.py" "$@"
