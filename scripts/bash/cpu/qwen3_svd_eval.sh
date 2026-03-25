#!/usr/bin/env bash
# Run Qwen3 SVD loop over ranks from assets/qwen_14b_svd_ranks.csv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Set CUDA_VISIBLE_DEVICES to control GPU/CPU selection (empty for CPU).
python "${ROOT}/scripts/examples/cpu/loop_svd_qwen3.py" \
  --config "${ROOT}/config/profiles/eval/svd_qwen3.yaml" \
  --input-csv "${ROOT}/assets/qwen_8b_svd_ranks.csv" \
  --out-csv "${ROOT}/logs/qwen3_loop_results.csv" \
  "$@"
