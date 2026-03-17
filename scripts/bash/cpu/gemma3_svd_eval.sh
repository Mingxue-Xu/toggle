#!/usr/bin/env bash
# Run Gemma-3 SVD loop over ranks from assets/gemma3_ranks.csv (CPU enforced).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
# ROOT="/home/mercy/cloud/act-remote-eval"

CUDA_VISIBLE_DEVICES="" python "${ROOT}/scripts/examples/cpu/loop_svd_gemma3.py" \
  --config "${ROOT}/config/svd_gemma3.yaml" \
  --input-csv "${ROOT}/assets/gemma3_ranks.csv" \
  --out-csv "${ROOT}/logs/gemma3_loop_results.csv" \
  "$@"
