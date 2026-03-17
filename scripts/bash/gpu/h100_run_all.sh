#!/usr/bin/env bash
# Run all H100 compression strategies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "Running All H100 Compression Strategies"
echo "============================================="

STRATEGIES=(
  "h100_activation_svd"
  "h100_weight_pruning"
  "h100_tucker"
  "h100_cp"
  "h100_entropy_svd"
  "h100_tensor_train"
  "h100_hybrid_mi"
  "h100_full_pipeline"
)

RESULTS=()

for strategy in "${STRATEGIES[@]}"; do
  echo ""
  echo ">>> Running ${strategy}..."
  echo "-------------------------------------------"

  if bash "${SCRIPT_DIR}/${strategy}.sh" "$@"; then
    RESULTS+=("${strategy}: SUCCESS")
  else
    RESULTS+=("${strategy}: FAILED")
  fi
done

echo ""
echo "============================================="
echo "Summary"
echo "============================================="
for result in "${RESULTS[@]}"; do
  echo "  $result"
done
