#!/usr/bin/env bash
# Run all H200 GPU compression strategies
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================="
echo "Running All H200 GPU Compression Strategies"
echo "============================================="

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. H200 GPU required."
    exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

STRATEGIES=(
  "h200_activation_svd"
  "h200_weight_pruning"
  "h200_tucker"
  "h200_cp"
  "h200_entropy_svd"
  "h200_tensor_train"
  "h200_hybrid_mi"
  "h200_full_pipeline"
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
