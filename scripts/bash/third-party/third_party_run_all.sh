#!/bin/bash
# Run All Third-Party Verification Tests
#
# Executes all third-party integration tests in sequence.
#
# Usage:
#   ./scripts/bash/third-party/third_party_run_all.sh [model_name]
#
# Example:
#   HF_TOKEN=... ./scripts/bash/third-party/third_party_run_all.sh Qwen/Qwen2.5-0.5B

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

MODEL="${1:-Qwen/Qwen2.5-0.5B}"

echo "============================================================"
echo "Running All Third-Party Verification Tests"
echo "============================================================"
echo "Model: ${MODEL}"
echo "============================================================"

# Activate an existing virtualenv or a repo-local .venv if available.
if [ -n "${VIRTUAL_ENV:-}" ] && [ -f "${VIRTUAL_ENV}/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "${VIRTUAL_ENV}/bin/activate"
elif [ -f "${ROOT_DIR}/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate"
fi

# Track timing
START_TIME=$(date +%s)

# §1: CoLA Backend Verification
echo ""
echo "============================================================"
echo "§1 CoLA Backend Verification"
echo "============================================================"

echo ""
echo "Running: cola_vs_torch_benchmark.sh"
"${SCRIPT_DIR}/cola_vs_torch_benchmark.sh" "${MODEL}" || echo "[WARNING] cola_vs_torch_benchmark.sh failed"

echo ""
echo "Running: lanczos_truncated_svd.sh"
"${SCRIPT_DIR}/lanczos_truncated_svd.sh" || echo "[WARNING] lanczos_truncated_svd.sh failed"

echo ""
echo "Running: lobpcg_ill_conditioned.sh"
"${SCRIPT_DIR}/lobpcg_ill_conditioned.sh" || echo "[WARNING] lobpcg_ill_conditioned.sh failed"

# §2: Information Flow Metrics Verification
echo ""
echo "============================================================"
echo "§2 Information Flow Metrics Verification"
echo "============================================================"

echo ""
echo "Running: info_flow_metrics_loading.sh"
"${SCRIPT_DIR}/info_flow_metrics_loading.sh" || echo "[WARNING] info_flow_metrics_loading.sh failed"

echo ""
echo "Running: intermediate_layer_analysis.sh"
"${SCRIPT_DIR}/intermediate_layer_analysis.sh" "${MODEL}" || echo "[WARNING] intermediate_layer_analysis.sh failed"

echo ""
echo "Running: metric_guided_compression.sh"
"${SCRIPT_DIR}/metric_guided_compression.sh" "${MODEL}" || echo "[WARNING] metric_guided_compression.sh failed"

echo ""
echo "Running: layer_sensitivity_correlation.sh"
"${SCRIPT_DIR}/layer_sensitivity_correlation.sh" "${MODEL}" || echo "[WARNING] layer_sensitivity_correlation.sh failed"

# §3: Combined Integration
echo ""
echo "============================================================"
echo "§3 Combined Integration"
echo "============================================================"

echo ""
echo "Running: third_party_full_pipeline.sh"
"${SCRIPT_DIR}/third_party_full_pipeline.sh" "${MODEL}" || echo "[WARNING] third_party_full_pipeline.sh failed"

echo ""
echo "Running: mi_hybrid_compression.sh"
"${SCRIPT_DIR}/mi_hybrid_compression.sh" "${MODEL}" || echo "[WARNING] mi_hybrid_compression.sh failed"

# Summary
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "============================================================"
echo "All Third-Party Tests Completed"
echo "============================================================"
echo "Total time: ${DURATION} seconds"
echo ""
echo "Results saved to:"
echo "  ${ROOT_DIR}/logs/third-party/"
echo ""
echo "Individual results:"
echo "  - cola_vs_torch/benchmark_results.json"
echo "  - lanczos_showcase/lanczos_benchmark_results.json"
echo "  - lobpcg_ill_conditioned/lobpcg_benchmark_results.json"
echo "  - info_flow_metrics/metrics_loading_results.json"
echo "  - intermediate_layer_analysis/layer_analysis_results.json"
echo "  - metric_guided_compression/compression_results.json"
echo "  - layer_sensitivity_correlation/correlation_results.json"
echo "  - full_pipeline/full_pipeline_report.json"
echo "  - mi_hybrid_compression/mi_hybrid_results.json"
