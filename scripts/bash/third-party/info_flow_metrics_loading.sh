#!/bin/bash
# §2.1 External Metrics Loading Verification
#
# Verifies that ExternalMetricsBackend correctly loads information_flow metrics.
#
# Usage:
#   ./scripts/bash/third-party/info_flow_metrics_loading.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"

WORKSPACE="${ROOT_DIR}/logs/third-party/info_flow_metrics"

echo "============================================================"
echo "External Metrics Loading Verification"
echo "============================================================"
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

# Run verification
python "${ROOT_DIR}/scripts/examples/third-party/info_flow_metrics_loading.py" \
    --workspace "${WORKSPACE}" \
    --module-path "information_flow.experiments.utils.metrics.metric_functions"

echo ""
echo "Results saved to: ${WORKSPACE}/metrics_loading_results.json"
