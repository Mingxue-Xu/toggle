#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "env/bin/python3" ]]; then
    PYTHON_BIN="env/bin/python3"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "python3 not found; create a venv at env/ or set PYTHON_BIN." >&2
    exit 1
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set; Gemma model download may fail." >&2
fi

# Allow overriding CUDA linalg backend and fallback; defaults stay CUDA's choice unless set.
TORCH_LINALG_PREFERRED="${TORCH_LINALG_PREFERRED:-}"
TORCH_LINALG_FALLBACK="${TORCH_LINALG_FALLBACK:-magma}"

OUT_DIR="$ROOT/../logs/profiling"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

COMMANDS=(
 "scripts/examples/cpu/svd_gemma3.py --cola no"
  "scripts/examples/cpu/svd_gemma3.py --cola yes"
  "scripts/examples/cpu/profile_compressed.py --config config/svd_gemma3.yaml --cola no"
  "scripts/examples/cpu/profile_compressed.py --config config/svd_gemma3.yaml --cola yes"
)
NAMES=(
  "svd_gemma3_no_cola"
  "svd_gemma3_cola"
  "svd_gemma3_compressed_no_cola"
  "svd_gemma3_compressed_cola"
)

summary_line() {
  local key="$1"
  local log="$2"
  awk -F': ' -v k="$key" '$1 == k {print $2}' "$log" | head -n 1
}

for i in "${!COMMANDS[@]}"; do
  cmd="${COMMANDS[$i]}"
  name="${NAMES[$i]}"
  log="${OUT_DIR}/${name}_${STAMP}.log"
  echo "Running ${cmd} ..."
  TORCH_LINALG_PREFERRED="$TORCH_LINALG_PREFERRED" \
  TORCH_LINALG_FALLBACK="$TORCH_LINALG_FALLBACK" \
  /usr/bin/time -v "$PYTHON_BIN" $cmd 2>&1 | tee "$log"
done

echo ""
echo "== Runtime Summary =="
for i in "${!COMMANDS[@]}"; do
  name="${NAMES[$i]}"
  log="${OUT_DIR}/${name}_${STAMP}.log"
  elapsed="$(awk -F': ' '/Elapsed \(wall clock\) time/ {print $2}' "$log" | head -n 1)"
  max_rss="$(awk -F': ' '/Maximum resident set size/ {print $2}' "$log" | head -n 1)"
  total_s="$(summary_line "total" "$log")"
  inference_s="$(summary_line "inference" "$log")"
  base_inf_s="$(summary_line "baseline_inference" "$log")"
  comp_inf_s="$(summary_line "compressed_inference" "$log")"
  input_bs="$(summary_line "input_batch_size" "$log")"
  input_seq="$(summary_line "input_seq_len" "$log")"
  eval_only=false
  if [[ -n "$inference_s" ]]; then
    base_inf_s="$inference_s"
    comp_inf_s="$inference_s"
    eval_only=true
  fi
  echo "${name}:"
  echo "  elapsed_wall: ${elapsed:-n/a}"
  echo "  max_rss_kb: ${max_rss:-n/a}"
  echo "  timing_total_s: ${total_s:-n/a}"
  echo "  memory_input_batch_size: ${input_bs:-n/a}"
  echo "  memory_input_seq_len: ${input_seq:-n/a}"
  if [[ "$eval_only" == true ]]; then
    echo "  timing_inference_s: ${inference_s:-n/a}"
  else
    echo "  timing_baseline_inference_s: ${base_inf_s:-n/a}"
    echo "  timing_compressed_inference_s: ${comp_inf_s:-n/a}"
  fi
done
