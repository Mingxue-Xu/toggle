# Remote Runs Guide

Instructions for running Toggle compression scripts on remote GPU servers (H100 80GB VRAM or equivalent).

---

## Prerequisites

### Environment Setup

```bash
# Activate the Python environment
source /path/to/env/bin/activate

# Set HuggingFace token for model access
export HF_TOKEN=<your_token>

# Navigate to codebase
cd /path/to/toggle
```

### Required Dependencies

- PyTorch with CUDA support (BF16 capable)
- `transformers`, `datasets`, `lm-eval`
- `tensorly` (for Tucker, CP, TT decomposition)
- `cola-ml` (for Lanczos/LOBPCG SVD backends)
- `information_flow` (third-party metrics package)

---

## 1. GPU Runs (H100 Optimized)

Scripts location: `scripts/examples/gpu/` and `scripts/bash/gpu/`

### 1.1 Quick Start: Run All Compression Strategies

```bash
./scripts/bash/gpu/h100_run_all.sh
```

### 1.2 Individual Compression Method Runs

| Method | Command |
|--------|---------|
| Activation SVD | `./scripts/bash/gpu/h100_activation_svd.sh` |
| Tucker decomposition | `./scripts/bash/gpu/h100_tucker.sh` |
| CP decomposition | `./scripts/bash/gpu/h100_cp.sh` |
| Tensor-Train | `./scripts/bash/gpu/h100_tensor_train.sh` |
| Weight pruning | `./scripts/bash/gpu/h100_weight_pruning.sh` |
| Entropy-based SVD | `./scripts/bash/gpu/h100_entropy_svd.sh` |
| MI hybrid | `./scripts/bash/gpu/h100_hybrid_mi.sh` |

### 1.3 Full Pipeline Runs by Model Tier

```bash
# Mid-size models (8B-9B) - Quick validation
./scripts/bash/gpu/h100_full_pipeline.sh --tier mid

# Large models (14B-27B) - Main results
./scripts/bash/gpu/h100_full_pipeline.sh --tier large

# Flagship models (70B+) - Paper highlights
./scripts/bash/gpu/h100_full_pipeline.sh --tier flagship
```

### 1.4 ASVD/SVD-LLM Pipeline

Full pipeline: `CalibrationCollector → ActivationScaling → DataWhitening → PPLSensitivity → BinarySearchRank → SVD → ClosedFormUpdate → LMHarness`

```bash
./scripts/bash/gpu/asvd_svdllm_pipeline.sh
```

Or directly with Python:

```bash
python scripts/examples/gpu/asvd_svdllm_pipeline.py \
    --model meta-llama/Llama-3.1-8B \
    --calibration_samples 256 \
    --target_ratio 0.5 \
    --dtype bfloat16
```

### 1.5 Benchmark Reproducibility Runs

Generate benchmark tables for paper submission:

```bash
# Run by tier
./scripts/bash/gpu/benchmark_reproducibility.sh --tier mid
./scripts/bash/gpu/benchmark_reproducibility.sh --tier large
./scripts/bash/gpu/benchmark_reproducibility.sh --tier flagship
```

Models tested per tier:

| Tier | Models |
|------|--------|
| mid | Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B |
| large | Qwen2.5-14B, Gemma-2-27B, Mixtral-8x7B |
| flagship | Llama-3.1-70B, Qwen2.5-72B |

### 1.6 Perplexity Evaluation

```bash
./scripts/bash/gpu/perplexity_evaluation.sh
```

Evaluates on WikiText-2, C4, and PTB datasets.

### 1.7 Memory Profiling

```bash
./scripts/bash/gpu/memory_profiling.sh --model meta-llama/Llama-3.1-70B
```

Tracks GPU VRAM usage and tokens/second throughput.

### 1.8 Baseline vs Compressed Comparison

```bash
./scripts/bash/gpu/comparison_baseline_compressed.sh
```

---

## 2. Third-Party Integration Runs

Scripts location: `scripts/examples/third-party/` and `scripts/bash/third-party/`

### 2.1 Quick Start: Run All Third-Party Tests

```bash
./scripts/bash/third-party/third_party_run_all.sh
```

### 2.2 CoLA Backend Verification

#### CoLA vs Torch SVD Comparison

Compares runtime, memory, and accuracy of CoLA (Lanczos) vs PyTorch dense SVD:

```bash
./scripts/bash/third-party/cola_vs_torch_benchmark.sh
```

Or:

```bash
python scripts/examples/third-party/cola_vs_torch_benchmark.py
```

Expected results:
- CoLA 2-4x faster on matrices > 4096x4096
- Memory: CoLA < 0.7x PyTorch
- Reconstruction error: < 1% difference

#### Lanczos Truncated SVD

Demonstrates speedup for truncated SVD (k << min(m,n)):

```bash
./scripts/bash/third-party/lanczos_truncated_svd.sh
```

#### LOBPCG for Ill-Conditioned Layers

Tests LOBPCG on high condition number layers:

```bash
./scripts/bash/third-party/lobpcg_ill_conditioned.sh
```

### 2.3 Information Flow Metrics Verification

#### External Metrics Loading

Verifies `ExternalMetricsBackend` correctly loads `information_flow` metrics:

```bash
./scripts/bash/third-party/info_flow_metrics_loading.sh
```

Expected: 7+ metrics loaded (`prompt_entropy`, `dataset_entropy`, `effective_rank`, `curvature`, `infonce`, `lidar`, `dime`)

#### Intermediate Layer Analysis

Reproduces paper finding that intermediate layers (50-75% depth) outperform final layer:

```bash
./scripts/bash/third-party/intermediate_layer_analysis.sh
```

#### Metric-Guided Compression

Compares uniform vs info_flow guided rank allocation:

```bash
./scripts/bash/third-party/metric_guided_compression.sh
```

Expected: 3-4% better accuracy at same compression ratio.

#### Layer Sensitivity Correlation

Validates info_flow metrics correlate with compression sensitivity:

```bash
./scripts/bash/third-party/layer_sensitivity_correlation.sh
```

### 2.4 Combined Integration Runs

#### Full Pipeline: Info Flow + CoLA

End-to-end pipeline using both third-party libraries:

```bash
./scripts/bash/third-party/third_party_full_pipeline.sh
```

Pipeline: `CalibrationCollector → ActivationMetricsPlugin (info_flow) → LayerMetricSelector → BinarySearchRank → SVD (CoLA) → ClosedFormUpdate → LMHarness`

#### Mutual Information Hybrid Compression

MI-guided method selection (SVD vs Tucker):

```bash
./scripts/bash/third-party/mi_hybrid_compression.sh
```

---

## 3. Model Memory Requirements

| Model | FP16 Size | Fits in 80GB | Notes |
|-------|-----------|--------------|-------|
| Llama-3.1-70B | ~140GB | No | Requires multi-GPU or offloading |
| Qwen2.5-72B | ~144GB | No | Requires multi-GPU or offloading |
| Mixtral-8x7B | ~90GB | Marginal | Use gradient checkpointing |
| Gemma-2-27B | ~54GB | Yes | Comfortable headroom |
| Llama-3.1-8B / Qwen2.5-14B / Gemma-2-9B | ~16-28GB | Yes | Can run multiple experiments |

---

## 4. Directory Structure

```
scripts/
├── examples/
│   ├── cpu/              # CPU-only scripts (small models: 0.5B-2B)
│   ├── gpu/              # H100 GPU scripts (large models: 8B-70B+)
│   │   ├── benchmark_reproducibility.py
│   │   ├── asvd_svdllm_pipeline.py
│   │   ├── memory_profiling.py
│   │   ├── perplexity_evaluation.py
│   │   ├── comparison_baseline_compressed.py
│   │   ├── h100_activation_svd.py
│   │   ├── h100_tucker.py
│   │   ├── h100_cp.py
│   │   ├── h100_tensor_train.py
│   │   ├── h100_weight_pruning.py
│   │   ├── h100_entropy_svd.py
│   │   ├── h100_hybrid_mi.py
│   │   ├── h100_full_pipeline.py
│   │   └── profile_compressed.py
│   └── third-party/      # Third-party integration scripts
│       ├── cola_vs_torch_benchmark.py
│       ├── lanczos_truncated_svd.py
│       ├── lobpcg_ill_conditioned.py
│       ├── info_flow_metrics_loading.py
│       ├── intermediate_layer_analysis.py
│       ├── metric_guided_compression.py
│       ├── layer_sensitivity_correlation.py
│       ├── third_party_full_pipeline.py
│       └── mi_hybrid_compression.py
└── bash/
    ├── cpu/              # CPU bash scripts
    ├── gpu/              # H100 GPU bash scripts
    │   ├── h100_run_all.sh
    │   ├── h100_activation_svd.sh
    │   ├── h100_tucker.sh
    │   ├── h100_cp.sh
    │   ├── h100_tensor_train.sh
    │   ├── h100_weight_pruning.sh
    │   ├── h100_entropy_svd.sh
    │   ├── h100_hybrid_mi.sh
    │   ├── h100_full_pipeline.sh
    │   ├── benchmark_reproducibility.sh
    │   ├── asvd_svdllm_pipeline.sh
    │   ├── memory_profiling.sh
    │   ├── perplexity_evaluation.sh
    │   └── comparison_baseline_compressed.sh
    └── third-party/      # Third-party integration bash scripts
        ├── third_party_run_all.sh
        ├── cola_vs_torch_benchmark.sh
        ├── lanczos_truncated_svd.sh
        ├── lobpcg_ill_conditioned.sh
        ├── info_flow_metrics_loading.sh
        ├── intermediate_layer_analysis.sh
        ├── metric_guided_compression.sh
        ├── layer_sensitivity_correlation.sh
        ├── third_party_full_pipeline.sh
        └── mi_hybrid_compression.sh
```

---

## 5. Execution Priority

### P0 (Blocking for Paper)

1. Test coverage: `pytest --cov=src --cov-report=term-missing` (target 90%+)
2. Benchmark table: `benchmark_reproducibility.sh --tier mid --tier large --tier flagship`
3. Compression ratio vs accuracy plots

### P1 (High Priority)

4. ASVD/SVD-LLM full pipeline on 70B model
5. Memory profile comparisons
6. Perplexity evaluation on WikiText
7. Scale analysis across model sizes

### P2 (Paper Content)

8. Cross-method comparison
9. Multiple compression ratios
10. MoE-specific analysis (Mixtral)

---

## 6. Troubleshooting

### Out of Memory

For 70B+ models, use:

```bash
# Gradient checkpointing
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Or reduce batch size in script
--batch_size 2
```

### CoLA Not Found

```bash
pip install cola-ml
```

### Information Flow Not Found

```bash
pip install information-flow

# Or clone and install from source
git clone https://github.com/OFSkean/information_flow.git
pip install -e ./information_flow
```

---

## References

- [CoLA](https://github.com/wilson-labs/cola) - Compositional Linear Algebra
- [Information Flow](https://github.com/OFSkean/information_flow) - Layer-wise metrics
- [ASVD](https://github.com/hahnyuan/ASVD4LLM) - Activation-aware SVD
- [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM) - Truncation-aware SVD
