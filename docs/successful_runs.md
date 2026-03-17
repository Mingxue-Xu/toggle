# Successful Runs Summary


## 1. GPU Compression Strategies - Successful Runs

### Activation-Guided SVD

**Bash Script**: `scripts/bash/gpu/h200_activation_svd.sh`
- Calls: `scripts/examples/gpu/h200_activation_svd.py`
- Config: `config/h200_activation_svd.yaml`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Compressed params | 1,135,054,848 |
| Compression ratio | 0.1413 (85.9% reduction) |
| Layers compressed | 224 (all 7 types x 32 blocks) |
| Time | 343-348s |
| GPU memory | 12.20 GB |

---

### Tensor-Train (TT)

**Bash Script**: `scripts/bash/gpu/h200_tensor_train.sh`
- Calls: `scripts/examples/gpu/h200_tensor_train.py`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Compressed params | 4,275,704,448 |
| Compression ratio | 0.5324 |
| Layers compressed | 64 (up_proj + down_proj only) |
| Time | 1,129.45s (~19 min) |
| GPU memory | 7.96 GB |

---

### Hybrid MI

**Bash Script**: `scripts/bash/gpu/h200_hybrid_mi.sh`
- Calls: `scripts/examples/gpu/h200_hybrid_mi.py`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Compressed params | 3,896,379,392 |
| Compression ratio | 0.4852 |
| Layers compressed | 224 (all 7 types x 32 blocks) |
| Analysis time | 0.35s |
| Compression time | 338.81s |
| GPU memory | 12.20 GB |

---

### Weight Pruning

**Bash Script**: `scripts/bash/gpu/h200_weight_pruning.sh`
- Calls: `scripts/examples/gpu/h200_weight_pruning.py`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Non-zero params | 4,549,427,155 |
| Effective ratio | 0.5665 |
| Compression time | 1.21s |
| GPU memory | 15.78 GB |

---

### Tucker Decomposition (Run 3)

**Bash Script**: `scripts/bash/gpu/h200_tucker.sh`
- Calls: `scripts/examples/gpu/h200_tucker.py`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Compressed params | 7,100,755,008 |
| Compression ratio | 0.8842 |
| Layers compressed | 96 (q_proj + k_proj + gate_proj x 32 blocks) |
| Time | 2,400.15s (~40 min) |
| GPU memory | 13.29 GB |

---

### CP Decomposition (Run 3)

**Bash Script**: `scripts/bash/gpu/h200_cp.sh`
- Calls: `scripts/examples/gpu/h200_cp.py`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Compressed params | 7,795,445,760 |
| Compression ratio | 0.9708 |
| Layers compressed | 64 (q_proj + v_proj x 32 blocks) |
| Time | 303.71s |
| GPU memory | 14.55 GB |

---

### Entropy-Based SVD (Run 3)

**Bash Script**: `scripts/bash/gpu/h200_entropy_svd.sh`
- Calls: `scripts/examples/gpu/h200_entropy_svd.py`

| Metric | Value |
|--------|-------|
| Model | Llama-3.1-8B |
| Original params | 8,030,261,248 |
| Compressed params | 3,967,095,008 |
| Compression ratio | 0.4940 |
| Layers compressed | 224 (all 7 types x 32 blocks) |
| Analysis time | 9.21s |
| Compression time | 172.60s |
| GPU memory | 12.43 GB |

---

## 2. Third-Party Agent Tests - Successful Runs

### CoLA Backend Verification (Agent 1)

**Bash Script**: `scripts/bash/third-party/run_agent_1.sh`
- Orchestrates: `cola_vs_torch_benchmark.sh`, `lanczos_truncated_svd.sh`, `lobpcg_ill_conditioned.sh`, `info_flow_metrics_loading.sh`
- Models: Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B, Qwen/Qwen2.5-7B

**Status**: SUCCESS - 6/6 tests passed in 2002 seconds (~33 min)

| # | Script | Bash File | Model | Time | Key Finding |
|---|--------|-----------|-------|------|-------------|
| 1 | lanczos_truncated_svd | `lanczos_truncated_svd.sh` | (synthetic 512-4096) | 32s | CoLA Lanczos 0.13-1.45x speedup. Best on 4096x4096 matrices. |
| 2 | lobpcg_ill_conditioned | `lobpcg_ill_conditioned.sh` | (synthetic 512x512) | 5s | Torch and CoLA Lanczos converge across condition numbers 1e2-1e8. |
| 3 | info_flow_metrics_loading | `info_flow_metrics_loading.sh` | (synthetic tensors) | 2s | BasicMetricsBackend loaded 6 metrics. |
| 4 | cola_vs_torch_benchmark | `cola_vs_torch_benchmark.sh` | Qwen/Qwen2.5-0.5B | 552s | CoLA slower on small 896-dim matrices (<1x speedup). |
| 5 | cola_vs_torch_benchmark | `cola_vs_torch_benchmark.sh` | meta-llama/Llama-3.1-8B | 873s | CoLA **4-6.5x faster** on 4096-dim matrices. |
| 6 | cola_vs_torch_benchmark | `cola_vs_torch_benchmark.sh` | Qwen/Qwen2.5-7B | 538s | CoLA **3.3-4.3x faster** on 3584-dim matrices. |

---

### Information Flow Analysis (Agent 2 - Run 3 Post-Fix)

**Bash Script**: `scripts/bash/third-party/run_agent_2.sh`
- Orchestrates: `intermediate_layer_analysis.sh`, `metric_guided_compression.sh`, `layer_sensitivity_correlation.sh`
- Models: Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B, Qwen/Qwen2.5-7B

**Model**: Qwen/Qwen2.5-0.5B

| Strategy | Perplexity |
|----------|------------|
| uniform | 50.03 |
| l2_norm guided | 71.15 |
| std guided | 71.15 |

---

### Combined Integration (Agent 3 - Run 2)

**Bash Script**: `scripts/bash/third-party/run_agent_3.sh`
- Orchestrates: `third_party_full_pipeline.sh`, `mi_hybrid_compression.sh`
- Models: Qwen/Qwen2.5-0.5B, meta-llama/Llama-3.1-8B, Qwen/Qwen2.5-7B

| # | Script | Bash File | Time | Ratio | Best PPL |
|---|--------|-----------|------|-------|----------|
| 1 | third_party_full_pipeline | `third_party_full_pipeline.sh` | 7s | 0.9676 | N/A (eval skipped) |
| 2 | mi_hybrid_compression | `mi_hybrid_compression.sh` | 28s | 0.9668 | 49.22 (mi_hybrid) |

---

## 3. Third-Party Benchmarks - Successful Runs

### Metric-Guided Compression (Run 3, Post-Fix)

**Bash Script**: `scripts/bash/third-party/metric_guided_compression.sh`
- Calls: `scripts/examples/third-party/metric_guided_compression.py`
- Args: `--target-ratio 0.7 --n-samples 32`

**Model**: Qwen/Qwen2.5-0.5B | **Target ratio**: 0.7

| Strategy | Actual Ratio | Perplexity | Time |
|----------|-------------|------------|------|
| uniform | 0.9668 | 50.03 | 1.66s |
| l2_norm guided | 0.9635 | 71.15 | 1.54s |
| std guided | 0.9635 | 71.15 | 2.91s |

---

### MI Hybrid Compression (Post-Fix)

**Bash Script**: `scripts/bash/third-party/mi_hybrid_compression.sh`
- Calls: `scripts/examples/third-party/mi_hybrid_compression.py`
- Args: `--target-ratio 0.7 --mi-threshold 0.5 --n-samples 32`

**Model**: Qwen/Qwen2.5-0.5B | **Target ratio**: 0.7

| Strategy | Time | PPL | SVD layers | Tucker layers |
|----------|------|-----|-----------|--------------|
| uniform_svd | 1.94s | 49.22 | 24 | 0 |
| uniform_tucker | 2.13s | 49.23 | 0 | 24 |
| mi_hybrid | 1.16s | 49.22 | 22 | 2 |
| mi_hybrid_cola | 4.30s | 298.69 | 22 | 2 |

---

### CoLA vs Torch Benchmark

**Bash Script**: `scripts/bash/third-party/cola_vs_torch_benchmark.sh`
- Calls: `scripts/examples/third-party/cola_vs_torch_benchmark.py`
- Args: `--ranks "32,64,128" --layers "0,4,8" --warmup-iters 2 --bench-iters 5`

**Model**: Qwen/Qwen2.5-7B | **Layers**: 0, 4, 8 | **Ranks**: 32, 64, 128

| Layer 0, Rank 32 | Torch | CoLA |
|------------------|-------|------|
| Time | 0.838s | 0.203s |
| Memory | 2,995 MB | 2,755 MB |
| Error | 0.850 | 1.113 |
| **Speedup** | - | **4.13x** |

---

### Lanczos Benchmark

**Bash Script**: `scripts/bash/third-party/lanczos_truncated_svd.sh`
- Calls: `scripts/examples/third-party/lanczos_truncated_svd.py`
- Args: `--sizes "512,1024,2048,4096" --ranks "16,32,64,128" --bench-iters 3 --max-lanczos-iters 100`

| Matrix Size | Rank | CoLA Speedup | Error Ratio |
|-------------|------|-------------|-------------|
| 512 | 16-128 | 0.13-0.73x | 1.01-1.97x |
| 1024 | 16-128 | 0.30-1.13x | ~1.0x |
| 2048 | 16-128 | 0.45-1.45x | ~1.0x |
| 4096 | 16-128 | 0.80-1.45x | ~1.0x |

---

### Layer Sensitivity Correlation

**Bash Script**: `scripts/bash/third-party/layer_sensitivity_correlation.sh`
- Calls: `scripts/examples/third-party/layer_sensitivity_correlation.py`
- Args: `--compression-rank 32 --n-samples 32`

**Model**: Qwen/Qwen2.5-7B | **Rank**: 32 | **Layers**: 28

| Metric | Correlation |
|--------|-------------|
| l2_norm | -0.023 |
| l1_norm | -0.006 |
| mean | +0.146 |
| std | -0.023 |
| max_abs | -0.016 |

---

## Summary

### GPU Compression Strategies: 7/7 SUCCESS

**Master Script**: `scripts/bash/gpu/h200_run_all.sh` (runs all 7 strategies sequentially)

| Strategy | Bash Script | Ratio | Time | Status |
|----------|-------------|-------|------|--------|
| Activation SVD | `h200_activation_svd.sh` | 0.1413 | 348s | SUCCESS |
| Tensor-Train | `h200_tensor_train.sh` | 0.5324 | 1129s | SUCCESS |
| Hybrid MI | `h200_hybrid_mi.sh` | 0.4852 | 339s | SUCCESS |
| Weight Pruning | `h200_weight_pruning.sh` | 0.5665 | 1.2s | SUCCESS |
| Tucker | `h200_tucker.sh` | 0.8842 | 2400s | SUCCESS |
| CP | `h200_cp.sh` | 0.9708 | 304s | SUCCESS |
| Entropy SVD | `h200_entropy_svd.sh` | 0.4940 | 173s | SUCCESS |

### Third-Party Tests: All with finite PPL included

**Master Script**: `scripts/bash/third-party/third_party_run_all.sh` (runs all tests)

- CoLA verification (Agent 1): `run_agent_1.sh` - 6/6 tests passed
- Info flow (Agent 2): `run_agent_2.sh` - Metric-guided post-fix: All finite PPL (50.03-71.15)
- Combined integration (Agent 3): `run_agent_3.sh` - MI hybrid post-fix: All finite PPL (49.22-298.69)
