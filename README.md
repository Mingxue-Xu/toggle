# Toggle

A configuration-driven LLM compression framework using low-rank factorization and layer removal.

## Features

| Category | Options |
|----------|---------|
| **Compression Algorithms** | SVD, Tucker, CP (CANDECOMP/PARAFAC), Tensor Train, Weight Pruning |
| **Rank Selection Criteria** | Activation-informed algorithms like [ASVD](https://github.com/hahnyuan/ASVD4LLM) and [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM), the metrics included in [information_flow](https://github.com/OFSkean/information_flow), entropies, Fisher Information, stable rank |
| **Algebra Backends** | [CoLA](https://github.com/wilson-labs/cola) (Lanczos, LOBPCG), PyTorch, [TensorLy](https://github.com/tensorly/tensorly) |
| **Target Layers** | MLP projections, Attention (Q/K/V/O), Embeddings |
| **Evaluation** | [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) integration, self-defined metrics |
| **Profiling** | Memory (RSS, VMS, GPU allocated/reserved, peak), time (per-phase), subprocess isolation |

## Quick Start

### 1. Install
In a conda environment or virtualenv, install with `pip`

```bash
pip install -r requirements.txt
```

### 2. Configure

Create a YAML config file specifying your compression pipeline:

```yaml
# config/my_compression.yaml
model:
  name: Qwen/Qwen3-14B
  source: hf
  device: cuda

compression:
  svd:
    backend: cola
    cola:
      algorithm: lanczos

factorization:
  objects:
    - "model.layers[*].mlp.gate_proj"
    - "model.layers[*].mlp.up_proj"
    - "model.layers[*].self_attn.q_proj"
  func_name: svd
  rank: 64

evaluation:
  type: lm_eval
  tasks: [arc_easy]
```

### 3. Run

```bash
python scripts/examples/gpu/h100_activation_svd.py --config config/my_compression.yaml
```

## Configuration Options

| Section | Key | Description |
|---------|-----|-------------|
| `model` | `name` | HuggingFace model ID or local path |
| `model` | `device` | `cuda` or `cpu` |
| `analysis` | `type` | `activation_metrics`, `fisher_information`, `weight_metrics` |
| `compression.svd` | `backend` | `cola`, `torch`, `tensorly` |
| `factorization` | `objects` | Layer patterns to compress (supports wildcards) |
| `factorization` | `func_name` | `svd`, `tucker`, `cp`, `tensor_train` |
| `evaluation` | `type` | `lm_eval` for benchmark tasks |

## Example Scripts

### CPU Examples (`scripts/examples/cpu/`)

Lightweight scripts for small models and development/testing:

| Script | Description |
|--------|-------------|
| `svd_gemma3.py` | Single-pass SVD compression for Gemma-3 |
| `svd_qwen3.py` | Single-pass SVD compression for Qwen-3 |
| `loop_svd_gemma3.py` | Iterative SVD compression for Gemma-3 |
| `loop_svd_qwen3.py` | Iterative SVD compression for Qwen-3 |
| `asvd_svdllm_pipeline.py` | Combined ASVD and SVD-LLM pipeline |
| `benchmark_reproducibility.py` | Reproducibility benchmarks |
| `perplexity_evaluation.py` | Perplexity evaluation utilities |
| `memory_profiling.py` | Memory usage profiling |
| `profile_compressed.py` | Profile compressed model performance |
| `comparison_baseline_compressed.py` | Compare baseline vs compressed models |

### GPU Examples (`scripts/examples/gpu/`)

H100-optimized scripts with enhanced memory management for large models:

| Script | Description |
|--------|-------------|
| `h100_activation_svd.py` | Activation-guided SVD compression (H100) |
| `h100_entropy_svd.py` | Entropy-based rank selection (H100) |
| `h100_hybrid_mi.py` | Mutual information hybrid compression (H100) |
| `h100_full_pipeline.py` | Complete analysis + compression + evaluation (H100) |
| `h100_tucker.py` | Tucker decomposition compression (H100) |
| `h100_cp.py` | CP (CANDECOMP/PARAFAC) decomposition (H100) |
| `h100_tensor_train.py` | Tensor-Train decomposition (H100) |
| `h100_weight_pruning.py` | Weight pruning compression (H100) |
| `asvd_svdllm_pipeline.py` | Combined ASVD and SVD-LLM pipeline |
| `benchmark_reproducibility.py` | Reproducibility benchmarks |
| `perplexity_evaluation.py` | Perplexity evaluation utilities |
| `memory_profiling.py` | Memory usage profiling |
| `profile_compressed.py` | Profile compressed model performance |
| `comparison_baseline_compressed.py` | Compare baseline vs compressed models |

### Third-Party Integrations (`scripts/examples/third-party/`)

| Script | Description |
|--------|-------------|
| `cola_vs_torch_benchmark.py` | CoLA vs PyTorch SVD performance comparison |
| `lanczos_truncated_svd.py` | Lanczos algorithm for truncated SVD |
| `lobpcg_ill_conditioned.py` | LOBPCG for ill-conditioned matrices |
| `info_flow_metrics_loading.py` | Information flow metric computation |
| `layer_sensitivity_correlation.py` | Layer sensitivity analysis |
| `metric_guided_compression.py` | Metric-driven compression selection |
| `mi_hybrid_compression.py` | Mutual information hybrid approach |
| `third_party_full_pipeline.py` | Full pipeline with third-party backends |

## Project Structure

```
toggle/
├── config/                 # YAML configuration files
├── scripts/
│   ├── bash/              # Shell scripts (cpu/, gpu/, third-party/)
│   └── examples/          # Python examples (cpu/, gpu/, third-party/)
├── src/
│   ├── config/            # Config loader
│   ├── framework/         # Core framework (layers, state, IO)
│   ├── orchestration/     # Pipeline orchestration
│   └── plugins/
│       ├── analysis/      # Metrics and layer selection
│       ├── compression/   # SVD, Tucker, CP, pruning
│       ├── evaluation/    # lm-eval integration
│       └── models/        # Model utilities
└── tests/                 # Unit, integration, and e2e tests
```

## Documentation

- [Remote Runs Guide](docs/remote_runs.md) — Instructions for running on GPU servers
- [Successful Runs](docs/successful_runs.md) — H200 GPU benchmark results (7 compression strategies, third-party tests)

## License

MIT
