# Goldcrest

[![PyPI version](https://img.shields.io/pypi/v/goldcrest.svg)](https://pypi.org/project/goldcrest/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

LLM compression framework focusing on low-rank factorization (matrices and tensors), managing your whole experiment pipelines - **compression**, **evaluation**, **profiling** and **logging**.


### Latest

**Mar. 24th**: Published to [pypi](https://pypi.org/project/goldcrest/), renamed to `goldcrest` (a tiny bird with bright crest) to avoid naming collision. 

**Mar. 23rd**: Passed all the tests after debugging with agents (Claude Code + Codex) and PyCharm. A good IDE is pretty helpful for identifying dead code and keeping track of what is going on, especially when agents get high. Also added some feature documents.

**Mar. 19th**: Fixed the identified slop with Codex. Working with agents of different "tastes" is helpful! More documentation is coming...

**Mar. 18th**: The current codebase seems a bit slop, both human and agent (Claude Code) contributors are investigating it.


## Features

| Category | Options |
|----------|---------|
| [Compression Algorithms](docs/compression_algorithm.md) | SVD, Tucker, CP (CANDECOMP/PARAFAC), Tensor Train, weight pruning (layer removal) |
| [Rank Selection Criteria](docs/rank-selection.md) | Activation-informed algorithms like [ASVD](https://github.com/hahnyuan/ASVD4LLM) and [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM), entropies, Fisher Information, stable rank, and metrics included in [information_flow](https://github.com/OFSkean/information_flow) (external download is required) |
| [Algebra Backends](docs/algebra_backends.md) | [CoLA](https://github.com/wilson-labs/cola) (Lanczos, LOBPCG), PyTorch, [TensorLy](https://github.com/tensorly/tensorly) |
| [Target Layers](docs/layer_types.md) | MLP, attention layers (Q/K/V/O), embedding layers |
| [Evaluation](docs/evaluation.md) | [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) integration, self-defined language task evaluation metrics |
| [Profiling](docs/profiling.md)| Memory (RSS, VMS, GPU allocated/reserved, peak), latency (per-phase), subprocess isolation |
| [Logging](docs/logging.md) | CSV logging for experiments, including models, evaluations, and compression runs, with cross-experiment comparison |

## Quick Start

### 1. Install

```bash
pip install goldcrest
```

With optional dependencies:
```bash
pip install goldcrest[eval]    # lm-eval-harness integration
pip install goldcrest[cola]    # CoLA algebra backend
```

### 2. Configure

Create a YAML config file specifying your compression pipeline, this is an example:

```yaml
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
  type: lm_eval         #   lm-eval-harness
  tasks: [arc_easy]
```
More example configurations are in [config/](config/). 

### 3. Run

```bash
python scripts/examples/gpu/h100_activation_svd.py --config config/h100_activation_svd.yaml
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

There are more detailed clarification about configuration settings for [compression algorithms](docs/compression_algorithm.md), and [layer types to be compressed](docs/layer_types.md).

## Example Scripts

### CPU Examples (`scripts/examples/cpu/`)

Lightweight scripts for development and testing.

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

GPU scripts for H100 and H200 workloads:

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
| `h200_activation_svd.py` | Activation-guided SVD compression (H200) |
| `h200_entropy_svd.py` | Entropy-based rank selection (H200) |
| `h200_hybrid_mi.py` | Mutual information hybrid compression (H200) |
| `h200_full_pipeline.py` | Complete analysis + compression + evaluation (H200) |
| `h200_tucker.py` | Tucker decomposition compression (H200) |
| `h200_cp.py` | CP (CANDECOMP/PARAFAC) decomposition (H200) |
| `h200_tensor_train.py` | Tensor-Train decomposition (H200) |
| `h200_weight_pruning.py` | Weight pruning compression (H200) |
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
goldcrest/
├── config/                # Experiment configs plus base/ and profiles/
│   ├── base/
│   └── profiles/
├── docs/                  # Documentation and changelogs
├── logs/                  # Runtime logs
├── scripts/
│   ├── bash/              # Shell scripts (cpu/, gpu/, third-party/)
│   ├── examples/          # Python examples (cpu/, gpu/, third-party/)
│   ├── logs/              # Script execution logs
│   └── utils/             # Utility scripts
├── tests/                 # Pytest regression and integration suite
└── goldcrest/
    ├── config/            # Config loader
    ├── framework/         # Core framework (layers, state, IO)
    ├── orchestration/     # Pipeline orchestration
    └── plugins/
        ├── analysis/      # Metrics and layer selection
        ├── compression/   # SVD, Tucker, CP, pruning
        ├── evaluation/    # lm-eval integration
        └── models/        # Model utilities
```

## Documentation

- [Architecture](docs/architecture.md) — Plugin-based design, EventBus, and workflow system
- [Changelogs](docs/changelogs.md) — Notes on recent fixes
- [Successful Runs](docs/successful_runs.md) — H200 GPU benchmark results (7 compression strategies, third-party tests)


## Cite as
```bibtext
@software{xu_goldcrest,
  author       = {Xu, Mingxue and Mandic, Danilo},
  title        = {Goldcrest: Full-Pipeline LLM Compression Framework for Low-Rank Factorization},
  month        = mar,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19211420},
  url          = {https://doi.org/10.5281/zenodo.19211420},
}
```
