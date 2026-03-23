# Compression Methods Guide

This document covers the four low-rank compression algorithms available through
`ModelConsolidator`, plus block pruning via `PruningPlugin`, and how to set the
relevant hyperparameters.

For low-rank methods, the standard surgery path replaces targeted `nn.Linear`
modules with `FactorLinear` layers that store factorized weights. Embeddings are
narrower: only token-level/vector-granularity compression is supported, which
rebuilds the embedding as `FactorEmbedding` with one `FactorLayer` per token.
Matrix-level embedding surgery is unsupported. At inference time
`FactorLinear.forward` contracts factors into a full weight matrix, while
`FactorEmbedding.forward` contracts only the requested token rows. For details
see [Compression for Different Layer Types](layer_types.md).

## Quick Reference

| Method | `compression_method` | Key hyperparameter | Best for |
|---|---|---|---|
| SVD | `"svd"` | `rank` (int) | 2D weight matrices, fine-grained rank control |
| Tucker | `"tucker"` | `tucker_ranks` (list of ints) | 2D/higher-order tensors, per-mode rank control |
| CP | `"cp"` | `cp_rank` (int) | Higher-order tensors, maximum compression |
| Tensor Train | `"tensor_train"` | `tensor_ranks` (list of ints) | Higher-order tensors, scalable to many modes |
| Block Pruning | -- | `num_deleted_block` (int) | Removing entire transformer blocks |

---

## Configuration example

A profile-driven pipeline config shape:

```yaml
model:
  name: google/gemma-3-270m-it
  source: hf
  device: auto
  trust_remote_code: true

runtime:
  use_cola_default: true

compression:
  svd:
    backend: cola
    cola:
      algorithm: lanczos

factorization:
  objects:                            # target modules — wildcards expand to all layers
    - "model.layers[*].self_attn.q_proj"
    - "model.layers[*].self_attn.k_proj"
    - "model.layers[*].self_attn.v_proj"
    - "model.layers[*].self_attn.o_proj"
    - "model.layers[*].mlp.gate_proj"
    - "model.layers[*].mlp.up_proj"
    - "model.layers[*].mlp.down_proj"
  func_name: "svd"                    # or "tucker", "cp", "tensor_train"
  rank: 16

evaluation:
  type: lm_eval
  tasks: [wikitext, arc_easy]
  limit: 2048
  batch_size: 32
  device: cuda

memory_inference:                     # optional profiling with random inputs
  enabled: true
  isolate_subprocess: true
  device: auto
  repeats: 1
  input:
    kind: ids
    batch_size: 1
    seq_len: 128
```

Base defaults are in `config/base/default.yaml`. Model-specific profiles live under `config/profiles/` (e.g., `eval/svd_qwen3.yaml` for Qwen3 evaluation, `profile/svd_gemma3.yaml` for Gemma3 with memory profiling).

Important: this profile format is consumed by the example runners under
`scripts/examples/...`, not passed verbatim to `ModelConsolidator`. Those
scripts map:

- `factorization.func_name` -> `compression_method`
- `factorization.objects` -> `target_modules`
- `factorization.rank` -> `rank`
- `compression.svd.backend_when_*` -> `svd_backend` / `svd_backend_config`

`ModelConsolidator` itself does not interpret `backend_when_cola` or
`backend_when_no_cola`; direct API use should pass `svd_backend` and optionally
`svd_backend_config`.

## Common setup

All low-rank methods share the same `ModelConsolidator` (`consolidator.py:27`) workflow:

```python
from src.framework.context import PipelineContext
from src.plugins.compression.consolidator import ModelConsolidator

ctx = PipelineContext(config={}, workspace_dir=tmp_path)
ctx.state.model = model
ctx.state.tokenizer = tokenizer

consolidator = ModelConsolidator(
    compression_method="svd",                      # or "tucker", "cp", "tensor_train"
    target_modules=["model.layers[0].mlp.gate_proj"],
    rank=32,                                       # method-specific params as kwargs
    svd_backend="torch",
)
consolidator.initialize(ctx)
result = consolidator.compress_model_with_surgery(model)
```

`compress_model_with_surgery` (`consolidator.py:917`) returns a result with `result.parameters["layers_replaced"]` (list of replaced layer paths) and `result.parameters["surgery_stats"]` (per-layer original/compressed param counts).

Target modules support wildcards: `"model.layers[*].mlp.gate_proj"` expands to all 18 layers.

For `nn.Linear`, the intended path is the default `granularity="matrix"`. For
embeddings, call `compress_model_with_surgery(..., granularity="vector",
tensor_size=[...])`; matrix-level embedding surgery is unsupported.

## 1. SVD

Truncated SVD produces 3 factors: U, diag(S), Vt. Lower rank = more compression.

```python
# From test_e2e_compression_algorithms.py:63-68
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=[
        "model.layers[0].mlp.gate_proj",       # (2048, 640)
        "model.layers[0].mlp.down_proj",       # (640, 2048)
        "model.layers[0].self_attn.q_proj",    # (1024, 640)
        "model.layers[0].self_attn.v_proj",    # (256, 640)
    ],
    rank=32,
    svd_backend="torch",
)
```

After surgery, replaced layers are `FactorLinear` with 3 factors:

```python
# From test_e2e_compression_algorithms.py:115-119
replaced = model.model.layers[1].self_attn.q_proj
assert isinstance(replaced, FactorLinear)
factors = list(replaced.weight.factors)
assert len(factors) == 3   # U, diag(S), Vt
```

Rank controls compression: lower rank = fewer params. Verified by comparing ranks 16, 64, 128 on the same layer (test line 211-257).

## 2. Tucker

Tucker decomposition with per-mode rank control. Two ranks for 2D weight matrices.

```python
# From test_e2e_compression_algorithms.py:277-283
consolidator = ModelConsolidator(
    compression_method="tucker",
    target_modules=["model.layers[0].mlp.gate_proj"],
    tucker_ranks=[32, 32],
    device="cpu",
    backend="pytorch",
)
```

Surgery stats record per-layer compression:

```python
# From test_e2e_compression_algorithms.py:367-368
stats = result.parameters["surgery_stats"]["model.layers[1].mlp.down_proj"]
assert stats["compressed_params"] < stats["original_params"]
```

## 3. CP (CANDECOMP/PARAFAC)

Single-rank CP decomposition. Works on attention and MLP projections.

```python
# From test_e2e_compression_algorithms.py:397-402
consolidator = ModelConsolidator(
    compression_method="cp",
    target_modules=["model.layers[0].self_attn.q_proj"],
    cp_rank=16,
    backend="pytorch",
)
```

Lower `cp_rank` = fewer params. Verified by comparing ranks 4 vs 16 (test line 462-504).

## 4. Tensor Train

TT decomposition uses explicit `tensor_ranks` for tensors that are already
higher-order. For 2D weight matrices, the current implementation auto-reshapes
the matrix to 3D internally and auto-derives the effective working ranks from
the reshaped dimensions instead of honoring the supplied list directly.

```python
# Constructor input used in test_e2e_compression_algorithms.py:524-529
consolidator = ModelConsolidator(
    compression_method="tensor_train",
    target_modules=["model.layers[0].mlp.gate_proj"],
    tensor_ranks=[1, 4, 1],
    backend="pytorch",
)
```

Important: for `gate_proj` `(2048, 640)`, the implementation reshapes to
`(1024, 2, 640)` and derives working ranks `[1, 4, 2, 1]`, not `[1, 4, 1]`
(`tensor_train.py:68-83`).

`down_proj` `(640, 2048)` follows the same 2D auto-reshape path. In that case
the derived working ranks happen to be `[1, 4, 4, 1]`, which matches the test
input:

```python
# From test_e2e_compression_algorithms.py:599-603
consolidator = ModelConsolidator(
    compression_method="tensor_train",
    target_modules=["model.layers[1].mlp.down_proj"],
    tensor_ranks=[1, 4, 4, 1],                    # auto-reshapes 2D to 3D
    backend="pytorch",
)
```

## 5. Block Pruning

Removes entire transformer blocks by activation metric scores. Uses `PruningPlugin` (`pruning.py:57`), not `ModelConsolidator`.

Requires an activation report with per-layer statistics in nested format:

```python
# Report format -- from test_e2e_compression_algorithms.py:643-650
{"per_layer": [
    {"name": "model.layers.0.self_attn.q_proj",
     "statistics": {
         "compute_lda_matrix.l2_norm": {"median": 12.5},   # nested, not flat
     }},
    ...
]}
```

Usage:

```python
# From test_e2e_compression_algorithms.py:668-681
from src.plugins.compression.pruning import PruningPlugin

pruning = PruningPlugin(
    name="pruning",
    **{"pruning": {                                # passed as **kwargs, not configuration=
        "selection_metric_type": "compute_lda_matrix.l2_norm.median",
        "report_path": str(report_path),
        "num_deleted_block": 2,                    # or use ratio_deleted_block=0.25
        "min_keep": 1,
        "container_path": "model.layers",
        "group_prefix": "layers",
    }},
)
pruning.initialize(ctx)
result = pruning.execute(model=model)
```

Note: config must be passed via `**{"pruning": {...}}`, not `configuration={"pruning": {...}}`, because `Plugin.__init__` stores `**kwargs` directly as `self._configuration` (`plugins.py:41`).

The `selection_metric_type` format is `"<base>.<reducer>"`, where `reducer`
must be `.mean` or `.median`. The selector splits on the last `.` and looks up
`statistics[base][reducer]` (`layer_selector.py:47-53`).
