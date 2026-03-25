# Rank Selection in the Goldcrest Compression Pipeline

This document describes how to integrate rank-selection strategies into Goldcrest's
SVD compression pipeline. Three production-ready approaches are covered:

1. **ASVD** — activation-aware SVD with PPL-based sensitivity probing
2. **SVD-LLM** — data whitening + optional closed-form update, orchestrated as a
   single pipeline plugin
3. **Other strategies** — stable rank, Fisher information, histogram entropy, and
   optional external metrics from [information_flow](https://github.com/OFSkean/information_flow) (in this case, you need to download it locally)

All approaches share the same `PipelineContext` / `StateManager` machinery, but
not all of them write rank decisions the same way. ASVD and SVD-LLM persist
resolved ranks under `svd.ranks.<layer_name>`. The analysis-oriented utilities
in section 3 write reports or score tensors into state; `LayerSVDRankDecider`
returns a `layer_ranks` mapping and the caller either passes a chosen rank
directly to `ModelConsolidator` or manually mirrors it into `svd.ranks.*`.

---

## Prerequisites

```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.layers import FactorLinear
from goldcrest.plugins.compression.consolidator import ModelConsolidator
```

Load a model and build the shared context:

```python
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-270m-it",
    torch_dtype=torch.float32,
    trust_remote_code=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

context = PipelineContext(config={}, workspace_dir="workspace/rank_selection")
context.state.model = model
context.state.tokenizer = tokenizer
```

Create a calibration dataloader (used by most strategies):

```python
def make_text_loader(tokenizer, texts, *, seq_len=32, include_labels=False):
    samples = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        batch = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if include_labels:
            batch["labels"] = encoded["input_ids"].clone()
        samples.append(batch)
    return DataLoader(samples, batch_size=None)

calibration_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can be compressed using low-rank factorization.",
    "Singular value decomposition is a fundamental linear algebra technique.",
]
calibration_loader = make_text_loader(tokenizer, calibration_texts, seq_len=32)
```

---

## 1. ASVD: Activation-Aware SVD with PPL Rank Selection

ASVD selects per-layer ranks by probing each layer at several compression ratios
and measuring the resulting perplexity delta. The full pipeline has four stages:

1. **Calibration collection** — record input activations via a forward pass
2. **Activation scaling** — compute per-column ASVD scaling factors
3. **PPL sensitivity probing** — temporarily compress each layer and measure PPL
4. **Binary-search rank allocation** — find per-layer ranks meeting a global
   parameter-ratio budget

### Stage 1: Calibration Collection

```python
from goldcrest.plugins.compression.calibration_collector import CalibrationCollectorPlugin

targets = [
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.self_attn.q_proj",
]

collector = CalibrationCollectorPlugin(
    n_samples=4,
    target_modules=targets,
    name="calibration",
)
collector.initialize(context)
collector_result = collector.execute(dataloader=calibration_loader)
```

After execution the state contains:

| State key | Content |
|---|---|
| `calibration.activations.<layer>` | Input activations per layer |
| `calibration.outputs.<layer>` | Output tensors per layer |
| `calibration.xtx.<layer>` | X^T X matrices for whitening |
| `calibration.layer_names` | List of collected layer names |
| `calibration.collected` | Boolean flag |
| `calibration.sample_count` | Number of samples collected |

### Stage 2: Activation Scaling

```python
from goldcrest.plugins.compression.svd_activation_scaling import ActivationScalingPlugin

scaling = ActivationScalingPlugin(method="abs_mean", alpha=0.5, name="activation_scaling")
scaling.initialize(context)
scaling_result = scaling.execute(target_layers=targets)
```

This writes `svd.scaling.<layer_name>` vectors into state. The mathematical
motivation is:

- Standard SVD minimises `||W - W_approx||_F^2`
- ASVD minimises `||S(W - W_approx)||_F^2` where `S = diag(scaling)^alpha`

Available methods: `"abs_mean"`, `"abs_max"`, `"fisher"`.

### Stage 3: PPL Sensitivity Probing

```python
from goldcrest.plugins.compression.svd_ppl_sensitivity import PPLSensitivityPlugin

sensitivity = PPLSensitivityPlugin(
    param_ratios=[0.5],
    use_activation_scaling=True,
    name="ppl_sensitivity",
)
sensitivity.initialize(context)
sensitivity_result = sensitivity.execute(
    model=model,
    tokenizer=tokenizer,
    eval_dataloader=make_text_loader(
        tokenizer,
        calibration_texts,
        seq_len=32,
        include_labels=True,
    ),
    target_layers=targets,
)
```

State written:

| State key | Content |
|---|---|
| `svd.sensitivity.ppl.<layer>` | Dict mapping ratio -> PPL delta |
| `svd.sensitivity.order` | Layers sorted by max sensitivity |
| `svd.sensitivity.baseline_ppl` | Baseline PPL before any compression |
| `svd.sensitivity.mode` | `"act_aware_ppl"` or `"plain_ppl"` |

### Stage 4: Binary-Search Rank Allocation

```python
from goldcrest.plugins.compression.svd_binary_search_rank import BinarySearchRankPlugin

ranker = BinarySearchRankPlugin(
    target_mode="param_ratio",
    param_ratio_target=0.1,
    sensitivity_source="ppl",
    min_rank=8,
    name="binary_search_rank",
)
ranker.initialize(context)
rank_result = ranker.execute(model=model, target_layers=targets)
```

The plugin reads the full per-layer PPL sensitivity surface from
`svd.sensitivity.ppl.<layer>` and uses a binary search over
`(layer, ratio, score)` candidates to meet the global parameter budget.

Ranks are written to `svd.ranks.<layer_name>` in state.

`rank_result` contains:

```python
{
    "ranks": {"model.layers.0.mlp.gate_proj": 42, ...},
    "total_original_params": ...,
    "total_compressed_params": ...,
    "compression_ratio": ...,
}
```

### Stage 5: Compress with Decided Ranks

```python
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=targets,
    rank=64,                    # fallback rank (overridden by state)
    svd_backend="torch",
    use_activation_scaling=True,
)
consolidator.initialize(context)
compression_result = consolidator.compress_model_with_surgery(model)
```

`ModelConsolidator` checks `svd.ranks.<layer>` in state first. The `rank`
constructor argument serves only as a fallback when no state-driven rank exists.

After compression, target layers become `FactorLinear` instances:

```python
for target in targets:
    modules = dict(model.named_modules())
    layer = modules[target]
    assert isinstance(layer, FactorLinear)
    chosen_rank = context.state.get(f"svd.ranks.{target}")
    factor_shapes = [tuple(f.weight.shape) for f in layer.weight.factors]
    print(f"{target}: rank={chosen_rank}, factors={factor_shapes}")
```

### Saving the Compressed Model

```python
from goldcrest.framework.compressed_io import save_compressed_to_safetensors

manifest = {
    "type": "gemma_asvd_ppl_rank_selection",
    "base_model": "google/gemma-3-270m-it",
    "modules_replaced": compression_result.parameters.get("layers_replaced", []),
    "save_format": "safetensors_state_dict",
}
manifest_path, weights_path = save_compressed_to_safetensors(model, manifest, save_dir)
```

**Source:** ` gemma_asvd_ppl_rank_selection.py`

---

## 2. SVD-LLM: Data Whitening + Closed-Form Update

SVD-LLM wraps calibration, whitening, rank resolution, SVD, and optional
closed-form U refinement into a single `SVDLLMPipelinePlugin`. Layers are
processed sequentially so each subsequent layer is calibrated against the
already-compressed model state.

### Single-Plugin Usage

```python
from goldcrest.plugins.compression.svdllm_pipeline import SVDLLMPipelinePlugin

targets = [
    "model.layers.0.mlp.gate_proj",
    "model.layers.0.self_attn.q_proj",
]

pipeline = SVDLLMPipelinePlugin(
    target_modules=targets,
    rank_ratio=0.1,
    min_rank=8,
    regularization=1e-6,
    svd_backend="torch",
    use_closed_form_update=True,
    name="svdllm_pipeline",
)
pipeline.initialize(context)
compression_result = pipeline.execute(
    model=model,
    dataloader=calibration_loader,
    n_samples=4,
)
```

### What happens inside the pipeline

For each target layer the plugin runs, in order:

1. **Rank resolution** — resolves from `rank`, `rank_ratio`, or pre-populated
   `svd.ranks.<layer>` in state
2. **CalibrationCollectorPlugin** — always collects `X^T X`, and collects
   activations / outputs only when `use_closed_form_update=True`
3. **DataWhiteningPlugin** — Cholesky-decomposes X^T X to produce whitening
   matrices `L` and `L_inv`
4. **ModelConsolidator** — SVD-compresses with `use_data_whitening=True` and
   optionally `use_closed_form_update=True`

The rank formula when using `rank_ratio`:

```
k = floor(m * n * rank_ratio / (m + n))
```

clamped to `[min_rank, min(m, n)]`.

### Result structure

```python
{
    "targets": ["model.layers.0.mlp.gate_proj", ...],
    "layer_results": {
        "<state_name>": {
            "surgery_name": "...",
            "state_name": "...",
            "rank": 42,
            "calibration": {...},
            "whitening": {...},
            "compression": {...},
        },
        ...
    },
    "layers_processed": 2,
    "layers_replaced": ["model.layers.0.mlp.gate_proj", ...],
    "failures": {},
    "use_closed_form_update": True,
}
```

### Whitening state

| State key | Content |
|---|---|
| `svd.whitening.L.<layer>` | Cholesky factor L |
| `svd.whitening.L_inv.<layer>` | Inverse L for reconstruction |

Mathematical background — given calibration inputs X:

```
X^T X = L @ L^T            (Cholesky decomposition)
W_white = W @ L             (whitening transform)
V_original = L^{-1} @ V    (inverse after truncation)
```

### Saving the SVD-LLM Model

```python
from goldcrest.framework.compressed_io import save_compressed_to_safetensors

layers_replaced = compression_result.get("layers_replaced", [])
layer_results = compression_result.get("layer_results", {}) or {}

selected_ranks = {}
for state_name, layer_result in layer_results.items():
    surgery_name = str(layer_result.get("surgery_name") or state_name)
    rank = layer_result.get("rank")
    if rank is not None:
        selected_ranks[surgery_name] = int(rank)

manifest = {
    "type": "gemma_svdllm_rank_selection",
    "base_model": "google/gemma-3-270m-it",
    "modules_replaced": layers_replaced,
    "save_format": "safetensors_state_dict",
    "use_closed_form_update": compression_result.get("use_closed_form_update"),
    "selected_ranks": selected_ranks,
}
manifest_path, weights_path = save_compressed_to_safetensors(model, manifest, save_dir)
```

### Disabling Closed-Form Update

Pass `use_closed_form_update=False` (or `--disable-closed-form-update` in the
standalone script) to skip the refinement step. The SVD-LLM pipeline still
applies data whitening; only the post-SVD update of the U factor is skipped.

**Source:** ` gemma_svdllm_rank_selection.py`

---

## 3. Other Rank-Selection Strategies

The E2E test suite (`tests/e2e/test_e2e_rank_selection.py`) demonstrates four
additional analysis signals and heuristics that can feed into the same
compression backend.

### 3a. Stable Rank

Stable rank is `(frobenius_norm / spectral_norm)^2`. The
`LayerSVDRankDecider` utility reads an activation-metrics report, computes a
block-level base rank from the report statistics, scales it by the selected
block score, and returns a `layer_ranks` mapping keyed by bracket-index module
paths such as `model.layers[0].self_attn.q_proj`.

```python
from goldcrest.plugins.analysis.layer_svd_rank_decider import LayerSVDRankDecider

# Build an activation report (normally produced by ActivationMetricsPlugin)
per_layer = []
for i in range(18):
    for mod in ["self_attn.q_proj", "mlp.gate_proj"]:
        per_layer.append({
            "name": f"model.layers.{i}.{mod}",
            "statistics": {
                "compute_lda_matrix.frobenius_norm": {"median": float(40 + i * 3)},
                "compute_lda_matrix.spectral_norm": {"median": float(4 + i * 0.3)},
            },
        })
report = {"per_layer": per_layer}

cfg = {
    "rank_strategy": "stable_rank",
    "selection_metric_type": "compute_lda_matrix.frobenius_norm.median",
    "min_rank": 8,
    "max_rank": 64,
}
decider = LayerSVDRankDecider(cfg)
decision = decider.decide(model, report)
ranks = decision.get("layer_ranks", {})
# ranks: {"model.layers[0].self_attn.q_proj": 12, ...}
```

The current tests feed the decided rank directly into compression rather than
writing it back to `svd.ranks.*`:

```python
target_name = sorted(ranks.keys())[0]
chosen_rank = ranks[target_name]

consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=[target_name],
    rank=chosen_rank,
    svd_backend="torch",
)
consolidator.initialize(context)
result = consolidator.compress_model_with_surgery(model)
```

### 3b. Fisher Information

Fisher information approximates parameter importance via squared gradients. In
the current codebase it produces per-layer Fisher tensors in state and a
normalised scalar score map via `compute_importance_scores()`. Those scores are
useful for custom allocation logic, and Fisher tensors can also drive
activation-aware scaling via `ActivationScalingPlugin(method="fisher")`, but
there is no built-in Fisher-specific rank allocator in the referenced tests.

```python
from goldcrest.plugins.analysis.fisher_information import FisherInformationPlugin

fisher = FisherInformationPlugin(n_samples=2, name="fisher")
fisher.initialize(context)

loader = make_text_loader(tokenizer, calibration_texts, seq_len=32, include_labels=True)
fisher.execute(dataloader=loader)

fisher_data = context.state.get("calibration.fisher", {})
# fisher_data: {"model.layers.0.mlp.gate_proj": <tensor>, ...}

scores = fisher.compute_importance_scores(fisher_data)
# scores: {"model.layers.0.mlp.gate_proj": 0.73, ...}  (normalised to [0, 1])
```

### 3c. Histogram Entropy

Histogram entropy is computed as part of the same activation-metrics pass. It
quantifies the spread of activation values per layer. Layers with low entropy
(more concentrated activations) tend to tolerate more aggressive compression.

```python
ctx = PipelineContext(
    config={
        "analysis": {
            "metrics": {"names": ["compute_lda_matrix"]},
            "selection": {"include_names": ["mlp"]},
            "aggregation": {"reductions": ["mean"]},
            "compute": {"move_to_cpu": True, "seed": 42},
        },
    },
    workspace_dir=tmp_path,
)
ctx.state.model = model
ctx.state.tokenizer = tokenizer

plugin = ActivationMetricsPlugin(name="activation_metrics")
plugin.initialize(ctx)
plugin.execute(dataloader=calibration_loader)

report = ctx.state.get("analysis.activation_metrics")
for entry in report["per_layer"]:
    entropy = entry["statistics"].get(
        "compute_lda_matrix.histogram_entropy", {}
    ).get("mean")
    if entropy is not None:
        print(f"{entry['name']}: histogram_entropy={entropy}")
```

### 3d. [`information_flow`](https://github.com/OFSkean/information_flow) Metrics (External Optional Integration)

Activation metrics including Frobenius norm and mutual information are computed
by `ActivationMetricsPlugin`. These feed the stable-rank decider or can be used
to build custom rank-allocation logic. The built-in examples/tests use the
derived metrics from `compute_lda_matrix`.

Optional external metrics from
[`information_flow`](https://github.com/OFSkean/information_flow) can also be
loaded through `ExternalMetricsBackend`, but there is an important practical
detail: at the time of writing the upstream repository is not published on PyPI
and does not include `setup.py` or `pyproject.toml`. In this codebase the
reliable integration path is therefore:

1. Clone `information_flow` into a local directory, for example
   `third_party/information_flow`
2. Point `analysis.metrics.import.file` at
   `third_party/information_flow/experiments/utils/metrics/metric_functions.py`
3. Set `name_prefix: "metric_"` so exported names such as
   `metric_effective_rank` become `effective_rank`
4. Use `selection_metric_type` such as `effective_rank.median` or
   `dime.median` when driving `LayerSVDRankDecider`

```bash
git clone https://github.com/OFSkean/information_flow.git third_party/information_flow
```

```python
from goldcrest.plugins.analysis.activation_metrics import ActivationMetricsPlugin
from goldcrest.plugins.analysis.layer_svd_rank_decider import LayerSVDRankDecider

ctx = PipelineContext(
    config={
        "analysis": {
            "type": "activation_metrics",
            "metrics": {
                "names": "all",
                "import": {
                    "file": "third_party/information_flow/experiments/utils/metrics/metric_functions.py",
                    "name_prefix": "metric_",
                },
            },
            "selection": {
                "include_names": ["mlp", "self_attn"],
                "module_types": ["GemmaMLP", "GemmaAttention"],
            },
            "aggregation": {"reductions": ["mean", "median"]},
            "compute": {"move_to_cpu": True, "seed": 42},
        },
        "svd_compression": {
            "rank_strategy": "stable_rank",
            "selection_metric_type": "effective_rank.median",
            "min_rank": 16,
            "target_param_ratio": 0.5,
        },
    },
    workspace_dir=tmp_path,
)
ctx.state.model = model
ctx.state.tokenizer = tokenizer

plugin = ActivationMetricsPlugin(name="activation_metrics")
plugin.initialize(ctx)
report = plugin.execute(dataloader=calibration_loader)

decider = LayerSVDRankDecider(ctx.config["svd_compression"])
decision = decider.decide(model, report)
ranks = decision.get("layer_ranks", {})
```

If you want to inspect the metric keys that were actually emitted into the
report, iterate through `report["per_layer"][i]["statistics"]` and use keys of
the form `<metric>.mean` or `<metric>.median` as `selection_metric_type`.
Scalar external metrics commonly become keys like `effective_rank`,
`prompt_entropy`, or `dime`.

One implementation detail matters when using the built-in decider:
`LayerSVDRankDecider` uses `selection_metric_type` to score blocks, but its base
rank heuristic is still derived from `compute_lda_matrix.frobenius_norm` and
`compute_lda_matrix.spectral_norm`. In practice that means `names: "all"` is
the safest configuration, because it keeps both the built-in LDA-derived stats
and the external `information_flow` metrics available in the same report.

---

## Pipeline State Key Reference

All rank-selection plugins communicate through `PipelineContext.state`. The
table below lists every key used in the rank-selection flow.

| State key pattern | Written by | Read by |
|---|---|---|
| `calibration.activations.<layer>` | `CalibrationCollectorPlugin` | `ActivationScalingPlugin`, `ClosedFormUpdatePlugin` |
| `calibration.outputs.<layer>` | `CalibrationCollectorPlugin` | `ClosedFormUpdatePlugin` |
| `calibration.xtx.<layer>` | `CalibrationCollectorPlugin` | `DataWhiteningPlugin` |
| `calibration.layer_names` | `CalibrationCollectorPlugin` | `ActivationScalingPlugin`, `DataWhiteningPlugin`, `ClosedFormUpdatePlugin` |
| `calibration.collected` | `CalibrationCollectorPlugin` | `ActivationScalingPlugin`, `DataWhiteningPlugin`, `ClosedFormUpdatePlugin` |
| `calibration.sample_count` | `CalibrationCollectorPlugin` | informational |
| `calibration.fisher.<layer>` | `FisherInformationPlugin` | `ActivationScalingPlugin` (method=`"fisher"`) |
| `svd.scaling.<layer>` | `ActivationScalingPlugin` | `ModelConsolidator` / `SVD` (when `use_activation_scaling=True`) |
| `svd.whitening.L.<layer>` | `DataWhiteningPlugin` | `ModelConsolidator` / `SVD` (when `use_data_whitening=True`) |
| `svd.whitening.L_inv.<layer>` | `DataWhiteningPlugin` | `ModelConsolidator` / `SVD` (when `use_data_whitening=True`) |
| `svd.sensitivity.ppl.<layer>` | `PPLSensitivityPlugin` | `BinarySearchRankPlugin` |
| `svd.sensitivity.order` | `PPLSensitivityPlugin` | informational |
| `svd.sensitivity.baseline_ppl` | `PPLSensitivityPlugin` | informational |
| `svd.sensitivity.mode` | `PPLSensitivityPlugin` | informational |
| `svd.ranks.<layer>` | `BinarySearchRankPlugin`, `SVDLLMPipelinePlugin` | `SVD`, `ModelConsolidator`, `SVDLLMPipelinePlugin` |
| `analysis.activation_metrics` | `ActivationMetricsPlugin` | `LayerSVDRankDecider` |

---

## Choosing a Strategy

| Strategy | When to use | Data required |
|---|---|---|
| **ASVD + PPL sensitivity** | Maximum control; explicit PPL-aware budget allocation | Calibration data + eval set for PPL probing |
| **SVD-LLM pipeline** | Single-command whitening + compression; sequential layer-by-layer | Calibration data |
| **Stable rank** | Fast heuristic from activation statistics; no eval pass needed | Activation-metrics report |
| **Fisher information** | Gradient-based importance when labelled data is available; useful for Fisher-aware scaling or custom rank logic | Labelled calibration data (requires backward passes) |
| **Information flow / entropy** | Exploratory profiling before choosing a budget; optional external `information_flow` metrics require a local checkout plus direct file or module import through `ExternalMetricsBackend` | Calibration data |

---

## Standalone Scripts

Two runnable scripts wrap the full rank-selection flow for quick experiments:

```bash
# ASVD + PPL rank selection
HF_TOKEN=... python  gemma_asvd_ppl_rank_selection.py \
    --targets "model.layers.0.mlp.gate_proj,model.layers.0.self_attn.q_proj" \
    --param-ratio-target 0.1 \
    --ppl-ratios "0.5" \
    --min-rank 8

# SVD-LLM rank selection
HF_TOKEN=... python  gemma_svdllm_rank_selection.py \
    --targets "model.layers.0.mlp.gate_proj,model.layers.0.self_attn.q_proj" \
    --param-ratio-target 0.1 \
    --min-rank 8 \
    --regularization 1e-6
```

Both scripts load the model, run baseline LMHarness evaluation, compress, run
post-compression evaluation, and save the result to safetensors.
