# Toggle Architecture

For a easy extension regarding different LLMs (Qwen, Gemma, Llama, etc) and various compression strategies (i.e. different low-rank strategies on different kinds of layers, what criteria should be used to choose ranks), toggle adopts a **factory pattern and plugin-based architecture**, which are typical object-oriented programming (OOP) design patterns. This design choice addresses two key redundancy concerns: 
- functions reused across different compression pipelines are abstracted into shared plugins; 
- interfaces with third-party dependencies (e.g., HuggingFace transformers, lm-eval-harness) are isolated behind well-defined boundaries. 

Powered by this OOP design, rather than hard-coding a single compression pipeline, Toggle treats each operation -- model loading, layer analysis, compression, evaluation -- as a self-contained plugin that communicates through a centralized event bus.

Furthermore, frequently accessed resources like model weights are managed independently through the `ModelManager` and `PipelineContext`, decoupling model lifecycle from any specific compression pipeline.



## Design Philosophy

- **Plugin-based** -- Every operation is a plugin with a well-defined lifecycle (`initialize`, `execute`, `cleanup`).
- **Event-driven** -- A thread-safe `EventBus` with 47 event types decouples plugins from one another.
- **Composable** -- Workflows define directed acyclic graphs of steps with explicit dependencies.
- **Reproducible** -- YAML-driven configuration with sensible defaults in `config/base/default.yaml`.

## System Overview

```
                    +---------------------+
                    | PipelineOrchestrator |
                    +----------+----------+
                               |
                    +----------v----------+
                    |      Workflow        |
                    | (DAG of WorkflowSteps)|
                    +----------+----------+
                               |
              +----------------+----------------+
              |                |                |
     +--------v------+  +-----v------+  +------v-------+
     |    Plugins     |  |  EventBus  |  | PipelineState|
     | (compression,  |  | (pub/sub)  |  | (state mgmt) |
     |  evaluation,   |  +------------+  +--------------+
     |  analysis,     |
     |  models)       |
     +----------------+
```

## Core Components

| Component | Module | Role |
|---|---|---|
| `PipelineOrchestrator` | `src/orchestration/orchestrator.py` | Manages workflows, plugins, and events |
| `Workflow` / `WorkflowStep` | `src/orchestration/workflow.py` | Defines pipeline steps and their dependencies |
| `WorkflowExecutor` | `src/orchestration/executor.py` | Executes workflow steps respecting dependency order |
| `Plugin` | `src/framework/plugins.py` | Abstract base class with lifecycle management |
| `PluginRegistry` | `src/framework/plugins.py` | Registration and discovery of plugins |
| `EventBus` | `src/framework/events.py` | Thread-safe pub/sub event system |
| `PipelineContext` | `src/framework/context.py` | Centralized execution context shared across plugins |
| `PipelineState` / `StateManager` | `src/framework/state.py` | State persistence and management |
| `ModelManager` | `src/framework/model_manager.py` | Unified model access, validation, and lifecycle management |

## Supported Compression Methods

| Method | Plugin(s) | Description |
|---|---|---|
| SVD | `SVD` | Standard singular value decomposition with configurable rank |
| ASVD | `ActivationScaling`, `DataWhitening`, `ClosedFormUpdate` | Activation-guided SVD with calibration-based scaling and whitening |
| SVD-LLM | `PPLSensitivity`, `BinarySearchRank` | Perplexity-aware rank selection via binary search |
| Tucker | `Tucker` | Tucker decomposition for multi-dimensional tensors |
| CP | `CP` | CANDECOMP/PARAFAC decomposition |
| Tensor-Train | `TensorTrain` | Tensor-train decomposition for high-order tensors |
| Structured Pruning | `PruningPlugin` | Weight magnitude and gradient-based structured pruning |
| KV-Cache Compression | `KVCacheCalibrator`, `KVCacheRuntime` | Projection-based KV-cache compression for inference |

## Plugin Lifecycle

Each plugin follows a standard lifecycle managed by the framework:

1. **initialize()** -- Called once when the plugin is first loaded. Use this for one-time setup.
2. **execute()** -- The main entry point. Calls `do_execute()` internally with error handling and event emission.
3. **cleanup()** -- Called after execution completes. Use this for resource cleanup.

The `execute()` method on the base class handles lifecycle events, error handling, and EventBus integration automatically -- you only need to implement `do_execute()`.

## EventBus

The `EventBus` is a thread-safe publish-subscribe system that decouples plugins from one another. Plugins can:

- **Emit events** to notify other components of state changes
- **Subscribe to events** to react to changes from other plugins

This design enables loose coupling and makes it easy to add new functionality without modifying existing code.

## Resource Management

Toggle separates model lifecycle management from pipeline execution through two complementary components: `PipelineContext` and `ModelManager`.

### PipelineContext

`PipelineContext` (`src/framework/context.py`) is the centralized execution context shared across all plugins. It provides:

- **Shared state** (`context.state`) -- Thread-safe `PipelineState` instance (protected by `RLock`) holding models, compression artifacts, evaluation results, and workflow status.
- **Event communication** (`context.event_bus`) -- `EventBus` instance for inter-plugin pub/sub messaging.
- **Configuration access** (`context.get_config(key)`) -- Dot-notation access to nested YAML configuration (e.g., `"model.name"`, `"compression.rank"`).
- **Resource sharing** (`context.set_resource()` / `get_resource()`) -- Store and retrieve shared resources by name without tight coupling between plugins.
- **Workspace management** (`context.get_workspace_path()`) -- Build paths within the pipeline's working directory for artifacts and logs.
- **Reproducibility** (`context.get_seed()`) -- Automatic seed initialization from config paths: `seed`, `analysis.compute.seed`, `compute.seed`, `runtime.seed`.
- **State persistence** (`context.save_state()` / `load_state()`) -- Serialize and restore pipeline state to/from JSON.

All plugins receive the same `PipelineContext` instance, enabling coordination without direct dependencies:

```python
# Plugin A stores a resource
context.set_resource("calibration_data", activations)

# Plugin B retrieves it later
data = context.get_resource("calibration_data")
```

### ModelManager

`ModelManager` (`src/framework/model_manager.py`) provides a unified interface for model access and validation, decoupling model lifecycle from individual plugins. Key methods:

| Method | Purpose |
|--------|---------|
| `get_model(context, type)` | Retrieve model by type (`"baseline"` or `"compressed"`) with validation |
| `set_model(context, model, type)` | Store model in context after validation |
| `get_tokenizer(context, type)` | Retrieve tokenizer with fallback logic |
| `validate_model(model, expected_type)` | Validate model is callable and matches expected type |
| `get_model_info(model)` | Return metadata: parameter count, device, compression status |

**Model types:**
- `"baseline"` / `"original"` -- The uncompressed reference model (stored in `context.state.original_model`)
- `"compressed"` / `"current"` -- The working model after compression (stored in `context.state.model`)

**Validation logic:**
- Compressed models: Checks for `FactorEmbedding` or `FactorLinear` layers from the `tensorizer` module
- Baseline models: Validates `forward` and `parameters` attributes with non-zero parameter count

This separation ensures plugins request models through a validated interface rather than directly accessing state:

```python
# Instead of: model = context.state.model
manager = ModelManager()
model = manager.get_model(context, "compressed")  # Validated access
```

## Workflow Definition

Workflows are defined as directed acyclic graphs (DAGs) of `WorkflowStep` objects. Each step specifies:

- **name** -- A unique identifier for the step
- **plugin** -- The plugin class to execute
- **depends_on** -- A list of step names that must complete before this step runs

The `WorkflowExecutor` automatically resolves dependencies and executes steps in the correct order.

## Configuration System

Toggle uses a hierarchical YAML configuration system:

1. **Base defaults** (`config/base/default.yaml`) -- Shared defaults for all pipelines
2. **Method configs** (`config/h100_*.yaml`) -- Method-specific configurations
3. **Profile configs** (`config/profiles/`) -- Model-specific overrides

Configuration values cascade from base to specific, allowing you to override only what you need.

## Directory Structure

```
src/
├── config/
│   └── loader.py             # YAML config loading and validation
├── framework/
│   ├── context.py            # PipelineContext
│   ├── events.py             # EventBus, PipelineEvent
│   ├── plugins.py            # Plugin, PluginRegistry
│   ├── state.py              # PipelineState, StateManager
│   ├── layers.py             # Layer utilities
│   ├── memory_profiler.py    # GPU/CPU memory profiling
│   ├── model_manager.py      # Model loading helpers
│   ├── statistics.py         # Statistical utilities
│   ├── strategy_factory.py   # Compression strategy factory
│   ├── compressed_io.py      # Compressed model I/O
│   ├── eval_interface.py     # Evaluation interface
│   ├── reproducibility.py    # Seed management, config hashing
│   └── inference_subprocess.py
├── orchestration/
│   ├── orchestrator.py       # PipelineOrchestrator
│   ├── workflow.py           # Workflow, WorkflowStep
│   └── executor.py           # WorkflowExecutor
└── plugins/
    ├── models/
    │   └── loader.py         # Model loading plugin
    ├── compression/
    │   ├── svd.py            # SVD
    │   ├── svd_activation_scaling.py   # ASVD scaling
    │   ├── svd_data_whitening.py       # ASVD whitening
    │   ├── svd_closed_form_update.py   # ASVD closed-form update
    │   ├── svd_ppl_sensitivity.py      # SVD-LLM sensitivity
    │   ├── svd_binary_search_rank.py   # SVD-LLM rank search
    │   ├── tucker.py         # Tucker decomposition
    │   ├── cp.py             # CP decomposition
    │   ├── tensor_train.py   # Tensor-train decomposition
    │   ├── pruning.py        # Structured pruning
    │   ├── kv_cache_projection_calibrator.py
    │   ├── kv_cache_projection_runtime.py
    │   └── ...               # Backend, utilities, consolidation
    ├── evaluation/
    │   ├── base.py           # Evaluation base classes
    │   ├── baseline_eval.py  # Baseline evaluation
    │   ├── compressed_eval.py# Compressed model evaluation
    │   ├── lm_eval.py        # lm-eval-harness integration
    │   └── csv_logger.py     # CSV result logging
    └── analysis/
        ├── weight_metrics.py # Weight analysis metrics
        ├── activation_metrics.py
        ├── fisher_information.py
        ├── layer_selector.py
        ├── layer_svd_rank_decider.py
        ├── memory_inference.py
        ├── report_loader.py
        ├── metric_utils.py   # Metric computation utilities
        └── pruning_plugin.py # Pruning analysis plugin
```
