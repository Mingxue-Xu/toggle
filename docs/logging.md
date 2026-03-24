# CSV Logging Guide

Source: `src/plugins/evaluation/csv_logger.py`

## Setup

```python
from src.plugins.evaluation.csv_logger import CSVLogger, ResultComparator

logger = CSVLogger(output_dir="./logs/csv")        # creates 4 CSV files
comparator = ResultComparator(logger)               # cross-experiment analysis
```

The constructor creates `experiments.csv`, `models.csv`, `evaluations.csv`, and `compressions.csv` with headers immediately if they do not already exist.


| File | Columns |
|------|---------|
| `experiments.csv` | experiment_id, start_time, end_time, duration_minutes, config_file, user, status, error_message |
| `models.csv` | experiment_id, timestamp, model_name, model_type, parameter_count, size_mb, loading_time, device, precision, architecture_info |
| `compressions.csv` | experiment_id, timestamp, compression_method, original_size_mb, compressed_size_mb, compression_ratio, compression_time, parameters, memory_usage_mb |
| `evaluations.csv` | experiment_id, timestamp, model_type, evaluation_type, plugin_name, task_name, metric_name, metric_value, evaluation_params, execution_time |

Dict fields (`architecture_info`, `parameters`, `evaluation_params`) are serialized as JSON.


## Experiment sessions

Every logging call requires an active session. Wrap your work in `start_experiment` / `end_experiment`:

```python
exp_id = logger.start_experiment("svd_rank32", config_file="config.yaml", user="alice")
# ... all log_* calls go here ...
logger.end_experiment(status="completed")           # API also accepts "failed", "cancelled", "skipped"
```

In the e2e test suite this is automatic via the `_experiment_session` autouse fixture in `conftest.py`:

```python
@pytest.fixture(autouse=True)
def _experiment_session(csv_logger, request):
    test_name = request.node.name
    csv_logger.start_experiment(test_name, config_file="e2e_test")
    yield
    status = "completed"
    if hasattr(request.node, "rep_call") and request.node.rep_call.failed:
        status = "failed"
    elif hasattr(request.node, "rep_setup") and request.node.rep_setup.skipped:
        status = "skipped"
    csv_logger.end_experiment(status=status)
```

## log_model_info

Records model metadata before and after compression. Writes to `models.csv`.

```python
# csv_logger.py:190
log_model_info(model_name: str, model_type: str, model_info: Dict[str, Any],
               loading_time: float, device: str, precision: str) -> None
```

Common `model_info` keys are `"num_parameters"` and `"size_mb"` -- these are extracted into dedicated CSV columns. If they are missing, `parameter_count` and `size_mb` default to `0` / `0.0`; the full dict is always stored as JSON in `architecture_info`.

```python
# From test_e2e_compression_algorithms.py:83-84
csv_logger.log_model_info(MODEL_ID, "baseline",
    {"num_parameters": original_params, "size_mb": orig_mb}, 0.0, DEVICE, "float32")
csv_logger.log_model_info(MODEL_ID, "compressed",
    {"num_parameters": compressed_params, "size_mb": comp_mb}, 0.0, DEVICE, "float32")

# Custom fields are allowed; extracted numeric columns will fall back to 0 / 0.0
csv_logger.log_model_info(MODEL_ID + "_asvd_svd", "compressed",
    {"params": param_count(model), "layers_replaced": len(targets)}, 0.0, DEVICE, "float32")
```

## log_compression_results

Records one compression operation. Writes to `compressions.csv`.

```python
# csv_logger.py:275
log_compression_results(compression_method: str, compression_results: Dict[str, Any],
                        parameters: Dict[str, Any]) -> None
```

Recognized `compression_results` keys are `original_size_mb`, `compressed_size_mb`, `compression_ratio`, `compression_time`, `memory_usage_mb`. Missing values default to `0.0`, except `compression_ratio` which defaults to `1.0`. The `parameters` dict is stored as JSON.

```python
# From test_e2e_compression_algorithms.py:85-89
csv_logger.log_compression_results("svd", {
    "original_size_mb": orig_mb,
    "compressed_size_mb": comp_mb,
    "compression_ratio": original_params / max(compressed_params, 1),
    "compression_time": elapsed,
    "memory_usage_mb": 0.0,
}, {"rank": 32, "svd_backend": "torch", "target_modules": [...]})
```

Valid `compression_method` values used across the test suite: `"svd"`, `"tucker"`, `"cp"`, `"tensor_train"`, `"block_pruning"`, `"svd+asvd"`, `"svd+whitening"`, `"svd+stable_rank"`.

## log_evaluation_results

Records evaluation or profiling metrics. Writes to `evaluations.csv`.

```python
# csv_logger.py:222
log_evaluation_results(model_type: str, evaluation_type: str, plugin_name: str,
                       task_results: Dict[str, Any], evaluation_params: Dict[str, Any],
                       execution_time: float) -> None
```

`task_results` is flattened: `{task_name: {metric_name: value}}` produces one CSV row per metric. A non-dict value is stored with `metric_name="result"`. `metric_value` can therefore be numeric, boolean, or string depending on what you log.

```python
# Evaluation -- from test_e2e_evaluation.py:95-102
csv_logger.log_evaluation_results(
    model_type="compressed", evaluation_type="lm_eval",
    plugin_name="lm_eval_harness",
    task_results={"arc_easy": {"result": str(result)}},
    evaluation_params={"tasks": ["arc_easy"], "limit": 4},
    execution_time=eval_time)

# Inference check -- from test_e2e_compression_algorithms.py:169-171
csv_logger.log_evaluation_results("compressed", "profile", "inference_check",
    {"inference": {"output_length": len(output), "valid": True}},
    {"prompt": "The capital of France is"}, inference_elapsed)

# Memory profiling -- from test_e2e_profiling.py:72-79
csv_logger.log_evaluation_results(
    model_type="compressed", evaluation_type="profile",
    plugin_name="memory_rss",
    task_results={"rss": {
        "pre_rss_mb": profile.pre_execution.rss_mb,
        "post_rss_mb": profile.post_execution.rss_mb}},
    evaluation_params={"phase": "svd_compression"},
    execution_time=elapsed)
```

## ResultComparator

Reads the CSV files back for cross-experiment analysis. `compare_experiments(...)` returns the raw CSV rows loaded by `csv.DictReader`, so field values are strings.

```python
comparator = ResultComparator(logger)

# Compare multiple experiments by ID
comparison = comparator.compare_experiments(["exp_rank16_...", "exp_rank64_..."])
# Returns: {experiments, model_comparison, evaluation_comparison, compression_comparison}

# Analyze compression efficiency (baseline vs compressed retention)
analysis = comparator.analyze_compression_efficiency("exp_id")
# Returns: {experiment_id, compression_ratio, performance_retention, efficiency_score}
```

`analyze_compression_efficiency` matches `model_type="baseline"` rows against `model_type="compressed"` rows by `task_name + metric_name`, computing retention ratios and `efficiency_score = compression_ratio * avg_retention`.

Important constraints:
- It uses the first parseable `compression_ratio` found in `compressions.csv` for that experiment.
- It only compares numeric `metric_value`s. Non-numeric rows (for example stringified `str(result)` payloads or boolean flags such as `valid=True`) are ignored.
