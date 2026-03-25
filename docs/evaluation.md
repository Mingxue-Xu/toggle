# Evaluation in the Goldcrest Compression Pipeline

This document describes how to evaluate compressed (and baseline) models using
Goldcrest's evaluation plugins. Two complementary approaches are covered:

1. **LMHarness** — EleutherAI `lm-eval-harness` integration for standard
   benchmark tasks (arc_easy, hellaswag, wikitext, etc.)
2. **Self-defined evaluation** — `ModelEvalInterface` for direct perplexity and
   log-likelihood computation, and `LMEvaluator` for lightweight task evaluation
   without the external `lm_eval` dependency

Both approaches plug into the same `PipelineContext` and can log results through
the `CSVLogger` for structured experiment tracking.

---

## Prerequisites

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from goldcrest.framework.context import PipelineContext
from goldcrest.framework.eval_interface import ModelEvalInterface
from goldcrest.plugins.compression.consolidator import ModelConsolidator
```

Load a model and set up the context:

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

ctx = PipelineContext(config={}, workspace_dir="workspace/evaluation")
ctx.state.model = model
ctx.state.tokenizer = tokenizer
```

Compress target layers (shared setup for the examples below):

```python
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=["model.layers[0].self_attn.q_proj"],
    rank=64,
    svd_backend="torch",
)
consolidator.initialize(ctx)
consolidator.compress_model_with_surgery(model)
```

---

## 1. LMHarness: lm-eval-harness Integration

`LMHarness` wraps the EleutherAI LM Evaluation Harness so that any model
already loaded in memory — baseline or compressed — can be evaluated on standard
benchmarks without reloading from a checkpoint.

### Requirements

```bash
pip install lm_eval
```

If `lm_eval` is not installed, `LMHarness.evaluate_task()` raises
`ImportError`. The end-to-end tests guard this path with `pytest.skip(...)`
before running the plugin.

### Single-Task Evaluation

```python
from goldcrest.plugins.evaluation.lm_eval import LMHarness

evaluator = LMHarness(
    tasks=["arc_easy"],
    batch_size=1,
    max_length=256,
    device=str(model.device),
    limit=4,
    backend="auto",
    name="lm_eval",
)
evaluator.initialize(ctx)
result = evaluator.execute()
```

`result` is a `Dict[str, ModelEvaluationResult]` keyed by task name. Each
`ModelEvaluationResult` contains:

| Field | Type | Description |
|---|---|---|
| `task_name` | `str` | Name of the evaluation task |
| `metrics` | `Dict[str, float]` | Task-specific metric values |
| `num_samples` | `Any` | Sample-count payload copied from `lm_eval`'s `n-samples` output for the task (commonly an `int`) |
| `evaluation_time` | `float` | Wall-clock seconds |
| `model_name` | `Optional[str]` | Model identifier |
| `additional_info` | `Optional[Dict]` | Extra metadata |

### Multi-Task Evaluation

Pass multiple task names to evaluate in a single call:

```python
evaluator = LMHarness(
    tasks=["arc_easy", "hellaswag"],
    batch_size=1,
    max_length=256,
    device=str(model.device),
    limit=2,
    backend="auto",
    name="lm_eval",
)
evaluator.initialize(ctx)
result = evaluator.execute()
```

### Baseline (Uncompressed) Evaluation

Run the same evaluator on the uncompressed model to establish a baseline:

```python
ctx = PipelineContext(config={}, workspace_dir="workspace/evaluation")
ctx.state.model = model
ctx.state.tokenizer = tokenizer

evaluator = LMHarness(
    tasks=["arc_easy"],
    batch_size=1,
    max_length=256,
    device=str(model.device),
    limit=4,
    backend="auto",
    name="lm_eval",
)
evaluator.initialize(ctx)
result = evaluator.execute()
```

No compression step is needed; the evaluator works identically on baseline and
compressed models.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tasks` | `List[str]` | `None` | Benchmark tasks to evaluate |
| `batch_size` | `int` | `1` | Evaluation batch size |
| `max_length` | `int` | `2048` | Maximum sequence length |
| `device` | `str` | `"auto"` | `"auto"`, `"cuda"`, or `"cpu"` |
| `limit` | `int` | `None` | Cap the number of evaluation examples |
| `backend` | `str` | `"auto"` | `"auto"` / `"adapter"` / `"in_memory"` use the in-memory model; `"hf"` / `"hf-causal"` reload from checkpoint |
| `model_type` | `str` | `"auto"` | Hint: `"compressed"`, `"baseline"`, or `"auto"` |
| `hf_model_name` | `str` | `None` | Required when `backend="hf"` |
| `allow_checkpoint_reload` | `bool` | `False` | Must be `True` for `backend="hf"` with an in-memory model |
| `name` | `str` | (required) | Plugin instance name |

### Supported Tasks

`LMHarness.get_supported_tasks()` exposes the curated list below. At execution
time, `LMHarness.evaluate_task()` only warns when a task is outside that list
and still passes the task string through to `lm_eval`.

The curated built-in task list is:

| Category | Tasks |
|---|---|
| Common-sense reasoning | `hellaswag`, `winogrande`, `piqa`, `siqa`, `commonsense_qa` |
| Reading comprehension | `arc_easy`, `arc_challenge`, `openbookqa`, `boolq`, `race` |
| Mathematical reasoning | `gsm8k`, `math_algebra`, `math_counting_and_probability` |
| Knowledge and facts | `truthfulqa_mc1`, `truthfulqa_mc2`, `mmlu` |
| Language understanding | `copa`, `wsc`, `wsc273` |
| Multiple choice | `swag`, `lambada_openai` |
| Perplexity | `wikitext`, `ptb_new` |

**Source:** `tests/e2e/test_e2e_evaluation.py`, class `TestLMHarness`

---

## 2. Self-Defined Language Task Evaluation

For evaluation without the external `lm_eval` dependency, Goldcrest provides two
interfaces:

- **`ModelEvalInterface`** — direct log-likelihood and rolling log-likelihood
  computation on arbitrary text
- **`LMEvaluator`** — a plugin wrapper around `ModelEvalInterface` for a fixed
  set of built-in classification and perplexity tasks

### 2a. Perplexity via ModelEvalInterface

`loglikelihood_rolling` computes the rolling log-likelihood of each text, which
is the standard basis for perplexity measurement.

```python
from goldcrest.framework.eval_interface import ModelEvalInterface

eval_interface = ModelEvalInterface(
    model=model,
    tokenizer=tokenizer,
    device=str(model.device),
)

test_texts = [
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
]
results = eval_interface.loglikelihood_rolling(test_texts)

for i, r in enumerate(results):
    ll = r[0] if isinstance(r, tuple) else r
    print(f"text_{i}: rolling_ll={ll}")
    assert isinstance(ll, float)
    assert ll < 0, "Log-likelihood should be negative"
```

Each entry in `results` is a `(log_likelihood,)` tuple. The log-likelihood is
always a finite negative float for well-formed input.

### 2b. Log-Likelihood of Continuations

`loglikelihood` scores how likely a continuation is given a context prefix. This
is the primitive behind most multiple-choice benchmarks.

```python
eval_interface = ModelEvalInterface(
    model=model,
    tokenizer=tokenizer,
    device=str(model.device),
)

requests = [
    ("The sky is", " blue"),
    ("Water is", " wet"),
]
results = eval_interface.loglikelihood(requests)

for i, r in enumerate(results):
    ll, is_greedy = r
    print(f"request_{i}: ll={ll}, is_greedy={is_greedy}")
    assert isinstance(ll, float)
```

Each entry is a `(log_likelihood, is_greedy)` tuple where `is_greedy` is `True`
when the continuation tokens match the model's argmax predictions.

### 2c. Baseline vs Compressed Perplexity Comparison

A common workflow is to measure perplexity before and after compression to
quantify quality loss:

```python
test_texts = ["The cat sat on the mat and looked around."]

# Baseline
eval_baseline = ModelEvalInterface(
    model=model, tokenizer=tokenizer, device=str(model.device),
)
baseline_results = eval_baseline.loglikelihood_rolling(test_texts)
baseline_ll = baseline_results[0][0] if isinstance(baseline_results[0], tuple) else baseline_results[0]

# Compress (wildcard targets all layers)
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=["model.layers[*].mlp.gate_proj"],
    rank=16,
    svd_backend="torch",
)
consolidator.initialize(ctx)
consolidator.compress_model_with_surgery(model)

# Compressed
eval_compressed = ModelEvalInterface(
    model=model, tokenizer=tokenizer, device=str(model.device),
)
compressed_results = eval_compressed.loglikelihood_rolling(test_texts)
compressed_ll = compressed_results[0][0] if isinstance(compressed_results[0], tuple) else compressed_results[0]

import math
assert math.isfinite(baseline_ll)
assert math.isfinite(compressed_ll)
print(f"Baseline LL: {baseline_ll:.4f}")
print(f"Compressed LL: {compressed_ll:.4f}")
print(f"Delta: {compressed_ll - baseline_ll:.4f}")
```

Both values should be finite negative numbers. The compressed model may have a
slightly lower (worse) log-likelihood depending on rank and target selection.

### 2d. LMEvaluator Plugin

`LMEvaluator` is a plugin-based wrapper around `ModelEvalInterface`. In the
current implementation it:

- Uses `context.state` to discover the model and tokenizer
- Uses `execute(num_samples=...)` or `context.config["evaluation"]` limit keys
  to cap examples
- Uses `context.config["test_settings"]` only as a fallback source of sample
  data when `datasets` is unavailable or dataset loading fails

It does not require the external `lm_eval` package.

```python
from goldcrest.plugins.evaluation.lm_eval import LMEvaluator

ctx = PipelineContext(
    config={
        "evaluation": {
            "limit": 4,
        },
        "test_settings": {
            # Used as fallback sample data if Hugging Face `datasets`
            # is unavailable or dataset loading fails.
            "perplexity_test_cases": [
                "The capital of France is Paris.",
                "Machine learning is a subset of artificial intelligence.",
            ],
        },
    },
    workspace_dir="workspace/evaluation",
)
ctx.state.model = model
ctx.state.tokenizer = tokenizer

# Compress first
consolidator = ModelConsolidator(
    compression_method="svd",
    target_modules=["model.layers[0].self_attn.v_proj"],
    rank=32,
    svd_backend="torch",
)
consolidator.initialize(ctx)
consolidator.compress_model_with_surgery(model)

# Evaluate
evaluator = LMEvaluator(tasks=["wikitext"], name="self_eval")
evaluator.initialize(ctx)
result = evaluator.execute()
```

`LMEvaluator` currently reads these configuration surfaces:

| Config key | Type | Default | Description |
|---|---|---|---|
| `execute(num_samples=...)` | `int` | `None` | Highest-precedence sample cap for the current run |
| `evaluation.limit` / `evaluation.max_examples` / `evaluation.num_samples` / `evaluation.default_num_samples` | `int` | `64` | Sample-cap fallback chain when `num_samples` is not passed |
| `test_settings.classification_test_cases` | `List[Dict]` | built-in fallback | Fallback examples shaped like `{context, choices, answer}` |
| `test_settings.perplexity_test_cases` | `List[str]` | built-in fallback | Fallback texts for `wikitext` / `ptb_new` |

Current `LMEvaluator.get_supported_tasks()` returns:

- Classification: `hellaswag`, `winogrande`, `piqa`, `siqa`, `arc_easy`,
  `arc_challenge`, `openbookqa`, `boolq`, `copa`, `wsc`
- Perplexity: `wikitext`, `ptb_new`

Dataset-backed loaders are currently implemented for `wikitext`, `ptb_new`,
`arc_easy`, `arc_challenge`, `hellaswag`, `winogrande`, `piqa`, `boolq`, and
`siqa`. Other supported tasks fall back to config/default sample cases.

Note: `evaluation.device`, `evaluation.batch_size`, and
`evaluation.max_length` are initialized to constructor-time defaults in the
current code path; `initialize(ctx)` does not re-read those keys from
`PipelineContext.config`.

**Source:** `tests/e2e/test_e2e_evaluation.py`, class `TestSelfDefinedEval`

---

## ModelEvalInterface API Reference

```python
class ModelEvalInterface:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "auto",
        batch_size: int = 1,
        max_length: int = 2048,
    ): ...

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        """Score (context, continuation) pairs.
        Returns list of (log_likelihood, is_greedy) tuples."""

    def loglikelihood_rolling(
        self, requests: List[str]
    ) -> List[Tuple[float]]:
        """Compute rolling log-likelihood for perplexity.
        Returns list of (log_likelihood,) tuples."""

    def tokenize(self, string: str) -> List[int]:
        """Tokenize a string (compatibility method)."""

    def detokenize(self, tokens: List[int]) -> str:
        """Detokenize a list of tokens (compatibility method)."""

    @property
    def max_gen_toks(self) -> int:
        """Maximum generation tokens (default 256)."""
```

---

## CSV Logging

All evaluation tests in `test_e2e_evaluation.py` log results through
`CSVLogger`, which writes structured records for experiment tracking. The four
record types are:

| Record | Logged by | Key fields |
|---|---|---|
| `ExperimentSession` | test harness | `experiment_id`, `start_time`, `status` |
| `ModelRecord` | `log_model_info()` | `model_name`, `model_type`, `parameter_count`, `size_mb` |
| `EvaluationRecord` | `log_evaluation_results()` | `model_type`, `evaluation_type`, `plugin_name`, `task_name`, `metric_name`, `metric_value` |
| `CompressionRecord` | `log_compression_results()` | `compression_method`, `original_size_mb`, `compressed_size_mb`, `compression_ratio` |

`CSVLogger.log_*()` requires an active experiment session started with
`start_experiment(...)`; otherwise it raises `RuntimeError`. The logger flattens
each nested `task_results` mapping into one `EvaluationRecord` per metric.

Example logging flow:

```python
from goldcrest.plugins.evaluation.csv_logger import CSVLogger

csv_logger = CSVLogger("workspace/evaluation_logs")
csv_logger.start_experiment("evaluation_demo", config_file="inline")

csv_logger.log_evaluation_results(
    model_type="compressed",
    evaluation_type="lm_eval",
    plugin_name="lm_eval_harness",
    task_results={"arc_easy": {"result": str(result)}},
    evaluation_params={"tasks": ["arc_easy"], "limit": 4},
    execution_time=eval_time,
)

csv_logger.end_experiment(status="completed")
```

---

## Choosing an Evaluation Approach

| Approach | When to use | External deps |
|---|---|---|
| **LMHarness** | Standard benchmarks (arc_easy, hellaswag, mmlu, wikitext, etc.) with canonical scoring | `lm_eval` package required |
| **ModelEvalInterface** | Direct rolling log-likelihood / log-likelihood on custom text; quick A/B comparison between baseline and compressed | None (built-in) |
| **LMEvaluator** | Built-in task evaluation with dataset-backed or fallback sample loading; lightweight alternative to LMHarness | None (built-in) |
