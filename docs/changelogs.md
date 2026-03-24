# Changelogs

## 2026-03-18

- Initial public release.
- Compression: SVD, ASVD, SVD-LLM, Tucker, CP, Tensor-Train, pruning, KV-cache.
- Analysis: activation metrics, weight metrics, Fisher information, layer sensitivity.
- Evaluation: `lm_eval` integration.
- Configuration: YAML-driven pipelines with defaults.
- Runtime: H100/H200 scripts, CPU scripts, memory profiling, latency measurement.
- Integrations: CoLA, TensorLy, `information_flow`.
- Docs: remote runs guide and benchmark results.

## 2026-03-19

- `LMHarness` now defaults to safe in-memory evaluation instead of silently reloading checkpoints.
- `WorkflowExecutor` now separates constructor config from runtime kwargs and forwards `context`, `model`, and `tokenizer` correctly.
- Evaluation and compression plugins were brought back onto the framework `do_execute()` lifecycle.
- Package-safe imports were repaired, including the import-time break in `compressed_eval.py`.
- The main Hugging Face loader path now normalizes auth kwargs and uses `AutoModelForCausalLM`.
- Critical paths now have pytest coverage, including a toy end-to-end workflow.


## 2026-03-22
- `ModelConsolidator._create_factor_layer_from_compressed_data`: fix the problem of Tucker factors clone in `FactorLayer`

## 2026-03-23

- Added full e2e test suite (103 tests) covering all 31 features on real Gemma-3-270m-it: compression algorithms, rank selection, algebra backends, target layers, evaluation, and profiling.
- Bug fixes driven by e2e tests: removed dead code in compression plugins (svd, cp, tucker, tensor_train, tensorizer, base), consolidated model surgery logic in `ModelConsolidator`, and fixed `activation_metrics`/`metric_utils` metric computation.
- Added docs: `algebra_backends.md`, `compression_algorithm.md`, `layer_types.md`, `logging.md`.
- CSV logging integration across all e2e tests via `CSVLogger` fixtures in conftest.